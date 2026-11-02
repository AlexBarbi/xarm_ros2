#!/usr/bin/env python3
"""
Sort the pieces on the table of shared/world.usd into their bins with MoveIt 2, finding them with the wrist camera
(xArm5 + xArm gripper + RealSense, e.g. xarm5_moveit_isaac.launch.py).

Nothing about the pieces is given: one depth image of the table is enough to find them, piece_vision.py recognizes
each part number from the CAD (shared/piece_models.json) and works out where to hold it. PIECE_BINS says which bin
each part number goes to. The table is scanned again until no piece is left on it.

The arm goes from point to point the shortest way: along the straight line when the IK can follow it, planned around
what blocks it otherwise, and always with the fewest turns of the joints. The table and the bins are behind the base,
so this takes joint1 past +-180 deg: the launch has to give the arm its full range (limited:=false, the default of
xarm5_moveit_isaac.launch.py), and the same goes for the joints of shared/world.usd.

Only moveit_msgs is needed (no moveit_py): the script talks to the move_group node through
  /move_action (MoveGroup), /compute_ik, /compute_fk, /compute_cartesian_path, /execute_trajectory,
  /apply_planning_scene, /get_planning_scene
and drives the gripper directly: the action of the robot driver (xarm_gripper/gripper_action) on the real arm, the
trajectory controller in simulation. Neither says anything about force, so everything it knows about a piece comes from
drive_joint in /joint_states: the fingers are sent past the piece, and it is there as long as they stay short of that.

Usage:
  python3 pick_and_place.py             # sort everything
  python3 pick_and_place.py --dry-run   # only scan, and print what is on the table and how it would be taken
  python3 pick_and_place.py --depth-topic /camera/aligned_depth_to_color/image_noisy   # with depth_noise.py running
"""

import argparse
import math
import sys
import threading
import time
import xml.etree.ElementTree as ET
from collections import deque

import numpy as np
import rclpy
from rclpy.action import ActionClient, get_action_names_and_types
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
from rclpy.utilities import remove_ros_args
from scipy.spatial.transform import Rotation

from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory, GripperCommand
from geometry_msgs.msg import Point, Pose, Quaternion
from moveit_msgs.action import ExecuteTrajectory, MoveGroup
from moveit_msgs.msg import (AttachedCollisionObject, CollisionObject, Constraints, JointConstraint, MoveItErrorCodes,
                             PlanningScene, PlanningSceneComponents)
from moveit_msgs.srv import ApplyPlanningScene, GetCartesianPath, GetPlanningScene, GetPositionFK, GetPositionIK
from sensor_msgs.msg import CameraInfo, Image, JointState
from shape_msgs.msg import SolidPrimitive
from std_msgs.msg import String
from trajectory_msgs.msg import JointTrajectoryPoint

from piece_vision import FINGER_MARGIN, MAX_GAP, PieceLibrary, TableMap, finger_position

# ---------------------------------------------------------------- what goes where: part number -> bin
PIECE_BINS = {
    '100000049205': 0,
    '100000051189': 1,
    '100000057387': 2,
    '100000047465': 3,
    '100000047481': 4,
    # '100000047527': 5,
    '100000047531': 6,
    # '100000052737': 7,
    '100000053152': 8,
    # '100000056753': 9,
    '100000057388': 10,
}

# ---------------------------------------------------------------- robot
FRAME = 'link_base'                 # every position below is in this frame
ARM_GROUP = 'xarm5'
ARM_JOINTS = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5']
TURNING_JOINTS = ['joint1', 'joint5']   # one full turn away is the same pose of the arm
ARRIVAL_TOLERANCE = 0.03            # rad: the controller gives up at 0.01, and the grip force of the simulated gripper
                                    # alone pushes the wrist that far
TCP_LINK = 'link_tcp'               # between the fingertips, 0.172 m from the gripper base
CAMERA_LINK = 'camera_color_optical_frame'
DEPTH_TOPIC = '/camera/aligned_depth_to_color/image_raw'
CAMERA_INFO_TOPIC = '/camera/color/camera_info'
GRIPPER_ACTIONS = {'xarm_gripper/gripper_action': GripperCommand,     # end of the action name on the real arm
                   'xarm_gripper_traj_controller/follow_joint_trajectory': FollowJointTrajectory}   # in simulation
GRIPPER_JOINT = 'drive_joint'
GRIPPER_OPEN = 0.0
GRIPPER_CLOSED = 0.85               # fingers touching each other
GRIPPER_SPEED = 1.0                 # rad/s of the free motions of the fingers: faster, opening right after a grip tears
                                    # the simulated gripper linkage apart and takes the arm with it (URDF limit: 2)
GRIPPER_RELEASE = 0.3               # rad opened to let go of a piece: enough, and quick
GRIPPER_CLOSE_SPEED = 0.35          # rad/s closing on a piece: slow enough to stop on it instead of slamming it
GRIPPER_STALL_TIME = 0.2            # s without finger motion while closing = contact
GRIPPER_STALL_TOL = 0.005           # rad of finger motion still counted as stopped
GRIPPER_HELD_ERROR = 0.01           # rad short of the target = stopped by something, closer = nothing between the fingers
GRIPPER_MISS_MARGIN = 0.008         # m: the fingers are sent this much closer than the width of the piece
GRIPPER_FREE_SAMPLES = 3            # samples of drive_joint in a row at the target, while carrying = the piece is gone
GRIPPER_TIMEOUT = 30.0              # s of wall clock: Isaac Sim can run much slower than real time
TOUCH_LINKS = ['xarm_gripper_base_link', 'left_outer_knuckle', 'left_finger', 'left_inner_knuckle',
               'right_outer_knuckle', 'right_finger', 'right_inner_knuckle', 'link_tcp']

# ---------------------------------------------------------------- scene (shared/world.usd)
TABLE_TOP_Z = -0.006                # surface the pieces and the bins stand on
TABLE_SIZE = (0.8, 1.5, 0.03)
TABLE_CENTER_XY = (-0.34, -0.2)

# open bins: centers, inner size, outer height, wall/floor thickness
BINS_XY = [(-0.12, 0.28), (-0.28, 0.28), (-0.44, 0.28), (-0.60, 0.28),
           (-0.12, 0.44), (-0.28, 0.44), (-0.44, 0.44),                 # (-0.60, 0.44) is out of reach
           (-0.12, 0.12), (-0.28, 0.12), (-0.44, 0.12), (-0.60, 0.12)]
BIN_BOTTOM_Z = TABLE_TOP_Z
BIN_INNER = (0.114, 0.114)          # x, y
BIN_HEIGHT = 0.07
BIN_WALL = 0.008

# part of the table the camera looks for pieces in: up to the bins, without the base of the robot
WORK_X = (-0.66, 0.0)
WORK_Y = (-0.62, 0.04)
ROBOT_KEEP_OUT = (0.0, 0.0, 0.14)
MODELS = 'shared/piece_models.json'

# ---------------------------------------------------------------- motion
SCAN_XYZ = (-0.377, -0.242, 0.397)  # TCP, gripper pointing down: the camera sees 77% of the work area from here
MORE_VIEWS = [(-0.16, -0.32), (-0.44, -0.20)]   # x, y at the same height, 93% with these: a table is not empty before
LOOK_REACH = 0.5                    # m from the base: further than this the arm cannot look straight down from SCAN_XYZ's height
MAX_LOOKS = 3                       # closer looks at what is not recognized, for each scan of the table
FREE_SPEED = 1.0                    # fraction of the joint limits
CARRY_SPEED = 1.0                   # holding a piece
LINEAR_SPEED = 1.0                  # straight moves near a piece
ACCELERATION = 0.8
ABOVE_PIECE = 0.035                 # fingertips above the top of the piece before going down
PLAN_ATTEMPTS = 3                   # a random planner can return a path that fails the final validation
LINEAR_ATTEMPTS = 3                 # the IK of a straight line gives up now and then, on the same line
LINE_DONE = 0.999                   # part of a straight line that counts as all of it
LINE_STEP = 0.005                   # m between the poses of a straight line: wider, the wrist turns in kinks that slow it all
MAX_SHORT_MOTION = 0.5              # rad of any joint for a short move: more means the arm would fold over
MAX_LINE_STEP = 0.5                 # rad of any joint between two points of a straight line: more is the IK flipping the
                                    # arm over, not a line
IK_ATTEMPTS = 5                     # the IK restarts from random seeds: some land on the arm reaching backwards
MIN_ELBOW = -math.pi                # joint3 further than this: the elbow folded the other way, reachable but a long way
CARRY_CLEARANCE = 0.035             # bottom of a carried piece above the bin rims and the pieces on the table
RELEASE_CLEARANCE = 0.015           # bottom of a piece above the rim of its bin when let go: from higher it bounces out
BIN_CORNER = 0.037                  # from the center of a bin, along x and y, to where a tall piece is dropped
CORNERS = [(-1, 1), (1, 1), (-1, -1), (1, -1)]  # the arm comes from -y: the far ones first, it does not pass over them
PIECE_MARGIN = 0.01                 # around the pieces on the table, for the planner
PLANNING_TIME = 2.0
GRASP_ATTEMPTS = 3                  # grasps of a piece in a row: when one slips the piece stays where it is
MAX_ROUNDS = 4                      # scans of the table with something left to do

PIECE_PREFIX = 'piece'
TABLE_ID = 'table'


def top_down(yaw):
    # Rz(yaw) * Rx(pi): TCP z axis pointing down, rotated by yaw about the vertical
    return Quaternion(x=math.cos(yaw / 2), y=math.sin(yaw / 2), z=0.0, w=0.0)


def make_pose(x, y, z, q=None):
    return Pose(position=Point(x=float(x), y=float(y), z=float(z)), orientation=q or Quaternion())


def wrapped(angle):
    """The same angle in -pi..pi."""
    return math.remainder(angle, 2 * math.pi)


def box(size, pose):
    return SolidPrimitive(type=SolidPrimitive.BOX, dimensions=[float(s) for s in size]), pose


class PieceDropped(RuntimeError):
    """The piece fell on the way to its bin: it is not where the next grasp would look for it."""


class PickAndPlace(Node):

    def __init__(self, depth_topic=DEPTH_TOPIC):
        super().__init__('pick_and_place')
        self.move_client = ActionClient(self, MoveGroup, 'move_action')
        self.execute_client = ActionClient(self, ExecuteTrajectory, 'execute_trajectory')
        self.gripper_client = None
        self.ik_client = self.create_client(GetPositionIK, 'compute_ik')
        self.fk_client = self.create_client(GetPositionFK, 'compute_fk')
        self.cartesian_client = self.create_client(GetCartesianPath, 'compute_cartesian_path')
        self.apply_scene_client = self.create_client(ApplyPlanningScene, 'apply_planning_scene')
        self.get_scene_client = self.create_client(GetPlanningScene, 'get_planning_scene')
        self.library = PieceLibrary(MODELS, PIECE_BINS)
        self.joints = {}
        self.joints_stamp = 0.0
        self.joint_limits = {}
        self.tall_in_bin = [0] * len(BINS_XY)
        self.bin_top = 0.0                          # height above the table of the tallest piece standing in a bin
        self.create_subscription(String, 'robot_description', self.on_robot_description,
                                 QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        # (stamp [s], position) of drive_joint, stamps in simulated time with Isaac Sim
        self.finger_samples = deque(maxlen=1000)
        self.finger_lock = threading.Lock()
        self.create_subscription(JointState, 'joint_states', self.on_joint_states, 10)
        self.carrying = False                       # arm motions watch the grip meanwhile
        self.depth = None
        self.camera_info = None
        self.create_subscription(Image, depth_topic, self.on_depth, qos_profile_sensor_data)
        self.create_subscription(CameraInfo, CAMERA_INFO_TOPIC, self.on_camera_info, qos_profile_sensor_data)

        # Node.executor is taken by rclpy
        self.spin_executor = SingleThreadedExecutor()
        self.spin_executor.add_node(self)
        self.spin_thread = threading.Thread(target=self.spin_executor.spin, daemon=True)
        self.spin_thread.start()

    def close(self):
        self.spin_executor.shutdown()
        self.spin_thread.join(timeout=2.0)
        self.destroy_node()

    def on_robot_description(self, msg):
        for joint in ET.fromstring(msg.data).iter('joint'):
            limit = joint.find('limit')
            if joint.get('name') in ARM_JOINTS and limit is not None:
                self.joint_limits[joint.get('name')] = (float(limit.get('lower')), float(limit.get('upper')))

    def on_joint_states(self, msg):
        self.joints.update(zip(msg.name, msg.position))
        self.joints_stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if GRIPPER_JOINT in msg.name:
            with self.finger_lock:
                self.finger_samples.append((self.joints_stamp, msg.position[msg.name.index(GRIPPER_JOINT)]))

    def on_depth(self, msg):
        self.depth = msg

    def on_camera_info(self, msg):
        self.camera_info = msg

    def finger_samples_since(self, stamp):
        with self.finger_lock:
            return [(t, p) for t, p in self.finger_samples if t >= stamp]

    # ------------------------------------------------------------ plumbing

    def find_gripper(self, timeout):
        """The real gripper is an action of the robot driver, in simulation a controller moves it."""
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            names = [name for name, _ in get_action_names_and_types(self)]
            for ending, kind in GRIPPER_ACTIONS.items():
                for name in names:
                    if name.endswith(ending):
                        self.get_logger().info('gripper: {}'.format(name))
                        return ActionClient(self, kind, name), kind
            time.sleep(0.2)
        raise RuntimeError('no gripper action, looked for {}'.format(list(GRIPPER_ACTIONS)))

    def wait_for_servers(self, timeout=15.0):
        self.gripper_client, self.gripper_kind = self.find_gripper(timeout)
        for client in (self.move_client, self.execute_client, self.gripper_client):
            if not client.wait_for_server(timeout_sec=timeout):
                raise RuntimeError('action server {} not available'.format(client._action_name))
        for client in (self.ik_client, self.fk_client, self.cartesian_client, self.apply_scene_client, self.get_scene_client):
            if not client.wait_for_service(timeout_sec=timeout):
                raise RuntimeError('service {} not available'.format(client.srv_name))

    @staticmethod
    def wait(future, timeout, what, check=None):
        end = time.monotonic() + timeout
        while not future.done():
            if time.monotonic() > end:
                raise RuntimeError('{}: timed out after {:.0f} s'.format(what, timeout))
            if check:
                check()
            time.sleep(0.005)
        return future.result()

    def call(self, client, request, what, timeout=30.0):
        return self.wait(client.call_async(request), timeout, what)

    def run_action(self, client, goal, what, timeout=120.0, check=None):
        handle = self.wait(client.send_goal_async(goal), 10.0, what)
        if not handle.accepted:
            raise RuntimeError('{}: goal rejected'.format(what))
        result = self.wait(handle.get_result_async(), timeout, what, check and (lambda: check(handle, what))).result
        if check:
            check(handle, what)     # a move can end with the piece gone, and the next may be the release
        return result

    # ------------------------------------------------------------ planning scene

    def apply_scene(self, scene):
        scene.is_diff = True
        scene.robot_state.is_diff = True
        return self.call(self.apply_scene_client, ApplyPlanningScene.Request(scene=scene), 'apply planning scene').success

    def add_objects(self, *objects):
        scene = PlanningScene()
        scene.world.collision_objects = list(objects)
        if not self.apply_scene(scene):
            raise RuntimeError('could not add {} to the planning scene'.format([o.id for o in objects]))

    def remove_objects(self, *ids):
        scene = PlanningScene()
        scene.world.collision_objects = [CollisionObject(id=i, operation=CollisionObject.REMOVE) for i in ids]
        self.apply_scene(scene)

    def collision_object(self, object_id, pose, *primitives):
        co = CollisionObject(id=object_id, operation=CollisionObject.ADD, pose=pose)
        co.header.frame_id = FRAME
        co.primitives = [p for p, _ in primitives]
        co.primitive_poses = [p for _, p in primitives]
        return co

    def bin_object(self, index):
        lx, ly = BIN_INNER
        t, h = BIN_WALL, BIN_HEIGHT
        return self.collision_object(
            'bin{}'.format(index), make_pose(BINS_XY[index][0], BINS_XY[index][1], BIN_BOTTOM_Z),
            box((lx + 2 * t, ly + 2 * t, t), make_pose(0, 0, t / 2)),                 # floor
            box((t, ly + 2 * t, h), make_pose(+(lx + t) / 2, 0, h / 2)),              # walls
            box((t, ly + 2 * t, h), make_pose(-(lx + t) / 2, 0, h / 2)),
            box((lx, t, h), make_pose(0, +(ly + t) / 2, h / 2)),
            box((lx, t, h), make_pose(0, -(ly + t) / 2, h / 2)))

    def table_object(self):
        # top surface 5 mm under TABLE_TOP_Z, so the robot base and the pieces do not touch it
        return self.collision_object(
            TABLE_ID, make_pose(TABLE_CENTER_XY[0], TABLE_CENTER_XY[1], TABLE_TOP_Z - 0.005 - TABLE_SIZE[2] / 2),
            box(TABLE_SIZE, Pose()))

    def piece_object(self, piece, margin=0.0):
        size = (piece.length + margin, piece.width + margin, piece.height + margin)
        pose = make_pose(piece.center[0], piece.center[1], TABLE_TOP_Z + 0.002 + piece.height / 2,
                         Quaternion(z=math.sin(piece.angle / 2), w=math.cos(piece.angle / 2)))
        return self.collision_object('{}{}'.format(PIECE_PREFIX, piece.label), pose, box(size, Pose()))

    def detach_object(self, object_id, link_name=TCP_LINK):
        aco = AttachedCollisionObject(link_name=link_name)
        aco.object.id = object_id
        aco.object.operation = CollisionObject.REMOVE
        scene = PlanningScene()
        scene.robot_state.attached_collision_objects = [aco]
        if not self.apply_scene(scene):
            raise RuntimeError('could not detach {}'.format(object_id))
        # the detached object goes back into the world: take it out, it is in a bin now
        self.remove_objects(object_id)

    def attach_object(self, obj):
        aco = AttachedCollisionObject(link_name=TCP_LINK, object=obj, touch_links=TOUCH_LINKS)
        scene = PlanningScene()
        scene.robot_state.attached_collision_objects = [aco]
        if not self.apply_scene(scene):
            raise RuntimeError('could not attach {}'.format(obj.id))

    def clear_pieces(self):
        """Take the pieces of a previous scan or of an aborted run out of the planning scene."""
        request = GetPlanningScene.Request()
        request.components.components = (PlanningSceneComponents.ROBOT_STATE_ATTACHED_OBJECTS |
                                          PlanningSceneComponents.WORLD_OBJECT_NAMES)
        scene = self.call(self.get_scene_client, request, 'get planning scene').scene
        for aco in scene.robot_state.attached_collision_objects:
            if aco.object.id.startswith(PIECE_PREFIX):
                self.detach_object(aco.object.id, aco.link_name)
        old = [o.id for o in scene.world.collision_objects if o.id.startswith(PIECE_PREFIX)]
        if old:
            self.remove_objects(*old)

    # ------------------------------------------------------------ camera

    def link_pose(self, link, joints=None):
        """(rotation matrix, translation) of a link in FRAME, now or with the joints at these positions."""
        request = GetPositionFK.Request()
        request.header.frame_id = FRAME
        request.fk_link_names = [link]
        request.robot_state.is_diff = True
        if joints:
            request.robot_state.joint_state.name = list(joints)
            request.robot_state.joint_state.position = list(joints.values())
        response = self.call(self.fk_client, request, '{} pose'.format(link))
        if response.error_code.val != MoveItErrorCodes.SUCCESS or not response.pose_stamped:
            raise RuntimeError('{} pose: MoveIt error {}'.format(link, response.error_code.val))
        pose = response.pose_stamped[0].pose
        q, p = pose.orientation, pose.position
        return Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix(), np.array([p.x, p.y, p.z])

    def depth_image(self):
        """Latest depth image (meters), taken after the arm stopped."""
        arrived = self.joints_stamp
        end = time.monotonic() + 20.0
        while True:
            msg = self.depth
            if msg is not None and msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9 > arrived + 0.05:
                break
            if time.monotonic() > end:
                raise RuntimeError('no depth image')
            time.sleep(0.01)
        if msg.encoding == '32FC1':
            return np.frombuffer(bytes(msg.data), np.float32).reshape(msg.height, msg.width).astype(float)
        return np.frombuffer(bytes(msg.data), np.uint16).reshape(msg.height, msg.width) / 1000.0

    def scan(self):
        """Look at the table from above and return its map."""
        self.move_to(make_pose(*SCAN_XYZ, top_down(0.0)), 'scan', FREE_SPEED)
        table = TableMap(TABLE_TOP_Z, WORK_X, WORK_Y, [ROBOT_KEEP_OUT])
        self.add_view(table)
        return table

    def look_around(self, table):
        """Add the views of the part of the table the first one does not cover."""
        for x, y in MORE_VIEWS:
            try:
                self.move_to(make_pose(x, y, SCAN_XYZ[2], top_down(0.0)), 'look around', FREE_SPEED)
                self.add_view(table)
            except RuntimeError as e:
                self.get_logger().warn(str(e))

    def add_view(self, table):
        depth = self.depth_image()
        rotation, translation = self.link_pose(CAMERA_LINK)
        k = self.camera_info.k
        table.add_view(depth, k[0], k[4], k[2], k[5], rotation, translation)

    def look_closer(self, table, piece):
        """Add the view from above a piece: at the edge of the first one, or behind something else, it was only partly seen."""
        x, y = piece.center
        scale = min(LOOK_REACH / math.hypot(x, y), 1.0)
        self.move_to(make_pose(x * scale, y * scale, SCAN_XYZ[2], top_down(0.0)), 'closer look', FREE_SPEED)
        self.add_view(table)

    # ------------------------------------------------------------ motion

    def compute_ik(self, pose, what):
        """Collision-free joint positions of the arm for the TCP at pose, seeded from the current state."""
        request = GetPositionIK.Request()
        ik = request.ik_request
        ik.group_name = ARM_GROUP
        ik.ik_link_name = TCP_LINK
        # starting from the arm already facing the pose keeps the solver from folding it over to get there
        facing = math.atan2(pose.position.y, pose.position.x)
        ik.robot_state.is_diff = True
        ik.robot_state.joint_state.name = ['joint1']
        ik.robot_state.joint_state.position = [self.fewest_turns({'joint1': facing})['joint1']]
        ik.avoid_collisions = True
        ik.pose_stamped.header.frame_id = FRAME
        ik.pose_stamped.pose = pose
        ik.timeout.sec = 1
        for _ in range(IK_ATTEMPTS):
            response = self.call(self.ik_client, request, what)
            if response.error_code.val != MoveItErrorCodes.SUCCESS:
                raise RuntimeError('{}: no collision-free IK solution for ({:.3f}, {:.3f}, {:.3f}), out of reach or in '
                                   'collision? (MoveIt error {})'.format(what, pose.position.x, pose.position.y,
                                                                         pose.position.z, response.error_code.val))
            joint_state = response.solution.joint_state
            joints = {name: position for name, position in zip(joint_state.name, joint_state.position)
                      if name in ARM_JOINTS}
            if abs(wrapped(joints['joint1'] - facing)) < math.pi / 2 and joints['joint3'] > MIN_ELBOW:
                return self.fewest_turns(joints)
        raise RuntimeError('{}: only reachable with the arm folded the other way'.format(what))

    def fewest_turns(self, joints):
        """The same pose of the arm, with the joints that turn all the way round closest to where they are now."""
        for name in TURNING_JOINTS:
            if name not in joints:
                continue
            lower, upper = self.joint_limits[name]
            options = [joints[name] + turns * 2 * math.pi for turns in (-2, -1, 0, 1, 2)]
            joints[name] = min((o for o in options if lower <= o <= upper), key=lambda o: abs(o - self.joints[name]))
        return joints

    def move_to(self, pose, what, speed):
        """Take the TCP to pose (in FRAME) the shortest way: the straight line, or a planned motion if it is blocked."""
        fraction = self.follow_line(pose, what, speed)
        if fraction >= LINE_DONE:
            return
        self.get_logger().info('{}: {:.0%} of the straight line is feasible, going around'.format(what, fraction))
        # A 5-DOF arm reaches exact top-down poses only: MoveIt's pose goal sampler perturbs the orientation within
        # the tolerances and then finds no IK, so solve IK for the exact pose and plan to that joint goal.
        self.get_logger().info('{}: moving to ({:.3f}, {:.3f}, {:.3f})'.format(
            what, pose.position.x, pose.position.y, pose.position.z))
        self.move_joints(self.compute_ik(pose, what), what, speed)

    def move_joints(self, joints, what, speed):
        goal = MoveGroup.Goal()
        request = goal.request
        request.group_name = ARM_GROUP
        request.start_state.is_diff = True
        request.goal_constraints = [Constraints(joint_constraints=[
            JointConstraint(joint_name=name, position=position, tolerance_above=0.001, tolerance_below=0.001, weight=1.0)
            for name, position in joints.items()])]
        request.num_planning_attempts = 2
        request.allowed_planning_time = PLANNING_TIME
        request.max_velocity_scaling_factor = speed
        request.max_acceleration_scaling_factor = ACCELERATION
        request.workspace_parameters.header.frame_id = FRAME
        request.workspace_parameters.min_corner.x = request.workspace_parameters.min_corner.y = -1.0
        request.workspace_parameters.min_corner.z = -0.5
        request.workspace_parameters.max_corner.x = request.workspace_parameters.max_corner.y = 1.0
        request.workspace_parameters.max_corner.z = 1.5
        goal.planning_options.plan_only = False

        for _ in range(PLAN_ATTEMPTS):
            result = self.run_action(self.move_client, goal, what, check=self.check_grip)
            if result.error_code.val not in (MoveItErrorCodes.INVALID_MOTION_PLAN, MoveItErrorCodes.PLANNING_FAILED):
                break
        self.check_arrival(result, joints, what)

    def check_arrival(self, result, joints, what):
        if result.error_code.val == MoveItErrorCodes.CONTROL_FAILED and all(
                abs(self.joints[name] - position) < ARRIVAL_TOLERANCE for name, position in joints.items()):
            return
        if result.error_code.val != MoveItErrorCodes.SUCCESS:
            raise RuntimeError('{}: MoveIt error {}'.format(what, result.error_code.val))

    def line_poses(self, start, pose):
        """Poses along the straight line from the arm at start to pose, joint5 turning evenly. Left to blend the
        orientation, the wrist swings to and fro to make up for joint1, which turns fastest in the middle of a line."""
        origin = self.link_pose(TCP_LINK, start)[1]
        target = np.array([pose.position.x, pose.position.y, pose.position.z])
        count = max(int(np.linalg.norm(target - origin) / LINE_STEP), 1)
        base = start['joint1']
        wrist = start['joint5']
        end_yaw = 2 * math.atan2(pose.orientation.y, pose.orientation.x)
        wrist_turn = wrapped(math.atan2(target[1], target[0]) - end_yaw - wrist)    # yaw = joint1 - joint5, looking down
        poses = []
        for i in range(1, count):
            point = origin + (target - origin) * i / count
            turned = base + wrapped(math.atan2(point[1], point[0]) - base)
            poses.append(make_pose(*point, top_down(turned - wrist - wrist_turn * i / count)))
        return poses + [pose]

    def follow_line(self, pose, what, speed):
        """Move the TCP along the straight line to pose if the arm can follow all of it. Returns how much of it it can."""
        request = GetCartesianPath.Request()
        request.header.frame_id = FRAME
        # The arm sags a hair off vertical, and a 5-DOF arm cannot tilt sideways: blending that tilt away while joint1
        # turns has no IK, even at 0.001 rad. So the line is worked out from the tool exactly vertical, joint4 being
        # what keeps it so, and the trajectory then starts from where the arm really is.
        start = {name: self.joints[name] for name in ARM_JOINTS}
        start['joint4'] = -(start['joint2'] + start['joint3'])
        request.start_state.is_diff = True
        request.start_state.joint_state.name = list(start)
        request.start_state.joint_state.position = list(start.values())
        request.group_name = ARM_GROUP
        request.link_name = TCP_LINK
        request.waypoints = self.line_poses(start, pose)
        request.max_step = LINE_STEP
        request.revolute_jump_threshold = MAX_LINE_STEP
        request.avoid_collisions = True
        request.max_velocity_scaling_factor = speed
        request.max_acceleration_scaling_factor = ACCELERATION
        for _ in range(LINEAR_ATTEMPTS):
            response = self.call(self.cartesian_client, request, what)
            if response.error_code.val == MoveItErrorCodes.SUCCESS and response.fraction >= LINE_DONE:
                self.get_logger().info('{}: straight line to ({:.3f}, {:.3f}, {:.3f})'.format(
                    what, pose.position.x, pose.position.y, pose.position.z))
                trajectory = response.solution.joint_trajectory
                trajectory.points[0].positions = [self.joints[name] for name in trajectory.joint_names]
                result = self.run_action(self.execute_client, ExecuteTrajectory.Goal(trajectory=response.solution), what,
                                         check=self.check_grip)
                self.check_arrival(result, dict(zip(trajectory.joint_names, trajectory.points[-1].positions)), what)
                return response.fraction
        return response.fraction

    def move_linear(self, pose, what):
        """Straight TCP line to pose, or a free motion to it if that keeps the arm in the same configuration."""
        fraction = self.follow_line(pose, what, LINEAR_SPEED)
        if fraction >= LINE_DONE:
            return
        # the IK of a free motion restarts from random seeds and can answer with the arm folded the other way: going
        # there from close to the table or a bin would slam the arm into them
        self.get_logger().warn('{}: only {:.0%} of the straight line is feasible'.format(what, fraction))
        for _ in range(LINEAR_ATTEMPTS):
            joints = self.compute_ik(pose, what)
            if max(abs(position - self.joints[name]) for name, position in joints.items()) < MAX_SHORT_MOTION:
                self.move_joints(joints, what, LINEAR_SPEED)
                return
        raise RuntimeError('{}: only reachable by folding the arm over'.format(what))

    def gripper_goal(self, position, duration):
        """The time to get there is only for the simulated gripper: the real one has one speed, set in the driver."""
        if self.gripper_kind is GripperCommand:
            goal = GripperCommand.Goal()
            goal.command.position = float(position)
            return goal
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = [GRIPPER_JOINT]
        goal.trajectory.points = [JointTrajectoryPoint(
            positions=[float(position)], time_from_start=Duration(sec=int(duration), nanosec=int(duration % 1 * 1e9)))]
        return goal

    def gripper(self, position, what, wait=True):
        travel = abs(position - self.finger_samples_since(0.0)[-1][1])
        goal = self.gripper_goal(position, max(travel / GRIPPER_SPEED, 0.1))
        if not wait:
            self.gripper_client.send_goal_async(goal)   # the fingers get there while the arm moves
            return
        try:
            result = self.run_action(self.gripper_client, goal, '{} gripper'.format(what), timeout=GRIPPER_TIMEOUT)
            if getattr(result, 'error_code', 0) != 0:
                self.get_logger().warn('{} gripper: controller says {} {}'.format(what, result.error_code, result.error_string))
        except RuntimeError as e:
            self.get_logger().warn(str(e))

    def fingers_free(self, count=1):
        """Nothing stops the fingers: the last samples of drive_joint are at the target they were sent to."""
        samples = self.finger_samples_since(0.0)[-count:]
        return len(samples) == count and all(self.grip_target - p < GRIPPER_HELD_ERROR for _, p in samples)

    def check_grip(self, motion, what):
        """Stop an arm motion as soon as the piece it carries is gone, wherever that happens on the way."""
        if not self.carrying or not self.fingers_free(GRIPPER_FREE_SAMPLES):
            return
        joints = dict(self.joints)          # the arm is well past this point by the time it has stopped
        self.carrying = False
        self.wait(motion.cancel_goal_async(), 5.0, what)
        self.wait(motion.get_result_async(), 10.0, what)
        position = self.link_pose(TCP_LINK, joints)[1]
        raise PieceDropped('lost during {}, around ({:.3f}, {:.3f}, {:.3f})'.format(what, *position))

    def close_on(self, width):
        """Close on a piece. Raises when the fingers miss it."""
        # The fingers are sent past the piece: they stop on it and keep pressing, with the force the gripper is limited
        # to (maxForce of the drive in Isaac Sim), also if it shifts between them. Slow, not to slam it. Staying short
        # of the target is all there is to know that a piece is there: the driver reports no force, in any way.
        start_stamp, start_position = self.finger_samples_since(0.0)[-1]
        self.grip_target = finger_position(max(width - GRIPPER_MISS_MARGIN, 0.0))
        goal = self.gripper_goal(self.grip_target, (self.grip_target - start_position) / GRIPPER_CLOSE_SPEED)
        handle = self.wait(self.gripper_client.send_goal_async(goal), 10.0, 'close gripper')
        if not handle.accepted:
            raise RuntimeError('close gripper: goal rejected')
        closed = handle.get_result_async()
        end = time.monotonic() + GRIPPER_TIMEOUT
        while not closed.done():
            if time.monotonic() > end:
                raise RuntimeError('close gripper: timed out after {:.0f} s'.format(GRIPPER_TIMEOUT))
            time.sleep(0.005)
            samples = self.finger_samples_since(start_stamp)
            if self.fingers_free() or not samples:
                continue
            # stalled: no motion over the last GRIPPER_STALL_TIME, once the fingers have started (or could have)
            stamp, position = samples[-1]
            started = position - start_position > 2 * GRIPPER_STALL_TOL or stamp - start_stamp > 1.0
            recent = [p for t, p in samples if t >= stamp - GRIPPER_STALL_TIME]
            if started and stamp - start_stamp >= GRIPPER_STALL_TIME and max(recent) - min(recent) < GRIPPER_STALL_TOL:
                break
        self.grip_contact = self.finger_samples_since(0.0)[-1][1]
        if self.fingers_free():
            raise RuntimeError('close gripper: nothing between the fingers at {:.2f} rad'.format(self.grip_contact))
        self.get_logger().info('close gripper: contact at {:.3f} rad, sent to {:.3f}'.format(self.grip_contact, self.grip_target))

    # ------------------------------------------------------------ task

    def closest_yaw(self, x, y, z, yaw):
        """Yaw for this grasp: the gripper is symmetric, half a turn is the same grasp, so the one the wrist is closest to."""
        options = []
        for option in (yaw, yaw + math.pi):
            try:
                joints = self.compute_ik(make_pose(x, y, z, top_down(option)), 'grasp')
                options.append((abs(joints['joint5'] - self.joints['joint5']), option))
            except RuntimeError as e:
                error = e
        if not options:
            raise error
        return min(options)[1]

    def pick_and_drop(self, piece, grasp, bin_index, tallest_left):
        """Take a piece and let it fall into a bin, carrying it over everything left on the table."""
        z = TABLE_TOP_Z + grasp.z
        above = max(z, TABLE_TOP_Z + piece.height + ABOVE_PIECE)
        yaw = self.closest_yaw(grasp.x, grasp.y, above, grasp.yaw)
        q = top_down(yaw)
        obj = self.piece_object(piece)
        self.gripper(finger_position(min(grasp.width + 2 * FINGER_MARGIN, MAX_GAP)), 'open', wait=False)
        self.move_to(make_pose(grasp.x, grasp.y, above, q), 'above piece', FREE_SPEED)
        self.remove_objects(obj.id)         # the fingers go around it
        self.move_linear(make_pose(grasp.x, grasp.y, z, q), 'approach')
        self.close_on(grasp.width)
        self.attach_object(obj)
        self.carrying = True
        try:
            self.carry_to_bin(piece, grasp, bin_index, tallest_left, z, yaw)
        finally:
            self.carrying = False
            self.gripper(self.grip_contact - GRIPPER_RELEASE, 'release')    # never go on with a piece in the gripper
            self.detach_object(obj.id)

    def carry_to_bin(self, piece, grasp, bin_index, tallest_left, z, yaw):
        carry_z = z + max(BIN_HEIGHT, self.bin_top, tallest_left) + CARRY_CLEARANCE
        try:
            self.move_linear(make_pose(grasp.x, grasp.y, carry_z, top_down(yaw)), 'lift')
        except PieceDropped as e:
            raise RuntimeError('the piece slipped out: {}'.format(e))   # it is still about where it was
        # the wrist stays as it is and the piece turns with the arm, which the bin does not mind: the center of the
        # piece goes over the center of the bin, and the TCP keeps its offset from it, turned as well
        center_x, center_y = self.bin_spot(piece, bin_index)
        turn = wrapped(math.atan2(center_y, center_x) - math.atan2(grasp.y, grasp.x))
        offset_x, offset_y = grasp.x - piece.center[0], grasp.y - piece.center[1]
        bin_x = center_x + offset_x * math.cos(turn) - offset_y * math.sin(turn)
        bin_y = center_y + offset_x * math.sin(turn) + offset_y * math.cos(turn)
        q = top_down(yaw + turn)
        self.move_to(make_pose(bin_x, bin_y, carry_z, q), 'over bin {}'.format(bin_index), CARRY_SPEED)
        # everything drops in from the rim: going lower would take the fingers over it, or into the bin
        release_z = z + BIN_HEIGHT + RELEASE_CLEARANCE
        if carry_z > release_z:
            self.move_linear(make_pose(bin_x, bin_y, release_z, q), 'lower')

    def bin_spot(self, piece, bin_index):
        """Low pieces land in the middle of a bin. Tall ones in a corner: in the middle they fall over and lean out of it."""
        x, y = BINS_XY[bin_index]
        if piece.height <= BIN_HEIGHT:
            return x, y
        side_x, side_y = CORNERS[self.tall_in_bin[bin_index] % len(CORNERS)]
        return x + side_x * BIN_CORNER, y + side_y * BIN_CORNER

    def find_pieces(self, table):
        """The pieces in the map, taking apart the ones that touch as far as their heights allow."""
        pieces = []
        for piece in table.segment():
            parts = [] if self.library.identify(piece)[0] else table.split(piece, self.library)
            pieces.extend(parts or [piece])
        return pieces

    def recognize(self, table, tried):
        """(piece, part number, grasp) of everything that can be taken, and how many pieces there are in all."""
        plans = []
        pieces = self.find_pieces(table)
        for _ in range(MAX_LOOKS):
            unknown = [piece for piece in pieces if not self.library.identify(piece)[0]]
            if not unknown:
                break
            try:
                self.look_closer(table, unknown[0])
            except RuntimeError as e:
                self.get_logger().warn(str(e))
                break
            again = self.find_pieces(table)
            if len([piece for piece in again if not self.library.identify(piece)[0]]) >= len(unknown):
                pieces = again
                break               # nothing more to see from up here
            pieces = again
        for piece in pieces:
            number, error = self.library.identify(piece)
            what = '{:.0f} x {:.0f} x {:.0f} mm at ({:.3f}, {:.3f})'.format(
                piece.length * 1000, piece.width * 1000, piece.height * 1000, piece.center[0], piece.center[1])
            if number is None:
                self.get_logger().warn('unknown piece, {}: left on the table'.format(what))
                continue
            grasp = table.grasp(piece, tried)
            if grasp is None:
                self.get_logger().warn('{}, {}: no way to hold it, left on the table'.format(number, what))
                continue
            self.get_logger().info('{}, {}: bin {}, grasp ({:.3f}, {:.3f}, {:.3f}) yaw {:.0f} deg, {:.0f} mm wide'.format(
                number, what, PIECE_BINS[number], grasp.x, grasp.y, grasp.z, math.degrees(grasp.yaw), grasp.width * 1000))
            plans.append((piece, number, grasp))
        return plans, len(pieces)

    def sort_table(self, dry_run):
        start = time.monotonic()
        self.wait_for_servers()
        end = time.monotonic() + 15.0
        while not self.finger_samples_since(0.0) or len(self.joint_limits) < len(ARM_JOINTS):
            if time.monotonic() > end:
                raise RuntimeError('no joint states or no robot_description published')
            time.sleep(0.05)
        self.clear_pieces()
        self.add_objects(self.table_object(), *[self.bin_object(i) for i in range(len(BINS_XY))])
        placed = 0
        total = 0
        tried = []          # grasps that have been used: a piece still on the table did not work with them
        for round_number in range(MAX_ROUNDS + 1):
            table = self.scan()
            plans, on_table = self.recognize(table, tried)
            if dry_run or not plans:
                self.look_around(table)
                plans, on_table = self.recognize(table, tried)
            total = max(total, placed + on_table)
            if dry_run or not plans or round_number == MAX_ROUNDS:
                break
            self.clear_pieces()
            self.add_objects(*[self.piece_object(piece, PIECE_MARGIN) for piece, _, _ in plans])
            plans.sort(key=lambda plan: -plan[0].height)   # the tall ones topple if something touches them
            for i, (piece, number, grasp) in enumerate(plans):
                tallest_left = plans[i + 1][0].height if i + 1 < len(plans) else 0.0
                dropped = False
                for _ in range(GRASP_ATTEMPTS):
                    tried.append(grasp)
                    try:
                        self.pick_and_drop(piece, grasp, PIECE_BINS[number], tallest_left)
                        placed += 1
                        if piece.height > BIN_HEIGHT:
                            self.tall_in_bin[PIECE_BINS[number]] += 1
                            self.bin_top = max(self.bin_top, BIN_WALL + piece.height)
                        self.get_logger().info('{} placed, {:.1f} s since the start'.format(number, time.monotonic() - start))
                        break
                    except PieceDropped as e:
                        self.get_logger().warn('{}: {}'.format(number, e))
                        dropped = True
                        break
                    except RuntimeError as e:
                        self.get_logger().warn('{}: {}'.format(number, e))
                    grasp = table.grasp(piece, tried)
                    if grasp is None:
                        break
                if dropped:
                    break       # it can have landed anywhere, even on the next piece: look at the table again
        self.gripper(GRIPPER_OPEN, 'open')
        missing = 0 if dry_run else max(total - placed - on_table, 0)
        self.get_logger().info('done: {} dropped into the bins, {} on the table, {} lost, {:.1f} s'.format(
            placed, on_table, missing, time.monotonic() - start))
        return on_table + missing


def main():
    parser = argparse.ArgumentParser(description='Sort the pieces on the table into their bins (xArm5 + gripper + wrist camera)')
    parser.add_argument('--dry-run', action='store_true', help='only scan, and print what is on the table')
    parser.add_argument('--depth-topic', default=DEPTH_TOPIC, help='depth image to use, e.g. the one of depth_noise.py')
    args = parser.parse_args(remove_ros_args(sys.argv)[1:])

    rclpy.init(args=sys.argv)
    node = PickAndPlace(args.depth_topic)
    try:
        if node.sort_table(args.dry_run) and not args.dry_run:
            sys.exit(1)
    except RuntimeError as e:
        node.get_logger().error(str(e))
        sys.exit(1)
    except KeyboardInterrupt:
        pass
    finally:
        node.close()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
