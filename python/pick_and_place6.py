#!/usr/bin/env python3
"""
Sort the pieces on the table of shared/world_xarm6.usd into their bins with MoveIt 2, finding them with the wrist camera
(xArm6 + xArm gripper + RealSense, xarm6_moveit_isaac.launch.py). It is pick_and_place.py, the sorter of the xArm5, with
what the sixth axis adds: the tool can point any way, not only straight down.

  - Straight lines turn the tool evenly from where it is to where it goes. The xArm5 has to work them out from the tool
    held exactly vertical, and make its wrist follow the base along the way.
  - A piece the arm cannot reach from straight above, at the far end of the belt, is taken with the gripper leaning back
    toward the base about the line between the fingers: they close across the piece as from above, and it goes into its
    bin leaning the same way, flat as it lay.
  - A piece at the edge of what the camera sees, out of reach from above, is looked at from as far as the arm reaches,
    the camera turned towards it.
  - The full bin of the rejects is emptied low over the belt. Held by the wall nearest to the belt, it is rolled about
    that wall (the fingers hold it firm that way, about the line between them it slips round), down under the fingers
    and up on the other side, with its lowest point always just over what lies on the belt: its pieces slide out over
    the opposite wall from a few centimeters. The xArm5 can only tip it about its wrist, which swings it high.

Everything else is pick_and_place.py (see pick_and_place.md): one depth image finds the pieces, piece_vision.py
recognizes them from the CAD, the parts of the mold in shared/mold.txt go into their bins, the bins are found at the
start, the web page of sorter_web.py shows and changes it all, --conveyor works with the belt round after round. The arm
needs its full range (limited:=false, the default of xarm6_moveit_isaac.launch.py), and so do the joints of the world.

Only moveit_msgs is needed (no moveit_py): the script talks to the move_group node through
  /move_action (MoveGroup), /compute_ik, /compute_fk, /compute_cartesian_path, /execute_trajectory,
  /apply_planning_scene, /get_planning_scene
and drives the gripper directly: the action of the robot driver (xarm_gripper/gripper_action) on the real arm, the
trajectory controller in simulation. Neither says anything about force, so everything it knows about a piece comes from
drive_joint in /joint_states: the fingers are sent past the piece, and it is there as long as they stay short of that.

Usage (from ros_ws):
  python3 pick_and_place6.py             # sort everything
  python3 pick_and_place6.py --dry-run   # only scan, and print what is on the table and how it would be taken
  python3 pick_and_place6.py --depth-topic /camera/aligned_depth_to_color/image_noisy   # with depth_noise.py running
  python3 pick_and_place6.py --conveyor  # round after round: waits for the belt to stop, sorts, tells it to go on (Ctrl-C ends)
  python3 pick_and_place6.py --mold 940017355   # change the mold in production, for the robot and the belt
"""

import argparse
import json
import math
import os
import sys
import threading
import time
import xml.etree.ElementTree as ET
from collections import deque
from pathlib import Path

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
                             PlanningScene, PlanningSceneComponents, RobotTrajectory)
from moveit_msgs.srv import ApplyPlanningScene, GetCartesianPath, GetPlanningScene, GetPositionFK, GetPositionIK
from sensor_msgs.msg import CameraInfo, Image, JointState
from shape_msgs.msg import SolidPrimitive
from std_msgs.msg import String
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from piece_vision import (FINGER_MARGIN, MAX_GAP, MIN_ENGAGE, PAD_WIDTH, TOP_PERCENTILE, PieceLibrary, TableMap,
                          finger_position, mold_parts)
from sorter_web import Board, WebServer, default_state_file

# ---------------------------------------------------------------- what goes where
MOLD_FILE = os.environ.get('MOLD_FILE', 'shared/mold.txt')  # the mold in production: its parts are what comes on the belt
BINS_CHANGED_FILE = 'shared/.bins_changed'  # bins changed on the web page, one per line, for Isaac Sim to empty them
WEB_PORT = 8080                     # of the web page: the bins, what is in them, what the robot is doing

# ---------------------------------------------------------------- robot
FRAME = 'link_base'                 # every position below is in this frame
ARM_GROUP = 'xarm6'
ARM_JOINTS = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
TURNING_JOINTS = ['joint1', 'joint4', 'joint6']     # one full turn away is the same pose of the arm
WRIST_JOINTS = ['joint4', 'joint5', 'joint6']
ARRIVAL_TOLERANCE = 0.03            # rad: the controller gives up at 0.01, and the grip force of the simulated gripper
                                    # alone pushes the wrist that far
TCP_LINK = 'link_tcp'               # between the fingertips, 0.172 m from the gripper base
CAMERA_LINK = 'camera_color_optical_frame'
DEPTH_TOPIC = '/aligned_depth_to_color/image_raw'   # end of the name: /camera/... in Isaac Sim, /camera/camera/... with
                                                    # realsense-ros 4 (started with align_depth.enable:=true)
CAMERA_INFO_TOPIC = '/color/camera_info'            # next to it
GRIPPER_ACTIONS = {'xarm_gripper/gripper_action': GripperCommand,     # end of the action name on the real arm
                   'xarm_gripper_traj_controller/follow_joint_trajectory': FollowJointTrajectory}   # in simulation
GRIPPER_JOINT = 'drive_joint'
GRIPPER_OPEN = 0.0
GRIPPER_CLOSED = 0.85               # fingers touching each other
GRIPPER_FORCE = 20.0                # N on the piece (the real gripper does 10 to 50): a few N hold these parts, more only
                                    # strains them; in Isaac Sim it is the maxForce of drive_joint in world.usd, 20 N x 2
                                    # fingers x 0.055 m of lever = 2.2 N*m (one motor drives both fingers)
GRIPPER_SPEED = 1.0                 # rad/s of the free motions of the fingers: faster, opening right after a grip tears
                                    # the simulated gripper linkage apart and takes the arm with it (URDF limit: 2)
GRIPPER_RELEASE = 0.3               # rad opened to let go of a piece: enough, and quick
GRIPPER_CLOSE_SPEED = 0.35          # rad/s closing on a piece: slow enough to stop on it instead of slamming it
GRIPPER_STALL_TIME = 0.2            # s without finger motion while closing = contact
GRIPPER_STALL_TOL = 0.005           # rad of finger motion still counted as stopped
GRIPPER_HELD_ERROR = 0.01           # rad short of the target = stopped by something, closer = nothing between the fingers
GRIPPER_MISS_MARGIN = 0.008         # m: the fingers are sent this much closer than the width of the piece
GRIPPER_PRESS = 0.07                # rad past the contact at least: with less, the fingers bending under the grip get so
                                    # close to the target that a piece still held looks gone
GRIPPER_FREE_SAMPLES = 3            # samples of drive_joint in a row at the target, while carrying = the piece is gone
GRIPPER_SETTLE_TIMEOUT = 5.0        # s of wall clock waiting for the fingers to stop after the lift
GRIPPER_TIMEOUT = 30.0              # s of wall clock: Isaac Sim can run much slower than real time
TOUCH_LINKS = ['xarm_gripper_base_link', 'left_outer_knuckle', 'left_finger', 'left_inner_knuckle',
               'right_outer_knuckle', 'right_finger', 'right_inner_knuckle', 'link_tcp']

# ---------------------------------------------------------------- scene (shared/world_xarm6.usd, as world.usd)
TABLE_TOP_Z = -0.006                # surface the pieces and the bins stand on
TABLE_SIZE = (0.8, 1.5, 0.03)
TABLE_CENTER_XY = (-0.34, -0.2)

# The bins are found at the start: looked for in SURVEY_X x SURVEY_Y, from SURVEY_VIEWS. They are numbered on the left
# (y > 0) from the nearest to the robot, then on the right the same way; the parts of the mold go into the first ones, the
# pieces nobody knows into the last one.
SURVEY_X = (-0.80, 0.06)
SURVEY_Y = (-0.60, 0.60)
# TCP x, y, yaw at the scan height: the camera over (-0.30, +-0.25) and (-0.45, +-0.25), turned so that it is further out
# than the TCP, where the arm still reaches. At the height of the rims an image covers less than on the table: from one
# view on each side the wall of a bin nearest to the robot was out of it, and the bin not closed.
SURVEY_VIEWS = [(-0.2675, 0.1825, math.pi / 2), (-0.4175, 0.1825, math.pi / 2),
                (-0.3325, -0.1825, -math.pi / 2), (-0.4825, -0.1825, -math.pi / 2)]

# where the camera looks for pieces: the belt (x -1.70..-0.16, y -0.10..0.10 in world.usd) from a little further than the
# arm reaches (-0.66), so that the last piece of a round is seen whole, and a little on its sides, up to the bins, without
# the base of the robot
WORK_X = (-0.74, -0.04)
WORK_Y = (-0.165, 0.165)
ROBOT_KEEP_OUT = (0.0, 0.0, 0.14)
MODELS = 'shared/piece_models.json'
BELT_FILE = 'shared/.conveyor_belt'     # what the belt of shared/conveyor_cycle.py is doing: moving, stopped or off
ROBOT_FILE = 'shared/.conveyor_robot'   # what this script is doing, for the belt: waiting, working or done
BELT_SILENCE = 120.0                    # s of wall clock without a new round of the belt before saying what may be wrong

# ---------------------------------------------------------------- motion
SCAN_XYZ = (-0.4525, 0.0675, 0.397)     # TCP, gripper pointing down and turned by SCAN_YAW: the camera (68 mm off the TCP,
SCAN_YAW = -math.pi / 2                 # 0.55 m above the belt) is over the middle of the pick zone, at (-0.42, 0), and the
                                        # long side of its image runs along the belt: it sees x -0.80..-0.04, y +-0.29
MORE_VIEWS = [(-0.3325, 0.0675), (-0.5725, 0.0675)]     # TCP x, y of two more views along the belt, same height and yaw:
                                                        # what a piece hides from one of them, the other sees
LOOK_REACH = 0.5                    # m from the base: further than this the arm cannot look straight down from SCAN_XYZ's height
MAX_LOOKS = 3                       # closer looks at what is not recognized, for each scan of the table
LOOK_TILT = math.radians(40)        # at most, the camera turned toward a piece out of reach from above
FREE_SPEED = 1.0                    # fraction of the joint limits
CARRY_SPEED = 1.0                   # holding a piece
LINEAR_SPEED = 1.0                  # straight moves near a piece
ACCELERATION = 1.0
ABOVE_PIECE = 0.035                 # fingertips above the top of the piece before going down
GRASP_TILTS = (math.radians(15), math.radians(30))     # the gripper leaning back toward the base, the least that works,
                                    # when the arm cannot take a piece from straight above
PLAN_ATTEMPTS = 3                   # a random planner can return a path that fails the final validation
LINEAR_ATTEMPTS = 3                 # the IK of a straight line gives up now and then, on the same line
LINE_DONE = 0.999                   # part of a straight line that counts as all of it
LINE_STEP = 0.005                   # m between the poses of a straight line: wider, the wrist turns in kinks that slow it all
MAX_SHORT_MOTION = 0.5              # rad of any joint for a short move: more means the arm would fold over
MAX_LINE_STEP = 0.5                 # rad of any joint between two points of a straight line: more is the IK flipping the
                                    # arm over, not a line
IK_ATTEMPTS = 5                     # the IK restarts from random seeds: some land on the arm reaching backwards
MIN_ELBOW = -math.pi                # joint3 further than this: the elbow folded the other way, reachable but a long way
TRAVEL_CLEARANCE = 0.06             # TCP above the bin rims and what stands in them, going for a piece
CARRY_CLEARANCE = 0.035             # bottom of a carried piece above the bin rims and the pieces on the table
RELEASE_CLEARANCE = 0.015           # bottom of a piece above the rim of its bin when let go: from higher it bounces out
BIN_CORNER_GAP = 0.02               # m from the inner walls of a bin to where a tall piece is dropped
# emptying the rejects on the belt (empty_on_belt, pour): a bin is taken by the wall nearest to the belt, lifted over
# everything, carried over the belt, rolled about the wall in the fingers until its pieces slide out over the opposite
# one, rolled back and put back where it was
DUMP_GRIP_DEPTH = 0.03              # m of the wall between the fingers, from the rim down
DUMP_BOTTOM = 0.17                  # m of the bottom of a lifted bin above the table: over the other bins and the belt
DUMP_AT = -0.40                     # x on the middle line of the belt where the pieces fall, in the middle of what the camera
                                    # sees: when the bin is turned along the belt; not turned, they fall in front of it
DUMP_TURNS = (math.radians(135), math.radians(125), math.radians(120))  # rolled this far: the wall the pieces slide out
                                    # on is then 45 to 30 deg steep, and friction holds nothing. Further, the gripper and
                                    # the wrist go lower than the rim and the bin has to stay higher
DUMP_FALL_STEP = 0.02               # m: falls this close count the same, and of those the farthest roll goes first
DUMP_CLEARANCE = (0.02, 0.05, 0.09, 0.14)   # m of the bin, the gripper and the wrist, all the way round, over what
                                    # lies highest on the belt: higher the arm has more room
DUMP_STEP = math.radians(10)        # between the poses of the roll
DUMP_SWEEP = 0.08                   # m the bin moves on while its pieces slide out, so that they do not land in a heap
DUMP_SWEEP_FROM = math.radians(80)  # of the roll: from here the pieces start sliding
DUMP_WRIST = ((0.172, 0.045), (0.27, 0.06))     # m up the gripper from the TCP, and m of room around: the gripper base
                                    # and the wrist, which go round with the bin and must stay over the belt as well
DUMP_SPEED = 0.5                    # rolling the bin: the pieces slide out instead of being flung
DUMP_LIMIT_MARGIN = 0.1             # rad the rolled arm keeps from the end of any joint: the bin sags it a little
DUMP_ARRIVAL = 0.1                  # rad off its goal an arm holding a bin still counts as arrived
DUMP_WAIT = 2.0                     # s of wall clock for what is in it to fall out
PIECE_MARGIN = 0.01                 # around the pieces on the table, for the planner
NEIGHBOR_GAP = 0.03                 # pieces closer than this to the one being taken: room for the fingers comes from the map
PLANNING_TIME = 2.0
GRASP_ATTEMPTS = 3                  # grasps of a piece in a row: when one misses the piece stays where it is
MAX_ROUNDS = 4                      # scans of the table with something left to do

PIECE_PREFIX = 'piece'
BIN_PREFIX = 'bin'
TABLE_ID = 'table'


def top_down(yaw):
    # Rz(yaw) * Rx(pi): TCP z axis pointing down, rotated by yaw about the vertical
    return Quaternion(x=math.cos(yaw / 2), y=math.sin(yaw / 2), z=0.0, w=0.0)


def rotation(q):
    return Rotation.from_quat([q.x, q.y, q.z, q.w])


def quaternion(r):
    x, y, z, w = r.as_quat()
    return Quaternion(x=float(x), y=float(y), z=float(z), w=float(w))


def pointing(direction, yaw):
    """The TCP with its z axis along direction, turned about it as close as can be to top_down(yaw)."""
    z = np.asarray(direction, float) / np.linalg.norm(direction)
    x = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    x -= z * (x @ z)
    x /= np.linalg.norm(x)
    return quaternion(Rotation.from_matrix(np.column_stack([x, np.cross(z, x), z])))


def leaning(yaw, tilt, x, y):
    """top_down(yaw) leaning back by tilt toward the base, seen from (x, y), about the line between the fingers (y of the
    TCP): they close across the piece as from above, only the gripper comes in at a slant."""
    along = math.cos(yaw) * x + math.sin(yaw) * y      # x of the TCP, the width of the pads, against the reach
    return quaternion(rotation(top_down(yaw)) * Rotation.from_euler('y', math.copysign(tilt, along)))


def duration(seconds):
    return Duration(sec=int(seconds), nanosec=int(seconds % 1 * 1e9))


def close_to(a, b, gap):
    """Two pieces on the table nearer than gap, give or take: the circles around their rectangles."""
    reach = (math.hypot(a.length, a.width) + math.hypot(b.length, b.width)) / 2
    return math.hypot(a.center[0] - b.center[0], a.center[1] - b.center[1]) < reach + gap


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

    def __init__(self, depth_topic=None, mold=None):
        super().__init__('pick_and_place')
        self.move_client = ActionClient(self, MoveGroup, 'move_action')
        self.execute_client = ActionClient(self, ExecuteTrajectory, 'execute_trajectory')
        self.gripper_client = None
        self.ik_client = self.create_client(GetPositionIK, 'compute_ik')
        self.fk_client = self.create_client(GetPositionFK, 'compute_fk')
        self.cartesian_client = self.create_client(GetCartesianPath, 'compute_cartesian_path')
        self.apply_scene_client = self.create_client(ApplyPlanningScene, 'apply_planning_scene')
        self.get_scene_client = self.create_client(GetPlanningScene, 'get_planning_scene')
        self.mold = None
        self.bins = []                              # found at the start, piece_vision.Bin
        self.library = None
        self.board = Board(default_state_file('shared'), mold_parts(MODELS), (WORK_X[0], WORK_X[1], WORK_Y[0], WORK_Y[1]))
        self.board.on_mold = self.mold_from_page
        self.board.on_changed = self.bin_changed
        self.recovered = None                       # pieces sorted since the rejects were last emptied on the belt
        self.load_mold(mold)
        self.joints = {}
        self.joints_stamp = 0.0
        self.joint_limits = {}
        self.tall_in_bin = []
        self.bin_top = 0.0                          # height above the table of the tallest piece standing in a bin
        self.create_subscription(String, 'robot_description', self.on_robot_description,
                                 QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        # (stamp [s], position) of drive_joint, stamps in simulated time with Isaac Sim
        self.finger_samples = deque(maxlen=1000)
        self.finger_lock = threading.Lock()
        self.create_subscription(JointState, 'joint_states', self.on_joint_states, 10)
        self.carrying = False                       # arm motions watch the grip meanwhile
        self.maybe_lost = None                      # the motion during which the fingers closed all the way, if they did
        self.neighbors = []                         # boxes of pieces out of the planning scene while one next to them is taken
        self.depth = None
        self.camera_info = None
        self.depth_topic = depth_topic              # None: the one found

        # Node.executor is taken by rclpy
        self.spin_executor = SingleThreadedExecutor()
        self.spin_executor.add_node(self)
        self.spin_thread = threading.Thread(target=self.spin_executor.spin, daemon=True)
        self.spin_thread.start()

    def close(self):
        self.spin_executor.shutdown()
        self.spin_thread.join(timeout=2.0)
        self.destroy_node()

    def load_mold(self, mold=None):
        """The mold in production, from MOLD_FILE, or the one given, which is written there for the belt."""
        if mold:
            temporary = Path(MOLD_FILE + '.tmp')
            temporary.write_text(mold + '\n')
            temporary.replace(MOLD_FILE)
        else:
            try:
                mold = Path(MOLD_FILE).read_text().strip()
            except OSError as e:
                raise RuntimeError('no mold: {} ({}), give one with --mold'.format(MOLD_FILE, e))
        if mold == self.mold:
            return
        parts = mold_parts(MODELS)
        if mold not in parts:
            raise RuntimeError('mold {} unknown, the ones in {}: {}'.format(mold, MODELS, ', '.join(sorted(parts))))
        self.library = PieceLibrary(MODELS, parts[mold])
        self.mold = mold
        self.board.set_mold(mold)
        if self.bins:
            self.log_bins_use()

    def log_bins_use(self):
        """What the bins are for (set on the web page), in the words batch_test.py reads."""
        parts = ['{} into bin {}'.format(code, self.board.bins_for(code)[0])
                 for code in self.board.molds[self.mold] if self.board.bins_for(code)]
        rejects = self.board.bins_for(None)
        self.get_logger().info('mold {}: {}, unknown pieces into bin {}'.format(
            self.mold, ', '.join(parts) or 'no bin for its parts', rejects[0] if rejects else 'none'))

    def mold_from_page(self, mold):
        """A mold chosen on the web page: for the belt, and for the robot at its next look at the table."""
        temporary = Path(MOLD_FILE + '.tmp')
        temporary.write_text(mold + '\n')
        temporary.replace(MOLD_FILE)

    def bin_changed(self, found):
        """A full bin changed for an empty one, says the web page. Isaac Sim empties it (shared/conveyor_cycle.py)."""
        with open(BINS_CHANGED_FILE, 'a') as f:
            f.write(json.dumps({'center': list(found['center']), 'length': found['length'], 'width': found['width'],
                                'angle': found['angle']}) + '\n')
        self.recovered = None

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

    def find_camera(self, timeout):
        """The aligned depth image of the wrist camera, wherever its driver puts it, and the camera info next to it."""
        end = time.monotonic() + timeout
        depth = self.depth_topic
        while depth is None:
            found = sorted(name for name, _ in self.get_topic_names_and_types() if name.endswith(DEPTH_TOPIC))
            if found:
                depth = found[0]
            elif time.monotonic() > end:
                raise RuntimeError('no topic ending in {}: is the camera driver running, with the depth aligned to the '
                                   'color image?'.format(DEPTH_TOPIC))
            else:
                time.sleep(0.2)
        info = depth[:depth.rindex('/aligned_depth_to_color/')] + CAMERA_INFO_TOPIC if '/aligned_depth_to_color/' in depth \
            else depth.rsplit('/', 2)[0] + CAMERA_INFO_TOPIC
        self.get_logger().info('camera: {}, {}'.format(depth, info))
        self.create_subscription(Image, depth, self.on_depth, qos_profile_sensor_data)
        self.create_subscription(CameraInfo, info, self.on_camera_info, qos_profile_sensor_data)

    def wait_for_servers(self, timeout=15.0):
        self.gripper_client, self.gripper_kind = self.find_gripper(timeout)
        self.find_camera(timeout)
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
        found = self.bins[index]
        lx, ly, t, h = found.length, found.width, found.wall, found.rim
        return self.collision_object(
            '{}{}'.format(BIN_PREFIX, index), make_pose(found.center[0], found.center[1], TABLE_TOP_Z,
                                                        Quaternion(z=math.sin(found.angle / 2), w=math.cos(found.angle / 2))),
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
        self.load_mold()                    # the web page can have changed it
        self.board.set_status('guarda il nastro')
        self.move_to(make_pose(*SCAN_XYZ, top_down(SCAN_YAW)), 'scan', FREE_SPEED)
        table = TableMap(TABLE_TOP_Z, WORK_X, WORK_Y, [ROBOT_KEEP_OUT])
        self.add_view(table)
        return table

    def look_around(self, table):
        """Add the views of the part of the table the first one does not cover."""
        for x, y in MORE_VIEWS:
            try:
                self.move_to(make_pose(x, y, SCAN_XYZ[2], top_down(SCAN_YAW)), 'look around', FREE_SPEED)
                self.add_view(table)
            except RuntimeError as e:
                self.get_logger().warn(str(e))

    def add_view(self, table):
        depth = self.depth_image()
        rotation, translation = self.link_pose(CAMERA_LINK)
        k = self.camera_info.k
        table.add_view(depth, k[0], k[4], k[2], k[5], rotation, translation)

    def look_closer(self, table, piece):
        """Add the view from above a piece: at the edge of the first one, or behind something else, it was only partly seen.
        Out of reach from above, from as far as the arm reaches, the camera turned towards it."""
        x, y = piece.center
        scale = min(LOOK_REACH / math.hypot(x, y), 1.0)
        tcp = np.array([x * scale, y * scale, SCAN_XYZ[2]])
        towards = np.array([x, y, TABLE_TOP_Z]) - tcp
        across = math.hypot(towards[0], towards[1])
        if across > -towards[2] * math.tan(LOOK_TILT):
            towards[:2] *= -towards[2] * math.tan(LOOK_TILT) / across
        try:
            self.move_to(make_pose(*tcp, pointing(towards, 0.0)), 'closer look', FREE_SPEED)
        except RuntimeError as e:
            if scale >= 1.0:
                raise
            self.get_logger().warn('{}: from straight above instead'.format(e))
            self.move_to(make_pose(*tcp, top_down(0.0)), 'closer look', FREE_SPEED)
        self.add_view(table)

    # ------------------------------------------------------------ motion

    def compute_ik(self, pose, what, avoid_collisions=True):
        """Joint positions of the arm for the TCP at pose, collision-free unless told otherwise, seeded from the current
        state."""
        request = GetPositionIK.Request()
        ik = request.ik_request
        ik.group_name = ARM_GROUP
        ik.ik_link_name = TCP_LINK
        # starting from the arm already facing the pose keeps the solver from folding it over to get there
        facing = math.atan2(pose.position.y, pose.position.x)
        ik.robot_state.is_diff = True
        ik.robot_state.joint_state.name = ['joint1']
        ik.robot_state.joint_state.position = [self.fewest_turns({'joint1': facing})['joint1']]
        ik.avoid_collisions = avoid_collisions
        ik.pose_stamped.header.frame_id = FRAME
        ik.pose_stamped.pose = pose
        ik.timeout.sec = 1
        for _ in range(IK_ATTEMPTS):
            response = self.call(self.ik_client, request, what)
            if response.error_code.val != MoveItErrorCodes.SUCCESS:
                raise RuntimeError('{}: no IK solution for ({:.3f}, {:.3f}, {:.3f}), out of reach or in '
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
        # to the joints of the IK, not to a pose goal: they come with the fewest turns of the joints that go all the way round
        self.get_logger().info('{}: moving to ({:.3f}, {:.3f}, {:.3f})'.format(
            what, pose.position.x, pose.position.y, pose.position.z))
        self.move_joints(self.compute_ik(pose, what), what, speed)

    def move_joints(self, joints, what, speed, tolerance=ARRIVAL_TOLERANCE):
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
        self.check_arrival(result, joints, what, tolerance)

    def check_arrival(self, result, joints, what, tolerance=ARRIVAL_TOLERANCE):
        if result.error_code.val == MoveItErrorCodes.CONTROL_FAILED and all(
                abs(self.joints[name] - position) < tolerance for name, position in joints.items()):
            return
        if result.error_code.val != MoveItErrorCodes.SUCCESS:
            raise RuntimeError('{}: MoveIt error {}'.format(what, result.error_code.val))

    def cartesian(self, poses, what, speed):
        """MoveIt's straight path from where the arm is through these poses, the tool turning evenly between them: how much
        of it the arm can follow, and the path."""
        request = GetCartesianPath.Request()
        request.header.frame_id = FRAME
        request.start_state.is_diff = True
        request.group_name = ARM_GROUP
        request.link_name = TCP_LINK
        request.waypoints = poses
        request.max_step = LINE_STEP
        request.revolute_jump_threshold = MAX_LINE_STEP
        request.avoid_collisions = True
        request.max_velocity_scaling_factor = speed
        request.max_acceleration_scaling_factor = ACCELERATION
        response = self.call(self.cartesian_client, request, what)
        times = [p.time_from_start.sec + p.time_from_start.nanosec * 1e-9 for p in response.solution.joint_trajectory.points]
        if response.error_code.val != MoveItErrorCodes.SUCCESS:
            return 0.0, response.solution
        if any(later <= earlier for earlier, later in zip(times, times[1:])):
            # all of it followed, but the time parametrization of MoveIt failed on it and left every time at 0: the
            # controller refuses that
            self.get_logger().warn('{}: MoveIt could not time the straight line'.format(what))
            return 0.0, response.solution
        return response.fraction, response.solution

    def execute(self, path, what, check=None, tolerance=ARRIVAL_TOLERANCE):
        """Carry out a path of MoveIt, from where the arm really is now."""
        trajectory = path.joint_trajectory
        trajectory.points[0].positions = [self.joints[name] for name in trajectory.joint_names]
        result = self.run_action(self.execute_client, ExecuteTrajectory.Goal(trajectory=path), what, check=check)
        self.check_arrival(result, dict(zip(trajectory.joint_names, trajectory.points[-1].positions)), what, tolerance)

    def follow_line(self, pose, what, speed):
        """Move the TCP along the straight line to pose if the arm can follow all of it. Returns how much of it it can."""
        for _ in range(LINEAR_ATTEMPTS):
            fraction, path = self.cartesian([pose], what, speed)
            if fraction >= LINE_DONE:
                self.get_logger().info('{}: straight line to ({:.3f}, {:.3f}, {:.3f})'.format(
                    what, pose.position.x, pose.position.y, pose.position.z))
                self.execute(path, what, check=self.check_grip)
                return fraction
        return fraction

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
            goal.command.max_effort = GRIPPER_FORCE     # the xArm driver of this workspace ignores it, the gripper keeps its own
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
        """Stop the lift as soon as the piece is gone. Over the bins only take note: a piece can still be held there."""
        if not self.carrying or not self.fingers_free(GRIPPER_FREE_SAMPLES):
            return
        if self.grip_target < GRIPPER_CLOSED:
            # A piece can turn or settle between the fingers, flat to on its side, and they close on it up to the target
            # while it is still held: the true poses in Isaac showed it lifted 10 cm with the fingers there.
            self.grip_target = GRIPPER_CLOSED
            self.gripper(self.grip_target, 'press', wait=False)
            self.get_logger().info('{}: the fingers closed further, pressing to {:.2f} rad'.format(what, self.grip_target))
            return
        if what != 'lift':
            # Closed all the way is not sure either: a thin part of a piece between the pads holds it, and Isaac showed
            # one carried into its bin like that. Opening here could drop it into another bin, so it goes on to its own.
            self.maybe_lost = self.maybe_lost or what
            return
        joints = dict(self.joints)          # the arm is well past this point by the time it has stopped
        self.carrying = False
        if motion:
            self.wait(motion.cancel_goal_async(), 5.0, what)
            self.wait(motion.get_result_async(), 10.0, what)
        position = self.link_pose(TCP_LINK, joints)[1]
        raise PieceDropped('lost during {}, around ({:.3f}, {:.3f}, {:.3f})'.format(what, *position))

    def check_grip_at_rest(self, what):
        """Check the grip once the fingers have stopped: a press can still be closing them on a piece that slipped."""
        for _ in range(2):          # again after a press
            target = self.grip_target
            begin = self.joints_stamp
            end = time.monotonic() + GRIPPER_SETTLE_TIMEOUT
            while time.monotonic() < end:
                recent = [p for t, p in self.finger_samples_since(self.joints_stamp - GRIPPER_STALL_TIME)]
                if recent and self.joints_stamp - begin >= GRIPPER_STALL_TIME and max(recent) - min(recent) < GRIPPER_STALL_TOL:
                    break
                time.sleep(0.01)
            self.check_grip(None, what)
            if self.grip_target == target:
                return

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
        if self.grip_target - self.grip_contact < GRIPPER_PRESS:    # the width was a guess, the contact is measured
            self.grip_target = min(self.grip_contact + GRIPPER_PRESS, GRIPPER_CLOSED)
            self.gripper(self.grip_target, 'press', wait=False)
        self.get_logger().info('close gripper: contact at {:.3f} rad, sent to {:.3f}'.format(self.grip_contact, self.grip_target))

    # ------------------------------------------------------------ task

    def wrist_turn(self, joints):
        return sum(abs(joints[name] - self.joints[name]) for name in WRIST_JOINTS)

    def closest_yaw(self, x, y, z, yaw):
        """Yaw for this grasp: the gripper is symmetric, half a turn is the same grasp, so the one the wrist is closest to."""
        options = []
        for option in (yaw, yaw + math.pi):
            try:
                options.append((self.wrist_turn(self.compute_ik(make_pose(x, y, z, top_down(option)), 'grasp')), option))
            except RuntimeError as e:
                error = e
        if not options:
            raise error
        return min(options)[1]

    def grasp_pose(self, grasp, above):
        """How to hold a piece: yaw, orientation and lean of the gripper, and how much higher the TCP goes for it. From
        straight above if the arm reaches the piece so, else leaning back toward the base, the least that works; the
        yaw the wrist is closest to."""
        z = TABLE_TOP_Z + grasp.z
        error = None
        for tilt in (0.0,) + GRASP_TILTS:
            higher = PAD_WIDTH / 2 * math.sin(tilt)     # the pads lean: their lower corners go that much deeper
            if tilt and grasp.engage - higher < MIN_ENGAGE:
                break                                   # the fingers would hold too little of it
            options = []
            for yaw in (grasp.yaw, grasp.yaw + math.pi):
                q = leaning(yaw, tilt, grasp.x, grasp.y) if tilt else top_down(yaw)
                try:
                    joints = self.compute_ik(make_pose(grasp.x, grasp.y, max(above, z + higher), q), 'above piece')
                    # down there the fingers are around the piece, which is in the planning scene until then
                    self.compute_ik(make_pose(grasp.x, grasp.y, z + higher, q), 'grasp', avoid_collisions=False)
                    options.append((self.wrist_turn(joints), yaw, q))
                except RuntimeError as e:
                    error = e
            if options:
                if tilt:
                    self.get_logger().info('grasp: out of reach from straight above, the gripper leaning back {:.0f} deg'.format(
                        math.degrees(tilt)))
                _, yaw, q = min(options, key=lambda option: option[0])
                return yaw, q, tilt, higher
        raise error

    def pick_and_drop(self, piece, grasp, bin_index, tallest_left, others=()):
        """Take a piece and let it fall into a bin, carrying it over everything left on the table."""
        z = TABLE_TOP_Z + grasp.z
        above = max(z, TABLE_TOP_Z + piece.height + ABOVE_PIECE)
        yaw, q, tilt, higher = self.grasp_pose(grasp, above)
        z += higher
        obj = self.piece_object(piece)
        self.gripper(finger_position(min(grasp.width + 2 * FINGER_MARGIN, MAX_GAP)), 'open', wait=False)
        # at the height of the rims the straight line back from a bin grazes the bins on the way, so down only over the piece
        travel = max(above, TABLE_TOP_Z + max(self.bin_rim(), self.bin_top) + TRAVEL_CLEARANCE)
        self.move_to(make_pose(grasp.x, grasp.y, travel, q), 'above piece', FREE_SPEED)
        # The fingers go around it, and next to the pieces it touches: the height map has room for them there, the boxes
        # of those with the margin for the planner do not, and the piece in the gripper would start in collision with them.
        self.neighbors = [self.piece_object(other, PIECE_MARGIN) for other in others if close_to(piece, other, NEIGHBOR_GAP)]
        self.remove_objects(obj.id, *[o.id for o in self.neighbors])
        try:
            self.move_linear(make_pose(grasp.x, grasp.y, z, q), 'approach')
            self.close_on(grasp.width)
            self.attach_object(obj)
            self.carrying = True
            self.maybe_lost = None
            try:
                self.carry_to_bin(piece, grasp, bin_index, tallest_left, z, yaw, q, tilt)
            finally:
                self.carrying = False
                self.gripper(self.grip_contact - GRIPPER_RELEASE, 'release')    # never go on with a piece in the gripper
                self.detach_object(obj.id)
        finally:
            self.restore_neighbors()
        if self.maybe_lost:
            raise PieceDropped('the fingers closed all the way during {}: the piece may have fallen on the way'.format(self.maybe_lost))

    def restore_neighbors(self):
        if self.neighbors:
            self.add_objects(*self.neighbors)
            self.neighbors = []

    def carry_to_bin(self, piece, grasp, bin_index, tallest_left, z, yaw, q, tilt):
        carry_z = z + max(self.bin_rim(), self.bin_top, tallest_left) + CARRY_CLEARANCE
        try:
            self.move_linear(make_pose(grasp.x, grasp.y, carry_z, q), 'lift')
            self.check_grip_at_rest('lift')     # still over the table: after this, the fingers never open before the bin
            self.restore_neighbors()
            self.board.set_phase('carry')
        except PieceDropped as e:
            # it falls back from up to the carry height and lands centimeters away: grasps from the old map miss it
            raise PieceDropped('the piece slipped out: {}'.format(e))
        # the wrist stays as it is and the piece turns with the arm, which the bin does not mind: the center of the
        # piece goes over the center of the bin, and the TCP keeps its offset from it, turned as well
        center_x, center_y = self.bin_spot(piece, bin_index)
        turn = wrapped(math.atan2(center_y, center_x) - math.atan2(grasp.y, grasp.x))
        offset_x, offset_y = grasp.x - piece.center[0], grasp.y - piece.center[1]
        bin_x = center_x + offset_x * math.cos(turn) - offset_y * math.sin(turn)
        bin_y = center_y + offset_x * math.sin(turn) + offset_y * math.cos(turn)
        q = top_down(yaw + turn)
        lower = 0.0
        if tilt:
            # taken leaning, it goes in leaning the same way, flat as it lay. If the arm cannot do that over the bin, it
            # goes in from straight above: the piece then hangs tilted, and its lowest corner is lower
            leaned = leaning(yaw + turn, tilt, bin_x, bin_y)
            try:
                self.compute_ik(make_pose(bin_x, bin_y, carry_z, leaned), 'over bin {}'.format(bin_index))
                q = leaned
            except RuntimeError:
                lower = max(piece.length, piece.width) / 2 * math.sin(tilt)
        self.move_to(make_pose(bin_x, bin_y, carry_z + lower, q), 'over bin {}'.format(bin_index), CARRY_SPEED)
        # everything drops in from the rim: going lower would take the fingers over it, or into the bin
        release_z = z + lower + self.bins[bin_index].rim + RELEASE_CLEARANCE
        if carry_z + lower > release_z:
            self.move_linear(make_pose(bin_x, bin_y, release_z, q), 'lower')

    def bin_rim(self):
        """Top of the highest bin above the table: what the arm passes over."""
        return max(found.rim for found in self.bins)

    def bin_spot(self, piece, bin_index):
        """Low pieces land in the middle of a bin. Tall ones in a corner: in the middle they fall over and lean out of it.
        The corners far from the robot first: the arm then does not pass over what stands in them."""
        found = self.bins[bin_index]
        x, y = found.center
        if piece.height <= found.rim:
            return x, y
        c, s = math.cos(found.angle), math.sin(found.angle)
        du, dv = max(found.length / 2 - BIN_CORNER_GAP, 0.0), max(found.width / 2 - BIN_CORNER_GAP, 0.0)
        corners = sorted(((x + u * c - v * s, y + u * s + v * c) for u in (-du, du) for v in (-dv, dv)),
                         key=lambda corner: -math.hypot(*corner))
        return corners[self.tall_in_bin[bin_index] % len(corners)]

    def bins_text(self, number):
        """Where a piece of this number goes, for the log."""
        if number and self.board.bins_for(number):
            return 'bin {}'.format(', '.join(str(i) for i in self.board.bins_for(number)))
        rejects = self.board.bins_for(None)
        return 'reject bin {}'.format(', '.join(str(i) for i in rejects)) if rejects else 'no bin'

    def bin_for(self, number):
        """The bin a piece goes into now: one of its bins with room; for a piece no bin is for, the rejects. When they
        are all full the robot waits until the web page says one was changed."""
        if number is None or not self.board.bins_for(number):
            number = None
        return self.board.target(number)

    def empty_full_rejects(self):
        """Empty a full bin of rejects on the belt: True if it did. Not when none of those emptied the last time came out
        known: then it waits to be changed, like any other full bin."""
        for index in self.board.bins_for(None):
            if self.board.full(index) and self.recovered != 0:
                return self.empty_on_belt(index)
        return False

    def empty_on_belt(self, index):
        """Take a bin by its wall nearest to the belt, roll it over above the belt until its pieces slide out, put it back.
        True if it did; if the arm cannot roll it, the bin waits to be changed, like any other full bin."""
        found = self.bins[index]
        on_belt = self.on_belt_height()             # before taking it: what the bin has to keep clear of
        self.board.set_status('svuota la cassa {} sul nastro'.format(index))
        self.get_logger().info('bin {} is full of rejects: emptying it on the belt, where what lies highest is {:.0f} mm '
                               'high'.format(index, on_belt * 1000))
        belt_x, belt_y = (WORK_X[0] + WORK_X[1]) / 2, (WORK_Y[0] + WORK_Y[1]) / 2
        c, s = math.cos(found.angle), math.sin(found.angle)
        out_l, out_w = (found.length + found.wall) / 2, (found.width + found.wall) / 2
        walls = [((found.center[0] + u * c - v * s, found.center[1] + u * s + v * c), normal)
                 for u, v, normal in ((out_l, 0, found.angle), (-out_l, 0, found.angle),
                                      (0, out_w, found.angle + math.pi / 2), (0, -out_w, found.angle + math.pi / 2))]
        (gx, gy), normal = min(walls, key=lambda w: math.hypot(w[0][0] - belt_x, w[0][1] - belt_y))
        grip_z = TABLE_TOP_Z + found.rim - DUMP_GRIP_DEPTH
        lift_z = grip_z + DUMP_BOTTOM               # the whole bin goes up as much
        travel = TABLE_TOP_Z + max(self.bin_rim(), self.bin_top) + TRAVEL_CLEARANCE
        yaw = self.closest_yaw(gx, gy, travel, normal + math.pi / 2)   # the fingers close across the wall
        q = top_down(yaw)
        obj = self.bin_object(index)
        self.gripper(finger_position(min(found.wall + 2 * FINGER_MARGIN, MAX_GAP)), 'open', wait=False)
        self.move_to(make_pose(gx, gy, travel, q), 'above the wall of bin {}'.format(index), FREE_SPEED)
        self.remove_objects(obj.id)                 # the fingers go around its wall
        holding = False
        try:
            self.move_linear(make_pose(gx, gy, grip_z, q), 'onto the wall')
            self.close_on(found.wall)
            self.attach_object(obj)
            holding = True
            self.move_linear(make_pose(gx, gy, lift_z, q), 'lift the bin')
            poured = self.pour(found, (gx, gy, grip_z), yaw, lift_z, on_belt)
            self.move_linear(make_pose(gx, gy, grip_z + 0.003, q), 'put the bin down')
        finally:
            self.gripper(self.finger_samples_since(0.0)[-1][1] - GRIPPER_RELEASE, 'release')
            if holding:
                self.detach_object(obj.id)
            self.add_objects(self.bin_object(index))
        self.move_linear(make_pose(gx, gy, travel, q), 'leave the bin')
        self.recovered = 0
        if not poured:
            self.get_logger().warn('bin {}: the arm cannot roll it over above the belt, put back full: it has to be '
                                   'changed'.format(index))
            self.board.set_status('cassa {}: non si riesce a svuotarla sul nastro, va cambiata'.format(index))
            return False
        self.board.emptied(index)
        self.tall_in_bin[index] = 0
        self.get_logger().info('bin {} emptied on the belt'.format(index))
        return True

    def on_belt_height(self):
        """Height above the table of what lies highest on the belt, pieces or its sides: the bin rolls over it."""
        table = self.scan()
        height = np.divide(table.views, table.view_count, out=np.zeros(table.shape), where=table.view_count > 0)
        return float(np.percentile(height[table.view_count > 0], TOP_PERCENTILE)) if table.view_count.any() else 0.0

    def pour(self, found, grip, yaw, lift_z, on_belt):
        """Roll the bin held at grip over the belt until its pieces slide out, and back to where it was lifted: True if it
        could. It rolls about the wall in the fingers, down first, under them, and up on the other side: its top turns
        toward where the bin hung from the fingers, the gripper and the wrist go round with it the other way, and the
        pieces slide out over the opposite wall. All the way round, its lowest point stays just over what lies on the
        belt. Over the belt the bin hangs as it was taken, its pieces falling in front of where it stood, or turned along
        the belt, toward the robot or away from it. How far it rolls, how low and which way: the first one the arm can
        follow, of those the pieces fall from lowest the farthest roll first."""
        gx, gy, gz = grip
        center = np.array([found.center[0] - gx, found.center[1] - gy])
        into = center / np.linalg.norm(center)      # from the wall in the fingers into the bin
        c, s = math.cos(found.angle), math.sin(found.angle)
        out_l, out_w = found.length / 2 + found.wall, found.width / 2 + found.wall
        # what goes round, from the TCP, as the bin hangs from it: its corners, and the gripper and the wrist above
        points = [(np.array([found.center[0] + u * c - v * s - gx, found.center[1] + u * s + v * c - gy, z - gz]), 0.0)
                  for u in (-out_l, out_l) for v in (-out_w, out_w) for z in (TABLE_TOP_Z, TABLE_TOP_Z + found.rim)]
        points += [(np.array([0.0, 0.0, up]), room) for up, room in DUMP_WRIST]
        lip = np.array([2 * center[0], 2 * center[1], TABLE_TOP_Z + found.rim - gz])    # the rim the pieces slide over
        # which way the bin points over the belt, from the wall in the fingers, and x where its pieces fall
        ways = [(into, found.center[0]), (np.array([1.0, 0.0]), DUMP_AT), (np.array([-1.0, 0.0]), DUMP_AT)]
        options = []
        for angle in DUMP_TURNS:
            steps = int(math.ceil(angle / DUMP_STEP))
            for way, fall_x in ways:
                turn = wrapped(math.atan2(way[1], way[0]) - math.atan2(into[1], into[0]))
                carried = Rotation.from_euler('z', turn)
                axis = np.array([-way[1], way[0], 0.0])         # up x way: along the wall in the fingers
                rolls = [Rotation.from_rotvec(axis * angle * k / steps) for k in range(steps + 1)]
                end = (rolls[-1] * carried).apply(lip)
                x, y = fall_x - end[0], -end[1]                  # the pieces fall on the middle line of the belt
                # while they slide out the bin moves on, toward where it rolls: they land along a line, not in a heap
                moved = [[((r * carried).apply(p), room) for p, room in points] for r in rolls]
                spots = []
                for k in range(steps + 1):
                    on = min(max((angle * k / steps - DUMP_SWEEP_FROM) / (angle - DUMP_SWEEP_FROM), 0.0), 1.0) - 0.5
                    spots.append((x - way[0] * DUMP_SWEEP * on, y - way[1] * DUMP_SWEEP * on))
                for clearance in DUMP_CLEARANCE:
                    # the TCP as low as every point leaves room for, over what is under it: the belt and what lies on it,
                    # or the table beside it, where the bin stood (the other bins are in the planning scene)
                    heights = [TABLE_TOP_Z + clearance + max(self.floor_at(sx + q[0], sy + q[1], on_belt) + room - q[2]
                                                             for q, room in pose) for pose, (sx, sy) in zip(moved, spots)]
                    fall = heights[-1] + end[2] - (TABLE_TOP_Z + on_belt)  # of the rim they slide over, above the belt
                    options.append((int(fall / DUMP_FALL_STEP), -angle, fall, clearance, way, spots, rolls, carried, heights))
        options.sort(key=lambda option: option[:2])             # the lowest fall, and of those the farthest roll
        hanging_from = rotation(top_down(yaw))
        chosen = None
        for _, minus_angle, fall, clearance, way, spots, rolls, carried, heights in options:
            hanging = carried * hanging_from
            poses = [make_pose(*spots[0], lift_z, quaternion(hanging))]     # over the belt first, as high as it was lifted
            poses += [make_pose(*spot, z, quaternion(r * hanging)) for spot, r, z in zip(spots, rolls, heights)]
            fraction, path = self.cartesian(poses, 'roll the bin over', DUMP_SPEED)
            margin = self.limit_margin(path) if fraction >= LINE_DONE else 0.0
            way_text = '{:.0f} deg, {:.0f} mm clear, bin pointing {:.0f} deg, the pieces sliding out {:.0f} mm over the belt'.format(
                -math.degrees(minus_angle), clearance * 1000, math.degrees(math.atan2(way[1], way[0])), fall * 1000)
            if fraction >= LINE_DONE and margin >= DUMP_LIMIT_MARGIN:
                chosen = path
                break
            self.get_logger().info('roll {}: {:.0f}% of it possible, {:.2f} rad from the end of a joint'.format(
                way_text, fraction * 100, margin))
        if not chosen:
            return False
        self.get_logger().info('rolling the bin over: {}'.format(way_text))
        rolled = True
        try:
            self.execute(chosen, 'roll the bin over', tolerance=DUMP_ARRIVAL)
        except RuntimeError as e:
            self.get_logger().warn(str(e))
            rolled = False
        if rolled:
            time.sleep(DUMP_WAIT)
        # back the way it went, to over its place: a new line from the rolled bin has to be worked out from an arm
        # sagging under it, the path it followed is known to be free
        try:
            self.retrace(chosen, 'roll the bin back')
        except RuntimeError as e:
            self.get_logger().warn(str(e))
            start = chosen.joint_trajectory
            self.move_joints(dict(zip(start.joint_names, start.points[0].positions)), 'bin back over its place', CARRY_SPEED,
                             DUMP_ARRIVAL)
        return rolled

    @staticmethod
    def floor_at(x, y, on_belt):
        """Height above the table of what is under (x, y), for a bin rolled over the belt."""
        return on_belt if WORK_X[0] < x < WORK_X[1] and abs(y) < WORK_Y[1] else 0.0

    def limit_margin(self, path):
        """How close a path gets to the end of any joint [rad]."""
        names = path.joint_trajectory.joint_names
        margin = math.inf
        for point in path.joint_trajectory.points:
            for name, position in zip(names, point.positions):
                low, high = self.joint_limits.get(name, (-math.inf, math.inf))
                margin = min(margin, position - low, high - position)
        return margin

    def retrace(self, path, what):
        """Back along a path the arm has followed, all of it or as far as it got, to where it began."""
        names = path.joint_trajectory.joint_names
        points = path.joint_trajectory.points
        here = [self.joints[name] for name in names]
        reached = min(range(len(points)), key=lambda i: max(abs(a - b) for a, b in zip(points[i].positions, here)))
        if reached == 0:
            return
        stamp = [p.time_from_start.sec + p.time_from_start.nanosec * 1e-9 for p in points]
        back = [JointTrajectoryPoint(positions=list(p.positions), velocities=[-v for v in p.velocities],
                                     accelerations=list(p.accelerations), time_from_start=duration(stamp[reached] - t))
                for p, t in zip(points[reached::-1], stamp[reached::-1])]
        back[0].positions = here
        for point in (back[0], back[-1]):       # still at both ends: the controller refuses a last point that is not
            point.velocities = [0.0] * len(names)
            point.accelerations = [0.0] * len(names)
        trajectory = RobotTrajectory(joint_trajectory=JointTrajectory(joint_names=list(names), points=back))
        result = self.run_action(self.execute_client, ExecuteTrajectory.Goal(trajectory=trajectory), what)
        self.check_arrival(result, dict(zip(names, back[-1].positions)), what, DUMP_ARRIVAL)

    def find_bins(self):
        """How many bins there are, where and how large: looking from above at both sides of the belt and over it."""
        objects = self.call(self.get_scene_client, GetPlanningScene.Request(components=PlanningSceneComponents(
            components=PlanningSceneComponents.WORLD_OBJECT_NAMES)), 'get planning scene').scene.world.collision_objects
        old = [o.id for o in objects if o.id.startswith(BIN_PREFIX)]
        if old:
            self.remove_objects(*old)               # of a run before: the bins may have moved since
        table = TableMap(TABLE_TOP_Z, SURVEY_X, SURVEY_Y, [ROBOT_KEEP_OUT])
        for x, y, yaw in SURVEY_VIEWS:
            self.move_to(make_pose(x, y, SCAN_XYZ[2], top_down(yaw)), 'look for the bins', FREE_SPEED)
            self.add_view(table)
        found = table.bins()
        left = sorted((b for b in found if b.center[1] > 0), key=lambda b: math.hypot(*b.center))
        right = sorted((b for b in found if b.center[1] <= 0), key=lambda b: math.hypot(*b.center))
        self.bins = left + right
        if len(self.bins) < 2:
            raise RuntimeError('{} bins found, at least two are needed: one for the parts, one for the rejects'.format(len(self.bins)))
        self.tall_in_bin = [0] * len(self.bins)
        for i, b in enumerate(self.bins):
            self.get_logger().info('bin {}: {:.0f} x {:.0f} mm inside, rim {:.0f} mm, at ({:.3f}, {:.3f}), turned {:.0f} deg'.format(
                i, b.length * 1000, b.width * 1000, b.rim * 1000, b.center[0], b.center[1], math.degrees(b.angle) % 180))
        self.board.set_bins(self.bins)

    def find_pieces(self, table):
        """The pieces in the map, taking apart the ones that touch as far as their heights allow."""
        pieces = []
        for piece in table.segment():
            parts = [] if self.library.identify(piece)[0] else table.split(piece, self.library)
            pieces.extend(parts or [piece])
        return pieces

    def unsure(self, piece):
        return piece.partial or not self.library.identify(piece)[0]

    def recognize(self, table, tried):
        """(piece, part number, grasp) of everything that can be taken, and how many pieces there are in all."""
        plans = []
        pieces = self.find_pieces(table)
        for _ in range(MAX_LOOKS):
            unknown = [piece for piece in pieces if self.unsure(piece)]
            if not unknown:
                break
            try:
                self.look_closer(table, unknown[0])
            except RuntimeError as e:
                self.get_logger().warn(str(e))
                break
            again = self.find_pieces(table)
            if len([piece for piece in again if self.unsure(piece)]) >= len(unknown):
                pieces = again
                break               # nothing more to see from up here
            pieces = again
        seen = []           # all of them, for the page: (piece, part number, whether it is taken)
        for piece in pieces:
            number, error = self.library.identify(piece)
            what = '{:.0f} x {:.0f} x {:.0f} mm at ({:.3f}, {:.3f})'.format(
                piece.length * 1000, piece.width * 1000, piece.height * 1000, piece.center[0], piece.center[1])
            if piece.partial:   # at the edge of what the camera reaches: a part of one piece can look like a whole other one
                self.get_logger().warn('unknown piece, {}: only partly in view, left on the table'.format(what))
                seen.append((piece, None, False))
                continue
            name = number or 'unknown piece'
            grasp = table.grasp(piece, tried)
            if grasp is None:
                self.get_logger().warn('{}, {}: no way to hold it, left on the table'.format(name, what))
                seen.append((piece, number, False))
                continue
            self.get_logger().info('{}, {}: {}, grasp ({:.3f}, {:.3f}, {:.3f}) yaw {:.0f} deg, {:.0f} mm wide'.format(
                name, what, self.bins_text(number),
                grasp.x, grasp.y, grasp.z, math.degrees(grasp.yaw), grasp.width * 1000))
            plans.append((piece, number, grasp))
            seen.append((piece, number, True))
        self.board.set_pieces(seen)
        return plans, len(pieces)

    def prepare(self):
        """Servers, and the table and the bins in the planning scene."""
        self.wait_for_servers()
        end = time.monotonic() + 15.0
        while not self.finger_samples_since(0.0) or len(self.joint_limits) < len(ARM_JOINTS):
            if time.monotonic() > end:
                raise RuntimeError('no joint states or no robot_description published')
            time.sleep(0.05)
        self.board.set_status('cerca le casse')
        self.find_bins()
        self.log_bins_use()
        self.clear_pieces()
        self.add_objects(self.table_object(), *[self.bin_object(i) for i in range(len(self.bins))])

    def sort_pieces(self, dry_run=False):
        """Clear the table: returns how many pieces went into their bins, into the reject bin, and how many are left."""
        start = time.monotonic()
        placed = 0
        rejected = 0
        total = 0
        tried = []          # grasps that have been used: a piece still on the table did not work with them
        self.empty_full_rejects()               # full from before: after a restart, or a run that ended there
        for round_number in range(MAX_ROUNDS + 1):
            table = self.scan()
            plans, on_table = self.recognize(table, tried)
            if dry_run or not plans:
                self.look_around(table)
                plans, on_table = self.recognize(table, tried)
            total = max(total, placed + rejected + on_table)
            if dry_run or not plans or round_number == MAX_ROUNDS:
                break
            # The unknown ones only when nothing known is left: often two touching pieces, which come apart once the
            # ones next to them are gone, and are known on the next look.
            plans = [plan for plan in plans if plan[1]] or plans
            self.clear_pieces()
            self.add_objects(*[self.piece_object(piece, PIECE_MARGIN) for piece, _, _ in plans])
            plans.sort(key=lambda plan: -plan[0].height)   # the tall ones topple if something touches them
            for i, (piece, number, grasp) in enumerate(plans):
                tallest_left = plans[i + 1][0].height if i + 1 < len(plans) else 0.0
                name = number or 'unknown piece'
                bin_index = self.bin_for(number)
                if bin_index is None:
                    self.get_logger().warn('{}: no bin is for it, left on the table'.format(name))
                    continue
                known = self.board.use[bin_index] == number
                dropped = False
                for _ in range(GRASP_ATTEMPTS):
                    tried.append(grasp)
                    try:
                        self.board.set_status('prende {} per la cassa {}'.format(number[-5:] if number else 'uno sconosciuto', bin_index),
                                              {'code': number, 'from': list(piece.center), 'to': bin_index, 'phase': 'pick'})
                        self.pick_and_drop(piece, grasp, bin_index, tallest_left, [p for p, _, _ in plans[i + 1:]])
                        self.board.add(bin_index, number if known else None)
                        self.board.set_status('{} nella cassa {}'.format(number[-5:] if known else 'scarto', bin_index))
                        if piece.height > self.bins[bin_index].rim:
                            self.tall_in_bin[bin_index] += 1
                            self.bin_top = max(self.bin_top, self.bins[bin_index].wall + piece.height)
                        if known:
                            placed += 1
                            if self.recovered is not None:
                                self.recovered += 1
                            self.get_logger().info('{} placed, {:.1f} s since the start'.format(number, time.monotonic() - start))
                        else:
                            rejected += 1
                            self.get_logger().info('unknown piece into the reject bin, {:.1f} s since the start'.format(
                                time.monotonic() - start))
                            dropped = self.empty_full_rejects()     # then they are on the belt: look at it again
                        break
                    except PieceDropped as e:
                        self.get_logger().warn('{}: {}'.format(name, e))
                        self.board.set_status('{}: perso, guarda di nuovo'.format(name))
                        dropped = True
                        break
                    except RuntimeError as e:
                        self.get_logger().warn('{}: {}'.format(name, e))
                        self.board.set_status('{}: {}'.format(name, e))
                    grasp = table.grasp(piece, tried)
                    if grasp is None:
                        break
                if dropped:
                    break       # it can have landed anywhere, even on the next piece: look at the table again
        self.gripper(GRIPPER_OPEN, 'open')
        self.board.set_status('fatto: {} nelle casse, {} negli scarti, {} lasciati'.format(placed, rejected, on_table))
        missing = 0 if dry_run else max(total - placed - rejected - on_table, 0)
        self.get_logger().info('done: {} dropped into the bins, {} into the reject bin, {} on the table, {} lost, {:.1f} s'.format(
            placed, rejected, on_table, missing, time.monotonic() - start))
        return placed, rejected, on_table + missing

    def sort_table(self, dry_run):
        """How many pieces did not go into their bins."""
        self.prepare()
        _, rejected, left = self.sort_pieces(dry_run)
        return rejected + left

    def belt_says(self):
        try:
            return Path(BELT_FILE).read_text().split()
        except OSError:
            return []

    def tell_belt(self, *state):
        """Written whole in one go: the belt must never read half of it."""
        temporary = Path(ROBOT_FILE + '.tmp')
        temporary.write_text(' '.join(state))
        temporary.replace(ROBOT_FILE)

    def follow_conveyor(self):
        """Round after round: a piece is never looked for on a moving belt, and the belt does not move under the robot."""
        self.prepare()
        self.move_to(make_pose(*SCAN_XYZ, top_down(SCAN_YAW)), 'scan', FREE_SPEED)
        self.tell_belt('waiting')
        self.board.set_status('aspetta che il nastro si fermi')
        self.get_logger().info('waiting for the belt to stop')
        last = None                 # the number of the round, not the moving in between, which can pass unseen
        placed = 0
        rounds = 0
        waiting_since = time.monotonic()
        try:
            while True:
                belt = self.belt_says()
                if time.monotonic() - waiting_since > BELT_SILENCE:
                    waiting_since = time.monotonic()
                    self.get_logger().warn(
                        'still waiting for the belt, {} says {}. If the belt is stopped in Isaac Sim: it writes that file next to '
                        'the world it plays, so open /workspace/world_xarm6.usd; its container user must be able to write there '
                        '(setfacl -m u:1234:rwx,d:u:1234:rwx shared, from ros_ws); and Isaac Sim must have started with '
                        '--/app/scripting/ignoreWarningDialog=true, or the belt script does not run'.format(
                            os.path.abspath(BELT_FILE), ' '.join(belt) or 'nothing (not there)'))
                if belt[:1] == ['off']:
                    last = None         # Stop in Isaac Sim: its script may start counting again
                if belt[:1] == ['stopped'] and belt[1] != last:
                    last = belt[1]
                    rounds += 1
                    self.load_mold()    # it can change between rounds
                    self.tell_belt('working', last)
                    self.get_logger().info('belt stopped, round {}'.format(last))
                    try:
                        done, _, left = self.sort_pieces()
                        placed += done
                        if left:
                            self.get_logger().warn('round {}: {} pieces left that the robot cannot take, the belt carries '
                                                   'what is still on it off its end'.format(last, left))
                    except RuntimeError as e:
                        self.get_logger().error('round {}: {}'.format(last, e))
                    self.tell_belt('done', last)
                    self.board.set_pieces([])      # the belt carries off what is left
                    self.board.set_status('giro {} finito, aspetta il nastro'.format(last))
                    self.get_logger().info('{} rounds, {} pieces dropped into the bins so far'.format(rounds, placed))
                    waiting_since = time.monotonic()
                time.sleep(0.1)
        finally:
            Path(ROBOT_FILE).unlink(missing_ok=True)    # gone: the belt goes on by itself


def main():
    parser = argparse.ArgumentParser(description='Sort the pieces on the table into their bins (xArm6 + gripper + wrist camera)')
    parser.add_argument('--dry-run', action='store_true', help='only scan, and print what is on the table')
    parser.add_argument('--depth-topic', help='depth image to use, e.g. the one of depth_noise.py (default: the topic ending '
                                              'in {})'.format(DEPTH_TOPIC))
    parser.add_argument('--conveyor', action='store_true', help='work with the belt round after round, until Ctrl-C')
    parser.add_argument('--web-port', type=int, default=WEB_PORT, help='of the web page (0: none)')
    parser.add_argument('--mold', help='the mold in production (e.g. 940016509): written into {} for the belt as well, which '
                                       'otherwise says it'.format(MOLD_FILE))
    args = parser.parse_args(remove_ros_args(sys.argv)[1:])

    rclpy.init(args=sys.argv)
    node = PickAndPlace(args.depth_topic, args.mold)
    if args.web_port:
        WebServer(node.board, args.web_port)
        node.get_logger().info('web page: http://<this PC>:{}'.format(args.web_port))
    try:
        if args.conveyor:
            node.follow_conveyor()
        elif node.sort_table(args.dry_run) and not args.dry_run:
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
