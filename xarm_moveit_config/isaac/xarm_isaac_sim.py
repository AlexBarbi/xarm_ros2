#!/usr/bin/env python3
# Software License Agreement (BSD License)
#
# Isaac Sim side of xarm_moveit_config/launch/_robot_moveit_isaac.launch.py
#
# Run with the Isaac Sim python (written for Isaac Sim 4.5 / 5.x / 6.x, legacy 4.2 names as fallback):
#   <isaac_sim>/python.sh /tmp/xarm_isaac/xarm_isaac_sim.py --urdf /tmp/xarm_isaac/xarm6.urdf
# The launch file copies this script, the URDF and its meshes into /tmp/xarm_isaac/ (mounted as
# /xarm_isaac when Isaac Sim runs in a container).
#
# Builds the xarm_gazebo table world around the robot (robot base = world origin, same as MoveIt),
# imports the URDF written by the launch file and wires a ROS 2 action graph:
#   Isaac -> ROS 2: /clock, /isaac_joint_states
#   ROS 2 -> Isaac: /isaac_joint_commands (position + velocity targets from topic_based_ros2_control),
#                   one topic per ros2_control hardware block, e.g. /isaac_joint_commands/XArmGripperSystem
#   RealSense (camera at the URDF camera_color_optical_frame, topics of the Gazebo plugin):
#     /camera/color/image_raw, /camera/color/camera_info, /camera/aligned_depth_to_color/image_raw,
#     /camera/depth/camera_info, optionally /camera/depth/color/points

import argparse
import math
import os
import tempfile
import xml.etree.ElementTree as ET

parser = argparse.ArgumentParser(description='xArm scene for MoveIt in Isaac Sim')
parser.add_argument('--urdf', required=True, help='URDF written by _robot_moveit_isaac.launch.py')
parser.add_argument('--joint-commands-topic', nargs='+', default=['isaac_joint_commands'],
                    help='one per ros2_control hardware block (arm, gripper)')
parser.add_argument('--joint-states-topic', default='isaac_joint_states')
parser.add_argument('--headless', action='store_true')
parser.add_argument('--livestream', action='store_true', help='headless + WebRTC stream (Isaac Sim WebRTC Streaming Client)')
parser.add_argument('--no-table', action='store_true', help='robot only, no table and ground')
parser.add_argument('--physics-hz', type=int, default=120)
parser.add_argument('--render-hz', type=int, default=60, help='app loop rate limit, keeps sim close to real time')
# USD angular drive units are per degree: 1047.2 Nm/deg = 6e4 Nm/rad, 52.36 Nms/deg = 3e3 Nms/rad
parser.add_argument('--stiffness', type=float, default=1047.2, help='joint drive stiffness (per degree)')
parser.add_argument('--damping', type=float, default=52.36, help='joint drive damping (per degree)')
parser.add_argument('--no-camera', action='store_true', help='no RealSense color/depth publishing')
parser.add_argument('--camera-frame', default='camera_color_optical_frame', help='URDF link of the camera optical frame')
parser.add_argument('--camera-pointcloud', action='store_true', help='also publish the depth point cloud (heavy)')
parser.add_argument('--camera-width', type=int, default=640)
parser.add_argument('--camera-height', type=int, default=480)
parser.add_argument('--camera-hfov', type=float, default=69.4, help='horizontal field of view in degrees (D435i color)')
parser.add_argument('--camera-hz', type=float, default=30.0, help='publish rate, rounded to a whole divisor of --render-hz')
args, _ = parser.parse_known_args()

# SimulationApp must be created before any omni / pxr import
from isaacsim import SimulationApp

app_config = {'headless': args.headless or args.livestream}
if args.livestream:
    app_config.update({'hide_ui': False, 'width': 1280, 'height': 720,
                       'window_width': 1920, 'window_height': 1080, 'display_options': 3286})
simulation_app = SimulationApp(app_config)

import carb
import numpy as np
import omni.graph.core as og
import omni.kit.app
import omni.kit.commands
import omni.timeline
import omni.usd
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics


def enable_first(*names):
    manager = omni.kit.app.get_app().get_extension_manager()
    for name in names:
        if manager.set_extension_enabled_immediate(name, True):
            return name
    raise RuntimeError('none of these Isaac Sim extensions could be enabled: {}'.format(names))


# Isaac Sim >= 4.5 renamed omni.isaac.* to isaacsim.*
ros2_ext = enable_first('isaacsim.ros2.bridge', 'omni.isaac.ros2_bridge')
enable_first('isaacsim.asset.importer.urdf', 'omni.importer.urdf')
if args.livestream:
    simulation_app.set_setting('/app/window/drawMouse', True)
    enable_first('omni.kit.livestream.webrtc', 'omni.kit.livestream.native')
LEGACY = ros2_ext == 'omni.isaac.ros2_bridge'
ROS2_NODES = 'omni.isaac.ros2_bridge' if LEGACY else 'isaacsim.ros2.bridge'
CORE_NODES = 'omni.isaac.core_nodes' if LEGACY else 'isaacsim.core.nodes'
simulation_app.update()

# Gazebo table world (xarm_gazebo/worlds/table_gz.world) expressed in the robot base frame.
# Gazebo spawn: x=-0.2 y=-0.5 z=1.021 yaw=+90deg. Fuel 'Table' at (0, -0.84, 0): 1.5 x 0.8 x 0.03 top
# centered at z=1.0, legs at (+-0.68, +-0.38). Sizes are approximate.
GZ_ROBOT_XYZ = (-0.2, -0.5, 1.021)


def gz_to_robot(x, y, z):
    dx, dy, dz = x - GZ_ROBOT_XYZ[0], y - GZ_ROBOT_XYZ[1], z - GZ_ROBOT_XYZ[2]
    return (dy, -dx, dz)


# Objects on the table in the robot base frame (x forward, y left). The robot sits at the table edge
# facing away from it, so everything is placed behind it (x < 0) within reach:
# three open bins on the left (+y), loose pieces to sort into them on the right (-y)
TABLE_TOP_Z = gz_to_robot(0.0, 0.0, 1.015)[2]
RED, GREEN, BLUE = (0.8, 0.1, 0.1), (0.1, 0.6, 0.15), (0.1, 0.25, 0.8)
BIN_SIZE = (0.13, 0.13, 0.07)  # outer x, y, height
BIN_WALL = 0.008
BINS = [((-0.12, 0.28), RED), ((-0.28, 0.28), GREEN), ((-0.44, 0.28), BLUE)]
PIECE_SIZE = 0.04
PIECE_MASS = 0.05
PIECES = [
    ((-0.15, -0.22), RED), ((-0.27, -0.22), BLUE), ((-0.39, -0.22), GREEN),
    ((-0.15, -0.34), GREEN), ((-0.27, -0.34), RED), ((-0.39, -0.34), BLUE),
]


def add_static_box(stage, path, center, size, color):
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    cube.AddTranslateOp().Set(Gf.Vec3d(*center))
    cube.AddScaleOp().Set(Gf.Vec3f(*size))
    cube.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())


def add_bin(stage, path, xy, color):
    # open box: floor and four walls standing on the table
    (x, y), (sx, sy, h), w, z = xy, BIN_SIZE, BIN_WALL, TABLE_TOP_Z
    UsdGeom.Xform.Define(stage, path)
    add_static_box(stage, path + '/Floor', (x, y, z + w / 2), (sx, sy, w), color)
    for i, side in enumerate((1, -1)):
        add_static_box(stage, path + '/WallX{}'.format(i), (x + side * (sx - w) / 2, y, z + h / 2), (w, sy, h), color)
        add_static_box(stage, path + '/WallY{}'.format(i), (x, y + side * (sy - w) / 2, z + h / 2), (sx - 2 * w, w, h), color)


def add_piece(stage, path, xy, color):
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(PIECE_SIZE)
    cube.AddTranslateOp().Set(Gf.Vec3d(xy[0], xy[1], TABLE_TOP_Z + PIECE_SIZE / 2 + 0.001))
    cube.AddOrientOp().Set(Gf.Quatf(1.0))
    cube.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    UsdPhysics.RigidBodyAPI.Apply(cube.GetPrim())
    UsdPhysics.MassAPI.Apply(cube.GetPrim()).CreateMassAttr(PIECE_MASS)


def build_world(stage):
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.Xform.Define(stage, '/World')

    scene = UsdPhysics.Scene.Define(stage, '/World/physicsScene')
    scene.CreateGravityDirectionAttr(Gf.Vec3f(0.0, 0.0, -1.0))
    scene.CreateGravityMagnitudeAttr(9.81)
    PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim()).CreateTimeStepsPerSecondAttr(args.physics_hz)

    UsdLux.DomeLight.Define(stage, '/World/DomeLight').CreateIntensityAttr(1000.0)
    sun = UsdLux.DistantLight.Define(stage, '/World/Sun')
    sun.CreateIntensityAttr(3000.0)
    sun.AddRotateXYZOp().Set(Gf.Vec3f(-45.0, 0.0, 30.0))

    if args.no_table:
        return
    add_static_box(stage, '/World/Ground', gz_to_robot(0.0, 0.0, -0.05), (20.0, 20.0, 0.1), (0.5, 0.5, 0.5))
    # world x/y swap in the robot frame (yaw 90deg), so the 1.5 m side runs along robot y
    add_static_box(stage, '/World/Table/Top', gz_to_robot(0.0, -0.84, 1.0), (0.8, 1.5, 0.03), (0.6, 0.45, 0.3))
    for i, (lx, ly) in enumerate([(0.68, 0.38), (0.68, -0.38), (-0.68, 0.38), (-0.68, -0.38)]):
        add_static_box(stage, '/World/Table/Leg{}'.format(i), gz_to_robot(lx, -0.84 + ly, 0.5),
                       (0.04, 0.04, 1.0), (0.3, 0.3, 0.3))
    for i, (xy, color) in enumerate(BINS):
        add_bin(stage, '/World/Bins/Bin{}'.format(i), xy, color)
    for i, (xy, color) in enumerate(PIECES):
        add_piece(stage, '/World/Pieces/Piece{}'.format(i), xy, color)


def import_robot(stage, urdf_path):
    try:
        from isaacsim.asset.importer.urdf import URDFImporter, URDFImporterConfig
    except ImportError:
        URDFImporter = None
    if URDFImporter is not None:
        # Isaac Sim >= 6 converts the URDF to a USD file with one Physics variant per engine, referenced here.
        # Written to a temp dir, the bundle folder is not writable for the container user
        config = URDFImporterConfig(urdf_path=urdf_path, usd_path=tempfile.mkdtemp(prefix='xarm_isaac_'), fix_base=True)
        usd_path = URDFImporter(config).import_urdf()
        robot = stage.DefinePrim('/World/{}'.format(os.path.splitext(os.path.basename(urdf_path))[0]), 'Xform')
        robot.GetReferences().AddReference(usd_path)
        if robot.GetVariantSets().HasVariantSet('Physics'):
            robot.GetVariantSets().SetSelection('Physics', 'physx')
        return robot.GetPath().pathString

    _, config = omni.kit.commands.execute('URDFCreateImportConfig')
    config.merge_fixed_joints = False
    config.fix_base = True
    config.make_default_prim = False
    config.self_collision = False
    config.create_physics_scene = False
    config.import_inertia_tensor = True
    config.convex_decomp = False
    config.distance_scale = 1.0
    status, robot_path = omni.kit.commands.execute(
        'URDFParseAndImportFile', urdf_path=urdf_path, import_config=config)
    if not status:
        raise RuntimeError('URDF import failed: {}'.format(urdf_path))
    return robot_path


def find_articulation_root(stage):
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            return prim.GetPath().pathString
    raise RuntimeError('no articulation root found after URDF import')


def tune_joint_drives(stage):
    for prim in stage.Traverse():
        if prim.IsA(UsdPhysics.RevoluteJoint):
            drive = UsdPhysics.DriveAPI.Apply(prim, 'angular')
        elif prim.IsA(UsdPhysics.PrismaticJoint):
            drive = UsdPhysics.DriveAPI.Apply(prim, 'linear')
        else:
            continue
        # mimic joints (gripper fingers) follow their leader, a drive would fight it
        if any('Mimic' in schema for schema in prim.GetAppliedSchemas()):
            continue
        drive.CreateTypeAttr().Set('force')
        drive.CreateStiffnessAttr().Set(args.stiffness)
        drive.CreateDampingAttr().Set(args.damping)
        drive.CreateTargetPositionAttr().Set(0.0)


def urdf_origin(joint):
    # 4x4 transform of a URDF joint origin, rpy as fixed-axis roll, pitch, yaw
    origin = joint.find('origin')
    xyz = [float(v) for v in origin.get('xyz', '0 0 0').split()] if origin is not None else [0.0] * 3
    roll, pitch, yaw = [float(v) for v in origin.get('rpy', '0 0 0').split()] if origin is not None else [0.0] * 3
    cr, sr, cp, sp, cy, sy = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch), math.cos(yaw), math.sin(yaw)
    transform = np.eye(4)
    transform[:3, :3] = [[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                         [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                         [-sp, cp * sr, cp * cr]]
    transform[:3, 3] = xyz
    return transform


def add_camera(stage, robot_path):
    # The optical frame is a massless URDF link the importer may merge away: walk up its fixed joints
    # to the first link simulated as a rigid body and attach the camera there
    bodies = {prim.GetName(): prim.GetPath().pathString
              for prim in Usd.PrimRange(stage.GetPrimAtPath(robot_path)) if prim.HasAPI(UsdPhysics.RigidBodyAPI)}
    joints = {joint.find('child').get('link'): joint for joint in ET.parse(args.urdf).getroot().iter('joint')}
    link, link_to_frame = args.camera_frame, np.eye(4)
    while link not in bodies:
        joint = joints.get(link)
        if joint is None or joint.get('type') != 'fixed':
            carb.log_warn('xarm isaac: {} not found on a rigid body, no camera'.format(args.camera_frame))
            return None
        link_to_frame = urdf_origin(joint) @ link_to_frame
        link = joint.find('parent').get('link')

    # USD cameras look along -Z with +Y up, ROS optical frames along +Z with +Y down
    link_to_camera = link_to_frame @ np.diag([1.0, -1.0, -1.0, 1.0])
    camera = UsdGeom.Camera.Define(stage, bodies[link] + '/RealSense')
    # Gf matrices act on row vectors
    camera.AddTransformOp().Set(Gf.Matrix4d(link_to_camera.T.tolist()))
    aperture = 20.955
    camera.CreateHorizontalApertureAttr(aperture)
    camera.CreateVerticalApertureAttr(aperture * args.camera_height / args.camera_width)
    camera.CreateFocalLengthAttr(aperture / (2 * math.tan(math.radians(args.camera_hfov) / 2)))
    # D435i depth range
    camera.CreateClippingRangeAttr(Gf.Vec2f(0.1, 10.0))
    return camera.GetPath().pathString


def set_target(stage, node_path, attr_name, target_path):
    prim = stage.GetPrimAtPath(node_path)
    rel = prim.GetRelationship(attr_name) or prim.CreateRelationship(attr_name)
    rel.SetTargets([Sdf.Path(target_path)])


def build_ros2_graph(stage, articulation_root, camera_path):
    nodes = [
        ('OnPlaybackTick', 'omni.graph.action.OnPlaybackTick'),
        ('Context', ROS2_NODES + '.ROS2Context'),
        ('SimTime', CORE_NODES + '.IsaacReadSimulationTime'),
        ('PublishClock', ROS2_NODES + '.ROS2PublishClock'),
        ('PublishJointState', ROS2_NODES + '.ROS2PublishJointState'),
    ]
    connections = [
        ('OnPlaybackTick.outputs:tick', 'PublishClock.inputs:execIn'),
        ('OnPlaybackTick.outputs:tick', 'PublishJointState.inputs:execIn'),
        ('Context.outputs:context', 'PublishClock.inputs:context'),
        ('Context.outputs:context', 'PublishJointState.inputs:context'),
        ('SimTime.outputs:simulationTime', 'PublishClock.inputs:timeStamp'),
        ('SimTime.outputs:simulationTime', 'PublishJointState.inputs:timeStamp'),
    ]
    values = [
        ('PublishClock.inputs:topicName', 'clock'),
        ('PublishJointState.inputs:topicName', args.joint_states_topic),
    ]
    # a subscriber and controller per commands topic, applying the newest message each frame
    # (the subscriber takes one message per frame, by default the oldest of 10 queued)
    controller_names = []
    for i, topic in enumerate(args.joint_commands_topic):
        subscriber, controller = 'SubscribeJointState{}'.format(i or ''), 'ArticulationController{}'.format(i or '')
        controller_names.append(controller)
        nodes += [
            (subscriber, ROS2_NODES + '.ROS2SubscribeJointState'),
            (controller, CORE_NODES + '.IsaacArticulationController'),
        ]
        connections += [
            ('OnPlaybackTick.outputs:tick', subscriber + '.inputs:execIn'),
            ('OnPlaybackTick.outputs:tick', controller + '.inputs:execIn'),
            ('Context.outputs:context', subscriber + '.inputs:context'),
        ] + [(subscriber + '.outputs:' + output, controller + '.inputs:' + output)
             for output in ('jointNames', 'positionCommand', 'velocityCommand', 'effortCommand')]
        values += [
            (subscriber + '.inputs:topicName', topic),
            (subscriber + '.inputs:queueSize', 1),
        ]
    if camera_path:
        # One render product at the color optical frame gives color and depth, so depth is aligned to color.
        # Topics of the Gazebo plugin, xarm_description/urdf/camera/realsense.gazebo.xacro
        camera_topics = [  # (node, helper, sensor type, topic)
            ('ColorImage', 'ROS2CameraHelper', 'rgb', 'camera/color/image_raw'),
            ('ColorInfo', 'ROS2CameraInfoHelper', None, 'camera/color/camera_info'),
            ('DepthImage', 'ROS2CameraHelper', 'depth', 'camera/aligned_depth_to_color/image_raw'),
            ('DepthInfo', 'ROS2CameraInfoHelper', None, 'camera/depth/camera_info'),
        ]
        if args.camera_pointcloud:
            camera_topics.append(('PointCloud', 'ROS2CameraHelper', 'depth_pcl', 'camera/depth/color/points'))
        # the helpers publish every rendered frame unless told to skip some
        frame_skip = max(0, round(args.render_hz / args.camera_hz) - 1)
        nodes.append(('RenderProduct', CORE_NODES + '.IsaacCreateRenderProduct'))
        connections.append(('OnPlaybackTick.outputs:tick', 'RenderProduct.inputs:execIn'))
        values += [
            ('RenderProduct.inputs:width', args.camera_width),
            ('RenderProduct.inputs:height', args.camera_height),
        ]
        for name, helper, sensor_type, topic in camera_topics:
            nodes.append((name, ROS2_NODES + '.' + helper))
            connections += [
                ('RenderProduct.outputs:execOut', name + '.inputs:execIn'),
                ('RenderProduct.outputs:renderProductPath', name + '.inputs:renderProductPath'),
                ('Context.outputs:context', name + '.inputs:context'),
            ]
            values += [
                (name + '.inputs:topicName', topic),
                (name + '.inputs:frameId', args.camera_frame),
                (name + '.inputs:frameSkipCount', frame_skip),
            ]
            if sensor_type:
                values.append((name + '.inputs:type', sensor_type))

    keys = og.Controller.Keys
    og.Controller.edit(
        {'graph_path': '/ActionGraph', 'evaluator_name': 'execution'},
        {keys.CREATE_NODES: nodes, keys.CONNECT: connections, keys.SET_VALUES: values},
    )
    if camera_path:
        set_target(stage, '/ActionGraph/RenderProduct', 'inputs:cameraPrim', camera_path)
    set_target(stage, '/ActionGraph/PublishJointState', 'inputs:targetPrim', articulation_root)
    for name in controller_names:
        node_path = '/ActionGraph/' + name
        set_target(stage, node_path, 'inputs:targetPrim', articulation_root)
        # Isaac Sim <= 4.2 selects the robot by string path
        controller = og.Controller.node(node_path)
        if controller.get_attribute_exists('inputs:robotPath'):
            og.Controller.set(og.Controller.attribute(node_path + '.inputs:robotPath'), articulation_root)
        if controller.get_attribute_exists('inputs:usePath'):
            og.Controller.set(og.Controller.attribute(node_path + '.inputs:usePath'), True)


def main():
    stage = omni.usd.get_context().get_stage()
    build_world(stage)
    robot_path = import_robot(stage, args.urdf)
    simulation_app.update()
    tune_joint_drives(stage)
    articulation_root = find_articulation_root(stage)
    camera_path = None if args.no_camera else add_camera(stage, robot_path)
    build_ros2_graph(stage, articulation_root, camera_path)
    carb.log_info('xarm isaac: articulation {} on /{} and /{}, camera {}'.format(
        articulation_root, args.joint_states_topic, args.joint_commands_topic, camera_path))

    settings = carb.settings.get_settings()
    settings.set('/app/runLoops/main/rateLimitEnabled', True)
    settings.set('/app/runLoops/main/rateLimitFrequency', args.render_hz)

    # Do not stop/restart the timeline while ROS is running: sim time jumps back and MoveIt/TF complain.
    omni.timeline.get_timeline_interface().play()
    while simulation_app.is_running():
        simulation_app.update()
    simulation_app.close()


if __name__ == '__main__':
    main()
