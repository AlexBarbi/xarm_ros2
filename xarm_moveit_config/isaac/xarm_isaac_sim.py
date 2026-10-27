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
#   ROS 2 -> Isaac: /isaac_joint_commands (position + velocity targets from topic_based_ros2_control)

import argparse
import os
import tempfile

parser = argparse.ArgumentParser(description='xArm scene for MoveIt in Isaac Sim')
parser.add_argument('--urdf', required=True, help='URDF written by _robot_moveit_isaac.launch.py')
parser.add_argument('--joint-commands-topic', default='isaac_joint_commands')
parser.add_argument('--joint-states-topic', default='isaac_joint_states')
parser.add_argument('--headless', action='store_true')
parser.add_argument('--livestream', action='store_true', help='headless + WebRTC stream (Isaac Sim WebRTC Streaming Client)')
parser.add_argument('--no-table', action='store_true', help='robot only, no table and ground')
parser.add_argument('--physics-hz', type=int, default=120)
parser.add_argument('--render-hz', type=int, default=60, help='app loop rate limit, keeps sim close to real time')
# USD angular drive units are per degree: 1047.2 Nm/deg = 6e4 Nm/rad, 52.36 Nms/deg = 3e3 Nms/rad
parser.add_argument('--stiffness', type=float, default=1047.2, help='joint drive stiffness (per degree)')
parser.add_argument('--damping', type=float, default=52.36, help='joint drive damping (per degree)')
args, _ = parser.parse_known_args()

# SimulationApp must be created before any omni / pxr import
from isaacsim import SimulationApp

app_config = {'headless': args.headless or args.livestream}
if args.livestream:
    app_config.update({'hide_ui': False, 'width': 1280, 'height': 720,
                       'window_width': 1920, 'window_height': 1080, 'display_options': 3286})
simulation_app = SimulationApp(app_config)

import carb
import omni.graph.core as og
import omni.kit.app
import omni.kit.commands
import omni.timeline
import omni.usd
from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdLux, UsdPhysics


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


def set_target(stage, node_path, attr_name, target_path):
    prim = stage.GetPrimAtPath(node_path)
    rel = prim.GetRelationship(attr_name) or prim.CreateRelationship(attr_name)
    rel.SetTargets([Sdf.Path(target_path)])


def build_ros2_graph(stage, articulation_root):
    keys = og.Controller.Keys
    og.Controller.edit(
        {'graph_path': '/ActionGraph', 'evaluator_name': 'execution'},
        {
            keys.CREATE_NODES: [
                ('OnPlaybackTick', 'omni.graph.action.OnPlaybackTick'),
                ('Context', ROS2_NODES + '.ROS2Context'),
                ('SimTime', CORE_NODES + '.IsaacReadSimulationTime'),
                ('PublishClock', ROS2_NODES + '.ROS2PublishClock'),
                ('PublishJointState', ROS2_NODES + '.ROS2PublishJointState'),
                ('SubscribeJointState', ROS2_NODES + '.ROS2SubscribeJointState'),
                ('ArticulationController', CORE_NODES + '.IsaacArticulationController'),
            ],
            keys.CONNECT: [
                ('OnPlaybackTick.outputs:tick', 'PublishClock.inputs:execIn'),
                ('OnPlaybackTick.outputs:tick', 'PublishJointState.inputs:execIn'),
                ('OnPlaybackTick.outputs:tick', 'SubscribeJointState.inputs:execIn'),
                ('OnPlaybackTick.outputs:tick', 'ArticulationController.inputs:execIn'),
                ('Context.outputs:context', 'PublishClock.inputs:context'),
                ('Context.outputs:context', 'PublishJointState.inputs:context'),
                ('Context.outputs:context', 'SubscribeJointState.inputs:context'),
                ('SimTime.outputs:simulationTime', 'PublishClock.inputs:timeStamp'),
                ('SimTime.outputs:simulationTime', 'PublishJointState.inputs:timeStamp'),
                ('SubscribeJointState.outputs:jointNames', 'ArticulationController.inputs:jointNames'),
                ('SubscribeJointState.outputs:positionCommand', 'ArticulationController.inputs:positionCommand'),
                ('SubscribeJointState.outputs:velocityCommand', 'ArticulationController.inputs:velocityCommand'),
                ('SubscribeJointState.outputs:effortCommand', 'ArticulationController.inputs:effortCommand'),
            ],
            keys.SET_VALUES: [
                ('PublishClock.inputs:topicName', 'clock'),
                ('PublishJointState.inputs:topicName', args.joint_states_topic),
                ('SubscribeJointState.inputs:topicName', args.joint_commands_topic),
            ],
        },
    )
    set_target(stage, '/ActionGraph/PublishJointState', 'inputs:targetPrim', articulation_root)
    set_target(stage, '/ActionGraph/ArticulationController', 'inputs:targetPrim', articulation_root)
    # Isaac Sim <= 4.2 selects the robot by string path
    controller = og.Controller.node('/ActionGraph/ArticulationController')
    if controller.get_attribute_exists('inputs:robotPath'):
        og.Controller.set(og.Controller.attribute('/ActionGraph/ArticulationController.inputs:robotPath'), articulation_root)
    if controller.get_attribute_exists('inputs:usePath'):
        og.Controller.set(og.Controller.attribute('/ActionGraph/ArticulationController.inputs:usePath'), True)


def main():
    stage = omni.usd.get_context().get_stage()
    build_world(stage)
    import_robot(stage, args.urdf)
    simulation_app.update()
    tune_joint_drives(stage)
    articulation_root = find_articulation_root(stage)
    build_ros2_graph(stage, articulation_root)
    carb.log_info('xarm isaac: articulation {} on /{} and /{}'.format(
        articulation_root, args.joint_states_topic, args.joint_commands_topic))

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
