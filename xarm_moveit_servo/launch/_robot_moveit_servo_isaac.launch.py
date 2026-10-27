#!/usr/bin/env python3
# Software License Agreement (BSD License)
#
# Keyboard teleop with MoveIt Servo in Isaac Sim, with MoveIt (move_group + RViz MotionPlanning) active too.
# Includes xarm_moveit_config/launch/_robot_moveit_isaac.launch.py (Isaac Sim, ros2_control, move_group, RViz)
# and adds servo_server and xarm_keyboard_input. Both drive the same trajectory controller: servo through its
# joint_trajectory topic, move_group through the FollowJointTrajectory action. The last command wins, and servo
# picks up from wherever a MoveIt execution left the arm.

from launch import LaunchDescription
from launch.actions import OpaqueFunction, IncludeLaunchDescription, ExecuteProcess, RegisterEventHandler, LogInfo
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from uf_ros_lib.moveit_configs_builder import MoveItConfigsBuilder
from uf_ros_lib.uf_robot_utils import load_yaml

# robot arguments with the defaults of _robot_moveit_isaac.launch.py, passed on to it so that
# servo's robot model is the one in Isaac Sim and move_group
ROBOT_ARGS = [
    ('dof', 7), ('robot_type', 'xarm'), ('prefix', ''), ('hw_ns', 'xarm'), ('limited', True),
    ('effort_control', False), ('velocity_control', False), ('model1300', False), ('robot_sn', ''),
    ('attach_to', 'world'), ('attach_xyz', '"0 0 0"'), ('attach_rpy', '"0 0 0"'),
    ('mesh_suffix', 'stl'), ('kinematics_suffix', ''), ('gripper_version', 'G1'),
    ('add_gripper', True), ('add_vacuum_gripper', False), ('add_bio_gripper', False),
    ('add_realsense_d435i', True), ('add_d435i_links', True), ('add_other_geometry', False),
    ('geometry_type', 'box'), ('geometry_mass', 0.1), ('geometry_height', 0.1), ('geometry_radius', 0.1),
    ('geometry_length', 0.1), ('geometry_width', 0.1), ('geometry_mesh_filename', ''),
    ('geometry_mesh_origin_xyz', '"0 0 0"'), ('geometry_mesh_origin_rpy', '"0 0 0"'),
    ('geometry_mesh_tcp_xyz', '"0 0 0"'), ('geometry_mesh_tcp_rpy', '"0 0 0"'),
]


def launch_setup(context, *args, **kwargs):
    robot_args = {name: LaunchConfiguration(name, default=default) for name, default in ROBOT_ARGS}
    dof = robot_args['dof']
    robot_type = robot_args['robot_type'].perform(context)
    prefix = robot_args['prefix'].perform(context)
    use_sim_time = LaunchConfiguration('use_sim_time', default=True).perform(context) in ('True', 'true')
    # command that opens a terminal for the keyboard node, e.g. 'gnome-terminal --wait --'; empty: run it yourself
    keyboard_terminal = LaunchConfiguration('keyboard_terminal', default='xterm -e').perform(context)

    xarm_type = '{}{}'.format(robot_type, dof.perform(context) if robot_type in ('xarm', 'lite') else '')

    moveit_config = MoveItConfigsBuilder(
        context=context,
        controllers_name='fake_controllers',
        ros2_control_plugin='topic_based_ros2_control/TopicBasedSystem',
        **robot_args,
    ).to_moveit_configs()

    robot_description_parameters = {}
    robot_description_parameters.update(moveit_config.robot_description)
    robot_description_parameters.update(moveit_config.robot_description_semantic)
    robot_description_parameters.update(moveit_config.robot_description_kinematics)
    robot_description_parameters.update(moveit_config.joint_limits)

    servo_yaml = load_yaml('xarm_moveit_servo', 'config/xarm_moveit_servo_config.yaml')
    servo_yaml['move_group_name'] = xarm_type
    servo_yaml['command_out_topic'] = '/{}{}_traj_controller/joint_trajectory'.format(prefix, xarm_type)
    # move_group owns the planning scene and /get_planning_scene, servo follows it,
    # so objects added in RViz are also avoided while jogging
    servo_yaml['is_primary_planning_scene_monitor'] = False
    servo_yaml['monitored_planning_scene_topic'] = '/monitored_planning_scene'
    servo_params = {'moveit_servo': servo_yaml}

    # robot moveit isaac launch
    # xarm_moveit_config/launch/_robot_moveit_isaac.launch.py
    robot_moveit_isaac_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution([FindPackageShare('xarm_moveit_config'), 'launch', '_robot_moveit_isaac.launch.py'])),
        launch_arguments=dict(robot_args, use_sim_time=str(use_sim_time).lower()).items(),
    )

    # servo and the keyboard node compare command stamps with their own clock, so both run on Isaac Sim time
    servo_node = Node(
        package='moveit_servo',
        executable='servo_node',
        name='servo_server',
        output='screen',
        parameters=[
            servo_params,
            robot_description_parameters,
            {'use_sim_time': use_sim_time},
        ],
    )

    if keyboard_terminal:
        # the keyboard node reads key presses from its terminal, ros2 launch does not give it one
        keyboard_node = Node(
            package='xarm_moveit_servo',
            executable='xarm_keyboard_input',
            prefix=keyboard_terminal,
            output='screen',
            parameters=[
                servo_params,
                {
                    'dof': dof,
                    'ros_queue_size': 10,
                    'use_sim_time': use_sim_time,
                },
            ],
        )
    else:
        keyboard_node = LogInfo(msg='Keyboard teleop not started (keyboard_terminal is empty), run it in another terminal:\n'
                                    '  ros2 run xarm_moveit_servo xarm_keyboard_input --ros-args -p use_sim_time:={}'.format(str(use_sim_time).lower()))

    # servo exits when it has no robot state 5 s after start, Isaac Sim takes much longer to come up:
    # start servo and the keyboard node once ros2_control publishes joint states
    wait_for_joint_states = ExecuteProcess(
        cmd=['ros2', 'topic', 'echo', '--once', servo_yaml['joint_topic'], 'sensor_msgs/msg/JointState'],
        output='log',
    )
    start_teleop = RegisterEventHandler(event_handler=OnProcessExit(
        target_action=wait_for_joint_states,
        # nothing to start when the launch is stopped before the robot came up
        on_exit=lambda event, _: [servo_node, keyboard_node] if event.returncode == 0 else [],
    ))

    return [
        robot_moveit_isaac_launch,
        start_teleop,
        wait_for_joint_states,
    ]


def generate_launch_description():
    return LaunchDescription([
        OpaqueFunction(function=launch_setup)
    ])
