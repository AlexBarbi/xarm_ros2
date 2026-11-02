#!/usr/bin/env python3
# Software License Agreement (BSD License)
#
# xarm5 + MoveIt on an Isaac Sim world started by hand, e.g. ros_ws/shared/world.usd with the robot
# asset next to it (./xarm5/xarm5.usda) and the ROS 2 action graph saved in it: open the world in
# Isaac Sim and press Play. Nothing of the scene is generated here, this launch only talks ROS 2 to it.
# ros2_control runs topic_based_ros2_control/TopicBasedSystem on the topics of the world's action graph:
#   Isaac -> ROS 2: /clock, /isaac_joint_states (isaac_joint_states_topic)
#   ROS 2 -> Isaac: /isaac_joint_commands (isaac_joint_commands_topic, arm),
#                   /isaac_joint_commands/XArmGripperSystem (gripper, one topic per ros2_control block)
#   RealSense: /camera/color/image_raw, /camera/aligned_depth_to_color/image_raw, ...
# Other args: add_gripper, add_realsense_d435i (as the robot in the world), use_sim_time, show_rviz.

import os
import re
import xml.etree.ElementTree as ET
import yaml
from ament_index_python import get_package_share_directory
from launch import LaunchDescription
from launch.actions import OpaqueFunction, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from uf_ros_lib.moveit_configs_builder import MoveItConfigsBuilder
from uf_ros_lib.uf_robot_utils import generate_ros2_control_params_temp_file

TOPIC_BASED_PLUGIN = 'topic_based_ros2_control/TopicBasedSystem'


def add_topic_params(urdf_xml, joint_commands_topic, joint_states_topic):
    # point every TopicBasedSystem block (arm and gripper) at the Isaac Sim topics, the names the action
    # graph of the world subscribes to: the first block on joint_commands_topic, the others on
    # joint_commands_topic/<block name> (blocks sharing one topic starve each other in Isaac Sim)
    root = ET.fromstring(urdf_xml)
    blocks = [ros2_control for ros2_control in root.iter('ros2_control')
              if (ros2_control.findtext('hardware/plugin') or '').strip() == TOPIC_BASED_PLUGIN]
    for i, ros2_control in enumerate(blocks):
        # TopicBasedSystem names its node after the block, xacro uses the plugin name ('/' is invalid there)
        block_name = re.sub(r'\W', '_', ros2_control.get('name'))
        ros2_control.set('name', block_name)
        commands_topic = '{}/{}'.format(joint_commands_topic, block_name) if i else joint_commands_topic
        hardware = ros2_control.find('hardware')
        for name, value in (('joint_commands_topic', commands_topic), ('joint_states_topic', joint_states_topic)):
            ET.SubElement(hardware, 'param', name=name).text = value
    return ET.tostring(root, encoding='unicode')


def launch_setup(context, *args, **kwargs):
    dof = LaunchConfiguration('dof', default=5)
    robot_type = LaunchConfiguration('robot_type', default='xarm')
    prefix = LaunchConfiguration('prefix', default='')
    hw_ns = LaunchConfiguration('hw_ns', default='xarm')
    limited = LaunchConfiguration('limited', default=True)
    attach_to = LaunchConfiguration('attach_to', default='world')
    attach_xyz = LaunchConfiguration('attach_xyz', default='"0 0 0"')
    attach_rpy = LaunchConfiguration('attach_rpy', default='"0 0 0"')
    # the robot description must match the robot in the world (xarm5/xarm5.usda)
    add_gripper = LaunchConfiguration('add_gripper', default=True)
    add_realsense_d435i = LaunchConfiguration('add_realsense_d435i', default=True)
    add_d435i_links = LaunchConfiguration('add_d435i_links', default=True)

    no_gui_ctrl = LaunchConfiguration('no_gui_ctrl', default=False)
    show_rviz = LaunchConfiguration('show_rviz', default=True)
    ros_namespace = LaunchConfiguration('ros_namespace', default='').perform(context)
    use_sim_time = LaunchConfiguration('use_sim_time', default=True).perform(context) in ('True', 'true')
    update_rate = int(LaunchConfiguration('update_rate', default=150).perform(context))
    joint_commands_topic = LaunchConfiguration('isaac_joint_commands_topic', default='/isaac_joint_commands').perform(context)
    joint_states_topic = LaunchConfiguration('isaac_joint_states_topic', default='/isaac_joint_states').perform(context)

    xarm_type = '{}{}'.format(robot_type.perform(context), dof.perform(context) if robot_type.perform(context) in ('xarm', 'lite') else '')

    ros2_control_params = generate_ros2_control_params_temp_file(
        os.path.join(get_package_share_directory('xarm_controller'), 'config', '{}_controllers.yaml'.format(xarm_type)),
        prefix=prefix.perform(context),
        add_gripper=add_gripper.perform(context) in ('True', 'true'),
        ros_namespace=ros_namespace,
        update_rate=update_rate,
        use_sim_time=use_sim_time,
        robot_type=robot_type.perform(context)
    )

    moveit_config = MoveItConfigsBuilder(
        context=context,
        controllers_name='fake_controllers',
        dof=dof,
        robot_type=robot_type,
        prefix=prefix,
        hw_ns=hw_ns,
        limited=limited,
        attach_to=attach_to,
        attach_xyz=attach_xyz,
        attach_rpy=attach_rpy,
        ros2_control_plugin=TOPIC_BASED_PLUGIN,
        ros2_control_params=ros2_control_params,
        add_gripper=add_gripper,
        add_realsense_d435i=add_realsense_d435i,
        add_d435i_links=add_d435i_links,
    ).to_moveit_configs()

    moveit_config.robot_description = {'robot_description': add_topic_params(
        moveit_config.robot_description['robot_description'], joint_commands_topic, joint_states_topic)}
    moveit_config_dump = yaml.dump(moveit_config.to_dict())

    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}, moveit_config.robot_description],
        remappings=[
            ('/tf', 'tf'),
            ('/tf_static', 'tf_static'),
        ]
    )

    # robot moveit common launch
    # xarm_moveit_config/launch/_robot_moveit_common2.launch.py
    robot_moveit_common_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution([FindPackageShare('xarm_moveit_config'), 'launch', '_robot_moveit_common2.launch.py'])),
        launch_arguments={
            'prefix': prefix,
            'attach_to': attach_to,
            'attach_xyz': attach_xyz,
            'attach_rpy': attach_rpy,
            'no_gui_ctrl': no_gui_ctrl,
            'show_rviz': show_rviz,
            'use_sim_time': str(use_sim_time).lower(),
            'moveit_config_dump': moveit_config_dump,
        }.items(),
    )

    # ros2 control launch
    # xarm_controller/launch/_ros2_control.launch.py
    ros2_control_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution([FindPackageShare('xarm_controller'), 'launch', '_ros2_control.launch.py'])),
        launch_arguments={
            'robot_description': yaml.dump(moveit_config.robot_description),
            'ros2_control_params': ros2_control_params,
        }.items(),
    )

    controllers = ['joint_state_broadcaster', '{}{}_traj_controller'.format(prefix.perform(context), xarm_type)]
    if add_gripper.perform(context) in ('True', 'true'):
        controllers.append('{}{}_gripper_traj_controller'.format(prefix.perform(context), robot_type.perform(context)))

    # one spawner for all controllers
    controller_nodes = [Node(
        package='controller_manager',
        executable='spawner',
        output='screen',
        arguments=controllers + [
            '--controller-manager', '{}/controller_manager'.format(ros_namespace),
            # with sim time the controller manager only cycles once Isaac Sim publishes /clock,
            # i.e. once the world is playing (default switch timeout is 5 s)
            '--switch-timeout', '300',
            '--service-call-timeout', '310',
        ],
        parameters=[{'use_sim_time': use_sim_time}],
    )]

    return [
        robot_state_publisher_node,
        robot_moveit_common_launch,
        ros2_control_launch,
    ] + controller_nodes


def generate_launch_description():
    return LaunchDescription([
        OpaqueFunction(function=launch_setup)
    ])
