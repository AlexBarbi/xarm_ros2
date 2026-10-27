#!/usr/bin/env python3
# Software License Agreement (BSD License)
#
# xarm5 + MoveIt in Isaac Sim (Isaac Sim replacement for xarm5_moveit_gazebo.launch.py).
# Start Isaac Sim too with isaac_sim_path:=<isaac_sim dir> (or $ISAAC_SIM_PATH),
# or in a container with isaac_docker_image:=nvcr.io/nvidia/isaac-sim:<version> [isaac_docker_args:='-v ...'].
# Other args: isaac_headless, isaac_livestream, isaac_table, isaac_camera, isaac_camera_pointcloud,
# isaac_urdf_path, isaac_ros_distro, use_sim_time, show_rviz.

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    hw_ns = LaunchConfiguration('hw_ns', default='xarm')

    # robot moveit isaac launch
    # xarm_moveit_config/launch/_robot_moveit_isaac.launch.py
    robot_moveit_isaac_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution([FindPackageShare('xarm_moveit_config'), 'launch', '_robot_moveit_isaac.launch.py'])),
        launch_arguments={
            'dof': '5',
            'robot_type': 'xarm',
            'hw_ns': hw_ns,
            'no_gui_ctrl': 'false',
        }.items(),
    )

    return LaunchDescription([
        robot_moveit_isaac_launch
    ])
