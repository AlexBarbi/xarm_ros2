#!/usr/bin/env python3
# Software License Agreement (BSD License)
#
# xarm5 + MoveIt in Isaac Sim (Isaac Sim replacement for xarm5_moveit_gazebo.launch.py).
# Start Isaac Sim too with isaac_sim_path:=<isaac_sim dir> (or $ISAAC_SIM_PATH),
# or in a container with isaac_docker_image:=nvcr.io/nvidia/isaac-sim:<version> [isaac_docker_args:='-v ...'].
# Other args: isaac_headless, isaac_livestream, isaac_table, isaac_camera, isaac_camera_pointcloud,
# isaac_urdf_path, isaac_ros_distro, use_sim_time, show_rviz.
# headless:=true runs without any window (no RViz, Isaac Sim started here headless). It is the default over SSH or without
# a display: RViz cannot open there, and when it exits the launch shuts everything down. headless:=false forces the windows.

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (AndSubstitution, LaunchConfiguration, NotSubstitution, OrSubstitution,
                                  PathJoinSubstitution)
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    hw_ns = LaunchConfiguration('hw_ns', default='xarm')
    headless = LaunchConfiguration('headless')
    show_rviz = LaunchConfiguration('show_rviz', default='true')
    isaac_headless = LaunchConfiguration('isaac_headless', default='false')
    no_display = 'SSH_CONNECTION' in os.environ or not os.environ.get('DISPLAY')

    # robot moveit isaac launch
    # xarm_moveit_config/launch/_robot_moveit_isaac.launch.py
    robot_moveit_isaac_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution([FindPackageShare('xarm_moveit_config'), 'launch', '_robot_moveit_isaac.launch.py'])),
        launch_arguments={
            'dof': '5',
            'robot_type': 'xarm',
            'hw_ns': hw_ns,
            'no_gui_ctrl': 'false',
            'limited': 'false',     # the arm's own range: with +-178 deg it cannot pass behind its base
            'show_rviz': AndSubstitution(show_rviz, NotSubstitution(headless)),
            'isaac_headless': OrSubstitution(isaac_headless, headless),
        }.items(),
    )

    return LaunchDescription([
        DeclareLaunchArgument('headless', default_value=str(no_display).lower(),
                              description='No windows: no RViz, Isaac Sim started here headless. Default: true over SSH or without a display'),
        robot_moveit_isaac_launch
    ])
