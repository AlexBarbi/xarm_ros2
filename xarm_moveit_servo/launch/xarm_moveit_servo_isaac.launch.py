#!/usr/bin/env python3
# Software License Agreement (BSD License)
#
# Keyboard teleop (MoveIt Servo) of xArm in Isaac Sim with MoveIt active as well, e.g.
#   ros2 launch xarm_moveit_servo xarm_moveit_servo_isaac.launch.py dof:=6 isaac_sim_path:=<isaac_sim dir>
# Isaac Sim args as in xarm_moveit_config/launch/xarm6_moveit_isaac.launch.py (isaac_sim_path, isaac_docker_image, ...).
# keyboard_terminal: command that opens the keyboard node's terminal (default 'xterm -e'), '' to run it yourself.

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    dof = LaunchConfiguration('dof', default=6)
    robot_type = LaunchConfiguration('robot_type', default='xarm')

    # robot moveit servo isaac launch
    # xarm_moveit_servo/launch/_robot_moveit_servo_isaac.launch.py
    robot_moveit_servo_isaac_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution([FindPackageShare('xarm_moveit_servo'), 'launch', '_robot_moveit_servo_isaac.launch.py'])),
        launch_arguments={
            'dof': dof,
            'robot_type': robot_type,
        }.items(),
    )

    return LaunchDescription([
        robot_moveit_servo_isaac_launch
    ])
