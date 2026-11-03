#!/usr/bin/env python3
"""The extremes of joint4, of the fingers and of how far the arm is from where it is told to be, in a run: a simulation
that blew up shows here even when the fingers are still near the table. usage: watch_joints.py out.json"""
import collections
import json
import os
import sys
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

ARM = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5']
HELD_FOR = 0.5      # s: a gap counts only if it lasts, the start of a motion opens a short one

worst = {'joint4': 0.0, 'finger_min': 0.0, 'tracking': 0.0}
command = {}
gaps = collections.deque()


def on_command(message):
    command.update(zip(message.name, message.position))


def on_joints(message):
    joints = dict(zip(message.name, message.position))
    worst['joint4'] = max(worst['joint4'], abs(joints.get('joint4', 0.0)))
    worst['finger_min'] = min(worst['finger_min'], joints.get('drive_joint', 0.0))
    if not all(j in command and j in joints for j in ARM):
        return
    now = time.monotonic()
    gaps.append((now, max(abs(joints[j] - command[j]) for j in ARM)))
    while gaps[0][0] < now - HELD_FOR:
        gaps.popleft()
        worst['tracking'] = max(worst['tracking'], min(g for _, g in gaps))


def main():
    rclpy.init()
    node = Node('watch_joints')
    node.create_subscription(JointState, '/joint_states', on_joints, 50)
    node.create_subscription(JointState, '/isaac_joint_commands', on_command, 50)
    temporary = sys.argv[1] + '.new'
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.5)
            with open(temporary, 'w') as f:
                json.dump(worst, f)
            os.replace(temporary, sys.argv[1])
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
