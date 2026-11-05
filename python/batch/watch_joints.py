#!/usr/bin/env python3
"""The extremes of the wrist (joint4 of the xArm5, joint5 of the xArm6), of the fingers and of how far the arm is from where it is told to be, in a run: a simulation
that blew up shows here even when the fingers are still near the table. usage: watch_joints.py out.json"""
import collections
import json
import os
import sys
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

ARM = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']    # those the arm has
HELD_FOR = 0.5      # s: a gap counts only if it lasts, the start of a motion opens a short one

worst = {'wrist': 0.0, 'finger_min': 0.0, 'tracking': 0.0}
command = {}
gaps = collections.deque()


def on_command(message):
    command.update(zip(message.name, message.position))


def on_joints(message):
    joints = dict(zip(message.name, message.position))
    wrist = 'joint5' if 'joint6' in joints else 'joint4'     # the one that bends the wrist, the others turn
    worst['wrist'] = max(worst['wrist'], abs(joints.get(wrist, 0.0)))
    worst['finger_min'] = min(worst['finger_min'], joints.get('drive_joint', 0.0))
    arm = [j for j in ARM if j in joints]
    if not all(j in command for j in arm):
        return
    now = time.monotonic()
    gaps.append((now, max(abs(joints[j] - command[j]) for j in arm)))
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
