"""The GripperCommand action of the real driver on top of the simulated gripper: same name ending, same behaviour (one speed,
no force, ends when the fingers stop). So pick_and_place runs in Isaac the code it runs on the real arm."""
import time
import rclpy
from rclpy.action import ActionClient, ActionServer
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory, GripperCommand
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectoryPoint
SPEED = 1.0                 # rad/s
rclpy.init()
n = rclpy.create_node('xarm_driver_stand_in')
group = ReentrantCallbackGroup()
client = ActionClient(n, FollowJointTrajectory, '/xarm_gripper_traj_controller/follow_joint_trajectory', callback_group=group)
now = [0.0]
n.create_subscription(JointState, '/joint_states', lambda m: now.__setitem__(0, dict(zip(m.name, m.position)).get('drive_joint', now[0])), 10, callback_group=group)
def execute(handle):
    target = handle.request.command.position
    d = max(abs(target - now[0]) / SPEED, 0.1)
    goal = FollowJointTrajectory.Goal()
    goal.trajectory.joint_names = ['drive_joint']
    goal.trajectory.points = [JointTrajectoryPoint(positions=[float(target)], time_from_start=Duration(sec=int(d), nanosec=int(d % 1 * 1e9)))]
    client.send_goal_async(goal)
    last, still = now[0], 0
    while still < 12 and abs(target - now[0]) > 0.004:
        time.sleep(0.1)
        still = still + 1 if abs(now[0] - last) < 0.003 else 0
        last = now[0]
    handle.succeed()
    r = GripperCommand.Result()
    r.position = now[0]
    return r
ActionServer(n, GripperCommand, '/xarm/xarm_gripper/gripper_action', execute, callback_group=group)
e = MultiThreadedExecutor(4); e.add_node(n); e.spin()
