"""Set the gripper motor of world.usd (maxForce of drive_joint, N*m) and save it. Inside the Isaac Sim container, with
nothing else open on world.usd: /isaac-sim/python.sh /workspace/set_grip_torque.py 2.2"""
import sys

from isaacsim import SimulationApp

app = SimulationApp({'headless': True})

from pxr import Usd  # noqa: E402

PATH = '/workspace/world.usd'
JOINT = '/World/xarm5/Physics/drive_joint'
stage = Usd.Stage.Open(PATH)
attribute = stage.GetPrimAtPath(JOINT).GetAttribute('drive:angular:physics:maxForce')
print('drive_joint maxForce was', attribute.Get(), flush=True)
attribute.Set(float(sys.argv[1]))
stage.GetRootLayer().Save()
print('drive_joint maxForce now', Usd.Stage.Open(PATH).GetPrimAtPath(JOINT).GetAttribute('drive:angular:physics:maxForce').Get(), flush=True)
app.close()
