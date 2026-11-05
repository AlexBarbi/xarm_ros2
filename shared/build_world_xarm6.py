"""world_xarm6.usd: world.usd with the xArm6 of the Isaac Sim assets (with the xArm gripper) in place of the xArm5,
everything else as it is (table, belt and its script, bins). Nothing is copied here: the robot is referenced from the
assets, so Isaac Sim needs to reach them when it opens the stage. Inside the Isaac Sim container, with nothing open on
world.usd:  /isaac-sim/python.sh /workspace/build_world_xarm6.py

The robot gets what world.usd gives the xArm5 by hand: the drives of the joints (the 6 has one more in the wrist, it takes
those of the others), the force of the gripper, the RealSense on the wrist where the URDF has it, and the ROS 2 graph of
joints, gripper, camera and clock, pointed at the new robot."""
import os

from isaacsim import SimulationApp

app = SimulationApp({'headless': True})

from isaacsim.storage.native import get_assets_root_path  # noqa: E402
from pxr import Sdf, Usd, UsdPhysics  # noqa: E402

HERE = '/workspace'
SOURCE = HERE + '/world.usd'
TARGET = HERE + '/world_xarm6.usd'
ROBOT = get_assets_root_path() + '/Isaac/Robots/Ufactory/xarm6/xarm6.usd'
VARIANTS = {'Physics': 'PhysX', 'Variant_Set': 'xarm_gripper', 'Sensor': 'None'}
OLD = '/World/xarm5'
NEW = '/World/xarm6'
DRIVE_ATTRIBUTES = ('drive:angular:physics:type', 'drive:angular:physics:stiffness', 'drive:angular:physics:damping',
                    'drive:angular:physics:targetPosition')
GRIPPER_FORCE = 'drive:angular:physics:maxForce'


def find(spec, name):
    if spec.name == name:
        return spec
    for child in spec.nameChildren:
        found = find(child, name)
        if found:
            return found
    return None


source = Sdf.Layer.FindOrOpen(SOURCE)
if os.path.exists(TARGET):
    os.remove(TARGET)
target = Sdf.Layer.CreateNew(TARGET)
target.TransferContent(source)

# what world.usd sets on the joints of the xArm5: the same on those of the 6
arm_drive = {name: source.GetPrimAtPath(OLD + '/Physics/joint2').attributes[name].default for name in DRIVE_ATTRIBUTES}
gripper = source.GetPrimAtPath(OLD + '/Physics/drive_joint')
gripper_drive = {name: gripper.attributes[name].default for name in DRIVE_ATTRIBUTES + (GRIPPER_FORCE,)}
camera = find(source.GetPrimAtPath(OLD), 'RealSense')

del target.GetPrimAtPath('/World').nameChildren[OLD.rsplit('/', 1)[1]]
robot = Sdf.CreatePrimInLayer(target, NEW)
robot.specifier = Sdf.SpecifierDef
robot.typeName = 'Xform'
robot.referenceList.Prepend(Sdf.Reference(ROBOT))
for name, choice in VARIANTS.items():
    robot.variantSelections[name] = choice
# the RealSense on the wrist: the same mount on link_eef as on the 5
camera_path = Sdf.Path(NEW + '/link6/link_eef/RealSense')
Sdf.CreatePrimInLayer(target, camera_path.GetParentPath())
assert Sdf.CopySpec(source, camera.path, target, camera_path)
target.Save()

stage = Usd.Stage.Open(target)
prim = stage.GetPrimAtPath(NEW)
# The gripper comes with an articulation root of its own, fixed to link6: one articulation for the whole robot, from
# the root of the arm, or the ROS 2 graph drives only one of them
roots = [p for p in Usd.PrimRange(prim) if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
arm_root = next(p for p in roots if '/gripper/' not in str(p.GetPath()))
for other in roots:
    if other != arm_root:
        other.RemoveAPI(UsdPhysics.ArticulationRootAPI)
roots = [arm_root.GetPath()]
print('RESULT articulation root', roots[0], flush=True)
for joint in Usd.PrimRange(prim):
    if not joint.IsA(UsdPhysics.RevoluteJoint) or not joint.HasAPI(UsdPhysics.DriveAPI, 'angular'):
        continue                    # the fingers follow drive_joint as mimic joints
    values = gripper_drive if joint.GetName() == 'drive_joint' else arm_drive
    for name, value in values.items():
        attribute = joint.GetAttribute(name)
        if not attribute:
            attribute = joint.CreateAttribute(name, Sdf.ValueTypeNames.Token if name.endswith(':type') else Sdf.ValueTypeNames.Float)
        attribute.Set(value)
    print('RESULT drive', joint.GetPath(), {k.split(':')[-1]: v for k, v in values.items()}, flush=True)
stage.Save()

# the ROS 2 graph, pointed at the new robot: one articulation, the gripper is part of it
for node in target.GetPrimAtPath('/ActionGraph').nameChildren:
    for relationship in node.relationships:
        targets = list(relationship.targetPathList.explicitItems) + list(relationship.targetPathList.prependedItems)
        if not targets:
            continue
        new_targets = [camera_path if t.name == 'RealSense' else roots[0] for t in targets]
        relationship.targetPathList.ClearEdits()
        relationship.targetPathList.explicitItems = new_targets
        print('RESULT graph', node.path.name, relationship.name, [str(t) for t in new_targets], flush=True)
target.Save()

stage = Usd.Stage.Open(TARGET)
print('RESULT check camera', bool(stage.GetPrimAtPath(camera_path)), 'joints',
      sorted(p.GetName() for p in Usd.PrimRange(stage.GetPrimAtPath(NEW)) if p.IsA(UsdPhysics.RevoluteJoint)), flush=True)
app.close()
