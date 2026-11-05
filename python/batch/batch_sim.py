#!/usr/bin/env python3
"""
The Isaac Sim side of batch_test.py: lays out the pieces of a run and tells where they really are.

Started by batch_test.py with the python of Isaac Sim and a folder, where config.json says what to load. Talks through
files in that folder: cmd.json from batch_test.py, reply.json back, and a heartbeat that stops when the renderer hangs.
Nothing is saved: the pieces added to the scene live in memory.
"""

import glob
import json
import math
import os
import random
import re
import sys
import traceback

import numpy as np
from isaacsim import SimulationApp

STATE = sys.argv[1]
CONFIG = json.load(open(os.path.join(STATE, 'config.json')))
with open(os.path.join(STATE, 'sim.pid'), 'w') as f:
    f.write(str(os.getpid()))

# the scripts of the stage are only allowed for the belt: the extension that runs them is on in any case
app = SimulationApp({'headless': True, 'extra_args': ['--/app/scripting/ignoreWarningDialog=true'] if CONFIG['belt'] else []})

from isaacsim.core.utils.extensions import enable_extension  # noqa: E402

enable_extension('isaacsim.ros2.bridge')
if CONFIG['belt']:
    enable_extension('omni.behavior.scripting.bundle')     # without it the script of the conveyor does not run
app.update()

import omni.physx  # noqa: E402
import omni.timeline  # noqa: E402
import omni.usd  # noqa: E402
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics  # noqa: E402

PIECES_PATH = '/World/Pieces'
BELT = '/World/Conveyor/Belt'
BELT_SPEED = 'physxSurfaceVelocity:surfaceVelocity'
BINS_PATH = '/World/Bins'
PARK = (3.0, 3.0, -0.98)        # on the ground, far from everything
SETTLE_FRAMES = 150             # after Play, for the pieces to come to rest
BELT_PIECES = 5                 # a round of the conveyor
EDGE = 0.04                     # m kept free at the near end of the pick zone and along the sides of the belt
FAR_X = -0.62                   # the far end of the pick zone, where a round of the belt leaves its last piece


class Piece:

    def __init__(self, prim):
        self.path = str(prim.GetPath())
        self.number = re.search(r'(?:STP_|_)(\d{12})(?:_|$)', prim.GetName()).group(1)
        to_body = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0).GetInverse()
        points = []
        for part in Usd.PrimRange(prim):
            if part.IsA(UsdGeom.Mesh):
                to_world = UsdGeom.Xformable(part).ComputeLocalToWorldTransform(0)
                raw = UsdGeom.Mesh(part).GetPointsAttr().Get()
                points.extend(to_body.Transform(to_world.Transform(p)) for p in raw[::max(len(raw) // 3000, 1)])
        self.points = np.array([[p[0], p[1], p[2]] for p in points])
        self.center = (self.points.min(axis=0) + self.points.max(axis=0)) / 2
        self.home = (prim.GetAttribute('xformOp:translate').Get(), prim.GetAttribute('xformOp:orient').Get())


def add_copies(stage):
    """As many pieces of each code as asked for: the codes the scene has none of come from the .usd next to the stage."""
    have = {}
    for prim in stage.GetPrimAtPath(PIECES_PATH).GetChildren():
        match = re.search(r'STP_(\d{12})_', prim.GetName())
        if match:
            have[match.group(1)] = have.get(match.group(1), 0) + 1
    files = {}
    for path in sorted(glob.glob(os.path.join(os.path.dirname(CONFIG['stage']), '**', '*.usd'), recursive=True)):
        match = re.search(r'STP_(\d{12})_', os.path.basename(path))
        if match:
            files.setdefault(match.group(1), path)
    for number, path in files.items():
        if number not in CONFIG['codes']:
            continue
        for i in range(max(CONFIG['pool'] - have.get(number, 0), 0)):
            prim = stage.DefinePrim('{}/pool_{}_{}'.format(PIECES_PATH, i, number), 'Xform')
            prim.GetPayloads().AddPayload(path)
            UsdGeom.Xformable(prim).AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*PARK))
            UsdGeom.Xformable(prim).AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(1, 0, 0, 0))
            UsdPhysics.RigidBodyAPI.Apply(prim)
            for _ in range(3):
                app.update()
            for part in Usd.PrimRange(prim):
                if part.IsA(UsdGeom.Mesh):
                    UsdPhysics.CollisionAPI.Apply(part)
                    UsdPhysics.MeshCollisionAPI.Apply(part).CreateApproximationAttr('convexDecomposition')
    while ctx.get_stage_loading_status()[2] > 0:
        app.update()


def place(stage, piece, orientation, x, y, z):
    turned = Gf.Rotation(orientation).TransformDir(Gf.Vec3d(*piece.center))
    prim = stage.GetPrimAtPath(piece.path)
    prim.GetAttribute('xformOp:translate').Set(Gf.Vec3d(x - turned[0], y - turned[1], z - turned[2]))
    attribute = prim.GetAttribute('xformOp:orient')
    attribute.Set(orientation if attribute.GetTypeName() == 'quatd' else Gf.Quatf(orientation))


def extent(piece, orientation):
    r = Gf.Rotation(orientation)
    turned = np.array([list(r.TransformDir(Gf.Vec3d(*p))) for p in piece.points[::5]])
    return turned.min(axis=0), turned.max(axis=0)


def pick_zone(stage):
    """(x0, x1, y0, y1) where pieces are laid: on the belt, wherever it is, in the part of it the camera looks at."""
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    belt = cache.ComputeWorldBound(stage.GetPrimAtPath(BELT)).ComputeAlignedRange()
    return (max(belt.GetMin()[0] + EDGE, FAR_X), min(belt.GetMax()[0], CONFIG['work_x'][1]) - EDGE,
            belt.GetMin()[1] + EDGE, belt.GetMax()[1] - EDGE)


def choose(rng, count, codes):
    """count pieces, each code as likely as the others, as long as there are pieces of it left."""
    left = {}
    for name, piece in pieces.items():
        if piece.number in codes:
            left.setdefault(piece.number, []).append(name)
    chosen = []
    while len(chosen) < count and left:
        number = rng.choice(sorted(left))
        chosen.append(left[number].pop())
        if not left[number]:
            del left[number]
    return chosen


def lay_out(stage, scenario, seed, count, codes):
    rng = random.Random(seed)
    if scenario == 'scene':
        chosen = [n for n in pieces if '/STP_' in pieces[n].path and pieces[n].number in codes]
    elif scenario == 'belt':
        chosen = []
    else:
        chosen = choose(rng, count, codes)
    x0, x1, y0, y1 = pick_zone(stage)
    if scenario == 'rejects':               # in the bin of the rejects, the last one of the sorter (right, furthest out)
        rejects = max((b for b in BINS if b['center'][1] <= 0), key=lambda b: math.hypot(*b['center']))
        (cx, cy), (lx, ly) = rejects['center'], rejects['inner']
        x0, x1, y0, y1 = cx - lx / 2 + EDGE, cx + lx / 2 - EDGE, cy - ly / 2 + EDGE, cy + ly / 2 - EDGE
    for i, name in enumerate(sorted(set(pieces) - set(chosen))):
        place(stage, pieces[name], Gf.Quatd(1, 0, 0, 0), PARK[0] + i * 0.3, PARK[1], PARK[2])
    if scenario == 'scene':
        for name in chosen:
            prim = stage.GetPrimAtPath(pieces[name].path)
            prim.GetAttribute('xformOp:translate').Set(pieces[name].home[0])
            prim.GetAttribute('xformOp:orient').Set(pieces[name].home[1])
        return chosen
    taken = []
    free = list(chosen)
    if scenario == 'touching':
        spots = [(x0 + (x1 - x0) * (i + 0.5) / 3, y0 + 0.01) for i in range(3)]     # pairs across the belt
        for cx, cy in spots[:len(chosen) // 2]:
            first, second = free.pop(), free.pop()
            lying = Gf.Quatd(math.cos(math.pi / 4), 0, math.sin(math.pi / 4), 0)
            lo, hi = extent(pieces[first], lying)
            place(stage, pieces[first], lying, cx, cy, 0.004 + (hi[2] - lo[2]) / 2)
            upright = Gf.Quatd(1, 0, 0, 0)
            lo2, hi2 = extent(pieces[second], upright)
            y = cy + (hi[1] - lo[1]) / 2 + 0.001 + (hi2[1] - lo2[1]) / 2
            place(stage, pieces[second], upright, cx, y, 0.004 + (hi2[2] - lo2[2]) / 2)
            taken.extend([(cx, cy), (cx, y)])
    for name in free:
        for _ in range(2000):
            x, y = rng.uniform(x0, x1), rng.uniform(y0, y1)
            if scenario == 'rejects' or all(math.hypot(x - u, y - v) > 0.11 for u, v in taken):
                break
        taken.append((x, y))
        yaw = rng.uniform(-math.pi, math.pi)
        orientation = Gf.Quatd(math.cos(yaw / 2), 0, 0, math.sin(yaw / 2))
        z = 0.07 if scenario != 'rejects' else RIM_Z + 0.04 + 0.03 * len(taken)   # dropped in, one over the other
        if scenario == 'tipped' and rng.random() < 0.7:
            roll = rng.choice([math.pi / 2, -math.pi / 2])
            orientation, z = orientation * Gf.Quatd(math.cos(roll / 2), math.sin(roll / 2), 0, 0), 0.09
        place(stage, pieces[name], orientation, x, y, z)
    return chosen


def turned_points(piece, position, rotation):
    x, y, z, w = rotation
    matrix = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                       [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                       [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    return piece.points @ matrix.T + np.array(position)


def belt_pieces(stage):
    prim = stage.GetPrimAtPath(PIECES_PATH)
    return [p for p in prim.GetChildren() if p.GetName().startswith('Belt_')]


def belt_speed(stage):
    prim = stage.GetPrimAtPath(BELT)
    return abs(prim.GetAttribute(BELT_SPEED).Get()[0]) if prim else 0.0


def snapshot(stage):
    for prim in belt_pieces(stage):
        name = prim.GetName()
        if name not in pieces:
            pieces[name] = Piece(prim)
            active.append(name)
    out = {'t': timeline.get_current_time(), 'pieces': {}, 'fingers': [], 'bins': BINS}
    for path in FINGERS:
        d = physx.get_rigidbody_transformation(path)
        if d['ret_val']:
            out['fingers'].append(list(d['position']))
    out['bin_poses'] = {}                   # where the bins really are: the robot can lift one
    for prim in stage.GetPrimAtPath(BINS_PATH).GetChildren():
        d = physx.get_rigidbody_transformation(str(prim.GetPath()))
        if d['ret_val']:
            out['bin_poses'][prim.GetName()] = [list(d['position']), list(d['rotation'])]
    for name in active:
        piece = pieces[name]
        d = physx.get_rigidbody_transformation(piece.path)
        if not d['ret_val'] or not len(piece.points):
            continue
        w = turned_points(piece, d['position'], d['rotation'])
        below = w[w[:, 2] < RIM_Z]       # under the rims: the walls of a bin hold that part
        out['pieces'][name] = {
            'number': piece.number, 'low': w.min(axis=0).tolist(), 'high': w.max(axis=0).tolist(),
            'mean': w.mean(axis=0).tolist(),
            'below_low': below.min(axis=0).tolist() if len(below) else None,
            'below_high': below.max(axis=0).tolist() if len(below) else None}
    return out


def robot_part(stage, name, kind=None):
    """The prim of this name in the robot, the rigid body if there are several: the robots of world.usd and
    world_xarm6.usd have the same parts in different places."""
    found = [p for p in stage.Traverse() if p.GetName() == name and (kind is None or p.IsA(kind))]
    if not found:
        raise RuntimeError('no {} in {}'.format(name, CONFIG['stage']))
    rigid = [p for p in found if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    return str((rigid or found)[0].GetPath())


def reply(message_id, **fields):
    temporary = os.path.join(STATE, 'reply.tmp')
    with open(temporary, 'w') as f:
        json.dump(dict(fields, id=message_id), f)
    os.replace(temporary, os.path.join(STATE, 'reply.json'))


ctx = omni.usd.get_context()
ctx.open_stage(CONFIG['stage'])
for _ in range(50):
    app.update()
while ctx.get_stage_loading_status()[2] > 0:
    app.update()
for _ in range(50):
    app.update()
stage = ctx.get_stage()
FINGERS = (robot_part(stage, 'left_finger'), robot_part(stage, 'right_finger'))
GRIPPER_DRIVE = robot_part(stage, 'drive_joint', UsdPhysics.RevoluteJoint)
print('fingers:', *FINGERS, flush=True)
if not CONFIG['belt'] and stage.GetPrimAtPath('/World/Conveyor'):
    with Usd.EditContext(stage, stage.GetSessionLayer()):       # in memory: without it the belt drops pieces on any run
        stage.GetPrimAtPath('/World/Conveyor').GetAttribute('omni:scripting:scripts').Set(Sdf.AssetPathArray([]))
if CONFIG.get('grip_torque'):
    with Usd.EditContext(stage, stage.GetSessionLayer()):       # for this batch only, world.usd stays as it is
        stage.GetPrimAtPath(GRIPPER_DRIVE).GetAttribute('drive:angular:physics:maxForce').Set(float(CONFIG['grip_torque']))
print('gripper motor:', stage.GetPrimAtPath(GRIPPER_DRIVE).GetAttribute('drive:angular:physics:maxForce').Get(), 'N*m', flush=True)
add_copies(stage)


def true_bins(stage):
    """The bins of the stage, as they are: center, inside size and rim, for the judge."""
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    found = []
    for prim in stage.GetPrimAtPath(BINS_PATH).GetChildren():
        box = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        walls = [cache.ComputeWorldBound(c).ComputeAlignedRange() for c in prim.GetChildren() if c.GetName().startswith('Wall')]
        wall = min(min(w.GetMax()[0] - w.GetMin()[0], w.GetMax()[1] - w.GetMin()[1]) for w in walls) if walls else 0.0
        found.append({'center': [(box.GetMin()[0] + box.GetMax()[0]) / 2, (box.GetMin()[1] + box.GetMax()[1]) / 2],
                      'inner': [box.GetMax()[0] - box.GetMin()[0] - 2 * wall, box.GetMax()[1] - box.GetMin()[1] - 2 * wall],
                      'rim': box.GetMax()[2]})
    return found


BINS = true_bins(stage)
RIM_Z = max(b['rim'] for b in BINS)
pieces = {p.GetName(): Piece(p) for p in stage.GetPrimAtPath(PIECES_PATH).GetChildren()}
active = []
timeline = omni.timeline.get_timeline_interface()
physx = omni.physx.get_physx_interface()
timeline.play()
for _ in range(10):
    app.update()
with open(os.path.join(STATE, 'ready'), 'w') as f:
    f.write(json.dumps({'pieces': len(pieces)}))

frame = 0
message = {'id': -1}
waiting = None          # {'id', 'frames', 'belt'}: a reset that is still coming to rest
command = os.path.join(STATE, 'cmd.json')
while app.is_running():
    app.update()
    frame += 1
    if frame % 30 == 0:
        with open(os.path.join(STATE, 'heartbeat'), 'w') as f:
            f.write(str(frame))
    try:
        if waiting:
            waiting['frames'] -= 1
            if waiting['belt']:
                ready = len(belt_pieces(stage)) >= BELT_PIECES and belt_speed(stage) == 0.0
                waiting['frames'] = SETTLE_FRAMES // 3 if not ready else waiting['frames']
            if waiting['frames'] <= 0:
                reply(waiting['id'], ok=True, snapshot=snapshot(stage))
                waiting = None
            continue
        if not os.path.exists(command):
            continue
        message = json.load(open(command))
        os.remove(command)
        if message['cmd'] == 'quit':
            reply(message['id'], ok=True)
            break
        if message['cmd'] == 'snapshot':
            reply(message['id'], ok=True, snapshot=snapshot(stage))
        if message['cmd'] == 'reset':
            timeline.stop()
            for _ in range(20):
                app.update()
            for name in [n for n in pieces if n.startswith('Belt_')]:
                del pieces[name]
            random.seed(message['seed'])        # the script of the conveyor draws from here
            active = lay_out(stage, message['scenario'], message['seed'], message['count'], message['codes'])
            app.update()
            timeline.play()
            waiting = {'id': message['id'], 'frames': SETTLE_FRAMES, 'belt': message['scenario'] == 'belt'}
    except Exception:
        reply(waiting['id'] if waiting else message['id'], ok=False, error=traceback.format_exc())
        waiting = None

timeline.stop()
app.close()
