"""
Cycle of the conveyor of world.usd: it is attached to /World/Conveyor, starts with Play and ends with Stop.

  PIECES random pieces fall on the far end of the belt, which takes them to the zone the robot reaches
  the belt stops there for WAIT seconds, then it all starts again

The times are simulated seconds, those the robot lives in: slower than the clock on the wall when Isaac Sim is.
Stop removes the pieces: save the stage after Stop, or they end up in the file. They cannot live in the session layer,
which would keep them out of it: physics writes where they are one layer below, and they would be drawn where they appeared.

Isaac Sim asks before running a script that comes with a stage: start it with --/app/scripting/ignoreWarningDialog=true,
or say yes to the dialog.
"""

import math
import os
import random
import re
from pathlib import Path

import carb
from omni.behavior.scripting.core import BehaviorScript
from pxr import Gf, Usd, UsdGeom, UsdPhysics

BELT = '/World/Conveyor/Belt'
SPEED_ATTRIBUTE = 'physxSurfaceVelocity:surfaceVelocity'
PIECES_PATH = '/World/Pieces'

CODES = [                       # the part numbers that fall: the active ones of PIECE_BINS in pick_and_place.py
    '100000047465',
    '100000047481',
    '100000047531',
    '100000049205',
    '100000051189',
    '100000053152',
    '100000057387',
    '100000057388',
]
PIECES = 5                      # for each round of the belt
WAIT = float(os.environ.get('CONVEYOR_WAIT', 3.0))   # s stopped in the pick zone
SPEED = 0.2                     # m/s of the belt, along +x
GAP = 0.105                     # m between the pieces on the belt: closer, they touch and the camera cannot tell them apart
DROP_X = -1.5                   # where they fall, out of reach of the arm
DROP_Y = (-0.15, -0.05)         # across the belt
DROP_HEIGHT = 0.13              # m above it
STOP_X = -0.16                  # where the first piece of a round would stop: 3 cm further on than it does, it lags falling
FIRST_DROP = 0.5                # s after Play


class ConveyorCycle(BehaviorScript):

    def on_init(self):
        self.made = []
        self.reset()

    def on_destroy(self):
        self.remove_pieces()

    def on_play(self):
        self.files = self.piece_files()
        self.reset()
        self.belt(SPEED)
        carb.log_warn('conveyor: running, {} part numbers'.format(len(self.files)))

    def on_stop(self):
        self.belt(0.0)
        self.remove_pieces()
        self.reset()

    def on_update(self, current_time, delta_time):
        self.clock += delta_time
        if self.waiting:
            if self.clock >= WAIT:
                carb.log_warn('conveyor: running')
                self.reset()
                self.belt(SPEED)
            return
        if self.dropped < PIECES and self.clock >= FIRST_DROP + self.dropped * GAP / SPEED:
            self.drop()
        if self.clock >= FIRST_DROP + (STOP_X - DROP_X) / SPEED:
            carb.log_warn('conveyor: stopped in the pick zone for {:.0f} s'.format(WAIT))
            self.belt(0.0)
            self.waiting = True
            self.clock = 0.0

    def reset(self):
        self.clock = 0.0
        self.dropped = 0
        self.waiting = False

    def belt(self, speed):
        self.stage.GetPrimAtPath(BELT).GetAttribute(SPEED_ATTRIBUTE).Set(Gf.Vec3f(speed, 0.0, 0.0))

    def piece_files(self):
        """One .usd for each part number, among the ones next to the stage (the way export_piece_models.py finds them)."""
        files = {}
        for path in sorted(Path(self.stage.GetRootLayer().realPath).parent.rglob('*.usd')):
            match = re.search(r'STP_(\d{9,})_', path.name)
            if match:
                files.setdefault(match.group(1), str(path))
        codes = os.environ.get('CONVEYOR_CODES', ','.join(CODES)).split(',')
        return {n: f for n, f in files.items() if n in codes}

    def drop(self):
        self.dropped += 1
        number = random.choice(sorted(self.files))
        path = '{}/Belt_{}_{}'.format(PIECES_PATH, len(self.made), number)
        prim = self.stage.DefinePrim(path, 'Xform')
        prim.GetPayloads().AddPayload(self.files[number])
        self.stage.Load(path)
        UsdPhysics.RigidBodyAPI.Apply(prim)
        for part in Usd.PrimRange(prim):
            if part.IsA(UsdGeom.Mesh):
                UsdPhysics.CollisionAPI.Apply(part)
                UsdPhysics.MeshCollisionAPI.Apply(part).CreateApproximationAttr('convexDecomposition')
        # any way up, with its center where it should fall from: the origin of a part is wherever the CAD had it
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
        center = cache.ComputeUntransformedBound(prim).ComputeAlignedRange().GetMidpoint()
        a, b, c, d = (random.gauss(0.0, 1.0) for _ in range(4))
        norm = math.sqrt(a * a + b * b + c * c + d * d)
        orientation = Gf.Quatd(a / norm, b / norm, c / norm, d / norm)
        turned = Gf.Rotation(orientation).TransformDir(center)
        top = self.stage.GetPrimAtPath(BELT).GetAttribute('xformOp:translate').Get()[2] + 0.0105
        target = Gf.Vec3d(DROP_X, random.uniform(*DROP_Y), top + DROP_HEIGHT)
        xform = UsdGeom.Xformable(prim)
        xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(target - turned)
        xform.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(orientation)
        self.made.append(path)
        carb.log_warn('conveyor: {} falling'.format(number))

    def remove_pieces(self):
        for path in self.made:
            self.stage.RemovePrim(path)
        self.made = []
