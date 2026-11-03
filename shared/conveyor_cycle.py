"""
Cycle of the conveyor of world.usd: it is attached to /World/Conveyor, starts with Play and ends with Stop.

  PIECES random pieces fall on the far end of the belt, which takes them to the zone the robot reaches: parts of the
  mold written in mold.txt next to the stage (or in $MOLD_FILE), read again at every round
  the belt stops there, and starts again after WAIT seconds, when pick_and_place.py --conveyor says it has finished,
  or without it as soon as there are no pieces left on the belt

The belt and the robot talk through two small files next to the stage. Here: .conveyor_belt, "moving N", "stopped N" or
"off", N being the round. From pick_and_place.py: .conveyor_robot, "waiting", "working N" or "done N", gone when it ends.

The times are simulated seconds, those the robot lives in: slower than the clock on the wall when Isaac Sim is.
Stop removes the pieces: save the stage after Stop, or they end up in the file. They cannot live in the session layer,
which would keep them out of it: physics writes where they are one layer below, and they would be drawn where they appeared.

Isaac Sim asks before running a script that comes with a stage: start it with --/app/scripting/ignoreWarningDialog=true,
or say yes to the dialog.
"""

import json
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
BELT_FILE = '.conveyor_belt'    # what the belt is doing, for pick_and_place.py
ROBOT_FILE = '.conveyor_robot'  # what pick_and_place.py is doing, for the belt
BINS_CHANGED = '.bins_changed'  # bins changed on the web page of pick_and_place.py, one per line: emptied here

MOLD_FILE = os.environ.get('MOLD_FILE', '')    # the mold in production; empty: mold.txt next to the stage
PIECES = 5                      # for each round of the belt
WAIT = float(os.environ.get('CONVEYOR_WAIT', 3.0))   # s stopped at least, the pieces left on the belt can make it longer
MAX_WAIT = 300.0                # s stopped at most: a robot that died must not stop the line for ever
LYING_HEIGHT = 0.08             # m of the center of a piece above the belt: higher, it is in the gripper
SPEED = 0.2                     # m/s of the belt, along +x
GAP = 0.105                     # m between the pieces on the belt: closer, they touch and the camera cannot tell them apart
DROP_FROM_START = 0.2           # m along the belt where they fall, out of reach of the arm
DROP_ACROSS = 0.05              # m to each side of the middle of the belt
DROP_HEIGHT = 0.13              # m above it
STOP_BEFORE_END = 0.0           # m from the end of the belt where the first piece of a round would stop: it lags 3 cm falling
FIRST_DROP = 0.5                # s after Play


def lying_on(low, high, center):
    """The center of a piece is over the belt, from low to high corner, and not up in the gripper."""
    return (low[0] <= center[0] <= high[0] and low[1] <= center[1] <= high[1] and center[2] < high[2] + LYING_HEIGHT)


class ConveyorCycle(BehaviorScript):

    def on_init(self):
        self.made = []
        self.round = 0                  # goes on after Stop: a "done" of before must not count for a round of now
        self.since_bins = 0.0
        self.bins_read = 0              # lines of BINS_CHANGED already done
        self.reset()

    def on_destroy(self):
        self.remove_pieces()
        self.tell('off')

    def on_play(self):
        self.mold, self.files = None, {}
        self.choose_parts()
        self.empty_changed_bins(skip=True)     # what was changed before Play is not in this scene
        # where the belt is now: it can be moved in the scene
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
        belt = cache.ComputeWorldBound(self.stage.GetPrimAtPath(BELT)).ComputeAlignedRange()
        self.drop_x = belt.GetMin()[0] + DROP_FROM_START
        self.middle_y = (belt.GetMin()[1] + belt.GetMax()[1]) / 2
        self.top = belt.GetMax()[2]
        self.travel = belt.GetMax()[0] - STOP_BEFORE_END - self.drop_x
        self.reset()
        self.belt(SPEED)
        self.tell('moving')
        carb.log_warn('conveyor: running, {} part numbers'.format(len(self.files)))

    def on_stop(self):
        self.belt(0.0)
        self.remove_pieces()
        self.reset()
        self.tell('off')

    def on_update(self, current_time, delta_time):
        self.clock += delta_time
        self.since_bins += delta_time
        if self.since_bins > 1.0:
            self.since_bins = 0.0
            self.empty_changed_bins()
        if self.waiting:
            if self.clock >= WAIT and self.clock >= self.next_check:
                self.next_check = self.clock + 1.0
                if not self.may_go_on():
                    return                  # moving now would take the pieces from under the robot
                carb.log_warn('conveyor: running')
                self.choose_parts()
                self.reset()
                self.belt(SPEED)
                self.tell('moving')
            return
        if self.dropped < PIECES and self.clock >= FIRST_DROP + self.dropped * GAP / SPEED:
            self.drop()
        if self.clock >= FIRST_DROP + self.travel / SPEED:
            self.round += 1
            carb.log_warn('conveyor: round {} stopped in the pick zone'.format(self.round))
            self.belt(0.0)
            self.waiting = True
            self.clock = 0.0
            self.tell('stopped')

    def reset(self):
        self.clock = 0.0
        self.next_check = 0.0
        self.dropped = 0
        self.waiting = False

    def may_go_on(self):
        """The robot decides when it is there: what it left on the belt, it left. Without it, nothing must be left."""
        robot = self.robot_says()
        left = self.pieces_on_belt()
        if robot == ['done', str(self.round)] or (not robot and not left):
            return True
        if self.clock < max(WAIT, MAX_WAIT):
            return False
        carb.log_warn('conveyor: going on after {:.0f} s, robot "{}", {} pieces on the belt'.format(self.clock, ' '.join(robot), left))
        return True

    def folder(self):
        return Path(self.stage.GetRootLayer().realPath).parent

    def tell(self, state):
        """Written whole in one go: pick_and_place.py must never read half of it."""
        try:
            temporary = self.folder() / (BELT_FILE + '.tmp')
            temporary.write_text('{} {}'.format(state, self.round))
            temporary.replace(self.folder() / BELT_FILE)
        except OSError as e:
            carb.log_warn('conveyor: cannot write {}: {}'.format(BELT_FILE, e))

    def robot_says(self):
        try:
            return (self.folder() / ROBOT_FILE).read_text().split()
        except OSError:
            return []

    def pieces_on_belt(self):
        """How many of the pieces are still lying on the belt."""
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
        belt = cache.ComputeWorldBound(self.stage.GetPrimAtPath(BELT)).ComputeAlignedRange()
        centers = [cache.ComputeWorldBound(self.stage.GetPrimAtPath(path)).ComputeAlignedRange().GetMidpoint() for path in self.made]
        return sum(lying_on(belt.GetMin(), belt.GetMax(), center) for center in centers)

    def belt(self, speed):
        self.stage.GetPrimAtPath(BELT).GetAttribute(SPEED_ATTRIBUTE).Set(Gf.Vec3f(speed, 0.0, 0.0))

    def choose_parts(self):
        """The parts of the mold in production, if it changed: one .usd for each part number, in the folder of the mold
        next to the stage (Stampo <mold>, the way export_piece_models.py finds them)."""
        try:
            mold = (Path(MOLD_FILE) if MOLD_FILE else self.folder() / 'mold.txt').read_text().strip()
        except OSError as e:
            carb.log_warn('conveyor: no mold ({}), keeping {}'.format(e, self.mold))
            return
        if mold == self.mold:
            return
        files = {}
        for path in sorted(self.folder().rglob('Stampo {}/**/*.usd'.format(mold))):
            match = re.search(r'STP_(\d{9,})_', path.name)
            if match:
                files.setdefault(match.group(1), str(path))
        if not files:
            carb.log_warn('conveyor: no part of mold {} next to the stage, keeping {}'.format(mold, self.mold))
            return
        self.mold, self.files = mold, files
        carb.log_warn('conveyor: mold {}, {}'.format(mold, ', '.join(sorted(files))))

    def drop(self):
        self.dropped += 1
        if not self.files:
            return
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
        target = Gf.Vec3d(self.drop_x, self.middle_y + random.uniform(-DROP_ACROSS, DROP_ACROSS), self.top + DROP_HEIGHT)
        xform = UsdGeom.Xformable(prim)
        xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(target - turned)
        xform.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(orientation)
        self.made.append(path)
        carb.log_warn('conveyor: {} falling'.format(number))

    def empty_changed_bins(self, skip=False):
        """A bin changed for an empty one on the web page: the pieces of the belt in it go. skip: only take note of the
        bins changed so far."""
        try:
            lines = (self.folder() / BINS_CHANGED).read_text().splitlines()
        except OSError:
            return
        if not skip:
            cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
            for line in lines[self.bins_read:]:
                try:
                    changed = json.loads(line)
                except ValueError:
                    continue
                c, s = math.cos(changed['angle']), math.sin(changed['angle'])
                gone = 0
                for path in list(self.made):
                    center = cache.ComputeWorldBound(self.stage.GetPrimAtPath(path)).ComputeAlignedRange().GetMidpoint()
                    dx, dy = center[0] - changed['center'][0], center[1] - changed['center'][1]
                    if (abs(dx * c + dy * s) < changed['length'] / 2 + 0.01 and abs(-dx * s + dy * c) < changed['width'] / 2 + 0.01
                            and center[2] < 0.3):
                        self.stage.RemovePrim(path)
                        self.made.remove(path)
                        gone += 1
                carb.log_warn('conveyor: bin at ({:.3f}, {:.3f}) changed, {} pieces taken away'.format(
                    changed['center'][0], changed['center'][1], gone))
        self.bins_read = len(lines)

    def remove_pieces(self):
        for path in self.made:
            self.stage.RemovePrim(path)
        self.made = []
