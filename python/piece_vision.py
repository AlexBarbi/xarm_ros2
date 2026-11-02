#!/usr/bin/env python3
"""
Find, recognize and plan top-down grasps of the pieces lying on a table, from a depth image. No ROS in here.

  TableMap        height map of the table from a depth view, split into separate pieces
  PieceLibrary    part numbers from the CAD: each part is recognized by its size in every pose it can rest in
  TableMap.grasp  best place to hold a piece: flat opposite sides, room for the fingers, close to its center of mass

All lengths in meters. Heights are above the table. The grasp yaw is the one of pick_and_place.top_down: at yaw 0 the
fingers close along y of the base.
"""

import json
import math
from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

CELL = 0.001                    # height map resolution
MIN_PIECE_HEIGHT = 0.004        # lower than this is the table
MAX_PIECE_HEIGHT = 0.14         # higher than this is the gripper in the view
MIN_PIECE_CELLS = 150
PIECE_MERGE_CELLS = 3           # parts of the height map closer than twice this are the same piece
MAX_HIDDEN_CELLS = 3000         # unseen areas smaller than this are shadows of what stands next to them
SPLIT_STEP = 0.002              # height between the slices that take apart pieces touching each other

SIGNATURE_TOLERANCE = 0.009     # worst difference of height / length / width from a resting pose of the part
SPLIT_TOLERANCE = 0.005         # the same for a slice of a blob: a part of one piece can look like another whole one

# xArm gripper, finger geometry from xarm_gripper.urdf.xacro and left_finger.stl
PAD_WIDTH = 0.032               # along the fingers' x
PAD_THICKNESS = 0.014           # along the closing direction
MAX_GAP = 0.085
MIN_GRASP_WIDTH = 0.01          # thinner: the fingers end up against their end stop, and the pieces slip out
MAX_GRASP_WIDTH = 0.068
FINGER_MARGIN = 0.008           # free space kept on each side of the object while the fingers come down
TIP_CLEARANCE = 0.003           # fingertips above what is under the fingers
TIP_TABLE_CLEARANCE = 0.003     # fingertips above the table
MIN_ENGAGE = 0.005              # how much of the object is between the pads: from the top of the object to the tips
MAX_ENGAGE = 0.032
MIN_CONTACT_ROWS = 6            # mm of pad width really touching both sides of the object
ENGAGE_WEIGHT = 500             # 2 mm deeper is worth 1 mm further from the center of mass: grasps near the top slip
YAW_STEP_DEG = 15
SAME_GRASP_DISTANCE = 0.012     # what counts as trying the same grasp again
SAME_GRASP_YAW = math.radians(20)


def finger_gap(position):
    """Distance between the pads at this drive_joint position."""
    return 2 * (0.035465 * math.cos(position) - 0.042039 * math.sin(position) + 0.035 - 0.025965)


def finger_tip(position):
    """How far the fingertips reach past link_tcp at this drive_joint position (negative: short of it)."""
    return 0.035465 * math.sin(position) + 0.042039 * math.cos(position) + 0.059098 + 0.061 - 0.172


_POSITIONS = np.linspace(0.0, 0.85, 86)
_GAPS = np.array([finger_gap(p) for p in _POSITIONS])


def finger_position(gap):
    """drive_joint position for this distance between the pads."""
    return float(np.interp(gap, _GAPS[::-1], _POSITIONS[::-1]))


def min_area_rectangle(points):
    """(length, width, angle of the length axis) of the smallest rectangle around 2D points."""
    hull = points[ConvexHull(points).vertices]
    edges = np.roll(hull, -1, axis=0) - hull
    best = None
    for angle in np.arctan2(edges[:, 1], edges[:, 0]):
        c, s = math.cos(angle), math.sin(angle)
        u, v = hull[:, 0] * c + hull[:, 1] * s, -hull[:, 0] * s + hull[:, 1] * c
        du, dv = np.ptp(u), np.ptp(v)
        if best is None or du * dv < best[0]:
            best = (du * dv, du, dv, angle) if du >= dv else (du * dv, dv, du, angle + math.pi / 2)
    return best[1], best[2], best[3]


def resting_rotations(points, center_of_mass):
    """Rotations of a part onto each stable pose: faces of its convex hull with the center of mass above them."""
    hull = ConvexHull(points)
    faces = []
    for simplex, equation in zip(hull.simplices, hull.equations):
        normal = equation[:3]
        for face in faces:
            if face['normal'] @ normal > 0.9995:
                face['vertices'].update(simplex)
                break
        else:
            faces.append({'normal': normal, 'offset': equation[3], 'vertices': set(simplex)})
    rotations = []
    for face in faces:
        if len(face['vertices']) < 3:
            continue
        axis = np.cross(face['normal'], (0.0, 0.0, -1.0))
        angle = math.atan2(np.linalg.norm(axis), -face['normal'][2])
        axis = axis / np.linalg.norm(axis) if np.linalg.norm(axis) > 1e-9 else np.array((1.0, 0.0, 0.0))
        rotation = Rotation.from_rotvec(axis * angle)   # the face down
        outline = rotation.apply(points[sorted(face['vertices'])])[:, :2]
        center = rotation.apply(center_of_mass)[:2]
        try:
            contour = ConvexHull(outline)
        except Exception:
            continue                                # a segment, no area to stand on
        if np.all(contour.equations[:, :2] @ center + contour.equations[:, 2] < -0.001):
            rotations.append(rotation)
    return rotations


class PieceLibrary:

    def __init__(self, path, part_numbers):
        with open(path) as f:
            models = json.load(f)
        self.signatures = []        # (part number, height, length, width) of each resting pose
        for number in part_numbers:
            points = np.array(models[number]['points'])
            center_of_mass = np.array(models[number]['center_of_mass'])
            for rotation in resting_rotations(points, center_of_mass):
                p = rotation.apply(points)
                length, width, _ = min_area_rectangle(p[:, :2])
                self.signatures.append((number, float(np.ptp(p[:, 2])), length, width))

    def identify(self, piece):
        """(part number or None, worst size difference from the best match)"""
        best, error = None, 1e9
        for number, height, length, width in self.signatures:
            e = max(abs(height - piece.height), abs(length - piece.length), abs(width - piece.width))
            if e < error:
                best, error = number, e
        return (best if error < SIGNATURE_TOLERANCE else None), error


@dataclass
class Piece:
    label: int                  # in TableMap.labels
    center: tuple               # (x, y) of the smallest rectangle around it
    height: float
    length: float
    width: float
    angle: float                # of the length axis
    center_of_mass: tuple       # (x, y), estimated from the height map


@dataclass
class Grasp:
    x: float
    y: float
    z: float                    # of the TCP, above the table
    yaw: float
    width: float                # of what ends up between the pads
    engage: float


def _same_grasp(x, y, yaw, grasp):
    turn = abs(yaw - grasp.yaw) % math.pi
    return math.hypot(x - grasp.x, y - grasp.y) < SAME_GRASP_DISTANCE and min(turn, math.pi - turn) < SAME_GRASP_YAW


class TableMap:

    def __init__(self, table_z, x_range, y_range, keep_out):
        """keep_out: (x, y, radius) circles where what the camera sees is not a piece."""
        self.table_z = table_z
        self.x0, self.y0 = x_range[0], y_range[0]
        self.shape = (int(round((x_range[1] - x_range[0]) / CELL)), int(round((y_range[1] - y_range[0]) / CELL)))
        self.height = np.zeros(self.shape)
        self.seen = np.zeros(self.shape, bool)
        self.keep_out = keep_out
        self.labels = None

    def add_view(self, depth, fx, fy, cx, cy, rotation, translation):
        """Project a depth image (meters) with the camera pose (optical frame in the frame of the table)."""
        v, u = np.nonzero(np.isfinite(depth) & (depth > 0.05) & (depth < 3.0))
        z = depth[v, u]
        points = np.stack([(u - cx) * z / fx, (v - cy) * z / fy, z], axis=1) @ rotation.T + translation
        ix = np.floor((points[:, 0] - self.x0) / CELL).astype(int)
        iy = np.floor((points[:, 1] - self.y0) / CELL).astype(int)
        h = points[:, 2] - self.table_z
        keep = (ix >= 0) & (ix < self.shape[0]) & (iy >= 0) & (iy < self.shape[1]) & (h < MAX_PIECE_HEIGHT)
        self.seen[ix[keep], iy[keep]] = True
        np.maximum.at(self.height, (ix[keep], iy[keep]), h[keep])

    def segment(self):
        """Split the height map into pieces."""
        height = ndimage.grey_closing(self.height, size=3)     # the camera pixels are larger than the cells
        xs = self.x0 + (np.arange(self.shape[0]) + 0.5) * CELL
        ys = self.y0 + (np.arange(self.shape[1]) + 0.5) * CELL
        gx, gy = np.meshgrid(xs, ys, indexing='ij')
        for x, y, radius in self.keep_out:
            height[np.hypot(gx - x, gy - y) < radius] = 0.0
        # opening drops specks, growing keeps the lobes of a piece joined by a thin neck together, and what a tall
        # lobe hides from the camera joins it to the next one: a shadow is not the table
        mask = height > MIN_PIECE_HEIGHT
        grown = ndimage.binary_dilation(ndimage.binary_opening(mask, iterations=1), iterations=PIECE_MERGE_CELLS)
        hidden, _ = ndimage.label(~ndimage.binary_closing(self.seen, iterations=1))
        hidden_cells = np.bincount(hidden.ravel())
        hidden_cells[0] = MAX_HIDDEN_CELLS
        self.labels, _ = ndimage.label(grown | (hidden_cells[hidden] < MAX_HIDDEN_CELLS), structure=np.ones((3, 3)))
        self.labels[~mask] = 0
        piece_cells = np.bincount(self.labels.ravel())
        is_piece = piece_cells >= MIN_PIECE_CELLS
        is_piece[0] = False
        self.labels[~is_piece[self.labels]] = 0
        height[self.labels == 0] = 0.0
        self.height = height
        return [self._piece(label) for label in np.flatnonzero(is_piece)]

    def _piece(self, label):
        cells = np.argwhere(self.labels == label)
        xy = (cells + 0.5) * CELL + (self.x0, self.y0)
        length, width, angle = min_area_rectangle(xy)
        c, s = math.cos(angle), math.sin(angle)
        u, v = xy[:, 0] * c + xy[:, 1] * s, -xy[:, 0] * s + xy[:, 1] * c
        mid_u, mid_v = (u.min() + u.max()) / 2, (v.min() + v.max()) / 2
        weights = self.height[cells[:, 0], cells[:, 1]]
        return Piece(int(label), (mid_u * c - mid_v * s, mid_u * s + mid_v * c), float(weights.max()),
                     length, width, angle, tuple(xy.T @ weights / weights.sum()))

    def split(self, piece, library):
        """Pieces that touch are one blob, but in higher and higher slices of it the taller ones come apart from the
        rest. Returns the ones the library knows as pieces of their own, plus the blob if something is left of it."""
        blob = self.labels == piece.label
        for level in np.arange(MIN_PIECE_HEIGHT + SPLIT_STEP, piece.height, SPLIT_STEP):
            slices, count = ndimage.label(ndimage.binary_opening(blob & (self.height > level)), structure=np.ones((3, 3)))
            known = []
            for part in range(1, count + 1):
                cells = slices == part
                if cells.sum() < MIN_PIECE_CELLS:
                    continue
                label = self.labels.max() + 1
                self.labels[cells] = label
                candidate = self._piece(label)
                if library.identify(candidate)[1] >= SPLIT_TOLERANCE:
                    self.labels[cells] = piece.label
                else:
                    known.append(candidate)
            if known:
                taken = ndimage.binary_dilation(blob & (self.labels != piece.label), iterations=PIECE_MERGE_CELLS + 1)
                left = np.count_nonzero(blob & ~taken)
                return known + ([piece] if left >= MIN_PIECE_CELLS else [])
        return []

    def _sample(self, grid, x, y):
        i = np.floor((x - self.x0) / CELL).astype(int)
        j = np.floor((y - self.y0) / CELL).astype(int)
        inside = (i >= 0) & (i < self.shape[0]) & (j >= 0) & (j < self.shape[1])
        values = np.zeros(x.shape, grid.dtype)
        values[inside] = grid[i[inside], j[inside]]
        return values

    def grasp(self, piece, tried=()):
        """Best top-down grasp of a piece, None when no pair of finger positions works. Grasps too close to the tried
        ones are left out."""
        half = int((max(piece.length, piece.width) / 2 + MAX_GRASP_WIDTH / 2 + PAD_THICKNESS + FINGER_MARGIN + 0.01) / CELL)
        pad = int(PAD_WIDTH / 2 / CELL)
        axis = np.arange(-half, half + 1) * CELL
        v_axis, u_axis = np.meshgrid(axis, axis, indexing='ij')
        finger = int(PAD_THICKNESS / CELL)
        margin = int(FINGER_MARGIN / CELL)
        best, best_score = None, 1e9
        for degrees in range(0, 180, YAW_STEP_DEG):
            yaw = math.radians(degrees)
            close = (math.sin(yaw), -math.cos(yaw))
            along = (math.cos(yaw), math.sin(yaw))
            x = piece.center[0] + u_axis * close[0] + v_axis * along[0]
            y = piece.center[1] + u_axis * close[1] + v_axis * along[1]
            height = self._sample(self.height, x, y)
            labels = self._sample(self.labels, x, y)
            for row in range(pad, 2 * half + 1 - pad, 2):
                band = height[row - pad:row + pad + 1]
                band_labels = labels[row - pad:row + pad + 1]
                mine = band_labels == piece.label
                if not mine.any():
                    continue
                top = band[mine].max()
                profile = band.max(axis=0)
                for engage in np.arange(MIN_ENGAGE, MAX_ENGAGE + 1e-6, 0.003):
                    tip = top - engage
                    if tip < TIP_TABLE_CLEARANCE:
                        break
                    solid = np.concatenate(([0], (profile > tip).astype(np.int8), [0]))
                    changes = np.flatnonzero(np.diff(solid))
                    for first, last in zip(changes[::2], changes[1::2] - 1):
                        width = (last - first + 1) * CELL
                        if not MIN_GRASP_WIDTH <= width <= MAX_GRASP_WIDTH:
                            continue
                        slice_labels = band_labels[:, first:last + 1][band[:, first:last + 1] > tip]
                        if not (slice_labels == piece.label).all():
                            continue                # another piece in the way, or not this piece
                        left, right = first - margin - finger, last + margin + finger + 1
                        if left < 0 or right > 2 * half + 1:
                            continue
                        if max(profile[left:first].max(), profile[last + 1:right].max()) > tip - TIP_CLEARANCE:
                            continue                # no room for a finger
                        touching = (band[:, first:first + 3].max(axis=1) > tip) & (band[:, last - 2:last + 1].max(axis=1) > tip)
                        rows = int(touching.sum())
                        if rows < MIN_CONTACT_ROWS:
                            continue
                        middle = (first + last) / 2 - half
                        center_u, center_v = middle * CELL, (row - half) * CELL
                        gx = piece.center[0] + center_u * close[0] + center_v * along[0]
                        gy = piece.center[1] + center_u * close[1] + center_v * along[1]
                        if any(_same_grasp(gx, gy, yaw, other) for other in tried):
                            continue
                        distance = math.hypot(gx - piece.center_of_mass[0], gy - piece.center_of_mass[1])
                        score = distance * 1000 + 40.0 / rows - engage * ENGAGE_WEIGHT
                        if score < best_score:
                            q = finger_position(width)
                            best_score = score
                            best = Grasp(gx, gy, float(tip + finger_tip(q)), yaw, width, float(engage))
        return best
