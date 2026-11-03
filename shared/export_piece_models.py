#!/usr/bin/env python3
"""
Export the CAD of the pieces to shared/piece_models.json, the library pick_and_place.py recognizes pieces with.
For every part number found in the .usd files next to this script: surface samples, center of mass and volume.

Reading USD needs pxr, which is only in the Isaac Sim container, so run it there (standard library only):
  docker exec isaac-sim bash -c 'U=$(ls -d /isaac-sim/extscache/omni.usd.libs-*); \
    PYTHONPATH=$U LD_LIBRARY_PATH=$U/bin /isaac-sim/kit/python/bin/python3 /workspace/export_piece_models.py'
"""

import json
import random
import re
from pathlib import Path

from pxr import Usd, UsdGeom

SAMPLES = 4000
HERE = Path(__file__).resolve().parent


def triangles(path):
    """World triangles of every mesh of a USD file, in meters."""
    stage = Usd.Stage.Open(str(path))
    scale = UsdGeom.GetStageMetersPerUnit(stage)
    out = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        points = [matrix.Transform(p) * scale for p in mesh.GetPointsAttr().Get()]
        indices = mesh.GetFaceVertexIndicesAttr().Get()
        start = 0
        for count in mesh.GetFaceVertexCountsAttr().Get():
            face = [points[i] for i in indices[start:start + count]]
            start += count
            out.extend((face[0], face[i], face[i + 1]) for i in range(1, count - 1))
    return out


def export(path):
    tris = triangles(path)
    areas, volume, moment = [], 0.0, [0.0, 0.0, 0.0]
    for a, b, c in tris:
        cross = (b - a) ^ (c - a)
        areas.append(cross.GetLength() / 2)
        signed = (a * (b ^ c)) / 6                  # signed volume of the tetrahedron with the origin
        volume += signed
        for i in range(3):
            moment[i] += signed * (a[i] + b[i] + c[i]) / 4
    samples = []
    for a, b, c in random.choices(tris, weights=areas, k=SAMPLES):
        r1, r2 = random.random(), random.random()
        if r1 + r2 > 1:
            r1, r2 = 1 - r1, 1 - r2
        p = a + (b - a) * r1 + (c - a) * r2
        samples.append([round(p[0], 5), round(p[1], 5), round(p[2], 5)])
    return {'file': str(path.relative_to(HERE)), 'volume': abs(volume),
            'center_of_mass': [m / volume for m in moment], 'points': samples}


def main():
    random.seed(0)
    models = {}
    for path in sorted(HERE.rglob('*.usd')):
        match = re.search(r'STP_(\d{9,})_', path.name)
        if match and match.group(1) not in models:
            models[match.group(1)] = export(path)
            print(match.group(1), '{:.1f} cm3'.format(models[match.group(1)]['volume'] * 1e6), path.name)
    (HERE / 'piece_models.json').write_text(json.dumps(models))
    print('wrote', HERE / 'piece_models.json')


if __name__ == '__main__':
    main()
