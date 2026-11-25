"""
vary_osm_heights.py
-------------------
A copy of the E-City layout with varied building heights (map name osm-heights). OpenStreetMap gives almost no
heights here (113 of 115 buildings are the importer's 12 m default), so these are ASSUMED: every building (the
boxes that touch) gets 2-4 floors of 3 m, and `--towers` larger ones, spread out, get 6-8 floors, so high
flights go over most roofs and around the towers. Footprints are unchanged; so is the osm-full map.

    python scripts/vary_osm_heights.py
"""

import argparse
import math
import os

import numpy as np
import yaml

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
SRC = os.path.join(ROOT, 'data', 'osm_city', 'bengaluru_ecity_phase1_rotated_v2_layout.yaml')
OUT = os.path.join(ROOT, 'data', 'osm_city', 'bengaluru_ecity_phase1_heights_layout.yaml')
FLOOR_M = 3.0


def points(b):
    c, s = math.cos(b.get('yaw', 0.0)), math.sin(b.get('yaw', 0.0))
    return [(b['center'][0] + u * b['half'][0] * c - v * b['half'][1] * s, b['center'][1] + u * b['half'][0] * s + v * b['half'][1] * c)
            for u in (-1, 0, 1) for v in (-1, 0, 1)]


def inside(b, p, margin=0.4):
    dx, dy = p[0] - b['center'][0], p[1] - b['center'][1]
    c, s = math.cos(b.get('yaw', 0.0)), math.sin(b.get('yaw', 0.0))
    return abs(dx * c + dy * s) < b['half'][0] + margin and abs(-dx * s + dy * c) < b['half'][1] + margin


def buildings(boxes):
    """Lists of box indices: boxes that touch are one building."""
    up = list(range(len(boxes)))

    def root(i):
        while up[i] != i:
            i = up[i]
        return i
    for i, a in enumerate(boxes):
        pts = points(a)
        for j, b in enumerate(boxes):
            if i != j and math.dist(a['center'], b['center']) < 150 and any(inside(b, p) for p in pts):
                up[root(i)] = root(j)
    out = {}
    for i in range(len(boxes)):
        out.setdefault(root(i), []).append(i)
    return list(out.values())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--towers', type=int, default=8)
    ap.add_argument('--seed', type=int, default=20)
    a = ap.parse_args()
    with open(SRC, encoding='utf-8') as f:
        boxes = yaml.safe_load(f)
    rng = np.random.default_rng(a.seed)
    groups = buildings(boxes)
    info = []
    for g in groups:
        area = sum(4 * boxes[i]['half'][0] * boxes[i]['half'][1] for i in g)
        info.append((area, tuple(np.mean([boxes[i]['center'] for i in g], axis=0)), g))
    floors = {k: int(rng.integers(2, 5)) for k in range(len(info))}              # 2-4 floors: 6, 9 or 12 m
    towers = []
    for k in sorted(range(len(info)), key=lambda k: -info[k][0]):                # towers: 350-1800 m2, 110 m apart
        if len(towers) < a.towers and 350 < info[k][0] < 1800 and all(math.dist(info[k][1], info[t][1]) > 110 for t in towers):
            towers.append(k)
            floors[k] = int(rng.integers(6, 9))
    for k, (_, _, g) in enumerate(info):
        for i in g:
            boxes[i]['height'] = floors[k] * FLOOR_M
    with open(OUT, 'w', encoding='utf-8', newline='\n') as f:
        f.write('# Footprints (c) OpenStreetMap contributors (ODbL). Heights are ASSUMED: scripts/vary_osm_heights.py.\n')
        yaml.safe_dump(boxes, f, sort_keys=False)
    n = np.bincount(list(floors.values()))
    print(f'wrote {os.path.relpath(OUT, ROOT)}: {len(info)} buildings, floors: ' + ', '.join(f'{k}: {v}' for k, v in enumerate(n) if v))
    for t in towers:
        print(f'  tower {floors[t] * FLOOR_M:.0f} m at ({info[t][1][0]:.0f}, {info[t][1][1]:.0f}), {info[t][0]:.0f} m2')


if __name__ == '__main__':
    main()
