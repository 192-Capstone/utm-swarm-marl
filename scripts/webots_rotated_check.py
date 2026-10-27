"""
webots_rotated_check.py
-----------------------
Do rotated boxes (layout key 'yaw') build correctly in Webots? Builds two rotated boxes and a
pillar in the Level-1 world (plain shapes) and in the realistic world (buildings), reads back
what Webots created (translation, rotation, size / footprint corners) and compares it with
the layout, and saves a top-down picture of each (needs a display).

Usage (repo root): python scripts/webots_rotated_check.py --out logs/webots/rotated
"""
import argparse
import math
import os
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.webots_adapter import WebotsAdapter     # noqa: E402

LAYOUT = [{'type': 'box', 'center': (2.0, 1.0), 'half': (1.5, 0.4), 'height': 8.0, 'yaw': math.radians(30)},
          {'type': 'box', 'center': (-2.0, -1.5), 'half': (0.8, 0.5), 'height': 8.0, 'yaw': math.radians(-60)},
          {'type': 'pillar', 'center': (-1.5, 2.5), 'radius': 0.5, 'height': 8.0}]


def expected_corners(ob):
    hx, hy = ob['half']
    c, s = math.cos(ob.get('yaw', 0.0)), math.sin(ob.get('yaw', 0.0))
    return np.array([[c * x - s * y, s * x + c * y] for x, y in ((hx, hy), (hx, -hy), (-hx, -hy), (-hx, hy))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/config_stage2_ring.yaml')
    ap.add_argument('--real', action='store_true')
    ap.add_argument('--out', default='logs/webots/rotated')
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    cfg['environment']['n_agents'] = 1
    cfg['webots'] = {'render': True, 'real': a.real}
    ad = WebotsAdapter(cfg)
    ad.reset_world({'drone_0': [0.0, 0.0, 2.0]}, {'drone_0': [4.0, 0.0, 2.0]}, LAYOUT)
    marks = ad.sup.getRoot().getField('children')
    for (x, y), col in (((3.5, 0.0), '1 0 0'), ((0.0, 3.5), '0 0 1')):     # red = +x, blue = +y
        marks.importMFNodeFromString(-1, f'Solid {{ translation {x} {y} 0.1 children [ Shape {{ appearance '
                                         f'PBRAppearance {{ baseColor {col} metalness 0 }} geometry Sphere {{ radius 0.25 }} }} ] }}')
    ad.sup.step(ad.basic_ms)
    root = ad._obstacle_root
    worst = 0.0
    for k, ob in enumerate(LAYOUT):
        nd = root.getMFNode(k)
        tr = nd.getField('translation').getSFVec3f()
        line = f"obstacle {k} ({ob['type']}, yaw {math.degrees(ob.get('yaw', 0.0)):+.0f} deg): translation {np.round(tr, 3).tolist()}"
        err = abs(tr[0] - ob['center'][0]) + abs(tr[1] - ob['center'][1])
        if ob['type'] == 'box':
            if a.real:      # SimpleBuilding: footprint corners relative to its translation
                cf = nd.getField('corners')
                got = np.array([cf.getMFVec2f(i) for i in range(cf.getCount())])
                e = np.abs(got - expected_corners(ob)).max()
                line += f" | corners max error {e:.2e} m"
            else:           # plain Solid: rotation field + Box size
                rot = nd.getField('rotation').getSFRotation()
                yaw = rot[3] * np.sign(rot[2] or 1)
                size = nd.getField('children').getMFNode(0).getField('geometry').getSFNode().getField('size').getSFVec3f()
                e = max(abs(math.remainder(yaw - ob['yaw'], 2 * math.pi)),
                        abs(size[0] - 2 * ob['half'][0]), abs(size[1] - 2 * ob['half'][1]))
                line += f" | rotation {np.round(rot, 4).tolist()} size {np.round(size, 3).tolist()} | max error {e:.2e}"
            err = max(err, e)
        worst = max(worst, err)
        print(line)
    print(f"worst error vs layout: {worst:.2e}")
    # top-down picture
    vp = ad.sup.getFromDef('VIEW') or next(ad.sup.getRoot().getField('children').getMFNode(i)
                                          for i in range(ad.sup.getRoot().getField('children').getCount())
                                          if ad.sup.getRoot().getField('children').getMFNode(i).getTypeName() == 'Viewpoint')
    vp.getField('position').setSFVec3f([0.0, 0.0, 200.0])
    # look straight down; Webots camera looks along its local +x, so pitch it by +90 deg,
    # then yaw so that world +x is to the right and +y up in the picture
    vp.getField('orientation').setSFRotation([-0.5773503, 0.5773503, 0.5773503, 2.0943951])
    # near-orthographic: far away with a narrow field of view (~9 m across at the ground)
    vp.getField('fieldOfView').setSFFloat(2 * math.atan(4.5 * 16 / 9 / 200.0))
    vp.getField('far').setSFFloat(400.0)
    for _ in range(5):
        ad.sup.step(ad.basic_ms)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    img = os.path.abspath(a.out + ('_real' if a.real else '_l1') + '.jpg')
    ad.sup.exportImage(img, 90)
    ad.sup.step(ad.basic_ms)
    print('image:', img)
    ad.close()


if __name__ == '__main__':
    main()
