"""
webots_lidar_check.py
---------------------
Does the analytic LiDAR of envs/webots_adapter.py (Level 1) read exactly what
PyBullet's ray casts read? No Webots needed: both scans are computed for the
same drone positions and obstacle layouts.

Random layouts from the training obstacle generator, drones placed anywhere,
including right next to obstacle surfaces (the close passes where the ray-start
rules matter). Reports the share of readings that differ by more than 1 cm and
the largest difference, for the 14-ray fan and the 32-ray ring.

Usage (repo root):  python scripts/webots_lidar_check.py [--layouts 200] [--drones 8]
"""
import argparse
import os
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.pybullet_adapter import PyBulletAdapter          # noqa: E402
from envs.webots_adapter import scan_lidar                 # noqa: E402
from envs import scenario_manager                          # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/config_stage2_ring.yaml')
    ap.add_argument('--layouts', type=int, default=200)
    ap.add_argument('--drones', type=int, default=8)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--rotated', action='store_true', help="give every box a random yaw")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    cfg['environment']['n_agents'] = a.drones
    cfg['environment']['collision_physics'] = False
    rng = np.random.RandomState(a.seed)
    pb = PyBulletAdapter(cfg)
    ocfg = scenario_manager.obstacle_config(cfg)
    n_fan = pb.n_lidar
    diffs_fan, diffs_ring, worst = [], [], (0.0, None)
    for k in range(a.layouts):
        spawns = rng.uniform(-6, 6, size=(a.drones, 3)); spawns[:, 2] = rng.uniform(0.5, 5, a.drones)
        goals = rng.uniform(-6, 6, size=(a.drones, 3)); goals[:, 2] = rng.uniform(0.5, 5, a.drones)
        layout = scenario_manager.generate_layout(
            rng, {f'drone_{i}': spawns[i].tolist() for i in range(a.drones)},
            {f'drone_{i}': goals[i].tolist() for i in range(a.drones)}, ocfg)
        if a.rotated:
            for ob in layout:
                if ob['type'] == 'box':
                    ob['yaw'] = float(rng.uniform(-np.pi, np.pi))
        if not layout:   # fall back to hand-made shapes
            layout = [{'type': 'pillar', 'center': (rng.uniform(-5, 5), rng.uniform(-5, 5)),
                       'radius': rng.uniform(0.3, 0.8), 'height': 8.0},
                      {'type': 'box', 'center': (rng.uniform(-5, 5), rng.uniform(-5, 5)),
                       'half': (rng.uniform(0.4, 1.2), rng.uniform(0.4, 1.2)), 'height': 8.0}]
        # half the drones hug an obstacle surface (0.16-0.5 m from it)
        for i in range(a.drones // 2):
            ob = layout[rng.randint(len(layout))]
            ang = rng.uniform(0, 2 * np.pi)
            u = np.array([np.cos(ang), np.sin(ang)])
            reach = ob['radius'] if ob['type'] == 'pillar' else float(np.max(np.abs(np.array(ob['half']) / np.maximum(np.abs(u), 1e-9)).min()))
            spawns[i, :2] = np.array(ob['center']) + u * (reach + rng.uniform(0.16, 0.5))
        sp = {f'drone_{i}': spawns[i].tolist() for i in range(a.drones)}
        gl = {f'drone_{i}': goals[i].tolist() for i in range(a.drones)}
        pb.reset_world(sp, gl, layout)
        pos = pb.get_positions().astype(np.float64)
        fan_pb, ring_pb = pb._scan_lidar(list(pos))
        headings = [pb._heading[f'drone_{i}'] for i in range(a.drones)]
        fan_an, ring_an = scan_lidar(pos, np.zeros(a.drones), headings, layout, pb.lidar_cfg, n_fan)
        df = np.abs(np.array(fan_pb) - np.array(fan_an))
        dr = np.abs(np.array(ring_pb) - np.array(ring_an))
        diffs_fan.append(df.ravel()); diffs_ring.append(dr.ravel())
        m = max(df.max(), dr.max())
        if m > worst[0]:
            worst = (m, k)
    pb.close()
    df = np.concatenate(diffs_fan); dr = np.concatenate(diffs_ring)
    print(f"{a.layouts} layouts x {a.drones} drones: {df.size} fan + {dr.size} ring readings")
    for name, d in (('fan (14)', df), ('ring (32)', dr)):
        print(f"  {name:10s} differ >1 cm: {100 * np.mean(d > 0.01):.3f}%   >5 cm: {100 * np.mean(d > 0.05):.3f}%   "
              f"max {d.max():.4f} m   mean {d.mean():.5f} m")
    print(f"  worst layout #{worst[1]}: {worst[0]:.4f} m")


if __name__ == '__main__':
    main()
