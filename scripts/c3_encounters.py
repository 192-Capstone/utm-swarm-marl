"""
c3_encounters.py
----------------
C3: how much drone traffic a Lidar run actually had, from its logs
(<root>/routes_<prefix>_<map>.route.lidar.npz, envs/webots_lidar.py ScanLog): per map, drone-steps
(flying drones only), and from the true positions of the flying drones at each control step, the
distance from each drone to its nearest flying neighbour:
    drone-steps with the nearest neighbour < 6 m (Lidar range), < 2 m, < 1 m
    episodes with any pair closer than 2 m / 1 m, the closest pair seen
The number of rays whose analytic ray hit a drone is in scripts/c3_lidar_compare.py's tables.

Usage (repo root): python scripts/c3_encounters.py --root eval_results/c3/dense12 --prefix lidar
Writes <root>/encounters_<prefix>.json and prints it.
"""
import argparse
import json
import os

import numpy as np

THRESH = (6.0, 2.0, 1.0)


def encounters(path):
    z = np.load(path)
    ep, k, drone, parked, all_pos = z['ep'], z['k'], z['drone'], z['parked'], z['all_pos']
    n = all_pos.shape[1]
    # rows come in blocks of n (drone 0 .. n-1) per logged control step
    assert len(ep) % n == 0 and np.all(drone.reshape(-1, n) == np.arange(n))
    P = all_pos.reshape(-1, n, n, 3)[:, 0]                       # (steps, n, 3)
    fly = ~parked.reshape(-1, n)
    E = ep.reshape(-1, n)[:, 0]
    d = np.linalg.norm(P[:, :, None] - P[:, None], axis=-1)
    d[:, np.arange(n), np.arange(n)] = np.inf
    d = np.where(fly[:, :, None] & fly[:, None, :], d, np.inf)
    nearest = d.min(axis=2)[fly]                                  # per flying drone-step
    pair_min = d.min(axis=(1, 2))                                 # per step
    out = {'steps': int(len(P)), 'drone_steps': int(fly.sum()), 'episodes': int(len(np.unique(E))),
           'closest_pair_m': float(pair_min.min())}
    for t in THRESH:
        out[f'drone_steps_nearest_lt_{t:g}m'] = int((nearest < t).sum())
        out[f'drone_steps_nearest_lt_{t:g}m_pct'] = 100.0 * float((nearest < t).mean())
    for t in THRESH[1:]:
        out[f'episodes_any_pair_lt_{t:g}m'] = int(len(np.unique(E[pair_min < t])))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--prefix', default='lidar')
    ap.add_argument('--maps', nargs='+', default=['easy', 'medium', 'hard', 'crop'])
    a = ap.parse_args()
    res = {}
    for m in a.maps:
        p = os.path.join(a.root, f'routes_{a.prefix}_{m}.route.lidar.npz')
        if os.path.exists(p):
            res[m] = encounters(p)
    json.dump(res, open(os.path.join(a.root, f'encounters_{a.prefix}.json'), 'w'), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
