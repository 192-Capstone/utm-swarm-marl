"""
webots_l2_hover.py
------------------
Hover jitter of the Level-2 Crazyflie (Part 3 / C2): how still is a quadrotor that
"holds a point"? Gives a drone-based reference for the success rule's "stopped"
(speed below 0.15 m/s for 31 control steps).

The drone starts away from the hold point, is put in position hold (envs/webots_l2.py
set_position_hold: Bitcraze's velocity PID with an outer loop v = 1.0 * (goal - pos),
capped at 0.3 m/s, altitude setpoint = goal z, run every 4 ms), flies there, and once it
has reached the point (within 5 cm and below 0.05 m/s for 1 s, or after 20 s at most)
its TRUE velocity (Supervisor) is logged every control step (32 ms) for --seconds.

Reports speed (3D, as the settle check uses, and horizontal) median / 95th percentile /
max, and the share of 1-s windows (31 control steps, every start step) whose max speed
is below 0.15 / 0.20 / 0.25 m/s. Writes a JSON next to --out.

Usage (repo root): python scripts/webots_l2_hover.py [--real] [--seconds 60] --out logs/webots/hover
"""
import argparse
import json
import os
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.webots_l2 import WebotsL2Adapter     # noqa: E402

WIN = 31                                       # control steps = the settle check's window


def window_share(speed, thr):
    m = np.array([speed[i:i + WIN].max() for i in range(len(speed) - WIN + 1)])
    return float(np.mean(m < thr))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/config_stage2_ring.yaml')
    ap.add_argument('--real', action='store_true', help="Bitcraze's official Crazyflie (swarm_l2_real.wbt)")
    ap.add_argument('--hold', type=float, nargs=3, default=[0.0, 0.0, 2.0])
    ap.add_argument('--start', type=float, nargs=3, default=[-1.0, 0.5, 1.5])
    ap.add_argument('--seconds', type=float, default=60.0)
    ap.add_argument('--out', default='logs/webots/hover')
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    cfg['environment']['n_agents'] = 1
    if a.real:
        cfg['webots'] = {'real': True}
    ad = WebotsL2Adapter(cfg)
    goal = np.array(a.hold)
    ad.reset_world({'drone_0': list(a.start)}, {'drone_0': list(a.hold)})
    ad.set_position_hold(goal)
    dt = ad.control_dt
    # fly to the point and wait until it is reached and calm
    calm, t_reach = 0, None
    for k in range(int(20.0 / dt)):
        ad.step_simulation()
        near = np.linalg.norm(ad._pos[0] - goal) < 0.05 and np.linalg.norm(ad._vel[0]) < 0.05
        calm = calm + 1 if near else 0
        if calm >= WIN:
            t_reach = (k + 1) * dt
            break
    # log for --seconds
    pos, vel = [], []
    for _ in range(int(round(a.seconds / dt))):
        ad.step_simulation()
        pos.append(ad._pos[0].copy()); vel.append(ad._vel[0].copy())
    ad.close()
    pos, vel = np.array(pos), np.array(vel)
    sp3 = np.linalg.norm(vel, axis=1)
    sph = np.linalg.norm(vel[:, :2], axis=1)
    res = {'model': 'official Crazyflie' if a.real else 'Crazyflie-equivalent',
           'hold': a.hold, 'start': a.start, 'reached_after_s': t_reach,
           'seconds_logged': len(sp3) * dt, 'control_step_s': dt,
           'noise_or_wind': 'none: the Level-2 world models no sensor noise and no wind',
           'speed_3d': {'median': float(np.median(sp3)), 'p95': float(np.percentile(sp3, 95)), 'max': float(sp3.max())},
           'speed_horizontal': {'median': float(np.median(sph)), 'p95': float(np.percentile(sph, 95)), 'max': float(sph.max())},
           'windows_1s_below': {str(t): window_share(sp3, t) for t in (0.15, 0.20, 0.25)},
           'position_error_m': {'median': float(np.median(np.linalg.norm(pos - goal, axis=1))),
                                'max': float(np.linalg.norm(pos - goal, axis=1).max())}}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out + ('_official' if a.real else '_equivalent') + '.json', 'w'), indent=2)
    print(json.dumps(res, indent=2))


if __name__ == '__main__':
    main()
