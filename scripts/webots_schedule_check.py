"""
webots_schedule_check.py
------------------------
Stage-3 demo groundwork: do scheduled departures work with real Crazyflies in Webots
(envs/webots_l2_multi.py park / place)? 4 drones in open space take off at 0, 2, 4 and 6 s,
the shipped policy flies each to its goal 6 m away, and each lands (is parked again) there.

Per drone: when it appeared and how far from its pad (it must appear within one control step of
its departure time, within 5 cm of the pad), whether it reached its goal and landed, and its max
tilt after release. Pass: every drone appears on time on its pad, reaches its goal and lands, no
tilt over 60 deg, and no adapter safety check fires (drop / start / controller log).

--sim pybullet runs the same in PyBullet (no Webots needed: checks the script itself).

Usage (repo root, low priority, nothing else heavy running; Webots needs a display):
  DISPLAY=:2 OMP_NUM_THREADS=4 nice -n 19 python scripts/webots_schedule_check.py \
      --checkpoint checkpoints/stage2_r1_crowd_rl_r7v3ft_s0_lab/checkpoint_final.pth
Writes logs/webots/schedule_check/<sim>.json (and the Webots log next to it).
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

CONFIG = 'configs/config_stage2_round7_eval_goalhold.yaml'
OUT = os.path.join(ROOT, 'logs', 'webots', 'schedule_check')
DEPART = (0.0, 2.0, 4.0, 6.0)
SPAWNS = [(-6.0, -3.0 + 2.0 * k, 1.0) for k in range(4)]
GOALS = [(0.0, -3.0 + 2.0 * k, 1.0) for k in range(4)]
MAX_TILT = 60.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--sim', choices=['webots_l2_multi', 'pybullet'], default='webots_l2_multi')
    args = ap.parse_args()

    import torch
    import yaml
    import eval_suite as es
    from envs.swarm_env import SwarmEnv

    cfg = yaml.safe_load(open(os.path.join(ROOT, CONFIG)))
    cfg.pop('scenario_mix', None)                        # as eval_suite
    n = len(DEPART)
    cfg['environment'].update(n_agents=n, collision_physics=False, crash_mode='remove', simulator=args.sim)
    cfg['environment']['schedule'] = {'enabled': True}
    os.makedirs(OUT, exist_ok=True)
    if args.sim != 'pybullet':
        cfg['webots'] = {'log': os.path.join(OUT, f'{args.sim}.webots.log')}
    env = SwarmEnv(cfg, device=torch.device('cpu'))
    pol = es.MappoPolicy(cfg, args.checkpoint, torch.device('cpu'), True)
    pol.restore_normalizer(env)
    np.random.seed(20); torch.manual_seed(20)

    key = [f'drone_{i}' for i in range(n)]
    obs = env.start_episode(dict(zip(key, map(list, SPAWNS))), dict(zip(key, map(list, GOALS))), obstacles=[],
                            departures=dict(zip(key, DEPART)))
    dt = env.control_dt
    appeared, pad_err, tilt = [None] * n, [None] * n, np.zeros(n)
    info, error, steps = {}, None, 0
    try:
        for step in range(cfg['environment']['max_steps']):
            obs, _, done, info = env.step(pol.act(obs, env, n))
            steps = step + 1
            pos = env.adapter.get_positions()
            for i in np.flatnonzero(~(env.waiting | env.landed | env.crashed)):
                if appeared[i] is None:
                    appeared[i] = steps * dt
                    pad_err[i] = float(np.linalg.norm(pos[i] - SPAWNS[i]))
                orn = getattr(env.adapter, '_orn', None)
                if orn is not None:
                    tilt[i] = max(tilt[i], float(np.degrees(np.abs(orn[i, :2]).max())))
            if done:
                break
    except RuntimeError as e:                            # an adapter safety check fired
        error = str(e)
    env.close()

    reached = list(info.get('drone_reached_goal', [False] * n))
    rows = []
    for i in range(n):
        on_time = appeared[i] is not None and DEPART[i] <= appeared[i] + 1e-9 <= DEPART[i] + 2 * dt
        rows.append({'drone': i, 'departure_s': DEPART[i], 'appeared_s': appeared[i],
                     'pad_error_m': pad_err[i], 'on_time': bool(on_time),
                     'reached_goal': bool(reached[i]), 'landed': bool(env.landed[i]),
                     'max_tilt_deg': float(tilt[i])})
        print(f"drone_{i}: departs {DEPART[i]:.1f} s, appeared {appeared[i]} s "
              f"({pad_err[i] if pad_err[i] is None else round(pad_err[i], 3)} m from its pad), "
              f"reached {reached[i]}, landed {bool(env.landed[i])}, max tilt {tilt[i]:.1f} deg")
    ok = (error is None and all(r['on_time'] and r['pad_error_m'] <= 0.05 and r['reached_goal'] and r['landed']
                                and r['max_tilt_deg'] <= MAX_TILT for r in rows))
    print(f"{'PASS' if ok else 'FAIL'} ({steps} steps, {steps * dt:.1f} s)" + (f": {error}" if error else ''))
    out = os.path.join(OUT, f'{args.sim}.json')
    json.dump({'sim': args.sim, 'pass': ok, 'error': error, 'steps': steps, 'drones': rows}, open(out, 'w'), indent=1)
    print('wrote', out)
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
