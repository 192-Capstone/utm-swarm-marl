"""
altitude_test.py
----------------
Can the learned policy fly a delivery at a higher cruise height? One drone on open ground (no
buildings), the E1 flight profile: take off at 1 m, climb to the cruise height over the pad, cruise
`--leg` m, descend over the goal pad. Flown with the shipped policy (+ goal-hold) as in the scenario
runs, at each height in --heights. Reports, per height: arrived, climb time, the cruise height held
(median, worst sag and overshoot over the cruise leg), and time to land.

    python scripts/altitude_test.py --heights 2 4 8 14 18
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import yaml

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import eval_suite as es                                            # noqa: E402
from envs.swarm_env import SwarmEnv                                 # noqa: E402
from modules.goal_hold import GoalHoldPolicy, goal_hold_enabled     # noqa: E402
from record_rollout import disable_all_collision_physics           # noqa: E402

CKPT = '../utm-swarm-marl/checkpoints/stage2_r1_crowd_rl_r7v3ft_s0_lab/checkpoint_final.pth'


def fly(cfg, ckpt, height, leg, seed, z_bias=0.0, alt_cap=None, land_step=None):
    cfg = json.loads(json.dumps(cfg))
    cfg['environment'].update(n_agents=1, crash_mode='remove', schedule={'enabled': True, 'land_latch_m': 2.4, 'land_vz_max': 0.5,
                                                                           'cruise_z_bias': z_bias, 'land_step_m': land_step},
                              route_follow={'enabled': True}, max_steps=int((60 + 4 * height + leg / 0.5) / 0.032))
    if alt_cap is not None:
        cfg.setdefault('observation', {})['altitude_cap'] = alt_cap
    env = SwarmEnv(cfg, device=torch.device('cpu'))
    disable_all_collision_physics(env, 1)
    policy = es.MappoPolicy(cfg, ckpt, torch.device('cpu'), True)
    policy.restore_normalizer(env)
    if goal_hold_enabled(cfg):
        policy = GoalHoldPolicy(policy, cfg)
    np.random.seed(seed)
    torch.manual_seed(seed)
    x0, x1 = -leg / 2, leg / 2
    route = [(x0, 0.0, 1.0), (x0, 0.0, height), (x1, 0.0, height), (x1, 0.0, 1.0)]
    obs = env.start_episode({'drone_0': [x0, 0.0, 1.0]}, {'drone_0': [x1, 0.0, 1.0]}, obstacles=[],
                            routes={'drone_0': route}, departures={'drone_0': 0.0})
    zs, xs, info = [], [], {}
    for step in range(cfg['environment']['max_steps']):
        obs, _, done, info = env.step(policy.act(obs, env, 1))
        p = env.adapter.get_positions()[0]
        xs.append(float(p[0])); zs.append(float(p[2]))
        if done:
            break
    env.close()
    t = np.arange(len(zs)) * 0.032
    zs, xs = np.array(zs), np.array(xs)
    up = np.flatnonzero(zs >= height - 0.6)
    cruise = (xs > x0 + 3) & (xs < x1 - 3)
    zc = zs[cruise] if cruise.any() else np.array([np.nan])
    return {'height': height, 'arrived': bool(info.get('drone_reached_goal', [False])[0]), 'crashed': bool(env.crashed[0]),
            'climb_s': round(float(t[up[0]]), 1) if len(up) else None,
            'cruise_median_z': round(float(np.median(zc)), 2), 'cruise_min_z': round(float(np.min(zc)), 2),
            'cruise_max_z': round(float(np.max(zc)), 2), 'max_z': round(float(zs.max()), 2),
            'end_s': round(float(t[-1]), 1), 'end_x': round(float(xs[-1]), 2)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', default='configs/config_stage2_round7_eval_goalhold.yaml')
    ap.add_argument('--checkpoint', default=CKPT)
    ap.add_argument('--heights', type=float, nargs='+', default=[2, 4, 8, 14, 18])
    ap.add_argument('--leg', type=float, default=30.0)
    ap.add_argument('--seed', type=int, default=20)
    ap.add_argument('--alt-cap', type=float, default=None, help="observation.altitude_cap (m): the policy's own height input capped")
    ap.add_argument('--land-step', type=float, default=None, help='SwarmEnv schedule.land_step_m (m): descend in steps')
    ap.add_argument('--z-bias', type=float, default=0.0, help='SwarmEnv schedule.cruise_z_bias (m): aim this much higher')
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config, encoding='utf-8'))
    cfg.pop('scenario_mix', None)
    cfg['environment']['collision_physics'] = False
    cfg.setdefault('obstacles', {})['enabled'] = False
    for h in a.heights:
        r = fly(cfg, a.checkpoint, h, a.leg, a.seed, a.z_bias, a.alt_cap, a.land_step)
        print(json.dumps(r), flush=True)


if __name__ == '__main__':
    main()
