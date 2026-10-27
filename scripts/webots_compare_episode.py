"""
webots_compare_episode.py
-------------------------
Step-by-step comparison of one episode in PyBullet and in Webots (Level 1).

Both simulators get the same scenario (same seed) and are flown in two ways:
  open loop   : the actions computed in PyBullet are sent to both simulators,
                so any difference is the simulators' integration alone
  closed loop : each simulator is flown by its own copy of the policy, so
                differences can grow through the policy's feedback
Per step the raw state (positions, velocities, 14-ray fan, 32-ray ring) is
compared; prints the largest differences and when they happen.

Usage (repo root):
  python scripts/webots_compare_episode.py --scenario pillar --policy repulsion
  python scripts/webots_compare_episode.py --scenario headon --n-agents 3 \
      --policy mappo --checkpoint checkpoints/stage2_imitation_ww_s0/bc_policy.pth
"""
import argparse
import copy
import os
import sys

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import eval_suite as es                                      # noqa: E402
from envs.swarm_env import SwarmEnv                          # noqa: E402
from record_rollout import disable_all_collision_physics     # noqa: E402
from baselines.repulsion_controller import RepulsionController  # noqa: E402


def make_env(cfg, sim):
    c = copy.deepcopy(cfg)
    c['environment']['simulator'] = sim
    env = SwarmEnv(c, device=torch.device('cpu'))
    disable_all_collision_physics(env, c['environment']['n_agents'])
    return env


def setup(env, scenario, seed, n):
    rng = np.random.RandomState(seed)
    np.random.seed(seed)
    if scenario in es.OBSTACLE_TESTS:
        return es.setup_obstacle_test(env, rng, scenario)
    if scenario in es.HELDOUT_TESTS:
        return es.setup_heldout_test(env, rng, scenario)
    return {'headon': es.setup_headon_scenario, 'crossing': es.setup_crossing_scenario,
            'swap': es.setup_swap_scenario}[scenario](env, rng, n)


def make_policy(a, cfg, env):
    if a.policy == 'repulsion':
        return RepulsionController(cfg)
    pol = es.MappoPolicy(cfg, a.checkpoint, torch.device('cpu'), True)
    pol.restore_normalizer(env)
    return pol


def snapshot(env, n):
    raw = env.adapter.get_raw_state()
    return {k: np.array([raw[f'drone_{i}'][k] for i in range(n)]) for k in ('position', 'velocity', 'lidar', 'lidar_ring')}


def run(a, cfg, mode, pb, wb):
    n = cfg['environment']['n_agents']
    obs_pb, obs_wb = setup(pb, a.scenario, a.seed, n), setup(wb, a.scenario, a.seed, n)
    assert pb.layout == wb.layout, "layouts differ"
    pol_pb, pol_wb = make_policy(a, cfg, pb), make_policy(a, cfg, wb)
    rows, done_pb, done_wb, info_pb, info_wb = [], False, False, {}, {}
    for step in range(cfg['environment']['max_steps']):
        act_pb = pol_pb.act(obs_pb, pb, n)
        act_wb = act_pb if mode == 'open' else pol_wb.act(obs_wb, wb, n)
        obs_pb, _, done_pb, info_pb = pb.step(act_pb)
        obs_wb, _, done_wb, info_wb = wb.step(act_wb)
        s_pb, s_wb = snapshot(pb, n), snapshot(wb, n)
        rows.append({k: np.abs(s_pb[k] - s_wb[k]).max() for k in s_pb})
        if done_pb or done_wb:
            break
    out = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    print(f"\n[{mode} loop] {a.scenario}, {n} drone(s), {a.policy}, seed {a.seed}: {len(rows)} steps")
    print(f"  ended: PyBullet step {len(rows)} {info_pb.get('reason') or ('running' if not done_pb else '?')} | "
          f"Webots {info_wb.get('reason') or ('running' if not done_wb else '?')}")
    for k, v in out.items():
        unit = 'm/s' if k == 'velocity' else 'm'
        big = np.where(v > 0.01)[0]
        print(f"  {k:10s} max diff {v.max():.6f} {unit} (step {int(v.argmax()) + 1}), mean {v.mean():.6f}; "
              f"steps with >1 cm: {len(big)}" + (f" (first at {big[0] + 1})" if len(big) else ''))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/config_stage2_ring.yaml')
    ap.add_argument('--scenario', default='pillar')
    ap.add_argument('--n-agents', type=int, default=3)
    ap.add_argument('--policy', choices=['repulsion', 'mappo'], default='repulsion')
    ap.add_argument('--checkpoint', default=None)
    ap.add_argument('--seed', type=int, default=10)
    ap.add_argument('--mode', choices=['open', 'closed', 'both'], default='both')
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    cfg.pop('scenario_mix', None)
    cfg['environment']['collision_physics'] = False
    obstacle = a.scenario in es.OBSTACLE_TESTS or a.scenario in es.HELDOUT_TESTS
    cfg['environment']['n_agents'] = 1 if obstacle else a.n_agents
    # one Webots per process: both modes reuse the same pair of environments
    pb, wb = make_env(cfg, 'pybullet'), make_env(cfg, 'webots')
    for mode in (['open', 'closed'] if a.mode == 'both' else [a.mode]):
        run(a, cfg, mode, pb, wb)
    pb.close(); wb.close()


if __name__ == '__main__':
    main()
