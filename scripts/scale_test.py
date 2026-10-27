"""
scale_test.py
-------------
B5 Phase D: does the policy hold up as the swarm grows, and what does a step cost?

Evaluation only, no training, no W&B. For each swarm size N the crowd "swap" test is played
with its area grown so the density stays that of the 8-drone test:
    reach = 10 m * sqrt(N / 8)          (swap places drones in [-reach, reach]^2, 1-5 m up)
    max_steps = config max_steps * sqrt(N / 8)   (flights get longer by the same factor)
Obstacles off, collision physics off (as the suite's crowd tests). Episodes, policy loading,
goal-hold and the success rules come from eval_suite.py, so the numbers mean the same thing.

Crashes: environment.crash_mode = remove (SwarmEnv). A crashed drone is parked out of the world
and the others keep flying, so one crash no longer ends 99 other flights. Both views are reported:
    success          strict, per episode: every drone settled and none crashed
    crash episodes   episodes with at least one crash (the strict safety view)
    arrived          per drone: share of all drone flights that settled at their goal
    crashed          per drone: share of all drone flights that crashed (crashes per 100 flights)

Each (N, neighbour index) is run on the same seeds, so brute / grid / h3 see identical episodes;
the script checks their outcomes are identical and reports the time per step:
    env step      everything in SwarmEnv.step (physics, LiDAR, rewards, observations)
    neighbours    the part spent in ObservationProcessor._build_neighbor_states

    python scripts/scale_test.py --config configs/config_stage2_round7_eval_goalhold.yaml \
        --checkpoint checkpoints/stage2_r1_crowd_rl_r7v3ft_s0/checkpoint_final.pth \
        --sizes 8 16 --index brute grid --seeds 20 --episodes 3
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import yaml

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, ROOT)

import eval_suite as es                                           # noqa: E402
from envs.swarm_env import SwarmEnv                               # noqa: E402
from modules.goal_hold import GoalHoldPolicy, goal_hold_enabled   # noqa: E402
from record_rollout import disable_all_collision_physics, setup_swap_scenario  # noqa: E402


class Timer:
    """Wraps a bound method and adds up the seconds spent in it."""

    def __init__(self, obj, name):
        self.total, self.calls = 0.0, 0
        inner = getattr(obj, name)

        def timed(*a, **k):
            t = time.perf_counter()
            try:
                return inner(*a, **k)
            finally:
                self.total += time.perf_counter() - t
                self.calls += 1
        setattr(obj, name, timed)


def run(base_cfg, args, n, kind):
    cfg = json.loads(json.dumps(base_cfg))
    scale = math.sqrt(n / 8)
    cfg.pop('scenario_mix', None)
    cfg['environment']['n_agents'] = n
    cfg['environment']['collision_physics'] = False
    cfg['environment']['crash_mode'] = 'remove'
    cfg['environment']['max_steps'] = max_steps = int(round(base_cfg['environment']['max_steps'] * scale))
    cfg.setdefault('obstacles', {})['enabled'] = False
    cfg['observation']['neighbor_index'] = kind
    dt = cfg['environment'].get('control_timestep', 0.032)

    env = SwarmEnv(cfg, device=torch.device('cpu'))
    disable_all_collision_physics(env, n)
    policy = es.MappoPolicy(cfg, args.checkpoint, torch.device('cpu'), True)
    policy.restore_normalizer(env)
    if goal_hold_enabled(cfg):
        policy = GoalHoldPolicy(policy, cfg)
    clip_val = env.obs_processor.normalizer.clip if env.obs_processor.schema == 'v1' else float('inf')
    step_t = Timer(env, 'step')
    nb_t = Timer(env.obs_processor, '_build_neighbor_states')

    episodes, agents = [], []
    t0 = time.time()
    for seed in args.seeds:
        np.random.seed(seed)
        torch.manual_seed(seed)
        rng = np.random.RandomState(seed)
        for ep in range(args.episodes):
            obs = setup_swap_scenario(env, rng, n, reach=10.0 * scale)
            e, ags = es.run_episode(env, policy, obs, n, max_steps, dt, False, clip_val)
            episodes.append({'seed': seed, 'episode': ep, **e})
            agents.extend({'seed': seed, 'episode': ep, **a, 'crashed': int(c)}
                          for a, c in zip(ags, env.crashed))
    env.close()
    s = es.summarize(episodes, agents, n, False)
    steps = max(step_t.calls, 1)
    row = {
        'n': n, 'index': kind, 'episodes': s['episodes'], 'max_steps': max_steps,
        'success': s['success_rate'], 'success_ci95': s['success_ci95'],
        'crash_episodes': float(np.mean([x['outcome'] == 'collision' for x in episodes])),
        'timeout': s['timeout_rate'],
        'arrived': float(np.mean([a['settled'] for a in agents])),
        'crashed': float(np.mean([a['crashed'] for a in agents])),
        'ms_per_step': 1000 * step_t.total / steps,
        'ms_neighbours_per_step': 1000 * nb_t.total / steps,
        'steps': step_t.calls, 'wall_s': round(time.time() - t0, 1),
        # what must match across indexes: every episode's outcome, length and final distances
        'fingerprint': [(x['outcome'], x['length_steps']) for x in episodes]
                       + [(a['settled'], a['crashed'], a['final_goal_dist']) for a in agents],
    }
    print(f"  N={n:4d} {kind:5s} success {100 * row['success']:5.1f}%  crash episodes "
          f"{100 * row['crash_episodes']:5.1f}%  | per drone: arrived {100 * row['arrived']:5.1f}% "
          f"crashed {100 * row['crashed']:5.2f}%  | "
          f"step {row['ms_per_step']:7.2f} ms (neighbours {row['ms_neighbours_per_step']:6.2f})  "
          f"{row['wall_s']:.0f}s", flush=True)
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--sizes', type=int, nargs='+', default=[8, 16, 32, 64, 100])
    ap.add_argument('--index', nargs='+', default=['brute', 'grid'], choices=['brute', 'grid', 'h3'])
    ap.add_argument('--seeds', type=int, nargs='+', default=[20, 21, 22, 23, 24])
    ap.add_argument('--episodes', type=int, default=20, help='episodes PER seed')
    ap.add_argument('--out', default=os.path.join('eval_results', 'scale_test', 'scale_test.json'))
    args = ap.parse_args()
    if any(s in range(10, 20) for s in args.seeds):
        ap.error('seeds 10-19 are reserved for the final test run')

    with open(args.config, encoding='utf-8') as f:
        base_cfg = yaml.safe_load(f)
    rows = []
    for n in args.sizes:
        for kind in args.index:
            rows.append(run(base_cfg, args, n, kind))
        same = [r for r in rows if r['n'] == n]
        ok = all(r['fingerprint'] == same[0]['fingerprint'] for r in same)
        print(f"  N={n:4d} identical outcomes across {[r['index'] for r in same]}: {ok}", flush=True)
        for r in same:
            r['identical_to_first_index'] = ok

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump({'config': args.config, 'checkpoint': args.checkpoint, 'seeds': args.seeds,
                   'episodes_per_seed': args.episodes,
                   'rows': [{k: v for k, v in r.items() if k != 'fingerprint'} for r in rows]}, f, indent=2)
    print(f"wrote {args.out}")


if __name__ == '__main__':
    main()
