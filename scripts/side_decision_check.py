"""
side_decision_check.py
----------------------
How does a policy fail on the single-obstacle tests? For each failed flight,
was the drone undecided about which side to pass on, or did it commit to a
side and still fail?

Per flight (one drone, one obstacle, eval_suite's pillar / building /
wide_building / long_wall setups, its own seeds), while the obstacle is
"ahead" (the straight line to the goal passes within 0.5 m of the surface and
the surface is within 4 m):
    flips   : how often the sideways command changes sign (only commands
              above 0.15 m/s count, so small wobble is ignored)
    hover   : steps flying slower than 0.2 m/s while more than 1 m from the
              goal (stopping in front of the obstacle)
and the outcome. A failed flight is classed as
    undecided : >= 2 flips, or >= 60 hover steps (~2 s)
    committed : otherwise (picked a side, kept it, still crashed / timed out)

If seeds of one recipe differ because the side choice is ambiguous to the
student (regression averages "left" and "right" into "straight on"), the weak
seeds' failures should be mostly undecided.

Usage (from the repo root):
    python scripts/side_decision_check.py --checkpoint <ckpt> [--name X] [--episodes 100]
"""
import argparse
import os
import sys

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import eval_suite as es                                    # noqa: E402
from envs.swarm_env import SwarmEnv                        # noqa: E402
from envs.scenario_manager import surface_distance         # noqa: E402


def run(checkpoint, config, episodes, seed):
    cfg = yaml.safe_load(open(config))
    cfg['environment']['collision_physics'] = False
    cfg['environment']['n_agents'] = 1
    env = SwarmEnv(cfg, device=torch.device('cpu'))
    pol = es.MappoPolicy(cfg, checkpoint, torch.device('cpu'), True)
    rows = []
    for kind in es.OBSTACLE_TESTS:
        rng = np.random.RandomState(seed)
        np.random.seed(seed)
        for _ in range(episodes):
            obs = es.setup_obstacle_test(env, rng, kind)
            ob = env.layout[0]
            flips, hover, last = 0, 0, 0
            for _ in range(cfg['environment']['max_steps']):
                p = env.adapter.get_positions()[0]
                g = env.goals[0]
                a = pol.act(obs, env, 1)
                v = np.array(a['drone_0'][:3])
                d_goal = float(np.linalg.norm(g - p))
                u = (g - p)[:2] / max(np.linalg.norm((g - p)[:2]), 1e-6)
                right = np.array([u[1], -u[0]])
                line = [surface_distance(ob, p[:2] + s * (g - p)[:2]) for s in np.linspace(0, 1, 30)]
                ahead = min(line) < 0.5 and surface_distance(ob, p[:2]) < 4.0
                if ahead:
                    lat = float(v[:2] @ right)
                    if abs(lat) > 0.15:
                        s = int(np.sign(lat))
                        flips += last != 0 and s != last
                        last = s
                    if np.linalg.norm(v) < 0.2 and d_goal > 1.0:
                        hover += 1
                obs, _, done, info = env.step(a)
                if done:
                    break
            reason = info.get('reason') or 'timeout'
            rows.append((kind, reason, flips, hover))
    env.close()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--name', default=None)
    ap.add_argument('--config', default='configs/config_stage2_ring.yaml')
    ap.add_argument('--episodes', type=int, default=100)
    ap.add_argument('--seed', type=int, default=30)
    a = ap.parse_args()
    rows = run(a.checkpoint, a.config, a.episodes, a.seed)
    name = a.name or a.checkpoint
    fails = [r for r in rows if r[1] != 'success']
    und = [r for r in fails if r[2] >= 2 or r[3] >= 60]
    print(f"\n{name}: {len(rows)} flights, success {100 * (1 - len(fails) / len(rows)):.1f}%")
    print(f"  failed flights: {len(fails)} = {sum(r[1] == 'collision' for r in fails)} crashes + "
          f"{sum(r[1] == 'timeout' for r in fails)} timeouts")
    if fails:
        print(f"  undecided (>=2 side flips or >=2 s hovering): {len(und)} = {100 * len(und) / len(fails):.0f}% of failures"
              f"   | committed but failed: {100 * (1 - len(und) / len(fails)):.0f}%")
    ok = [r for r in rows if r[1] == 'success']
    print(f"  mean side flips: failed {np.mean([r[2] for r in fails]) if fails else 0:.2f}, "
          f"successful {np.mean([r[2] for r in ok]) if ok else 0:.2f}")
    for kind in es.OBSTACLE_TESTS:
        k = [r for r in rows if r[0] == kind]
        kf = [r for r in k if r[1] != 'success']
        ku = [r for r in kf if r[2] >= 2 or r[3] >= 60]
        print(f"    {kind:14s} success {100 * (1 - len(kf) / len(k)):5.1f}%   failures undecided "
              f"{(100 * len(ku) / len(kf)) if kf else 0:5.0f}%")


if __name__ == '__main__':
    main()
