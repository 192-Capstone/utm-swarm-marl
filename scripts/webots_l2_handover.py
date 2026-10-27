"""
webots_l2_handover.py
---------------------
Part 4 (A4): a policy on Webots Level 2 (official Crazyflie unless --equivalent), flying all
the way ("policy-only") or handing over to a position hold near the goal ("hand-over").

Episodes are drawn exactly as eval_suite.py draws them for a single-drone obstacle test (per
seed: np.random / torch seeded, a RandomState for the scenario, episodes in order), and each
episode's scenario hash is stored, so variants can be matched episode by episode.

Hand-over (--handover): the first time the drone is within --handover-dist (0.5 m) of its goal,
policy commands stop and the drone is given a position setpoint at the goal
(envs/webots_l2.py set_position_hold: outer loop v = 1.0 * (goal - pos), capped at 0.3 m/s,
altitude setpoint = goal z; Bitcraze's velocity controller has no x/y position mode).

Per step the TRUE state is logged. Per episode: the environment's own outcome (strict rule:
3D distance < 0.5 m and 3D speed < 0.15 m/s for 31 consecutive steps), the same rule with
0.20 m/s, whether it ended within 0.5 m, where a timed-out drone ended, and at the goal:
median speed inside the goal sphere, longest still stretch (< 0.15 m/s inside the sphere),
time from first entering the sphere to settling. Writes one JSON per (variant, test).

--zero-orientation (off by default, Webots only): the policy's 3 orientation inputs [6:9]
(roll, pitch, yaw / pi) are set to 0 before it acts, as in PyBullet, where the
velocity-controlled body never tilts. Tilt statistics (|roll|, |pitch| per step) are logged
for every run.

Usage (repo root): python scripts/webots_l2_handover.py --variant V1 --test pillar \
    --policy mappo --checkpoint <ckpt> [--handover] --seeds 20 21 22 23 24 --episodes 100
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import eval_suite as es                                          # noqa: E402
from envs.swarm_env import SwarmEnv                              # noqa: E402
from baselines.repulsion_controller import RepulsionController  # noqa: E402


def longest_run(mask):
    best = cur = 0
    for m in mask:
        cur = cur + 1 if m else 0
        best = max(best, cur)
    return best


def first_settle(dist, speed, thr_d, thr_v, need):
    """Index of the step at which the settle rule is met (or None)."""
    cur = 0
    for k, (d, v) in enumerate(zip(dist, speed)):
        cur = cur + 1 if (d < thr_d and v < thr_v) else 0
        if cur >= need:
            return k
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', required=True)
    ap.add_argument('--test', required=True, choices=list(es.OBSTACLE_TESTS))
    ap.add_argument('--config', default='configs/config_stage2_round7_teacher.yaml')
    ap.add_argument('--policy', choices=['mappo', 'repulsion'], default='mappo')
    ap.add_argument('--checkpoint', default=None)
    ap.add_argument('--handover', action='store_true')
    ap.add_argument('--handover-dist', type=float, default=0.5)
    ap.add_argument('--zero-orientation', action='store_true',
                    help="force the policy's orientation inputs [6:9] to 0 (default off)")
    ap.add_argument('--seeds', type=int, nargs='+', default=[20, 21, 22, 23, 24])
    ap.add_argument('--episodes', type=int, default=100, help="episodes PER seed (as eval_suite)")
    ap.add_argument('--equivalent', action='store_true', help="my Crazyflie-equivalent model instead of the official one")
    ap.add_argument('--out', default='eval_results/webots_l2_handover')
    a = ap.parse_args()

    cfg = yaml.safe_load(open(a.config))
    cfg.pop('scenario_mix', None)                 # as eval_suite
    cfg['environment']['collision_physics'] = False
    cfg['environment']['n_agents'] = 1
    cfg['environment']['simulator'] = 'webots_l2'
    cfg['webots'] = {'real': not a.equivalent}
    env = SwarmEnv(cfg, device=torch.device('cpu'))
    ad = env.adapter
    if a.policy == 'mappo':
        pol = es.MappoPolicy(cfg, a.checkpoint, torch.device('cpu'), True)
        pol.restore_normalizer(env)
    else:
        pol = RepulsionController(cfg)
    E = cfg['environment']
    max_steps, thr_d, need = E['max_steps'], E.get('goal_threshold', 0.5), E.get('settle_steps', 31)
    thr_v = E.get('settle_speed', 0.15)
    dt = E.get('control_timestep', 0.032)

    rows, t0 = [], time.time()
    for seed in a.seeds:
        np.random.seed(seed); torch.manual_seed(seed)
        rng = np.random.RandomState(seed)
        for ep in range(a.episodes):
            obs = es.setup_obstacle_test(env, rng, a.test)
            sid = es.scenario_hash(env)
            goal = np.array(env.goals[0], dtype=float)
            dist, speed, hand_step, info, tilt = [], [], None, {}, []
            for step in range(max_steps):
                if a.zero_orientation:
                    obs = dict(obs)
                    lf = obs['local_fixed'].clone()
                    lf[:, 6:9] = 0.0
                    obs['local_fixed'] = lf
                act = pol.act(obs, env, 1)
                if a.handover and hand_step is None and np.linalg.norm(ad._pos[0] - goal) < a.handover_dist:
                    hand_step = step
                    ad.set_position_hold(goal)
                obs, _, done, info = env.step(act)          # actions are ignored while holding
                dist.append(float(np.linalg.norm(ad._pos[0] - goal)))
                speed.append(float(np.linalg.norm(ad._vel[0])))
                tilt.append((abs(float(ad._orn[0][0])), abs(float(ad._orn[0][1]))))
                if done:
                    break
            ad.set_position_hold(None)
            outcome = info.get('reason') or 'timeout'
            dist, speed = np.array(dist), np.array(speed)
            inside = dist < thr_d
            crash_k = len(dist) if outcome != 'collision' else len(dist) - 1
            s020 = first_settle(dist[:crash_k], speed[:crash_k], thr_d, 0.20, need)
            s015 = first_settle(dist, speed, thr_d, thr_v, need)
            entry = int(np.argmax(inside)) if inside.any() else None
            fd = float(dist[-1])
            to_cause = None
            if outcome == 'timeout':
                to_cause = 'inside_moving' if fd < thr_d else ('short_0.5_1' if fd < 1.0 else 'stuck_gt1')
            rows.append({'seed': seed, 'episode': ep, 'scenario_id': sid, 'outcome': outcome,
                         'success_strict': outcome == 'success', 'success_020': s020 is not None,
                         'ended_within_0.5': fd < thr_d, 'final_dist': fd, 'steps': len(dist),
                         'timeout_cause': to_cause, 'handover_step': hand_step,
                         'first_entry_step': entry,
                         'median_speed_in_goal': float(np.median(speed[inside])) if inside.any() else None,
                         'longest_still_in_goal': longest_run(inside & (speed < thr_v)),
                         'time_to_settle_s': (s015 - entry + 1) * dt if (s015 is not None and entry is not None) else None,
                         'max_speed': float(speed.max()),
                         'tilt_deg': {'roll_max': float(np.degrees(max(t[0] for t in tilt))),
                                      'pitch_max': float(np.degrees(max(t[1] for t in tilt))),
                                      'roll_p95': float(np.degrees(np.percentile([t[0] for t in tilt], 95))),
                                      'pitch_p95': float(np.degrees(np.percentile([t[1] for t in tilt], 95)))},
                         'tilt_samples_deg': [[round(float(np.degrees(r)), 2), round(float(np.degrees(q)), 2)]
                                              for r, q in tilt[::4]]})
        print(f"{time.strftime('%T')} {a.variant} {a.test} seed {seed}: "
              f"{np.mean([r['success_strict'] for r in rows if r['seed'] == seed]) * 100:.0f}% strict", flush=True)
    env.close()
    os.makedirs(a.out, exist_ok=True)
    out = {'variant': a.variant, 'test': a.test, 'policy': a.policy, 'checkpoint': a.checkpoint,
           'handover': a.handover, 'handover_dist': a.handover_dist, 'zero_orientation': a.zero_orientation,
           'seeds': a.seeds,
           'episodes_per_seed': a.episodes, 'config': a.config,
           'model': 'equivalent' if a.equivalent else 'official Crazyflie', 'wall_s': time.time() - t0,
           'episodes': rows}
    json.dump(out, open(os.path.join(a.out, f'{a.variant}_{a.test}.json'), 'w'))
    s = np.mean([r['success_strict'] for r in rows]) * 100
    print(f"{a.variant} {a.test}: {len(rows)} episodes, strict success {s:.1f}%, wall {time.time() - t0:.0f}s")


if __name__ == '__main__':
    main()
