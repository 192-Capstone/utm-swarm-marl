"""
eval_city_routes.py
-------------------
D7: does following a planned route help the learned policy in a city? Evaluation only.

Each episode is a D1 procedural city (or the E-City crop) with n drones on flights between
street points (envs/city_generator.py sample_flights; E-City: free, routable points). The same
episodes are flown twice, by the same policy:
    direct   the policy's goal is the real goal (as in Stage 2)
    route    environment.route_follow: the policy chases a carrot along its D7 route
             (envs/swarm_env.py, envs/route_planner.py RouteFollower.visible_target)
Scored per drone flight: settled at the goal (strict rule), crashed (drone or building),
stuck (timed out more than 1 m from the goal), and for both modes the cross-track error: the
distance from the drone to its planned route, every step (median / 95th percentile / max), the
number E1's safety radius should cover.

    python scripts/eval_city_routes.py --config configs/config_stage2_round7_eval_goalhold.yaml \
        --checkpoint <ckpt> --map city --level medium --drones 4 --seeds 20 --episodes 5

Seeds 10-19 are refused (reserved for test runs). Collision physics off (as the suite).

C5 (PyBullet vs Webots):
    --simulator pybullet (default) | webots_l2_multi (envs/webots_l2_multi.py; Webots log next
                to the output, <out>.<mode>.webots.log)
    --modes     which modes to fly (default: direct route)
Each mode runs in its OWN subprocess (one Webots connection per Python process; the episodes are
drawn again there from the same seeds). Besides the summary (<out>), the per-flight rows go to
<out stem>.<mode>.rows.json (seed, episode, drone, scenario_id, outcome, cross-track numbers;
Webots: max tilt and true speed), for scripts/c5_compare.py.

C3 (Webots Lidar, envs/webots_lidar.py; needs --simulator webots_l2_multi):
    --lidar compare   the drones carry a Lidar, the policy still flies on the analytic scan; both
                      scans are logged to <out stem>.<mode>.lidar.npz
    --lidar policy    the policy flies on the Lidar scan (logged the same way)
Without --lidar nothing changes (the option is not even recorded in the args).
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

import numpy as np
import torch
import yaml

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, ROOT)

import eval_suite as es                                           # noqa: E402
from envs.city_generator import generate_city, sample_flights      # noqa: E402
from envs.route_planner import build_grid, plan, simplify          # noqa: E402
from envs.scenario_manager import DRONE_RADIUS                     # noqa: E402
from envs.swarm_env import SwarmEnv                                # noqa: E402
from modules.goal_hold import GoalHoldPolicy, goal_hold_enabled    # noqa: E402
from record_rollout import disable_all_collision_physics           # noqa: E402

CROP = os.path.join(ROOT, 'data', 'osm_city', 'bengaluru_ecity_phase1_rotated_v2_crop50_cand1_layout.yaml')


def cross_track(xy, route):
    """Distance (m) from a point to a polyline."""
    best = np.inf
    for a, b in zip(route[:-1], route[1:]):
        ab = b - a
        L2 = float(ab @ ab)
        t = 0.0 if L2 < 1e-12 else float(np.clip((xy - a) @ ab / L2, 0.0, 1.0))
        best = min(best, float(np.linalg.norm(a + t * ab - xy)))
    return best


def scenario_id(spawns, goals):
    """Hash of an episode's spawns and goals (rounded to mm), to check the pairing afterwards."""
    keys = sorted(spawns, key=lambda k: int(k.split('_')[1]))
    a = np.round(np.array([spawns[k] for k in keys] + [goals[k] for k in keys], dtype=np.float64), 3)
    return hashlib.sha1(a.tobytes()).hexdigest()[:12]


def make_episode(rng, args):
    """layout, spawns, goals, routes for one episode (routes on the D7 grid, same rules as D1)."""
    if args.map == 'city':
        city = generate_city(rng, args.level)
        layout = city['layout']
        spawns, goals = sample_flights(rng, city, args.drones)
        grid = city['grid']
    else:
        from envs.osm_scenario import load_osm_layout
        layout = load_osm_layout(CROP)
        grid = build_grid(layout, ((-25, 25), (-25, 25)), 0.25, DRONE_RADIUS + 0.35)
        free = np.argwhere(~grid.occ)
        spawns, goals, used = {}, {}, []
        while len(spawns) < args.drones:
            a, b = (np.array(grid.to_world(tuple(free[rng.randint(len(free))]))) for _ in range(2))
            if not 8.0 <= np.linalg.norm(a - b) <= 25.0:
                continue
            if any(np.linalg.norm(p - q) < 2.0 for p in (a, b) for q in used):
                continue
            if plan(grid, a, b) is None:
                continue
            k = f'drone_{len(spawns)}'
            z = float(rng.uniform(2.0, 3.0))
            spawns[k], goals[k] = [*a, z], [*b, z]
            used += [a, b]
    routes = {}
    for k in spawns:
        p = plan(grid, spawns[k][:2], goals[k][:2])
        routes[k] = simplify(grid, p) if p is not None else [tuple(spawns[k][:2]), tuple(goals[k][:2])]
    return layout, spawns, goals, routes


def run(cfg, args, episodes, use_routes):
    cfg = json.loads(json.dumps(cfg))
    cfg['environment']['route_follow'] = {'enabled': use_routes}
    env = SwarmEnv(cfg, device=torch.device('cpu'))
    disable_all_collision_physics(env, args.drones)
    policy = es.MappoPolicy(cfg, args.checkpoint, torch.device('cpu'), True)
    policy.restore_normalizer(env)
    if goal_hold_enabled(cfg):
        policy = GoalHoldPolicy(policy, cfg)
    max_steps = cfg['environment']['max_steps']
    rows = []
    for ep in episodes:
        sid = scenario_id(ep['spawns'], ep['goals'])
        obs = env.start_episode(ep['spawns'], ep['goals'], obstacles=ep['layout'],
                                routes=ep['routes'] if use_routes else None)
        routes = {k: np.asarray(v, dtype=float) for k, v in ep['routes'].items()}
        xt = {k: [] for k in routes}
        info = {}
        for _ in range(max_steps):
            obs, _, done, info = env.step(policy.act(obs, env, args.drones))
            pos = env.adapter.get_positions()
            for i, k in enumerate(routes):
                xt[k].append(cross_track(pos[i, :2], routes[k]))
            if done:
                break
        pos = env.adapter.get_positions()
        crashed_ep = info.get('reason') == 'collision'
        st = env.adapter.episode_stats() if hasattr(env.adapter, 'episode_stats') else None
        for i, k in enumerate(routes):
            settled = bool(info['drone_reached_goal'][i])
            hit = bool(info.get('drone_collision', [False] * args.drones)[i]
                       or info.get('obstacle_collision', [False] * args.drones)[i])
            dist = float(np.linalg.norm(pos[i] - env.goals[i]))
            rows.append({'seed': ep['seed'], 'episode': ep['episode'], 'drone': i, 'settled': settled,
                         'crashed': hit, 'episode_crash': crashed_ep,
                         'stuck': (not settled) and (not crashed_ep) and dist > 1.0, 'final_dist': dist,
                         'xt_med': float(np.median(xt[k])), 'xt_p95': float(np.percentile(xt[k], 95)),
                         'xt_max': float(np.max(xt[k])), 'xt_all': xt[k], 'scenario_id': sid,
                         'reason': info.get('reason')})
            if st is not None:                     # Webots: lost-control numbers (eval_suite.lost_control)
                rows[-1].update(max_tilt_deg=max(st['max_roll_deg'][i], st['max_pitch_deg'][i]),
                                max_true_speed=st['max_speed'][i])
    env.close()
    return rows


def summary(rows):
    xt = np.concatenate([r['xt_all'] for r in rows])
    n = len(rows)
    return {'flights': n,
            'settled_pct': 100.0 * sum(r['settled'] for r in rows) / n,
            'crashed_pct': 100.0 * sum(r['crashed'] for r in rows) / n,
            'stuck_pct': 100.0 * sum(r['stuck'] for r in rows) / n,
            'episodes_with_crash_pct': 100.0 * np.mean([r['episode_crash'] for r in rows]),
            'cross_track_m': {'median': float(np.median(xt)), 'p95': float(np.percentile(xt, 95)),
                              'p99': float(np.percentile(xt, 99)), 'max': float(xt.max())}}


def draw_episodes(args):
    episodes = []
    for seed in args.seeds:
        rng = np.random.RandomState(seed)
        for ep in range(args.episodes):
            layout, spawns, goals, routes = make_episode(rng, args)
            episodes.append({'seed': seed, 'episode': ep, 'layout': layout, 'spawns': spawns,
                             'goals': goals, 'routes': routes})
    return episodes


def run_mode_subprocess(mode, rows_path):
    """Fly one mode in a fresh Python process (this script, same arguments, --worker-mode);
    returns its rows."""
    cmd = [sys.executable, os.path.abspath(__file__), *sys.argv[1:],
           '--worker-mode', mode, '--worker-out', rows_path]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"{mode} worker failed:\n{(p.stderr or p.stdout)[-3000:]}")
    with open(rows_path) as f:
        return json.load(f)['rows']


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--map', choices=['city', 'osm-crop'], default='city')
    ap.add_argument('--level', default='medium')
    ap.add_argument('--drones', type=int, default=4)
    ap.add_argument('--seeds', type=int, nargs='+', default=[20, 21, 22, 23, 24])
    ap.add_argument('--episodes', type=int, default=20, help='episodes PER seed')
    ap.add_argument('--max-steps', type=int, default=1500)
    ap.add_argument('--out', default=None)
    ap.add_argument('--simulator', choices=['pybullet', 'webots_l2_multi'], default='pybullet')
    ap.add_argument('--modes', nargs='+', choices=['direct', 'route'], default=['direct', 'route'])
    ap.add_argument('--lidar', choices=['compare', 'policy'], default=argparse.SUPPRESS,
                    help='C3: Webots Lidar on the drones (default: off)')
    ap.add_argument('--worker-mode', default=None, help=argparse.SUPPRESS)
    ap.add_argument('--worker-out', default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()
    if any(10 <= s <= 19 for s in args.seeds):
        ap.error('seeds 10-19 are reserved for test runs')
    if hasattr(args, 'lidar') and args.simulator != 'webots_l2_multi':
        ap.error('--lidar needs --simulator webots_l2_multi')
    worker = (args.worker_mode, args.worker_out)
    del args.worker_mode, args.worker_out
    path = args.out or os.path.join('eval_results', 'city_routes',
                                    f"{args.map}_{args.level}_n{args.drones}"
                                    f"{'' if args.simulator == 'pybullet' else '_' + args.simulator}.json")

    with open(args.config, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    cfg.pop('scenario_mix', None)
    cfg['environment'].update(n_agents=args.drones, collision_physics=False, max_steps=args.max_steps)
    cfg.setdefault('obstacles', {})['enabled'] = False      # the city layout is passed explicitly
    stem = os.path.splitext(path)[0]

    if worker[0]:                                           # one mode, in this (fresh) process
        mode = worker[0]
        if args.simulator != 'pybullet':
            cfg['environment']['simulator'] = args.simulator
            cfg['webots'] = {'log': os.path.abspath(f'{stem}.{mode}.webots.log')}
            if hasattr(args, 'lidar'):
                cfg['webots']['lidar'] = {'enabled': True, 'use_for_policy': args.lidar == 'policy',
                                          'log': os.path.abspath(f'{stem}.{mode}.lidar.npz')}
        episodes = draw_episodes(args)
        np.random.seed(args.seeds[0]); torch.manual_seed(args.seeds[0])
        rows = run(cfg, args, episodes, mode == 'route')
        with open(worker[1], 'w') as f:
            json.dump({'args': vars(args), 'mode': mode, 'rows': rows}, f)
        return

    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    out = {'args': vars(args)}
    for mode in ('direct', 'route'):
        if mode not in args.modes:
            continue
        t0 = time.time()
        rows_path = f'{stem}.{mode}.rows.json'
        rows = run_mode_subprocess(mode, rows_path)
        s = summary(rows)
        with open(rows_path, 'w') as f:                     # keep the rows, without the per-step track
            json.dump({'args': vars(args), 'mode': mode,
                       'rows': [{k: v for k, v in r.items() if k != 'xt_all'} for r in rows]}, f, indent=1)
        out[mode] = s
        xt = s['cross_track_m']
        print(f"{mode:6s} {args.map}/{args.level} {args.drones} drones, {s['flights']} flights: "
              f"settled {s['settled_pct']:5.1f}%  crashed {s['crashed_pct']:4.1f}%  stuck {s['stuck_pct']:4.1f}%  "
              f"| cross-track median {xt['median']:.2f} m, p95 {xt['p95']:.2f}, p99 {xt['p99']:.2f}, "
              f"max {xt['max']:.2f}  ({time.time() - t0:.0f} s)", flush=True)
    with open(path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"wrote {path}")


if __name__ == '__main__':
    main()
