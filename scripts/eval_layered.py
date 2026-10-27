"""
eval_layered.py
---------------
E1 v3, layered safety: is the learned policy still needed when E1 has planned the traffic, and
is E1 still needed when the policy can avoid? Evaluation only.

The same flights (random routable requests, as in eval_e1_flown.py) are flown in a 2 x 2:
    planning   no_e1    everyone leaves at its requested time, preferred layer
               e1       departure time and layer as E1 approved them
    pilot      learned  the policy as shipped (sees other drones: neighbour observations + LiDAR)
               blind    the same policy, blind to other drones (neighbour slots masked, LiDAR
                        ignores drones): it still flies its route and avoids buildings (caveat:
                        all-masked neighbours never occurred in training, so it may fly worse
                        in general, not only unaware of others)
               simple   no learning: a proportional controller towards the route's carrot
                        (the same carrot the policy follows), speed capped at the cruise speed;
                        it sees nothing, other drones or buildings
with the plan broken on purpose, the same way in all four (seeded):
    late       a share of flights takes off late (uniform in a range of seconds)
    slow       a share of flights flies slower (velocity command x factor)
    intruders  extra drones with no flight plan cross the area (E1 never hears of them)
Scored: arrived, crashed, close pairs (< 2.0 / < 1.4 m), closest approach; with E1, E3's
conformance alerts (utm/conformance.py) and, for each close pair, how long before it the first
alert of either drone came, counting only alerts in the --horizon seconds before (warning time;
an older alert is not a warning for it); and the other way round, the share
of alerts followed by a close pass of that drone within --horizon seconds (alert precision).

    python scripts/eval_layered.py --config configs/config_stage2_round7_eval_goalhold.yaml \
        --checkpoint <ckpt> --map city --flights 20 --window 60 --seeds 20 21 22

Seeds 10-19 are refused (reserved for test runs).
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
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import eval_suite as es                                           # noqa: E402
from eval_e1_flown import NEAR, plans                              # noqa: E402
from envs.swarm_env import SwarmEnv                                # noqa: E402
from modules.goal_hold import GoalHoldPolicy, goal_hold_enabled    # noqa: E402
from preview_route import load_map                                 # noqa: E402
from preview_strategic import random_requests                      # noqa: E402
from record_rollout import disable_all_collision_physics           # noqa: E402
from utm.conformance import ConformanceMonitor                     # noqa: E402
from utm.strategic_planner import StrategicPlanner                 # noqa: E402


class SimplePilot:
    """Flies straight at the route's carrot (env.adapter.goals), no sensing, no learning."""

    def __init__(self, kp=1.5, v_max=1.5):
        self.kp, self.v_max = kp, v_max

    def act(self, obs, env, n):
        pos = env.adapter.get_positions()
        out = {}
        for i in range(n):
            v = self.kp * (np.asarray(env.adapter.goals[f'drone_{i}'], dtype=float) - pos[i])
            sp = float(np.linalg.norm(v))
            if sp > self.v_max:
                v *= self.v_max / sp
            out[f'drone_{i}'] = [*v.tolist(), 0.0]
        return out


def disruptions(rng, flights, args):
    """Per flight: extra take-off delay (s) and speed factor; the same in all four conditions."""
    late = [float(rng.uniform(*args.late)) if rng.uniform() < args.p_late else 0.0 for _ in flights]
    slow = [args.slow if rng.uniform() < args.p_slow else 1.0 for _ in flights]
    return late, slow


def intruders(rng, sky, n, window):
    """Unplanned drones: routable random trips at random times, flown at layer 0's height."""
    out = []
    for k, fp in enumerate(random_requests(rng, sky, n, window=window)):
        r = sky.route(fp.start, fp.goal, 0)
        z0, zl = sky.s.start_z, sky.layer_z(0)
        wps = [(r[0][0], r[0][1], z0)] + [(x, y, zl) for x, y in r] + [(r[-1][0], r[-1][1], z0)]
        out.append({'id': f'intruder_{k}', 'start': fp.start, 'goal': fp.goal, 't': fp.t_earliest, 'route': wps})
    return out


def fly(cfg, args, layout, sky, flights, extra, late, slow, mode, pilot, seed):
    n_f = len(flights)
    n = n_f + len(extra)
    cfg = json.loads(json.dumps(cfg))
    cfg['environment'].update(n_agents=n, crash_mode='remove', schedule={'enabled': True},
                              route_follow={'enabled': True})
    if pilot == 'blind':
        cfg.setdefault('lidar', {})['sees_drones'] = False
    dt = cfg['environment'].get('control_timestep', 0.032)
    starts = [f['start'] for f in flights] + [x['start'] for x in extra]
    goals = [f['goal'] for f in flights] + [x['goal'] for x in extra]
    routes = [f[mode][1] for f in flights] + [x['route'] for x in extra]
    deps = [f[mode][0] + late[i] for i, f in enumerate(flights)] + [x['t'] for x in extra]
    factor = np.array(slow + [1.0] * len(extra))
    longest = max(sum(math.dist(a[:2], b[:2]) for a, b in zip(r, r[1:])) for r in routes)
    cfg['environment']['max_steps'] = int((max(deps) + 3.0 * longest / factor.min() + 30.0) / dt)
    env = SwarmEnv(cfg, device=torch.device('cpu'))
    disable_all_collision_physics(env, n)
    if pilot == 'simple':
        policy = SimplePilot(v_max=StrategicPlanner([], ((-1, 1), (-1, 1))).s.v_cruise)
    else:
        policy = es.MappoPolicy(cfg, args.checkpoint, torch.device('cpu'), True, blind_neighbors=(pilot == 'blind'))
        policy.restore_normalizer(env)
        if goal_hold_enabled(cfg):
            policy = GoalHoldPolicy(policy, cfg)
    np.random.seed(seed)
    torch.manual_seed(seed)
    z0 = sky.s.start_z
    key = [f'drone_{i}' for i in range(n)]
    obs = env.start_episode({k: [*s, z0] for k, s in zip(key, starts)}, {k: [*g, z0] for k, g in zip(key, goals)},
                            obstacles=layout, routes=dict(zip(key, routes)), departures=dict(zip(key, deps)))
    ids = [f['id'] for f in flights] + [x['id'] for x in extra]
    monitor = ConformanceMonitor(sky) if mode == 'e1' else None
    every = max(1, int(round(1.0 / dt)))
    close = {t: {} for t in NEAR}                 # pair -> first time below the threshold
    min_sep = np.inf
    info = {}
    for step in range(cfg['environment']['max_steps']):
        acts = policy.act(obs, env, n)
        for i in np.flatnonzero(factor < 1.0):
            a = acts[key[i]]
            acts[key[i]] = [*(np.asarray(a[:3]) * factor[i]).tolist(), *list(a[3:])]
        obs, _, done, info = env.step(acts)
        t = (step + 1) * dt
        air = np.flatnonzero(~(env.waiting | env.landed | env.crashed))
        pos = env.adapter.get_positions()
        if monitor is not None and (step + 1) % every == 0:
            monitor.check(t, {ids[i]: pos[i] for i in air})
        if len(air) > 1:
            p = pos[air]
            d = np.linalg.norm(p[:, None, :] - p[None, :, :], axis=2)
            iu = np.triu_indices(len(air), 1)
            dd = d[iu]
            min_sep = min(min_sep, float(dd.min()))
            for th in NEAR:
                for k in np.flatnonzero(dd < th):
                    close[th].setdefault((int(air[iu[0][k]]), int(air[iu[1][k]])), t)
        if done:
            break
    env.close()
    reached = np.asarray(info.get('drone_reached_goal', [False] * n), dtype=bool)
    out = {'flights': n_f, 'intruders': len(extra), 'arrived': int(reached[:n_f].sum()),
           'crashed': int(env.crashed[:n_f].sum()), 'crashed_all': int(env.crashed.sum()),
           'pairs_below_2.0': len(close[2.0]), 'pairs_below_1.4': len(close[1.4]),
           'pairs_below_1.4_with_intruder': sum(1 for a, b in close[1.4] if a >= n_f or b >= n_f),
           'min_separation_m': None if min_sep == np.inf else round(min_sep, 3)}
    if monitor is not None:
        warn = []
        for (a, b), t_close in close[1.4].items():
            first = [e['t'] for e in monitor.events
                     if e['flight'] in (ids[a], ids[b]) and e['state'] != 'ok'
                     and t_close - args.horizon <= e['t'] <= t_close]
            warn.append(None if not first else round(t_close - min(first), 2))
        close_t = {}                                  # flight id -> times it was in a close pair
        for (a, b), t_close in close[1.4].items():
            close_t.setdefault(ids[a], []).append(t_close)
            close_t.setdefault(ids[b], []).append(t_close)
        alerts = [e for e in monitor.events if e['state'] != 'ok']
        hit = [any(e['t'] <= tc <= e['t'] + args.horizon for tc in close_t.get(e['flight'], ())) for e in alerts]
        out['e3'] = {**monitor.summary(), 'events': monitor.events}
        out['close_pair_warning_s'] = warn
        out['alerts_followed_by_close_pass'] = int(sum(hit))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--map', choices=['city', 'osm-crop'], default='city')
    ap.add_argument('--level', default='medium')
    ap.add_argument('--flights', type=int, default=20)
    ap.add_argument('--window', type=float, default=60.0)
    ap.add_argument('--seeds', type=int, nargs='+', default=[20, 21, 22])
    ap.add_argument('--p-late', type=float, default=0.25)
    ap.add_argument('--late', type=float, nargs=2, default=[5.0, 15.0], help='take-off delay range (s)')
    ap.add_argument('--p-slow', type=float, default=0.25)
    ap.add_argument('--slow', type=float, default=0.6, help='speed factor of slow drones')
    ap.add_argument('--intruders', type=int, default=2)
    ap.add_argument('--modes', nargs='+', default=['no_e1', 'e1'], choices=['no_e1', 'e1'])
    ap.add_argument('--pilots', nargs='+', default=['simple', 'blind', 'learned'], choices=['simple', 'blind', 'learned'])
    ap.add_argument('--horizon', type=float, default=20.0, help='alert precision: close pass within this many s')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    if any(10 <= s <= 19 for s in args.seeds):
        ap.error('seeds 10-19 are reserved for test runs')

    with open(args.config, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    cfg.pop('scenario_mix', None)
    cfg['environment']['collision_physics'] = False
    cfg.setdefault('obstacles', {})['enabled'] = False

    rows = []
    for seed in args.seeds:
        args.seed = seed
        layout, bounds = load_map(args)
        sky = StrategicPlanner(layout, bounds)
        reqs = random_requests(np.random.RandomState(seed), sky, args.flights, window=args.window)
        decisions, flights = plans(sky, reqs)
        rng = np.random.RandomState(5000 + seed)
        late, slow = disruptions(rng, flights, args)
        extra = intruders(rng, sky, args.intruders, args.window)
        row = {'seed': seed, 'approved': len(flights), 'late': sum(x > 0 for x in late),
               'slow': sum(x < 1 for x in slow), 'intruders': len(extra)}
        for mode in args.modes:
            for pilot in args.pilots:
                t0 = time.time()
                r = fly(cfg, args, layout, sky, flights, extra, late, slow, mode, pilot, seed)
                row[f'{mode}/{pilot}'] = r
                print(f"seed {seed} {mode:5s} {pilot:7s}: arrived {r['arrived']}/{r['flights']}, crashed {r['crashed']} "
                      f"(+{r['crashed_all'] - r['crashed']} intruders), pairs < 1.4 m {r['pairs_below_1.4']} "
                      f"({r['pairs_below_1.4_with_intruder']} with an intruder), closest {r['min_separation_m']} m"
                      + (f", E3 alerts {r['e3']['alerts']}" if 'e3' in r else '') + f"  ({time.time() - t0:.0f} s)",
                      flush=True)
        rows.append(row)

    print('\nTOTAL (flights, intruders excluded from arrived/crashed)')
    for mode in args.modes:
        for pilot in args.pilots:
            R = [r[f'{mode}/{pilot}'] for r in rows]
            fl = sum(x['flights'] for x in R)
            line = (f"{mode:5s} {pilot:7s}: arrived {100 * sum(x['arrived'] for x in R) / fl:5.1f}%, "
                    f"crashed {100 * sum(x['crashed'] for x in R) / fl:4.1f}%, "
                    f"pairs < 1.4 m {sum(x['pairs_below_1.4'] for x in R)} "
                    f"({sum(x['pairs_below_1.4_with_intruder'] for x in R)} with an intruder)")
            if mode == 'e1':
                w = [v for x in R for v in x['close_pair_warning_s']]
                seen = [v for v in w if v is not None]
                n_al = sum(x['e3']['alerts'] for x in R)
                line += (f"; E3 alerts {n_al} ({sum(x['alerts_followed_by_close_pass'] for x in R)} followed by a "
                         f"close pass within {args.horizon:g} s), close pairs alerted before "
                         f"{len(seen)}/{len(w)}" + (f" (median {np.median(seen):.1f} s ahead)" if seen else ''))
            print(line)
    path = args.out or os.path.join('eval_results', 'e1_layered',
                                    f"{args.map}_{args.level}_f{args.flights}_w{args.window:g}.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump({'args': vars(args), 'trials': rows}, f, indent=1, default=str)
    print(f"wrote {path}")


if __name__ == '__main__':
    main()
