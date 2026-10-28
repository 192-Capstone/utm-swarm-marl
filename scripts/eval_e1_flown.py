"""
eval_e1_flown.py
----------------
E1 v2: does strategic planning make the flown traffic safer? Evaluation only.

Per trial: random routable flight requests on a D1 city or the E-City crop (n flights wanting to
leave within `window` seconds), planned by E1 (utm/strategic_planner.py). The flights E1 approves
are then flown twice by the learned policy, each drone following its D7 route (environment.
route_follow) with E1's layer profile (vertical climb at the pad, cruise at the layer's height,
vertical landing), taking off at its departure time and landing when settled (environment.
schedule); crashed drones leave the world (crash_mode remove):
    no_e1   everyone leaves at its requested time, in its preferred layer
    e1      departure time and layer as E1 approved them
Same flights, routes, policy and noise seeds in both. Scored: arrived, crashed, close encounters
(pairs of airborne drones that ever came closer than 2.0 m and 1.4 m: E1's separation target
and the policy's reaction distance), closest approach, E1's delays, and how late each drone
settled on its goal pad compared with the planned arrival (late_s; E1 holds the pad pad_after s).
In the e1 condition, E3's conformance monitor (utm/conformance.py) watches every airborne drone
once per second against its booking: alerts by state, and flights with at least one alert.

--zones FRAC adds temporary no-fly zones (D2, utm/airspace_zones.py): 10 m tiles covering about
FRAC of the map, each closed for 20-40 s starting at a random time in the window. E1 plans around
them (waiting or flying around, whichever is cheaper); both conditions are also scored for zone violations (an airborne drone inside an active
zone's (H3 cell, layer) keys): flights that ever entered one, and drone-seconds spent inside.

    python scripts/eval_e1_flown.py --config configs/config_stage2_round7_eval_goalhold.yaml \
        --checkpoint <ckpt> --map city --level medium --flights 20 --window 60 --seeds 20 21

Seeds 10-19 are refused (reserved for test runs).

C5 (PyBullet vs Webots):
    --simulator pybullet (default) | webots_l2_multi (envs/webots_l2_multi.py; Webots log next
                to the output, <out stem>.s<seed>.<mode>.webots.log)
Each (seed, condition) is flown in its OWN subprocess (one Webots connection per Python process,
and the number of drones changes from trial to trial); the trial (map, zones, requests, E1's
plan) is drawn again there from the same seed and checked against this process's (digest).
Besides the summary (<out>), per-flight rows go to <out stem>.rows.json (seed, condition, flight
id, arrived, crashed, zone entered, in a close pass below 2.0 / 1.4 m, late_s; Webots: max tilt
and true speed), for scripts/c5_compare.py.
"""

import argparse
import dataclasses
import hashlib
import json
import math
import os
import subprocess
import sys
import time

import numpy as np
import torch
import yaml

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import eval_suite as es                                           # noqa: E402
from envs.swarm_env import SwarmEnv                                # noqa: E402
from modules.goal_hold import GoalHoldPolicy, goal_hold_enabled    # noqa: E402
from preview_route import load_map                                 # noqa: E402
from preview_strategic import random_requests                      # noqa: E402
from record_rollout import disable_all_collision_physics           # noqa: E402
from utm.conformance import ConformanceMonitor                     # noqa: E402
from utm.airspace_zones import apply_zones, random_block_zones, tile_blocks, zone_keys  # noqa: E402
from utm.strategic_planner import Settings, StrategicPlanner       # noqa: E402

NEAR = (2.0, 1.4)


def plans(sky, reqs):
    """E1 decisions; and for the approved flights, both conditions' (t_dep, 3D waypoints)."""
    decisions = [sky.submit(fp) for fp in reqs]
    flights = []
    for fp, d in zip(reqs, decisions):
        if d.status != 'approved':
            continue
        r = sky.route(fp.start, fp.goal, fp.layer)
        z0, zl = sky.s.start_z, sky.layer_z(fp.layer)
        naive = [(r[0][0], r[0][1], z0)] + [(x, y, zl) for x, y in r] + [(r[-1][0], r[-1][1], z0)]
        flights.append({'id': fp.id, 'start': fp.start, 'goal': fp.goal,
                        'no_e1': (fp.t_earliest, naive), 'e1': (d.t_dep, d.waypoints), 'delay': d.delay,
                        'arrival': {'no_e1': fp.t_earliest + sky.profile(r, fp.layer)[1], 'e1': d.arrival}})
    return decisions, flights


def temporary_zones(rng, bounds, frac, window):
    (x0, x1), (y0, y1) = bounds
    zones = random_block_zones(tile_blocks(bounds, 10.0), (x1 - x0) * (y1 - y0), frac, rng)
    out = []
    for z in zones:
        t0 = float(rng.uniform(0.0, window))
        out.append(dataclasses.replace(z, active=[[t0, t0 + float(rng.uniform(20.0, 40.0))]]))
    return out


def zone_windows(sky, zones):
    """(cell, layer) -> active windows of the zones covering it."""
    out = {}
    for z in zones:
        for key in zone_keys(z, sky.sky, sky.s.n_layers):
            out.setdefault(key, []).extend(tuple(w) for w in z.active)
    return out


def per_flight(flights, mode, reached, crashed, entered, close, settled, stats=None):
    """One row per flight of a condition: its own outcome, for the paired (per-flight) tests.
    close: {threshold: set of (i, j) pairs}; settled: settle time (s) or nan, per drone."""
    rows = []
    for i, f in enumerate(flights):
        r = {'id': f['id'], 't_dep': float(f[mode][0]), 'arrived': bool(reached[i]), 'crashed': bool(crashed[i]),
             'zone_entered': i in entered,
             **{f'close_{t}': any(i in pr for pr in close[t]) for t in NEAR},
             'late_s': None if np.isnan(settled[i]) else round(float(settled[i] - f['arrival'][mode]), 2)}
        if stats is not None:
            r.update(max_tilt_deg=max(stats['max_roll_deg'][i], stats['max_pitch_deg'][i]),
                     max_true_speed=stats['max_speed'][i])
        rows.append(r)
    return rows


def fly(cfg, args, layout, flights, mode, seed, sky=None, windows=None):
    n = len(flights)
    cfg = json.loads(json.dumps(cfg))
    cfg['environment'].update(n_agents=n, crash_mode='remove', schedule={'enabled': True},
                              route_follow={'enabled': True})
    dt = cfg['environment'].get('control_timestep', 0.032)
    last = max(f[mode][0] for f in flights)
    longest = max(sum(math.dist(a[:2], b[:2]) for a, b in zip(f[mode][1], f[mode][1][1:])) for f in flights)
    cfg['environment']['max_steps'] = int((last + 3.0 * longest / 1.0 + 30.0) / dt)
    env = SwarmEnv(cfg, device=torch.device('cpu'))
    disable_all_collision_physics(env, n)
    policy = es.MappoPolicy(cfg, args.checkpoint, torch.device('cpu'), True)
    policy.restore_normalizer(env)
    if goal_hold_enabled(cfg):
        policy = GoalHoldPolicy(policy, cfg)
    np.random.seed(seed)
    torch.manual_seed(seed)
    z0 = 1.0
    spawns = {f'drone_{i}': [*f['start'], z0] for i, f in enumerate(flights)}
    goals = {f'drone_{i}': [*f['goal'], z0] for i, f in enumerate(flights)}
    obs = env.start_episode(spawns, goals, obstacles=layout,
                            routes={f'drone_{i}': f[mode][1] for i, f in enumerate(flights)},
                            departures={f'drone_{i}': f[mode][0] for i, f in enumerate(flights)})
    close = {t: set() for t in NEAR}
    min_sep = np.inf
    entered, inside_steps = set(), 0
    settled = np.full(n, np.nan)
    monitor = ConformanceMonitor(sky) if mode == 'e1' and sky is not None else None
    every = max(1, int(round(1.0 / dt)))
    info = {}
    for step in range(cfg['environment']['max_steps']):
        obs, _, done, info = env.step(policy.act(obs, env, n))
        air = np.flatnonzero(~(env.waiting | env.landed | env.crashed))
        settled[np.isnan(settled) & env.landed] = (step + 1) * dt
        if monitor is not None and (step + 1) % every == 0:
            pos = env.adapter.get_positions()
            monitor.check((step + 1) * dt, {flights[i]['id']: pos[i] for i in air})
        if windows:
            t = (step + 1) * dt
            for i, (x, y, z) in zip(air, env.adapter.get_positions()[air]):
                ivs = windows.get((sky.sky.cell(x, y), sky.layer_of(z)), ())
                if any(lo <= t < hi for lo, hi in ivs):
                    entered.add(int(i))
                    inside_steps += 1
        if len(air) > 1:
            p = env.adapter.get_positions()[air]
            d = np.linalg.norm(p[:, None, :] - p[None, :, :], axis=2)
            iu = np.triu_indices(len(air), 1)
            dd = d[iu]
            min_sep = min(min_sep, float(dd.min()))
            for t in NEAR:
                for k in np.flatnonzero(dd < t):
                    close[t].add((int(air[iu[0][k]]), int(air[iu[1][k]])))
        if done:
            break
    stats = env.adapter.episode_stats() if hasattr(env.adapter, 'episode_stats') else None
    env.close()
    reached = np.asarray(info.get('drone_reached_goal', [False] * n), dtype=bool)
    return {'flights': n, 'arrived': int(reached.sum()), 'crashed': int(env.crashed.sum()),
            'pairs_below_2.0': len(close[2.0]), 'pairs_below_1.4': len(close[1.4]),
            'min_separation_m': None if min_sep == np.inf else round(min_sep, 3),
            'zone_entries': len(entered), 'zone_seconds': round(inside_steps * dt, 2),
            'e3': None if monitor is None else {**monitor.summary(), 'events': monitor.events},
            'late_s': [None if np.isnan(t) else round(float(t - f['arrival'][mode]), 2)
                       for t, f in zip(settled, flights)],
            '_flights': per_flight(flights, mode, reached, env.crashed, entered, close, settled, stats)}


def trial(args, seed):
    """Everything E1 decides for one seed: layout, planner, zones and their windows, requests,
    decisions, approved flights. Deterministic in (args, seed)."""
    args.seed = seed
    layout, bounds = load_map(args)
    settings = Settings() if args.radius is None else dataclasses.replace(Settings(), radius=args.radius)
    sky = StrategicPlanner(layout, bounds, settings)
    zones = (temporary_zones(np.random.RandomState(1000 + seed), bounds, args.zones, args.window)
             if args.zones else [])
    apply_zones(sky, zones)
    windows = zone_windows(sky, zones)
    reqs = random_requests(np.random.RandomState(seed), sky, args.flights, window=args.window)
    decisions, flights = plans(sky, reqs)
    return layout, sky, zones, windows, reqs, decisions, flights


def flights_digest(flights):
    return hashlib.sha1(repr(flights).encode()).hexdigest()[:16]


def fly_subprocess(seed, mode, out_path):
    """Fly one (seed, condition) in a fresh Python process (this script, same arguments)."""
    cmd = [sys.executable, os.path.abspath(__file__), *sys.argv[1:],
           '--worker-seed', str(seed), '--worker-mode', mode, '--worker-out', out_path]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"seed {seed} {mode} worker failed:\n{(p.stderr or p.stdout)[-3000:]}")
    with open(out_path) as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--map', choices=['city', 'osm-crop'], default='city')
    ap.add_argument('--level', default='medium')
    ap.add_argument('--flights', type=int, default=20)
    ap.add_argument('--window', type=float, default=60.0, help='requested departures spread over this many s')
    ap.add_argument('--seeds', type=int, nargs='+', default=[20, 21, 22])
    ap.add_argument('--radius', type=float, default=None,
                    help="E1's safety radius (m; default Settings: 2.1 = 1.0 half-separation + 1.1 drift)")
    ap.add_argument('--hold-engage', type=float, default=None,
                    help='goal-hold engage distance (m) for these runs (config: 0.5); release = +0.2 m. '
                         'E1 landings: drones can stall ~0.6 m from the goal, outside a 0.5 m zone')
    ap.add_argument('--zones', type=float, default=0.0,
                    help='fraction of the map under temporary no-fly zones (0 = none)')
    ap.add_argument('--out', default=None)
    ap.add_argument('--simulator', choices=['pybullet', 'webots_l2_multi'], default='pybullet')
    ap.add_argument('--worker-seed', type=int, default=None, help=argparse.SUPPRESS)
    ap.add_argument('--worker-mode', default=None, help=argparse.SUPPRESS)
    ap.add_argument('--worker-out', default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()
    if any(10 <= s <= 19 for s in args.seeds):
        ap.error('seeds 10-19 are reserved for test runs')
    worker = (args.worker_seed, args.worker_mode, args.worker_out)
    del args.worker_seed, args.worker_mode, args.worker_out
    if worker[0] is not None and 10 <= worker[0] <= 19:
        ap.error('seeds 10-19 are reserved for test runs')
    path = args.out or os.path.join('eval_results', 'e1_flown',
                                    f"{args.map}_{args.level}_f{args.flights}_w{args.window:g}"
                                    f"{f'_z{args.zones:g}' if args.zones else ''}"
                                    f"{'' if args.simulator == 'pybullet' else '_' + args.simulator}.json")
    stem = os.path.splitext(path)[0]

    with open(args.config, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    cfg.pop('scenario_mix', None)
    cfg['environment']['collision_physics'] = False
    if args.hold_engage is not None:                        # this experiment only; config untouched
        cfg.setdefault('goal_hold', {}).update(engage_dist=args.hold_engage,
                                               release_dist=args.hold_engage + 0.2)
    cfg.setdefault('obstacles', {})['enabled'] = False      # the map layout is passed explicitly

    if worker[0] is not None:                               # one (seed, condition), in this fresh process
        seed, mode = worker[0], worker[1]
        if args.simulator != 'pybullet':
            cfg['environment']['simulator'] = args.simulator
            cfg['webots'] = {'log': os.path.abspath(f'{stem}.s{seed}.{mode}.webots.log')}
        layout, sky, zones, windows, reqs, decisions, flights = trial(args, seed)
        r = fly(cfg, args, layout, flights, mode, seed, sky, windows)
        with open(worker[2], 'w') as f:
            json.dump({'digest': flights_digest(flights), 'result': r}, f)
        return

    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    rows, flight_rows = [], []
    for seed in args.seeds:
        layout, sky, zones, windows, reqs, decisions, flights = trial(args, seed)
        row = {'seed': seed, 'requested': len(reqs), 'approved': len(flights), 'zones': len(zones),
               'rejected_by_zone': sum(1 for d in decisions if d.blockers),
               'around_zones': sum(1 for d in decisions if d.around_zones),
               'delay_mean_s': float(np.mean([f['delay'] for f in flights])) if flights else 0.0}
        for mode in ('no_e1', 'e1'):
            t0 = time.time()
            w = fly_subprocess(seed, mode, f'{stem}.s{seed}.{mode}.worker.json')
            if w['digest'] != flights_digest(flights):
                raise RuntimeError(f"seed {seed} {mode}: the worker drew a different E1 plan")
            r = w['result']
            flight_rows += [{'seed': seed, 'mode': mode, **fr} for fr in r.pop('_flights')]
            row[mode] = r
            print(f"seed {seed} {mode:5s}: {r['flights']} flights, arrived {r['arrived']}, crashed {r['crashed']}, "
                  f"pairs < 2.0 m {r['pairs_below_2.0']}, < 1.4 m {r['pairs_below_1.4']}, "
                  f"closest {r['min_separation_m']} m, zone entries {r['zone_entries']} "
                  f"({r['zone_seconds']} s)  ({time.time() - t0:.0f} s)", flush=True)
        rows.append(row)

    tot = {m: {k: sum(r[m][k] for r in rows) for k in ('flights', 'arrived', 'crashed', 'pairs_below_2.0',
                                                         'pairs_below_1.4', 'zone_entries', 'zone_seconds')}
           for m in ('no_e1', 'e1')}
    for m in ('no_e1', 'e1'):
        t = tot[m]
        print(f"TOTAL {m:5s}: {t['flights']} flights, arrived {100 * t['arrived'] / t['flights']:.1f}%, "
              f"crashed {100 * t['crashed'] / t['flights']:.1f}%, pairs < 2.0 m {t['pairs_below_2.0']}, "
              f"< 1.4 m {t['pairs_below_1.4']}, zone entries {t['zone_entries']} ({t['zone_seconds']:.1f} s)")
    print(f"E1: {sum(r['approved'] for r in rows)}/{sum(r['requested'] for r in rows)} approved, "
          f"mean delay {np.mean([r['delay_mean_s'] for r in rows]):.1f} s, "
          f"rejected with a zone in the way {sum(r['rejected_by_zone'] for r in rows)}, "
          f"flown around zones {sum(r['around_zones'] for r in rows)}")
    e3 = [r['e1']['e3'] for r in rows if r['e1'].get('e3')]
    if e3:
        by = {k: sum(x['by_state'][k] for x in e3) for k in e3[0]['by_state']}
        print(f"E3 alerts (e1): {sum(x['alerts'] for x in e3)} on {sum(x['flights_alerted'] for x in e3)} of "
              f"{sum(r['e1']['flights'] for r in rows)} flights; " + ', '.join(f'{k} {v}' for k, v in by.items() if v))
    for m in ('no_e1', 'e1'):
        late = np.array([x for r in rows for x in r[m]['late_s'] if x is not None])
        if len(late):
            print(f"settled vs planned arrival ({m}): median {np.median(late):+.1f} s, "
                  f"p95 {np.percentile(late, 95):+.1f} s, p99 {np.percentile(late, 99):+.1f} s, max {late.max():+.1f} s")
    with open(path, 'w') as f:
        json.dump({'args': vars(args), 'trials': rows, 'total': tot}, f, indent=2)
    with open(f'{stem}.rows.json', 'w') as f:
        json.dump({'args': vars(args), 'rows': flight_rows}, f, indent=1)
    print(f"wrote {path}")


if __name__ == '__main__':
    main()
