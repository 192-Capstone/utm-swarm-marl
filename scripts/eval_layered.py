"""
eval_layered.py
---------------
E1 v3, layered safety: is the learned policy still needed when E1 has planned the traffic, and
is E1 still needed when the policy can avoid? Evaluation only.

The same flights (random routable requests, as in eval_e1_flown.py) are flown in a 2 x 2:
    planning   no_e1    everyone leaves at its requested time, preferred layer
               e1       departure time and layer as E1 approved them
               e4       as e1, plus E4 (utm/contingency.py) reacting in flight to E3's alerts,
                        intruders and the mid-flight zone: re-planned drones get their new route
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
    zone       with --zone-at T, a 6 x 6 m no-fly zone appears at T: where it cuts the most
               planned routes (--zone-place busiest, the worst case) or on a random tile that
               cuts at least one (random); permanent from T, or active for --zone-for s only;
               every condition is scored for drones inside it while it is active
Scored: arrived, crashed, close pairs (< 2.0 / < 1.4 m), closest approach; with E1, E3's
conformance alerts (utm/conformance.py) and, for each close pair, how long before it the first
alert of either drone came, counting only alerts in the --horizon seconds before (warning time;
an older alert is not a warning for it); and the other way round, the share
of alerts followed by a close pass of that drone within --horizon seconds (alert precision); the
same two for the high-risk alerts only (what E4 acts on).

Webots (C5): --simulator webots_l2_multi flies one run (one seed, mode and pilot) per command;
--land-latch-m / --land-vz-max: C5's landing options (both off by default; scenario runs use
2.4 / 0.5, scripts/run_scenario.py).

E2 (intent): --risk intent rates E3's alerts by where the drones are going (each drone shares its
planned next 10 s along its route, utm/intent.py) instead of by the bookings; --telemetry N D L
makes the UTM see the drones only through their reports (utm/telemetry.py: GPS noise N m, delay
D s, loss L) instead of their true positions. Arrived / crashed / close pairs / zone entries are
always scored on the true positions.

    python scripts/eval_layered.py --config configs/config_stage2_round7_eval_goalhold.yaml \
        --checkpoint <ckpt> --map city --flights 20 --window 60 --seeds 20 21 22

Seeds 10-19 are refused (reserved for test runs).
"""

import argparse
import copy
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
from envs.swarm_env import LAND_RADIUS_M, SwarmEnv                 # noqa: E402
from modules.goal_hold import GoalHoldPolicy, goal_hold_enabled    # noqa: E402
from preview_route import load_map                                 # noqa: E402
from preview_strategic import random_requests                      # noqa: E402
from record_rollout import disable_all_collision_physics           # noqa: E402
from utm.airspace_zones import Zone, tile_blocks, zone_keys        # noqa: E402
from utm.conformance import ConformanceMonitor                     # noqa: E402
from utm.contingency import ContingencyManager                     # noqa: E402
from utm.intent import predicted_path, route_intent                # noqa: E402
from utm.strategic_planner import FlightPlan, StrategicPlanner     # noqa: E402
from utm.telemetry import TelemetryModel, silent                   # noqa: E402


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


def place_zone(sky, flights, bounds, place='busiest', rng=None, active=None, size=6.0):
    """A size x size m mid-flight zone: the tile whose zone keys the most approved flights have
    booked (busiest), or a random tile among those at least one has booked (random). active:
    [[t0, t1]] for a temporary zone, None for permanent. Returns (zone, flights it cuts)."""
    tiles = []
    for x0, x1, y0, y1 in tile_blocks(bounds, size):
        z = Zone('midflight', 'Mid-flight closure', polygon=[(x0, y0), (x1, y0), (x1, y1), (x0, y1)],
                 active=active or [])
        keys = zone_keys(z, sky.sky, sky.s.n_layers)
        n = sum(1 for f in flights if any(k in keys for k in sky.current[f['id']].reservation))
        tiles.append((n, z))
    if place == 'busiest':
        return max(tiles, key=lambda x: x[0])[::-1]
    cut = [x for x in tiles if x[0] > 0]
    return cut[rng.randint(len(cut))][::-1]


def fly(cfg, args, layout, sky, flights, extra, late, slow, mode, pilot, seed, zone=None, rec=None, zones=None):
    """One run. rec: an optional utm.flight_log.FlightLog that records it for the dashboard (E6).
    zones: [(t, Zone), ...] appearing mid-flight (E6 scenarios); zone (at args.zone_at) is one of them.
    extra: unplanned drones (intruders: id, start, goal, t, route); an entry with 'emergency': True
    is an emergency flight instead (E6): its drone waits on its pad until t, then it is filed with
    'priority' (E4 / E1 give it the air it needs) and takes off on the approved route."""
    zones = list(zones or []) + ([(args.zone_at, zone)] if zone is not None else [])
    plan = 'no_e1' if mode == 'no_e1' else 'e1'
    n_f = len(flights)
    n = n_f + len(extra)
    cfg = json.loads(json.dumps(cfg))
    sch = {'enabled': True}
    for k in ('land_latch_m', 'land_vz_max'):               # C5 landing options (absent = off)
        if getattr(args, k, None) is not None:
            sch[k] = getattr(args, k)
    cfg['environment'].update(n_agents=n, crash_mode='remove', schedule=sch, route_follow={'enabled': True})
    if getattr(args, 'simulator', 'pybullet') != 'pybullet':   # C5: one Webots connection per process
        cfg['environment']['simulator'] = args.simulator
    if pilot == 'blind':
        cfg.setdefault('lidar', {})['sees_drones'] = False
    dt = cfg['environment'].get('control_timestep', 0.032)
    starts = [f['start'] for f in flights] + [x['start'] for x in extra]
    goals = [f['goal'] for f in flights] + [x['goal'] for x in extra]
    routes = [f[plan][1] for f in flights] + [x['route'] for x in extra]
    deps = [f[plan][0] + late[i] for i, f in enumerate(flights)] + [np.inf if x.get('emergency') else x['t'] for x in extra]
    calls = {n_f + k: x for k, x in enumerate(extra) if x.get('emergency')}     # drone index -> emergency
    if calls and mode == 'no_e1':
        raise ValueError('emergency flights need E1 (mode e1 or e4)')
    factor = np.array(slow + [1.0] * len(extra))
    longest = max(sum(math.dist(a[:2], b[:2]) for a, b in zip(r, r[1:])) for r in routes)
    last = max([d for d in deps if np.isfinite(d)] + [x['t'] for x in calls.values()])
    extra_s = float(getattr(args, 'extra_time', 0.0))       # E6: room for E4's delays (zones waited out)
    # an emergency's route is in `longest`; E1 may still delay its take-off by up to max_delay
    wait_em = sky.s.max_delay if calls else 0.0
    cfg['environment']['max_steps'] = int((last + wait_em + 3.0 * longest / factor.min() + 30.0 + extra_s) / dt)
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
    sky = copy.deepcopy(sky)                      # E4 changes bookings: each run starts fresh
    monitor = ConformanceMonitor(sky, risk=args.risk) if mode in ('e1', 'e4') else None
    tm = None                                     # E2: the UTM hears reports, not the truth
    if monitor is not None and (args.telemetry is not None or args.risk == 'intent'):
        tm = TelemetryModel(*(args.telemetry or (0.0, 0.0, 0.0)), seed=9000 + seed)
    prev = {}                                     # drone -> true position a report period ago
    every_frame = None

    def status():                                 # per drone, utm/flight_log.py codes
        return np.where(env.cancelled, 4, np.where(env.crashed, 3, np.where(env.landed, 2, np.where(env.waiting, 0, 1))))

    if rec is not None:
        every_frame = max(1, int(round(rec.frame_dt / dt)))
        rec.start(sky, layout, dt, [
            {'id': ids[i], 'kind': 'flight' if i < n_f else ('emergency' if i in calls else 'intruder'),
             'operator': flights[i].get('operator', '') if i < n_f else '',
             'start': starts[i], 'goal': goals[i], 't_dep': deps[i] if np.isfinite(deps[i]) else None,
             'route': [] if i in calls else routes[i],
             'arrival': flights[i]['arrival'][plan] if i < n_f else None,
             'layer': sky.layer_of(routes[i][1][2]) if len(routes[i]) > 2 else 0} for i in range(n)],
            [z for _, z in zones])
    e4 = ContingencyManager(sky, monitor) if mode == 'e4' else None
    zkeys = [set(zone_keys(z, sky.sky, sky.s.n_layers)) for _, z in zones]
    zone_on, entered, inside_s, replans = [False] * len(zones), set(), 0.0, []
    called = {}                                   # emergency id -> {t, status, t_dep}
    every = max(1, int(round(1.0 / dt)))
    close = {t: {} for t in NEAR}                 # pair -> first time below the threshold
    min_sep = np.inf
    info = {}
    was_landed, was_crashed = env.landed.copy(), env.crashed.copy()
    for step in range(cfg['environment']['max_steps']):
        acts = policy.act(obs, env, n)
        for i in np.flatnonzero(factor < 1.0):
            a = acts[key[i]]
            acts[key[i]] = [*(np.asarray(a[:3]) * factor[i]).tolist(), *list(a[3:])]
        obs, _, done, info = env.step(acts)
        t = (step + 1) * dt
        air = np.flatnonzero(~(env.waiting | env.landed | env.crashed))
        pos = env.adapter.get_positions()
        if rec is not None and (step + 1) % every_frame == 0:
            rec.frame(t, pos, status())
        if rec is not None:                       # arrivals and crashes as events (the E5 dashboard)
            for i in np.flatnonzero(env.landed & ~was_landed):
                rec.event(t, 'arrived', flight=ids[i])
            for i in np.flatnonzero(env.crashed & ~was_crashed):
                rec.event(t, 'crashed', flight=ids[i])
            was_landed, was_crashed = env.landed.copy(), env.crashed.copy()
        if (step + 1) % every == 0:
            here, paths = {ids[i]: pos[i] for i in air}, None
            if tm is not None:
                states = {}
                for i in air:
                    vel = (pos[i] - prev[i]) / (every * dt) if i in prev else np.zeros(3)
                    f = env.followers.get(key[i]) if env.followers and i < n_f else None
                    cruising = (f is not None and env._climbed.get(key[i], False) and not env._landing[i]
                                and np.hypot(*(pos[i][:2] - env.goals[i][:2])) >= LAND_RADIUS_M)
                    states[ids[i]] = (pos[i], vel, route_intent(f, pos[i], np.hypot(*vel[:2]), t) if cruising else [])
                tm.send(t, states)
                tm.receive(t)
                for fid in silent(tm.latest, t):  # no report for LINK_TIMEOUT_S: landed, crashed or
                    tm.forget(fid)                #   link lost (never the simulator's truth)
                here = {fid: r.pos for fid, r in tm.latest.items()}
                if args.risk == 'intent':
                    paths = {fid: predicted_path(r, t) for fid, r in tm.latest.items()}
            prev = {i: pos[i].copy() for i in air}
            acts_e4 = []
            for k, (t_on, z) in enumerate(zones):
                if not zone_on[k] and t >= t_on:
                    zone_on[k] = True
                    if rec is not None:
                        rec.zone_on(t, z.id)
                    if e4 is not None:
                        acts_e4 += e4.add_zone(z, t, here)
            for i, x in calls.items():
                if ids[i] not in called and t >= x['t']:
                    fp = FlightPlan(ids[i], tuple(x['start']), tuple(x['goal']), 0, t, int(x.get('priority', 1)), 'emergency')
                    got = (e4.emergency(fp, t, here) if e4 is not None      # first: the emergency itself
                           else [{'t': t, 'flight': fp.id, 'why': 'emergency', 'decision': sky.submit(fp, now=t)}])
                    acts_e4 += got
                    d = got[0]['decision']
                    called[fp.id] = {'t': round(t, 2), 'status': d.status, 't_dep': d.t_dep}
                    if d.status == 'approved' and monitor is not None:
                        monitor.plans[fp.id] = d          # E3 watches it like any planned flight
                    elif d.status != 'approved':          # refused: it never takes off, the run doesn't wait
                        env.cancel(i)
            if e4 is not None:
                acts_e4 += e4.step(t, here, paths)
            elif monitor is not None:
                monitor.check(t, here, paths)
            if rec is not None:
                rec.tick(t, monitor, paths)
            for a in acts_e4:
                d = a['decision']
                if rec is not None:
                    rec.replan(a['t'], a['flight'], a['why'], d)
                replans.append({'t': a['t'], 'flight': a['flight'], 'why': a['why'], 'status': d.status, 'reason': d.reason})
                if a['flight'] not in ids:
                    continue
                i = ids.index(a['flight'])
                if d.status == 'approved':
                    env.set_route(i, d.waypoints, departure=d.t_dep)
                elif a['why'].endswith('-> cancelled') and env.waiting[i]:
                    env.cancel(i)
            live = set().union(*[zk for (_, z), zk, on in zip(zones, zkeys, zone_on)
                                 if on and (z.permanent or any(a <= t < b for a, b in z.active))]) if zones else set()
            if live:
                for i in air:
                    if (sky.sky.cell(pos[i][0], pos[i][1]), sky.layer_of(pos[i][2])) in live:
                        entered.add(ids[i])
                        inside_s += 1.0
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
    if rec is not None and (not rec.log['frames']['t'] or rec.log['frames']['t'][-1] != round(t, 3)):
        rec.frame(t, env.adapter.get_positions(), status())   # the final state, unless just recorded
    env.close()
    reached = np.asarray(info.get('drone_reached_goal', [False] * n), dtype=bool)
    home = {y['flight'] for y in replans if y['status'] == 'approved' and y['reason'] == 'return to base'}
    arrived = sum(1 for i in range(n_f) if reached[i] and ids[i] not in home)   # at the ORIGINAL goal
    out = {'flights': n_f, 'intruders': len(extra), 'arrived': arrived,
           'crashed': int(env.crashed[:n_f].sum()), 'crashed_all': int(env.crashed.sum()),
           'pairs_below_2.0': len(close[2.0]), 'pairs_below_1.4': len(close[1.4]),
           'pairs_below_1.4_with_intruder': sum(1 for a, b in close[1.4]
                                                if (a >= n_f and a not in calls) or (b >= n_f and b not in calls)),
           'min_separation_m': None if min_sep == np.inf else round(min_sep, 3),
           'zone_entries': len(entered), 'zone_seconds': inside_s,
           'emergencies': [{'id': ids[i], **called.get(ids[i], {'t': None, 'status': 'never called'}),
                            'arrived': bool(reached[i])} for i in calls],
           'cancelled': int(env.cancelled[:n_f].sum()),
           'returned': len(home),
           'replans': replans}
    if monitor is not None:
        warn, warn_high, detail = [], [], []
        for (a, b), t_close in close[1.4].items():
            before = [e for e in monitor.events if e['flight'] in (ids[a], ids[b]) and e['state'] != 'ok'
                      and t_close - args.horizon <= e['t'] <= t_close]
            first = [e['t'] for e in before]
            high = [e['t'] for e in before if e['risk'] == 'high']
            warn.append(None if not first else round(t_close - min(first), 2))
            warn_high.append(None if not high else round(t_close - min(high), 2))
            recent = [y for y in replans if y['flight'] in (ids[a], ids[b]) and t_close - 10.0 <= y['t'] <= t_close]
            detail.append({'a': ids[a], 'b': ids[b], 't': round(t_close, 2), 'warning_s': warn[-1],
                           'replanned_within_10s': recent})
        close_t = {}                                  # flight id -> times it was in a close pair
        for (a, b), t_close in close[1.4].items():
            close_t.setdefault(ids[a], []).append(t_close)
            close_t.setdefault(ids[b], []).append(t_close)
        alerts = [e for e in monitor.events if e['state'] != 'ok']
        hit = [any(e['t'] <= tc <= e['t'] + args.horizon for tc in close_t.get(e['flight'], ())) for e in alerts]
        out['high_alerts'] = sum(e['risk'] == 'high' for e in alerts)
        out['high_alerts_followed'] = int(sum(h for e, h in zip(alerts, hit) if e['risk'] == 'high'))
        out['close_pair_high_warning_s'] = warn_high
        if tm is not None:
            out['telemetry'] = {'sent': tm.sent, 'lost': tm.lost}
        out['e3'] = {**monitor.summary(), 'events': monitor.events}
        out['close_pair_warning_s'] = warn
        out['close_pairs'] = detail
        out['alerts_followed_by_close_pass'] = int(sum(hit))
    if rec is not None:
        rec.finish(out, monitor)
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
    ap.add_argument('--modes', nargs='+', default=['no_e1', 'e1'], choices=['no_e1', 'e1', 'e4'])
    ap.add_argument('--zone-at', type=float, default=None,
                    help='a no-fly zone appears at this time (s)')
    ap.add_argument('--zone-place', choices=['busiest', 'random'], default='busiest',
                    help='where: the tile that cuts the most routes, or a random tile that cuts one')
    ap.add_argument('--risk', choices=['booking', 'intent'], default='booking', help="E3's risk rule (intent: E2)")
    ap.add_argument('--simulator', choices=['pybullet', 'webots_l2_multi'], default='pybullet',
                    help='webots_l2_multi: one run per command (one seed, mode and pilot), one Webots per process')
    ap.add_argument('--land-latch-m', type=float, default=None, help='C5 landing option (default: off)')
    ap.add_argument('--land-vz-max', type=float, default=None, help='C5 landing option (default: off)')
    ap.add_argument('--telemetry', type=float, nargs=3, default=None, metavar=('NOISE_M', 'DELAY_S', 'LOSS'),
                    help='the UTM hears the drones through reports (default: true positions)')
    ap.add_argument('--zone-for', type=float, default=None, help='zone active this many s (default: permanent)')
    ap.add_argument('--pilots', nargs='+', default=['simple', 'blind', 'learned'], choices=['simple', 'blind', 'learned'])
    ap.add_argument('--horizon', type=float, default=20.0, help='alert precision: close pass within this many s')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    if any(10 <= s <= 19 for s in args.seeds):
        ap.error('seeds 10-19 are reserved for test runs')
    if args.simulator != 'pybullet' and len(args.seeds) * len(args.modes) * len(args.pilots) > 1:
        ap.error('Webots: one run per command (one seed, one mode, one pilot)')

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
        zone, cut = None, 0
        if args.zone_at is not None:
            active = None if args.zone_for is None else [[args.zone_at, args.zone_at + args.zone_for]]
            zone, cut = place_zone(sky, flights, sky.bounds, args.zone_place,
                                   np.random.RandomState(7000 + seed), active)
        row = {'seed': seed, 'approved': len(flights), 'late': sum(x > 0 for x in late),
               'slow': sum(x < 1 for x in slow), 'intruders': len(extra), 'zone_cuts': cut}
        for mode in args.modes:
            for pilot in args.pilots:
                t0 = time.time()
                r = fly(cfg, args, layout, sky, flights, extra, late, slow, mode, pilot, seed, zone)
                row[f'{mode}/{pilot}'] = r
                print(f"seed {seed} {mode:5s} {pilot:7s}: arrived {r['arrived']}/{r['flights']}, crashed {r['crashed']} "
                      f"(+{r['crashed_all'] - r['crashed']} intruders), pairs < 1.4 m {r['pairs_below_1.4']} "
                      f"({r['pairs_below_1.4_with_intruder']} with an intruder), closest {r['min_separation_m']} m"
                      + (f", E3 alerts {r['e3']['alerts']}" if 'e3' in r else '')
                      + (f", zone entries {r['zone_entries']}" if zone is not None else '')
                      + (f", re-plans {sum(x['status'] == 'approved' for x in r['replans'])}/{len(r['replans'])}"
                         if mode == 'e4' else '') + f"  ({time.time() - t0:.0f} s)",
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
            if args.zone_at is not None:
                line += f", zone entries {sum(x['zone_entries'] for x in R)} ({sum(x['zone_seconds'] for x in R):.0f} s)"
            if mode == 'e4':
                rp = [y for x in R for y in x['replans']]
                line += (f", re-plans {sum(y['status'] == 'approved' for y in rp)} ok / {len(rp)}, "
                         f"returned to base {sum(x['returned'] for x in R)}, cancelled {sum(x['cancelled'] for x in R)}")
            if mode in ('e1', 'e4'):
                w = [v for x in R for v in x['close_pair_warning_s']]
                seen = [v for v in w if v is not None]
                n_al = sum(x['e3']['alerts'] for x in R)
                line += (f"; E3 alerts {n_al} ({sum(x['alerts_followed_by_close_pass'] for x in R)} followed by a "
                         f"close pass within {args.horizon:g} s), close pairs alerted before "
                         f"{len(seen)}/{len(w)}" + (f" (median {np.median(seen):.1f} s ahead)" if seen else ''))
                wh = [v for x in R for v in x['close_pair_high_warning_s'] if v is not None]
                line += (f"; high-risk alerts {sum(x['high_alerts'] for x in R)} "
                         f"({sum(x['high_alerts_followed'] for x in R)} followed by a close pass), close pairs with a "
                         f"high-risk alert before {len(wh)}/{len(w)}" + (f" (median {np.median(wh):.1f} s)" if wh else ''))
            print(line)
    path = args.out or os.path.join('eval_results', 'e1_layered',
                                    f"{args.map}_{args.level}_f{args.flights}_w{args.window:g}.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump({'args': vars(args), 'trials': rows}, f, indent=1, default=str)
    print(f"wrote {path}")


if __name__ == '__main__':
    main()
