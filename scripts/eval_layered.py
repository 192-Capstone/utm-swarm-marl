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

D5 drone failures (PyBullet, mode e1 or e4): --failures P gives each approved flight a probability P of
one failure (--failure-types: motor, land, comms) at a random time in mid-flight, the same schedule in
every mode and pilot of a seed. The UTM learns of it only from the telemetry reports (a perfect link
unless --telemetry says otherwise): E3 detects it in e1 and e4, E4's handlers react in e4 only. Off by
default (nothing is drawn then). Scored: per failure its fire time, E3's detection and E4's actions and
the outcome (crashed / forced_down / arrived); false alerts (a failure state on a flight with no
failure); a forced-down drone has not arrived.

Seeds 10-19 are refused (reserved for test runs).
"""

import argparse
import copy
import json
import math
import os
import re
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
from envs.disturbances import FAILURE_TYPES, draw_failures          # noqa: E402
from envs.swarm_env import LAND_RADIUS_M, SwarmEnv                 # noqa: E402
from modules.goal_hold import GoalHoldPolicy, goal_hold_enabled    # noqa: E402
from preview_route import load_map                                 # noqa: E402
from preview_strategic import random_requests                      # noqa: E402
from record_rollout import disable_all_collision_physics           # noqa: E402
from utm.airspace_zones import Zone, tile_blocks, zone_keys        # noqa: E402
from utm.conformance import FAILURE_STATES, ConformanceMonitor                     # noqa: E402
from utm.contingency import ContingencyManager                     # noqa: E402
from utm.intent import INTENT_VZ, predicted_path, profile_intent, route_intent   # noqa: E402
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


_CAUSED = re.compile(r'^(?:lost link|landing|intrusion by) (.+)$')    # E4's 'why' of a re-plan around a failed flight
                                                                      # (utm/contingency.py; tests pin the formats)


def caused_by(action, fid):
    """Is this E4 action about the failed flight fid? A log entry (kind 'log') names its flight; a re-plan of
    another flight carries only its 'why' ('lost link <id>', 'landing <id>', 'intrusion by <id>')."""
    if action.get('kind') == 'log':
        return action['flight'] == fid
    m = _CAUSED.match(str(action['why']))
    return m is not None and m.group(1) == str(fid) and action['flight'] != fid


def resolve_failures(failures, ids, n_flights):
    """[{'flight': id or index, 't', 'type'}] -> the same with flight ids (planned flights only; an id wins over an index)."""
    out = []
    for f in failures:
        fl = f['flight']
        if fl in ids[:n_flights]:
            fid = fl
        elif isinstance(fl, (int, np.integer)) and not isinstance(fl, bool) and 0 <= fl < n_flights:
            fid = ids[fl]
        else:
            raise ValueError(f"failure for unknown flight {fl!r} (planned flights only)")
        out.append({'flight': fid, 't': float(f['t']), 'type': f['type']})
    return out


class FailureFeed:
    """D5: what the UTM hears of the drones' failures, and what is scored. env: SwarmEnv with failures on; tm:
    the TelemetryModel; failures: resolved (resolve_failures)."""

    def __init__(self, env, tm, ids, failures):
        self.env, self.tm, self.ids = env, tm, list(ids)
        self.sched = {f['flight']: f for f in failures}
        self.idx = {fid: self.ids.index(fid) for fid in self.sched}
        self._muted, self._reported = set(), set()
        self._reported_t = {}                     # drone index -> when its emergency report went out
        self.takeoff = {}                         # drone index -> the time it left its pad in this run
        self._seen = {}                           # flight -> (failure state, is its failure) last held by E3
        self._skipped_at = {}                     # failed-to-fire flight -> its true state when the failure came due
        self.transitions = {}                     # failed flight -> [(t, state)] E3 held it in
        self.false_alerts = []                    # failure states of flights with no failure (yet)
        self._was_failed, self._was_forced = [''] * len(self.ids), [False] * len(self.ids)

    def true_state(self, i):
        """Drone i as it truly is (never what the UTM hears): crashed, landed (a landing or a forced landing),
        cancelled, waiting (on its pad) or flying."""
        env = self.env
        return ('crashed' if env.crashed[i] else 'landed' if env.landed[i] else 'cancelled' if env.cancelled[i]
                else 'waiting' if env.waiting[i] else 'flying')

    def step(self, t, pos):
        """Every control step: mute a comms-lost drone (once); a motor failure sends its one emergency report now
        (the fall ends before the next 1 Hz report), through the link model like any report. True if it did."""
        env, sent = self.env, False
        for fid, i in self.idx.items():           # when each scheduled flight actually took off (E4 may move it)
            if i not in self.takeoff and not env.waiting[i] and not env.cancelled[i]:
                self.takeoff[i] = round(float(t), 3)
        if env.failures_skipped:                  # the env skipped a failure whose drone was not flying: note why
            for f in env.failures_skipped:
                fid = self.ids[int(f['drone'].split('_')[1])]
                self._skipped_at.setdefault(fid, self.true_state(int(f['drone'].split('_')[1])))
        for fid, i in self.idx.items():
            if env.comms_lost[i] and i not in self._muted:
                self._muted.add(i)
                self.tm.mute(fid)
            if env.failed[i] == 'motor' and env.emergency_report[i] and i not in self._reported:
                self._reported.add(i)
                self._reported_t[i] = t
                vel = np.asarray(env.adapter.get_raw_state()[f'drone_{i}']['velocity'], dtype=float)
                self.tm.send(t, {fid: (pos[i], vel, [], 'emergency')})
                self.tm.receive(t)
                sent = True
        return sent

    def reported_now(self, i, t):
        """Did drone i's emergency report go out at this very step? Its regular report of the step is then not
        sent: with the same t_sent it would replace the emergency one in tm.latest before E3 sees it."""
        return self._reported_t.get(i) == t

    def status(self, i):
        """The status in drone i's 1 Hz report: 'landing' during a forced landing, else 'ok'."""
        return 'landing' if self.env.failed[i] == 'land' and not self.env.landed[i] else 'ok'

    def observe(self, t, monitor):
        """After E3 ran: note the failure states it holds (first detection per failed flight, false alerts)."""
        for fid, s in monitor.state.items():
            own = fid in self.idx and bool(self.env.failed[self.idx[fid]])
            cur = (s, own) if s in FAILURE_STATES else None
            if self._seen.get(fid) == cur:
                continue
            self._seen[fid] = cur
            if cur is None:
                continue
            if own:
                self.transitions.setdefault(fid, []).append((round(float(t), 2), s))
            else:
                self.false_alerts.append({'flight': fid, 't': round(float(t), 2), 'state': s,
                                          'true_state': self.true_state(self.ids.index(fid)) if fid in self.ids else None})

    def new_events(self):
        """[('failure', i, type) / ('forced_down', i, '')] since the last call (for the recording)."""
        out = []
        for i in range(len(self.ids)):
            if self.env.failed[i] and not self._was_failed[i]:
                out.append(('failure', i, self.env.failed[i]))
            if self.env.forced_down[i] and not self._was_forced[i]:
                out.append(('forced_down', i, ''))
            self._was_failed[i], self._was_forced[i] = self.env.failed[i], bool(self.env.forced_down[i])
        return out

    def _why_skipped(self, fid, i):
        """Why a failure never fired: the drone was not flying when it came due (the state noted then), or, if the
        run never got that far, its final state; still flying or waiting at the end = the episode ended first."""
        state = self._skipped_at.get(fid) or self.true_state(i)
        return {'crashed': 'crashed before', 'landed': 'arrived before', 'cancelled': 'cancelled before'}.get(
            state, 'still on its pad when due' if fid in self._skipped_at else 'episode ended before')

    def results(self, reached, replans):
        """One row per scheduled failure. reached: drone_reached_goal per drone; replans: the run's E4 action log."""
        env, out = self.env, []
        for fid, f in self.sched.items():
            i = self.idx[fid]
            fired = bool(env.failed[i])
            tr = self.transitions.get(fid, [])
            outcome = ('crashed' if env.crashed[i] else 'forced_down' if env.forced_down[i]
                       else 'arrived' if reached[i] else 'unfinished')
            why = None if fired else self._why_skipped(fid, i)
            out.append({'flight': fid, 'type': f['type'], 't': f['t'],
                        'status': 'fired' if fired else 'skipped', 'reason': why,
                        'takeoff_t': self.takeoff.get(i),
                        'into_flight_s': round(float(env.failure_t[i]) - self.takeoff[i], 2) if fired and i in self.takeoff else None,
                        'fired_t': round(float(env.failure_t[i]), 2) if fired else None,
                        'e3': {'detected_t': tr[0][0], 'state': tr[0][1], 'transitions': tr} if tr else None,
                        'e4': [y for y in replans if fired and y['t'] >= env.failure_t[i] - 1e-6 and caused_by(y, fid)],
                        'outcome': outcome})
        return out


def fly(cfg, args, layout, sky, flights, extra, late, slow, mode, pilot, seed, zone=None, rec=None, zones=None, *, failures=None):
    """One run. rec: an optional utm.flight_log.FlightLog that records it for the dashboard (E6).
    zones: [(t, Zone), ...] appearing mid-flight (E6 scenarios); zone (at args.zone_at) is one of them.
    extra: unplanned drones (intruders: id, start, goal, t, route); an entry with 'emergency': True
    is an emergency flight instead (E6): its drone waits on its pad until t, then it is filed with
    'priority' (E4 / E1 give it the air it needs) and takes off on the approved route.
    failures: [{'flight': id or index, 't': s, 'type': 'motor' | 'land' | 'comms'}] (D5; None = none, today's
    behaviour exactly; [] = on with nobody failing): mode e1 or e4 and PyBullet only; the UTM then hears the
    drones only through reports (FailureFeed), E3 detects the failures and, in e4, E4 reacts."""
    zones = list(zones or []) + ([(args.zone_at, zone)] if zone is not None else [])
    on = failures is not None
    plan = 'no_e1' if mode == 'no_e1' else 'e1'
    n_f = len(flights)
    n = n_f + len(extra)
    cfg = json.loads(json.dumps(cfg))
    sch = {'enabled': True}
    for k in ('land_latch_m', 'land_vz_max', 'land_wait_clear_m', 'cruise_z_bias', 'land_step_m',
              'land_vz_high', 'land_slow_m', 'los_below_cruise'):   # flight options (absent = off)
        if getattr(args, k, None) is not None:
            sch[k] = getattr(args, k)
    cfg['environment'].update(n_agents=n, crash_mode='remove', schedule=sch, route_follow={'enabled': True})
    if on:                                        # cfg is this call's own copy (above)
        cfg['environment']['failures'] = {'enabled': True}
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
    if on:
        if mode not in ('e1', 'e4'):
            raise ValueError('failures need the UTM to hear the drones (mode e1 or e4)')
        if getattr(args, 'simulator', 'pybullet') != 'pybullet':
            raise ValueError('failures need adapter.set_falling: PyBullet only')
        all_ids = [f['id'] for f in flights] + [x['id'] for x in extra]
        failures = resolve_failures(failures, all_ids, n_f)
        sched = [{'drone': f'drone_{all_ids.index(f["flight"])}', 't': f['t'], 'type': f['type']} for f in failures]
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
                            obstacles=layout, routes=dict(zip(key, routes)), departures=dict(zip(key, deps)),
                            **({'failures': sched} if on else {}))
    ids = [f['id'] for f in flights] + [x['id'] for x in extra]
    sky = copy.deepcopy(sky)                      # E4 changes bookings: each run starts fresh
    tm = None                                     # E2: the UTM hears reports, not the truth
    if mode in ('e1', 'e4') and (on or args.telemetry is not None or args.risk == 'intent'):
        tm = TelemetryModel(*(args.telemetry or (0.0, 0.0, 0.0)), seed=9000 + seed)
    monitor = (ConformanceMonitor(sky, risk=args.risk, blind_margin=float(getattr(args, 'blind_margin', 0.0) or 0.0),
                                  **({'failures': True, 'noise_m': tm.noise} if on else {}))
               if mode in ('e1', 'e4') else None)
    feed = FailureFeed(env, tm, ids, failures) if on else None
    full = bool(getattr(args, 'intent_profile', False))   # E2 option D: intent with the climb and landing
    vz_down = float(getattr(args, 'land_vz_max', None) or INTENT_VZ)   # a landing cap, if on, sets the descent

    def phase(i, p):
        """A planned drone's flight phase for its intent ('climb' / 'cruise' / 'land'), None without a route."""
        if not (env.followers and i < n_f and key[i] in env.followers):
            return None
        if not env._climbed.get(key[i], False):
            return 'climb'
        return 'land' if env._landing[i] or np.hypot(*(p[:2] - env.goals[i][:2])) < LAND_RADIUS_M else 'cruise'

    def intent_of(i, p, vel, t):
        ph = phase(i, p)
        if full:
            return [] if ph is None else profile_intent(env.followers[key[i]], p, np.hypot(*vel[:2]), t, ph,
                                                        env.goals[i][2], sky.s.v_cruise, INTENT_VZ, vz_down)
        return route_intent(env.followers[key[i]], p, np.hypot(*vel[:2]), t) if ph == 'cruise' else []
    prev = {}                                     # drone -> true position a report period ago
    every_frame = None

    def status():                                 # per drone, utm/flight_log.py codes
        return np.where(env.cancelled, 4, np.where(env.crashed, 3, np.where(env.landed, 2, np.where(env.waiting, 0, 1))))

    if rec is not None:
        every_frame = max(1, int(round(rec.frame_dt / dt)))
        rec.start(sky, layout, dt, [
            {'id': ids[i], 'kind': 'flight' if i < n_f else ('emergency' if i in calls else 'intruder'),
             'operator': flights[i].get('operator', '') if i < n_f else calls[i].get('operator', 'emergency') if i in calls else '',
             'start': starts[i], 'goal': goals[i], 't_dep': deps[i] if np.isfinite(deps[i]) else None,
             'route': [] if i in calls else routes[i],
             'arrival': flights[i]['arrival'][plan] if i < n_f else None,
             'layer': sky.layer_of(routes[i][1][2]) if len(routes[i]) > 2 else 0} for i in range(n)],
            [z for _, z in zones])
    e4 = (ContingencyManager(sky, monitor, **({'failures': True, 'landing_descent_ms': env.failure_cfg['land_vz']}
                                               if on else {})) if mode == 'e4' else None)
    zkeys = [set(zone_keys(z, sky.sky, sky.s.n_layers)) for _, z in zones]
    zone_on, entered, inside_s, replans = [False] * len(zones), set(), 0.0, []

    def utm_check(t, here, paths, acts):          # E3 (and E4) on what the UTM now hears
        reports = {} if feed is None else {'reports': tm.latest}
        if e4 is not None:
            acts += e4.step(t, here, paths, **reports)
        elif monitor is not None:
            monitor.check(t, here, paths, **reports)
        if feed is not None:
            feed.observe(t, monitor)

    def apply_acts(acts):                         # E4's actions: recorded, and the new routes sent to the drones
        for a in acts:
            d = a['decision']
            if rec is not None:
                rec.replan(a['t'], a['flight'], a['why'], d)
            replans.append({'t': a['t'], 'flight': a['flight'], 'why': a['why'], 'status': d.status, 'reason': d.reason,
                            **({'kind': a['kind']} if 'kind' in a else {})})     # log entries (D5 handlers) only
            if a['flight'] not in ids:
                continue
            i = ids.index(a['flight'])
            if d.status == 'approved':
                env.set_route(i, d.waypoints, departure=d.t_dep)
            elif a['why'].endswith('-> cancelled') and env.waiting[i]:
                env.cancel(i)

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
        if feed is not None and feed.step(t, pos) and (step + 1) % every:
            # a motor failure's emergency report: E3 / E4 look at it now, not at the next 1 Hz tick (the fall
            # ends within a second); on a tick the check below does it. Same inputs as the tick's, without forgetting
            quiet = set(silent(tm.latest, t))
            now = {fid: r for fid, r in tm.latest.items() if fid not in quiet}
            acts_now = []
            utm_check(t, {fid: r.pos for fid, r in now.items()},
                      {fid: predicted_path(r, t) for fid, r in now.items()} if args.risk == 'intent' else None, acts_now)
            apply_acts(acts_now)
        if rec is not None and (step + 1) % every_frame == 0:
            rec.frame(t, pos, status())
        if rec is not None:                       # arrivals and crashes as events (the E5 dashboard)
            for i in np.flatnonzero(env.landed & ~was_landed):
                if not env.forced_down[i]:        # a forced landing is not an arrival
                    rec.event(t, 'arrived', flight=ids[i])
            for i in np.flatnonzero(env.crashed & ~was_crashed):
                rec.event(t, 'crashed', flight=ids[i])
            if feed is not None:
                for kind, i, what in feed.new_events():
                    rec.event(t, kind, flight=ids[i], **({'failure_type': what} if what else {}))
            was_landed, was_crashed = env.landed.copy(), env.crashed.copy()
        if (step + 1) % every == 0:
            here, paths = {ids[i]: pos[i] for i in air}, None
            if tm is not None:
                states = {}
                for i in air:
                    if feed is not None and feed.reported_now(i, t):
                        continue
                    vel = (pos[i] - prev[i]) / (every * dt) if i in prev else np.zeros(3)
                    states[ids[i]] = (pos[i], vel, intent_of(i, pos[i], vel, t),
                                      *([feed.status(i)] if feed is not None else []))
                tm.send(t, states)
                tm.receive(t)
                for fid in silent(tm.latest, t):  # no report for LINK_TIMEOUT_S: landed, crashed or
                    tm.forget(fid)                #   link lost (never the simulator's truth)
                here = {fid: r.pos for fid, r in tm.latest.items()}
                if args.risk == 'intent':
                    paths = {fid: predicted_path(r, t) for fid, r in tm.latest.items()}
                    monitor.blind = {fid for fid in tm.latest if fid not in monitor.plans}   # no flight plan: intruders
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
                    fp = FlightPlan(ids[i], tuple(x['start']), tuple(x['goal']), int(x.get('layer', 0)), t, int(x.get('priority', 1)),
                                    x.get('operator', 'emergency'))
                    got = (e4.emergency(fp, t, here) if e4 is not None      # first: the emergency itself
                           else [{'t': t, 'flight': fp.id, 'why': 'emergency', 'decision': sky.submit(fp, now=t)}])
                    acts_e4 += got
                    d = got[0]['decision']
                    called[fp.id] = {'t': round(t, 2), 'status': d.status, 't_dep': d.t_dep}
                    if d.status == 'approved' and monitor is not None:
                        monitor.plans[fp.id] = d          # E3 watches it like any planned flight
                    elif d.status != 'approved':          # refused: it never takes off, the run doesn't wait
                        env.cancel(i)
            utm_check(t, here, paths, acts_e4)
            if rec is not None:
                rec.tick(t, monitor, paths)
            apply_acts(acts_e4)
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
    arrived = sum(1 for i in range(n_f) if reached[i] and ids[i] not in home and not env.forced_down[i])   # at the ORIGINAL goal
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
    if on:
        out['forced_down'] = int(env.forced_down[:n_f].sum())
        out['failures'] = feed.results(reached, replans)
        out['false_alerts'] = feed.false_alerts
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
    ap.add_argument('--intent-profile', action='store_true',
                    help='E2 option D: intents include the climb and the landing')
    ap.add_argument('--blind-margin', type=float, default=0.0,
                    help='E2 option B (small): extra m of radius for pairs with a drone that has no flight plan')
    ap.add_argument('--simulator', choices=['pybullet', 'webots_l2_multi'], default='pybullet',
                    help='webots_l2_multi: one run per command (one seed, mode and pilot), one Webots per process')
    ap.add_argument('--land-latch-m', type=float, default=None, help='C5 landing option (default: off)')
    ap.add_argument('--land-vz-max', type=float, default=None, help='C5 landing option (default: off)')
    ap.add_argument('--telemetry', type=float, nargs=3, default=None, metavar=('NOISE_M', 'DELAY_S', 'LOSS'),
                    help='the UTM hears the drones through reports (default: true positions)')
    ap.add_argument('--zone-for', type=float, default=None, help='zone active this many s (default: permanent)')
    ap.add_argument('--pilots', nargs='+', default=['simple', 'blind', 'learned'], choices=['simple', 'blind', 'learned'])
    ap.add_argument('--failures', type=float, default=None, metavar='P',
                    help='D5: each approved flight has this probability of one failure (default: none, nothing drawn)')
    ap.add_argument('--failure-types', nargs='+', default=list(FAILURE_TYPES), choices=list(FAILURE_TYPES))
    ap.add_argument('--horizon', type=float, default=20.0, help='alert precision: close pass within this many s')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    if any(10 <= s <= 19 for s in args.seeds):
        ap.error('seeds 10-19 are reserved for test runs')
    if args.failures is not None:
        if not 0.0 <= args.failures <= 1.0:
            ap.error('--failures P: P must be in [0, 1]')
        if 'no_e1' in args.modes:
            ap.error('--failures needs the UTM to hear the drones: modes e1 / e4 only (drop no_e1)')
        if args.simulator != 'pybullet':
            ap.error('--failures is PyBullet only (Webots has no set_falling)')
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
        sched = None                                  # D5: one schedule per seed, flown in every mode and pilot
        if args.failures is not None:
            drawn = draw_failures(np.random.RandomState(8000 + seed),
                                  [(f['id'], f['e1'][0] + late[i], f['arrival']['e1'] + late[i])
                                   for i, f in enumerate(flights)], args.failures, tuple(args.failure_types))
            sched = [{'flight': d['drone'], 't': d['t'], 'type': d['type']} for d in drawn]
        row = {'seed': seed, 'approved': len(flights), 'late': sum(x > 0 for x in late),
               'slow': sum(x < 1 for x in slow), 'intruders': len(extra), 'zone_cuts': cut,
               **({'failures': sched} if sched is not None else {})}
        for mode in args.modes:
            for pilot in args.pilots:
                t0 = time.time()
                r = fly(cfg, args, layout, sky, flights, extra, late, slow, mode, pilot, seed, zone, failures=sched)
                row[f'{mode}/{pilot}'] = r
                acts = [y for y in r['replans'] if y['status'] != 'logged']     # D5 handlers' log entries are not re-plans
                print(f"seed {seed} {mode:5s} {pilot:7s}: arrived {r['arrived']}/{r['flights']}, crashed {r['crashed']} "
                      f"(+{r['crashed_all'] - r['crashed']} intruders), pairs < 1.4 m {r['pairs_below_1.4']} "
                      f"({r['pairs_below_1.4_with_intruder']} with an intruder), closest {r['min_separation_m']} m"
                      + (f", E3 alerts {r['e3']['alerts']}" if 'e3' in r else '')
                      + (f", zone entries {r['zone_entries']}" if zone is not None else '')
                      + (f", re-plans {sum(x['status'] == 'approved' for x in acts)}/{len(acts)}"
                         if mode == 'e4' else '')
                      + (f", failures fired {sum(x['status'] == 'fired' for x in r['failures'])}/{len(r['failures'])}, "
                         f"forced down {r['forced_down']}, false alerts {len(r['false_alerts'])}" if sched is not None else '')
                      + f"  ({time.time() - t0:.0f} s)",
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
                rp = [y for x in R for y in x['replans'] if y['status'] != 'logged']
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
            if args.failures is not None:
                F = [y for x in R for y in x['failures']]
                line += (f", failures fired {sum(y['status'] == 'fired' for y in F)}/{len(F)}, "
                         f"forced down {sum(x['forced_down'] for x in R)}, "
                         f"false alerts {sum(len(x['false_alerts']) for x in R)}")
            print(line)
    path = args.out or os.path.join('eval_results', 'e1_layered',
                                    f"{args.map}_{args.level}_f{args.flights}_w{args.window:g}.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump({'args': {k: v for k, v in vars(args).items()
                              if args.failures is not None or k not in ('failures', 'failure_types')}, 'trials': rows}, f, indent=1, default=str)
    print(f"wrote {path}")


if __name__ == '__main__':
    main()
