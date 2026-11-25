"""
utm_server.py
-------------
E5 phase 3: the live UTM. One process runs a scenario's city in PyBullet with the learned policy,
in real time, with E1 (Planner), E3 (Monitor) and E4 (Responder), and takes the client nodes'
bookings as they come, over a small HTTP API. The browser pages in dashboard/ are its consoles.

    python scripts/utm_server.py scenarios/ecity_rooftops.yaml          # then http://localhost:8770/
    python scripts/utm_server.py scenarios/ecity_rooftops.yaml --slots 16 --speed 2 --port 8770

From the scenario it takes the map, the places (hubs, drop points, hospital) and the utm / sim
settings; its requests, emergencies, intruders and zones are not replayed: the consoles file them.

Access (utm/access.py, docs/access.md): every call below except /api/scene needs a login (POST /api/login {role,
passcode} sets a session cookie; GET /api/me, POST /api/logout). An operator sees and books only its own flights; the admin
sees all; dashboard/data/ is admin-only. --access FILE picks the passcode file (deployment/access.yaml), --no-auth turns the
login off (everyone is the admin).

API (JSON):
    GET  /api/scene                     map, places, layers, operators: what the consoles draw
    GET  /api/state?since=N              time, every flight (status, position, route) and the
                                         events from N on
    POST /api/request   {operator, to[, from, pad]}      a booking: the reply is E1's decision
    POST /api/emergency {operator, from, to[, layer]}    an emergency flight (E4 gives it priority)
    POST /api/bird      {flight | place | at:[x,y[,z]][, speed 1-3, count 1-4]}   admin only: birds join the simulation
                                         (aimed to cross a flight's route ahead of it, or to be at a place); the state then
                                         also lists the birds, and each flight whether its bird guard is active
    POST /api/zone      {polygon[, for_s]}               admin only: a no-fly zone from now on (local metres; E4
                                                         re-plans the flights it clashes with)

Drones: the simulator has a fixed number of drones (--slots). They wait parked; an approved booking
gets a free one, which waits on its pad until its booked departure (SwarmEnv.refly) and is free
again once it has landed (or was cancelled or crashed). A booking with no drone free is refused.
The UTM sees the true positions (no telemetry model) and E3 / E4 run once per second, as in fly().
"""

import argparse
import json
import os
import socket
import sys
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, unquote, urlparse

import numpy as np
import torch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import eval_suite as es                                            # noqa: E402
import run_scenario as rs                                          # noqa: E402
from envs import birds as birds_mod                                # noqa: E402
from envs.swarm_env import SwarmEnv                                 # noqa: E402
from modules.bird_guard import BirdGuardPolicy, guard_enabled       # noqa: E402
from modules.goal_hold import GoalHoldPolicy, goal_hold_enabled     # noqa: E402
from record_rollout import disable_all_collision_physics           # noqa: E402
from render_scenario import describe, short                        # noqa: E402
from utm import access as access_mod                              # noqa: E402  (who may see what)
from utm import flight_db, speeds                                  # noqa: E402  (the flight log, E1's flight profile)
from utm import scenario as scn                                    # noqa: E402
from utm.airspace_zones import Zone                                # noqa: E402
from utm.conformance import ConformanceMonitor                     # noqa: E402
from utm.contingency import ContingencyManager                     # noqa: E402
from utm.strategic_planner import FlightPlan                       # noqa: E402

FLIGHT_OPTIONS = ('land_latch_m', 'land_vz_max', 'land_wait_clear_m', 'cruise_z_bias', 'land_step_m',
                  'land_vz_high', 'land_slow_m', 'los_below_cruise', 'pad_queue_m')     # as eval_layered.fly()
STATUS = ('waiting', 'flying', 'landed', 'crashed', 'cancelled')
LEAD_S = 2.0                                     # s: a booking may leave this soon at the earliest

# Birds in the live system (docs/birds_report.md, the final configuration of the bird work): the onboard detector (360 deg,
# 12 m, Kalman q=2, birds not in the neighbour slots) and the bird guard, with a pool of bird slots that /api/bird fills.
BIRD_POOL = 12                                   # birds flying at once
BIRDS_PER_CALL = 4
BIRD_SPEED_M_S = (1.0, 3.0)
BIRD_PLACE_T_S = 5.0                             # a bird sent to a place or point is there in this many seconds
AVOID_QUIET_S = 2.0                              # a drone's guard quiet this long: that avoidance is over
BIRD_TRAIL = 10                                  # snapshots (~1 s) of trail
AIM_AHEAD_S = (2.0, 16.0)                        # a bird aimed at a flight crosses its route where the drone is this many s from now
LIVE_BIRDS = {'enabled': True, 'count': BIRD_POOL, 'radius': 0.12,
              'detector': {'enabled': True, 'mode': 'realistic', 'range_m': 12.0, 'fov_deg': 360.0, 'tracker': 'kalman',
                           'kalman_q': 2.0, 'share': False, 'birds_in_slots': False},
              'guard': {'enabled': True, 'layer_spacing_m': 2.0}}


def fmt(t):
    return f'{int(t // 60)}:{int(t % 60):02d}'


class LiveUTM:
    def __init__(self, sc, slots=16, speed=1.0, seed=20, db=None, birds=False):
        self.sc, self.speed, self.n = sc, float(speed), int(slots)
        self.layout, self.bounds, self.sky = rs.planner_for(sc)
        self.db_path = db                          # the flight log (utm/flight_db.py): written as things happen; None: off
        self.rec = flight_db.LiveRecorder(db, sc, self.speed, speeds.from_settings(self.sky.s)) if db else None
        cfg, args = rs.sim_setup(sc)
        sch = {'enabled': True}
        for k in FLIGHT_OPTIONS:
            if getattr(args, k, None) is not None:
                sch[k] = getattr(args, k)
        cfg['environment'].update(n_agents=self.n, crash_mode='remove', schedule=sch,
                                  route_follow={'enabled': True}, max_steps=10 ** 9)
        self.pool = None                           # the birds (envs/birds.py BirdPool); None: birds are off
        if birds:
            cfg['environment']['birds'] = json.loads(json.dumps(LIVE_BIRDS))
        self.dt = cfg['environment'].get('control_timestep', 0.032)
        self.env = SwarmEnv(cfg, device=torch.device('cpu'))
        disable_all_collision_physics(self.env, self.n)
        policy = es.MappoPolicy(cfg, args.checkpoint, torch.device('cpu'), True)
        policy.restore_normalizer(self.env)
        self.policy = GoalHoldPolicy(policy, cfg) if goal_hold_enabled(cfg) else policy
        self.guard = None
        if birds:
            if guard_enabled(cfg):                 # after goal-hold, as eval_layered.fly() does
                self.policy = self.guard = BirdGuardPolicy(self.policy, cfg)
                self.guard.skip_idle = True        # no bird tracked, no guard active: no per-drone work
            zs = [float(self.sky.layer_z(k)) for k in range(self.sky.s.n_layers)]
            self.z_band = (max(0.3, min(zs) - 0.5), max(zs) + 0.5)
            self.pool = birds_mod.BirdPool(BIRD_POOL, self.dt, LIVE_BIRDS['radius'], [list(b) for b in self.sky.bounds])
            self.bird_rng = np.random.default_rng([int(seed), 0xB12D])
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.places = sc['places']
        self.hubs = {p['operator']: name for name, p in self.places.items() if p.get('kind') == 'hub'}
        z0 = self.sky.s.start_z
        park = self._pad(next(iter(self.hubs.values()))) if self.hubs else (0.0, 0.0)
        zl = self.sky.layer_z((self.sky.s.layers_open or (0,))[0])
        key = [f'drone_{i}' for i in range(self.n)]
        here = [park[0], park[1], z0]
        self.obs = self.env.start_episode({k: here for k in key}, {k: here for k in key}, obstacles=self.layout,
                                          routes={k: [tuple(here), (park[0], park[1], zl), tuple(here)] for k in key},
                                          departures={k: np.inf for k in key}, birds=self.pool)
        self.monitor = ConformanceMonitor(self.sky, risk='booking')
        self.e4 = ContingencyManager(self.sky, self.monitor)
        self.flights = {}                         # flight id -> what the consoles show
        self.slot = {}                            # flight id -> drone slot while it has one
        self.free = list(range(self.n))
        self.pad_used = {}                        # (hub, pad index) -> last booking time
        self.events = []
        self.bird_name = {}                       # pool slot -> 'b3' while that bird flies
        self.bird_trail = {}                      # pool slot -> its last positions
        self.n_birds = 0
        self.avoid = {}                           # drone slot -> [flight id, closest bird (m), last time its guard was active]
        self.zones = []                           # no-fly zones filed so far
        self.n_mon = 0                            # E3 events already published
        self.lock = threading.Lock()
        self.k, self.t = 0, 0.0
        self.every = max(1, int(round(1.0 / self.dt)))
        self.running = True
        self.snap = {'t': 0.0, 'free': self.n, 'flights': []}   # what /api/state serves: read without the lock

    # ------------------------------------------------------------------ places
    def _pad(self, name, pick=False):
        p = self.places[name]
        if p.get('kind') != 'hub':
            return tuple(map(float, p['at'][:2]))
        pads = p['pads']
        k = min(range(len(pads)), key=lambda j: (self.pad_used.get((name, j), -1e9), j)) if pick else 0
        if pick:
            self.pad_used[name, k] = self.t
        return tuple(map(float, pads[k][:2]))

    def _layer(self, a, b):
        d = self.sc['demand']
        short_m, short_layers = d.get('short_trip_m'), d.get('short_layers')
        ls = short_layers if short_m and short_layers and np.hypot(b[0] - a[0], b[1] - a[1]) < float(short_m) \
            else d.get('layers', [0])
        return int(ls[len(self.flights) % len(ls)])

    def _event(self, kind, text, **extra):
        self.events.append({'t': round(self.t, 2), 'type': kind, 'text': text, **extra})
        if self.rec:
            self.rec.event(self.events[-1])

    # ------------------------------------------------------------------ bookings
    def _start(self, fid, d, kind, operator, frm, to):
        i = self.free.pop(0)
        z0 = self.sky.s.start_z
        self.env.refly(i, (d.waypoints[0][0], d.waypoints[0][1], z0), d.waypoints, d.t_dep)
        self.slot[fid] = i
        self.monitor.plans[fid] = d
        self.flights[fid] = {'id': fid, 'kind': kind, 'operator': operator, 'from': frm, 'to': to,
                             'status': 'waiting', 't_dep': round(d.t_dep, 2), 'arrival': round(d.arrival, 2),
                             'layer_z': round(self.sky.layer_z(d.layer), 1), 'route': [list(map(float, w)) for w in d.waypoints],
                             'pos': [d.waypoints[0][0], d.waypoints[0][1], z0], 'replans': 0, 'slot': i}
        if self.rec:
            self.rec.booked(self.flights[fid], self.t, d.layer)

    def _reply(self, fid, d, operator, frm, to):
        if d.status != 'approved':
            self._event('rejected', f'{operator}: {fid} {frm} → {to} refused ({d.reason})', flight=fid, node=operator,
                        reason=d.reason, frm=frm, to=to)
            return {'flight': fid, 'status': 'rejected', 'reason': d.reason}
        f = self.flights[fid]
        self._event('booked', f"{operator}: {fid} {frm} → {to} approved, take-off {fmt(d.t_dep)} at "
                    f"{f['layer_z']:.0f} m", flight=fid, node=operator)
        return {'flight': fid, 'status': 'approved', 'take_off': f['t_dep'], 'arrival': f['arrival'],
                'layer_m': f['layer_z'], 'delay_s': round(d.delay, 1), 'from': frm, 'to': to}

    def request(self, operator, to, frm=None, pad=None):
        with self.lock:
            frm = frm or self.hubs.get(operator)
            if frm not in self.places or to not in self.places:
                return {'status': 'rejected', 'reason': 'unknown place (or no hub for this operator)'}
            if not self.free:
                return {'status': 'rejected', 'reason': 'no drone free'}
            a = self._pad(frm, pick=True) if pad is None else tuple(map(float, self.places[frm]['pads'][int(pad)][:2]))
            b = self._pad(to, pick=True)
            fid = f'f{len(self.flights)}'
            d = self.sky.submit(FlightPlan(fid, a, b, layer=self._layer(a, b), t_earliest=self.t + LEAD_S,
                                           operator=operator))
            if d.status == 'approved':
                self._start(fid, d, 'flight', operator, frm, to)
            return self._reply(fid, d, operator, frm, to)

    def emergency(self, operator, frm, to, layer=None):
        with self.lock:
            if frm not in self.places or to not in self.places:
                return {'status': 'rejected', 'reason': 'unknown place'}
            if not self.free:
                return {'status': 'rejected', 'reason': 'no drone free'}
            a, b = self._pad(frm), self._pad(to)
            fid = f'emergency_{sum(f["kind"] == "emergency" for f in self.flights.values())}'
            layer = int(self.sc['demand'].get('layers', [0])[0]) if layer is None else int(layer)   # high: the long-trip layer
            fp = FlightPlan(fid, a, b, layer, self.t + LEAD_S, 1, operator)
            acts = self.e4.emergency(fp, self.t, self._airborne())
            d = acts[0]['decision']
            if d.status == 'approved':
                self._start(fid, d, 'emergency', operator, frm, to)
            reply = self._reply(fid, d, operator, frm, to)
            self._apply(acts[1:])
            return reply

    def zone(self, polygon, for_s=None):
        with self.lock:
            xy = [(float(p[0]), float(p[1])) for p in polygon]
            if len(xy) < 3:
                return {'status': 'error', 'reason': 'a zone needs at least 3 corners'}
            zid = f'zone_{len(self.zones)}'
            z = Zone(zid, 'Live closure', polygon=xy, active=[[self.t, self.t + float(for_s)]] if for_s else [])
            acts = self.e4.add_zone(z, self.t, self._airborne())
            plans = [a['decision'].status == 'approved' for a in acts if a.get('kind') != 'log']
            done, stuck = sum(plans), len(plans) - sum(plans)
            self.zones.append({'id': zid, 'polygon': [list(p) for p in xy], 't_on': round(self.t, 2),
                               'active': [[round(self.t, 2), round(self.t + float(for_s), 2)]] if for_s else []})
            self._event('zone_on', 'No-fly zone closed' + (f' for {int(float(for_s))} s' if for_s else ' until further notice')
                        + f': {done} flight{"" if done == 1 else "s"} re-planned' + (f', {stuck} with no free route yet' if stuck else ''), zone=zid)
            self._apply(acts)
            return {'status': 'approved', 'zone': zid, 'replanned': done}

    # ------------------------------------------------------------------ birds
    def _times(self, f):
        """Flight f's route as an array, and the time (s) E1's flight profile takes to reach each waypoint from the first."""
        r = np.array(f['route'], dtype=float)
        v = self.sky.s
        vd = v.v_descent if v.v_descent is not None else v.v_climb
        seg = np.diff(r, axis=0)
        h, dz = np.linalg.norm(seg[:, :2], axis=1), seg[:, 2]
        dt = np.maximum(h / v.v_cruise, np.abs(dz) / np.where(dz >= 0, v.v_climb, vd))
        return r, np.concatenate([[0.0], np.cumsum(dt)])

    def _aim_at_flight(self, f):
        """(aim point, seconds until the drone is there, the route's heading there) for a bird aimed at flight f: where the
        drone will be AIM_AHEAD_S from now, half of what is left of its flight at the most, by E1's flight profile."""
        r, T = self._times(f)
        pos = np.array(f['pos'], dtype=float)
        best, at = np.inf, 0.0
        for j in range(len(r) - 1):                  # where the drone is on its route: the nearest point in 3D
            d = r[j + 1] - r[j]
            u = float(np.clip(np.dot(pos - r[j], d) / max(float(np.dot(d, d)), 1e-9), 0.0, 1.0))
            e = float(np.linalg.norm(r[j] + u * d - pos))
            if e < best:
                best, at = e, float(T[j] + u * (T[j + 1] - T[j]))
        left = T[-1] - at
        tau = min(float(np.clip(0.5 * left, *AIM_AHEAD_S)), 0.9 * left)
        aim = np.array([np.interp(at + tau, T, r[:, c]) for c in range(3)])
        k = min(int(np.searchsorted(T, at + tau, side='right')) - 1, len(r) - 2)
        d = r[k + 1, :2] - r[k, :2]
        wait = max(float(f['t_dep']) - self.t, 0.0) if f['status'] == 'waiting' else 0.0
        return aim, wait + tau, float(np.arctan2(d[1], d[0])) if np.hypot(*d) > 0.05 else None

    def _level(self):
        return float(self.sky.layer_z((self.sky.s.layers_open or (0,))[0]))

    def add_bird(self, flight=None, place=None, at=None, speed=2.0, count=1, who='admin'):
        """Birds join the running simulation. flight: each is aimed to cross that flight's route ahead of it, timed to
        arrive with it; place / at ([x, y(, z)]): to be there BIRD_PLACE_T_S from now. Returns (http code, reply)."""
        if self.pool is None:
            return 404, {'status': 'error', 'reason': 'birds are off (--no-birds)'}
        if sum(x is not None for x in (flight, place, at)) != 1:
            raise ValueError('give exactly one target: flight, place or at')
        speed, count = float(speed), int(count)
        if not BIRD_SPEED_M_S[0] <= speed <= BIRD_SPEED_M_S[1]:
            raise ValueError(f'speed must be {BIRD_SPEED_M_S[0]:g}-{BIRD_SPEED_M_S[1]:g} m/s')
        if not 1 <= count <= BIRDS_PER_CALL:
            raise ValueError(f'count must be 1-{BIRDS_PER_CALL}')
        cfg = {'speed': [speed, speed], 'radius': LIVE_BIRDS['radius'], 'vz_max': 0.3, 'circle_radius': [2.0, 4.0],
               'max_turn_rate': 0.35}
        with self.lock:
            if len(self.pool.free_slots()) < count:
                return 409, {'status': 'error', 'reason': f'too many birds flying (at most {BIRD_POOL} at once)'}
            face = None
            if flight is not None:
                f = self.flights.get(flight)
                if f is None or f['status'] not in ('waiting', 'flying'):
                    raise ValueError(f'{flight} is not on its pad or flying')
                aim0, t_aim, face = self._aim_at_flight(f)
                near, extra = f'near {flight}', {'flight': flight, 'node': f['operator']}
            elif place is not None:
                if place not in self.places:
                    raise ValueError(f'unknown place {place!r}')
                p = self.places[place]
                xy = p['pads'][0][:2] if p.get('kind') == 'hub' else p['at'][:2]
                aim0, t_aim, near, extra = np.array([float(xy[0]), float(xy[1]), self._level()]), BIRD_PLACE_T_S, f'near {place}', {}
            else:
                if len(at) not in (2, 3):
                    raise ValueError('at must be [x, y] or [x, y, z]')
                z = float(at[2]) if len(at) == 3 else self._level()
                aim0, t_aim, near, extra = np.array([float(at[0]), float(at[1]), z]), BIRD_PLACE_T_S, \
                    f'at ({float(at[0]):.0f}, {float(at[1]):.0f})', {}
            aim0[2] = float(np.clip(aim0[2], *self.z_band))
            for j in range(count):                # several birds: the same crossing, 0.8 s apart, from different sides
                head = None if face is None else face + float(self.bird_rng.choice([-1.0, 1.0])) * \
                    (np.pi / 2 + float(self.bird_rng.uniform(-0.6, 0.6)))
                bird = birds_mod.Bird.planned(self.bird_rng, cfg, aim0, max(0.5, t_aim + (j - (count - 1) / 2.0) * 0.8),
                                              self.dt, 'cross', self.z_band, self.sky.bounds, fold=False, heading=head)
                slot = self.env.launch_bird(bird)
                self.n_birds += 1
                self.bird_name[slot], self.bird_trail[slot] = f'b{self.n_birds}', []
            text = ('bird added ' if count == 1 else f'{count} birds added ') + f'{near} ({who})'
            self._event('bird_added', text, **extra)
        return 200, {'status': 'ok', 'added': count, 'text': text}

    def _birds_step(self):
        """Per step: forget the birds that left, and follow each guarded drone's closest bird (for the flight log)."""
        pool = self.pool
        for slot in [s for s in self.bird_name if pool.t0[s] < 0]:
            del self.bird_name[slot], self.bird_trail[slot]
        active = self.guard.active if self.guard is not None else None   # (None until the guard has run once)
        cur = {i: fid for fid, i in self.slot.items()}
        for i in list(self.avoid):                # the flight ended or the drone was re-used: its note ends with it
            if cur.get(i) != self.avoid[i][0]:
                self._avoided(i)
        if active is None or not (active.any() or self.avoid):
            return
        act = pool.active()
        d = np.full(self.n, np.inf)
        if act.any():
            d = np.linalg.norm(self.env.adapter.get_positions()[:, None, :] - pool.positions()[None, act, :], axis=2).min(axis=1)
        for i in range(self.n):                   # one avoidance = from the guard taking over until it has been quiet for AVOID_QUIET_S
            if active[i] and i in cur:
                self.avoid.setdefault(i, [cur[i], np.inf, self.t])[2] = self.t
            if i in self.avoid:
                self.avoid[i][1] = min(self.avoid[i][1], float(d[i]))
                if self.t - self.avoid[i][2] >= AVOID_QUIET_S:
                    self._avoided(i)

    def _avoided(self, i):
        fid, closest, _ = self.avoid.pop(i)
        f = self.flights.get(fid)
        if f is None or bool(self.env.bird_crashed[i]) or not np.isfinite(closest):
            return
        self._event('bird_avoided', f'{fid} avoided a bird (closest {closest:.1f} m)', flight=fid, node=f['operator'])

    def _birds_view(self):
        """The flying birds for /api/state: id, position (m), the last positions (trail), radius."""
        out, pos = [], self.pool.positions()
        for slot, name in self.bird_name.items():
            p = [round(float(v), 2) for v in pos[slot]]
            tr = self.bird_trail[slot]
            out.append({'id': name, 'pos': p, 'trail': list(tr), 'r': float(self.pool.radii[slot])})
            tr.append(p)
            del tr[:-BIRD_TRAIL]
        return out

    # ------------------------------------------------------------------ flying
    def _airborne(self):
        pos = self.env.adapter.get_positions()
        fly = ~(self.env.waiting | self.env.landed | self.env.crashed)
        return {fid: pos[i] for fid, i in self.slot.items() if fly[i]}

    def _apply(self, acts):
        for a in acts:
            d, fid = a['decision'], a['flight']
            if a.get('kind') == 'log':
                self._event('log', f"{fid}: {a['why']}", flight=fid)
                continue
            e = {'type': 'replan', 'flight': fid, 'why': a['why'], 'status': d.status, 'reason': d.reason,
                 'around_traffic': getattr(d, 'around_traffic', False)}
            f = self.flights.get(fid)
            text = describe(e) or f"{fid} re-planned"
            extra = {'flight': fid, 'tag': short(e), 'why': a['why']}
            if f is not None:
                extra['node'] = f['operator']
            if fid in self.slot and d.status == 'approved':
                later = d.arrival - f['arrival'] if f else 0.0
                self.env.set_route(self.slot[fid], d.waypoints, departure=d.t_dep)
                f.update(t_dep=round(d.t_dep, 2), arrival=round(d.arrival, 2), replans=f['replans'] + 1,
                         layer_z=round(self.sky.layer_z(d.layer), 1), route=[list(map(float, w)) for w in d.waypoints])
                if self.rec:
                    self.rec.replanned(f)
                if later >= 1.0:
                    extra['later'] = round(later, 1)
            elif fid in self.slot and a['why'].endswith('-> cancelled') and self.env.waiting[self.slot[fid]]:
                self.env.cancel(self.slot[fid])
            self._event('replan', text, **extra)

    def _alerts(self):
        """E3's high-risk alerts become events addressed to the flight's operator (the console and
        the operator page show them as notifications)."""
        for e in self.monitor.events[self.n_mon:]:
            f = self.flights.get(e['flight'])
            text = describe({'type': 'alert', 'flight': e['flight'], 'state': e['state'], 'risk': e['risk'], 'other': e['other']})
            if e['state'] != 'ok' and e['risk'] == 'high' and text and f is not None:
                self._event('alert', text, flight=e['flight'], node=f['operator'], risk='high', state=e['state'], other=None if e['other'] is None else str(e['other']))
        self.n_mon = len(self.monitor.events)

    def _sync(self):
        """Flight statuses from the simulator; free the drones that are done."""
        pos = self.env.adapter.get_positions()
        code = np.where(self.env.cancelled, 4, np.where(self.env.crashed, 3, np.where(
            self.env.landed, 2, np.where(self.env.waiting, 0, 1))))
        for fid, i in list(self.slot.items()):
            f = self.flights[fid]
            s = STATUS[int(code[i])]
            if s != f['status']:
                if s == 'landed':
                    self._event('arrived', f"{fid} landed at {f['to']}", flight=fid, node=f['operator'])
                elif s == 'crashed':
                    hit = self.pool is not None and bool(self.env.bird_crashed[i])
                    self._event('crashed', f'{fid} hit a bird' if hit else f'{fid} crashed', flight=fid, node=f['operator'])
                f['status'] = s
                if s == 'flying':
                    f.setdefault('t_off', round(self.t, 2))       # its real take-off (the tracking page)
                if self.rec:
                    self.rec.status(fid, s, self.t, f)
            if s == 'flying':
                f['pos'] = [round(float(v), 2) for v in pos[i]]
            if s in ('landed', 'crashed', 'cancelled'):
                if s == 'landed':
                    f['pos'] = [f['route'][-1][0], f['route'][-1][1], self.sky.s.start_z]
                del self.slot[fid]
                self.free.append(i)

    def step(self):
        with self.lock:
            acts = self.policy.act(self.obs, self.env, self.n)
            self.obs, _, _, _ = self.env.step(acts)
            self.k += 1
            self.t = self.k * self.dt
            if self.k % self.every == 0:
                self._apply(self.e4.step(self.t, self._airborne()))
                self._alerts()
            if self.pool is not None:
                self._birds_step()
            self._sync()
            if self.rec and self.k % self.every == 0:         # the flight log's 1 Hz position track
                self.rec.track(self.t, [(fid, f['pos']) for fid, f in self.flights.items() if f['status'] == 'flying'])
            if self.k % 3 == 0:                   # ~10 snapshots per simulated second
                # display only: 'guard' stays on until the avoidance is over (AVOID_QUIET_S after the guard last had the controls)
                guarded = {fid: bool(self.guard.active[i]) or i in self.avoid for fid, i in self.slot.items()} if self.guard is not None and self.guard.active is not None else {}
                self.snap = {'t': round(self.t, 2), 'free': len(self.free),
                             'flights': [{**f, 'guard': guarded.get(f['id'], False)} if self.pool is not None else dict(f)
                                         for f in self.flights.values()],
                             'high': [fid for fid, r in self.monitor.risk.items() if r == 'high' and fid in self.slot]}
                if self.pool is not None:
                    self.snap['birds'] = self._birds_view()

    def run(self):
        wall0 = time.time()
        while self.running:
            self.step()
            lag = self.t / self.speed - (time.time() - wall0)
            if lag < -0.5:                        # fell behind (a booking held the lock): carry on from here,
                wall0 = time.time() - self.t / self.speed   # don't race to catch up (it looks like a jump)
                lag = 0.0
            time.sleep(lag if lag > 0 else 0.0005)    # always yield: the web requests get the lock too

    # ------------------------------------------------------------------ what the consoles read
    def scene(self):
        b = self.layout
        return {'schema': 'utm-live/1', 'name': self.sc['name'], 'bounds': self.bounds,
                'layers': [float(self.sky.layer_z(k)) for k in range(self.sky.s.n_layers)],
                'buildings': [{'c': [round(float(o['center'][0]), 2), round(float(o['center'][1]), 2)],
                               'h': [round(float(v), 2) for v in o['half']], 'z': round(float(o['height']), 2),
                               'yaw': round(float(o.get('yaw', 0.0)), 3)} for o in b],
                'places': [{'name': n, 'kind': p.get('kind', ''), 'operator': p.get('operator', ''),
                            'pads': p['pads'] if p.get('kind') == 'hub' else [p['at']]} for n, p in self.places.items()],
                'operators': sorted(self.hubs) + sorted({e.get('operator', 'Hospital') for e in self.sc['emergencies']}
                                                        or {'Hospital'}),
                'birds': {'on': self.pool is not None, 'pool': BIRD_POOL, 'per_call': BIRDS_PER_CALL,
                          'speed': list(BIRD_SPEED_M_S)},
                'slots': self.n, 'speeds': speeds.from_settings(self.sky.s),     # speeds: E1's flight profile (the ETA)
                'speed': (self.rec.pace() if self.rec else None) or self.speed,   # sim s per wall s, as it really runs
                'clock': self.rec.clock() if self.rec else []}                    # [t, wall epoch] knots: the tracking page's clock

    def note_reply(self, reply, body):
        """A refusal that never became a flight (no drone free, unknown place) goes to the flight log as Rejected."""
        if self.rec and reply.get('status') == 'rejected' and 'flight' not in reply:
            with self.lock:
                self.rec.refused(self.t, body.get('operator'), body.get('from') or self.hubs.get(body.get('operator')),
                                 body.get('to'), reply.get('reason'))

    def close(self):
        """Stops the simulation loop and closes the flight log (flights not finished are marked as such)."""
        self.running = False
        time.sleep(0.3)                           # let the loop finish its step
        return self.rec.close() if self.rec else 0

    def state(self, since=0):
        """The newest snapshot (no lock: the simulation is not held up) and the events from `since` on."""
        snap, ev = self.snap, self.events[since:]          # events only grow: a slice is safe
        return {**snap, 'events': ev, 'n_events': since + len(ev), 'zones': list(self.zones)}


class V6Server(ThreadingHTTPServer):
    """The ::1 side of localhost: browsers try it first, and with nothing there each request waits ~2 s."""
    address_family = socket.AF_INET6


def handler(live, root, access):
    """The HTTP side. `access` (utm/access.py) decides who may read and do what: every /api call except /api/scene,
    /api/login and /api/me needs a session; operators see only their own flights; the dashboard's data/ folder
    (every operator's flights) is for the admin only."""
    class H(SimpleHTTPRequestHandler):
        def __init__(self, *a, **k):
            super().__init__(*a, directory=root, **k)

        def log_message(self, *a):
            pass

        def end_headers(self):
            if not any(h.lower().startswith(b'cache-control') for h in getattr(self, '_headers_buffer', [])):
                self.send_header('Cache-Control', 'no-cache')          # a page changed on disk is not shown from the browser's cache
            super().end_headers()

        def _json(self, obj, code=200, cookie=None):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            if cookie:
                self.send_header('Set-Cookie', cookie)
            self.end_headers()
            self.wfile.write(body)

        # -- who is asking
        def _token(self):
            return access_mod.cookie_token(self.headers.get('Cookie'))

        def _role(self):
            return access.role_of(self._token())

        def _who(self, role):
            return {'role': role, 'admin': access.is_admin(role), 'no_auth': access.no_auth}

        def _need(self):
            """The caller's role, or None after answering 401."""
            role = self._role()
            if role is None:
                self._json({'status': 'error', 'reason': 'login required'}, 401)
            return role

        def _deny(self, why='not allowed'):
            self._json({'status': 'error', 'reason': why}, 403)

        def do_GET(self):
            u = urlparse(self.path)
            if u.path == '/api/scene':                                  # the map and places: public
                scene = live.scene()
                return self._json({**scene, 'roles': access.role_names(scene['operators']), 'no_auth': access.no_auth})
            if u.path == '/api/me':
                role = self._role()
                return self._json(self._who(role) if role else {'status': 'error', 'reason': 'login required'}, 200 if role else 401)
            if u.path == '/api/state':
                role = self._need()
                if role is None:
                    return
                st = live.state(int(parse_qs(u.query).get('since', ['0'])[0]))
                return self._json(access_mod.filter_state(st, role, access.is_admin(role)))
            if u.path in ('/api/flightlog', '/api/track'):         # the flight log database (not with --no-db)
                role = self._need()
                if role is None:
                    return
                if not live.db_path:
                    return self._json({'status': 'error', 'reason': 'the flight log is off (--no-db)'}, 404)
                admin, q = access.is_admin(role), parse_qs(u.query)
                db = flight_db.connect(live.db_path)               # a reader of its own: WAL, the simulation's writes go on
                try:
                    active = live.rec.run_id if live.rec else None
                    if u.path == '/api/flightlog':
                        log = flight_db.flightlog_dict(db, active or '', live.snap['t'])
                        return self._json(access_mod.filter_flightlog(log, role, admin))
                    tr = flight_db.track_dict(db, q.get('run', [active])[0], q['flight'][0])
                    if not access_mod.may_see_track(tr, role, admin):
                        return self._deny('not your flight')
                    return self._json(tr)
                except (KeyError, ValueError) as e:
                    if not admin:                                  # an unknown flight looks like someone else's: no probing
                        return self._deny('not your flight')
                    return self._json({'status': 'error', 'reason': str(e)}, 404)
                finally:
                    db.close()
            if u.path.startswith('/api/'):
                return self._json({'status': 'error', 'reason': 'not found'}, 404)
            if not access_mod.public_static(unquote(u.path)):         # data/*.json and anything else: admin only
                role = self._role()
                if role is None:
                    return self._json({'status': 'error', 'reason': 'login required'}, 401)
                if not access.is_admin(role):
                    return self._deny('admin only')
            return super().do_GET()

        do_HEAD = do_GET

        def do_POST(self):
            u = urlparse(self.path)
            origin = self.headers.get('Origin')
            if origin and urlparse(origin).netloc != self.headers.get('Host'):     # a page of another site: no
                return self._deny('cross-site request')
            try:
                body = json.loads(self.rfile.read(int(self.headers.get('Content-Length', 0))) or b'{}')
                if u.path == '/api/login':
                    if access.no_auth:
                        return self._json(self._who(access.admin))
                    token = access.login(str(body.get('role', '')), str(body.get('passcode', '')))
                    if token is None:
                        time.sleep(0.5)                                   # slows guessing a little
                        return self._json({'status': 'error', 'reason': 'wrong role or passcode'}, 401)
                    cookie = f'{access_mod.COOKIE}={token}; HttpOnly; SameSite=Strict; Path=/; Max-Age={int(access.ttl)}'
                    return self._json(self._who(access.role_of(token)), cookie=cookie)
                if u.path == '/api/logout':
                    access.logout(self._token())
                    return self._json({'status': 'ok'}, cookie=f'{access_mod.COOKIE}=; HttpOnly; SameSite=Strict; Path=/; Max-Age=0')
                if u.path == '/api/bird':                                  # birds join the simulation: the admin only
                    role = self._need()
                    if role is None:
                        return
                    if not access.is_admin(role):
                        return self._deny('admin only')
                    code, reply = live.add_bird(body.get('flight'), body.get('place'), body.get('at'),
                                                body.get('speed', 2.0), body.get('count', 1))
                    return self._json(reply, code)
                if u.path == '/api/zone':                                  # a no-fly zone: the admin only
                    role = self._need()
                    if role is None:
                        return
                    if not access.is_admin(role):
                        return self._deny('admin only')
                    return self._json(live.zone(body['polygon'], body.get('for_s')))
                if u.path not in ('/api/request', '/api/emergency'):
                    return self._json({'status': 'error', 'reason': 'not found'}, 404)
                role = self._need()
                if role is None:
                    return
                admin = access.is_admin(role)
                op = body.get('operator') if admin else role                # an operator books only as itself
                if not admin and body.get('operator') not in (None, role):
                    return self._deny(f'{role} cannot book for {body.get("operator")}')
                if u.path == '/api/request':
                    reply = live.request(op, body['to'], body.get('from'), body.get('pad'))
                else:
                    if not admin and role in live.hubs:                    # emergencies are filed by the emergency operator
                        return self._deny(f'{role} is a delivery operator: it cannot file emergencies')
                    reply = live.emergency(op, body['from'], body['to'], body.get('layer'))
                live.note_reply(reply, {**body, 'operator': op})
                return self._json(reply)
            except (KeyError, ValueError, TypeError) as e:
                return self._json({'status': 'error', 'reason': str(e)}, 400)
    return H


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('scenario')
    ap.add_argument('--port', type=int, default=8770)
    ap.add_argument('--slots', type=int, default=16, help='drones the simulator holds (bookings at once)')
    ap.add_argument('--speed', type=float, default=1.0, help='simulated seconds per wall second')
    ap.add_argument('--db', default=os.path.join(ROOT, 'eval_results', 'flight_log.db'),
                    help='the flight log database every booking, event and position is written to (utm/flight_db.py). Default: '
                         'the main one, eval_results/flight_log.db, so the recorded runs and the live flights are in one log')
    ap.add_argument('--no-db', action='store_true', help='do not write the flight log')
    ap.add_argument('--no-birds', action='store_true', help='no birds: no bird simulation, detector or guard (and no /api/bird)')
    ap.add_argument('--access', default=os.path.join(ROOT, 'deployment', 'access.yaml'),
                    help='the roles and their passcode hashes (utm/access.py, docs/access.md)')
    ap.add_argument('--no-auth', action='store_true', help='NO LOGIN: everyone is the admin (a demo switch; a warning is shown)')
    a = ap.parse_args()
    if a.no_auth:
        access = access_mod.Access(no_auth=True)
    else:
        try:
            access = access_mod.Access.load(os.path.abspath(a.access))
        except (OSError, ValueError) as e:
            sys.exit(f'no access file ({e}).\nCreate one (python -m utm.access set admin), or start with --no-auth '
                     f'to run with no login at all (everyone is the admin).')
    sc = scn.load(os.path.abspath(a.scenario))
    db = None if a.no_db else os.path.abspath(a.db)
    os.chdir(ROOT)                                # the scenario's relative paths (config, checkpoint) are the repo's
    print('building the city and the planner...', flush=True)
    live = LiveUTM(sc, a.slots, a.speed, sc['seed'], db=db, birds=not a.no_birds)
    threading.Thread(target=live.run, daemon=True).start()
    h = handler(live, os.path.join(ROOT, 'dashboard'), access)
    srv = ThreadingHTTPServer(('127.0.0.1', a.port), h)            # local only: both loopbacks, no network
    try:
        threading.Thread(target=V6Server(('::1', a.port), h).serve_forever, daemon=True).start()
    except OSError:
        pass                                      # no IPv6: 127.0.0.1 still works
    print(f'UTM live on http://localhost:{a.port}/  (operators: {", ".join(live.scene()["operators"])})', flush=True)
    if a.no_auth:
        print('*' * 78 + '\n* WARNING: --no-auth: NO LOGIN. Everyone who can reach this server is the admin\n'
              '* and sees and books for every operator. Demo use only.\n' + '*' * 78, flush=True)
    else:
        print(f'login required: roles {", ".join(access.role_names(live.scene()["operators"]))} ({a.access})', flush=True)
    print(f'flight log: {db} (run {live.rec.run_id}; --db FILE for another)' if db else 'flight log: off (--no-db)', flush=True)
    print(f'birds: on (up to {BIRD_POOL} at once; detector 360 deg / 12 m, Kalman, bird guard; the admin adds them, POST /api/bird)'
          if live.pool is not None else 'birds: off (--no-birds)', flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        live.close()


if __name__ == '__main__':
    main()
