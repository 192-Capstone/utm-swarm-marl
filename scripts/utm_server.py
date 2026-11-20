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

API (JSON):
    GET  /api/scene                      map, places, layers, operators: what the consoles draw
    GET  /api/state?since=N              time, every flight (status, position, route) and the
                                         events from N on
    POST /api/request   {operator, to[, from, pad]}      a booking: the reply is E1's decision
    POST /api/emergency {operator, from, to[, layer]}    an emergency flight (E4 gives it priority)

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
from urllib.parse import parse_qs, urlparse

import numpy as np
import torch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import eval_suite as es                                            # noqa: E402
import run_scenario as rs                                          # noqa: E402
from envs.swarm_env import SwarmEnv                                 # noqa: E402
from modules.goal_hold import GoalHoldPolicy, goal_hold_enabled     # noqa: E402
from record_rollout import disable_all_collision_physics           # noqa: E402
from render_scenario import describe, short                        # noqa: E402
from utm import scenario as scn                                    # noqa: E402
from utm.conformance import ConformanceMonitor                     # noqa: E402
from utm.contingency import ContingencyManager                     # noqa: E402
from utm.strategic_planner import FlightPlan                       # noqa: E402

FLIGHT_OPTIONS = ('land_latch_m', 'land_vz_max', 'land_wait_clear_m', 'cruise_z_bias', 'land_step_m',
                  'land_vz_high', 'land_slow_m', 'los_below_cruise')     # as eval_layered.fly()
STATUS = ('waiting', 'flying', 'landed', 'crashed', 'cancelled')
LEAD_S = 2.0                                     # s: a booking may leave this soon at the earliest


def fmt(t):
    return f'{int(t // 60)}:{int(t % 60):02d}'


class LiveUTM:
    def __init__(self, sc, slots=16, speed=1.0, seed=20):
        self.sc, self.speed, self.n = sc, float(speed), int(slots)
        self.layout, self.bounds, self.sky = rs.planner_for(sc)
        cfg, args = rs.sim_setup(sc)
        sch = {'enabled': True}
        for k in FLIGHT_OPTIONS:
            if getattr(args, k, None) is not None:
                sch[k] = getattr(args, k)
        cfg['environment'].update(n_agents=self.n, crash_mode='remove', schedule=sch,
                                  route_follow={'enabled': True}, max_steps=10 ** 9)
        self.dt = cfg['environment'].get('control_timestep', 0.032)
        self.env = SwarmEnv(cfg, device=torch.device('cpu'))
        disable_all_collision_physics(self.env, self.n)
        policy = es.MappoPolicy(cfg, args.checkpoint, torch.device('cpu'), True)
        policy.restore_normalizer(self.env)
        self.policy = GoalHoldPolicy(policy, cfg) if goal_hold_enabled(cfg) else policy
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
                                          departures={k: np.inf for k in key})
        self.monitor = ConformanceMonitor(self.sky, risk='booking')
        self.e4 = ContingencyManager(self.sky, self.monitor)
        self.flights = {}                         # flight id -> what the consoles show
        self.slot = {}                            # flight id -> drone slot while it has one
        self.free = list(range(self.n))
        self.pad_used = {}                        # (hub, pad index) -> last booking time
        self.events = []
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

    def _reply(self, fid, d, operator, frm, to):
        if d.status != 'approved':
            self._event('rejected', f'{operator}: {fid} {frm} → {to} refused ({d.reason})', flight=fid, node=operator)
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
                if later >= 1.0:
                    extra['later'] = round(later, 1)
            elif fid in self.slot and a['why'].endswith('-> cancelled') and self.env.waiting[self.slot[fid]]:
                self.env.cancel(self.slot[fid])
            self._event('replan', text, **extra)

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
                    self._event('crashed', f'{fid} crashed', flight=fid, node=f['operator'])
                f['status'] = s
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
            self._sync()
            if self.k % 3 == 0:                   # ~10 snapshots per simulated second
                self.snap = {'t': round(self.t, 2), 'free': len(self.free),
                             'flights': [dict(f) for f in self.flights.values()]}

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
                'slots': self.n}

    def state(self, since=0):
        """The newest snapshot (no lock: the simulation is not held up) and the events from `since` on."""
        snap, ev = self.snap, self.events[since:]          # events only grow: a slice is safe
        return {**snap, 'events': ev, 'n_events': since + len(ev)}


class V6Server(ThreadingHTTPServer):
    """The ::1 side of localhost: browsers try it first, and with nothing there each request waits ~2 s."""
    address_family = socket.AF_INET6


def handler(live, root):
    class H(SimpleHTTPRequestHandler):
        def __init__(self, *a, **k):
            super().__init__(*a, directory=root, **k)

        def log_message(self, *a):
            pass

        def _json(self, obj, code=200):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            u = urlparse(self.path)
            if u.path == '/api/scene':
                return self._json(live.scene())
            if u.path == '/api/state':
                return self._json(live.state(int(parse_qs(u.query).get('since', ['0'])[0])))
            return super().do_GET()

        def do_POST(self):
            u = urlparse(self.path)
            try:
                body = json.loads(self.rfile.read(int(self.headers.get('Content-Length', 0))) or b'{}')
                if u.path == '/api/request':
                    return self._json(live.request(body['operator'], body['to'], body.get('from'), body.get('pad')))
                if u.path == '/api/emergency':
                    return self._json(live.emergency(body['operator'], body['from'], body['to'], body.get('layer')))
            except (KeyError, ValueError, TypeError) as e:
                return self._json({'status': 'error', 'reason': str(e)}, 400)
            self._json({'status': 'error', 'reason': 'not found'}, 404)
    return H


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('scenario')
    ap.add_argument('--port', type=int, default=8770)
    ap.add_argument('--slots', type=int, default=16, help='drones the simulator holds (bookings at once)')
    ap.add_argument('--speed', type=float, default=1.0, help='simulated seconds per wall second')
    a = ap.parse_args()
    sc = scn.load(os.path.abspath(a.scenario))
    os.chdir(ROOT)                                # the scenario's relative paths (config, checkpoint) are the repo's
    print('building the city and the planner...', flush=True)
    live = LiveUTM(sc, a.slots, a.speed, sc['seed'])
    threading.Thread(target=live.run, daemon=True).start()
    h = handler(live, os.path.join(ROOT, 'dashboard'))
    srv = ThreadingHTTPServer(('127.0.0.1', a.port), h)            # local only: both loopbacks, no network
    try:
        threading.Thread(target=V6Server(('::1', a.port), h).serve_forever, daemon=True).start()
    except OSError:
        pass                                      # no IPv6: 127.0.0.1 still works
    print(f'UTM live on http://localhost:{a.port}/  (operators: {", ".join(live.scene()["operators"])})', flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        live.running = False


if __name__ == '__main__':
    main()
