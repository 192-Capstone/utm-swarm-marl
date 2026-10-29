"""
flight_log.py
-------------
E6: the recording of one scenario run, everything the dashboard (E5) needs to replay it. One
gzip-compressed JSON file. Provisional (schema version below): it will change with the
dashboard; bump the version when it does.

    {
      "schema":   "utm-flight-log/2",
      "scenario": {...},                                  # the scenario as run (utm/scenario.py)
      "sim":      {"simulator", "dt", "frame_dt"},
      "map":      {"bounds": [[xmin, xmax], [ymin, ymax]], "obstacles": [...], "layers": [z, ...]},
      "drones":   [{"id", "kind": "flight" | "intruder" | "emergency", "operator", "start", "goal",
                    "t_dep", "arrival", "layer", "route": [[x, y, z], ...]}],
      "cells":    {h3 id: [[x, y], ...]},                 # hexagon outlines (local m) of every booked cell
      "bookings": {flight id: [[h3 id, layer, t0, t1], ...]},          # E1's, at take-off
      "zones":    [{"id", "polygon", "active", "t_on"}],
      "frames":   {"t": [...], "pos": [[x, y, z] per drone] per frame, "status": [code per drone] per frame},
      "ticks":    [{"t", "risk": {id: [state, risk]}, "paths": {id: [[x, y, z], ...]}}],   # 1 Hz, UTM view
      "events":   [{"t", "type", ...}],                   # alert, replan, zone_on, arrived, crashed
                                                          #   (arrived / crashed: {"flight"}, when it lands / crashes)
      "summary":  {...}                                   # the run's results (eval_layered fly())
    }

Status codes (frames.status): 0 waiting on its pad, 1 flying, 2 landed, 3 crashed, 4 cancelled.
Positions are rounded to mm. Paths in ticks are the UTM's predicted paths (E2), every 2 s.
"""

import gzip
import json
import sys
import time

import numpy as np

SCHEMA = 'utm-flight-log/2'
WAITING, FLYING, LANDED, CRASHED, CANCELLED = range(5)


def _plain(v):
    """JSON-friendly: numpy -> python, tuples -> lists, recursively."""
    if isinstance(v, dict):
        return {str(k): _plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple, np.ndarray)):
        return [_plain(x) for x in v]
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    return v


def _r(a, digits=3):
    return np.round(np.asarray(a, dtype=float), digits).tolist()


class FlightLog:
    """Collects one run; fly() calls start / frame / tick / event, then save()."""

    def __init__(self, scenario=None, frame_dt=0.25, progress_s=None):
        self.frame_dt = float(frame_dt)
        self.progress_s = progress_s                  # print a progress line every this many sim seconds
        self._next_progress, self._wall0 = 0.0, time.time()
        self.log = {'schema': SCHEMA, 'scenario': _plain(scenario or {}), 'sim': {}, 'map': {},
                    'drones': [], 'cells': {}, 'bookings': {}, 'zones': [],
                    'frames': {'t': [], 'pos': [], 'status': []}, 'ticks': [], 'events': [], 'summary': {}}
        self._sky = None

    def start(self, sky, layout, dt, drones, zones=()):
        """sky: the StrategicPlanner (bookings, H3 cells); drones: list of dicts (see the schema)."""
        self._sky = sky
        self.log['sim'] = {'dt': float(dt), 'frame_dt': self.frame_dt}
        self.log['map'] = {'bounds': _plain(sky.bounds), 'obstacles': _plain(list(layout)),
                           'layers': [float(sky.layer_z(k)) for k in range(sky.s.n_layers)]}
        self.log['drones'] = [_plain(d) for d in drones]
        for d in drones:
            dec = sky.current.get(d['id']) if d.get('kind') == 'flight' else None
            if dec is not None:
                self.log['bookings'][d['id']] = self._keys(dec.reservation)
        for z in zones:
            self.log['zones'].append({'id': z.id, 'polygon': _plain(getattr(z, 'polygon', None)),
                                      'active': _plain(z.active), 't_on': None})

    def _keys(self, reservation):
        out = []
        for (cell, layer), ivs in reservation.items():
            if cell not in self.log['cells']:
                self.log['cells'][cell] = _r(self._sky.sky.boundary(cell))
            out += [[cell, int(layer), round(float(a), 2), round(float(b), 2)] for a, b in ivs]
        return out

    def frame(self, t, pos, status):
        self.log['frames']['t'].append(round(float(t), 3))
        self.log['frames']['pos'].append(_r(pos))
        self.log['frames']['status'].append([int(s) for s in status])
        if self.progress_s and t >= self._next_progress:
            self._next_progress = t + self.progress_s
            n = np.bincount(np.asarray(status, dtype=int), minlength=5)
            m, sec = divmod(int(time.time() - self._wall0), 60)
            print(f't={t:5.0f}s  flying {n[1]}  landed {n[2]}  waiting {n[0]}  crashed {n[3]}  (wall {m}m{sec:02d}s)',
                  file=sys.stderr, flush=True)

    def tick(self, t, monitor=None, paths=None):
        """The UTM's view at a 1 Hz tick: E3 state and risk per drone, E2 predicted paths."""
        rec = {'t': round(float(t), 2)}
        if monitor is not None:
            rec['risk'] = {f: [monitor.state.get(f, 'ok'), monitor.risk.get(f, '')] for f in monitor.state}
        if paths:
            rec['paths'] = {f: _r(p[::4]) for f, (_, p) in paths.items()}
        self.log['ticks'].append(rec)

    def event(self, t, kind, **extra):
        self.log['events'].append({'t': round(float(t), 2), 'type': kind, **_plain(extra)})

    def replan(self, t, flight, why, decision):
        """An E4 re-plan: the new route and booking (if approved)."""
        extra = {'flight': flight, 'why': why, 'status': decision.status, 'reason': decision.reason}
        if decision.status == 'approved':
            extra.update(route=_r(decision.waypoints), t_dep=float(decision.t_dep), arrival=float(decision.arrival),
                         booking=self._keys(decision.reservation))
        self.event(t, 'replan', **extra)

    def zone_on(self, t, zone_id):
        for z in self.log['zones']:
            if z['id'] == zone_id:
                z['t_on'] = round(float(t), 2)
        self.event(t, 'zone_on', zone=zone_id)

    def finish(self, summary, monitor=None):
        if monitor is not None:
            for e in monitor.events:
                if e['state'] != 'ok':
                    self.event(e['t'], 'alert', flight=e['flight'], state=e['state'], risk=e['risk'],
                               other=None if e['other'] is None else str(e['other']))
        self.log['events'].sort(key=lambda e: e['t'])
        self.log['summary'] = _plain({k: v for k, v in summary.items() if k not in ('e3', 'replans')})

    def save(self, path):
        with gzip.open(path, 'wt', encoding='utf-8') as f:
            json.dump(self.log, f, separators=(',', ':'))


def read(path):
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        log = json.load(f)
    if log.get('schema') != SCHEMA:
        raise ValueError(f"{path}: schema {log.get('schema')!r}, expected {SCHEMA!r}")
    return log
