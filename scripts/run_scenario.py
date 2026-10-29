"""
run_scenario.py
---------------
E6, system integration: one command runs the whole system on one scenario (utm/scenario.py):

    city (D1 / D3) -> demand (D4) -> E1 plans -> the learned policy flies them (PyBullet)
    -> E3 monitors (E2 reports / intent) -> E4 reacts to alerts, intruders, the zone
    -> E7 metrics (throughput, delay, fairness, safety per flight hour), and a recording for the
       dashboard (utm/flight_log.py)

    python scripts/run_scenario.py scenarios/ecity_demo.yaml
    python scripts/run_scenario.py scenarios/ecity_demo.yaml --record out.json.gz

The flying, monitoring and E4 loop is eval_layered.py's fly(), so a scenario run and the
evaluations behave the same. Prints the results and E7's metrics; writes the recording.
sim.simulator webots_l2_multi flies it in Webots (C5; DISPLAY set, one run per process).
C5's landing options are on by default here (sim.land_latch_m 2.4, sim.land_vz_max 0.5), and
two found in Webots: E1 holds a landing pad 10 s after arrival (utm.pad_after; E1's default 7 s
is PyBullet's p99 landing) and a drone waits above its pad while another is low over it
(sim.land_wait_clear_m 1.5).
"""

import argparse
import dataclasses
import json
import os
import sys
import time
import types

import numpy as np
import yaml

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import eval_layered as ev                                         # noqa: E402
from eval_e1_flown import plans                                    # noqa: E402
from preview_route import load_map                                 # noqa: E402
from preview_strategic import random_requests                      # noqa: E402
from utm import scenario as scn                                    # noqa: E402
from utm.airspace_zones import Zone                                # noqa: E402
from utm.demand import DemandSettings, generate_demand             # noqa: E402
from utm.flight_log import FlightLog                               # noqa: E402
from utm.metrics import safety_per_flight_hour, system_metrics     # noqa: E402
from utm.strategic_planner import FlightPlan, Settings, StrategicPlanner   # noqa: E402


def where(sc, v):
    """(x, y) of a place name (a hub: its first pad) or of [x, y]."""
    if isinstance(v, str):
        p = sc['places'][v]
        return tuple(map(float, p['pads'][0] if p.get('kind') == 'hub' else p['at']))
    return tuple(map(float, v[:2]))


def booked_requests(sc):
    """FlightPlans for the client nodes' requests (scenario requests), in time order. A hub end is
    the pad of that hub used longest ago (or the request's pad); the preferred layer follows the
    demand block's rule (layers; short_layers for trips under short_trip_m)."""
    hub_of = {p['operator']: name for name, p in sc['places'].items() if p.get('kind') == 'hub'}
    last = {}                                     # (hub, pad index) -> time of its last request
    d = sc['demand']
    layers, short_m, short_layers = d.get('layers', [0]), d.get('short_trip_m'), d.get('short_layers')

    def end(v, pad, t):
        if isinstance(v, str) and sc['places'][v].get('kind') == 'hub':
            pads = sc['places'][v]['pads']
            k = int(pad) if pad is not None else min(range(len(pads)), key=lambda j: (last.get((v, j), -1e9), j))
            last[v, k] = t
            return tuple(map(float, pads[k]))
        return where(sc, v)

    out = []
    for k, r in enumerate(sorted(sc['requests'], key=lambda r: float(r['t']))):
        t = float(r['t'])
        a = end(r.get('from', hub_of.get(r['operator'])), r.get('pad'), t)
        b = end(r['to'], None, t)
        short = short_m is not None and short_layers and np.hypot(b[0] - a[0], b[1] - a[1]) < float(short_m)
        layer = int(r.get('layer', (short_layers if short else layers)[k % len(short_layers if short else layers)]))
        out.append(FlightPlan(f'f{k}', a, b, layer=layer, t_earliest=t, operator=str(r['operator'])))
    return out


def requests(sc, layout, bounds, sky):
    """FlightPlans for the scenario's demand: the client nodes' requests, D4, or E1's random
    requests (flights + window)."""
    if sc['requests']:
        return booked_requests(sc)
    d = sc['demand']
    if 'flights' in d:
        return random_requests(np.random.RandomState(sc['seed']), sky, int(d['flights']),
                               window=float(d.get('window', 30.0)))
    d = dict(d)
    pads = d.pop('pads', None)                    # fixed pads: [x, y, kind] (kind hub / drop)
    keys = {f.name for f in dataclasses.fields(DemandSettings)}
    unknown = set(d) - keys
    if unknown:
        raise ValueError(f'demand: unknown settings {sorted(unknown)} (see utm/demand.py DemandSettings)')
    pads = None if pads is None else [tuple(p) for p in pads]
    return generate_demand(layout, bounds, DemandSettings(**d), seed=sc['seed'], pads=pads).plans


def _layer_route(sky, start, goal, what, layer=0):
    r = sky.route(tuple(start), tuple(goal), layer)
    if r is None:
        raise ValueError(f'{what}: no route from {start} to {goal}')
    z0, zl = sky.s.start_z, sky.layer_z(layer)
    return [(r[0][0], r[0][1], z0)] + [(a, b, zl) for a, b in r] + [(r[-1][0], r[-1][1], z0)]


def extra_drones(sc, sky, rng, window):
    """The intruders (a count: random trips; or a list) and the emergency flights' drones."""
    ix = sc['intruders']
    if isinstance(ix, int):
        out = ev.intruders(rng, sky, ix, window)
    else:
        out = [{'id': f'intruder_{k}', 'start': where(sc, x['start']), 'goal': where(sc, x['goal']), 't': float(x['t']),
                'route': _layer_route(sky, where(sc, x['start']), where(sc, x['goal']), f'intruder {k}',
                                      int(x.get('layer', 0)))}
               for k, x in enumerate(ix)]
    for k, e in enumerate(sc['emergencies']):
        e = {**e, 'start': where(sc, e.get('from', e.get('start'))), 'goal': where(sc, e.get('to', e.get('goal')))}
        out.append({'id': f'emergency_{k}', 'start': tuple(e['start']), 'goal': tuple(e['goal']),
                    't': float(e['t']), 'priority': int(e.get('priority', 1)), 'emergency': True,
                    'layer': int(e.get('layer', 0)), 'operator': e.get('operator', 'emergency'),   # its client node
                    'route': _layer_route(sky, e['start'], e['goal'], f'emergency {k}', int(e.get('layer', 0)))})   # until filed
    return out


def make_zones(sc, sky, flights):
    """[(t, Zone)] and how many approved flights each cuts."""
    out, cuts = [], []
    for k, z in enumerate(sc['zones']):
        active = [[float(z['at']), float(z['at']) + float(z['for'])]] if z.get('for') else None
        if 'polygon' in z:
            zone, cut = Zone(f'zone_{k}', 'Mid-flight closure', polygon=[tuple(p) for p in z['polygon']],
                             active=active or []), None
        else:
            zone, cut = ev.place_zone(sky, flights, sky.bounds, z.get('place', 'busiest'),
                                      np.random.RandomState(7000 + sc['seed'] + k), active)
            zone.id = f'zone_{k}'
        out.append((float(z['at']), zone))
        cuts.append(cut)
    return out, cuts


def checkpoint_path(path):
    """The scenario's checkpoint: as given (from the current directory), else from the repo root."""
    for p in (path, os.path.join(ROOT, path)):
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(f"checkpoint not found: {path!r} (looked from {os.getcwd()} and from the repo root "
                            f"{os.path.abspath(ROOT)}); set sim.checkpoint in the scenario")


def planner_for(sc):
    """The map (layout, bounds) and E1 with the scenario's utm settings."""
    ns = types.SimpleNamespace(**sc['map'])
    ns.level, ns.seed = sc['map'].get('level', 'medium'), sc['map'].get('seed', 0)
    layout, bounds = load_map(ns)
    s = Settings(pad_after=float(sc['utm']['pad_after']), layer_floor=float(sc['utm']['layer_floor']))
    if sc['utm']['max_delay'] is not None:
        s.max_delay = float(sc['utm']['max_delay'])
    s.n_layers = int(sc['utm']['n_layers'])
    if sc['utm']['layer_penalty'] is not None:
        s.layer_penalty = float(sc['utm']['layer_penalty'])
    s.traffic_detour = bool(sc['utm']['traffic_detour'])
    for k in ('v_climb', 'v_descent', 'v_descent_high', 'slow_band_m'):     # the flight profile E1 books (null: E1's)
        if sc['utm'][k] is not None:
            setattr(s, k, float(sc['utm'][k]))
    if sc['utm']['layers_open'] is not None:
        s.layers_open = tuple(int(k) for k in sc['utm']['layers_open'])
    sky = StrategicPlanner(layout, bounds, s)
    return layout, bounds, sky


def sim_setup(sc):
    """The simulator config and the flight options (args) the scenario flies with."""
    with open(sc['sim']['config'], encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    cfg.pop('scenario_mix', None)
    cfg['environment']['collision_physics'] = False
    cfg.setdefault('obstacles', {})['enabled'] = False
    if sc['sim']['altitude_cap'] is not None:
        cfg.setdefault('observation', {})['altitude_cap'] = float(sc['sim']['altitude_cap'])
    if sc['sim']['position_origin'] is not None:
        cfg.setdefault('observation', {})['position_origin'] = sc['sim']['position_origin']
    if sc['sim']['hold_engage_m'] is not None:          # goal-hold: engage here, release 0.2 m further out
        e = float(sc['sim']['hold_engage_m'])
        cfg.setdefault('goal_hold', {}).update(engage_dist=e, release_dist=e + 0.2)
    ckpt = checkpoint_path(sc['sim']['checkpoint']) if sc['sim']['pilot'] == 'learned' else sc['sim']['checkpoint']
    return cfg, types.SimpleNamespace(checkpoint=ckpt, horizon=20.0, risk=sc['utm']['risk'],
                                 simulator=sc['sim']['simulator'], land_latch_m=sc['sim']['land_latch_m'],
                                 land_vz_max=sc['sim']['land_vz_max'], land_wait_clear_m=sc['sim']['land_wait_clear_m'],
                                 cruise_z_bias=sc['sim']['cruise_z_bias'], land_step_m=sc['sim']['land_step_m'],
                                 land_vz_high=sc['sim']['land_vz_high'], land_slow_m=sc['sim']['land_slow_m'],
                                 los_below_cruise=sc['sim']['los_below_cruise'] or None,
                                 telemetry=sc['utm']['telemetry'], zone_at=None,
                                 extra_time=60.0 + sum(float(z.get('for') or 60.0) for z in sc['zones']))


def run(sc, record=None):
    """Runs one scenario; returns (results, FlightLog)."""
    layout, bounds, sky = planner_for(sc)
    reqs = requests(sc, layout, bounds, sky)
    op = {fp.id: fp.operator for fp in reqs}
    decisions, flights = plans(sky, reqs)
    for f in flights:
        f['operator'] = op.get(f['id'], '')
    window = max((fp.t_earliest for fp in reqs), default=0.0) + 1.0
    rng = np.random.RandomState(5000 + sc['seed'])
    dz = sc['disruptions']
    late, slow = ev.disruptions(rng, flights, types.SimpleNamespace(**dz))
    extra = extra_drones(sc, sky, rng, window)
    zones, cuts = make_zones(sc, sky, flights)

    duration = float(sc['demand'].get('duration_s', sc['demand'].get('window', 30.0)))
    e7 = system_metrics(sky, reqs, duration)                      # before fly(): E1's decisions as filed
    cfg, args = sim_setup(sc)
    rec = FlightLog(sc, progress_s=10.0)
    out = ev.fly(cfg, args, layout, sky, flights, extra, late, slow, sc['utm']['mode'], sc['sim']['pilot'],
                 sc['seed'], rec=rec, zones=zones)
    out.update(requested=len(reqs), approved=len(flights), zone_cuts=cuts)
    if e7['flight_s'] > 0:
        e7['safety'] = safety_per_flight_hour(out['pairs_below_1.4'], out['crashed'], e7['flight_s'])
    out['e7'] = e7
    rec.log['summary']['e7'] = e7
    path = record or sc.get('record')
    if path:
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        rec.save(path)
    return out, rec


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('scenario')
    ap.add_argument('--record', default=None, help='recording path (default: the scenario\'s record)')
    a = ap.parse_args()
    sc = scn.load(a.scenario)
    t0 = time.time()
    out, rec = run(sc, a.record)
    keep = ('requested', 'approved', 'flights', 'arrived', 'crashed', 'cancelled', 'returned', 'pairs_below_1.4',
            'zone_entries', 'zone_cuts', 'high_alerts', 'emergencies')
    print(json.dumps({k: out.get(k) for k in keep if k in out}, indent=1))
    e7 = out['e7']
    print(f"E7: throughput {e7['throughput_per_h']:.0f} flights/h, delay mean {e7['delay_mean_s']:.1f} s "
          f"(p95 {e7['delay_p95_s']:.1f}), fairness (Jain) {e7['fairness_jain']:.3f}, "
          f"{e7['flight_s'] / 3600:.2f} planned flight h"
          + (f", {e7['safety']['close_passes_per_h']:.1f} close passes / {e7['safety']['crashes_per_h']:.1f} crashes "
             "per flight h" if 'safety' in e7 else ''))
    print(f"recorded {len(rec.log['frames']['t'])} frames, {len(rec.log['events'])} events "
          f"-> {a.record or sc.get('record')}  ({time.time() - t0:.0f} s)")


if __name__ == '__main__':
    main()
