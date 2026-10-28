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
from utm.strategic_planner import Settings, StrategicPlanner       # noqa: E402


def requests(sc, layout, bounds, sky):
    """FlightPlans for the scenario's demand: D4, or E1's random requests (flights + window)."""
    d = sc['demand']
    if 'flights' in d:
        return random_requests(np.random.RandomState(sc['seed']), sky, int(d['flights']),
                               window=float(d.get('window', 30.0)))
    keys = {f.name for f in dataclasses.fields(DemandSettings)}
    unknown = set(d) - keys
    if unknown:
        raise ValueError(f'demand: unknown settings {sorted(unknown)} (see utm/demand.py DemandSettings)')
    return generate_demand(layout, bounds, DemandSettings(**d), seed=sc['seed']).plans


def _layer0_route(sky, start, goal, what):
    r = sky.route(tuple(start), tuple(goal), 0)
    if r is None:
        raise ValueError(f'{what}: no route from {start} to {goal}')
    z0, zl = sky.s.start_z, sky.layer_z(0)
    return [(r[0][0], r[0][1], z0)] + [(a, b, zl) for a, b in r] + [(r[-1][0], r[-1][1], z0)]


def extra_drones(sc, sky, rng, window):
    """The intruders (a count: random trips; or a list) and the emergency flights' drones."""
    ix = sc['intruders']
    if isinstance(ix, int):
        out = ev.intruders(rng, sky, ix, window)
    else:
        out = [{'id': f'intruder_{k}', 'start': tuple(x['start']), 'goal': tuple(x['goal']), 't': float(x['t']),
                'route': _layer0_route(sky, x['start'], x['goal'], f'intruder {k}')} for k, x in enumerate(ix)]
    for k, e in enumerate(sc['emergencies']):
        out.append({'id': f'emergency_{k}', 'start': tuple(e['start']), 'goal': tuple(e['goal']),
                    't': float(e['t']), 'priority': int(e.get('priority', 1)), 'emergency': True,
                    'route': _layer0_route(sky, e['start'], e['goal'], f'emergency {k}')})   # until filed
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


def run(sc, record=None):
    """Runs one scenario; returns (results, FlightLog)."""
    ns = types.SimpleNamespace(**sc['map'])
    ns.level, ns.seed = sc['map'].get('level', 'medium'), sc['map'].get('seed', 0)
    layout, bounds = load_map(ns)
    sky = StrategicPlanner(layout, bounds, Settings(pad_after=float(sc['utm']['pad_after'])))
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

    with open(sc['sim']['config'], encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    cfg.pop('scenario_mix', None)
    cfg['environment']['collision_physics'] = False
    cfg.setdefault('obstacles', {})['enabled'] = False
    duration = float(sc['demand'].get('duration_s', sc['demand'].get('window', 30.0)))
    e7 = system_metrics(sky, reqs, duration)                      # before fly(): E1's decisions as filed
    ckpt = checkpoint_path(sc['sim']['checkpoint']) if sc['sim']['pilot'] == 'learned' else sc['sim']['checkpoint']
    args = types.SimpleNamespace(checkpoint=ckpt, horizon=20.0, risk=sc['utm']['risk'],
                                 simulator=sc['sim']['simulator'], land_latch_m=sc['sim']['land_latch_m'],
                                 land_vz_max=sc['sim']['land_vz_max'], land_wait_clear_m=sc['sim']['land_wait_clear_m'],
                                 telemetry=sc['utm']['telemetry'], zone_at=None,
                                 extra_time=60.0 + sum(float(z.get('for') or 60.0) for z in sc['zones']))
    rec = FlightLog(sc)
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
