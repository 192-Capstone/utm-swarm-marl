"""
scenario.py
-----------
E6: a scenario file describes one end-to-end run of the system (scripts/run_scenario.py):
city -> demand -> E1 plan -> fly (learned policy) -> E3 / E2 monitor -> E4 -> metrics, recorded
for the dashboard. YAML, every key optional except map (defaults below).

    name: ecity_demo
    seed: 20                         # demand, disruptions, intruders, zone placement
    map:  {map: osm-crop}            # or {map: city, level: medium, seed: 3}
    demand:                          # D4 (utm/demand.py DemandSettings: any of its fields), or
      rate: 10                       #   {flights: 20, window: 30} for E1's random requests
      duration_s: 30
      mode: hubs                     #   random | hubs (n_hubs depots, n_drops drop points)
      n_hubs: 2
      n_drops: 8                     #   or pads: [[x, y, hub], [x, y, drop], ...] for fixed sites
                                     #   ([x, y, hub, op0]: op0's depot, its flights are op0's)
    places:                          # named sites, used by name below (and on the dashboard)
      West Hub: {kind: hub, operator: op0, pads: [[-28.8, 13.7], [-22.8, 13.7]]}   # op0's depot
      Clinic:   {kind: drop, at: [35.4, 90.7]}       # kind: drop | hospital | any word
    requests:                        # instead of demand: each client node's bookings, in order
      - {t: 5, operator: op0, to: Clinic}            # from a free pad of op0's hub (or from: a place,
                                     #   pad: 0-based pad of a hub); a place name or [x, y]
    disruptions: {p_late: 0.25, late: [5, 15], p_slow: 0.25, slow: 0.6}
    intruders: 2                     # a count (random trips), or a list of {t, start, goal}
    zones:                           # no-fly zones appearing mid-flight (zone: one of them, same keys)
      - {at: 15, place: busiest, for: 60}        #   where it cuts the most routes (or random), 60 s
      - {at: 40, polygon: [[x, y], ...]}         #   a given outline (local m), permanent
    emergencies:                     # emergency flights: filed at t with priority (E4 makes room)
      - {t: 30, start: [x, y], goal: [x, y], priority: 1}
    utm:  {mode: e4, risk: booking, telemetry: null,   # mode: no_e1 | e1 | e4; telemetry [noise, delay, loss]
           pad_after: 10,                    # s E1 holds a landing pad after arrival (E1 default 7:
                                             #   PyBullet's p99 landing; Webots lands ~2 s later)
           layer_floor: 1,                   # m: the lowest layer's bottom (layers 2 m thick); 13 puts
                                             #   E-City's layers at 14 / 16 m, over its 12 m roofs
           max_delay: null,                  # s a request may wait for a slot (null: E1's default, 60)
           n_layers: 2, layers_open: null,   # layers (2 m each from layer_floor); the ones flights may use
                                             #   (null: all; leave out the ones among the roofs)
           layer_penalty: null,              # s per layer away from the preferred one (null: E1's 4 s;
                                             #   with far-apart layers a climb costs more than that)
           v_climb: null, v_descent: null,   # the profile E1 books (null: 0.5 m/s up and down);
           v_descent_high: null, slow_band_m: null}   #   with v_descent_high: fast while high, then
                                             #   v_descent for the last slow_band_m (sim: land_vz_high)
                                             # traffic_detour: E1 also tries a same-layer route around
                                             #   other flights' air (booked when it would pass) before
                                             #   changing layer or waiting
    sim:  {config: ..., checkpoint: ..., pilot: learned, simulator: pybullet,   # or webots_l2_multi (C5)
           land_latch_m: 2.4, land_vz_max: 0.5,          # C5's landing options (null: off)
           land_wait_clear_m: 1.5,           # hold above the pad while another drone is low over it
           cruise_z_bias: 0, altitude_cap: null, land_step_m: null}   # for cruise above ~6 m: 0.9, 3.5, 2
                                             #   (scripts/altitude_test.py: aim above the policy's ~0.8 m
                                             #   sag, keep its own-height input in its training range,
                                             #   descend in 2 m steps)
                                             # position_origin: goal on maps bigger than the 50 m
                                             #   training arena (own x, y input read from the carrot)
                                             # hold_engage_m: goal-hold takes over this close (default 0.5;
                                             #   1.0 when drones settle just outside 0.5 m over a pad)
                                             # land_vz_high: m/s descending while more than land_slow_m
                                             #   (default 3) over the pad; land_vz_max only below that
                                             # los_below_cruise: a roof over 1 m below a drone's cruise
                                             #   height doesn't block its line of sight (flying over roofs)
    failures:                        # D5: drone failures (utm.mode e1 or e4, PyBullet), absent = none
      p: 0.2                         #   drawn: each approved flight fails with probability p, mid-flight,
      types: [motor, land, comms]    #   its type drawn from types (own RNG: 8000 + seed, as eval_layered)
    # or listed: [{flight: f3, t: 30, type: motor}, ...]    (t: s, flight: an id of the run's flights)
    record: eval_results/scenarios/ecity_demo.json.gz

Seeds 10-19 are refused (reserved for test runs), as in the evaluation scripts.
"""

import copy

import yaml

DEFAULTS = {
    'name': 'scenario',
    'seed': 20,
    'demand': {'rate': 10, 'duration_s': 30, 'mode': 'random'},
    'disruptions': {'p_late': 0.0, 'late': [5.0, 15.0], 'p_slow': 0.0, 'slow': 0.6},
    'intruders': 0,
    'zone': None,
    'zones': [],
    'emergencies': [],
    'places': {},
    'requests': [],
    'utm': {'mode': 'e4', 'risk': 'booking', 'telemetry': None, 'pad_after': 10.0, 'layer_floor': 1.0, 'max_delay': None,
            'n_layers': 2, 'layers_open': None, 'layer_penalty': None,
            'v_climb': None, 'v_descent': None, 'v_descent_high': None, 'slow_band_m': None,
            'traffic_detour': False},
    'sim': {'config': 'configs/config_stage2_round7_eval_goalhold.yaml',
            'checkpoint': '../utm-swarm-marl/checkpoints/stage2_r1_crowd_rl_r7v3ft_s0_lab/checkpoint_final.pth',
            'pilot': 'learned', 'simulator': 'pybullet', 'land_latch_m': 2.4, 'land_vz_max': 0.5,
            'land_wait_clear_m': 1.5, 'cruise_z_bias': 0.0, 'altitude_cap': None, 'land_step_m': None,
            'position_origin': None, 'hold_engage_m': None, 'land_vz_high': None, 'land_slow_m': None,
            'los_below_cruise': False},
    'record': None,
}
FAILURE_TYPES = ('motor', 'land', 'comms')        # envs/disturbances.py FAILURE_TYPES
MODES, RISKS, PILOTS = ('no_e1', 'e1', 'e4'), ('booking', 'intent'), ('simple', 'blind', 'learned')
SIMULATORS = ('pybullet', 'webots_l2_multi')


def _merge(base, over):
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def validate(sc):
    """Raises ValueError naming the first problem."""
    def bad(msg):
        raise ValueError(f"scenario {sc.get('name')!r}: {msg}")
    if not isinstance(sc.get('map'), dict) or sc['map'].get('map') not in ('city', 'osm-crop', 'osm-full'):
        bad("map.map must be city, osm-crop or osm-full")
    seeds = [sc['seed']] + ([sc['map']['seed']] if 'seed' in sc['map'] else [])
    if any(10 <= int(s) <= 19 for s in seeds):
        bad('seeds 10-19 are reserved for test runs')
    d = sc['demand']
    if not sc['requests'] and 'flights' not in d and (d.get('rate', 0) <= 0 or d.get('duration_s', 0) <= 0):
        bad('demand needs flights (+ window) or rate > 0 and duration_s > 0 (or give requests)')
    places = sc['places']
    for name, p in places.items():
        if p.get('kind') == 'hub':
            if not p.get('operator') or not p.get('pads'):
                bad(f'place {name!r}: a hub needs operator and pads [[x, y], ...]')
        elif 'at' not in p:
            bad(f'place {name!r} needs at: [x, y]')
    hubs = {p['operator'] for p in places.values() if p.get('kind') == 'hub'}
    for r in sc['requests']:
        if not {'t', 'operator', 'to'} <= set(r):
            bad('a request needs t, operator and to')
        for k in ('to', 'from'):
            if isinstance(r.get(k), str) and r[k] not in places:
                bad(f'request: unknown place {r[k]!r}')
        if 'from' not in r and r['operator'] not in hubs:
            bad(f"request: {r['operator']} has no hub to fly from (give from:)")
    if sc['utm']['mode'] not in MODES:
        bad(f'utm.mode must be one of {MODES}')
    if sc['utm']['risk'] not in RISKS:
        bad(f'utm.risk must be one of {RISKS}')
    tel = sc['utm'].get('telemetry')
    if tel is not None and len(tel) != 3:
        bad('utm.telemetry must be [noise_m, delay_s, loss] or null')
    if sc['sim']['pilot'] not in PILOTS:
        bad(f'sim.pilot must be one of {PILOTS}')
    if sc['sim']['simulator'] not in SIMULATORS:
        bad(f'sim.simulator must be one of {SIMULATORS}')
    for z in sc['zones']:
        if 'at' not in z:
            bad('a zone needs at (s)')
        if 'polygon' not in z and z.get('place', 'busiest') not in ('busiest', 'random'):
            bad('a zone needs a polygon or place: busiest | random')
    for e in sc['emergencies']:
        if not ({'t', 'start', 'goal'} <= set(e) or {'t', 'from', 'to'} <= set(e)) or int(e.get('priority', 1)) <= 0:
            bad('an emergency needs t, start, goal or from, to (and priority > 0)')
        for k in ('from', 'to', 'start', 'goal'):
            if isinstance(e.get(k), str) and e[k] not in places:
                bad(f'emergency: unknown place {e[k]!r}')
    if sc['emergencies'] and sc['utm']['mode'] == 'no_e1':
        bad('emergencies need E1 (utm.mode e1 or e4)')
    ix = sc['intruders']
    if not isinstance(ix, int) and not all({'t', 'start', 'goal'} <= set(x) for x in ix):
        bad('intruders: a count, or a list of {t, start, goal}')
    fl = sc.get('failures')                           # not in DEFAULTS: absent stays out of the recording
    if fl is not None:
        if sc['utm']['mode'] not in ('e1', 'e4'):
            bad('failures need the UTM to hear the drones (utm.mode e1 or e4)')
        if sc['sim']['simulator'] != 'pybullet':
            bad('failures are PyBullet only (Webots has no set_falling)')
        if isinstance(fl, dict):
            if set(fl) - {'p', 'types'} or 'p' not in fl:
                bad('failures: {p: 0-1, types: [motor, land, comms]} or a list of {flight, t, type}')
            if isinstance(fl['p'], bool) or not isinstance(fl['p'], (int, float)) or not 0.0 <= fl['p'] <= 1.0:
                bad(f"failures.p must be a number in [0, 1], got {fl['p']!r}")
            types = fl.get('types', list(FAILURE_TYPES))
            if not isinstance(types, (list, tuple)) or not types or any(x not in FAILURE_TYPES for x in types):
                bad(f'failures.types must be a non-empty list of {FAILURE_TYPES}, got {types!r}')
            fl['types'] = list(types)
        elif isinstance(fl, list):
            ids = {f'f{k}' for k in range(len(sc['requests']))} if sc['requests'] else None   # else known at run time
            for f in fl:
                if not isinstance(f, dict) or set(f) != {'flight', 't', 'type'}:
                    bad('a failure needs flight, t and type (and nothing else)')
                if f['type'] not in FAILURE_TYPES:
                    bad(f"failure type {f['type']!r} must be one of {FAILURE_TYPES}")
                if isinstance(f['t'], bool) or not isinstance(f['t'], (int, float)) or f['t'] < 0:
                    bad(f"failure t must be a number >= 0, got {f['t']!r}")
                if ids is not None and f['flight'] not in ids:
                    bad(f"failure for unknown flight {f['flight']!r} (this scenario's flights: f0..f{len(ids) - 1})")
            if len({f['flight'] for f in fl}) < len(fl):
                bad('at most one failure per flight')
        else:
            bad('failures: {p, types} or a list of {flight, t, type}')
    return sc


def load(path_or_dict):
    """A scenario (dict) from a YAML file or a dict, with the defaults filled in and checked."""
    if isinstance(path_or_dict, dict):
        raw = path_or_dict
    else:
        with open(path_or_dict, encoding='utf-8') as f:
            raw = yaml.safe_load(f) or {}
    sc = _merge(DEFAULTS, raw)
    if sc.get('zone'):                                 # one zone: shorthand for a one-item list
        sc['zones'] = [sc['zone']] + list(sc['zones'])
    sc.pop('zone', None)
    return validate(sc)
