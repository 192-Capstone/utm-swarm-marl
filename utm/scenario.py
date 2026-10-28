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
      n_drops: 8
    disruptions: {p_late: 0.25, late: [5, 15], p_slow: 0.25, slow: 0.6}
    intruders: 2                     # a count (random trips), or a list of {t, start, goal}
    zones:                           # no-fly zones appearing mid-flight (zone: one of them, same keys)
      - {at: 15, place: busiest, for: 60}        #   where it cuts the most routes (or random), 60 s
      - {at: 40, polygon: [[x, y], ...]}         #   a given outline (local m), permanent
    emergencies:                     # emergency flights: filed at t with priority (E4 makes room)
      - {t: 30, start: [x, y], goal: [x, y], priority: 1}
    utm:  {mode: e4, risk: booking, telemetry: null,   # mode: no_e1 | e1 | e4; telemetry [noise, delay, loss]
           pad_after: 10}                    # s E1 holds a landing pad after arrival (E1 default 7:
                                             #   PyBullet's p99 landing; Webots lands ~2 s later)
    sim:  {config: ..., checkpoint: ..., pilot: learned, simulator: pybullet,   # or webots_l2_multi (C5)
           land_latch_m: 2.4, land_vz_max: 0.5,          # C5's landing options (null: off)
           land_wait_clear_m: 1.5}           # hold above the pad while another drone is low over it
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
    'utm': {'mode': 'e4', 'risk': 'booking', 'telemetry': None, 'pad_after': 10.0},
    'sim': {'config': 'configs/config_stage2_round7_eval_goalhold.yaml',
            'checkpoint': '../utm-swarm-marl/checkpoints/stage2_r1_crowd_rl_r7v3ft_s0_lab/checkpoint_final.pth',
            'pilot': 'learned', 'simulator': 'pybullet', 'land_latch_m': 2.4, 'land_vz_max': 0.5,
            'land_wait_clear_m': 1.5},
    'record': None,
}
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
    if 'flights' not in d and (d.get('rate', 0) <= 0 or d.get('duration_s', 0) <= 0):
        bad('demand needs flights (+ window) or rate > 0 and duration_s > 0')
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
        if not {'t', 'start', 'goal'} <= set(e) or int(e.get('priority', 1)) <= 0:
            bad('an emergency needs t, start, goal (and priority > 0)')
    if sc['emergencies'] and sc['utm']['mode'] == 'no_e1':
        bad('emergencies need E1 (utm.mode e1 or e4)')
    ix = sc['intruders']
    if not isinstance(ix, int) and not all({'t', 'start', 'goal'} <= set(x) for x in ix):
        bad('intruders: a count, or a list of {t, start, goal}')
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
