"""
disturbances.py
---------------
D5 dynamic events for SwarmEnv (envs/swarm_env.py): drone failures. Off by default; then nothing
here is used and no random numbers are drawn. (Wind gusts: future work, the policy was not
trained with wind.)

Failures (environment.failures; FAILURE_TYPES) are given per episode as a schedule,
SwarmEnv.start_episode(failures=[{'drone': 'drone_3', 't': 21.4, 'type': 'motor'}, ...]);
draw_failures() draws one, seeded, at random times in the middle of the flights.
"""

import numpy as np

FAILURE_TYPES = ('motor', 'land', 'comms')
FAILURE_DEFAULTS = {'enabled': False, 'land_vz': 0.5, 'land_z': 1.0, 'land_tol': 0.5, 'gravity': 9.81,
                    'fall_v_max': 10.0, 'fall_drag_tau': 1.0, 'p_emergency_report': 1.0, 'seed': 0}


def failure_config(config):
    """environment.failures with defaults filled in, checked; None when off.
        land_vz          forced landing: commanded vz is clipped to +-land_vz (m/s)
        land_z           ... down to this height above the ground or roof under the drone
        land_tol         ... landed once within land_tol of that point and settled
        gravity          motor failure: the drone falls with this acceleration (m/s^2) ...
        fall_v_max       ... up to this vertical speed (m/s) ...
        fall_drag_tau    ... while its horizontal speed decays with this time constant (s)
        p_emergency_report  probability that a drone whose motor fails still sends one report
                         with status 'emergency' (Remote ID's emergency status); SwarmEnv draws it
                         per failure and exposes it in env.emergency_report, the eval sends it
        seed             for that draw (np.random.default_rng([seed, episode]))"""
    raw = (config.get('environment', {}).get('failures') or {})
    if not raw.get('enabled', False):
        return None
    unknown = set(raw) - set(FAILURE_DEFAULTS)
    if unknown:
        raise ValueError(f"environment.failures has unknown keys: {sorted(unknown)}")
    cfg = {**FAILURE_DEFAULTS, **{k: v for k, v in raw.items()}}
    for k in ('land_vz', 'land_z', 'land_tol', 'gravity', 'fall_v_max', 'fall_drag_tau'):
        cfg[k] = float(cfg[k])
        if cfg[k] <= 0:
            raise ValueError(f'environment.failures.{k} must be > 0')
    cfg['p_emergency_report'] = float(cfg['p_emergency_report'])
    cfg['seed'] = int(cfg['seed'])
    if not 0.0 <= cfg['p_emergency_report'] <= 1.0:
        raise ValueError('environment.failures.p_emergency_report must be in [0, 1]')
    return cfg


def check_failures(failures, n_agents):
    """A failure schedule, checked and sorted by time: [{'drone', 't', 'type'}, ...]."""
    out = []
    seen = set()
    for f in failures:
        key, t, kind = f['drone'], float(f['t']), f['type']
        if kind not in FAILURE_TYPES:
            raise ValueError(f'failure type must be one of {FAILURE_TYPES}, got {kind!r}')
        if not key.startswith('drone_') or not 0 <= int(key.split('_')[1]) < n_agents:
            raise ValueError(f'failure for unknown drone {key!r}')
        if key in seen:
            raise ValueError(f'{key} has more than one failure')
        if not t >= 0:
            raise ValueError(f'failure time must be >= 0, got {t}')
        seen.add(key)
        out.append({'drone': key, 't': t, 'type': kind})
    return sorted(out, key=lambda f: (f['t'], int(f['drone'].split('_')[1])))


def draw_failures(rng, flights, p, types=FAILURE_TYPES, frac=(0.2, 0.8)):
    """A seeded failure schedule. flights: [(drone key, t_dep, t_arr), ...] (E1's booked departure
    and arrival); each fails with probability p, at t_dep + u (t_arr - t_dep), u uniform in frac
    (mid-flight), with a type drawn uniformly from types. rng: np.random.RandomState or Generator.
    Draws the same number of random numbers per flight whatever happens (3), so the schedule of
    one flight does not depend on the others'."""
    if not 0.0 <= p <= 1.0 or not types or not 0.0 <= frac[0] <= frac[1] <= 1.0:
        raise ValueError('draw_failures: p in [0, 1], types non-empty, 0 <= frac[0] <= frac[1] <= 1')
    out = []
    for key, t_dep, t_arr in flights:
        hit, u, k = rng.uniform(), rng.uniform(*frac), rng.uniform()
        if hit < p:
            kind = types[min(int(k * len(types)), len(types) - 1)]
            out.append({'drone': key, 't': float(t_dep + u * (t_arr - t_dep)), 'type': kind})
    return out


def roof_under(layout, x, y, margin):
    """Height of the highest obstacle within margin (m) of (x, y) horizontally, 0 if none."""
    from envs.scenario_manager import surface_distance
    top = 0.0
    for ob in layout:
        if surface_distance(ob, (x, y, 0.0)) < margin:
            top = max(top, float(ob['height']))
    return top
