"""
birds.py
--------
Birds: small, fast, non-cooperative moving obstacles (opt-in: environment.birds, off by default).

A bird is a sphere that flies a pre-computed path and never reacts to drones. Every path passes
through an AIM POINT on a drone's planned path at about the time that drone is expected there, so
an episode always has an encounter, however short the flight (a bird that starts 10 m out at 1 m/s
would otherwise arrive long after a 5 s flight is over):

    aim point   a random point on one drone's path (its straight start-goal line, or the route
                the drone follows), 20-100 % of the way along
    aim time    departure + (arc length to the aim point) / drone_speed + U(-jitter, +jitter) s
    path        generated from the aim point, so the bird is at the aim point at the aim time:
                  cross   straight line, random direction
                  turn    constant speed, gentle random turns (heading rate <= max_turn_rate)
                  circle  a circle of radius circle_radius through the aim point
                a small random vertical speed (|vz| <= vz_max) in all modes; x, y and z are
                reflected at the bounds / the altitude band, so a bird stays inside them

Scale note: real birds fly 8-20 m/s. The drones fly at 2 m/s with a 6 m LiDAR and 0.10-0.29 s of
lag, so a real-speed head-on bird cannot be dodged reactively by any controller: speeds here
(default 1-3 m/s) are scaled to the simulation.

Determinism: the birds of one episode come from their own random generator, seeded with
[environment.birds.seed, hash of the drones' spawns and paths]. Nothing is drawn from np.random,
so the same episode flies the same drones, noise and lag with and without birds, and bird i is
the same bird whatever the count.

Config (environment.birds; off when the block is missing or enabled is false):
    birds:
      enabled: true
      count: 2
      speed: [1.0, 3.0]            # m/s, drawn per bird
      radius: 0.12                 # m
      modes: [cross, turn, circle]
      seed: 0
      drone_speed: 2.0             # m/s, what the aim time assumes
      jitter_s: 1.0                # aim time +- this
      z_band: null                 # [lo, hi] m; default: the paths' heights +- 0.5 m
      bounds: null                 # [[xmin, xmax], [ymin, ymax]]; default: the paths' box +- margin
      margin: 12.0                 # m, for the default bounds
      circle_radius: [2.0, 4.0]    # m
      max_turn_rate: 0.35          # rad/s (turn mode)
      vz_max: 0.3                  # m/s
"""

import hashlib

import numpy as np

MODES = ('cross', 'turn', 'circle')

DEFAULTS = {
    'enabled': False, 'count': 0, 'speed': [1.0, 3.0], 'radius': 0.12,
    'modes': list(MODES), 'seed': 0, 'drone_speed': 2.0, 'jitter_s': 1.0,
    'z_band': None, 'bounds': None, 'margin': 12.0, 'circle_radius': [2.0, 4.0],
    'max_turn_rate': 0.35, 'vz_max': 0.3,
}


def validate_block(block):
    """The block with the defaults filled in; ValueError for a bad one."""
    block = dict(block or {})
    unknown = set(block) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"birds: unknown settings {sorted(unknown)}")
    cfg = {**DEFAULTS, **block}
    if isinstance(cfg['count'], bool) or not isinstance(cfg['count'], int) or cfg['count'] < 0:
        raise ValueError(f"birds.count must be an integer >= 0, got {cfg['count']!r}")
    for k in ('speed', 'circle_radius'):
        v = cfg[k]
        if not (isinstance(v, (list, tuple)) and len(v) == 2 and 0 < v[0] <= v[1]):
            raise ValueError(f"birds.{k} must be [lo, hi] with 0 < lo <= hi, got {v!r}")
    for k in ('radius', 'drone_speed', 'max_turn_rate'):
        if not cfg[k] > 0:
            raise ValueError(f"birds.{k} must be positive")
    if cfg['jitter_s'] < 0 or cfg['vz_max'] < 0 or cfg['margin'] < 0:
        raise ValueError("birds.jitter_s, vz_max and margin must be >= 0")
    if not cfg['modes'] or any(m not in MODES for m in cfg['modes']):
        raise ValueError(f"birds.modes must be a non-empty list of {MODES}, got {cfg['modes']!r}")
    if cfg['z_band'] is not None and not (len(cfg['z_band']) == 2 and cfg['z_band'][0] < cfg['z_band'][1]):
        raise ValueError("birds.z_band must be [lo, hi]")
    return cfg


def config(full_config):
    """environment.birds, validated; None when birds are off."""
    block = (full_config.get('environment') or {}).get('birds')
    if not block or not block.get('enabled', False):
        return None
    cfg = validate_block(block)
    return cfg if cfg['count'] > 0 else None


def _fold(x, lo, hi):
    """Reflect x into [lo, hi] (a triangle wave): identity inside."""
    w = hi - lo
    y = np.mod(np.asarray(x, dtype=float) - lo, 2.0 * w)
    return lo + np.where(y > w, 2.0 * w - y, y)


class Bird:
    """One bird: position(k) is where it is after k control steps."""

    def __init__(self, table_fn, radius, mode='custom', speed=0.0, aim=None, t_aim=None):
        self._fn = table_fn           # n -> (n, 3) positions of steps 0..n-1
        self._table = table_fn(256)
        self.radius = float(radius)
        self.mode, self.speed = mode, float(speed)
        self.aim = None if aim is None else np.asarray(aim, dtype=float)
        self.t_aim = t_aim

    def position(self, k):
        while k >= len(self._table):
            self._table = self._fn(2 * len(self._table))
        return self._table[k]

    @classmethod
    def straight(cls, p0, v, radius, dt):
        """A bird flying a straight line from p0 at velocity v (m/s): for tests and hand-built scenes."""
        p0, v = np.asarray(p0, dtype=float), np.asarray(v, dtype=float)
        return cls(lambda n: p0 + np.arange(n)[:, None] * dt * v, radius, 'straight', float(np.linalg.norm(v)))

    @classmethod
    def planned(cls, rng, cfg, aim, t_aim, dt, mode, z_band, bounds):
        """A bird of this mode that is at `aim` (x, y, z) at time t_aim."""
        speed = float(rng.uniform(*cfg['speed']))
        theta0 = float(rng.uniform(0.0, 2.0 * np.pi))
        vz = 0.0 if mode == 'circle' else float(rng.uniform(-cfg['vz_max'], cfg['vz_max']))
        if mode == 'circle':
            R = float(rng.uniform(*cfg['circle_radius']))
            rate0, (a1, a2), (T1, T2), (p1, p2) = float(rng.choice([-1.0, 1.0])) * speed / R, (0.0, 0.0), (1.0, 1.0), (0.0, 0.0)
        elif mode == 'turn':
            rate0 = 0.0
            a1, a2 = (float(x) for x in rng.uniform(0.3, 0.5, 2) * cfg['max_turn_rate'])
            T1, T2 = (float(x) for x in rng.uniform(4.0, 10.0, 2))
            p1, p2 = (float(x) for x in rng.uniform(0.0, 2.0 * np.pi, 2))
        else:
            rate0, (a1, a2), (T1, T2), (p1, p2) = 0.0, (0.0, 0.0), (1.0, 1.0), (0.0, 0.0)
        ka = max(1, int(round(t_aim / dt)))
        (xlo, xhi), (ylo, yhi) = bounds
        zlo, zhi = z_band

        def raw(n):
            t = np.arange(n) * dt
            rate = rate0 + a1 * np.sin(2 * np.pi * t / T1 + p1) + a2 * np.sin(2 * np.pi * t / T2 + p2)
            th = theta0 + np.cumsum(rate) * dt
            step = speed * dt * np.stack([np.cos(th), np.sin(th)], axis=1)
            return np.vstack([[0.0, 0.0], np.cumsum(step, axis=0)[:-1]]), t

        xy_aim = raw(ka + 1)[0][ka]
        offset = np.asarray(aim[:2], dtype=float) - xy_aim

        def table(n):
            n = max(n, ka + 1)
            xy, t = raw(n)
            xy = xy + offset
            z = aim[2] + vz * (t - ka * dt)
            return np.stack([_fold(xy[:, 0], xlo, xhi), _fold(xy[:, 1], ylo, yhi), _fold(z, zlo, zhi)], axis=1)

        return cls(table, cfg['radius'], mode, speed, aim, ka * dt)


class BirdField:
    """The birds of one episode; step() moves them on by one control step."""

    def __init__(self, birds, dt):
        self.birds, self.dt, self.k = list(birds), float(dt), 0
        self.radii = np.array([b.radius for b in self.birds], dtype=float)

    def __len__(self):
        return len(self.birds)

    def positions(self, k=None):
        k = self.k if k is None else k
        return np.array([b.position(k) for b in self.birds], dtype=float).reshape(-1, 3)

    def step(self):
        self.k += 1
        return self.positions()

    @classmethod
    def generate(cls, cfg, paths, dt, seed_parts=(), t_dep=None):
        """cfg: validated environment.birds; paths: one (m, 3) array per drone (its straight line or
        route); t_dep: departure time per drone (s, default 0)."""
        paths = [np.asarray(p, dtype=float) for p in paths]
        t_dep = np.zeros(len(paths)) if t_dep is None else np.asarray(t_dep, dtype=float)
        digest = hashlib.sha1(np.concatenate([np.round(p, 3).ravel() for p in paths]).tobytes()).digest()
        rng = np.random.default_rng([int(cfg['seed']), int.from_bytes(digest[:8], 'little'), *map(int, seed_parts)])
        allp = np.vstack(paths)
        if cfg['z_band'] is not None:
            z_band = tuple(float(z) for z in cfg['z_band'])
        else:
            z_band = (max(0.3, float(allp[:, 2].min()) - 0.5), float(allp[:, 2].max()) + 0.5)
        if cfg['bounds'] is not None:
            bounds = tuple(tuple(float(v) for v in b) for b in cfg['bounds'])
        else:
            lo, hi = allp[:, :2].min(axis=0) - cfg['margin'], allp[:, :2].max(axis=0) + cfg['margin']
            bounds = ((float(lo[0]), float(hi[0])), (float(lo[1]), float(hi[1])))
        seg = [np.linalg.norm(np.diff(p[:, :2], axis=0), axis=1) for p in paths]
        total = np.array([s.sum() for s in seg])
        birds = []
        for _ in range(cfg['count']):                      # one draw sequence per bird: bird i never depends on the count
            d = int(rng.integers(len(paths)))
            frac = float(rng.uniform(0.2, 1.0))
            mode = str(rng.choice(cfg['modes']))
            jit = float(rng.uniform(-cfg['jitter_s'], cfg['jitter_s']))
            s = frac * total[d]
            cum = np.concatenate([[0.0], np.cumsum(seg[d])])
            aim = np.array([np.interp(s, cum, paths[d][:, c]) for c in range(3)])
            t_aim = max(0.5, t_dep[d] + s / cfg['drone_speed'] + jit)
            birds.append(Bird.planned(rng, cfg, aim, t_aim, dt, mode, z_band, bounds))
        return cls(birds, dt)
