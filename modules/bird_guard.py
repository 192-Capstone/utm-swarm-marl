"""
bird_guard.py
-------------
Stage 3a+: a classical bird-avoidance layer, in the style of goal-hold. The learned policy reacts to a
bird in a neighbour slot too weakly (docs/birds_report.md, stage 3a), so, as the project already hands the
final stop to a classical controller, a small rule layer takes over the velocity command only while a bird
is a threat. Order: policy -> goal-hold -> bird guard (the guard has the final say only while active).

Honest inputs: only the detector's TRACKED birds (envs/bird_detector.py: its own range, field of view,
noise, latency and occlusion), the drone's own noisy state (what the policy sees), the LiDAR ring and the
other drones' shared states. The guard never sees a bird's true position (a bird the detector has not
tracked triggers nothing). The oracle detector can be used as an upper bound in the evaluation.

Per flying drone, every step:
    threat   closest point of approach (the teacher's, baselines/repulsion_controller.cpa) between the drone
             and a tracked bird, both moved predict_s (0.4 s: the lag, as the teacher does) ahead, both
             keeping their velocities: a threat if the predicted miss is under miss_m within horizon_s and
             the bird is closing.
    evasion  while active: eight candidate velocities, the policy's command plus evade_speed (1.5 m/s) sideways
             (perpendicular to the relative motion, either side), up, down, or a diagonal of those. A
             candidate scores the smallest of
                 predicted miss to the birds,
                 predicted miss to the other DRONES (their shared states, never the neighbour slots, which may
                 hold birds) + drone_margin_m,
                 the LiDAR ring distance in the candidate's horizontal direction (obstacles).
             Vertical candidates are left out at the edge of the altitude band: centred on the drone's current
             target height (the carrot's for routed flights, the goal's otherwise), half-width
             min(climb_m, layer_frac * layer_spacing_m) when the scenario has layers, climb_m otherwise, and
             never below z_floor_m. The best candidate replaces the command (per axis capped at v_max).
    safety   while active the teacher's push-away from drones inside close_dist and from LiDAR obstacle hits
             inside obs_close is added, so the guard cannot walk a drone into a neighbour or a wall.
    release  hysteresis: the guard stays active until, for release_steps steps, every tracked bird has passed
             (opening) or its predicted miss is over release_m.

Config (environment.birds.guard; off when the block is missing or enabled is false; needs
environment.birds.detector):
    guard:
      enabled: true
      horizon_s: 3.0         # look this far ahead
      miss_m: 1.0            # a threat below this predicted miss
      release_m: 1.4         # released above this (and when the bird is opening)
      release_steps: 5
      predict_s: null        # lag look-ahead; default: the teacher's predict_s, else 0.4
      evade_speed: 1.5       # m/s added to the command
      climb_m: 1.5           # vertical half-width of the altitude band
      layer_spacing_m: null  # set by scenarios with layers (2 m); then half-width <= layer_frac * this
      layer_frac: 0.45
      z_floor_m: 0.8
      drone_margin_m: 0.3
      brake: []              # stage 3a++: extra candidates that scale the command (e.g. [0.0, 0.5]: hover, half speed),
                             # scored like the others
      turn_s: 0              # s, off at 0: a bird that is turning is predicted along a curve, not a straight line. Its
                             # turn rate is the slope of its track's heading over the last turn_s (from half of it on);
                             # needs tracks that keep their bird (detector.share off)
      turn_lag_s: 0.3        # s the tracker's heading lags a turning bird (the heading is moved on by this)
      turn_min: 0.2          # rad/s, a slower turn counts as straight (the tracks are noisy)
"""

import numpy as np

from baselines.repulsion_controller import RepulsionController, cpa
from envs import lidar as lidar_mod

DEFAULTS = {
    'enabled': False, 'horizon_s': 3.0, 'miss_m': 1.0, 'release_m': 1.4, 'release_steps': 5, 'predict_s': None,
    'evade_speed': 1.5, 'climb_m': 1.5, 'layer_spacing_m': None, 'layer_frac': 0.45, 'z_floor_m': 0.8,
    'drone_margin_m': 0.3, 'brake': [], 'turn_s': 0.0, 'turn_lag_s': 0.3, 'turn_min': 0.2,
}
PARKED_Z = 1.0e3          # parked drones sit far away (adapter.park)
DRONE_NEAR_M = 8.0        # other drones further than this do not matter within the horizon
RING_CONE = np.cos(np.radians(25.0))
TURN_MAX = 1.6            # rad/s, the fastest turn believed
TURN_SPEED = 0.3          # m/s, a slower bird has no heading to speak of
PATH_DT = 0.1             # s between the points of a predicted curve


def validate(block):
    """The guard block with the defaults filled in, or None when it is missing / not enabled."""
    if not block:
        return None
    block = dict(block)
    unknown = set(block) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"birds.guard: unknown settings {sorted(unknown)}")
    cfg = {**DEFAULTS, **block}
    if not cfg['enabled']:
        return None
    for k in ('horizon_s', 'miss_m', 'evade_speed', 'climb_m'):
        if not cfg[k] > 0:
            raise ValueError(f"birds.guard.{k} must be positive")
    if cfg['release_m'] < cfg['miss_m']:
        raise ValueError("birds.guard.release_m must be >= miss_m")
    if cfg['release_steps'] < 1 or cfg['z_floor_m'] < 0 or cfg['drone_margin_m'] < 0:
        raise ValueError("birds.guard: release_steps >= 1, z_floor_m >= 0, drone_margin_m >= 0")
    if cfg['layer_spacing_m'] is not None and not cfg['layer_spacing_m'] > 0:
        raise ValueError("birds.guard.layer_spacing_m must be positive or null")
    if not 0 < cfg['layer_frac'] <= 0.5:
        raise ValueError("birds.guard.layer_frac must be in (0, 0.5]")
    if cfg['turn_s'] < 0 or cfg['turn_lag_s'] < 0 or cfg['turn_min'] < 0:
        raise ValueError("birds.guard: turn_s, turn_lag_s and turn_min must be >= 0")
    cfg['release_steps'] = int(cfg['release_steps'])
    cfg['brake'] = [float(x) for x in (cfg['brake'] or [])]
    if any(not 0.0 <= x < 1.0 for x in cfg['brake']):
        raise ValueError("birds.guard.brake: scales must be in [0, 1)")
    return cfg


def guard_enabled(config):
    from envs import birds as birds_mod
    b = birds_mod.config(config)
    return b is not None and b['guard'] is not None


def band_half_width(cfg):
    """Vertical half-width of the evasion band (m): a layered scenario must not push a drone into another
    layer's air (layers are 2 m apart, so 0.9 m), otherwise climb_m."""
    if cfg['layer_spacing_m'] is not None:
        return min(cfg['climb_m'], cfg['layer_frac'] * cfg['layer_spacing_m'])
    return cfg['climb_m']


# the eight candidate evasions: (sideways, vertical) weights, sideways +1 = right of the relative motion
CANDIDATES = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]


class BirdGuardPolicy:
    def __init__(self, inner, config):
        from envs import birds as birds_mod
        b = birds_mod.config(config)
        if b is None or b['guard'] is None:
            raise ValueError("BirdGuardPolicy needs environment.birds.guard enabled")
        if b['detector'] is None:
            raise ValueError("environment.birds.guard needs environment.birds.detector (the guard uses tracked birds only)")
        self.cfg = b['guard']
        rc = RepulsionController(config)                      # the teacher's margins and lag look-ahead
        if self.cfg['turn_s'] > 0 and b['detector']['share']:
            raise ValueError("birds.guard.turn_s needs tracks that keep their bird (birds.detector.share off)")
        self.dt = float(config['environment'].get('control_timestep', 0.032))
        self.turn_steps = int(round(self.cfg['turn_s'] / self.dt))
        self.times = np.arange(0.0, self.cfg['horizon_s'] + 1e-9, PATH_DT)
        self.heading = {}             # (drone, bird) -> the unwrapped headings of its track, newest last
        self.predict_s = float(self.cfg['predict_s'] if self.cfg['predict_s'] is not None
                               else (rc.predict_s if rc.predict_s > 0 else 0.4))
        self.close_dist, self.k_close = rc.close_dist, rc.k_close
        self.obs_close, self.k_obs_close = rc.obs_close, rc.k_obs_close
        self.v_max = float(config['environment'].get('v_max', 2.0))
        self.half_width = band_half_width(self.cfg)
        self.inner = inner
        self._episode = None
        self.active = None
        self.quiet = None
        self.skip_idle = False        # True (the live server): nothing tracked and nothing active -> hand the command back at once
        self.guard_stats = {'drone_steps': 0, 'active_steps': 0, 'activations': 0}
        self.ep = {'drone_steps': 0, 'active_steps': 0, 'activated': 0}

    def __getattr__(self, name):
        if name == 'inner':
            raise AttributeError(name)
        return getattr(self.inner, name)

    # ----------------------------------------------------------------- episode
    def _start_episode_if_new(self, env, n):
        episode = getattr(env, 'episode_index', None)
        if self.active is None or len(self.active) != n or episode != self._episode:
            self.active = np.zeros(n, dtype=bool)
            self.quiet = np.zeros(n, dtype=np.int64)
            self.heading = {}
            self.ep = {'drone_steps': 0, 'active_steps': 0, 'activated': 0}
            self._episode = episode

    def pop_episode(self):
        """This episode's guard numbers (eval_suite adds them to the episode row)."""
        return {'guard_drone_steps': self.ep['drone_steps'], 'guard_active_steps': self.ep['active_steps'],
                'guard_activated': self.ep['activated']}

    # --------------------------------------------------------------------- act
    def act(self, obs, env, n_agents):
        actions = self.inner.act(obs, env, n_agents)
        self._start_episode_if_new(env, n_agents)
        tracks = getattr(env, '_bird_tracks', None)
        raw = getattr(env, 'last_obs_raw', None)
        if tracks is None or raw is None:
            return actions
        known, tpos, tvel = tracks
        if self.skip_idle and not known.any() and not self.active.any():       # (the step counters then stay at rest too)
            return actions
        keys = [f'drone_{i}' for i in range(n_agents)]
        pos = np.array([raw[k]['position'] for k in keys], dtype=float)
        vel = np.array([raw[k]['velocity'] for k in keys], dtype=float)
        flying = pos[:, 2] < PARKED_Z
        tau = self.predict_s
        for i in range(n_agents):
            if not flying[i]:
                self.active[i] = False
                continue
            self.ep['drone_steps'] += 1
            self.guard_stats['drone_steps'] += 1
            b = np.flatnonzero(known[i])
            p_now = pos[i] + vel[i] * tau
            bp = np.asarray(tpos[i, b], dtype=float) + np.asarray(tvel[i, b], dtype=float) * tau
            bv = np.asarray(tvel[i, b], dtype=float)
            paths = None
            if self.turn_steps:
                bp, bv, paths = self._curves(i, b, known.shape[1], np.asarray(tpos[i, b], dtype=float), bv)
            threat, opening_or_safe, worst = self._assess(p_now, vel[i], bp, bv, paths)
            if threat:
                if not self.active[i]:
                    self.guard_stats['activations'] += 1
                    self.ep['activated'] = 1
                self.active[i], self.quiet[i] = True, 0
            elif self.active[i]:
                self.quiet[i] = self.quiet[i] + 1 if opening_or_safe else 0
                if self.quiet[i] >= self.cfg['release_steps']:
                    self.active[i] = False
            if not self.active[i]:
                continue
            self.ep['active_steps'] += 1
            self.guard_stats['active_steps'] += 1
            if len(bp) == 0:                          # latched, but the track has dropped: nothing to evade (it releases soon)
                continue
            a = [float(x) for x in actions[keys[i]]]
            a[:3] = self._command(i, env, keys[i], np.array(a[:3], dtype=float), pos, vel, p_now, vel[i], bp, bv, worst, raw, paths)
            actions[keys[i]] = a
        return actions

    # ---------------------------------------------------------------- turning birds
    def _curves(self, i, b, n_birds, tpos, tvel):
        """The tracked birds b of drone i, turning as their tracks have been: (positions predict_s from now, velocities
        there, their paths over the horizon from there: (birds, times, 3))."""
        for q in range(n_birds):
            if q not in b:
                self.heading.pop((i, q), None)               # the track dropped: a new one starts afresh
        tau, bp, bv = self.predict_s, np.array(tpos), np.array(tvel)
        paths = np.zeros((len(b), len(self.times), 3))
        for j, q in enumerate(b):
            v, speed, rate = tvel[j], float(np.hypot(*tvel[j][:2])), 0.0
            if speed < TURN_SPEED:
                self.heading.pop((i, q), None)
            else:
                h = self.heading.setdefault((i, q), [])
                th = float(np.arctan2(v[1], v[0]))
                h.append(th if not h else h[-1] + (th - h[-1] + np.pi) % (2 * np.pi) - np.pi)
                del h[:-self.turn_steps - 1]
                if len(h) > self.turn_steps // 2:
                    x = np.arange(len(h)) * self.dt
                    x = x - x.mean()
                    rate = float(np.clip(x @ np.asarray(h) / (x @ x), -TURN_MAX, TURN_MAX))
                    if abs(rate) < self.cfg['turn_min']:
                        rate = 0.0
            th = float(np.arctan2(v[1], v[0])) + rate * self.cfg['turn_lag_s']

            def at(t):
                if rate == 0.0:
                    d = np.outer(t, v[:2])
                else:
                    d = speed / rate * np.stack([np.sin(th + rate * t) - np.sin(th), np.cos(th) - np.cos(th + rate * t)], 1)
                return tpos[j] + np.column_stack([d, v[2] * t])
            paths[j] = at(self.times + tau)
            bp[j] = paths[j, 0]
            if rate != 0.0:
                bv[j, :2] = speed * np.cos(th + rate * tau), speed * np.sin(th + rate * tau)
        return bp, bv, paths

    def _miss_paths(self, p_now, v, paths):
        """Predicted miss to each bird on its path, and when: (misses, times, distances now)."""
        d = np.linalg.norm(p_now + self.times[:, None] * v - paths, axis=2)
        k = d.argmin(axis=1)
        return d[np.arange(len(d)), k], self.times[k], d[:, 0]

    # ------------------------------------------------------------------ threat
    def _assess(self, p_now, v, bp, bv, paths=None):
        """(threat, every bird opening or safe (for the release), the most threatening bird's index or None)."""
        if len(bp) == 0:
            return False, True, None
        miss_m, rel_m, H = self.cfg['miss_m'], self.cfg['release_m'], self.cfg['horizon_s']
        threat, safe, worst, worst_miss = False, True, None, np.inf
        curved = None if paths is None else self._miss_paths(p_now, v, paths)
        for k in range(len(bp)):
            p = p_now - bp[k]
            rv = v - bv[k]
            if curved is None:
                t_star, c = cpa(p, rv, H)
                dc, d = float(np.linalg.norm(c)), float(np.linalg.norm(p))
                closing = float(p @ rv) < 0.0 or d < miss_m
            else:                                    # (a bird on a curve may be opening now and come back)
                dc, t_star, d = (float(x[k]) for x in curved)
                closing = t_star > 0.0 or d < miss_m
            if dc < miss_m and (t_star < H or d < miss_m) and closing:
                threat = True
            if closing and dc < rel_m:
                safe = False                         # still closing with a predicted miss under the release threshold
            if closing and dc < worst_miss:
                worst, worst_miss = k, dc
        return threat, safe, worst

    # ---------------------------------------------------------------- evasion
    def _min_miss(self, p_now, v_c, others_p, others_v, paths=None):
        """Smallest predicted miss of a drone moving at v_c to the given bodies (inf for none)."""
        if paths is not None:
            return float(self._miss_paths(p_now, v_c, paths)[0].min())
        miss = np.inf
        H = self.cfg['horizon_s']
        for q, w in zip(others_p, others_v):
            _, c = cpa(p_now - q, v_c - w, H)
            miss = min(miss, float(np.linalg.norm(c)))
        return miss

    def _command(self, i, env, key, base, pos, vel, p_now, v, bp, bv, worst, raw, paths=None):
        cfg, tau = self.cfg, self.predict_s
        n = len(pos)
        # other DRONES (shared states), not the neighbour slots (which may hold birds); parked ones are far away
        idx = [j for j in range(n) if j != i and pos[j][2] < PARKED_Z and np.linalg.norm(pos[j] - pos[i]) < DRONE_NEAR_M]
        dp = [pos[j] + vel[j] * tau for j in idx]
        dv = [vel[j] for j in idx]
        # LiDAR ring (obstacles; hits on drones dropped)
        ring = np.asarray(raw[key].get('lidar_ring', []), dtype=float)
        rng_max = env.lidar_range
        ring_d = np.full(0, rng_max)
        dirs = np.zeros((0, 3))
        if len(ring):
            yaw = env.adapter._heading.get(key, 0.0)
            dirs = lidar_mod.ring_directions(yaw, len(ring))
            ring_d = ring.copy()
            hit = ring_d < rng_max - 1e-3
            if hit.any() and idx:
                pts = pos[i][None, :2] + ring_d[hit, None] * dirs[hit][:, :2]
                near = np.min(np.linalg.norm(pts[:, None, :] - pos[idx][None, :, :2], axis=2), axis=1)
                dropped = np.flatnonzero(hit)[near <= 0.35]
                ring_d[dropped] = rng_max
        # direction of the relative motion to the most threatening bird (or the nearest one)
        k = worst if worst is not None else int(np.argmin(np.linalg.norm(p_now - bp, axis=1)))
        rv_h = (v - bv[k])[:2]
        n_rv = float(np.linalg.norm(rv_h))
        right = np.array([rv_h[1], -rv_h[0], 0.0]) / n_rv if n_rv > 1e-6 else np.array([0.0, -1.0, 0.0])
        # altitude band around the drone's current target height
        zc = float(env.adapter.goals[key][2])
        lo, hi = max(zc - self.half_width, cfg['z_floor_m']), zc + self.half_width
        z = float(pos[i][2])
        best_v, best_score = base.copy(), -np.inf

        def score(v_c, horiz):
            s = self._min_miss(p_now, v_c, bp, bv, paths)
            if idx:
                s = min(s, self._min_miss(p_now, v_c, dp, dv) + cfg['drone_margin_m'])
            if horiz is not None and len(ring_d):
                cone = dirs[:, :2] @ horiz >= RING_CONE
                if cone.any():
                    s = min(s, float(ring_d[cone].min()))
            return s
        base_score = score(base, None)
        best_score = base_score
        for side, vert in CANDIDATES:
            if vert > 0 and z >= hi or vert < 0 and z <= lo:
                continue                                         # at the edge of the band: no further that way
            u = side * right + np.array([0.0, 0.0, float(vert)])
            u = u / np.linalg.norm(u)
            v_c = np.clip(base + cfg['evade_speed'] * u, -self.v_max, self.v_max)
            horiz = None
            if side != 0:
                h = side * right[:2]
                horiz = h / np.linalg.norm(h)
            sc = score(v_c, horiz)
            if sc > best_score + 1e-9:
                best_v, best_score = v_c, sc
        for scale in cfg['brake']:                                       # slow down / hover: the command scaled
            sc = score(base * scale, None)
            if sc > best_score + 1e-9:
                best_v, best_score = base * scale, sc
        out = best_v
        # teacher-style push-away while active: other drones inside close_dist, LiDAR obstacle hits inside obs_close
        for j in idx:
            d_vec = pos[i] - pos[j]
            d = float(np.linalg.norm(d_vec))
            if 1e-6 < d < self.close_dist:
                out = out + self.k_close * (self.close_dist - d) / self.close_dist * (d_vec / d)
        if len(ring_d):
            hit = ring_d < rng_max - 1e-3
            if hit.any():
                j = int(np.argmin(np.where(hit, ring_d, np.inf)))
                if ring_d[j] < self.obs_close:
                    away = -dirs[j]
                    out = out + self.k_obs_close * (self.obs_close - ring_d[j]) / self.obs_close * away
        return np.clip(out, -self.v_max, self.v_max).tolist()
