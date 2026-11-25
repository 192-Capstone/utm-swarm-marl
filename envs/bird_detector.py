"""
bird_detector.py
----------------
Stage 3a: an ONBOARD bird detector with a tracker, so a drone can be told about birds without being
given their true positions. Opt-in: environment.birds.detector (off by default).

Per drone and bird, every control step:
    visible     the bird is within range_m, inside the horizontal field of view (fov_deg centred on the
                drone's velocity direction; 360 = a multi-camera rig), and (realistic mode, occlusion on)
                no building / obstacle is between them (one ray, ignoring drones and birds)
    detection   with probability p_near (distance 0) falling linearly to p_far (at range_m)
    measurement the bird's position + N(0, sigma_m) per axis, delivered latency_steps control steps late
    tracker     one alpha-beta filter per (drone, bird): position and velocity. A track is reported once
                it has confirm_hits measurements, is extrapolated to the current step, coasts coast_s
                after its last measurement and then drops.
Only birds are detected here. Other drones still come from shared states (neighbour_source).

Tracker (stage 3a++): `tracker: alphabeta` (default) or `kalman`, a per-axis constant-velocity Kalman filter
(process noise: the bird's acceleration, kalman_q; measurement noise: sigma_m; irregular gaps handled).

Shared tracks (stage 3a++, `share: true`, default off; realistic mode only): each drone also receives the tracks of
the drones within share_range_m (30 m, the neighbour sensing radius), as they were share_latency_steps ago (the
comms latency, on top of the sender's own detection latency), moved on by the sender's velocity, their age
increased by the latency. Own and received tracks are then MERGED by geometry: sorted freshest first (then most
hits, own first), a track within share_gate_m of one already kept is a duplicate and dropped. The result has up to
n_drones * n_birds tracks per drone (no bird ids).
NOTE on generosity: without sharing the detector associates each drone's measurements to its own tracks PERFECTLY
(tracks are indexed by the true bird), which a real tracker would not; shared tracks use the geometric association
above. The two are therefore not equally generous: sharing is the more realistic of the two.

oracle mode (an UPPER BOUND for the evaluation, never a result): 360 degrees, no occlusion, no noise, no
misses, no latency, true velocity; the range is the same range_m. It isolates what the sensing limits cost.

THE VALUES BELOW ARE ASSUMPTIONS (like the sensor-noise levels), NOT MEASUREMENTS of any camera or radar:
    range_m 8, fov_deg 120, p_near 0.9, p_far 0.5, sigma_m 0.1, latency_steps 3 (0.096 s), coast_s 0.5,
    alpha 0.4, beta 0.05, confirm_hits 2.

Randomness: its own generator, seeded per episode by SwarmEnv (not the global stream, not the sensor-noise
stream), so drones, noise and bird paths are the same with and without the detector. The generator draws the
same numbers every step whatever is visible, so the draws of one episode do not depend on the geometry.

Config (environment.birds.detector):
    detector:
      enabled: true
      mode: realistic          # realistic | oracle
      range_m: 8.0
      fov_deg: 120.0
      p_near: 0.9
      p_far: 0.5
      sigma_m: 0.1
      latency_steps: 3
      coast_s: 0.5
      alpha: 0.4
      beta: 0.05
      confirm_hits: 2
      occlusion: true
      seed: 0
      birds_in_slots: true     # tracked birds also fill neighbour slots (stage 3a). false: the 5 slots stay for
                               # drones and the tracks are for the bird guard only (modules/bird_guard.py)
      tracker: alphabeta       # alphabeta | kalman
      kalman_q: 1.0            # m/s^2, the bird's acceleration (process noise), kalman only
      kalman_v0: 3.0           # m/s, initial velocity uncertainty of a new track, kalman only
      share: false             # drones also use the tracks of drones within share_range_m (merged by geometry)
      share_range_m: 30.0
      share_latency_steps: 3
      share_gate_m: 1.0        # tracks closer than this are the same bird
"""

from collections import deque

import numpy as np

DEFAULTS = {
    'enabled': False, 'mode': 'realistic', 'range_m': 8.0, 'fov_deg': 120.0, 'p_near': 0.9, 'p_far': 0.5,
    'sigma_m': 0.1, 'latency_steps': 3, 'coast_s': 0.5, 'alpha': 0.4, 'beta': 0.05, 'confirm_hits': 2,
    'occlusion': True, 'seed': 0, 'birds_in_slots': True, 'tracker': 'alphabeta', 'kalman_q': 1.0, 'kalman_v0': 3.0,
    'share': False, 'share_range_m': 30.0, 'share_latency_steps': 3, 'share_gate_m': 1.0,
}
TRACKERS = ('alphabeta', 'kalman')
MODES = ('realistic', 'oracle')
MOVING = 0.2          # m/s: slower than this a drone keeps its last heading


def validate(block):
    """The detector block with the defaults filled in, or None when it is missing / not enabled."""
    if not block:
        return None
    block = dict(block)
    unknown = set(block) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"birds.detector: unknown settings {sorted(unknown)}")
    cfg = {**DEFAULTS, **block}
    if not cfg['enabled']:
        return None
    if cfg['mode'] not in MODES:
        raise ValueError(f"birds.detector.mode must be one of {MODES}, got {cfg['mode']!r}")
    if not cfg['range_m'] > 0:
        raise ValueError("birds.detector.range_m must be positive")
    if not 0 < cfg['fov_deg'] <= 360:
        raise ValueError("birds.detector.fov_deg must be in (0, 360]")
    if not (0 <= cfg['p_far'] <= cfg['p_near'] <= 1):
        raise ValueError("birds.detector needs 0 <= p_far <= p_near <= 1")
    if cfg['sigma_m'] < 0 or cfg['coast_s'] < 0 or cfg['latency_steps'] < 0 or cfg['confirm_hits'] < 1:
        raise ValueError("birds.detector: sigma_m, coast_s, latency_steps >= 0 and confirm_hits >= 1")
    if not (0 < cfg['alpha'] <= 1 and 0 <= cfg['beta'] <= 1):
        raise ValueError("birds.detector: alpha in (0, 1], beta in [0, 1]")
    cfg['latency_steps'], cfg['confirm_hits'] = int(cfg['latency_steps']), int(cfg['confirm_hits'])
    cfg['birds_in_slots'] = bool(cfg['birds_in_slots'])
    if cfg['tracker'] not in TRACKERS:
        raise ValueError(f"birds.detector.tracker must be one of {TRACKERS}, got {cfg['tracker']!r}")
    if not (cfg['kalman_q'] > 0 and cfg['kalman_v0'] > 0):
        raise ValueError("birds.detector: kalman_q and kalman_v0 must be positive")
    cfg['share'] = bool(cfg['share'])
    if not (cfg['share_range_m'] > 0 and cfg['share_gate_m'] > 0 and cfg['share_latency_steps'] >= 0):
        raise ValueError("birds.detector: share_range_m and share_gate_m must be positive, share_latency_steps >= 0")
    cfg['share_latency_steps'] = int(cfg['share_latency_steps'])
    return cfg


class BirdDetector:
    """One episode's detector + trackers for n drones and b birds. step() returns what each drone now
    knows: (known (n, b) bool, pos (n, b, 3), vel (n, b, 3)), in the world frame."""

    def __init__(self, cfg, n_drones, n_birds, dt, rng, headings=None, blocked=None):
        """headings: (n, 2) initial horizontal heading per drone (default +x); blocked(starts, ends) ->
        bool array: True where a building / obstacle is on the segment (realistic mode with occlusion)."""
        self.cfg, self.n, self.b, self.dt, self.rng = cfg, int(n_drones), int(n_birds), float(dt), rng
        self.oracle = cfg['mode'] == 'oracle'
        self.blocked = blocked
        self.heading = (np.tile([1.0, 0.0], (self.n, 1)) if headings is None
                        else np.asarray(headings, dtype=float).copy())
        self.coast = int(round(cfg['coast_s'] / self.dt))
        shape = (self.n, self.b)
        self.x, self.v = np.zeros(shape + (3,)), np.zeros(shape + (3,))
        self.last = np.full(shape, -10 ** 9, dtype=np.int64)     # step of the last measurement
        self.hits = np.zeros(shape, dtype=np.int64)
        self.kalman = cfg['tracker'] == 'kalman'
        if self.kalman:                                           # per-axis covariance [[pxx, pxv], [pxv, pvv]]
            self.pxx, self.pxv, self.pvv = np.zeros(shape), np.zeros(shape), np.zeros(shape)
        self.share = bool(cfg['share']) and not self.oracle
        self._history = deque(maxlen=cfg['share_latency_steps'] + 1)   # own tracks of the last steps (for sharing)
        self._pending = []                                        # (step measured, detected, measured pos)
        self.visible = np.zeros(shape, dtype=bool)                # last step's visibility (tests, stats)

    # -------------------------------------------------------------- geometry
    def _update_heading(self, drone_vel):
        v = np.asarray(drone_vel, dtype=float)[:, :2]
        sp = np.linalg.norm(v, axis=1)
        moving = sp > MOVING
        self.heading[moving] = v[moving] / sp[moving, None]

    def _fov_ok(self, rel):
        if self.cfg['fov_deg'] >= 360.0:
            return np.ones(rel.shape[:2], dtype=bool)
        h = np.linalg.norm(rel[..., :2], axis=2)
        cos = np.where(h > 1e-9, np.einsum('nbc,nc->nb', rel[..., :2], self.heading) / np.maximum(h, 1e-9), 1.0)
        return cos >= np.cos(np.radians(self.cfg['fov_deg'] / 2.0))

    # ------------------------------------------------------------------ step
    def step(self, k, drone_pos, drone_vel, bird_pos, bird_vel):
        drone_pos, bird_pos = np.asarray(drone_pos, dtype=float), np.asarray(bird_pos, dtype=float)
        bird_vel = np.asarray(bird_vel, dtype=float)
        rel = bird_pos[None, :, :] - drone_pos[:, None, :]
        dist = np.linalg.norm(rel, axis=2)
        in_range = dist <= self.cfg['range_m']
        if self.oracle:
            self.visible = in_range
            known = in_range
            return known, np.broadcast_to(bird_pos[None], rel.shape).copy(), np.broadcast_to(bird_vel[None], rel.shape).copy()

        self._update_heading(drone_vel)
        vis = in_range & self._fov_ok(rel)
        if self.cfg['occlusion'] and self.blocked is not None and vis.any():
            ii, jj = np.nonzero(vis)
            vis[ii, jj] = ~self.blocked(drone_pos[ii], bird_pos[jj])
        self.visible = vis
        # the same draws every step, whatever is visible
        u = self.rng.random((self.n, self.b))
        noise = self.rng.normal(0.0, self.cfg['sigma_m'], (self.n, self.b, 3))
        p = self.cfg['p_near'] + (self.cfg['p_far'] - self.cfg['p_near']) * np.clip(dist / self.cfg['range_m'], 0, 1)
        det = vis & (u < p)
        self._pending.append((k, det, bird_pos[None, :, :] + noise))
        L = self.cfg['latency_steps']
        while self._pending and self._pending[0][0] + L <= k:           # deliver what has arrived
            self._update(*self._pending.pop(0))
        age = (k - self.last)
        known = (self.hits >= self.cfg['confirm_hits']) & (age <= self.coast)
        pos = self.x + self.v * (self.dt * np.maximum(age, 0))[..., None]
        if self.share:
            own = (known.copy(), pos.copy(), self.v.copy(), age.copy(), self.hits.copy())
            self._history.append(own)
            return self._merge(drone_pos, own)
        return known, pos, self.v.copy()

    def _merge(self, drone_pos, own):
        """Own tracks + the tracks received from drones in comms range, merged by geometry (see the module doc)."""
        cfg, dt = self.cfg, self.dt
        Ls = cfg['share_latency_steps']
        old = self._history[0] if len(self._history) == Ls + 1 else None        # what the others had Ls steps ago
        K = self.n * self.b
        out_known = np.zeros((self.n, K), dtype=bool)
        out_pos, out_vel = np.zeros((self.n, K, 3)), np.zeros((self.n, K, 3))
        o_known, o_pos, o_vel, o_age, o_hits = own
        for i in range(self.n):
            cands = []                                           # (age, -hits, 0 own / 1 received, pos, vel)
            for b in np.flatnonzero(o_known[i]):
                cands.append((int(o_age[i, b]), -int(o_hits[i, b]), 0, o_pos[i, b], o_vel[i, b]))
            if old is not None:
                s_known, s_pos, s_vel, s_age, s_hits = old
                for j in range(self.n):
                    if j == i or np.linalg.norm(drone_pos[j] - drone_pos[i]) > cfg['share_range_m']:
                        continue
                    for b in np.flatnonzero(s_known[j]):
                        age = int(s_age[j, b]) + Ls
                        if age > self.coast:
                            continue
                        cands.append((age, -int(s_hits[j, b]), 1, s_pos[j, b] + s_vel[j, b] * (Ls * dt), s_vel[j, b]))
            cands.sort(key=lambda c: (c[0], c[1], c[2]))         # freshest, then most confident, own first
            kept = []
            for c in cands:
                if all(np.linalg.norm(c[3] - a[3]) > cfg['share_gate_m'] for a in kept):
                    kept.append(c)
            for s, c in enumerate(kept[:K]):
                out_known[i, s], out_pos[i, s], out_vel[i, s] = True, c[3], c[4]
        return out_known, out_pos, out_vel

    def _update_kalman(self, km, det, z):
        cfg, dt = self.cfg, self.dt
        R = max(cfg['sigma_m'] ** 2, 1e-6)
        fresh = det & ((self.hits == 0) | ((km - self.last) > self.coast))
        cont = det & ~fresh
        self.x[fresh], self.v[fresh], self.hits[fresh], self.last[fresh] = z[fresh], 0.0, 1, km
        self.pxx[fresh], self.pxv[fresh], self.pvv[fresh] = R, 0.0, cfg['kalman_v0'] ** 2
        if cont.any():
            D = ((km - self.last)[cont] * dt)                    # seconds since the last measurement
            q2 = cfg['kalman_q'] ** 2
            pxx, pxv, pvv = self.pxx[cont], self.pxv[cont], self.pvv[cont]
            xp = self.x[cont] + self.v[cont] * D[:, None]
            pxx_p = pxx + 2 * D * pxv + D ** 2 * pvv + q2 * D ** 4 / 4
            pxv_p = pxv + D * pvv + q2 * D ** 3 / 2
            pvv_p = pvv + q2 * D ** 2
            S = pxx_p + R
            kx, kv = pxx_p / S, pxv_p / S
            r = z[cont] - xp
            self.x[cont] = xp + kx[:, None] * r
            self.v[cont] = self.v[cont] + kv[:, None] * r
            self.pxx[cont], self.pxv[cont], self.pvv[cont] = (1 - kx) * pxx_p, (1 - kx) * pxv_p, pvv_p - kv * pxv_p
            self.hits[cont] += 1
            self.last[cont] = km

    def _update(self, km, det, z):
        if self.kalman:
            return self._update_kalman(km, det, z)
        a, bt, dt = self.cfg['alpha'], self.cfg['beta'], self.dt
        fresh = det & ((self.hits == 0) | ((km - self.last) > self.coast))
        cont = det & ~fresh
        self.x[fresh], self.v[fresh], self.hits[fresh], self.last[fresh] = z[fresh], 0.0, 1, km
        if cont.any():
            g = (km - self.last)[cont]                                   # steps since the last measurement
            gdt = (g * dt)[:, None]
            xp = self.x[cont] + self.v[cont] * gdt
            r = z[cont] - xp
            second = self.hits[cont] == 1
            v_new = self.v[cont] + bt * r / gdt
            v_new[second] = np.clip((z[cont] - self.x[cont])[second] / gdt[second], -6.0, 6.0)   # first velocity: finite difference
            self.x[cont] = xp + a * r
            self.v[cont] = v_new
            self.hits[cont] += 1
            self.last[cont] = km
