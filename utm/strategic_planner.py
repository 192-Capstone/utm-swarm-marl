"""
strategic_planner.py
--------------------
E1: approve, delay, move or reject flight plans before take-off, so that approved flights never
share airspace at the same time.

    sky = StrategicPlanner(layout, bounds)
    d = sky.submit(FlightPlan('f1', start=(-10, 0), goal=(12, 3)))
    d.status, d.layer, d.t_dep, d.delay, d.waypoints          # 'approved', 0, 0.0, 0.0, [(x, y, z), ...]

Flow for one plan (first come, first served):
  1. route    the D7 planner on the layer's map (buildings lower than the layer's floor ignored)
  2. profile  take off from the start pad at start_z, climb to the layer's centre height,
              cruise along the simplified route, descend at the goal pad
  3. reserve  every dt along the profile: the H3 cells within `radius` of the drone, in the
              altitude layer it is in, for a time window padded by a buffer that grows with
              time since departure (timing errors build up along a route); plus the pads:
              pad_mode 'window' (default): the start pad shortly before take-off and the goal
              pad shortly after landing, as in real UTM, where a drone is on the ground before
              and after its flight (in the sim: kept off-map until departure and removed once
              it has settled, e.g. with the adapter's park); pad_mode 'episode': the start pad
              from t = 0 until departure and the goal pad open-ended, for a drone that hovers on
              its pad for the whole episode (blocks far more airspace: ~50% fewer approvals)
  4. check    against the ledger of approved flights and the blocked (cell, layer) keys
  5. resolve  options (layer, route, delay) by cost = arrival delay vs the fastest plan
              + layer_penalty x layers changed; the cheapest conflict-free one is booked;
              none within max_delay -> rejected. With no-fly zones (D2, apply_zones) each layer
              has two routes: the direct one and one around every zone the flight may not enter
              (masked on the route map exactly where the reservation would touch the zone), so
              the cost decides between waiting for a temporary zone and flying around it

A reservation is a set of keys (H3 cell, layer), each with time intervals (s). Two flights
conflict when they share a key with overlapping intervals.

Scale (Settings) is the drone-sized sim world by default: H3 resolution 15 (0.6 m edge), 2 layers
of 2 m from 1 m (cruise at 2 m and 4 m), separation target 2 m -> radius 1 m + 1.1 m tracking
margin (the D7 route-following p99 cross-track error), cruise 1.5 m/s, climb 0.5 m/s,
buffer 1 s + 10% of the time since departure (max 3 s), max delay 60 s. Real-scale studies only
change the numbers.
"""

from dataclasses import dataclass, field

import numpy as np

from envs.airspace import Airspace
from envs.route_planner import Grid, build_grid, plan, simplify
from envs.scenario_manager import DRONE_RADIUS

INF = float('inf')


@dataclass
class Settings:
    resolution: int = 15
    layer_floor: float = 1.0
    layer_height: float = 2.0
    n_layers: int = 2
    start_z: float = 1.0              # pad hover height (the policy never learned to take off)
    v_cruise: float = 1.5             # m/s
    v_climb: float = 0.5              # m/s
    dt: float = 0.25                  # s between profile samples
    radius: float = 2.1               # m: separation 2 m / 2 + tracking margin 1.1 m (D7 p99 cross-track)
    buffer_base: float = 1.0          # s
    buffer_frac: float = 0.10         # s per s since departure
    buffer_max: float = 3.0           # s
    max_delay: float = 60.0           # s
    pad_mode: str = 'window'          # 'window' | 'episode' (see the module docstring)
    pad_before: float = 2.0           # s the start pad is held before take-off ('window')
    pad_after: float = 7.0            # s the goal pad is held after landing ('window'): the flown p99 of
                                      # settling after the planned arrival (E1 v2 runs: p50 0.9, p95 4.6, p99 6.5 s)
    delay_step: float = 0.5           # s
    layer_penalty: float = 4.0        # s per layer away from the preferred one
    grid_res: float = 0.25            # m, route planner grid
    inflate: float = DRONE_RADIUS + 0.35


@dataclass
class FlightPlan:
    id: str
    start: tuple                      # (x, y) pad, local metres
    goal: tuple
    layer: int = 0                    # preferred cruise layer
    t_earliest: float = 0.0           # s
    priority: int = 0                 # informational for now (E4)
    operator: str = ''                # who flies it (E7 fairness between operators)


@dataclass
class Decision:
    id: str
    status: str                       # 'approved' | 'rejected'
    layer: int = None
    t_dep: float = None
    delay: float = None               # t_dep - t_earliest
    arrival: float = None
    cost: float = None
    waypoints: list = field(default_factory=list)       # (x, y, z)
    reservation: dict = field(default_factory=dict)     # (cell, layer) -> [(t0, t1), ...]
    reason: str = ''
    around_zones: bool = False        # flies the route around the no-fly zones
    blockers: list = field(default_factory=list)        # 'blocked' / 'zone:<id>' that were in the way


def _merge(intervals):
    out = []
    for lo, hi in sorted(intervals):
        if out and lo <= out[-1][1]:
            out[-1][1] = max(out[-1][1], hi)
        else:
            out.append([lo, hi])
    return [tuple(iv) for iv in out]


class StrategicPlanner:
    def __init__(self, layout, bounds, settings=None, airspace=None, blocked=()):
        """layout: obstacle layout (envs/scenario_manager.py format). bounds: ((xmin, xmax),
        (ymin, ymax)). blocked: (cell, layer) keys nobody may use (no-fly zones, D2)."""
        self.s = settings or Settings()
        if self.s.pad_mode not in ('window', 'episode'):
            raise ValueError(f"pad_mode must be 'window' or 'episode', got {self.s.pad_mode!r}")
        self.sky = airspace or Airspace(resolution=self.s.resolution, layer_floor=self.s.layer_floor,
                                        layer_height=self.s.layer_height)
        self.layout, self.bounds = list(layout), bounds
        self.blocked = set(blocked)
        self.exempt = {}                  # flight id -> ledger owners it may share keys with (D2)
        self.zone_keys = {}               # 'zone:<id>' -> its (cell, layer) keys, routed around (D2)
        self.ledger = {}                  # (cell, layer) -> [(t0, t1, flight id)]
        self.decisions = []
        self._grids = {}
        self._centres = {}
        self._edge = self.sky.edge_length()
        self._k = self.sky.k_for_radius(self.s.radius)

    # ------------------------------------------------------------- geometry
    def layer_of(self, z):
        """Altitude layer of a height, kept inside 0 .. n_layers - 1: the profile starts and ends
        exactly at start_z (the floor), where rounding could otherwise give layer -1."""
        return min(max(self.sky.layer(z + 1e-6), 0), self.s.n_layers - 1)

    def layer_z(self, layer):
        lo, hi = self.sky.layer_range(layer)
        return 0.5 * (lo + hi)

    def _grid(self, layer):
        if layer not in self._grids:
            lo, _ = self.sky.layer_range(layer)
            self._grids[layer] = build_grid(self.layout, self.bounds, self.s.grid_res, self.s.inflate,
                                            min_height=lo)
        return self._grids[layer]

    def _zone_grid(self, layer, zones):
        """The layer's map with the zones' cells added as obstacles: every grid point whose
        reservation (cells_near) would reach one of their cells on this layer, plus one grid step
        for the samples between grid points."""
        cache = (layer, zones)
        if cache not in self._grids:
            g = self._grid(layer)
            occ = g.occ.copy()
            reach = self.s.radius + self._edge + g.res
            cells = {c for z in zones for c, k in self.zone_keys[z] if k == layer}
            for c in cells:
                if c not in self._centres:
                    self._centres[c] = self.sky.center(c)
                cx, cy = self._centres[c]
                ix0, ix1 = np.searchsorted(g.xs, cx - reach), np.searchsorted(g.xs, cx + reach, side='right')
                iy0, iy1 = np.searchsorted(g.ys, cy - reach), np.searchsorted(g.ys, cy + reach, side='right')
                if ix0 >= ix1 or iy0 >= iy1:
                    continue
                gx, gy = np.meshgrid(g.xs[ix0:ix1], g.ys[iy0:iy1])
                occ[iy0:iy1, ix0:ix1] |= (gx - cx) ** 2 + (gy - cy) ** 2 <= reach * reach
            self._grids[cache] = Grid(occ, g.xs, g.ys, g.res)
        return self._grids[cache]

    def route(self, start, goal, layer, avoid=frozenset()):
        """Simplified D7 route (x, y) on the layer's map, or None. avoid: zone owners
        ('zone:<id>', see zone_keys) to fly around."""
        avoid = frozenset(z for z in avoid if any(k == layer for _, k in self.zone_keys[z]))
        g = self._zone_grid(layer, avoid) if avoid else self._grid(layer)
        p = plan(g, start, goal)
        return None if p is None else simplify(g, p)

    def cells_near(self, x, y):
        """H3 cells that come within `radius` of (x, y): the disk around the drone's cell, keeping
        only cells whose centre is within radius + one edge (a hexagon's corner distance)."""
        reach = self.s.radius + self._edge
        out = []
        for c in self.sky.disk(self.sky.cell(x, y), self._k):
            if c not in self._centres:
                self._centres[c] = self.sky.center(c)
            cx, cy = self._centres[c]
            if (cx - x) ** 2 + (cy - y) ** 2 <= reach * reach:
                out.append(c)
        return out

    # ------------------------------------------------------------- profile
    def profile(self, route_xy, layer):
        """Samples (tau, x, y, z) from departure (tau = 0) to arrival at start_z over the goal."""
        s = self.s
        z0, zl = s.start_z, self.layer_z(layer)
        climb = abs(zl - z0) / s.v_climb
        pts = np.asarray(route_xy, dtype=float)
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        cum = np.concatenate([[0.0], np.cumsum(seg)])
        cruise = cum[-1] / s.v_cruise
        total = 2 * climb + cruise
        out = []
        for tau in np.append(np.arange(0.0, total, s.dt), total):
            if tau < climb:
                x, y, z = pts[0, 0], pts[0, 1], z0 + (zl - z0) * tau / climb
            elif tau <= climb + cruise:
                d = (tau - climb) * s.v_cruise
                k = min(int(np.searchsorted(cum, d, side='right')) - 1, len(seg) - 1)
                t = 0.0 if seg[k] < 1e-9 else (d - cum[k]) / seg[k]
                x, y = pts[k] + t * (pts[k + 1] - pts[k])
                z = zl
            else:
                x, y, z = pts[-1, 0], pts[-1, 1], zl + (z0 - zl) * (tau - climb - cruise) / climb
            out.append((float(tau), float(x), float(y), float(z)))
        return out, total

    def _relative_reservation(self, prof):
        """(cell, layer) -> merged intervals, relative to departure."""
        s, res = self.s, {}
        for tau, x, y, z in prof:
            b = min(s.buffer_max, s.buffer_base + s.buffer_frac * tau)
            layer = self.layer_of(z)
            for c in self.cells_near(x, y):
                res.setdefault((c, layer), []).append((tau - b, tau + b))
        return {k: _merge(v) for k, v in res.items()}

    def _absolute(self, rel, t_dep, start_cells, goal_cells, total):
        """Shift the flight to t_dep and add the pads (pad_mode)."""
        s = self.s
        out = {k: [(lo + t_dep, hi + t_dep) for lo, hi in v] for k, v in rel.items()}
        pad = self.layer_of(s.start_z)
        b_end = min(s.buffer_max, s.buffer_base + s.buffer_frac * total)
        arrival = t_dep + total
        if s.pad_mode == 'episode':
            start_iv, goal_iv = (0.0, t_dep + s.buffer_base), (arrival - b_end, INF)
        else:
            start_iv = (t_dep - s.pad_before, t_dep + s.buffer_base)
            goal_iv = (arrival - b_end, arrival + s.pad_after)
        for c in start_cells:
            out.setdefault((c, pad), []).append(start_iv)
        for c in goal_cells:
            out.setdefault((c, pad), []).append(goal_iv)
        return {k: _merge(v) for k, v in out.items()}

    # ------------------------------------------------------------- ledger
    def conflicts(self, reservation, fid=None):
        """Flight ids (or 'blocked') sharing a key with an overlapping interval. Owners in
        exempt[fid] don't count (a flight authorised into a zone)."""
        hit = set()
        skip = self.exempt.get(fid, ())
        for key, ivs in reservation.items():
            if key in self.blocked:
                hit.add('blocked')
            for lo2, hi2, owner in self.ledger.get(key, ()):
                if owner not in skip and any(lo < hi2 and lo2 < hi for lo, hi in ivs):
                    hit.add(owner)
        return hit

    def _book(self, fid, reservation):
        for key, ivs in reservation.items():
            self.ledger.setdefault(key, []).extend((lo, hi, fid) for lo, hi in ivs)

    # ------------------------------------------------------------- submit
    def submit(self, fp):
        s = self.s
        options = {}
        avoid = frozenset(self.zone_keys) - self.exempt.get(fp.id, set())
        for layer in range(s.n_layers):
            direct = self.route(fp.start, fp.goal, layer)
            around = self.route(fp.start, fp.goal, layer, avoid) if avoid else None
            for kind, r in ((0, direct), (1, around)):
                if r is None or (kind == 1 and r == direct):
                    continue
                prof, total = self.profile(r, layer)
                options[layer, kind] = (r, total, self._relative_reservation(prof))
        if not options:
            d = Decision(fp.id, 'rejected', reason='no route on any layer')
            self.decisions.append(d)
            return d
        fastest = min(fp.t_earliest + total for _, total, _ in options.values())
        cands = []
        for (layer, kind), (_, total, _) in options.items():
            for delay in np.arange(0.0, s.max_delay + 1e-9, s.delay_step):
                arrival = fp.t_earliest + delay + total
                cost = (arrival - fastest) + s.layer_penalty * abs(layer - fp.layer)
                cands.append((cost, float(delay), layer, kind))
        cands.sort()
        start_cells, goal_cells = self.cells_near(*fp.start), self.cells_near(*fp.goal)
        blockers = set()
        for cost, delay, layer, kind in cands:
            r, total, rel = options[layer, kind]
            t_dep = fp.t_earliest + delay
            res = self._absolute(rel, t_dep, start_cells, goal_cells, total)
            hit = self.conflicts(res, fp.id)
            blockers |= {h for h in hit if h == 'blocked' or str(h).startswith('zone:')}
            if not hit:
                self._book(fp.id, res)
                zl = self.layer_z(layer)
                wps = ([(r[0][0], r[0][1], s.start_z)] + [(x, y, zl) for x, y in r]
                       + [(r[-1][0], r[-1][1], s.start_z)])
                d = Decision(fp.id, 'approved', layer, t_dep, delay, t_dep + total, float(cost),
                             wps, res, around_zones=bool(kind))
                self.decisions.append(d)
                return d
        why = f'no conflict-free option within {s.max_delay} s'
        if blockers:
            why += ' (zones in the way: ' + ', '.join(sorted(blockers)) + ')'
        d = Decision(fp.id, 'rejected', reason=why, blockers=sorted(blockers))
        self.decisions.append(d)
        return d

    def summary(self):
        ok = [d for d in self.decisions if d.status == 'approved']
        n = len(self.decisions)
        delays = np.array([d.delay for d in ok]) if ok else np.zeros(0)
        return {
            'requests': n,
            'approved': len(ok),
            'approved_pct': 100.0 * len(ok) / n if n else 0.0,
            'delay_mean_s': float(delays.mean()) if ok else 0.0,
            'delay_p95_s': float(np.percentile(delays, 95)) if ok else 0.0,
            'delayed_pct': 100.0 * float((delays > 0).mean()) if ok else 0.0,
            'layer_use': {k: sum(d.layer == k for d in ok) for k in range(self.s.n_layers)},
        }
