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

Changing a booking (E4, utm/contingency.py): release(id, t) frees what a flight booked after t;
replan(id, position, t) books a new plan from where the drone is now (no start pad, the climb or
descent to the chosen layer starts at its current height); the first `grace` s are not checked
(they are already happening: the onboard policy handles them), and if no option fits, the old
booking is kept. A request with priority > 0 (an emergency) may take air booked by lower-priority
flights: it is booked as if they weren't there; those still on the ground are re-submitted, and
airborne ones are returned in Decision.displaced for re-planning from their positions.

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
    layers_open: tuple = None         # layers flights may use (None: all); e.g. not the ones among the roofs
    start_z: float = 1.0              # pad hover height (the policy never learned to take off)
    v_cruise: float = 1.5             # m/s
    v_climb: float = 0.5              # m/s
    v_descent: float = None           # m/s down (None: v_climb); the last slow_band_m when the next is set
    v_descent_high: float = None      # m/s descending while higher than slow_band_m over the pad
    slow_band_m: float = 3.0          # m: the last metres down at v_descent (a gentle touchdown, C5)
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
    traffic_detour: bool = False      # also try a same-layer route around the cells other flights have
                                      # booked at the times this one would pass them (_detour)
    detour_margin_s: float = 5.0      # s: a detour keeps this much more time from other traffic than
                                      # the booking buffer (flown timing drifts by more than 1-3 s; on
                                      # the same layer that is all that keeps two drones apart)
    grid_res: float = 0.25            # m, route planner grid
    inflate: float = DRONE_RADIUS + 0.35


@dataclass
class FlightPlan:
    id: str
    start: tuple                      # (x, y) pad, local metres
    goal: tuple
    layer: int = 0                    # preferred cruise layer
    t_earliest: float = 0.0           # s
    priority: int = 0                 # > 0: emergency, may displace lower-priority bookings (E4)
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
    around_traffic: bool = False      # flies a same-layer detour around other flights' booked air
    blockers: list = field(default_factory=list)        # 'blocked' / 'zone:<id>' that were in the way
    displaced: list = field(default_factory=list)       # airborne flights an emergency took air from (E4)


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
        self.plans = {}                   # flight id -> its FlightPlan (E4 re-planning)
        self.current = {}                 # flight id -> its booked Decision now
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

    def _detour(self, fp, layer, route, avoid, z_start, snap, tries=6):
        """A same-layer route around other flights' booked air, or None (none needed or none found).
        Time-aware: a cell counts only if someone else has it booked when this flight (leaving at
        t_earliest) would pass it; cells near its own pads don't (a delay handles those). Those cells
        become obstacles and the route is planned again, up to `tries` times as the new route meets
        other traffic; the usual conflict check then judges the result like any other option."""
        near_pads = set(self.cells_near(*fp.start)) | set(self.cells_near(*fp.goal))
        base = self._zone_grid(layer, avoid) if avoid else self._grid(layer)
        cells, r = set(), route
        for _ in range(tries):
            prof, total = self.profile(r, layer, z_start)
            res = self._absolute(self._relative_reservation(prof), fp.t_earliest, [], [], total)
            busy = {c for (c, k), ivs in res.items()
                    if k == layer and c not in near_pads
                    and self._traffic((c, k), [(lo - self.s.detour_margin_s, hi + self.s.detour_margin_s)
                                               for lo, hi in ivs], fp.id)}
            if not busy - cells:
                return r if cells else None
            cells |= busy
            g = self._block_cells(base, cells, layer)
            p = plan(g, fp.start, fp.goal, snap)
            if p is None:
                return None
            r = simplify(g, p)                    # straightened around the traffic too
        return None

    def _traffic(self, key, ivs, fid):
        """Another flight (not a zone, not blocked air, not one fid may enter) has key booked in ivs."""
        skip = set(self.exempt.get(fid, ())) | {fid}
        return any(owner not in skip and not str(owner).startswith('zone:') and lo < hi2 and lo2 < hi
                   for lo2, hi2, owner in self.ledger.get(key, ()) for lo, hi in ivs)

    def _zone_grid(self, layer, zones):
        """The layer's map with the zones' cells added as obstacles: every grid point whose
        reservation (cells_near) would reach one of their cells on this layer, plus one grid step
        for the samples between grid points."""
        cache = (layer, zones)
        if cache not in self._grids:
            cells = {c for z in zones for c, k in self.zone_keys[z] if k == layer}
            self._grids[cache] = self._block_cells(self._grid(layer), cells, layer)
        return self._grids[cache]

    def _block_cells(self, g, cells, layer):
        """Grid g with these H3 cells (on `layer`) added as obstacles: every grid point whose
        reservation (cells_near) would reach one of them, plus one grid step for the samples between
        grid points."""
        occ = g.occ.copy()
        reach = self.s.radius + self._edge + g.res
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
        return Grid(occ, g.xs, g.ys, g.res)

    def route(self, start, goal, layer, avoid=frozenset(), snap=1.0):
        """Simplified D7 route (x, y) on the layer's map, or None. avoid: zone owners
        ('zone:<id>', see zone_keys) to fly around. snap: how far (m) start and goal may be from
        the nearest free grid cell (a drone re-planned mid-air may be inside a zone's margin)."""
        avoid = frozenset(z for z in avoid if any(k == layer for _, k in self.zone_keys[z]))
        g = self._zone_grid(layer, avoid) if avoid else self._grid(layer)
        p = plan(g, start, goal, snap)
        return None if p is None else simplify(g, p)

    def _clear_below(self, xy, layer):
        """A drone at xy may descend straight to this layer: no building of that layer's map (grown by
        the planner's margin) under it. Re-planned mid-air onto a lower layer, the route starts with that
        drop where the drone is; over a roof it would land on it (E-City, f14 at 15 m over a 12 m roof)."""
        g = self._grid(layer)
        return not g.inside(xy) or g.free(g.to_cell(xy))

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
    def profile(self, route_xy, layer, z_start=None):
        """Samples (tau, x, y, z) from departure (tau = 0) to arrival at start_z over the goal.
        z_start: height the climb starts from (default start_z, a pad; E4 re-plans mid-air)."""
        s = self.s
        z0 = s.start_z if z_start is None else float(z_start)
        zl = self.layer_z(layer)
        climb = abs(zl - z0) / s.v_climb
        drop = abs(zl - s.start_z)
        high = 0.0 if s.v_descent_high is None else max(0.0, drop - s.slow_band_m)   # m descended fast
        slow_v = s.v_climb if s.v_descent is None else s.v_descent
        fast = high / s.v_descent_high if high else 0.0
        descent = fast + (drop - high) / slow_v
        pts = np.asarray(route_xy, dtype=float)
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        cum = np.concatenate([[0.0], np.cumsum(seg)])
        cruise = cum[-1] / s.v_cruise
        total = climb + cruise + descent
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
                u = tau - climb - cruise                  # s into the descent: fast, then slow
                down = u * s.v_descent_high if u < fast else high + (u - fast) * slow_v
                x, y, z = pts[-1, 0], pts[-1, 1], zl - np.sign(zl - s.start_z) * min(down, drop)
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
    def conflicts(self, reservation, fid=None, after=None):
        """Flight ids (or 'blocked') sharing a key with an overlapping interval. Owners in
        exempt[fid] don't count (a flight authorised into a zone), nor does fid itself; with
        `after`, only the parts of the reservation from that time on are checked (E4: the next
        seconds are already happening)."""
        hit = set()
        skip = set(self.exempt.get(fid, ())) | {fid}
        if after is not None:
            reservation = {k: [(max(lo, after), hi) for lo, hi in ivs if hi > after]
                           for k, ivs in reservation.items()}
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
    def submit(self, fp, now=None):
        """Decide a request (first come, first served) and book it. A request with priority > 0
        may displace lower-priority bookings: flights still on the ground at `now` (default: the
        request's earliest departure) are re-submitted here; airborne ones are listed in
        Decision.displaced, to be re-planned from their positions (utm/contingency.py)."""
        self.plans[fp.id] = fp
        before = dict(self.current)
        d = self._submit(fp)
        t = fp.t_earliest if now is None else now
        for fid in list(d.displaced):
            old = before.get(fid)
            if old is not None and old.t_dep >= t:           # not flying yet: just file it again
                p = self.plans[fid]
                self._submit(FlightPlan(p.id, p.start, p.goal, p.layer, max(p.t_earliest, t),
                                        p.priority, p.operator))
                d.displaced.remove(fid)
        return d

    def _submit(self, fp, z_start=None, pad_start=True, after=None, max_delay=None, snap=1.0):
        s = self.s
        max_delay = s.max_delay if max_delay is None else max_delay
        options = {}
        avoid = frozenset(self.zone_keys) - self.exempt.get(fp.id, set())
        for layer in (range(s.n_layers) if s.layers_open is None else s.layers_open):
            if z_start is not None and self.layer_z(layer) < z_start - 1e-6 and not self._clear_below(fp.start, layer):
                continue                  # mid-air: never step down onto a roof (the drop is straight down)
            direct = self.route(fp.start, fp.goal, layer, snap=snap)
            around = self.route(fp.start, fp.goal, layer, avoid, snap) if avoid else None
            swerve = (self._detour(fp, layer, around or direct, avoid, z_start, snap)
                      if s.traffic_detour and layer == fp.layer and (around or direct) is not None else None)
            for kind, r in ((0, direct), (1, around), (2, swerve)):
                if r is None or (kind == 1 and r == direct):
                    continue
                prof, total = self.profile(r, layer, z_start)
                options[layer, kind] = (r, total, self._relative_reservation(prof))
        if not options:
            d = Decision(fp.id, 'rejected', reason='no route on any layer')
            self.decisions.append(d)
            return d
        fastest = min(fp.t_earliest + total for _, total, _ in options.values())
        cands = []
        for (layer, kind), (_, total, _) in options.items():
            for delay in np.arange(0.0, max_delay + 1e-9, s.delay_step):
                arrival = fp.t_earliest + delay + total
                cost = (arrival - fastest) + s.layer_penalty * abs(layer - fp.layer)
                cands.append((cost, float(delay), layer, kind))
        cands.sort()
        start_cells = self.cells_near(*fp.start) if pad_start else []
        goal_cells = self.cells_near(*fp.goal)
        blockers = set()
        for cost, delay, layer, kind in cands:
            r, total, rel = options[layer, kind]
            t_dep = fp.t_earliest + delay
            res = self._absolute(rel, t_dep, start_cells, goal_cells, total)
            hit = self.conflicts(res, fp.id, after)
            blockers |= {h for h in hit if h == 'blocked' or str(h).startswith('zone:')}
            moved = sorted(h for h in hit if self._yields(h, fp))
            if len(moved) == len(hit):                      # no clash, or only flights that give way
                for h in moved:
                    self.release(h, -INF)
                self._book(fp.id, res)
                zl = self.layer_z(layer)
                z0 = s.start_z if z_start is None else float(z_start)
                wps = ([(r[0][0], r[0][1], z0)] + [(x, y, zl) for x, y in r]
                       + [(r[-1][0], r[-1][1], s.start_z)])
                d = Decision(fp.id, 'approved', layer, t_dep, delay, t_dep + total, float(cost),
                             wps, res, around_zones=kind == 1 or (kind == 2 and bool(avoid)),
                             around_traffic=kind == 2, displaced=moved)
                self.decisions.append(d)
                self.current[fp.id] = d
                return d
        why = f'no conflict-free option within {max_delay} s'
        if blockers:
            why += ' (zones in the way: ' + ', '.join(sorted(blockers)) + ')'
        d = Decision(fp.id, 'rejected', reason=why, blockers=sorted(blockers))
        self.decisions.append(d)
        return d

    def _yields(self, owner, fp):
        """Does a booking owner give way to this request? Only a flight of lower priority, and
        only to an emergency (priority > 0); never a zone or an intruder's protective volume."""
        if fp.priority <= 0 or owner == 'blocked' or str(owner).startswith(('zone:', 'intruder:')):
            return False
        other = self.plans.get(owner)
        return other is not None and other.priority < fp.priority

    # ------------------------------------------------------------- changing a booking (E4)
    def release(self, fid, t):
        """Free what flight fid has booked from time t on (its windows are cut at t)."""
        for key in list(self.ledger):
            kept = []
            for lo, hi, owner in self.ledger[key]:
                if owner != fid:
                    kept.append((lo, hi, owner))
                elif lo < t:
                    kept.append((lo, min(hi, t), owner))
            if kept:
                self.ledger[key] = kept
            else:
                del self.ledger[key]

    def replan(self, fid, position, t, grace=2.0, snap=3.5):
        """Book a new plan for an airborne flight from its current position (x, y, z) at time t
        to its original goal; the first `grace` s are not checked. It leaves now (a flying drone
        can't wait on a pad: hovering in place would not be booked), so only the layer and the
        route can change; its start may be up to `snap` m from free air (inside a zone's margin).
        If a zone (or blocked air) is what keeps it from its goal, it returns to its start pad
        instead (Decision.reason 'return to base'); traffic alone never sends it home. If nothing
        fits, the old booking is restored and the rejected Decision returned."""
        p = self.plans[fid]
        ledger, current = {k: list(v) for k, v in self.ledger.items()}, self.current.get(fid)
        self.release(fid, t)
        here = (float(position[0]), float(position[1]))
        for goal, note in ((p.goal, ''), (p.start, 'return to base')):
            fp = FlightPlan(fid, here, goal, p.layer, float(t), p.priority, p.operator)
            d = self._submit(fp, z_start=float(position[2]), pad_start=False, after=t + grace,
                             max_delay=0.0, snap=snap)
            if d.status == 'approved':
                d.reason = note
                break
            if not d.blockers:                      # only traffic in the way: don't give up the goal
                break
        if d.status != 'approved':
            self.ledger = ledger
            if current is not None:
                self.current[fid] = current
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
