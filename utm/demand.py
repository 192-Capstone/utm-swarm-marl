"""
demand.py
---------
D4: traffic demand. Flight requests (where from, where to, when, who) for the strategic planner
(E1), the experiments and the dashboard (E5), from low to very high density.

    from utm.demand import DemandSettings, generate_demand, save_demand, load_demand
    dem = generate_demand(layout, bounds, DemandSettings(rate=20, mode='hubs'), seed=3)
    for fp in dem.plans:                       # E1 FlightPlan objects, sorted by t_earliest
        sky.submit(fp)
    save_demand('demand.json', dem)            # replay the exact same demand later
    dem.report                                 # requested / kept / dropped (unroutable), ...

Density is flights per minute per district, a district being 50 m x 50 m = 2,500 m2 (one D1
city). A bigger map gets proportionally more flights: rate x map area / 2,500 per minute. Levels
used in the experiments: LEVELS = 5, 10, 20, 40, 80 (low -> very high).

Departures are a Poisson process at that rate over [0, duration_s). An optional peak
(centre_s, width_s, factor) multiplies the rate by up to `factor` with a Gaussian bump around
centre_s (rush hour); it is drawn by thinning, so the count stays Poisson.

Where pads may go (the pad area): cells of the largest connected free region of layer 0's map
(every pair is then connected) that are also at least pad_clearance m from every building (all
heights) and edge m from the map border, and of those, the largest 4-connected part. Flown E1
runs showed the policy won't settle on a pad close to a wall (it hovers ~1 m off and lands late
or never), hence the defaults 1.5 m and 2 m. Pockets reachable only through gaps narrower than
about 2 x pad_clearance (a courtyard behind a narrow opening) are excluded on purpose: a pad there
has clearance, but every flight to it squeezes past walls. report['pad_area_share_all_parts'] vs
report['pad_area_share'] shows what that costs. pad_clearance=0 and edge=0 give the whole region
(the generator as it was before these settings). Pads are cell centres, so the clearance holds
exactly at the pad.

Origins / destinations (mode):
    random  pad-area points, trips at least min_trip m long; with max_trip set, the goal is
            drawn at a uniform distance in [min_trip, max_trip] around the start (short trips
            on a big map),
            pads at least pad_sep m from the pads of flights departing within pad_window_s
    hubs    n_hubs depots spread over the pad area (farthest-point) and n_drops drop points;
            a share hub_share of the flights go depot -> drop point, the rest drop point -> depot.
            Each depot is a cluster of pads_per_hub pads within hub_radius of its centre, pad_sep
            apart; a flight uses one pad of its depot at random. With a single pad per depot,
            every depot flight takes off from or lands on that one pad, so E1 has to queue them
            all on it and the depot pad, not the airspace, limits capacity.
    or pass `pads`: a plain list of (x, y) or (x, y, kind), e.g. from D2's vertiports later;
    'random' then picks two pads per flight, 'hubs' uses kind 'hub' as depots and the rest as
    drop points. Given pads are never moved or dropped for clearance: the report says how close
    they are (pads_below_clearance, pads_below_edge).

Every request has a D7 route on layer 0 (E1's own layer-0 map). Pads drawn from the pad area
(random points, generated hubs and drops) always have one, since the region is connected the way
A* moves, so they skip the A* check. Given pads may not: each such request is checked with A*, and
unroutable ones are dropped and counted (report['dropped_unroutable']), so E1's approval rate
measures airspace capacity, not map connectivity; keep_unroutable=True keeps them (raw demand).

Settings are validated (ValueError on a negative rate, min_trip > max_trip, ...; also when
pad_clearance and edge leave no cell for pads). Anything that gives fewer flights or pads than
asked shows in the report: no_valid_pair (no pair found within max_tries draws),
dropped_unroutable, pads_requested vs pads_placed and depot_pads (hubs mode), pad_area_share.
The report also gives pad_clearance_min and pad_edge_min: how close the pads actually used are to
a building and to the border.

Each flight also gets an operator id (op0, op1, ... with configurable shares) for the fairness
metrics (E7) and a preferred cruise layer.

random_requests() is the small fixed-count generator the E1 scripts use. It applies the same
pad_clearance / edge rule but NOT the largest-part rule: it must give exactly the requests of the
E1 scripts' version (PR #27), which the flown E1 results were made with. 0, 0 gives its output from
before pad clearance.
"""

import json
from collections import Counter
from dataclasses import asdict, dataclass, field, fields

import numpy as np

from envs.route_planner import _surface_distance_grid, build_grid, plan
from utm.strategic_planner import FlightPlan, StrategicPlanner

DISTRICT_AREA = 2500.0                          # m2: one 50 m x 50 m D1 city
LEVELS = (5, 10, 20, 40, 80)                    # flights / min / district: low -> very high
FORMAT = 'utm-demand'
VERSION = 2                                     # 2: pad_clearance, edge, pads_per_hub, hub_radius
OLD_SETTINGS = {1: {'pad_clearance': 0.0, 'edge': 0.0, 'pads_per_hub': 1}}   # what version 1 used


@dataclass
class DemandSettings:
    rate: float = 10.0                          # flights / min / district (2,500 m2)
    duration_s: float = 120.0                   # departures in [0, duration_s)
    peak: tuple = None                          # (centre_s, width_s, factor) or None
    mode: str = 'random'                        # 'random' | 'hubs'
    min_trip: float = 8.0                       # m, straight-line start -> goal
    max_trip: float = None                      # m, or None (no limit)
    pad_sep: float = 3.0                        # m between pads of flights close in time ('random')
    pad_window_s: float = 10.0                  # s: "close in time" for pad_sep
    n_hubs: int = 3                             # 'hubs': depots
    n_drops: int = 20                           # 'hubs': drop points
    hub_share: float = 0.8                      # 'hubs': share of flights depot -> drop point
    operators: dict = field(default_factory=lambda: {'op0': 0.25, 'op1': 0.25, 'op2': 0.25, 'op3': 0.25})
    layers: tuple = (0,)                        # preferred cruise layer, drawn uniformly
    keep_unroutable: bool = False
    max_tries: int = 50                         # draws per flight before giving up on a request
    pad_clearance: float = 1.5                  # m from any building to a generated pad
    edge: float = 2.0                           # m from the map border to a generated pad
    pads_per_hub: int = 1                       # 'hubs': pads per depot (1 = one pad per depot)
    hub_radius: float = 6.0                     # 'hubs': m, a depot's pads lie within this of its centre


@dataclass
class Pad:
    x: float
    y: float
    kind: str = ''                              # e.g. 'hub', 'drop'; free text
    group: int = -1                             # depot index of a generated hub pad (-1: none)


@dataclass
class Demand:
    plans: list                                 # FlightPlan, sorted by t_earliest
    pads: list                                  # Pad (hubs mode or given pads; [] for random)
    settings: DemandSettings
    report: dict
    seed: int = None
    bounds: tuple = None


# ----------------------------------------------------------------------------- the map
def layer0_grid(layout, bounds, settings=None):
    """E1's layer-0 routing grid (the same map E1 routes on for its lowest layer)."""
    return StrategicPlanner(layout, bounds, settings)._grid(0)


def map_area(bounds):
    (x0, x1), (y0, y1) = bounds
    return float((x1 - x0) * (y1 - y0))


def rate_per_min(settings, bounds):
    """Flights per minute for this map: the per-district rate scaled by map area."""
    return settings.rate * map_area(bounds) / DISTRICT_AREA


def _runs(mask):
    """Horizontal runs of True cells: run k is row rows[k], columns [starts[k], ends[k])."""
    ny = mask.shape[0]
    pad = np.zeros((ny, 1), dtype=bool)
    d = np.diff(np.hstack([pad, mask, pad]).astype(np.int8), axis=1)
    rows, starts = np.nonzero(d == 1)
    ends = np.nonzero(d == -1)[1]                # (same row-major order as the starts)
    return rows, starts, ends


class Cells:
    """A set of grid cells kept as horizontal runs: uniform sampling and membership tests without
    listing every cell (the full E-City map has ~8.5 M cells)."""

    def _set_runs(self, grid, rows, starts, ends):
        self.grid = grid
        self.rows, self.starts, self.ends = rows, starts, ends
        lengths = (ends - starts).astype(np.int64)
        self.n_cells = int(lengths.sum())
        self._cdf = np.cumsum(lengths) / max(lengths.sum(), 1)    # pick a run by its length
        self._row_first = np.searchsorted(rows, np.arange(grid.occ.shape[0] + 1))

    @classmethod
    def from_mask(cls, grid, mask):
        c = cls.__new__(cls)
        c._set_runs(grid, *_runs(mask))
        return c

    def mask(self):
        m = np.zeros(self.grid.occ.shape, dtype=bool)
        for row, a, b in zip(self.rows, self.starts, self.ends):
            m[row, a:b] = True
        return m

    def sample(self, rng):
        """A uniformly random cell centre (x, y) of the set."""
        k = min(int(np.searchsorted(self._cdf, rng.uniform(), side='right')), len(self.rows) - 1)
        col = self.starts[k] + rng.randint(self.ends[k] - self.starts[k])
        return self.grid.to_world((int(self.rows[k]), int(col)))

    def contains(self, xy):
        iy, ix = self.grid.to_cell(xy)
        if not self.grid.inside(xy):
            return False
        lo, hi = self._row_first[iy], self._row_first[iy + 1]
        return bool(np.any((self.starts[lo:hi] <= ix) & (ix < self.ends[lo:hi])))


def _largest_part(mask):
    """The largest 4-connected part of a boolean mask, as runs (rows, starts, ends), plus the
    number of True cells in all parts; None when the mask is empty. Works on runs, not cells."""
    ny = mask.shape[0]
    rows, starts, ends = _runs(mask)
    n = len(rows)
    if n == 0:
        return None
    parent = np.arange(n)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    row_first = np.searchsorted(rows, np.arange(ny + 1))
    for y in range(ny - 1):
        i, i_end = row_first[y], row_first[y + 1]
        j, j_end = row_first[y + 1], row_first[y + 2]
        while i < i_end and j < j_end:
            if starts[i] < ends[j] and starts[j] < ends[i]:      # columns overlap: touching
                a, b = find(i), find(j)
                if a != b:
                    parent[b] = a
            if ends[i] < ends[j]:
                i += 1
            else:
                j += 1
    roots =np.array([find(k) for k in range(n)], dtype=np.int64)
    lengths = (ends - starts).astype(np.int64)
    size = np.bincount(roots, weights=lengths)
    keep = roots == int(np.argmax(size))
    return rows[keep], starts[keep], ends[keep], int(lengths.sum())


class Region(Cells):
    """The largest connected free region of a grid, as horizontal runs of free cells.
    Connectivity is 4-neighbour, which is exactly what A* (8 moves, no corner cutting) can reach.
    Works from runs rather than cells, so the full E-City map (~8.5 M cells) takes a second."""

    def __init__(self, grid):
        part = _largest_part(~grid.occ)
        if part is None:
            raise ValueError("the map has no free cell")
        rows, starts, ends, all_free = part
        self._set_runs(grid, rows, starts, ends)
        self.free_cells = all_free
        self.cells_all_parts = self.n_cells
        self._pad_areas = {}


# ----------------------------------------------------------------------------- pad clearance
def surface_clearance(layout, points, min_height=0.0):
    """Horizontal distance (m) from each point to the nearest obstacle surface taller than
    min_height (negative inside one; inf with no obstacle)."""
    p = np.asarray(points, dtype=float).reshape(-1, 2)
    d = np.full(len(p), np.inf)
    for ob in layout:
        if ob.get('height', np.inf) > min_height:
            d = np.minimum(d, _surface_distance_grid(ob, p[:, 0], p[:, 1]))
    return d


def edge_distance(bounds, points):
    """Distance (m) from each point to the nearest map border."""
    p = np.asarray(points, dtype=float).reshape(-1, 2)
    (x0, x1), (y0, y1) = bounds
    return np.min([p[:, 0] - x0, x1 - p[:, 0], p[:, 1] - y0, y1 - p[:, 1]], axis=0)


def pad_mask(grid, layout, bounds, pad_clearance, edge):
    """Cells whose centre is >= pad_clearance from every building (all heights) and >= edge inside
    the bounds (not yet limited to the connected region). The same expression as the E1 scripts'
    pad rule (PR #27), so random_requests matches theirs exactly."""
    pad_map = build_grid(layout, bounds, grid.res, pad_clearance)
    if pad_map.occ.shape != grid.occ.shape:
        raise ValueError("pad_mask: the clearance grid doesn't match the layer-0 grid (bounds?)")
    (x0, x1), (y0, y1) = bounds
    gx, gy = np.meshgrid(grid.xs, grid.ys)
    inside = (gx >= x0 + edge) & (gx <= x1 - edge) & (gy >= y0 + edge) & (gy <= y1 - edge)
    return ~pad_map.occ & inside


def pad_area(region, layout, bounds, pad_clearance=1.5, edge=2.0):
    """Where generated pads may go: the largest 4-connected part of the region's cells that pass
    pad_mask (kept on the region, so repeated calls on the same map are free). Its
    cells_all_parts counts every part, before keeping the largest. pad_clearance = edge = 0 gives
    the region itself."""
    if pad_clearance <= 0 and edge <= 0:
        return region
    key = (float(pad_clearance), float(edge))
    if key not in region._pad_areas:
        part = _largest_part(region.mask() & pad_mask(region.grid, layout, bounds, pad_clearance, edge))
        if part is None:
            raise ValueError(f"pad_clearance {pad_clearance} m and edge {edge} m leave no cell for pads "
                             f"in the connected region ({region.n_cells} cells): lower them")
        rows, starts, ends, all_parts = part
        area = Cells.__new__(Cells)
        area._set_runs(region.grid, rows, starts, ends)
        area.cells_all_parts = all_parts
        region._pad_areas[key] = area
    return region._pad_areas[key]


# ----------------------------------------------------------------------------- time
def departure_times(rng, rate_min, duration_s, peak=None):
    """Poisson departure times (s) at rate_min flights per minute over [0, duration_s), sorted.
    peak (centre_s, width_s, factor): rate x (1 + (factor - 1) exp(-((t - centre) / width)^2 / 2)),
    drawn by thinning."""
    lam = rate_min / 60.0
    if peak is None:
        n = rng.poisson(lam * duration_s)
        return np.sort(rng.uniform(0.0, duration_s, n))
    c, w, f = (float(v) for v in peak)
    if w <= 0 or f < 1:
        raise ValueError("peak needs width_s > 0 and factor >= 1")
    lam_max = lam * f
    n = rng.poisson(lam_max * duration_s)
    t = np.sort(rng.uniform(0.0, duration_s, n))
    accept = rng.uniform(0.0, 1.0, n) * lam_max < lam * (1 + (f - 1) * np.exp(-0.5 * ((t - c) / w) ** 2))
    return t[accept]


# ----------------------------------------------------------------------------- pads
def _as_pads(pads):
    out = []
    for p in pads:
        if isinstance(p, Pad):
            out.append(p)
        elif len(p) == 2:
            out.append(Pad(float(p[0]), float(p[1])))
        else:
            out.append(Pad(float(p[0]), float(p[1]), str(p[2])))
    return out


def hub_pads(rng, area, n_hubs, n_drops, min_sep=3.0, candidates=400, per_hub=1, radius=6.0):
    """n_hubs depots spread over the area (farthest-point sampling over random candidates), each a
    cluster of up to per_hub pads within radius of its centre (the centre is the first), then
    n_drops drop points; every added pad is at least min_sep from every other pad. Pads that don't
    fit are left out (the caller reports the shortfall). per_hub = 1 draws exactly as before."""
    cand = np.array([area.sample(rng) for _ in range(candidates)])
    hubs = [cand[rng.randint(len(cand))]]
    while len(hubs) < n_hubs:
        d = np.min(np.linalg.norm(cand[:, None, :] - np.array(hubs)[None, :, :], axis=2), axis=1)
        hubs.append(cand[int(np.argmax(d))])
    pads = [Pad(float(x), float(y), 'hub', i) for i, (x, y) in enumerate(hubs)]
    g = area.grid
    for i, (cx, cy) in enumerate(hubs if per_hub > 1 else []):
        tries = 0
        while sum(p.group == i for p in pads) < per_hub and tries < 200 * per_hub:
            tries += 1
            r, ang = radius * np.sqrt(rng.uniform()), rng.uniform(0.0, 2 * np.pi)
            q = (cx + r * np.cos(ang), cy + r * np.sin(ang))
            if not area.contains(q):
                continue
            x, y = g.to_world(g.to_cell(q))
            if np.hypot(x - cx, y - cy) <= radius and all(np.hypot(x - p.x, y - p.y) >= min_sep for p in pads):
                pads.append(Pad(float(x), float(y), 'hub', i))
    tries = 0
    while sum(p.kind == 'drop' for p in pads) < n_drops and tries < 100 * n_drops:
        tries += 1
        x, y = area.sample(rng)
        if all(np.hypot(x - p.x, y - p.y) >= min_sep for p in pads):
            pads.append(Pad(float(x), float(y), 'drop'))
    return pads


# ----------------------------------------------------------------------------- generator
def validate(s, given_pads=False):
    """ValueError with a clear message for settings that can't give sensible demand."""
    if s.mode not in ('random', 'hubs'):
        raise ValueError(f"mode must be 'random' or 'hubs', got {s.mode!r}")
    if not s.rate >= 0:
        raise ValueError(f"rate must be >= 0 flights / min / district, got {s.rate}")
    if not s.duration_s >= 0:
        raise ValueError(f"duration_s must be >= 0, got {s.duration_s}")
    if not s.min_trip >= 0:
        raise ValueError(f"min_trip must be >= 0, got {s.min_trip}")
    if s.max_trip is not None and not s.max_trip >= s.min_trip:
        raise ValueError(f"max_trip ({s.max_trip}) must be >= min_trip ({s.min_trip})")
    if not s.pad_sep >= 0 or not s.pad_window_s >= 0:
        raise ValueError(f"pad_sep and pad_window_s must be >= 0, got {s.pad_sep}, {s.pad_window_s}")
    if not 0.0 <= s.hub_share <= 1.0:
        raise ValueError(f"hub_share must be in [0, 1], got {s.hub_share}")
    if int(s.max_tries) < 1:
        raise ValueError(f"max_tries must be >= 1, got {s.max_tries}")
    if s.mode == 'hubs' and not given_pads and (int(s.n_hubs) < 1 or int(s.n_drops) < 1):
        raise ValueError(f"mode 'hubs' needs n_hubs >= 1 and n_drops >= 1, got {s.n_hubs}, {s.n_drops}")
    if len(s.layers) == 0 or any(int(k) < 0 for k in s.layers):
        raise ValueError(f"layers must be a non-empty list of layers >= 0, got {s.layers}")
    if not s.pad_clearance >= 0 or not s.edge >= 0:
        raise ValueError(f"pad_clearance and edge must be >= 0 m, got {s.pad_clearance}, {s.edge}")
    if int(s.pads_per_hub) < 1 or not s.hub_radius >= 0:
        raise ValueError(f"pads_per_hub must be >= 1 and hub_radius >= 0, got {s.pads_per_hub}, {s.hub_radius}")
    if s.operators:
        w = np.array([float(v) for v in s.operators.values()])
        if (w < 0).any() or w.sum() <= 0:
            raise ValueError("operators: shares must be >= 0 and not all 0")


def _kinds(pads):
    """Number of distinct pad positions of each kind ('' counts as 'pad')."""
    seen = {}
    for p in pads:
        seen.setdefault(p.kind or 'pad', set()).add((p.x, p.y))
    return {k: len(v) for k, v in sorted(seen.items())}


def generate_demand(layout, bounds, settings=None, seed=0, pads=None, planner_settings=None,
                    grid=None, region=None):
    """Demand for one map: E1 FlightPlans sorted by departure time, plus a report.
    layout, bounds: as for StrategicPlanner. pads: optional list of (x, y[, kind]) / Pad.
    planner_settings: E1 Settings whose layer-0 map is used for sampling and the route check.
    grid / region: the layer-0 grid / its Region, to reuse across calls on the same map (they
    must come from this layout, bounds and planner_settings); built here when not given."""
    s = settings or DemandSettings()
    validate(s, given_pads=pads is not None)
    if s.operators:
        ops = list(s.operators)
        w = np.array([float(s.operators[k]) for k in ops])
        w = w / w.sum()
    rng = np.random.RandomState(seed)
    if region is not None:
        grid = region.grid
    else:
        grid = grid if grid is not None else layer0_grid(layout, bounds, planner_settings)
        region = Region(grid)
    source = 'given' if pads is not None else ('generated' if s.mode == 'hubs' else 'free')
    area = region if source == 'given' else pad_area(region, layout, bounds, s.pad_clearance, s.edge)
    pads = _as_pads(pads) if pads is not None else (
        hub_pads(rng, area, s.n_hubs, s.n_drops, s.pad_sep, per_hub=int(s.pads_per_hub),
                 radius=s.hub_radius) if s.mode == 'hubs' else [])
    hubs = [p for p in pads if p.kind == 'hub']
    others = [p for p in pads if p.kind != 'hub']
    if s.mode == 'hubs' and (not hubs or not others):
        why = (f": only {_kinds(pads)} fit in the pad area ({area.n_cells} cells) with pad_sep {s.pad_sep}, "
               f"pad_clearance {s.pad_clearance}, edge {s.edge}" if source == 'generated' else "")
        raise ValueError(f"mode 'hubs' needs at least one pad of kind 'hub' and one other pad{why}")
    if s.mode == 'random' and source == 'given' and len(pads) < 2:
        raise ValueError(f"mode 'random' with given pads needs at least 2 pads, got {len(pads)}")
    check_route = source == 'given'             # pad-area pads are always connected (see top)
    depots = {}                                 # depot -> its hub pads (a given hub is its own depot)
    for i, p in enumerate(hubs):
        depots.setdefault(p.group if p.group >= 0 else ('given', i), []).append(p)
    depots = list(depots.values())
    clustered = any(len(dp) > 1 for dp in depots)

    times = departure_times(rng, rate_per_min(s, bounds), s.duration_s, s.peak)
    plans, recent = [], []                      # recent: (t, start, goal) for pad_sep
    dropped = no_pair = 0
    for t in times:
        pair = None
        for _ in range(s.max_tries):
            if s.mode == 'hubs':
                if clustered:                    # a depot, then one of its pads
                    dp = depots[rng.randint(len(depots))]
                    h = dp[rng.randint(len(dp))]
                else:                            # one pad per depot: the draws of version 1
                    h = hubs[rng.randint(len(hubs))]
                o = others[rng.randint(len(others))]
                a, b = ((h.x, h.y), (o.x, o.y)) if rng.uniform() < s.hub_share else ((o.x, o.y), (h.x, h.y))
            elif pads:
                i, j = rng.choice(len(pads), 2, replace=False)
                a, b = (pads[i].x, pads[i].y), (pads[j].x, pads[j].y)
            elif s.max_trip is not None:         # goal drawn around the start: a big map with
                a = area.sample(rng)             # short trips would reject almost every pair
                r = rng.uniform(s.min_trip, s.max_trip)
                ang = rng.uniform(0.0, 2 * np.pi)
                q = (a[0] + r * np.cos(ang), a[1] + r * np.sin(ang))
                if not area.contains(q):
                    continue
                b = grid.to_world(grid.to_cell(q))
            else:
                a, b = area.sample(rng), area.sample(rng)
            L =float(np.hypot(b[0] - a[0], b[1] - a[1]))
            if L < s.min_trip or (s.max_trip is not None and L > s.max_trip):
                continue
            if not pads:                         # free points: keep pads of nearby-in-time flights apart
                near = [q for tq, sq, gq in recent if abs(t - tq) < s.pad_window_s for q in (sq, gq)]
                if any(np.hypot(p[0] - q[0], p[1] - q[1]) < s.pad_sep for p in (a, b) for q in near):
                    continue
            pair = (tuple(map(float, a)), tuple(map(float, b)))
            break
        if pair is None:
            no_pair += 1
            continue
        a, b = pair
        # drawn before the route check, so keep_unroutable doesn't change the other requests
        op = ops[rng.choice(len(ops), p=w)] if s.operators else ''
        layer = int(s.layers[rng.randint(len(s.layers))])
        if check_route and plan(grid, a, b) is None:
            dropped += 1
            if not s.keep_unroutable:
                continue
        plans.append(FlightPlan(f"f{len(plans)}", a, b, layer=layer, t_earliest=float(t), operator=op))
        recent = [r for r in recent if t - r[0] < s.pad_window_s] + [(float(t), a, b)]

    report = {
        'map_area_m2': map_area(bounds),
        'districts': map_area(bounds) / DISTRICT_AREA,
        'rate_per_min_district': s.rate,
        'rate_per_min_map': rate_per_min(s, bounds),
        'duration_s': s.duration_s,
        'requested': int(len(times)),
        'no_valid_pair': int(no_pair),
        'dropped_unroutable': int(0 if s.keep_unroutable else dropped),
        'unroutable_kept': int(dropped if s.keep_unroutable else 0),
        'kept': len(plans),
        'region_share': region.n_cells / max(region.free_cells, 1),
        'operators': dict(Counter(p.operator for p in plans)),
        'pads_source': source,                  # 'free' (random points), 'generated' (hubs), 'given'
        'pads_requested': ({'drop': int(s.n_drops), 'hub': int(s.n_hubs) * int(s.pads_per_hub)}
                           if source == 'generated' else _kinds(pads)),
        'pads_placed': _kinds(pads),            # distinct positions; < requested when they didn't fit
        'depot_pads': [len({(p.x, p.y) for p in dp}) for dp in depots],    # pads per depot
        'pad_area_share_all_parts': area.cells_all_parts / max(region.n_cells, 1),   # before the
        'pad_area_share': area.n_cells / max(region.n_cells, 1),           # largest-part rule, after
    }
    report.update(_clearance_report(layout, bounds, s, pads if pads else None, plans))
    return Demand(plans, pads, s, report, seed, tuple(tuple(map(float, b)) for b in bounds))


def _clearance_report(layout, bounds, s, pads, plans):
    """How close pads are to buildings and to the border: the minimum over the pads actually used
    (None when no flight or no building), and how many pads (the given / generated ones, or the
    used ones for random points) are below pad_clearance / edge."""
    used = sorted({q for p in plans for q in (p.start, p.goal)})
    pool = [(p.x, p.y) for p in pads] if pads else used
    cl_used, ed_used = surface_clearance(layout, used), edge_distance(bounds, used)
    cl_pool, ed_pool = surface_clearance(layout, pool), edge_distance(bounds, pool)
    finite = cl_used[np.isfinite(cl_used)]
    return {
        'pad_clearance_min': float(finite.min()) if len(finite) else None,
        'pad_edge_min': float(ed_used.min()) if used else None,
        'pads_below_clearance': int(np.sum(cl_pool < s.pad_clearance - 1e-9)),
        'pads_below_edge': int(np.sum(ed_pool < s.edge - 1e-9)),
    }


# ----------------------------------------------------------------------------- save / load
def save_demand(path, demand):
    """JSON with everything needed to replay the demand exactly (plans, pads, settings, report)."""
    s = asdict(demand.settings)
    data = {
        'format': FORMAT, 'version': VERSION, 'seed': demand.seed, 'bounds': demand.bounds,
        'settings': s, 'report': demand.report,
        'pads': [asdict(p) for p in demand.pads],
        'flights': [{'id': p.id, 'start': list(p.start), 'goal': list(p.goal), 'layer': p.layer,
                     't_earliest': p.t_earliest, 'priority': p.priority, 'operator': p.operator}
                    for p in demand.plans],
    }
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=1)


def load_demand(path):
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    if data.get('format') != FORMAT:
        raise ValueError(f"{path} is not a demand file (format {data.get('format')!r})")
    version = data.get('version')
    if version not in (1, VERSION):
        raise ValueError(f"{path}: demand format version {version}, expected 1 or {VERSION}")
    known = {f.name for f in fields(DemandSettings)}
    s = {**OLD_SETTINGS.get(version, {}), **{k: v for k, v in data['settings'].items() if k in known}}
    for k in ('peak', 'layers'):
        if s.get(k) is not None:
            s[k] = tuple(s[k])
    plans = [FlightPlan(fl['id'], tuple(fl['start']), tuple(fl['goal']), layer=fl.get('layer', 0),
                        t_earliest=fl['t_earliest'], priority=fl.get('priority', 0),
                        operator=fl.get('operator', ''))
             for fl in data['flights']]
    bounds = tuple(tuple(b) for b in data['bounds']) if data.get('bounds') else None
    return Demand(plans, [Pad(**p) for p in data.get('pads', [])], DemandSettings(**s),
                  data.get('report', {}), data.get('seed'), bounds)


# ----------------------------------------------------------------------------- E1 preview helper
def random_requests(rng, sky, n, min_sep=3.0, window=30.0, pad_clearance=1.5, edge=2.0):
    """n requests between free points of layer 0's map that have a route, pads >= min_sep apart,
    >= pad_clearance from every building and >= edge from the border. Identical to the E1
    scripts' version (PR #27): no largest-part rule here. pad_clearance = edge = 0 gives its
    output from before pad clearance."""
    g = sky._grid(0)
    ok = ~g.occ
    if pad_clearance > 0 or edge > 0:
        ok = ok & pad_mask(g, sky.layout, sky.bounds, pad_clearance, edge)
    free = np.argwhere(ok)
    if len(free) == 0:
        raise ValueError(f"pad_clearance {pad_clearance} m and edge {edge} m leave no free cell for pads")
    pads, out, tries = [], [], 0
    while len(out) < n and tries < 50 * n:
        tries += 1
        a, b = (g.to_world(tuple(free[rng.randint(len(free))])) for _ in range(2))
        if np.hypot(a[0] - b[0], a[1] - b[1]) < 8.0:
            continue
        if any(np.hypot(p[0] - q[0], p[1] - q[1]) < min_sep for p in (a, b) for q in pads):
            continue
        if sky.route(a, b, 0) is None:
            continue
        pads += [a, b]
        out.append(FlightPlan(f"f{len(out)}", a, b, t_earliest=float(rng.uniform(0, window))))
    return out
