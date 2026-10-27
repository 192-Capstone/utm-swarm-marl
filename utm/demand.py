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

Origins / destinations (mode):
    random  free points of the largest connected free region of layer 0's map (every pair is
            then connected), trips at least min_trip m long; with max_trip set, the goal is
            drawn at a uniform distance in [min_trip, max_trip] around the start (short trips
            on a big map),
            pads at least pad_sep m from the pads of flights departing within pad_window_s
    hubs    n_hubs depots spread over the region (farthest-point) and n_drops drop points;
            a share hub_share of the flights go depot -> drop point, the rest drop point -> depot
    or pass `pads`: a plain list of (x, y) or (x, y, kind), e.g. from D2's vertiports later;
    'random' then picks two pads per flight, 'hubs' uses kind 'hub' as depots and the rest as
    drop points.

Every request is checked for a D7 route on layer 0 (E1's own layer-0 map). Unroutable ones are
dropped and counted (report['dropped_unroutable']), so E1's approval rate measures airspace
capacity, not map connectivity; keep_unroutable=True keeps them (raw demand).

Each flight also gets an operator id (op0, op1, ... with configurable shares) for the fairness
metrics (E7) and a preferred cruise layer.

random_requests() is the small fixed-count generator the E1 preview has used since E1; it is
kept unchanged for that.
"""

import json
from collections import Counter
from dataclasses import asdict, dataclass, field, fields

import numpy as np

from envs.route_planner import plan
from utm.strategic_planner import FlightPlan, StrategicPlanner

DISTRICT_AREA = 2500.0                          # m2: one 50 m x 50 m D1 city
LEVELS = (5, 10, 20, 40, 80)                    # flights / min / district: low -> very high
FORMAT = 'utm-demand'
VERSION = 1


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


@dataclass
class Pad:
    x: float
    y: float
    kind: str = ''                              # e.g. 'hub', 'drop'; free text


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


class Region:
    """The largest connected free region of a grid, as horizontal runs of free cells.
    Connectivity is 4-neighbour, which is exactly what A* (8 moves, no corner cutting) can reach.
    Works from runs rather than cells, so the full E-City map (~8.5 M cells) takes a second."""

    def __init__(self, grid):
        self.grid = grid
        free = ~grid.occ
        ny = free.shape[0]
        pad = np.zeros((ny, 1), dtype=bool)
        d = np.diff(np.hstack([pad, free, pad]).astype(np.int8), axis=1)
        rows, starts = np.nonzero(d == 1)            # run k: row rows[k], columns [starts, ends)
        ends = np.nonzero(d == -1)[1]                # (same row-major order as the starts)
        n = len(rows)
        if n == 0:
            raise ValueError("the map has no free cell")
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
        roots = np.array([find(k) for k in range(n)], dtype=np.int64)
        lengths = (ends - starts).astype(np.int64)
        size = np.bincount(roots, weights=lengths)
        keep = roots == int(np.argmax(size))
        self.rows, self.starts, self.ends = rows[keep], starts[keep], ends[keep]
        self.n_cells = int(lengths[keep].sum())
        self.free_cells = int(lengths.sum())
        self._cdf = np.cumsum(lengths[keep]) / lengths[keep].sum()   # pick a run by its length
        self._row_first = np.searchsorted(self.rows, np.arange(ny + 1))

    def sample(self, rng):
        """A uniformly random cell centre (x, y) of the region."""
        k = min(int(np.searchsorted(self._cdf, rng.uniform(), side='right')), len(self.rows) - 1)
        col = self.starts[k] + rng.randint(self.ends[k] - self.starts[k])
        return self.grid.to_world((int(self.rows[k]), int(col)))

    def contains(self, xy):
        iy, ix = self.grid.to_cell(xy)
        if not self.grid.inside(xy):
            return False
        lo, hi = self._row_first[iy], self._row_first[iy + 1]
        return bool(np.any((self.starts[lo:hi] <= ix) & (ix < self.ends[lo:hi])))


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


def hub_pads(rng, region, n_hubs, n_drops, min_sep=3.0, candidates=400):
    """n_hubs depots spread over the region (farthest-point sampling over random candidates), then
    n_drops drop points at least min_sep from every other pad."""
    cand = np.array([region.sample(rng) for _ in range(candidates)])
    hubs = [cand[rng.randint(len(cand))]]
    while len(hubs) < n_hubs:
        d = np.min(np.linalg.norm(cand[:, None, :] - np.array(hubs)[None, :, :], axis=2), axis=1)
        hubs.append(cand[int(np.argmax(d))])
    pads = [Pad(float(x), float(y), 'hub') for x, y in hubs]
    tries = 0
    while sum(p.kind == 'drop' for p in pads) < n_drops and tries < 100 * n_drops:
        tries += 1
        x, y = region.sample(rng)
        if all(np.hypot(x - p.x, y - p.y) >= min_sep for p in pads):
            pads.append(Pad(float(x), float(y), 'drop'))
    return pads


# ----------------------------------------------------------------------------- generator
def generate_demand(layout, bounds, settings=None, seed=0, pads=None, planner_settings=None):
    """Demand for one map: E1 FlightPlans sorted by departure time, plus a report.
    layout, bounds: as for StrategicPlanner. pads: optional list of (x, y[, kind]) / Pad.
    planner_settings: E1 Settings whose layer-0 map is used for sampling and the route check."""
    s = settings or DemandSettings()
    if s.mode not in ('random', 'hubs'):
        raise ValueError(f"mode must be 'random' or 'hubs', got {s.mode!r}")
    if s.operators:
        ops = list(s.operators)
        w = np.array([float(s.operators[k]) for k in ops])
        if (w < 0).any() or w.sum() <= 0:
            raise ValueError("operators: shares must be >= 0 and not all 0")
        w = w / w.sum()
    rng = np.random.RandomState(seed)
    grid = layer0_grid(layout, bounds, planner_settings)
    region = Region(grid)
    pads = _as_pads(pads) if pads is not None else (
        hub_pads(rng, region, s.n_hubs, s.n_drops, s.pad_sep) if s.mode == 'hubs' else [])
    hubs = [p for p in pads if p.kind == 'hub']
    others = [p for p in pads if p.kind != 'hub']
    if s.mode == 'hubs' and (not hubs or not others):
        raise ValueError("mode 'hubs' needs at least one pad of kind 'hub' and one other pad")

    times = departure_times(rng, rate_per_min(s, bounds), s.duration_s, s.peak)
    plans, recent = [], []                      # recent: (t, start, goal) for pad_sep
    dropped = no_pair = 0
    for t in times:
        pair = None
        for _ in range(s.max_tries):
            if s.mode == 'hubs':
                h, o = hubs[rng.randint(len(hubs))], others[rng.randint(len(others))]
                a, b = ((h.x, h.y), (o.x, o.y)) if rng.uniform() < s.hub_share else ((o.x, o.y), (h.x, h.y))
            elif pads:
                i, j = rng.choice(len(pads), 2, replace=False)
                a, b = (pads[i].x, pads[i].y), (pads[j].x, pads[j].y)
            elif s.max_trip is not None:         # goal drawn around the start: a big map with
                a = region.sample(rng)           # short trips would reject almost every pair
                r = rng.uniform(s.min_trip, s.max_trip)
                ang = rng.uniform(0.0, 2 * np.pi)
                q = (a[0] + r * np.cos(ang), a[1] + r * np.sin(ang))
                if not region.contains(q):
                    continue
                b = grid.to_world(grid.to_cell(q))
            else:
                a, b = region.sample(rng), region.sample(rng)
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
        if plan(grid, a, b) is None:
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
    }
    return Demand(plans, pads, s, report, seed, tuple(tuple(map(float, b)) for b in bounds))


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
    if data.get('version') != VERSION:
        raise ValueError(f"{path}: demand format version {data.get('version')}, expected {VERSION}")
    known = {f.name for f in fields(DemandSettings)}
    s = {k: v for k, v in data['settings'].items() if k in known}
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
def random_requests(rng, sky, n, min_sep=3.0, window=30.0):
    """n requests between free points of layer 0's map that have a route, pads >= min_sep apart.
    (The E1 preview's fixed-count generator, moved here unchanged.)"""
    g = sky._grid(0)
    free = np.argwhere(~g.occ)
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
