"""
demand_checks.py
----------------
Checks for D4 demand (utm/demand.py), shared by the tests and scripts/demand_stats.py.

    check_demand(d, region)        invariants of one Demand -> list of problems ([] = all fine)
    ks_exponential(x, rate)        Kolmogorov-Smirnov test of samples against Exp(rate)
    chi_square(obs, exp)           Pearson chi-square goodness of fit
    poisson_dispersion(counts, m)  is the variance of Poisson counts what it should be
    peak_expected(...)             expected departures per time bin under the peak curve
    quarter_shares(region)         share of the region's cells in each map quarter
    safe_seeds(n)                  seeds for statistics, never the reserved test-run seeds 10-14

numpy + math only (scipy is not a dependency). p-values: Kolmogorov's asymptotic series, and the
regularized incomplete gamma function for chi-square.
"""

import math

import numpy as np

from envs.route_planner import plan
from utm.demand import edge_distance, pad_area, surface_clearance

RESERVED_SEEDS = range(10, 15)                  # the final test run's seeds: never used here


# ----------------------------------------------------------------------------- seeds
def check_seeds(seeds):
    bad = sorted(set(int(s) for s in seeds) & set(RESERVED_SEEDS))
    if bad:
        raise ValueError(f"seeds {bad} are reserved for the final test run")
    return list(seeds)


def safe_seeds(n, start=100):
    """n consecutive seeds from start, skipping the reserved ones."""
    out, s = [], start
    while len(out) < n:
        if s not in RESERVED_SEEDS:
            out.append(s)
        s += 1
    return out


# ----------------------------------------------------------------------------- statistics
def kolmogorov_p(d, n):
    """P(D_n >= d) for the one-sample KS statistic (asymptotic, with Stephens' small-n factor)."""
    if d <= 0 or n <= 0:
        return 1.0
    en = math.sqrt(n)
    lam = (en + 0.12 + 0.11 / en) * d
    if lam < 0.27:                              # series converges slowly here, and p is ~1
        return 1.0
    s = sum((-1) ** (k - 1) * math.exp(-2.0 * k * k * lam * lam) for k in range(1, 101))
    return min(max(2.0 * s, 0.0), 1.0)


def ks_exponential(x, rate):
    """(D, p): KS test of samples x against the exponential distribution with this rate (1/s)."""
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    if n == 0:
        return 0.0, 1.0
    cdf = 1.0 - np.exp(-rate * x)
    i = np.arange(1, n + 1)
    d = float(max(np.max(i / n - cdf), np.max(cdf - (i - 1) / n)))
    return d, kolmogorov_p(d, n)


def gammq(a, x):
    """Regularized upper incomplete gamma Q(a, x) (series below a + 1, continued fraction above)."""
    if a <= 0 or x < 0:
        raise ValueError("gammq needs a > 0 and x >= 0")
    if x == 0:
        return 1.0
    log_front = -x + a * math.log(x) - math.lgamma(a)
    if x < a + 1.0:
        ap, term = a, 1.0 / a
        total = term
        for _ in range(10000):
            ap += 1.0
            term *= x / ap
            total += term
            if abs(term) < abs(total) * 1e-15:
                break
        return min(max(1.0 - total * math.exp(log_front), 0.0), 1.0)
    tiny = 1e-300
    b = x + 1.0 - a
    c, d = 1.0 / tiny, 1.0 / b
    h = d
    for i in range(1, 10000):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        d = tiny if abs(d) < tiny else d
        c = b + an / c
        c = tiny if abs(c) < tiny else c
        d = 1.0 / d
        step = d * c
        h *= step
        if abs(step - 1.0) < 1e-15:
            break
    return min(max(math.exp(log_front) * h, 0.0), 1.0)


def chi2_sf(stat, dof):
    """P(X >= stat) for X ~ chi-square(dof)."""
    return gammq(dof / 2.0, stat / 2.0)


def chi_square(observed, expected, dof=None):
    """(stat, dof, p): Pearson chi-square of counts against expected counts. dof defaults to
    bins - 1 (expected scaled to the same total); pass bins when the total is not fixed."""
    o = np.asarray(observed, dtype=float)
    e = np.asarray(expected, dtype=float)
    if (e <= 0).any():
        raise ValueError("expected counts must be > 0")
    stat = float(np.sum((o - e) ** 2 / e))
    dof = len(o) - 1 if dof is None else int(dof)
    return stat, dof, chi2_sf(stat, dof)


def poisson_dispersion(counts, mean):
    """(variance / mean, p): two-sided test that counts with known Poisson mean have variance = mean,
    using (N - 1) s^2 / mean ~ chi-square(N - 1)."""
    c = np.asarray(counts, dtype=float)
    n = len(c)
    stat = (n - 1) * float(np.var(c, ddof=1)) / mean
    upper = chi2_sf(stat, n - 1)
    return float(np.var(c, ddof=1)) / mean, min(1.0, 2.0 * min(upper, 1.0 - upper))


def peak_expected(rate_min, edges, peak=None):
    """Expected departures in each bin [edges[i], edges[i+1]) for one run, at rate_min flights per
    minute with the optional (centre_s, width_s, factor) peak of departure_times()."""
    lam = rate_min / 60.0
    edges = np.asarray(edges, dtype=float)
    out = lam * np.diff(edges)
    if peak is not None:
        c, w, f = (float(v) for v in peak)
        z = (edges - c) / (w * math.sqrt(2.0))
        bump = w * math.sqrt(math.pi / 2.0) * np.diff([math.erf(v) for v in z])
        out = out + lam * (f - 1.0) * bump
    return out


# ----------------------------------------------------------------------------- space
def quarter_of(points, centre):
    """Quarter index per point: 0 = x < cx, y < cy; 1 = x >= cx, y < cy; 2, 3 the same with y >= cy."""
    p = np.asarray(points, dtype=float).reshape(-1, 2)
    return (p[:, 0] >= centre[0]).astype(int) + 2 * (p[:, 1] >= centre[1]).astype(int)


def quarter_shares(region, centre):
    """Share of the region's cells in each quarter (same split as quarter_of on cell centres)."""
    g = region.grid
    k = int(np.searchsorted(g.xs, centre[0], side='left'))       # columns with x < cx
    left = np.clip(np.minimum(region.ends, k) - region.starts, 0, None)
    right = (region.ends - region.starts) - left
    top = g.ys[region.rows] >= centre[1]
    n = np.array([left[~top].sum(), right[~top].sum(), left[top].sum(), right[top].sum()], dtype=float)
    return n / n.sum()


# ----------------------------------------------------------------------------- invariants
def check_demand(d, region, route=True, tol=1e-9, layout=None):
    """Problems with one Demand (an empty list when everything holds). region: the Region of the
    layer-0 grid it was generated on. route=True runs real A* on every kept request. layout: the
    map's obstacles; when given, every generated pad must be >= pad_clearance from every building
    (exact surface distance) and >= edge from the border."""
    s, r, plans = d.settings, d.report, d.plans
    out = []
    t = [p.t_earliest for p in plans]
    if any(b < a for a, b in zip(t, t[1:])):
        out.append("departure times are not sorted")
    if plans and (t[0] < 0.0 or t[-1] >= s.duration_s):
        out.append(f"departure times outside [0, {s.duration_s}): {t[0]:.3f} .. {t[-1]:.3f}")
    ids = [p.id for p in plans]
    if len(set(ids)) != len(ids):
        out.append("flight ids are not unique")

    if r['kept'] != len(plans):
        out.append(f"report kept {r['kept']} != {len(plans)} plans")
    if r['requested'] != r['kept'] + r['no_valid_pair'] + r['dropped_unroutable']:
        out.append(f"report: requested {r['requested']} != kept {r['kept']} + no_valid_pair "
                   f"{r['no_valid_pair']} + dropped_unroutable {r['dropped_unroutable']}")
    if sum(r['operators'].values()) != len(plans):
        out.append("report operators don't add up to the kept flights")
    for k, n in r['pads_placed'].items():
        if n > r['pads_requested'].get(k, n):
            out.append(f"more {k} pads placed ({n}) than requested ({r['pads_requested'][k]})")

    for p in plans:
        L = math.hypot(p.goal[0] - p.start[0], p.goal[1] - p.start[1])
        if L < s.min_trip - tol or (s.max_trip is not None and L > s.max_trip + tol):
            out.append(f"{p.id}: trip {L:.3f} m outside [{s.min_trip}, {s.max_trip}]")
        if p.layer not in s.layers:
            out.append(f"{p.id}: layer {p.layer} not in {s.layers}")
        if s.operators and p.operator not in s.operators:
            out.append(f"{p.id}: unknown operator {p.operator!r}")

    source = r['pads_source']
    if source == 'free':
        pts = [q for p in plans for q in (p.start, p.goal)]
    else:
        pts = [(q.x, q.y) for q in d.pads] if source == 'generated' else []
        pad_xy = {(q.x, q.y): q.kind for q in d.pads}
        for p in plans:
            if p.start not in pad_xy or p.goal not in pad_xy:
                out.append(f"{p.id}: start or goal is not one of the pads")
            elif s.mode == 'hubs' and (pad_xy[p.start] == 'hub') == (pad_xy[p.goal] == 'hub'):
                out.append(f"{p.id}: a hubs-mode flight needs exactly one hub end")
    for q in pts:
        if not region.contains(q):
            out.append(f"pad {tuple(round(v, 3) for v in q)} is outside the connected region")
    if pts and layout is not None:
        area = pad_area(region, layout, d.bounds, s.pad_clearance, s.edge)
        for q in pts:
            if region.contains(q) and not area.contains(q):
                out.append(f"pad {tuple(round(v, 3) for v in q)} is outside the pad area (its largest part)")
        cl, ed = surface_clearance(layout, pts), edge_distance(d.bounds, pts)
        for q, c, e in zip(pts, cl, ed):
            if c < s.pad_clearance - tol:
                out.append(f"pad {tuple(round(v, 3) for v in q)} is {c:.2f} m from a building "
                           f"(pad_clearance {s.pad_clearance})")
            if e < s.edge - tol:
                out.append(f"pad {tuple(round(v, 3) for v in q)} is {e:.2f} m from the border (edge {s.edge})")
    if source == 'generated':                   # depots' and drop points' pads pad_sep apart
        xy = sorted({(q.x, q.y) for q in d.pads})
        for i, a in enumerate(xy):
            for b in xy[i + 1:]:
                if math.hypot(a[0] - b[0], a[1] - b[1]) < s.pad_sep - tol:
                    out.append(f"pads {a} and {b} closer than pad_sep {s.pad_sep}")
        for q in d.pads:
            if q.kind == 'hub' and q.group >= 0:
                centre = next(c for c in d.pads if c.kind == 'hub' and c.group == q.group)
                if math.hypot(q.x - centre.x, q.y - centre.y) > s.hub_radius + tol:
                    out.append(f"depot {q.group} pad ({q.x}, {q.y}) beyond hub_radius {s.hub_radius}")

    if source == 'free' and s.pad_sep > 0:
        for i, p in enumerate(plans):
            for o in plans[i + 1:]:
                if o.t_earliest - p.t_earliest >= s.pad_window_s:
                    break
                for a in (p.start, p.goal):
                    for b in (o.start, o.goal):
                        if math.hypot(a[0] - b[0], a[1] - b[1]) < s.pad_sep - tol:
                            out.append(f"{p.id} / {o.id}: pads {math.hypot(a[0] - b[0], a[1] - b[1]):.2f} m "
                                       f"apart within {s.pad_window_s} s")

    if route and not s.keep_unroutable:
        for p in plans:
            if plan(region.grid, p.start, p.goal) is None:
                out.append(f"{p.id}: no A* route on layer 0")
    return out
