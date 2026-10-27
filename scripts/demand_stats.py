"""
demand_stats.py
---------------
D4 demand (utm/demand.py) checked as a random generator, over many seeds:

    counts     flights per run: mean and variance vs rate x duration x area (Poisson: var = mean)
    gaps       inter-departure gaps vs the exponential distribution (KS test)
    peak       departure times vs the rush-hour curve (chi-square over time bins)
    operators  operator shares vs the configured shares (chi-square)
    hub_share  share of hubs-mode flights leaving a depot vs hub_share (chi-square)
    spread     pads over the free region: share per map quarter vs that quarter's share of the
               connected region (Region.sample: chi-square; generated starts: max difference)

    python scripts/demand_stats.py                  # all of the above, default number of seeds
    python scripts/demand_stats.py --seeds 400      # more seeds
    python scripts/demand_stats.py --invariants     # also the invariant check over every map x
                                                    # mode x level, 120 s, seeds 0-3, real A*

Seeds start at 100 and never include 10-14 (reserved for the final test run). The fast versions
of these checks are tests/test_demand_stats.py, which call the functions below with fewer seeds.
Output: printed, and eval_results/demand_stats/<name>.md (eval_results/ is not tracked by git).
"""

import argparse
import math
import os
import sys
import time

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)

from envs.city_generator import generate_city  # noqa: E402
from envs.osm_scenario import load_osm_layout  # noqa: E402
from utm.demand import (LEVELS, DemandSettings, Region, departure_times,  # noqa: E402
                        generate_demand, layer0_grid, rate_per_min)
from utm.demand_checks import (check_demand, check_seeds, chi_square, ks_exponential,  # noqa: E402
                               peak_expected, poisson_dispersion, quarter_of, quarter_shares,
                               safe_seeds)

CROP = os.path.join(ROOT, "data", "osm_city", "bengaluru_ecity_phase1_rotated_v2_crop50_cand1_layout.yaml")
B = ((-25.0, 25.0), (-25.0, 25.0))


def load_maps(names=("empty", "empty-wide", "easy", "medium", "hard", "crop")):
    """name -> (layout, bounds, Region of the layer-0 grid)."""
    out = {}
    for name in names:
        if name == "empty":
            layout, bounds = [], B
        elif name == "empty-wide":
            layout, bounds = [], ((-50.0, 50.0), (-25.0, 25.0))
        elif name == "crop":
            layout, bounds = load_osm_layout(CROP), B
        else:
            c = generate_city(np.random.RandomState(0), name)
            W, H = c["size"]
            layout, bounds = c["layout"], ((-W / 2, W / 2), (-H / 2, H / 2))
        out[name] = (layout, bounds, Region(layer0_grid(layout, bounds)))
    return out


def two_sided_p(z):
    return math.erfc(abs(z) / math.sqrt(2.0))


# ----------------------------------------------------------------------------- the checks
def stat_counts(maps, seeds, rates=(5, 20, 80), duration=120.0, names=("empty", "empty-wide", "crop")):
    """Requested flights per run vs the Poisson mean rate x area / 2,500 x duration / 60."""
    rows = []
    for name in names:
        layout, bounds, region = maps[name]
        for rate in rates:
            s = DemandSettings(rate=rate, duration_s=duration)
            mu = rate_per_min(s, bounds) * duration / 60.0
            n = np.array([generate_demand(layout, bounds, s, seed=k, region=region).report["requested"]
                          for k in seeds])
            z = (n.mean() - mu) / math.sqrt(mu / len(n))
            ratio, p_var = poisson_dispersion(n, mu)
            rows.append(dict(map=name, rate=rate, expected=mu, mean=float(n.mean()), p_mean=two_sided_p(z),
                             var_over_mean=ratio, p_var=p_var))
    return rows


def stat_gaps(seeds, rate=20.0, duration=3600.0):
    """Gaps between consecutive departures (one hour per seed, so cutting at the end is negligible)."""
    gaps = []
    for k in seeds:
        t = departure_times(np.random.RandomState(k), rate, duration)
        gaps.append(np.diff(np.concatenate([[0.0], t])))
    g = np.concatenate(gaps)
    lam = rate / 60.0
    d, p = ks_exponential(g, lam)
    return dict(n=len(g), mean_gap=float(g.mean()), expected_gap=1.0 / lam, ks_d=d, p=p)


def stat_peak(maps, seeds, rate=20.0, duration=120.0, peak=(60.0, 15.0, 3.0), bin_s=10.0):
    """Departure times of generated demand (empty map) vs the expected count per bin."""
    layout, bounds, region = maps["empty"]
    s = DemandSettings(rate=rate, duration_s=duration, peak=peak)
    t = np.concatenate([[p.t_earliest for p in generate_demand(layout, bounds, s, seed=k, region=region).plans]
                        for k in seeds])
    edges = np.arange(0.0, duration + bin_s / 2, bin_s)
    obs = np.histogram(t, edges)[0]
    exp = len(seeds) * peak_expected(rate_per_min(s, bounds), edges, peak)
    stat, dof, p = chi_square(obs, exp, dof=len(obs))            # total not fixed: dof = bins
    return dict(edges=edges, observed=obs, expected=exp, chi2=stat, dof=dof, p=p)


def stat_operators(maps, seeds, shares=None, name="medium"):
    shares = shares or {"a": 0.5, "b": 0.3, "c": 0.15, "d": 0.05}
    layout, bounds, region = maps[name]
    s = DemandSettings(rate=40, duration_s=120.0, operators=shares)
    ops = [p.operator for k in seeds for p in generate_demand(layout, bounds, s, seed=k, region=region).plans]
    keys = list(shares)
    obs = np.array([ops.count(k) for k in keys])
    exp = len(ops) * np.array([shares[k] for k in keys]) / sum(shares.values())
    stat, dof, p = chi_square(obs, exp)
    return dict(n=len(ops), shares=dict(zip(keys, (obs / len(ops)).round(4))), chi2=stat, dof=dof, p=p)


def stat_hub_share(maps, seeds, hub_share=0.8, name="medium"):
    layout, bounds, region = maps[name]
    s = DemandSettings(rate=40, duration_s=120.0, mode="hubs", hub_share=hub_share)
    out_of_hub = n = 0
    for k in seeds:
        d = generate_demand(layout, bounds, s, seed=k, region=region)
        hubs = {(q.x, q.y) for q in d.pads if q.kind == "hub"}
        out_of_hub += sum(p.start in hubs for p in d.plans)
        n += len(d.plans)
    stat, dof, p = chi_square([out_of_hub, n - out_of_hub], [n * hub_share, n * (1 - hub_share)])
    return dict(n=n, share=out_of_hub / n, expected=hub_share, chi2=stat, dof=dof, p=p)


def stat_spread(maps, seeds, draws=20000, names=("empty", "easy", "medium", "hard", "crop")):
    """Per map quarter: share of the connected region, of Region.sample draws, and of the starts of
    generated random-mode demand (trip length and pad spacing reject some pairs, so starts are close
    to, not exactly, uniform)."""
    rows = []
    for name in names:
        layout, bounds, region = maps[name]
        centre = (0.5 * sum(bounds[0]), 0.5 * sum(bounds[1]))
        area = quarter_shares(region, centre)
        rng = np.random.RandomState(seeds[0])
        q = np.bincount(quarter_of([region.sample(rng) for _ in range(draws)], centre), minlength=4)
        live = area > 0
        stat, dof, p = chi_square(q[live], draws * area[live])
        s = DemandSettings(rate=40, duration_s=120.0)
        starts = [p.start for k in seeds for p in generate_demand(layout, bounds, s, seed=k, region=region).plans]
        gen = np.bincount(quarter_of(starts, centre), minlength=4) / len(starts)
        rows.append(dict(map=name, region=area, sampled=q / draws, p_sample=p, starts=gen, n_starts=len(starts),
                         max_diff=float(np.max(np.abs(gen - area)))))
    return rows


def run_invariants(maps, seeds=(0, 1, 2, 3), duration=120.0, names=("easy", "medium", "hard", "crop")):
    seeds = check_seeds(seeds)
    modes = {"random": {}, "random-short": dict(max_trip=20.0), "hubs": dict(mode="hubs")}
    n_runs = n_flights = 0
    problems = []
    for name in names:
        layout, bounds, region = maps[name]
        for mode, kw in modes.items():
            for level in LEVELS:
                for k in seeds:
                    d = generate_demand(layout, bounds, DemandSettings(rate=level, duration_s=duration, **kw),
                                        seed=k, region=region)
                    found = check_demand(d, region, route=True)
                    problems += [f"{name} {mode} {level} seed {k}: {m}" for m in found]
                    n_runs += 1
                    n_flights += len(d.plans)
    return dict(runs=n_runs, flights=n_flights, problems=problems)


# ----------------------------------------------------------------------------- report
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=300, help="number of seeds (from 100, skipping 10-14)")
    ap.add_argument("--invariants", action="store_true", help="also the full invariant check (slow)")
    ap.add_argument("--name", default="stats")
    args = ap.parse_args()

    seeds = check_seeds(safe_seeds(args.seeds))
    t0 = time.time()
    maps = load_maps()
    md = [f"# D4 demand statistics ({len(seeds)} seeds, {seeds[0]}..{seeds[-1]})", ""]

    def out(line=""):
        print(line, flush=True)
        md.append(line)

    out("## Flight count per run (Poisson: mean = rate x area / 2,500 x duration / 60, variance = mean)")
    out("| map | rate | expected | mean | p (mean) | var / mean | p (var) |")
    out("|---|---|---|---|---|---|---|")
    for r in stat_counts(maps, seeds):
        out(f"| {r['map']} | {r['rate']} | {r['expected']:.1f} | {r['mean']:.2f} | {r['p_mean']:.3f} | "
            f"{r['var_over_mean']:.3f} | {r['p_var']:.3f} |")

    g = stat_gaps(seeds)
    out()
    out("## Inter-departure gaps vs exponential (20 / min, 1 h per seed)")
    out(f"{g['n']} gaps, mean {g['mean_gap']:.3f} s (expected {g['expected_gap']:.3f} s), "
        f"KS D = {g['ks_d']:.4f}, p = {g['p']:.3f}")

    pk = stat_peak(maps, seeds)
    out()
    out("## Departures vs the peak curve (20 / min, peak at 60 s, width 15 s, x3)")
    out("| bin (s) | observed | expected |")
    out("|---|---|---|")
    for a, b, o, e in zip(pk["edges"], pk["edges"][1:], pk["observed"], pk["expected"]):
        out(f"| {a:.0f}-{b:.0f} | {o} | {e:.1f} |")
    out(f"chi-square {pk['chi2']:.2f} on {pk['dof']} dof, p = {pk['p']:.3f}")

    op = stat_operators(maps, seeds)
    out()
    out("## Operator shares (configured a 0.5, b 0.3, c 0.15, d 0.05)")
    out(f"{op['n']} flights: {op['shares']}, chi-square {op['chi2']:.2f} on {op['dof']} dof, p = {op['p']:.3f}")

    hs = stat_hub_share(maps, seeds)
    out()
    out("## hub_share (hubs mode, configured 0.8)")
    out(f"{hs['n']} flights: {hs['share']:.4f} leave a depot, chi-square {hs['chi2']:.2f}, p = {hs['p']:.3f}")

    out()
    out("## Spread over the free region (quarters SW, SE, NW, NE)")
    out("| map | region share | Region.sample | p | generated starts | max diff |")
    out("|---|---|---|---|---|---|")
    fmt = lambda v: " ".join(f"{x:.3f}" for x in v)  # noqa: E731
    for r in stat_spread(maps, seeds[:60]):
        out(f"| {r['map']} | {fmt(r['region'])} | {fmt(r['sampled'])} | {r['p_sample']:.3f} | "
            f"{fmt(r['starts'])} | {r['max_diff']:.3f} |")

    if args.invariants:
        inv = run_invariants(maps)
        out()
        out("## Invariants (D1 easy / medium / hard + crop x random / random-short / hubs x levels 5-80 "
            "x seeds 0-3, 120 s, real A* on every request)")
        out(f"{inv['runs']} runs, {inv['flights']} flights, {len(inv['problems'])} problems")
        for m in inv["problems"][:20]:
            out(f"- {m}")

    out()
    out(f"({time.time() - t0:.0f} s)")
    out_dir = os.path.normpath(os.path.join(ROOT, "eval_results", "demand_stats"))
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{args.name}.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print(f"saved {path}")


if __name__ == "__main__":
    main()
