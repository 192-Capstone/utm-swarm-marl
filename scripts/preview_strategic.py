"""
preview_strategic.py
--------------------
Picture of the E1 strategic planner (utm/strategic_planner.py) on a D1 city or the E-City crop.
Random flight requests between free points are submitted first come, first served (plan only:
nothing is flown). Three panels:

    the plan        every approved route, coloured by its cruise layer; delayed flights labelled
    two snapshots   the airspace booked at that moment: each flight's cells merged into one
                    outline (filled by the layer the drone is in), and a dot where the drone is.
                    Outlines in the same layer never overlap: that is E1's guarantee.

    python scripts/preview_strategic.py --map osm-crop --flights 30
    python scripts/preview_strategic.py --map city --level hard --seed 2 --flights 40

Output: eval_results/strategic_preview_<map>.png (eval_results/ is not tracked by git).
"""

import argparse
import os
import sys

import h3
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Polygon  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from preview_route import draw_obstacle, load_map  # noqa: E402
from envs.route_planner import build_grid  # noqa: E402
from utm.strategic_planner import FlightPlan, Settings, StrategicPlanner  # noqa: E402

LAYER_COLOURS = ["#2f6fb3", "#e08a1e", "#3a9a5b", "#a04ac0"]      # layer 0, 1, ...


def random_requests(rng, sky, n, min_sep=3.0, window=30.0, pad_clearance=1.5, edge=2.0):
    """n requests between pads that have a route on layer 0, pads >= min_sep apart, each pad at
    least pad_clearance from any building and edge from the map's border. (Pads hugging a wall,
    as close as the route map allows (~0.5 m), made the policy hover ~1 m off instead of settling:
    it keeps its trained distance from buildings. Real vertiports are not squeezed against walls
    either; D2's examples sit 2-7 m from buildings.)"""
    g = sky._grid(0)
    pad_map = build_grid(sky.layout, sky.bounds, g.res, pad_clearance)
    (x0, x1), (y0, y1) = sky.bounds
    gx, gy = np.meshgrid(g.xs, g.ys)
    inside = (gx >= x0 + edge) & (gx <= x1 - edge) & (gy >= y0 + edge) & (gy <= y1 - edge)
    free = np.argwhere(~g.occ & ~pad_map.occ & inside)
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


def outlines(sky, cells):
    """Merge H3 cells into outline polygons (local x, y)."""
    shape = h3.cells_to_h3shape(list(cells))
    polys = [shape] if hasattr(shape, "outer") else list(shape)
    return [np.array([sky.sky.to_local(lat, lng) for lat, lng in p.outer]) for p in polys]


def position(sky, d, t):
    """(x, y, z) of an approved flight at time t, or None when it is on the ground."""
    route = [p[:2] for p in d.waypoints[1:-1]]
    prof, total = sky.profile(route, d.layer)
    tau = t - d.t_dep
    if tau < 0 or tau > total:
        return None
    i = min(int(round(tau / sky.s.dt)), len(prof) - 1)
    return prof[i][1:]


def draw_base(ax, layout, bounds):
    for ob in layout:
        draw_obstacle(ax, ob)
    ax.set_xlim(*bounds[0]); ax.set_ylim(*bounds[1]); ax.set_aspect("equal")
    ax.set_facecolor("#f7f7f4")
    ax.tick_params(labelsize=8)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--map", choices=["city", "osm-crop"], default="osm-crop")
    ap.add_argument("--level", default="medium")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--flights", type=int, default=30)
    ap.add_argument("--pad-mode", choices=["window", "episode"], default="window")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    layout, bounds = load_map(args)
    sky = StrategicPlanner(layout, bounds, Settings(pad_mode=args.pad_mode))
    rng = np.random.RandomState(args.seed)
    for fp in random_requests(rng, sky, args.flights):
        sky.submit(fp)
    s = sky.summary()
    ok = [d for d in sky.decisions if d.status == "approved"]
    print(f"{s['approved']}/{s['requests']} approved, delay mean {s['delay_mean_s']:.1f} s "
          f"(p95 {s['delay_p95_s']:.1f} s), {s['delayed_pct']:.0f}% delayed, layers {s['layer_use']}")

    # the two busiest moments, at least 10 s apart
    ts = np.arange(0.0, max(d.arrival for d in ok), 0.5)
    busy = np.array([sum(position(sky, d, t) is not None for d in ok) for t in ts])
    t1 = ts[int(np.argmax(busy))]
    far = np.abs(ts - t1) >= 10.0
    t2 = ts[far][int(np.argmax(busy[far]))] if far.any() else t1
    snaps = sorted([t1, t2])

    fig, axes = plt.subplots(1, 3, figsize=(19, 6.6))
    ax = axes[0]
    draw_base(ax, layout, bounds)
    for d in ok:
        c = LAYER_COLOURS[d.layer]
        w = np.array([p[:2] for p in d.waypoints])
        ax.plot(w[:, 0], w[:, 1], "-", color=c, lw=1.6, alpha=0.9)
        ax.plot(*w[0], "o", color=c, ms=4)
        ax.plot(*w[-1], "x", color=c, ms=6, mew=1.8)
        if d.delay > 0:
            ax.annotate(f"+{d.delay:.0f}s", w[0], xytext=(3, 3), textcoords="offset points",
                        fontsize=7, color=c)
    ax.set_title(f"the plan: {s['approved']}/{s['requests']} approved, "
                 f"{s['delayed_pct']:.0f}% delayed (mean {s['delay_mean_s']:.0f} s)", fontsize=10)

    for ax, t in zip(axes[1:], snaps):
        draw_base(ax, layout, bounds)
        n_air = 0
        for d in ok:
            booked = {}
            for (cell, layer), ivs in d.reservation.items():
                if any(lo <= t <= hi for lo, hi in ivs):
                    booked.setdefault(layer, set()).add(cell)
            for layer, cells in booked.items():
                for poly in outlines(sky, cells):
                    ax.add_patch(Polygon(poly, closed=True, facecolor=LAYER_COLOURS[layer], alpha=0.22,
                                         edgecolor=LAYER_COLOURS[layer], lw=1.0))
            p = position(sky, d, t)
            if p is not None:
                n_air += 1
                ax.plot(p[0], p[1], "o", color=LAYER_COLOURS[sky.layer_of(p[2])], ms=5,
                        mec="white", mew=0.8)
        ax.set_title(f"airspace booked at t = {t:.1f} s ({n_air} drones in the air)", fontsize=10)

    handles = [Patch(facecolor=LAYER_COLOURS[k], alpha=0.5,
                     label=f"layer {k} ({sky.sky.layer_range(k)[0]:g}-{sky.sky.layer_range(k)[1]:g} m)")
               for k in range(sky.s.n_layers)]
    handles += [Line2D([], [], marker="o", ls="", color="0.3", label="start / drone"),
                Line2D([], [], marker="x", ls="", color="0.3", label="goal")]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, fontsize=9)
    fig.suptitle(f"E1 strategic planner on {args.map} (H3 resolution {sky.s.resolution}, "
                 f"pads: {args.pad_mode})", fontsize=12)
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    out = args.out or os.path.join("eval_results", f"strategic_preview_{args.map}.png")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=120)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
