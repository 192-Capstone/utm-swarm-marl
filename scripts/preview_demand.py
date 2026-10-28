"""
preview_demand.py
-----------------
A quick look at D4 demand (utm/demand.py): every flight request as a start -> goal line on the
map, coloured by operator, one panel per mode (random | hubs | hubs with a cluster of pads per
depot). Light green: the connected free region; dark green: the pad area inside it (at least
pad_clearance from buildings and edge from the border), where pads are drawn. Obstacles are grey.

    python scripts/preview_demand.py --map osm-crop --rate 20
    python scripts/preview_demand.py --map city --level medium --rate 20 --seed 3

Output: eval_results/demand_preview/demand_<map>.png (eval_results/ is not tracked by git).
"""

import argparse
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from preview_route import draw_obstacle, load_map  # noqa: E402
from utm.demand import DemandSettings, Region, generate_demand, layer0_grid, pad_area  # noqa: E402
from utm.demand_checks import check_seeds  # noqa: E402

OP_COLOURS = ["#2f6fb3", "#e08a1e", "#3a9a5b", "#a04ac0", "#c0392b", "#7f8c8d"]
PANELS = (("random", dict(mode="random")), ("hubs", dict(mode="hubs")),
          ("hubs, 3 pads per depot", dict(mode="hubs", pads_per_hub=3)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--map", choices=["city", "osm-crop"], default="osm-crop")
    ap.add_argument("--level", default="medium", help="D1 city level (with --map city)")
    ap.add_argument("--seed", type=int, default=0, help="demand seed (the city itself uses seed 0)")
    ap.add_argument("--rate", type=float, default=20.0, help="flights / min / district")
    ap.add_argument("--duration", type=float, default=120.0)
    ap.add_argument("--pad-clearance", type=float, default=DemandSettings.pad_clearance)
    ap.add_argument("--edge", type=float, default=DemandSettings.edge)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    check_seeds([args.seed])

    layout, bounds = load_map(argparse.Namespace(map=args.map, level=args.level, seed=0))
    region = Region(layer0_grid(layout, bounds))
    g = region.grid
    shade = region.mask() * 1.0 + pad_area(region, layout, bounds, args.pad_clearance, args.edge).mask() * 1.0
    extent = (g.xs[0] - g.res / 2, g.xs[-1] + g.res / 2, g.ys[0] - g.res / 2, g.ys[-1] + g.res / 2)

    fig, axes = plt.subplots(1, len(PANELS), figsize=(20, 7.4))
    ops = None
    for ax, (title, kw) in zip(axes, PANELS):
        s = DemandSettings(rate=args.rate, duration_s=args.duration, pad_clearance=args.pad_clearance,
                           edge=args.edge, **kw)
        d = generate_demand(layout, bounds, s, seed=args.seed, region=region)
        ops = list(s.operators)
        ax.imshow(np.where(shade > 0, shade, np.nan), origin="lower", extent=extent, cmap="Greens",
                  vmin=0, vmax=3.2, alpha=0.6, interpolation="nearest")
        for ob in layout:
            draw_obstacle(ax, ob)
        for p in d.plans:
            c = OP_COLOURS[ops.index(p.operator) % len(OP_COLOURS)] if p.operator in ops else "0.3"
            ax.plot([p.start[0], p.goal[0]], [p.start[1], p.goal[1]], "-", color=c, lw=0.9, alpha=0.7)
            ax.plot(*p.start, "o", color=c, ms=2.5)
        for q in d.pads:
            if q.kind == "hub":
                ax.plot(q.x, q.y, "*", color="black", ms=15, mec="white", mew=1.0, zorder=5)
            else:
                ax.plot(q.x, q.y, "s", color="white", ms=5, mec="black", mew=0.8, zorder=5)
        r = d.report
        cl = "-" if r["pad_clearance_min"] is None else f"{r['pad_clearance_min']:.2f}"
        ed = "-" if r["pad_edge_min"] is None else f"{r['pad_edge_min']:.2f}"
        ax.set_title(f"{title}: {r['kept']} of {r['requested']} requests kept, {r['no_valid_pair']} no pair\n"
                     f"pads placed {r['pads_placed'] or '-'}; closest pad {cl} m to a building, "
                     f"{ed} m to the edge", fontsize=9.5)
        ax.set_xlim(*bounds[0]); ax.set_ylim(*bounds[1]); ax.set_aspect("equal")
        ax.tick_params(labelsize=8)

    handles = [Line2D([], [], color=OP_COLOURS[i % len(OP_COLOURS)], label=k) for i, k in enumerate(ops)]
    handles += [Line2D([], [], marker="o", ls="", color="0.3", ms=4, label="start"),
                Line2D([], [], marker="*", ls="", color="black", ms=12, label="depot (hub)"),
                Line2D([], [], marker="s", ls="", color="white", mec="black", ms=6, label="drop point"),
                Line2D([], [], marker="s", ls="", color="#c7e9c0", ms=10, label="connected free region"),
                Line2D([], [], marker="s", ls="", color="#74c476", ms=10, label="pad area")]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, fontsize=9)
    name = args.map if args.map != "city" else f"city-{args.level}"
    fig.suptitle(f"D4 demand on {name}: {args.rate:g} flights / min / district for {args.duration:g} s, "
                 f"seed {args.seed}, pads >= {args.pad_clearance:g} m from buildings and "
                 f">= {args.edge:g} m from the edge", fontsize=12)
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    out = args.out or os.path.normpath(os.path.join(ROOT, "eval_results", "demand_preview", f"demand_{name}.png"))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=110)
    print(f"saved {os.path.normpath(out)}")


if __name__ == "__main__":
    main()
