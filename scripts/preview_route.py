"""
preview_route.py
----------------
Draw planned routes on a map, to check the route planner (D7) by eye.

Examples:
    python scripts/preview_route.py --map city --level medium --seed 0 --flights 6
    python scripts/preview_route.py --map osm-crop --flights 6
    python scripts/preview_route.py --map osm-full --res 0.5 --flights 6

Writes route_preview_<map>.png: buildings (grey), blocked cells (light red), the raw A* path
(thin), the simplified waypoints (thick, dots), starts (squares) and goals (crosses).
"""

import argparse
import math
import os
import sys
import time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from envs.route_planner import build_grid, path_length, plan, simplify  # noqa: E402
from envs.scenario_manager import DRONE_RADIUS  # noqa: E402

CROP = "data/osm_city/bengaluru_ecity_phase1_rotated_v2_crop50_cand1_layout.yaml"


def load_map(args):
    if args.map == "city":
        from envs.city_generator import generate_city

        city = generate_city(np.random.RandomState(args.seed), args.level)
        W, H = city["size"]
        return city["layout"], ((-W / 2, W / 2), (-H / 2, H / 2))

    from envs.osm_scenario import load_osm_layout, layout_bounds, DEFAULT_LAYOUT_PATH

    layout = load_osm_layout(
        CROP if args.map == "osm-crop" else DEFAULT_LAYOUT_PATH
    )

    if args.map == "osm-crop":
        # Keep planning strictly inside the 50 m crop.
        return layout, ((-25, 25), (-25, 25))

    x0, y0, x1, y1 = layout_bounds(layout)
    return layout, ((x0, x1), (y0, y1))


def draw_obstacle(ax, ob):
    if ob["type"] == "pillar":
        ax.add_patch(Circle(ob["center"], ob["radius"], color="0.45"))
        return

    hx, hy = ob["half"]
    c = math.cos(ob.get("yaw", 0.0))
    s = math.sin(ob.get("yaw", 0.0))

    pts = [
        (
            ob["center"][0] + c * u - s * v,
            ob["center"][1] + s * u + c * v,
        )
        for u, v in ((-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy))
    ]

    ax.add_patch(Polygon(pts, closed=True, color="0.45"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--map",
        choices=["city", "osm-crop", "osm-full"],
        default="city",
    )
    ap.add_argument("--level", default="medium")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--flights", type=int, default=6)
    ap.add_argument("--res", type=float, default=0.25)
    ap.add_argument("--inflate", type=float, default=DRONE_RADIUS + 0.35)
    args = ap.parse_args()

    layout, bounds = load_map(args)

    t0 = time.time()
    grid = build_grid(layout, bounds, args.res, args.inflate)
    t_grid = time.time() - t0
    print(
        f"grid build: {t_grid:.3f} s | "
        f"grid {grid.shape} @ {args.res:.2f} m | "
        f"inflate {args.inflate:.2f} m"
    )

    rng = np.random.RandomState(args.seed + 1)
    free = np.argwhere(~grid.occ)

    fig, ax = plt.subplots(figsize=(9, 9))
    (x0, x1), (y0, y1) = bounds

    ax.imshow(
        np.where(grid.occ, 1.0, np.nan),
        origin="lower",
        extent=(x0, x1, y0, y1),
        cmap="Reds",
        vmin=0,
        vmax=3,
        interpolation="nearest",
    )

    for ob in layout:
        draw_obstacle(ax, ob)

    colours = plt.cm.tab10.colors
    planned = 0

    for k in range(args.flights):
        a = grid.to_world(tuple(free[rng.randint(len(free))]))
        b = grid.to_world(tuple(free[rng.randint(len(free))]))

        t1 = time.time()
        raw = plan(grid, a, b)
        dt = time.time() - t1

        col = colours[k % 10]
        ax.plot(*a, "s", color=col, ms=7)
        ax.plot(*b, "x", color=col, ms=9, mew=2)

        if raw is None:
            print(f"flight {k}: no route ({dt:.2f} s)")
            continue

        planned += 1
        pts = simplify(grid, raw)

        ax.plot(*zip(*raw), "-", color=col, lw=0.8, alpha=0.6)
        ax.plot(*zip(*pts), "o-", color=col, lw=2.2, ms=4)

        print(
            f"flight {k}: {len(raw)} cells -> {len(pts)} waypoints, "
            f"{path_length(pts):.1f} m (straight {math.dist(a, b):.1f} m), "
            f"{dt:.2f} s"
        )

    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")

    ax.set_title(
        f"{args.map} grid {grid.shape} @ {args.res} m, "
        f"inflate {args.inflate:.2f} m, {planned}/{args.flights} planned, "
        f"grid built in {t_grid:.2f} s"
    )

    if args.map == "city":
        out = f"route_preview_city_{args.level}.png"
    else:
        out = f"route_preview_{args.map}-new.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print("saved", out)


if __name__ == "__main__":
    main()