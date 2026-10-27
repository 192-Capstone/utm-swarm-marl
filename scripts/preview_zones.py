"""
preview_zones.py
----------------
Picture of D2 no-fly zones (utm/airspace_zones.py) on a map, one panel per E1 altitude layer:
buildings, each zone's outline (red = prohibited, orange = authorisation, dashed = temporary,
with its time window), the H3 key cells zone_keys() gives for that layer, and the vertiports.
Also prints a table per zone and whether each vertiport is clear of the zones (E1 rejects a
flight whose pad reservation, radius 2.1 m, touches a zone key).

    python scripts/preview_zones.py --map ecity_crop50 --zones data/airspace/ecity_crop50_example.geojson
    python scripts/preview_zones.py --map d1_medium_seed3 --zones data/airspace/d1_medium_seed3_example.geojson

Output: eval_results/zones_preview_<map>.png (eval_results/ is not tracked by git), or --out.
"""

import argparse
import math
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Polygon  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from preview_route import draw_obstacle  # noqa: E402
from utm.airspace_zones import load_zones, map_airspace, zone_keys  # noqa: E402
from utm.strategic_planner import Settings, StrategicPlanner  # noqa: E402

CROP = os.path.join(ROOT, "data/osm_city/bengaluru_ecity_phase1_rotated_v2_crop50_cand1_layout.yaml")
COLOURS = {"prohibited": "#d62728", "authorisation": "#ff8c00"}
MAPS = ("ecity_crop50", "d1_medium_seed3")


def load_map(name):
    """(layout, bounds, airspace) for a map name."""
    if name == "ecity_crop50":
        from envs.osm_scenario import load_osm_layout

        return load_osm_layout(CROP), ((-25.0, 25.0), (-25.0, 25.0)), map_airspace("ecity_crop50")
    from envs.city_generator import generate_city

    city = generate_city(np.random.RandomState(3), "medium")
    W, H = city["size"]
    return city["layout"], ((-W / 2, W / 2), (-H / 2, H / 2)), map_airspace("d1")


def band(z):
    lo = "surface" if z.lower_m is None else f"{z.lower_m:g} m"
    hi = "no ceiling" if z.upper_m is None else f"{z.upper_m:g} m"
    return f"{lo} - {hi}"


def window(z):
    return "permanent" if z.permanent else ", ".join(f"{t0:g}-{t1:g} s" for t0, t1 in z.active)


def hex_distance(sky, cell, p):
    """Distance (m) from point p to a cell's hexagon (0 inside)."""
    poly = np.asarray(sky.boundary(cell))
    a, b = poly, np.roll(poly, -1, axis=0)
    ab, ap = b - a, np.asarray(p) - a
    t = np.clip(np.sum(ap * ab, axis=1) / np.sum(ab * ab, axis=1), 0.0, 1.0)
    d = float(np.min(np.linalg.norm(ap - t[:, None] * ab, axis=1)))
    cross = ab[:, 0] * ap[:, 1] - ab[:, 1] * ap[:, 0]
    return 0.0 if (np.all(cross >= 0) or np.all(cross <= 0)) else d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", required=True, choices=MAPS)
    ap.add_argument("--zones", required=True, help="GeoJSON zone file")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    layout, bounds, sky = load_map(args.map)
    zones, ports, meta = load_zones(args.zones, sky)
    s = Settings()
    keys = {z.id: zone_keys(z, sky, s.n_layers) for z in zones}

    # ---- table per zone
    print(f"{args.map}: {len(zones)} zones, {len(ports)} vertiports ({meta.get('description', '')})")
    print(f"{'zone':22s} {'restriction':14s} {'height band':22s} {'window':14s} "
          + " ".join(f"L{k} cells" for k in range(s.n_layers)) + "   area m2")
    for z in zones:
        per = [sum(1 for _, layer in keys[z.id] if layer == k) for k in range(s.n_layers)]
        print(f"{z.id:22s} {z.restriction:14s} {band(z):22s} {window(z):14s} "
              + " ".join(f"{n:8d}" for n in per) + f"   {z.area():7.1f}")

    # ---- vertiports vs zones. Pads are in layer 0; E1 reserves cells_near(pad) there.
    planner = StrategicPlanner(layout, bounds, s, airspace=sky)
    pad_layer = planner.layer_of(s.start_z)
    print(f"\nvertiports (pad layer {pad_layer}, E1 radius {s.radius} m):")
    for v in ports:
        near = set(planner.cells_near(v.x, v.y))
        notes, nearest = [], (math.inf, None)
        for z in zones:
            cells = {c for c, layer in keys[z.id] if layer == pad_layer}
            if not cells:
                continue
            inside = z.distance([(v.x, v.y)])[0] == 0
            gap = min(hex_distance(sky, c, (v.x, v.y)) for c in cells)
            nearest = min(nearest, (gap, z.id))
            shared = len(near & cells)
            if inside or gap <= s.radius or shared:
                notes.append(f"{z.id}: inside={inside}, {gap:.2f} m to its cells, "
                             f"{shared} pad cells shared ({window(z)})")
        status = "; ".join(notes) if notes else f"clear (nearest: {nearest[1]}, {nearest[0]:.2f} m to its cells)"
        print(f"  {v.id:20s} ({v.x:6.2f}, {v.y:6.2f}) {v.kind:9s} {status}")

    # ---- picture
    (xmin, xmax), (ymin, ymax) = bounds
    fig, axes = plt.subplots(1, s.n_layers, figsize=(7.5 * s.n_layers, 7.5))
    for k, ax in enumerate(np.atleast_1d(axes)):
        lo, hi = sky.layer_range(k)
        for ob in layout:
            draw_obstacle(ax, ob)
        n_cells = 0
        for z in zones:
            col = COLOURS[z.restriction]
            cells = [c for c, layer in keys[z.id] if layer == k]
            n_cells += len(cells)
            for c in cells:
                ax.add_patch(Polygon(sky.boundary(c), closed=True, facecolor=col, alpha=0.25,
                                     edgecolor=col, linewidth=0.3))
            outline = z.outline()
            ax.add_patch(Polygon(outline, closed=True, fill=False, edgecolor=col, linewidth=2.0,
                                 linestyle="--" if not z.permanent else "-"))
            cx, cy = np.mean(outline, axis=0)
            label = z.name if z.permanent or window(z) in z.name else f"{z.name}\n[{window(z)}]"
            if not cells:
                label += "\n(not in this layer)"
            ax.text(cx, cy, label, ha="center", va="center", fontsize=7, color=col, weight="bold",
                    bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=1))
        for v in ports:
            ax.plot(v.x, v.y, marker="^", color="#1f4fd1", markersize=9, markeredgecolor="k")
            ax.text(v.x + 0.6, v.y + 0.6, v.id, fontsize=7, color="#1f4fd1")
        ax.set_title(f"layer {k}: {lo:g}-{hi:g} m, {n_cells} key cells")
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal")
        ax.set_xlabel("x (m, east)")
        ax.set_ylabel("y (m, north)")
        ax.grid(alpha=0.2)
    handles = [Patch(facecolor="0.45", label="building"),
               Line2D([], [], color=COLOURS["prohibited"], lw=2, label="prohibited (red)"),
               Line2D([], [], color=COLOURS["authorisation"], lw=2, label="authorisation (yellow)"),
               Line2D([], [], color="k", lw=2, ls="--", label="temporary (dashed)"),
               Patch(facecolor="0.5", alpha=0.25, label="H3 key cells (res %d)" % sky.res),
               Line2D([], [], marker="^", color="#1f4fd1", ls="", markeredgecolor="k", label="vertiport")]
    fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=8, frameon=False)
    fig.suptitle(f"D2 zones: {args.map} ({os.path.basename(args.zones)}) - made-up example, not real restrictions")
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    out = args.out or os.path.join(ROOT, "eval_results", f"zones_preview_{args.map}.png")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"\nsaved {os.path.relpath(out, ROOT) if args.out is None else out}")


if __name__ == "__main__":
    main()
