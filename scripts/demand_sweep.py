"""
demand_sweep.py
---------------
D4 x E1, plan only: generate demand at each density level and submit it to the strategic planner.
One row per (map, mode, level, seed): requests generated / dropped as unroutable / submitted,
E1 approval %, mean and p95 delay, layer use, and approval per operator (a first look at E7
fairness). This is the input for the capacity curves (approval and delay against demand).

    python scripts/demand_sweep.py                                   # default: crop + medium city
    python scripts/demand_sweep.py --maps city:easy city:medium city:hard osm-crop \
        --modes random hubs --levels 5 10 20 40 80 --seeds 0 1 2
    python scripts/demand_sweep.py --save-demand                     # also keep each demand JSON

Output: eval_results/demand_sweep/<name>.csv and .md (eval_results/ is not tracked by git).
A run at 80 flights / min takes about a minute in E1 (plan only, CPU); nothing is flown.
"""

import argparse
import csv
import os
import sys
import time

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from preview_route import load_map  # noqa: E402
from utm.demand import LEVELS, DemandSettings, generate_demand, save_demand  # noqa: E402
from utm.strategic_planner import StrategicPlanner  # noqa: E402

COLUMNS = ["map", "mode", "level", "seed", "requested", "dropped", "submitted", "approved_pct",
           "delay_mean_s", "delay_p95_s", "delayed_pct", "layer1_pct", "op_approved_pct", "seconds"]


def parse_map(spec):
    """'osm-crop', 'osm-full' or 'city:<level>[:<seed>]' -> (label, argparse namespace for load_map)."""
    parts = spec.split(":")
    if parts[0] == "city":
        level = parts[1] if len(parts) > 1 else "medium"
        seed = int(parts[2]) if len(parts) > 2 else 0
        return f"city-{level}-{seed}", argparse.Namespace(map="city", level=level, seed=seed)
    if parts[0] in ("osm-crop", "osm-full"):
        return parts[0], argparse.Namespace(map=parts[0], level="medium", seed=0)
    raise SystemExit(f"unknown map {spec!r}: use osm-crop, osm-full or city:<easy|medium|hard>[:seed]")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--maps", nargs="+", default=["osm-crop", "city:medium"])
    ap.add_argument("--modes", nargs="+", default=["random"], choices=["random", "hubs"])
    ap.add_argument("--levels", nargs="+", type=float, default=list(LEVELS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--duration", type=float, default=120.0, help="s of departures")
    ap.add_argument("--max-trip", type=float, default=None, help="m (useful on osm-full)")
    ap.add_argument("--name", default="sweep")
    ap.add_argument("--save-demand", action="store_true", help="also write each demand as JSON")
    args = ap.parse_args()

    out_dir = os.path.join("eval_results", "demand_sweep")
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for spec in args.maps:
        label, ns = parse_map(spec)
        layout, bounds = load_map(ns)
        for mode in args.modes:
            for level in args.levels:
                for seed in args.seeds:
                    t0 = time.time()
                    s = DemandSettings(rate=level, duration_s=args.duration, mode=mode, max_trip=args.max_trip)
                    dem = generate_demand(layout, bounds, s, seed=seed)
                    if args.save_demand:
                        save_demand(os.path.join(out_dir, f"{args.name}_{label}_{mode}_{level:g}_s{seed}.json"), dem)
                    sky = StrategicPlanner(layout, bounds)
                    for fp in dem.plans:
                        sky.submit(fp)
                    sm = sky.summary()
                    ok = {d.id for d in sky.decisions if d.status == "approved"}
                    per_op = {}
                    for fp in dem.plans:
                        n, a = per_op.get(fp.operator, (0, 0))
                        per_op[fp.operator] = (n + 1, a + (fp.id in ok))
                    n_ok = max(sm["approved"], 1)
                    rows.append({
                        "map": label, "mode": mode, "level": f"{level:g}", "seed": seed,
                        "requested": dem.report["requested"], "dropped": dem.report["dropped_unroutable"],
                        "submitted": sm["requests"], "approved_pct": f"{sm['approved_pct']:.1f}",
                        "delay_mean_s": f"{sm['delay_mean_s']:.1f}", "delay_p95_s": f"{sm['delay_p95_s']:.1f}",
                        "delayed_pct": f"{sm['delayed_pct']:.0f}",
                        "layer1_pct": f"{100.0 * sm['layer_use'].get(1, 0) / n_ok:.0f}",
                        "op_approved_pct": " ".join(f"{k}:{100.0 * a / n:.0f}" for k, (n, a) in sorted(per_op.items())),
                        "seconds": f"{time.time() - t0:.1f}",
                    })
                    r = rows[-1]
                    print(f"{label:16s} {mode:6s} {level:>4g}/min seed {seed}: {r['submitted']:>4} flights, "
                          f"{r['approved_pct']:>5}% approved, delay {r['delay_mean_s']} s (p95 {r['delay_p95_s']}), "
                          f"dropped {r['dropped']}, {r['seconds']} s", flush=True)

    csv_path = os.path.join(out_dir, f"{args.name}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)

    # summary over seeds: one line per (map, mode, level)
    md = ["| map | mode | flights/min/district | flights | approved % | mean delay s | p95 delay s |",
          "|---|---|---|---|---|---|---|"]
    keys = []
    for r in rows:
        k = (r["map"], r["mode"], r["level"])
        if k not in keys:
            keys.append(k)
    for k in keys:
        g = [r for r in rows if (r["map"], r["mode"], r["level"]) == k]
        mean = lambda c: np.mean([float(r[c]) for r in g])  # noqa: E731
        md.append(f"| {k[0]} | {k[1]} | {k[2]} | {mean('submitted'):.0f} | {mean('approved_pct'):.1f} | "
                  f"{mean('delay_mean_s'):.1f} | {mean('delay_p95_s'):.1f} |")
    md_path = os.path.join(out_dir, f"{args.name}.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print("\n" + "\n".join(md))
    print(f"\nsaved {csv_path} and {md_path}")


if __name__ == "__main__":
    main()
