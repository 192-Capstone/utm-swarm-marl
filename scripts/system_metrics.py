"""
system_metrics.py
-----------------
E7, plan only: D4 demand at each density level -> E1 -> system metrics (utm/metrics.py):
throughput, delay, flight hours, approval and delay per operator, fairness (with the number of
requests per operator next to it: with 15-30 requests, one flight moves an operator's approval by
3-7 points). Nothing is flown;
safety per flight hour needs a flown run (scripts/eval_layered.py) and utm.metrics.safety_per_flight_hour.

    python scripts/system_metrics.py                                         # crop + medium city
    python scripts/system_metrics.py --modes random hubs --levels 5 10 20 40 --seeds 20 21 22
    python scripts/system_metrics.py --shares 0.7 0.1 0.1 0.1                # one big operator

Output: eval_results/system_metrics/<name>.csv and .md (not tracked). Seeds 10-19 are refused.
"""

import argparse
import csv
import os
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from demand_sweep import RESERVED, parse_map  # noqa: E402
from preview_route import load_map  # noqa: E402
from utm.demand import DemandSettings, Region, generate_demand, layer0_grid  # noqa: E402
from utm.metrics import system_metrics  # noqa: E402
from utm.strategic_planner import StrategicPlanner  # noqa: E402

COLUMNS = ["map", "mode", "level", "seed", "requests", "approved_pct", "throughput_per_h", "delay_mean_s",
           "delay_p95_s", "flight_h", "fairness_jain", "delay_gap_s", "op_requests_min", "op_requests_max",
           "per_operator"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--maps", nargs="+", default=["osm-crop", "city:medium"])
    ap.add_argument("--modes", nargs="+", default=["random", "hubs"], choices=["random", "hubs"])
    ap.add_argument("--levels", nargs="+", type=float, default=[5, 10, 20, 40])
    ap.add_argument("--seeds", nargs="+", type=int, default=[20, 21, 22])
    ap.add_argument("--duration", type=float, default=120.0, help="s of departures")
    ap.add_argument("--shares", nargs="+", type=float, default=None, help="operator shares (default equal, 4)")
    ap.add_argument("--name", default="e7")
    args = ap.parse_args()
    if any(s in RESERVED for s in args.seeds):
        ap.error("seeds 10-19 are reserved for the final test run")
    ops = None if args.shares is None else {f"op{i}": v for i, v in enumerate(args.shares)}

    out_dir = os.path.join("eval_results", "system_metrics")
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for spec in args.maps:
        label, _, ns = parse_map(spec)
        layout, bounds = load_map(ns)
        region = Region(layer0_grid(layout, bounds))
        for mode in args.modes:
            for level in args.levels:
                for seed in args.seeds:
                    s = DemandSettings(rate=level, duration_s=args.duration, mode=mode)
                    if ops:
                        s.operators = ops
                    dem = generate_demand(layout, bounds, s, seed=seed, region=region)
                    sky = StrategicPlanner(layout, bounds)
                    for fp in dem.plans:
                        sky.submit(fp)
                    m = system_metrics(sky, dem.plans, args.duration)
                    rows.append({
                        "map": label, "mode": mode, "level": f"{level:g}", "seed": seed, "requests": m["requests"],
                        "approved_pct": f"{m['approved_pct']:.1f}", "throughput_per_h": f"{m['throughput_per_h']:.0f}",
                        "delay_mean_s": f"{m['delay_mean_s']:.1f}", "delay_p95_s": f"{m['delay_p95_s']:.1f}",
                        "flight_h": f"{m['flight_s'] / 3600:.3f}", "fairness_jain": f"{m['fairness_jain']:.3f}",
                        "delay_gap_s": f"{m['delay_gap_s']:.1f}",
                        "op_requests_min": min(o["requests"] for o in m["per_operator"].values()),
                        "op_requests_max": max(o["requests"] for o in m["per_operator"].values()),
                        "per_operator": " ".join(f"{k}:{o['requests']}/{o['approved_pct']:.0f}%/"
                                                 f"{'-' if o['delay_mean_s'] is None else round(o['delay_mean_s'], 1)}s"
                                                 for k, o in m["per_operator"].items()),
                    })
                    r = rows[-1]
                    print(f"{label:16s} {mode:6s} {level:>4g}/min seed {seed}: {r['requests']:>4} req, "
                          f"{r['approved_pct']:>5}% approved, {r['throughput_per_h']:>5}/h, delay {r['delay_mean_s']} s, "
                          f"Jain {r['fairness_jain']}, delay gap {r['delay_gap_s']} s", flush=True)

    with open(os.path.join(out_dir, f"{args.name}.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    md = [f"seeds {' '.join(map(str, args.seeds))}, {args.duration:g} s of departures, operators "
          f"{ops or 'equal (4)'}; mean over seeds", "",
          "| map | mode | flights/min/district | requests | approved % | throughput /h | mean delay s | p95 delay s "
          "| Jain (approval) | operator delay gap s | requests per operator (min-max over runs) |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for key in dict.fromkeys((r["map"], r["mode"], r["level"]) for r in rows):
        g = [r for r in rows if (r["map"], r["mode"], r["level"]) == key]
        mean = lambda c: np.mean([float(r[c]) for r in g])  # noqa: E731
        md.append(f"| {' | '.join(key)} | {mean('requests'):.0f} | {mean('approved_pct'):.1f} | "
                  f"{mean('throughput_per_h'):.0f} | {mean('delay_mean_s'):.1f} | {mean('delay_p95_s'):.1f} | "
                  f"{mean('fairness_jain'):.3f} | {mean('delay_gap_s'):.1f} | "
                  f"{min(r['op_requests_min'] for r in g)}-{max(r['op_requests_max'] for r in g)} |")
    with open(os.path.join(out_dir, f"{args.name}.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
