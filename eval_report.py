"""
eval_report.py — One Markdown table from every eval_suite.py summary.json.

    python eval_report.py                       # eval_results/v1
    python eval_report.py --root eval_results/v1 --out eval_results/v1/report.md
"""

import argparse
import glob
import json
import os


def fmt(x, nd=3):
    if x is None or (isinstance(x, float) and x != x):
        return "-"
    return f"{x:.{nd}f}" if isinstance(x, float) else str(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join("eval_results", "v1"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows, irows = [], []
    for path in sorted(glob.glob(os.path.join(args.root, "*", "summary.json"))):
        with open(path) as f:
            s = json.load(f)
        c = s["config"]
        if s.get("kind") == "interaction":
            lo, hi = s["collision_ci95"]
            extra = {"intruder": f"moved >0.1 m {fmt(s.get('moved_over_0.1m_rate'))}, "
                                 f"back on goal {fmt(s.get('back_on_goal_rate'))}",
                     "blocker": f"reached goal {fmt(s.get('reached_goal_rate'))}, "
                                f"detour {fmt(s.get('max_lateral_median'))} m",
                     "neighbour": f"moved away {fmt(s.get('moved_away_mean'))} m, "
                                  f"sep change {fmt(s.get('sep_change_mean'))} m"}[c["scenario"]]
            irows.append([c["scenario"], c["policy"], c["collision_physics"], str(s["episodes"]),
                          f"{fmt(s['collision_rate'])} [{fmt(lo)}-{fmt(hi)}]",
                          fmt(s["sep_below_0.3_rate"]), fmt(s["min_sep"]["median"]), extra])
            continue
        lo, hi = s["success_ci95"]
        rows.append([
            c["policy"], c["scenario"],
            (f"{c['goal_dist_max']:g} m" if c.get("goal_dist_max")
             else "1-2 m" if c["scenario"] == "random" else "scenario"),
            str(c["n_agents"]), c["collision_physics"], str(s["episodes"]),
            fmt(s["mean_initial_goal_dist"], 1),
            f"{fmt(s['success_rate'])} [{fmt(lo)}-{fmt(hi)}]",
            fmt(s["per_agent_success_pooled"]),
            fmt(s["collision_rate"]),
            fmt(s.get("sep_below_0.3_rate")),
            fmt(s.get("physical_contact_rate")),
            fmt(s.get("min_sep", {}).get("min")),
            fmt(s["timeout_rate"]),
            fmt(s["final_goal_err"]["p95"]),
            fmt(s["settle_time_s"]["median"], 2),
            fmt(s["path_efficiency_mean"]),
            fmt(s["obs_clip_frac_mean"], 4),
        ])

    header = ["policy", "scenario", "goal dist", "N", "physics", "eps", "mean dist (m)",
              "success [95% CI]", "per-agent", "collision <0.2", "sep <0.3", "contact",
              "min sep", "timeout", "goal err p95", "settle med (s)", "path eff", "obs clip"]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    if irows:
        ih = ["scenario", "policy", "physics", "eps", "ego collided [95% CI]",
              "sep <0.3", "min sep median", "behaviour"]
        lines += ["", "**Two-drone interaction tests** (drone_1 scripted)", "",
                  "| " + " | ".join(ih) + " |", "|" + "---|" * len(ih)]
        lines += ["| " + " | ".join(r) + " |" for r in irows]
    table = "\n".join(lines)
    print(table)
    if args.out:
        with open(args.out, "w") as f:
            f.write(table + "\n")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
