
"""D7 route planner benchmark."""
import time
import statistics
from pathlib import Path

import numpy as np

from envs.city_generator import generate_city
from envs.osm_scenario import load_osm_layout, layout_bounds
from envs.route_planner import build_grid, plan, simplify
from envs.scenario_manager import DRONE_RADIUS

CROP = Path(
    "data/osm_city/bengaluru_ecity_phase1_rotated_v2_crop50_cand1_layout.yaml"
)
FULL = Path(
    "data/osm_city/bengaluru_ecity_phase1_rotated_v2_layout.yaml"
)
PAIRS = 20
SEEDS = range(5)


def run_benchmark(name, grid, seed, grid_build_s):
    free = np.argwhere(~grid.occ)
    if len(free) < 2:
        raise RuntimeError(f"{name}: not enough free cells")

    rng = np.random.RandomState(seed)
    plan_times = []
    waypoint_counts = []
    successes = 0

    print(f"\n{name}, seed={seed}: testing {PAIRS} pairs...", flush=True)

    for i in range(PAIRS):
        a = free[rng.randint(len(free))]
        b = free[rng.randint(len(free))]

        while np.array_equal(a, b):
            b = free[rng.randint(len(free))]

        start = grid.to_world((int(a[0]), int(a[1])))
        goal = grid.to_world((int(b[0]), int(b[1])))

        t0 = time.perf_counter()
        path = plan(grid, start, goal, snap_dist=0.0)
        elapsed = time.perf_counter() - t0
        plan_times.append(elapsed)

        if path is not None:
            successes += 1

            t0 = time.perf_counter()
            simplified = simplify(grid, path)
            simplify_time = time.perf_counter() - t0

            waypoint_counts.append(len(simplified))
        else:
            simplify_time = 0.0

        print(
            f"  Pair {i + 1:02}/{PAIRS}: "
            f"{'success' if path is not None else 'no route'}; "
            f"plan={elapsed:.3f}s; simplify={simplify_time:.3f}s",
            flush=True,
        )

    return {
        "scenario": name,
        "seed": seed,
        "grid_s": grid_build_s,
        "median_s": statistics.median(plan_times),
        "max_s": max(plan_times),
        "success": successes,
        "avg_waypoints": (
            statistics.mean(waypoint_counts) if waypoint_counts else 0.0
        ),
    }


def make_grid(layout, bounds, res, inflate):
    t0 = time.perf_counter()
    grid = build_grid(
        layout,
        bounds=bounds,
        res=res,
        inflate=inflate,
    )
    return grid, time.perf_counter() - t0


def main():
    rows = []

    # Procedural cities: five seeds per difficulty.
    for level in ("easy", "medium", "hard"):
        for seed in SEEDS:
            city = generate_city(
                np.random.RandomState(seed),
                level=level,
            )

            width, height = city["size"]
            bounds = (
                (-width / 2, width / 2),
                (-height / 2, height / 2),
            )

            grid, build_s = make_grid(
                city["layout"],
                bounds,
                res=city["res"],
                inflate=DRONE_RADIUS + 0.35,
            )

            rows.append(
                run_benchmark(
                    f"city-{level}",
                    grid,
                    seed,
                    build_s,
                )
            )

    # OSM crop: exact crop bounds, 20 pairs.
    crop = load_osm_layout(CROP)

    grid, build_s = make_grid(
        crop,
        ((-25, 25), (-25, 25)),
        res=0.25,
        inflate=0.5,
    )

    rows.append(
        run_benchmark("osm-crop", grid, 0, build_s)
    )

    # Full OSM: use the exact layout bounds, at 0.5 m resolution.
    full = load_osm_layout(FULL)
    xmin, ymin, xmax, ymax = layout_bounds(full)
    bounds = ((xmin, xmax), (ymin, ymax))

    grid, build_s = make_grid(
        full,
        bounds,
        res=0.5,
        inflate=0.5,
    )

    rows.append(
        run_benchmark("osm-full", grid, 0, build_s)
    )

    print("\nD7 ROUTE PLANNER BENCHMARK RESULTS")
    print(
        f"{'Scenario':<13} {'Seed':>4} {'Grid(s)':>10} "
        f"{'Median(s)':>11} {'Max(s)':>10} {'Routes':>10} "
        f"{'Avg waypoints':>15}"
    )
    print("-" * 86)

    for r in rows:
        print(
            f"{r['scenario']:<13} {r['seed']:>4} "
            f"{r['grid_s']:>10.4f} {r['median_s']:>11.4f} "
            f"{r['max_s']:>10.4f} "
            f"{r['success']:>5}/{PAIRS:<4} "
            f"{r['avg_waypoints']:>15.2f}"
        )


if __name__ == "__main__":
    main()
