from pathlib import Path
import random
import sys

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from envs.pybullet_adapter import PyBulletAdapter
from envs import scenario_manager
from envs.route_planner import build_grid, plan, simplify, path_length
from modules.grid_path_planner import connected_components, grid_distance

CONFIG = ROOT / "configs/config_stage2_obstacles.yaml"
LAYOUT = ROOT / "data/osm_city/bengaluru_ecity_phase1_rotated_v2_crop50_cand1_layout.yaml"

STEP = 0.5
GRID_MIN, GRID_MAX = -22.0, 22.0
OBSTACLE_MARGIN = 1.25
Z = 2.0
SPEED = 0.7


def main():
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    layout = yaml.safe_load(LAYOUT.read_text(encoding="utf-8"))

    if not isinstance(layout, list) or not layout:
        raise RuntimeError("The obstacle layout is empty or invalid.")

    # Build the shared D7 route-planning grid.
    grid = build_grid(
        layout,
        ((GRID_MIN, GRID_MAX), (GRID_MIN, GRID_MAX)),
        STEP,
        OBSTACLE_MARGIN,
    )

    free = {
        (iy, ix)
        for iy in range(grid.occ.shape[0])
        for ix in range(grid.occ.shape[1])
        if grid.free((iy, ix))
    }

    if not free:
        raise RuntimeError("The route planner found no free grid cells.")

    # Preserve the original connected-region restriction.
    regions = connected_components(free, grid.occ.shape[0])

    if not regions:
        raise RuntimeError("No connected free-space region found.")

    largest = max(regions, key=len)

    print(f"Obstacle boxes: {len(layout)}")
    print(f"Connected free-space regions: {len(regions)}")
    print(f"Largest region: {len(largest)} cells")
    print(f"Grid resolution: {STEP:.2f} m")
    print(f"Safety margin used by planner: {OBSTACLE_MARGIN:.2f} m")

    def xy(cell):
        """Convert a grid cell (iy, ix) into world coordinates (x, y)."""
        return [float(v) for v in grid.to_world(cell)]

    rng = random.Random(2026)
    candidates = sorted(largest)
    endpoints = []
    routes = []

    # Plan three routes with separated start and goal endpoints.
    for route_no in range(3):
        starts = [
            cell for cell in candidates
            if all(
                grid_distance(cell, p, STEP) >= 6.0
                for p in endpoints
            )
        ]

        if not starts:
            starts = candidates

        # Keep the longest route found, so the drone crosses the map rather
        # than taking the first route that is just long enough.
        picked = None
        best_length = 15.0

        for start in rng.sample(starts, min(30, len(starts))):
            targets = [
                cell for cell in candidates
                if grid_distance(start, cell, STEP) >= 15.0
                and all(
                    grid_distance(cell, p, STEP) >= 6.0
                    for p in endpoints
                )
            ]

            if not targets:
                targets = [
                    cell for cell in candidates
                    if grid_distance(start, cell, STEP) >= 15.0
                ]

            if not targets:
                continue

            targets.sort(
                key=lambda cell: grid_distance(start, cell, STEP),
                reverse=True,
            )

            # The most distant goals first.
            for goal in targets[:min(10, len(targets))]:
                route = plan(grid, xy(start), xy(goal))

                if route is not None and path_length(route) > best_length:
                    picked = (start, goal, route)
                    best_length = path_length(route)

        if picked is None:
            raise RuntimeError(
                f"Could not plan route {route_no + 1}. Try another crop "
                "or reduce the minimum route length."
            )

        start, goal, route = picked
        routes.append(route)
        endpoints.extend([start, goal])

    adapter = PyBulletAdapter(config, gui=False)
    passed = 0

    try:
        for route_no, path in enumerate(routes, start=1):
            # Retain the complete route for parking-clearance checks.
            # Simplified waypoints are used for scripted flight control.
            path_cells = set()

            for point in path:
                cell = grid.to_cell(point)
                if cell is not None:
                    path_cells.add(cell)

            waypoints = simplify(grid, path)

            # Find two parking positions with adequate route clearance.
            parking = []

            for min_gap in (4.0, 3.0, 2.5):
                pool = []

                for cell in free - path_cells:
                    gap = min(
                        grid_distance(cell, p, STEP)
                        for p in path_cells
                    )

                    if gap >= min_gap:
                        pool.append((gap, cell))

                pool.sort(reverse=True)
                parking = []

                for _, cell in pool:
                    if all(
                        grid_distance(cell, p, STEP) >= 2.5
                        for p in parking
                    ):
                        parking.append(cell)

                    if len(parking) == 2:
                        break

                if len(parking) == 2:
                    break

            if len(parking) != 2:
                raise RuntimeError("Could not find safe parking positions.")

            start_xy = list(path[0])
            goal_xy = list(path[-1])

            spawns = {
                "drone_0": [*start_xy, Z],
                "drone_1": [*xy(parking[0]), Z],
                "drone_2": [*xy(parking[1]), Z],
            }

            goals = {
                "drone_0": [*goal_xy, Z],
                "drone_1": spawns["drone_1"].copy(),
                "drone_2": spawns["drone_2"].copy(),
            }

            adapter.reset_world(spawns, goals, obstacles=layout)

            index = 1
            settled = 0
            collision = None
            min_gap = float("inf")
            reached = False
            max_steps = max(4000, len(path) * 100)

            print(
                f"\nRoute {route_no}: {path_length(path):.1f} m | "
                f"{len(waypoints)} simplified waypoints"
            )
            print(f"Start: {spawns['drone_0']}")
            print(f"Goal:  {goals['drone_0']}")

            for step_no in range(1, max_steps + 1):
                positions = adapter.get_positions()
                position = positions[0][:2]

                while index < len(waypoints):
                    target = np.array(waypoints[index], dtype=float)
                    delta = target - position

                    if np.linalg.norm(delta) >= 0.12:
                        break

                    index += 1

                if index >= len(waypoints):
                    velocity = np.zeros(2)
                else:
                    delta = (
                        np.array(waypoints[index], dtype=float) - position
                    )
                    d = float(np.linalg.norm(delta))

                    velocity = (
                        delta / d * min(SPEED, 1.5 * d)
                        if d > 1e-8
                        else np.zeros(2)
                    )

                actions = {
                    "drone_0": [
                        float(velocity[0]),
                        float(velocity[1]),
                        0.0,
                        0.0,
                    ],
                    "drone_1": [0.0, 0.0, 0.0, 0.0],
                    "drone_2": [0.0, 0.0, 0.0, 0.0],
                }

                adapter.apply_actions(actions)
                adapter.step_simulation()

                positions = adapter.get_positions()
                raw = adapter.get_raw_state()

                # Check geometric obstacle clearance for every drone.
                for i, pos in enumerate(positions):
                    gap = float(
                        scenario_manager.distance_to_obstacles(layout, pos)
                    )
                    min_gap = min(min_gap, gap)

                    if gap < scenario_manager.DRONE_RADIUS:
                        collision = f"obstacle, drone_{i}"
                        break

                if collision:
                    break

                # Check drone-to-drone separation.
                for i in range(len(positions)):
                    for j in range(i + 1, len(positions)):
                        if np.linalg.norm(
                            positions[i] - positions[j]
                        ) < float(
                            config["environment"].get("collision_radius", 0.2)
                        ):
                            collision = f"drone-drone, {i}/{j}"
                            break

                    if collision:
                        break

                if collision:
                    break

                dist_goal = float(
                    np.linalg.norm(
                        positions[0] - np.asarray(goals["drone_0"])
                    )
                )
                speed = float(
                    np.linalg.norm(raw["drone_0"]["velocity"])
                )

                if dist_goal < 0.5 and speed < 0.15:
                    settled += 1
                else:
                    settled = 0

                if settled >= 31:
                    reached = True
                    break

            final_positions = adapter.get_positions()
            final_distance = float(
                np.linalg.norm(
                    final_positions[0] - np.asarray(goals["drone_0"])
                )
            )

            ok = reached and collision is None
            passed += int(ok)

            print(f"Steps: {step_no}")
            print(f"Final goal distance: {final_distance:.3f} m")
            print(f"Minimum obstacle gap: {min_gap:.3f} m")
            print(f"Collision: {collision or 'none detected'}")
            print(f"Settled at goal: {reached}")
            print(f"RESULT: {'PASS' if ok else 'FAIL/INCOMPLETE'}")

            if not ok:
                print("Stopping after the first unsuccessful route.")
                break

    finally:
        adapter.close()
        print("\nPyBullet connection closed.")

    print(f"SUMMARY: {passed}/{len(routes)} routes passed.")
    return 0 if passed == len(routes) else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(
            f"ERROR: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        raise SystemExit(1)
