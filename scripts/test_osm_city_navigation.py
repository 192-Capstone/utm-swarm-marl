from pathlib import Path
import random
import math
import sys

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from envs.pybullet_adapter import PyBulletAdapter
from envs import scenario_manager
from modules.grid_path_planner import (
    build_free_cells, connected_components, breadth_first_search,
    reconstruct_path, grid_distance, grid_coordinates,
)

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

    coords = grid_coordinates(GRID_MIN, GRID_MAX, STEP)
    n = len(coords)
    free = build_free_cells(layout, coords, OBSTACLE_MARGIN)

    regions = connected_components(free, n)
    largest = max(regions, key=len)
    print(f"Obstacle boxes: {len(layout)}")
    print(f"Connected free-space regions: {len(regions)}")
    print(f"Largest region: {len(largest)} cells")
    print(f"Safety margin used by planner: {OBSTACLE_MARGIN:.2f} m")

    rng = random.Random(2026)
    candidates = sorted(largest)
    endpoints = []
    routes = []

    for route_no in range(3):
        starts = [
            c for c in candidates
            if all(grid_distance(c, p, STEP) >= 6.0 for p in endpoints)
        ]
        if not starts:
            starts = candidates

        picked = None
        for start in rng.sample(starts, min(80, len(starts))):
            distances, parents = breadth_first_search(start, largest, n)
            targets = [
                cell for cell, d in distances.items()
                if d * STEP >= 15.0
                and all(grid_distance(cell, p, STEP) >= 6.0 for p in endpoints)
            ]
            if not targets:
                targets = [
                    cell for cell, d in distances.items()
                    if d * STEP >= 15.0
                ]
            if targets:
                targets.sort(key=lambda cell: distances[cell], reverse=True)
                goal = rng.choice(targets[:min(20, len(targets))])
                picked = (start, goal, parents)
                break

        if picked is None:
            raise RuntimeError(
                f"Could not plan route {route_no + 1}. Try another crop "
                "or reduce the minimum route length."
            )

        start, goal, parents = picked
        path = reconstruct_path(goal, parents)
        routes.append(path)
        endpoints.extend([start, goal])

    def xy(cell):
        return [coords[cell[0]], coords[cell[1]]]

    adapter = PyBulletAdapter(config, gui=False)
    passed = 0

    try:
        for route_no, path in enumerate(routes, start=1):
            path_cells = set(path)

            parking = []
            for min_gap in (4.0, 3.0, 2.5):
                pool = []
                for cell in free - path_cells:
                    gap = min(grid_distance(cell, p, STEP) for p in path)
                    if gap >= min_gap:
                        pool.append((gap, cell))
                pool.sort(reverse=True)

                parking = []
                for _, cell in pool:
                    if all(grid_distance(cell, p, STEP) >= 2.5 for p in parking):
                        parking.append(cell)
                    if len(parking) == 2:
                        break
                if len(parking) == 2:
                    break

            if len(parking) != 2:
                raise RuntimeError("Could not find safe parking positions.")

            start_xy, goal_xy = xy(path[0]), xy(path[-1])
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
                f"\nRoute {route_no}: {len(path) - 1} grid segments, "
                f"{(len(path)-1)*STEP:.1f} m"
            )
            print(f"Start: {spawns['drone_0']}")
            print(f"Goal:  {goals['drone_0']}")

            for step_no in range(1, max_steps + 1):
                positions = adapter.get_positions()
                position = positions[0][:2]

                while index < len(path):
                    target = np.array(xy(path[index]), dtype=float)
                    delta = target - position
                    if np.linalg.norm(delta) >= 0.12:
                        break
                    index += 1

                if index >= len(path):
                    velocity = np.zeros(2)
                else:
                    delta = np.array(xy(path[index]), dtype=float) - position
                    d = float(np.linalg.norm(delta))
                    velocity = delta / d * min(SPEED, 1.5 * d) if d > 1e-8 else np.zeros(2)

                actions = {
                    "drone_0": [float(velocity[0]), float(velocity[1]), 0.0, 0.0],
                    "drone_1": [0.0, 0.0, 0.0, 0.0],
                    "drone_2": [0.0, 0.0, 0.0, 0.0],
                }
                adapter.apply_actions(actions)
                adapter.step_simulation()

                positions = adapter.get_positions()
                raw = adapter.get_raw_state()

                for i, pos in enumerate(positions):
                    gap = float(scenario_manager.distance_to_obstacles(layout, pos))
                    min_gap = min(min_gap, gap)
                    if gap < scenario_manager.DRONE_RADIUS:
                        collision = f"obstacle, drone_{i}"
                        break

                if collision:
                    break

                for i in range(len(positions)):
                    for j in range(i + 1, len(positions)):
                        if np.linalg.norm(positions[i] - positions[j]) < float(
                            config["environment"].get("collision_radius", 0.2)
                        ):
                            collision = f"drone-drone, {i}/{j}"
                            break
                    if collision:
                        break
                if collision:
                    break

                dist_goal = float(
                    np.linalg.norm(positions[0] - np.asarray(goals["drone_0"]))
                )
                speed = float(np.linalg.norm(raw["drone_0"]["velocity"]))

                if dist_goal < 0.5 and speed < 0.15:
                    settled += 1
                else:
                    settled = 0

                if settled >= 31:
                    reached = True
                    break

            final_positions = adapter.get_positions()
            final_distance = float(
                np.linalg.norm(final_positions[0] - np.asarray(goals["drone_0"]))
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
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
