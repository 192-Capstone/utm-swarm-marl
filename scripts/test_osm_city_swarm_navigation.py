from pathlib import Path
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
    reconstruct_path, grid_coordinates, approximate_diameter_path,
)

CONFIG = ROOT / "configs/config_stage2_obstacles.yaml"
LAYOUT = ROOT / "data/osm_city/bengaluru_ecity_phase1_rotated_v2_crop50_cand1_layout.yaml"

GRID_STEP = 0.5
GRID_MIN = -22.0
GRID_MAX = 22.0
OBSTACLE_MARGIN = 1.25
MIN_ROUTE_LENGTH = 12.0
ALTITUDE = 2.0
MAX_SPEED = 0.7
WAYPOINT_RADIUS = 0.12


def main():
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    layout = yaml.safe_load(LAYOUT.read_text(encoding="utf-8"))

    if not layout:
        raise RuntimeError(f"Empty obstacle layout: {LAYOUT}")

    coords = grid_coordinates(GRID_MIN, GRID_MAX, GRID_STEP)
    n = len(coords)
    free = build_free_cells(layout, coords, OBSTACLE_MARGIN)

    def choose_long_path(allowed):
        candidates = []
        for component in connected_components(allowed, n):
            if len(component) < 50:
                continue
            path = approximate_diameter_path(component, n)
            length = (len(path) - 1) * GRID_STEP
            if length >= MIN_ROUTE_LENGTH:
                candidates.append((length, path))

        if not candidates:
            return None
        candidates.sort(key=lambda item: item[0], reverse=True)
        return candidates[0][1]

    def make_three_routes():
        # Try progressively less separation if the crop is too constrained.
        for separation_m in (3.0, 2.5, 2.0, 1.5):
            routes = []
            allowed = set(free)
            separation_cells = math.ceil(separation_m / GRID_STEP)
            radius_sq = separation_cells ** 2

            for _ in range(3):
                path = choose_long_path(allowed)
                if path is None:
                    break

                routes.append(path)
                path_cells = set(path)

                # Reserve a buffer around every path before planning the next.
                allowed = {
                    cell for cell in allowed
                    if cell not in path_cells
                    and all(
                        (cell[0] - p[0]) ** 2 + (cell[1] - p[1]) ** 2
                        > radius_sq
                        for p in path
                    )
                }

            if len(routes) == 3:
                return routes, separation_m

        return None, None

    routes, route_separation = make_three_routes()
    if routes is None:
        raise RuntimeError(
            "Could not find three sufficiently long, separated routes in this crop. "
            "Try another crop rather than weakening collision checks."
        )

    def xy(cell):
        return [float(coords[cell[0]]), float(coords[cell[1]])]

    starts = [xy(route[0]) for route in routes]
    targets = [xy(route[-1]) for route in routes]

    spawns = {
        f"drone_{i}": [*starts[i], ALTITUDE]
        for i in range(3)
    }
    goals = {
        f"drone_{i}": [*targets[i], ALTITUDE]
        for i in range(3)
    }

    print(f"Obstacle boxes: {len(layout)}")
    print(f"Grid resolution: {GRID_STEP:.1f} m")
    print(f"Planner obstacle margin: {OBSTACLE_MARGIN:.2f} m")
    print(f"Planned route separation: at least {route_separation:.1f} m")
    for i, route in enumerate(routes):
        print(
            f"Drone {i}: route length {(len(route)-1)*GRID_STEP:.1f} m | "
            f"start {spawns[f'drone_{i}']} | goal {goals[f'drone_{i}']}"
        )

    adapter = PyBulletAdapter(config, gui=False)
    settled_steps = [0, 0, 0]
    reached = [False, False, False]
    waypoint_index = [1, 1, 1]
    min_obstacle_gap = [float("inf")] * 3
    min_drone_distance = float("inf")
    collision = None

    max_steps = max(5000, max(len(route) for route in routes) * 100)
    steps_used = 0

    try:
        adapter.reset_world(spawns, goals, obstacles=layout)

        # Validate all initial positions before moving.
        initial_positions = adapter.get_positions()
        for i, position in enumerate(initial_positions):
            gap = float(scenario_manager.distance_to_obstacles(layout, position))
            if gap < scenario_manager.DRONE_RADIUS:
                raise RuntimeError(
                    f"Drone {i} starts in collision/too close to an obstacle: {gap:.3f} m"
                )

        for step_no in range(1, max_steps + 1):
            steps_used = step_no
            positions = adapter.get_positions()
            actions = {}

            # All three drones receive actions each simulation step.
            for i, route in enumerate(routes):
                if reached[i]:
                    actions[f"drone_{i}"] = [0.0, 0.0, 0.0, 0.0]
                    continue

                position = positions[i][:2]
                while waypoint_index[i] < len(route):
                    target_xy = np.asarray(xy(route[waypoint_index[i]]))
                    delta = target_xy - position
                    if float(np.linalg.norm(delta)) >= WAYPOINT_RADIUS:
                        break
                    waypoint_index[i] += 1

                if waypoint_index[i] >= len(route):
                    velocity = np.zeros(2)
                else:
                    target_xy = np.asarray(xy(route[waypoint_index[i]]))
                    delta = target_xy - position
                    d = float(np.linalg.norm(delta))
                    velocity = (
                        delta / d * min(MAX_SPEED, 1.5 * d)
                        if d > 1e-8 else np.zeros(2)
                    )

                actions[f"drone_{i}"] = [
                    float(velocity[0]), float(velocity[1]), 0.0, 0.0
                ]

            adapter.apply_actions(actions)
            adapter.step_simulation()

            positions = adapter.get_positions()
            raw = adapter.get_raw_state()

            # Check geometric obstacle clearance and drone-to-drone separation.
            for i, position in enumerate(positions):
                gap = float(scenario_manager.distance_to_obstacles(layout, position))
                min_obstacle_gap[i] = min(min_obstacle_gap[i], gap)

                if gap < scenario_manager.DRONE_RADIUS:
                    collision = f"obstacle collision by drone_{i}"
                    break

            if collision:
                break

            for i in range(3):
                for j in range(i + 1, 3):
                    separation = float(np.linalg.norm(positions[i] - positions[j]))
                    min_drone_distance = min(min_drone_distance, separation)
                    threshold = float(
                        config["environment"].get("collision_radius", 0.2)
                    )
                    if separation < threshold:
                        collision = f"drone-to-drone proximity violation: {i}/{j}"
                        break
                if collision:
                    break

            if collision:
                break

            goal_threshold = float(config["environment"]["goal_threshold"])
            settle_speed = float(config["environment"]["settle_speed"])
            required_settle = int(config["environment"]["settle_steps"])

            for i in range(3):
                if reached[i]:
                    continue

                distance_to_goal = float(np.linalg.norm(
                    positions[i] - np.asarray(goals[f"drone_{i}"])
                ))
                speed = float(np.linalg.norm(raw[f"drone_{i}"]["velocity"]))

                if distance_to_goal < goal_threshold and speed < settle_speed:
                    settled_steps[i] += 1
                else:
                    settled_steps[i] = 0

                if settled_steps[i] >= required_settle:
                    reached[i] = True

            if all(reached):
                break

        positions = adapter.get_positions()
        print(f"\nSimulation steps: {steps_used}")
        print(f"Minimum drone-to-drone distance: {min_drone_distance:.3f} m")
        print(f"Collision detected: {collision or 'none'}")

        for i in range(3):
            distance_to_goal = float(np.linalg.norm(
                positions[i] - np.asarray(goals[f"drone_{i}"])
            ))
            ok = reached[i] and collision is None
            print(
                f"Drone {i}: goal distance={distance_to_goal:.3f} m | "
                f"min obstacle gap={min_obstacle_gap[i]:.3f} m | "
                f"settled={reached[i]} | {'PASS' if ok else 'FAIL/INCOMPLETE'}"
            )

        success = all(reached) and collision is None
        print(
            "\nRESULT: "
            + ("PASS â€” all three drones reached and settled simultaneously."
               if success
               else "FAIL/INCOMPLETE â€” inspect the route and collision results.")
        )

    finally:
        adapter.close()
        print("PyBullet connection closed.")

    return 0 if all(reached) and collision is None else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
