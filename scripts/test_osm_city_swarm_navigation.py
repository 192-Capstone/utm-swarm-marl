from pathlib import Path
import math
import random
import sys

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from envs.pybullet_adapter import PyBulletAdapter
from envs import scenario_manager
from envs.route_planner import build_grid, plan, simplify, path_length
from modules.grid_path_planner import connected_components

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

    # Build the shared D7 grid using the existing OSM crop bounds and margin.
    grid = build_grid(
        layout,
        ((GRID_MIN, GRID_MAX), (GRID_MIN, GRID_MAX)),
        GRID_STEP,
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

    print(f"Obstacle boxes: {len(layout)}")
    print(f"Free grid cells: {len(free)}")
    print(f"Grid resolution: {GRID_STEP:.2f} m")
    print(f"Planner obstacle margin: {OBSTACLE_MARGIN:.2f} m")

    rng = random.Random(2026)

    def xy(cell):
        """Convert a grid cell (iy, ix) to its world-coordinate centre."""
        return [float(v) for v in grid.to_world(cell)]

    def cell_distance(a, b):
        """Euclidean distance between grid cells, measured in metres."""
        return math.hypot(a[0] - b[0], a[1] - b[1]) * GRID_STEP

    def route_cells(points):
        """Rasterize route segments into cells for the separation buffer."""
        cells = set()

        if len(points) == 1:
            cells.add(grid.to_cell(points[0]))
            return cells

        for a, b in zip(points, points[1:]):
            distance = math.hypot(b[0] - a[0], b[1] - a[1])
            samples = max(1, math.ceil(distance / (GRID_STEP / 2)))

            for k in range(samples + 1):
                t = k / samples
                point = (
                    a[0] + t * (b[0] - a[0]),
                    a[1] + t * (b[1] - a[1]),
                )
                cells.add(grid.to_cell(point))

        return cells

    def choose_long_path(allowed):
        """Find a sufficiently long D7 route within remaining free space."""
        if len(allowed) < 50:
            return None

        # Connected components avoid trying endpoints in disconnected areas.
        grid_width = grid.occ.shape[1]
        components = connected_components(allowed, grid_width)
        components.sort(key=len, reverse=True)

        original_occ = grid.occ.copy()
        allowed_mask = np.zeros_like(grid.occ, dtype=bool)

        for cell in allowed:
            allowed_mask[cell] = True

        try:
            # Block cells outside the remaining available route space.
            grid.occ[:] = original_occ | ~allowed_mask

            for component in components:
                if len(component) < 50:
                    continue

                component_cells = sorted(component)
                starts_to_try = rng.sample(
                    component_cells,
                    min(64, len(component_cells)),
                )

                for start in starts_to_try:
                    goals = [
                        cell for cell in component_cells
                        if cell_distance(start, cell) >= MIN_ROUTE_LENGTH
                    ]

                    if not goals:
                        continue

                    goals.sort(
                        key=lambda cell: cell_distance(start, cell),
                        reverse=True,
                    )

                    # Search a wider selection of distant goals.
                    goal_pool = goals[:min(32, len(goals))]
                    rng.shuffle(goal_pool)

                    for goal in goal_pool[:16]:
                        route = plan(
                            grid,
                            xy(start),
                            xy(goal),
                            snap_dist=0.0,
                        )

                        if route is None:
                            continue

                        waypoints = simplify(grid, route)

                        if path_length(waypoints) >= MIN_ROUTE_LENGTH:
                            return waypoints

            return None

        finally:
            # Always restore the original occupancy grid.
            grid.occ[:] = original_occ

    def make_three_routes():
        # Preserve the original fallback separation thresholds.
        for separation_m in (3.0, 2.5, 2.0, 1.5):
            routes = []
            allowed = set(free)

            separation_cells = math.ceil(separation_m / GRID_STEP)
            radius_sq = separation_cells ** 2

            for _ in range(3):
                waypoints = choose_long_path(allowed)

                if waypoints is None:
                    break

                routes.append(waypoints)
                occupied_cells = route_cells(waypoints)

                # Reserve a buffer around the entire route before planning
                # the next route.
                allowed = {
                    cell for cell in allowed
                    if cell not in occupied_cells
                    and all(
                        (cell[0] - p[0]) ** 2
                        + (cell[1] - p[1]) ** 2
                        > radius_sq
                        for p in occupied_cells
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

    # D7 routes contain world-coordinate waypoints, not grid-cell indices.
    starts = [
        [float(v) for v in route[0]]
        for route in routes
    ]
    targets = [
        [float(v) for v in route[-1]]
        for route in routes
    ]

    spawns = {
        f"drone_{i}": [*starts[i], ALTITUDE]
        for i in range(3)
    }
    goals = {
        f"drone_{i}": [*targets[i], ALTITUDE]
        for i in range(3)
    }

    print(f"Planned route separation: at least {route_separation:.1f} m")

    for i, route in enumerate(routes):
        print(
            f"Drone {i}: route length {path_length(route):.1f} m | "
            f"{len(route)} waypoints | "
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
            gap = float(
                scenario_manager.distance_to_obstacles(layout, position)
            )

            if gap < scenario_manager.DRONE_RADIUS:
                raise RuntimeError(
                    f"Drone {i} starts in collision/too close to an obstacle: "
                    f"{gap:.3f} m"
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

                # Advance through waypoints that have already been reached.
                while waypoint_index[i] < len(route):
                    target_xy = np.asarray(
                        route[waypoint_index[i]],
                        dtype=float,
                    )
                    delta = target_xy - position

                    if float(np.linalg.norm(delta)) >= WAYPOINT_RADIUS:
                        break

                    waypoint_index[i] += 1

                if waypoint_index[i] >= len(route):
                    velocity = np.zeros(2)
                else:
                    target_xy = np.asarray(
                        route[waypoint_index[i]],
                        dtype=float,
                    )
                    delta = target_xy - position
                    d = float(np.linalg.norm(delta))

                    velocity = (
                        delta / d * min(MAX_SPEED, 1.5 * d)
                        if d > 1e-8
                        else np.zeros(2)
                    )

                actions[f"drone_{i}"] = [
                    float(velocity[0]),
                    float(velocity[1]),
                    0.0,
                    0.0,
                ]

            adapter.apply_actions(actions)
            adapter.step_simulation()

            positions = adapter.get_positions()
            raw = adapter.get_raw_state()

            # Check geometric obstacle clearance for every drone.
            for i, position in enumerate(positions):
                gap = float(
                    scenario_manager.distance_to_obstacles(layout, position)
                )
                min_obstacle_gap[i] = min(min_obstacle_gap[i], gap)

                if gap < scenario_manager.DRONE_RADIUS:
                    collision = f"obstacle collision by drone_{i}"
                    break

            if collision:
                break

            # Check drone-to-drone separation.
            for i in range(3):
                for j in range(i + 1, 3):
                    separation = float(
                        np.linalg.norm(positions[i] - positions[j])
                    )
                    min_drone_distance = min(
                        min_drone_distance,
                        separation,
                    )
                    threshold = float(
                        config["environment"].get("collision_radius", 0.2)
                    )

                    if separation < threshold:
                        collision = (
                            f"drone-to-drone proximity violation: {i}/{j}"
                        )
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

                distance_to_goal = float(
                    np.linalg.norm(
                        positions[i] - np.asarray(goals[f"drone_{i}"])
                    )
                )
                speed = float(
                    np.linalg.norm(raw[f"drone_{i}"]["velocity"])
                )

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
        print(
            f"Minimum drone-to-drone distance: "
            f"{min_drone_distance:.3f} m"
        )
        print(f"Collision detected: {collision or 'none'}")

        for i in range(3):
            distance_to_goal = float(
                np.linalg.norm(
                    positions[i] - np.asarray(goals[f"drone_{i}"])
                )
            )
            ok = reached[i] and collision is None

            print(
                f"Drone {i}: goal distance={distance_to_goal:.3f} m | "
                f"min obstacle gap={min_obstacle_gap[i]:.3f} m | "
                f"settled={reached[i]} | "
                f"{'PASS' if ok else 'FAIL/INCOMPLETE'}"
            )

        success = all(reached) and collision is None
        print("\nRESULT: " + ("PASS" if success else "FAIL/INCOMPLETE"))

        return 0 if success else 1

    finally:
        adapter.close()
        print("\nPyBullet connection closed.")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)