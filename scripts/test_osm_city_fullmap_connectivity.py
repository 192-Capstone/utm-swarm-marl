"""Coarse-grid route-connectivity diagnostic for the full OSM city map."""
from pathlib import Path
import math
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import yaml

from envs.osm_scenario import load_osm_layout, layout_bounds, sample_flights
from modules.grid_path_planner import (
    grid_coordinates,
    build_free_cells,
    connected_components,
    breadth_first_search,
    reconstruct_path,
)

CONFIG_PATH = Path("configs/config_stage3_osm_ecity.yaml")
STEP_M = 2.0
OBSTACLE_MARGIN_M = 1.25
BOUNDARY_MARGIN_M = 2.0
MAX_SNAP_DISTANCE_M = 3.0
SEEDS = (0, 1, 2)


def snap_to_free_cell(point, coordinates, free_cells, grid_min):
    """Snap an endpoint to a nearby free grid cell, if one is available."""
    x, y = float(point[0]), float(point[1])
    i0 = round((x - grid_min) / STEP_M)
    j0 = round((y - grid_min) / STEP_M)

    choices = []
    for i in range(max(0, i0 - 2), min(len(coordinates), i0 + 3)):
        for j in range(max(0, j0 - 2), min(len(coordinates), j0 + 3)):
            cell = (i, j)
            if cell not in free_cells:
                continue
            gx, gy = coordinates[i], coordinates[j]
            distance = math.hypot(gx - x, gy - y)
            if distance <= MAX_SNAP_DISTANCE_M:
                choices.append((distance, cell))

    if not choices:
        return None
    return min(choices, key=lambda item: item[0])


def main():
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    layout_path = Path(config["osm_scenario"]["layout_path"])
    layout = load_osm_layout(layout_path)
    xmin, ymin, xmax, ymax = layout_bounds(layout)

    grid_min = math.floor(min(xmin, ymin) / STEP_M) * STEP_M
    grid_max = math.ceil(max(xmax, ymax) / STEP_M) * STEP_M
    coordinates = grid_coordinates(grid_min, grid_max, STEP_M)

    print("Full-map layout:", layout_path)
    print("Obstacle boxes:", len(layout))
    print("Map bounds:", (xmin, ymin, xmax, ymax))
    print("Grid resolution:", STEP_M, "m")
    print("Obstacle margin:", OBSTACLE_MARGIN_M, "m")
    print("Building free grid cells... this may take a little while.")

    free_cells = build_free_cells(
        layout, coordinates, obstacle_margin=OBSTACLE_MARGIN_M
    )

    # Restrict route cells to the map's bounds, with a boundary margin.
    free_cells = {
        (i, j)
        for i, j in free_cells
        if xmin + BOUNDARY_MARGIN_M <= coordinates[i] <= xmax - BOUNDARY_MARGIN_M
        and ymin + BOUNDARY_MARGIN_M <= coordinates[j] <= ymax - BOUNDARY_MARGIN_M
    }

    if not free_cells:
        raise RuntimeError("No free grid cells found inside the map bounds.")

    components = connected_components(free_cells, len(coordinates))
    components.sort(key=len, reverse=True)

    print("Free grid cells:", len(free_cells))
    print("Connected free-space components:", len(components))
    print("Largest component cells:", len(components[0]))
    print("Largest component approximate area:",
          round(len(components[0]) * STEP_M * STEP_M, 1), "m^2")

    rng_seeds = SEEDS
    tested = 0
    reachable = 0
    unsnappable = 0
    disconnected = 0

    for seed in rng_seeds:
        rng = np.random.RandomState(seed)
        spawns, goals = sample_flights(
            layout,
            n_agents=3,
            rng=rng,
            min_goal_distance_m=15.0,
            max_goal_distance_m=45.0,
            obstacle_clearance_m=2.0,
            min_point_separation_m=5.0,
            boundary_margin_m=BOUNDARY_MARGIN_M,
            max_sampling_attempts=1000,
        )

        for drone_id in sorted(spawns):
            start = spawns[drone_id]
            goal = goals[drone_id]
            start_snap = snap_to_free_cell(
                start, coordinates, free_cells, grid_min
            )
            goal_snap = snap_to_free_cell(
                goal, coordinates, free_cells, grid_min
            )

            tested += 1
            print(
                f"\nSeed {seed} {drone_id}: "
                f"sampled straight-line distance="
                f"{math.dist(start[:2], goal[:2]):.1f} m"
            )

            if start_snap is None or goal_snap is None:
                unsnappable += 1
                print("RESULT: NO NEARBY FREE GRID CELL for start or goal")
                continue

            start_offset, start_cell = start_snap
            goal_offset, goal_cell = goal_snap
            print(f"Start snap offset: {start_offset:.2f} m")
            print(f"Goal snap offset:  {goal_offset:.2f} m")

            distances, parents = breadth_first_search(
                start_cell, free_cells, len(coordinates)
            )

            if goal_cell not in distances:
                disconnected += 1
                print("RESULT: DISCONNECTED at this grid resolution")
                continue

            path = reconstruct_path(goal_cell, parents)
            route_length = (len(path) - 1) * STEP_M
            reachable += 1
            print(f"Grid route length: {route_length:.1f} m")
            print(f"Grid route cells: {len(path)}")
            print("RESULT: CONNECTED")

    print("\n=== CONNECTIVITY SUMMARY ===")
    print("Pairs tested:", tested)
    print("Connected:", reachable)
    print("Disconnected on this grid:", disconnected)
    print("Could not snap endpoints:", unsnappable)
    print("Grid resolution:", STEP_M, "m")
    print(
        "Interpretation: coarse geometric connectivity diagnostic only; "
        "not a PyBullet flight or trained-policy evaluation."
    )


if __name__ == "__main__":
    main()
