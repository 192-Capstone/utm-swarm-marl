"""Load OSM-derived obstacle layouts and sample obstacle-free flights.

The scenario uses the map's own geometry bounds instead of assuming the legacy
50 m or 100 m random-world extent. It samples short vehicle-scale flights in a
much larger imported map; it does not guarantee global route connectivity or
replace the shared path-planning workstream.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import yaml

from envs.scenario_manager import surface_distance

DEFAULT_LAYOUT_PATH = Path("data/osm_city/bengaluru_ecity_phase1_rotated_v2_layout.yaml")


def validate_osm_layout(layout) -> list[dict]:
    """Validate the box-only OSM layout contract, including optional yaw."""
    if not isinstance(layout, list) or not layout:
        raise ValueError("OSM layout must be a non-empty YAML list")
    checked = []
    for idx, obstacle in enumerate(layout):
        if not isinstance(obstacle, dict) or obstacle.get("type") != "box":
            raise ValueError(f"OSM obstacle {idx} must be a box mapping")
        try:
            center = [float(v) for v in obstacle["center"]]
            half = [float(v) for v in obstacle["half"]]
            height = float(obstacle["height"])
            yaw = float(obstacle.get("yaw", 0.0))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"OSM obstacle {idx} has incomplete geometry fields") from exc
        if (len(center) != 2 or len(half) != 2
                or not all(math.isfinite(v) for v in (*center, *half, height, yaw))
                or min(half) <= 0 or height <= 0):
            raise ValueError(f"OSM obstacle {idx} has non-finite or non-positive geometry")
        checked_obstacle = dict(obstacle)
        checked_obstacle["center"] = center
        checked_obstacle["half"] = half
        checked_obstacle["height"] = height
        if abs(yaw) > 1e-12:
            checked_obstacle["yaw"] = yaw
        else:
            checked_obstacle.pop("yaw", None)
        checked.append(checked_obstacle)
    return checked


def load_osm_layout(path: str | Path) -> list[dict]:
    """Read and validate a generated YAML obstacle layout."""
    layout_path = Path(path)
    if not layout_path.is_file():
        raise FileNotFoundError(f"OSM scenario layout not found: {layout_path}")
    return validate_osm_layout(yaml.safe_load(layout_path.read_text(encoding="utf-8")))


def layout_bounds(layout: Sequence[dict]) -> tuple[float, float, float, float]:
    """Return xmin, ymin, xmax, ymax of rotated box footprints."""
    if not layout:
        raise ValueError("Cannot compute bounds of an empty OSM layout")
    lows_x, lows_y, highs_x, highs_y = [], [], [], []
    for obstacle in layout:
        if obstacle.get("type") != "box":
            raise ValueError("OSM city layouts currently support box obstacles only")
        cx, cy = obstacle["center"]
        hx, hy = obstacle["half"]
        yaw = float(obstacle.get("yaw", 0.0))
        c, s = abs(math.cos(yaw)), abs(math.sin(yaw))
        extent_x = hx * c + hy * s
        extent_y = hx * s + hy * c
        lows_x.append(cx - extent_x)
        highs_x.append(cx + extent_x)
        lows_y.append(cy - extent_y)
        highs_y.append(cy + extent_y)
    return min(lows_x), min(lows_y), max(highs_x), max(highs_y)


def _is_free(layout: Sequence[dict], x: float, y: float, clearance_m: float) -> bool:
    return all(surface_distance(obstacle, (x, y)) >= clearance_m for obstacle in layout)


def sample_flights(
    layout: Sequence[dict],
    n_agents: int,
    rng=None,
    *,
    min_goal_distance_m: float = 15.0,
    max_goal_distance_m: float = 45.0,
    obstacle_clearance_m: float = 2.0,
    min_point_separation_m: float = 5.0,
    boundary_margin_m: float = 2.0,
    min_altitude_m: float = 2.0,
    max_altitude_m: float = 4.0,
    max_sampling_attempts: int = 1000,
) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
    """Sample valid start/goal pairs from free space in the complete map extent.

    A bounded pool of free points is sampled once, then paired into flights.
    This avoids nested rejection loops growing exponentially when the map is
    dense. A valid pair is clear at its endpoints; route connectivity is left
    to the common planner/teacher and is not guaranteed by endpoint sampling.
    """
    import numpy as np

    rng = np.random if rng is None else rng
    if n_agents < 1:
        raise ValueError("n_agents must be at least one")
    finite_values = (min_goal_distance_m, max_goal_distance_m, obstacle_clearance_m,
                     min_point_separation_m, boundary_margin_m, min_altitude_m,
                     max_altitude_m)
    if not all(math.isfinite(float(v)) for v in finite_values):
        raise ValueError("OSM scenario sampling parameters must be finite")
    if min_goal_distance_m < 0 or max_goal_distance_m < min_goal_distance_m:
        raise ValueError("Goal-distance bounds are invalid")
    if obstacle_clearance_m < 0 or min_point_separation_m < 0 or boundary_margin_m < 0:
        raise ValueError("Clearances/margins must be non-negative")
    if min_altitude_m <= 0 or max_altitude_m < min_altitude_m:
        raise ValueError("Altitude bounds are invalid")
    if max_sampling_attempts < 1:
        raise ValueError("max_sampling_attempts must be positive")

    xmin, ymin, xmax, ymax = layout_bounds(layout)
    xmin += boundary_margin_m
    ymin += boundary_margin_m
    xmax -= boundary_margin_m
    ymax -= boundary_margin_m
    if xmin >= xmax or ymin >= ymax:
        raise ValueError("OSM bounds are too small for the configured boundary margin")

    # Collect candidate points with a single bounded rejection loop. The target
    # pool allows multiple possible start/goal pairings without repeatedly
    # scanning every obstacle for deeply nested attempts.
    target_candidates = min(max_sampling_attempts, max(200, n_agents * 80))
    candidates: list[tuple[float, float]] = []
    for _ in range(max_sampling_attempts):
        x = float(rng.uniform(xmin, xmax))
        y = float(rng.uniform(ymin, ymax))
        if _is_free(layout, x, y, obstacle_clearance_m):
            candidates.append((x, y))
            if len(candidates) >= target_candidates:
                break

    if len(candidates) < 2:
        raise RuntimeError(
            f"Only {len(candidates)} free candidate points were found in the OSM map; "
            "reduce obstacle_clearance_m or increase max_sampling_attempts."
        )

    def separated(point, used):
        return all(math.hypot(point[0] - other[0], point[1] - other[1]) >= min_point_separation_m
                   for other in used)

    # Randomise pair order while retaining deterministic behaviour for a seeded RNG.
    order = list(map(int, rng.permutation(len(candidates))))
    spawns: dict[str, list[float]] = {}
    goals: dict[str, list[float]] = {}
    used_points: list[tuple[float, float]] = []
    for i in range(n_agents):
        accepted = None
        for start_idx in order:
            start_xy = candidates[start_idx]
            if not separated(start_xy, used_points):
                continue
            goal_order = list(map(int, rng.permutation(len(candidates))))
            for goal_idx in goal_order:
                if goal_idx == start_idx:
                    continue
                goal_xy = candidates[goal_idx]
                if not separated(goal_xy, [*used_points, start_xy]):
                    continue
                distance = math.hypot(goal_xy[0] - start_xy[0], goal_xy[1] - start_xy[1])
                if min_goal_distance_m <= distance <= max_goal_distance_m:
                    accepted = start_xy, goal_xy
                    break
            if accepted is not None:
                break
        if accepted is None:
            raise RuntimeError(
                f"Could not sample a free start/goal pair for drone {i} from "
                f"{len(candidates)} candidates with goal distance "
                f"[{min_goal_distance_m:g}, {max_goal_distance_m:g}] m. "
                "Increase max_sampling_attempts or relax the separation/clearance constraints."
            )
        start_xy, goal_xy = accepted
        z = float(rng.uniform(min_altitude_m, max_altitude_m))
        key = f"drone_{i}"
        spawns[key] = [start_xy[0], start_xy[1], z]
        goals[key] = [goal_xy[0], goal_xy[1], z]
        used_points.extend((start_xy, goal_xy))
    return spawns, goals
