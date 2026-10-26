"""Small reusable 4-connected grid planner for map/layout smoke tests.

This module centralises D3's occupancy-grid, connected-component and BFS logic
so it can later be reused by the D1 procedural-city generator. It is a baseline
geometric planner, not the learned MARL policy or a production city-scale router.
"""
from __future__ import annotations

from collections import deque
import math
from typing import Iterable, Sequence

from envs.scenario_manager import surface_distance

Cell = tuple[int, int]


def grid_coordinates(grid_min: float, grid_max: float, step: float) -> list[float]:
    """Return regularly spaced coordinates including both bounds when aligned."""
    if not all(math.isfinite(float(v)) for v in (grid_min, grid_max, step)):
        raise ValueError("Grid bounds and step must be finite")
    if step <= 0 or grid_max < grid_min:
        raise ValueError("Grid step must be positive and grid_max >= grid_min")
    n_steps = round((grid_max - grid_min) / step)
    if not math.isclose(grid_min + n_steps * step, grid_max, abs_tol=1e-8):
        raise ValueError("Grid range must be an integer multiple of step")
    return [grid_min + i * step for i in range(n_steps + 1)]


def build_free_cells(layout: Sequence[dict], coordinates: Sequence[float],
                     obstacle_margin: float) -> set[Cell]:
    """Build free grid cells using true obstacle surface distances, including yaw."""
    if not math.isfinite(float(obstacle_margin)) or obstacle_margin < 0:
        raise ValueError("obstacle_margin must be finite and non-negative")
    free: set[Cell] = set()
    for j, y in enumerate(coordinates):
        for i, x in enumerate(coordinates):
            if all(surface_distance(obstacle, (x, y)) >= obstacle_margin
                   for obstacle in layout):
                free.add((i, j))
    return free


def grid_neighbors(cell: Cell, allowed: set[Cell], grid_size: int) -> Iterable[Cell]:
    """Yield four-connected neighbours that exist in the allowed cell set."""
    i, j = cell
    for nxt in ((i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)):
        if 0 <= nxt[0] < grid_size and 0 <= nxt[1] < grid_size and nxt in allowed:
            yield nxt


def connected_components(cells: set[Cell], grid_size: int) -> list[set[Cell]]:
    """Return connected components in deterministic seed order."""
    remaining = set(cells)
    result: list[set[Cell]] = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        component = {start}
        queue = deque([start])
        while queue:
            current = queue.popleft()
            for nxt in grid_neighbors(current, cells, grid_size):
                if nxt in remaining:
                    remaining.remove(nxt)
                    component.add(nxt)
                    queue.append(nxt)
        result.append(component)
    return result


def breadth_first_search(start: Cell, allowed: set[Cell], grid_size: int):
    """Return hop distances and parent links from ``start`` within ``allowed``."""
    if start not in allowed:
        raise ValueError("BFS start cell must be in the allowed set")
    distances = {start: 0}
    parents: dict[Cell, Cell | None] = {start: None}
    queue = deque([start])
    while queue:
        current = queue.popleft()
        for nxt in grid_neighbors(current, allowed, grid_size):
            if nxt not in distances:
                distances[nxt] = distances[current] + 1
                parents[nxt] = current
                queue.append(nxt)
    return distances, parents


def reconstruct_path(end: Cell, parents: dict[Cell, Cell | None]) -> list[Cell]:
    """Reconstruct a path returned by ``breadth_first_search``."""
    if end not in parents:
        raise ValueError("end cell is not reachable from the BFS start")
    path = []
    current = end
    while current is not None:
        path.append(current)
        current = parents[current]
    path.reverse()
    return path


def grid_distance(a: Cell, b: Cell, step: float) -> float:
    """Euclidean distance between grid cells in metres."""
    return math.hypot(a[0] - b[0], a[1] - b[1]) * step


def approximate_diameter_path(component: set[Cell], grid_size: int) -> list[Cell]:
    """Two BFS sweeps yield a useful long path (not a proven graph diameter)."""
    if not component:
        return []
    distances, _ = breadth_first_search(min(component), component, grid_size)
    endpoint = max(distances, key=distances.get)
    distances, parents = breadth_first_search(endpoint, component, grid_size)
    other = max(distances, key=distances.get)
    return reconstruct_path(other, parents)
