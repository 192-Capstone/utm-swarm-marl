"""Tests for the reusable grid planner used by D3 navigation smoke tests."""
import math

import pytest

from modules.grid_path_planner import (
    approximate_diameter_path,
    breadth_first_search,
    build_free_cells,
    connected_components,
    grid_coordinates,
    reconstruct_path,
)


def test_grid_coordinates_requires_aligned_step():
    assert grid_coordinates(-1.0, 1.0, 0.5) == [-1.0, -0.5, 0.0, 0.5, 1.0]
    with pytest.raises(ValueError):
        grid_coordinates(0.0, 1.0, 0.3)


def test_free_cells_respects_rotated_obstacle_geometry():
    obstacle = {"type": "box", "center": [0.0, 0.0], "half": [2.0, 0.5], "height": 5.0, "yaw": math.pi / 2}
    coords = grid_coordinates(-3.0, 3.0, 1.0)
    free = build_free_cells([obstacle], coords, obstacle_margin=0.0)
    # Vertical long-axis means cells near (0, 1) are inside, while (1, 0) is outside.
    assert (3, 4) not in free  # world (0, 1)
    assert (4, 3) in free      # world (1, 0)


def test_bfs_and_component_path_are_consistent():
    cells = {(x, y) for x in range(5) for y in range(5)} - {(2, y) for y in range(4)}
    components = connected_components(cells, 5)
    assert len(components) == 1
    distances, parents = breadth_first_search((0, 0), components[0], 5)
    path = reconstruct_path((4, 4), parents)
    assert len(path) - 1 == distances[(4, 4)]
    assert path[0] == (0, 0) and path[-1] == (4, 4)
    diameter_path = approximate_diameter_path(components[0], 5)
    assert len(diameter_path) > 1
