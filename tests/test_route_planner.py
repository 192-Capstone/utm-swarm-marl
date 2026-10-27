"""
test_route_planner.py
---------------------
Shared route planner (D7, envs/route_planner.py).

Run: python -m pytest tests/test_route_planner.py -q
"""

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from envs.route_planner import (Grid, RouteFollower, build_grid, path_length, plan, segment_free,
                                simplify, snap)
from envs.scenario_manager import DRONE_RADIUS, surface_distance

B = ((-10.0, 10.0), (-10.0, 10.0))
WALL = {'type': 'box', 'center': (0.0, 0.0), 'half': (0.3, 6.0), 'height': 8.0}


def test_empty_map_is_a_straight_line():
    g = build_grid([], B, res=0.25, inflate=0.5)
    pts = simplify(g, plan(g, (-8.0, -3.0), (7.0, 5.0)))
    assert len(pts) == 2
    assert abs(path_length(pts) - math.dist((-8, -3), (7, 5))) < 1e-9


def test_wall_forces_a_detour_and_segments_stay_free():
    g = build_grid([WALL], B, res=0.25, inflate=0.5)
    raw = plan(g, (-5.0, 0.0), (5.0, 0.0))
    pts = simplify(g, raw)
    assert raw is not None and len(pts) >= 3
    assert path_length(pts) > 10.0 + 1.0                       # longer than the straight 10 m
    assert path_length(pts) <= path_length(raw) + 1e-9          # shortcutting never lengthens
    assert all(segment_free(g, a, b) for a, b in zip(pts, pts[1:]))
    for x, y in pts[1:-1]:                                      # waypoints keep the margin
        assert surface_distance(WALL, (x, y)) >= 0.5 - g.res


def test_rotated_box_uses_its_true_shape_not_its_bounding_square():
    turned = {'type': 'box', 'center': (0.0, 0.0), 'half': (4.0, 0.3), 'height': 8.0, 'yaw': math.pi / 4}
    g = build_grid([turned], B, res=0.25, inflate=0.3)
    assert not g.free(g.to_cell((0.0, 0.0)))                    # centre of the bar: blocked
    assert not g.free(g.to_cell((2.0, 2.0)))                    # on the bar's axis: blocked
    assert g.free(g.to_cell((2.0, -2.0)))                       # inside the bounding square, off the bar: free


def test_unreachable_goal_returns_none():
    ring = [{'type': 'box', 'center': (0, 3), 'half': (3.5, 0.5), 'height': 8},
            {'type': 'box', 'center': (0, -3), 'half': (3.5, 0.5), 'height': 8},
            {'type': 'box', 'center': (3, 0), 'half': (0.5, 3.5), 'height': 8},
            {'type': 'box', 'center': (-3, 0), 'half': (0.5, 3.5), 'height': 8}]
    g = build_grid(ring, B, res=0.25, inflate=0.5)
    assert plan(g, (0.0, 0.0), (8.0, 8.0)) is None


def test_start_next_to_a_wall_is_snapped_to_a_free_cell():
    g = build_grid([WALL], B, res=0.25, inflate=0.5)
    start = (-0.6, 0.0)                                         # 0.3 m from the wall: blocked cell
    assert not g.free(g.to_cell(start))
    c = snap(g, start, max_dist=1.0)
    assert c is not None and g.free(c)
    assert plan(g, start, (5.0, 0.0)) is not None
    assert plan(g, start, (5.0, 0.0), snap_dist=0.0) is None    # no snapping -> blocked start


def test_min_height_ignores_low_buildings():
    low = dict(WALL, height=3.0)
    assert not build_grid([low], B, 0.25, 0.5).free((40, 40))   # cell near (0, 0): blocked
    g = build_grid([low], B, 0.25, 0.5, min_height=5.0)         # cruising above 5 m: low wall ignored
    assert g.free(g.to_cell((0.0, 0.0)))


def test_matches_d1_city_grid_and_reachability():
    from envs.city_generator import generate_city, route_length
    rng = np.random.RandomState(0)
    for level in ('easy', 'medium', 'hard'):
        city = generate_city(rng, level)
        W, H = city['size']
        g = build_grid(city['layout'], ((-W / 2, W / 2), (-H / 2, H / 2)), 0.25, DRONE_RADIUS + 0.35)
        assert np.array_equal(g.occ, city['occ'])
        for _ in range(8):
            a = tuple(rng.uniform(-22, 22, 2))
            b = tuple(rng.uniform(-22, 22, 2))
            old = route_length(city, a, b)
            new = plan(g, a, b, snap_dist=0.0)
            assert (old is None) == (new is None)
            if old:
                assert abs(path_length(new) / old - 1) < 0.08


def test_follower_runs_ahead_never_back_and_ends_at_the_goal():
    f = RouteFollower([(0, 0, 2), (10, 0, 2), (10, 10, 2)], lookahead=4.5)
    assert np.allclose(f.target((0, 0, 2)), (4.5, 0, 2))
    assert np.allclose(f.target((9.5, 0.2, 2)), (10, 4, 2))     # around the corner
    assert np.allclose(f.target((2, 0, 2)), (10, 4, 2))         # moving back doesn't pull the target back
    assert np.allclose(f.target((10, 8, 2)), (10, 10, 2))       # last stretch: the goal itself


def test_deterministic():
    g = build_grid([WALL], B, 0.25, 0.5)
    assert plan(g, (-5.0, 1.0), (5.0, -1.0)) == plan(g, (-5.0, 1.0), (5.0, -1.0))


def test_segment_check_matches_point_by_point():
    """segment_free checks all its samples at once; it must agree with checking each sample
    point through Grid.to_cell, including segments that leave the grid (clamped)."""
    layout = [WALL, {'type': 'pillar', 'center': (5.0, 5.0), 'radius': 1.0, 'height': 8.0},
              {'type': 'box', 'center': (-5.0, -5.0), 'half': (2.0, 1.0), 'yaw': 0.6, 'height': 8.0}]
    g = build_grid(layout, B, res=0.25, inflate=0.5)
    rng = np.random.RandomState(0)
    for _ in range(500):
        a, b = rng.uniform(-11, 11, 2), rng.uniform(-11, 11, 2)
        n = max(2, int(math.ceil(math.hypot(*(b - a)) / (g.res / 4))) + 1)
        slow = all(g.free(g.to_cell(a + t * (b - a))) for t in np.linspace(0.0, 1.0, n))
        assert segment_free(g, a, b) == slow
