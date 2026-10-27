"""
test_airspace.py
----------------
H3-indexed airspace (envs/airspace.py).

  A. Local metres <-> lat/lng round trip is sub-millimetre, and distances match H3's
     great-circle distance.
  B. A point lies inside the hexagon of its own cell; cell centres map back near the point.
  C. k_for_radius: grid_disk(cell, k) contains every point within the radius (brute force).
  D. Obstacles -> cells: every cell centre inside a (rotated) box is found; cover=True also
     catches partly-overlapping cells; pillars and tiny obstacles work.
  E. Altitude layers.

Run: python -m pytest tests/test_airspace.py -q
"""

import math
import os
import sys

import numpy as np
import pytest

h3 = pytest.importorskip('h3')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from envs.airspace import ANCHOR_ECITY, Airspace          # noqa: E402
from envs.scenario_manager import surface_distance        # noqa: E402


def inside(poly, p):
    """Point-in-polygon (ray casting)."""
    x, y = p
    hit = False
    for (x1, y1), (x2, y2) in zip(poly, poly[1:] + poly[:1]):
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            hit = not hit
    return hit


def test_local_latlng_round_trip_and_distances():
    sky = Airspace()
    rng = np.random.default_rng(0)
    xy = rng.uniform(-2000, 2000, (500, 2))
    lat, lng = sky.to_latlng(xy[:, 0], xy[:, 1])
    bx, by = sky.to_local(lat, lng)
    assert np.abs(np.column_stack([bx, by]) - xy).max() < 1e-6          # sub-micrometre
    assert sky.to_latlng(0.0, 0.0) == pytest.approx(ANCHOR_ECITY)
    for (x1, y1), (x2, y2) in zip(xy[:100], xy[100:200]):
        a, b = sky.to_latlng(x1, y1), sky.to_latlng(x2, y2)
        gc = h3.great_circle_distance((float(a[0]), float(a[1])), (float(b[0]), float(b[1])), unit='m')
        assert abs(gc - math.hypot(x2 - x1, y2 - y1)) < 1e-4 * max(gc, 1.0) + 1e-3


def test_point_is_inside_its_cell():
    sky = Airspace(resolution=12)
    rng = np.random.default_rng(1)
    for x, y in rng.uniform(-300, 300, (300, 2)):
        c = sky.cell(x, y)
        assert h3.get_resolution(c) == 12
        assert inside(sky.boundary(c), (x, y))
        cx, cy = sky.center(c)
        assert math.hypot(cx - x, cy - y) <= sky.edge_length() * 1.2   # within one cell radius
    assert sky.cells([[1.0, 2.0], [3.0, 4.0]]) == [sky.cell(1.0, 2.0), sky.cell(3.0, 4.0)]


@pytest.mark.parametrize('res,radius', [(11, 30.0), (12, 30.0), (12, 6.0), (13, 6.0)])
def test_disk_of_k_for_radius_contains_every_point_in_range(res, radius):
    sky = Airspace(resolution=res)
    k = sky.k_for_radius(radius)
    rng = np.random.default_rng(res)
    for _ in range(200):
        p = rng.uniform(-500, 500, 2)
        ring = sky.disk(sky.cell(*p), k)
        ang = rng.uniform(0, 2 * math.pi, 40)
        rad = radius * np.sqrt(rng.uniform(0, 1, 40))
        for a, r in zip(ang, rad):
            q = p + r * np.array([math.cos(a), math.sin(a)])
            assert sky.cell(*q) in ring


def test_box_and_pillar_to_cells():
    sky = Airspace(resolution=13)                       # 4.1 m hexagons
    box = {'type': 'box', 'center': (20.0, -10.0), 'half': (15.0, 4.0), 'height': 12.0, 'yaw': 0.5}
    exact = sky.cells_for_obstacle(box, cover=False)
    cover = sky.cells_for_obstacle(box, cover=True)
    assert exact and exact <= cover
    for c in exact:                                     # every cell found has its centre inside
        assert surface_distance(box, sky.center(c)) <= 1e-6
    # every cell whose centre is inside the box is found
    probe = sky.disk(sky.cell(20.0, -10.0), 8)
    for c in probe:
        if surface_distance(box, sky.center(c)) < -0.01:
            assert c in exact
    # cover: every cell touching the box (its centre within one cell radius of the box) is found
    for c in probe:
        if surface_distance(box, sky.center(c)) < sky.edge_length() * 0.5:
            assert c in cover
    pillar = {'type': 'pillar', 'center': (0.0, 0.0), 'radius': 6.0, 'height': 8.0}
    assert sky.cell(0.0, 0.0) in sky.cells_for_obstacle(pillar)
    tiny = {'type': 'pillar', 'center': (5.0, 5.0), 'radius': 0.3, 'height': 8.0}
    assert sky.cells_for_obstacle(tiny, cover=False) == {sky.cell(5.0, 5.0)}


def test_altitude_layers():
    sky = Airspace(layer_floor=0.0, layer_height=5.0)
    assert [sky.layer(z) for z in (0.0, 4.99, 5.0, 12.0)] == [0, 0, 1, 2]
    assert sky.layer_range(2) == (10.0, 15.0)
    assert sky.cell3d(1.0, 2.0, 7.0) == (sky.cell(1.0, 2.0), 1)
