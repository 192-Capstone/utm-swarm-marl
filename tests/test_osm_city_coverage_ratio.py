"""Regression tests for enforcing the configured rotated-box coverage ratio."""
from shapely.affinity import rotate
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union

from scripts.import_osm_city import geometry_to_boxes, obstacle_to_polygon


def assert_coverage_within_limit(footprint, obstacles, ratio_limit=1.15):
    assert obstacles
    covered = unary_union([obstacle_to_polygon(item) for item in obstacles])
    assert footprint.difference(covered).area <= max(1e-5, footprint.area * 1e-7)
    assert covered.area / footprint.area <= ratio_limit + 1e-3


def test_round_footprint_uses_refined_bands_to_meet_area_ratio():
    # A circular footprint has an MRR area ratio around 1.27, so one box is too loose.
    footprint = Point(0, 0).buffer(10, quad_segs=32)
    obstacles = geometry_to_boxes(
        footprint, 0, 0, 12, min_side_m=0, max_box_area_ratio=1.15
    )
    assert_coverage_within_limit(footprint, obstacles)
    assert len(obstacles) > 1
    assert len(obstacles) <= 20


def test_rotated_round_footprint_meets_area_ratio_without_gaps():
    footprint = rotate(Point(0, 0).buffer(10, quad_segs=24), 31, origin="centroid")
    obstacles = geometry_to_boxes(
        footprint, 0, 0, 12, min_side_m=0, max_box_area_ratio=1.15
    )
    assert_coverage_within_limit(footprint, obstacles)


def test_rotated_concave_footprint_remains_covered_within_limit():
    footprint = rotate(
        Polygon([(0, 0), (12, 0), (12, 2), (4, 2), (4, 8), (12, 8), (12, 10), (0, 10)]),
        23,
        origin="centroid",
    )
    obstacles = geometry_to_boxes(
        footprint, 0, 0, 12, min_side_m=0, max_box_area_ratio=1.15
    )
    assert_coverage_within_limit(footprint, obstacles)
