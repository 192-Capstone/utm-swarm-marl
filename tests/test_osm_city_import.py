"""Offline unit tests for the D3 OpenStreetMap importer helpers."""
import math
import unittest

from shapely.affinity import rotate
from shapely.geometry import Polygon, box as shapely_box
from shapely.ops import unary_union

from scripts.import_osm_city import (
    bounds_to_box,
    geometry_to_boxes,
    obstacle_to_polygon,
    parse_building_height,
    parse_building_height_with_source,
)


class BoundsToBoxTests(unittest.TestCase):
    def test_projects_bounds_to_local_metres(self):
        obstacle = bounds_to_box(
            bounds=(100.0, 200.0, 110.0, 220.0),
            origin_x=100.0,
            origin_y=200.0,
            height_m=12.0,
        )
        self.assertEqual(obstacle["type"], "box")
        self.assertEqual(obstacle["center"], [5.0, 10.0])
        self.assertEqual(obstacle["half"], [5.0, 10.0])
        self.assertEqual(obstacle["height"], 12.0)

    def test_skips_too_small_footprint(self):
        obstacle = bounds_to_box(
            bounds=(0.0, 0.0, 1.5, 8.0), origin_x=0.0, origin_y=0.0,
            height_m=9.0, min_side_m=2.0,
        )
        self.assertIsNone(obstacle)

    def test_skips_degenerate_or_non_finite_bounds(self):
        self.assertIsNone(bounds_to_box((0.0, 0.0, 0.0, 2.0), 0, 0, 10))
        self.assertIsNone(bounds_to_box((0.0, 0.0, math.inf, 2.0), 0, 0, 10))


class BuildingHeightTests(unittest.TestCase):
    def test_uses_height_in_metres(self):
        self.assertEqual(parse_building_height({"height": "14.5 m"}), 14.5)

    def test_converts_explicit_feet(self):
        self.assertEqual(parse_building_height({"height": "30 ft"}), 9.14)

    def test_uses_building_levels_when_height_missing(self):
        self.assertEqual(parse_building_height({"building:levels": "4"}), 12.0)

    def test_uses_default_when_no_height_tags(self):
        self.assertEqual(parse_building_height({}), 12.0)

    def test_height_source_is_recorded(self):
        self.assertEqual(parse_building_height_with_source({"height": "18 m"}), (18.0, "osm_height"))
        self.assertEqual(parse_building_height_with_source({"building:levels": "4"}), (12.0, "building_levels_estimate"))
        self.assertEqual(parse_building_height_with_source({}), (12.0, "default_assumption"))

    def test_height_tag_has_priority_over_levels(self):
        self.assertEqual(parse_building_height_with_source({"height": "15", "building:levels": "4"}), (15.0, "osm_height"))


def assert_footprint_covered(testcase, footprint, obstacles, max_ratio=1.15):
    box_polygons = [obstacle_to_polygon(ob) for ob in obstacles]
    covered = unary_union(box_polygons)
    testcase.assertLessEqual(footprint.difference(covered).area, 1e-5)
    ratio = covered.area / footprint.area
    testcase.assertGreaterEqual(ratio, 0.95)
    testcase.assertLessEqual(ratio, max_ratio + 1e-3)
    return ratio


class FootprintDecompositionTests(unittest.TestCase):
    def test_l_shaped_footprint_uses_rotated_boxes_and_is_covered(self):
        footprint = Polygon([(0, 0), (10, 0), (10, 3), (3, 3), (3, 10), (0, 10)])
        obstacles = geometry_to_boxes(footprint, 0, 0, 12, min_side_m=2, max_box_area_ratio=1.15)
        self.assertIsNotNone(obstacles)
        self.assertGreater(len(obstacles), 1)
        self.assertLessEqual(len(obstacles), 4)
        assert_footprint_covered(self, footprint, obstacles)

    def test_rectangle_merges_to_one_axis_aligned_box(self):
        footprint = Polygon([(0, 0), (10, 0), (10, 8), (0, 8)])
        obstacles = geometry_to_boxes(footprint, 0, 0, 10)
        self.assertEqual(len(obstacles), 1)
        self.assertAlmostEqual(obstacles[0]["half"][0], 5.001, places=3)
        self.assertAlmostEqual(obstacles[0]["half"][1], 4.001, places=3)
        self.assertNotIn("yaw", obstacles[0])
        assert_footprint_covered(self, footprint, obstacles)

    def test_rotated_rectangle_exports_yaw_in_radians(self):
        footprint = rotate(shapely_box(0, 0, 12, 3), 30, origin="center")
        obstacles = geometry_to_boxes(footprint, 0, 0, 9, min_side_m=0)
        self.assertEqual(len(obstacles), 1)
        self.assertAlmostEqual(obstacles[0]["yaw"], math.radians(30), places=5)
        assert_footprint_covered(self, footprint, obstacles)

    def test_rotated_concave_footprint_is_covered_with_low_overage(self):
        footprint = rotate(
            Polygon([(0, 0), (10, 0), (10, 3), (3, 3), (3, 10), (0, 10)]),
            27, origin="centroid",
        )
        obstacles = geometry_to_boxes(footprint, 0, 0, 12, min_side_m=0, max_box_area_ratio=1.15)
        self.assertIsNotNone(obstacles)
        self.assertGreater(len(obstacles), 1)
        self.assertLessEqual(len(obstacles), 4)
        assert_footprint_covered(self, footprint, obstacles)

    def test_hole_is_not_filled_by_box_union(self):
        footprint = Polygon(
            [(0, 0), (10, 0), (10, 10), (0, 10)],
            holes=[[(3, 3), (7, 3), (7, 7), (3, 7)]],
        )
        obstacles = geometry_to_boxes(footprint, 0, 0, 12, min_side_m=0)
        self.assertIsNotNone(obstacles)
        assert_footprint_covered(self, footprint, obstacles)
        hole = shapely_box(3, 3, 7, 7)
        covered = unary_union([obstacle_to_polygon(ob) for ob in obstacles])
        self.assertLessEqual(covered.intersection(hole).area, 0.02)

    def test_rejects_area_ratio_below_one(self):
        footprint = shapely_box(0, 0, 5, 5)
        self.assertIsNone(geometry_to_boxes(footprint, 0, 0, 10, max_box_area_ratio=0.99))


if __name__ == "__main__":
    unittest.main()
