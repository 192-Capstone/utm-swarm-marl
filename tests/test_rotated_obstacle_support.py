"""Shared geometry contract tests for optional yaw on box obstacles."""
import math

import numpy as np
import pytest

from envs.scenario_manager import footprint_radius, surface_distance
from baselines.repulsion_controller import closest_point


def test_legacy_box_without_yaw_remains_axis_aligned():
    obstacle = {"type": "box", "center": [0.0, 0.0], "half": [2.0, 0.5], "height": 10.0}
    assert surface_distance(obstacle, [0.0, 0.0]) == pytest.approx(-0.5)
    assert surface_distance(obstacle, [3.0, 0.0]) == pytest.approx(1.0)
    assert surface_distance(obstacle, [0.0, 1.5]) == pytest.approx(1.0)


def test_rotated_box_distance_uses_local_box_axes():
    yaw = math.pi / 4
    obstacle = {"type": "box", "center": [0.0, 0.0], "half": [2.0, 0.5], "height": 10.0, "yaw": yaw}
    # A point 1 m along the box's local short-axis has 0.5 m clearance.
    point = [-math.sin(yaw) * 1.0, math.cos(yaw) * 1.0]
    assert surface_distance(obstacle, point) == pytest.approx(0.5, abs=1e-7)
    # Three metres along the local long axis has 1 m clearance.
    point = [math.cos(yaw) * 3.0, math.sin(yaw) * 3.0]
    assert surface_distance(obstacle, point) == pytest.approx(1.0, abs=1e-7)


def test_rotating_box_does_not_change_its_footprint_radius():
    axis_box = {"type": "box", "center": [0.0, 0.0], "half": [3.0, 1.0], "height": 10.0}
    rotated = dict(axis_box, yaw=0.73)
    assert footprint_radius(axis_box) == pytest.approx(footprint_radius(rotated))


def test_yaw_is_optional_but_invalid_yaw_does_not_silently_pass():
    obstacle = {"type": "box", "center": [0.0, 0.0], "half": [1.0, 1.0], "height": 10.0, "yaw": float("nan")}
    # The geometry core expects finite data; the caller's schema validation is
    # responsible for rejecting malformed yaw. The distance function returns
    # NaN rather than a misleading finite clearance.
    assert math.isnan(surface_distance(obstacle, [3.0, 0.0]))


def test_teacher_closest_point_respects_rotated_box_axes():
    yaw = math.pi / 4
    obstacle = {"type": "box", "center": [2.0, -3.0], "half": [2.0, 0.5],
                "height": 10.0, "yaw": yaw}
    # Point is three metres along the box's local long axis; closest surface is
    # at local x=+2 and local y=0, then transformed back to world coordinates.
    q = np.array([2.0 + 3.0 * math.cos(yaw), -3.0 + 3.0 * math.sin(yaw)])
    expected = np.array([2.0 + 2.0 * math.cos(yaw), -3.0 + 2.0 * math.sin(yaw)])
    assert np.allclose(closest_point(obstacle, q), expected, atol=1e-7)


def test_teacher_closest_point_from_inside_returns_a_boundary_point():
    obstacle = {"type": "box", "center": [0.0, 0.0], "half": [2.0, 0.5],
                "height": 10.0, "yaw": math.pi / 4}
    result = closest_point(obstacle, np.array([0.0, 0.0]))
    assert surface_distance(obstacle, result) == pytest.approx(0.0, abs=1e-7)
