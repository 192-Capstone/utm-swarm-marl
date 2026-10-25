"""Offline tests for loading and sampling OSM city scenarios."""
from pathlib import Path

import numpy as np
import pytest
import yaml

from envs.scenario_manager import surface_distance
from envs.osm_scenario import (
    layout_bounds,
    load_osm_layout,
    sample_flights,
    validate_osm_layout,
)


def sample_layout():
    # Four corner markers define a 100 m-scale envelope while leaving a large
    # open centre for reproducible free-space flight sampling.
    return [
        {"type": "box", "center": [-50.0, -50.0], "half": [1.0, 1.0], "height": 12.0},
        {"type": "box", "center": [-50.0, 50.0], "half": [1.0, 1.0], "height": 12.0},
        {"type": "box", "center": [50.0, -50.0], "half": [1.0, 1.0], "height": 12.0},
        {"type": "box", "center": [50.0, 50.0], "half": [1.0, 1.0], "height": 12.0},
        {"type": "box", "center": [0.0, 0.0], "half": [3.0, 1.0], "height": 10.0, "yaw": 0.4},
    ]


def test_validate_layout_accepts_legacy_and_rotated_boxes():
    result = validate_osm_layout(sample_layout())
    assert len(result) == 5
    assert result[-1]["yaw"] == pytest.approx(0.4)
    assert "yaw" not in result[0]


def test_validate_layout_rejects_invalid_dimensions_and_non_box_objects():
    with pytest.raises(ValueError):
        validate_osm_layout([{"type": "box", "center": [0, 0], "half": [1, 0], "height": 2}])
    with pytest.raises(ValueError):
        validate_osm_layout([{"type": "pillar", "center": [0, 0], "radius": 1, "height": 2}])


def test_layout_bounds_accounts_for_rotated_extent():
    bounds = layout_bounds([{"type": "box", "center": [0, 0], "half": [2, 1], "height": 5, "yaw": np.pi / 2}])
    assert bounds == pytest.approx((-1.0, -2.0, 1.0, 2.0))


def test_load_osm_layout_reads_yaml_and_validates(tmp_path: Path):
    path = tmp_path / "layout.yaml"
    path.write_text(yaml.safe_dump(sample_layout()), encoding="utf-8")
    result = load_osm_layout(path)
    assert len(result) == 5
    with pytest.raises(FileNotFoundError):
        load_osm_layout(tmp_path / "missing.yaml")


def test_sample_flights_uses_full_map_bounds_and_returns_free_points():
    layout = sample_layout()
    spawns, goals = sample_flights(
        layout,
        3,
        rng=np.random.RandomState(2026),
        min_goal_distance_m=10.0,
        max_goal_distance_m=40.0,
        obstacle_clearance_m=2.0,
        min_point_separation_m=5.0,
        boundary_margin_m=2.0,
        min_altitude_m=2.0,
        max_altitude_m=4.0,
        max_sampling_attempts=1000,
    )
    assert set(spawns) == {"drone_0", "drone_1", "drone_2"}
    assert set(goals) == set(spawns)
    points = []
    for key in sorted(spawns):
        s = spawns[key]
        g = goals[key]
        assert 2.0 <= s[2] <= 4.0
        assert 2.0 <= g[2] <= 4.0
        assert 10.0 <= np.linalg.norm(np.asarray(s[:2]) - np.asarray(g[:2])) <= 40.0
        for point in (s[:2], g[:2]):
            assert all(surface_distance(ob, point) >= 2.0 for ob in layout)
            assert all(np.linalg.norm(np.asarray(point) - np.asarray(prior)) >= 5.0 for prior in points)
            points.append(point)


def test_sample_flights_rejects_too_small_attempt_limit():
    with pytest.raises(ValueError):
        sample_flights(sample_layout(), 1, rng=np.random.RandomState(1), max_sampling_attempts=0)


def test_stage4_osm_config_targets_full_rotated_layout_not_the_crop():
    config_path = Path(__file__).resolve().parents[1] / "configs" / "config_stage3_osm_ecity.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert config["scenario_mix"] == {"osm_ecity": 1.0}
    assert config["training"]["initial_curriculum_stage"] == 4
    assert config["osm_scenario"]["layout_path"].endswith("bengaluru_ecity_phase1_rotated_v2_layout.yaml")
    assert config["lidar"]["enabled"] is True
