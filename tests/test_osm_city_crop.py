import math
import pytest
from shapely.ops import unary_union
from shapely.affinity import translate
from scripts.crop_osm_city import crop_layout
from scripts.import_osm_city import obstacle_to_polygon


def make_box(center, half, height=12.0):
    return {
        "type": "box",
        "center": list(center),
        "half": list(half),
        "height": height,
    }


def test_box_inside_crop_is_preserved_and_recentered():
    item = make_box((1.0, 2.0), (1.0, 1.5))
    result = crop_layout([item], 0.0, 0.0, 10.0, 10.0)

    assert len(result) == 1
    assert result[0]["center"] == [1.0, 2.0]
    assert result[0]["half"] == [1.0, 1.5]
    assert result[0]["height"] == 12.0


def test_box_crossing_crop_boundary_is_clipped():
    item = make_box((5.0, 0.0), (2.0, 1.0))
    result = crop_layout([item], 0.0, 0.0, 10.0, 10.0)

    assert len(result) == 1
    assert result[0]["center"] == [4.0, 0.0]
    assert result[0]["half"][0] == pytest.approx(1.0)
    assert result[0]["half"][1] == pytest.approx(1.0)


def test_box_outside_crop_is_omitted():
    item = make_box((9.0, 9.0), (1.0, 1.0))

    assert crop_layout([item], 0.0, 0.0, 10.0, 10.0) == []


def test_multiple_boxes_are_clipped_independently():
    items = [
        make_box((-5.0, 0.0), (2.0, 1.0)),
        make_box((5.0, 0.0), (2.0, 1.0)),
    ]
    result = crop_layout(items, 0.0, 0.0, 10.0, 10.0)

    assert len(result) == 2
    assert all(item["half"][0] == pytest.approx(1.0) for item in result)


def test_invalid_crop_dimensions_are_rejected():
    with pytest.raises(ValueError):
        crop_layout([], 0.0, 0.0, 0.0, 10.0)


def test_zero_sized_obstacle_is_rejected():
    item = make_box((0.0, 0.0), (1.0, 0.0))

    with pytest.raises(ValueError):
        crop_layout([item], 0.0, 0.0, 10.0, 10.0)


def test_non_box_obstacle_is_rejected():
    pillar = {
        "type": "pillar",
        "center": [0.0, 0.0],
        "radius": 1.0,
        "height": 5.0,
    }

    with pytest.raises(ValueError):
        crop_layout([pillar], 0.0, 0.0, 10.0, 10.0)


def test_oriented_box_fully_inside_crop_preserves_yaw():
    item = make_box((1.0, 1.0), (2.0, 0.5))
    item["yaw"] = math.pi / 4
    result = crop_layout([item], 0.0, 0.0, 10.0, 10.0)
    assert result == [item]


def test_oriented_box_crossing_crop_boundary_is_clipped_and_stays_in_crop():
    from shapely.geometry import box as shapely_box

    item = make_box((4.5, 0.0), (2.0, 0.5))
    item["yaw"] = math.pi / 4
    crop_center = (0.0, 0.0)
    crop_width = crop_height = 10.0
    source_polygon = obstacle_to_polygon(item)
    source_clip = source_polygon.intersection(shapely_box(-5, -5, 5, 5))
    result = crop_layout([item], *crop_center, crop_width, crop_height)
    assert result
    # The clipped boundary pieces are re-boxed axis-aligned and kept within
    # the crop coordinate limits after rounding.
    assert all("yaw" not in obstacle for obstacle in result)
    for obstacle in result:
        x, y = obstacle["center"]
        hx, hy = obstacle["half"]
        assert -5.0001 <= x - hx <= 5.0001
        assert -5.0001 <= x + hx <= 5.0001
        assert -5.0001 <= y - hy <= 5.0001
        assert -5.0001 <= y + hy <= 5.0001
    crop_local = translate(source_clip, xoff=-crop_center[0], yoff=-crop_center[1])
    covered = unary_union([obstacle_to_polygon(obstacle) for obstacle in result])
    assert crop_local.difference(covered).area <= 1e-5
