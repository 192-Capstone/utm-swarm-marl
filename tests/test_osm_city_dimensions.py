from scripts.import_osm_city import bounds_to_box


def test_tiny_dimension_that_rounds_to_zero_is_rejected():
    # Positive raw depth, but half-depth rounds to 0.000 m.
    result = bounds_to_box(
        (0.0, 0.0, 46.154, 0.0004),
        origin_x=0.0,
        origin_y=0.0,
        height_m=12.0,
        min_side_m=0.0,
    )
    assert result is None


def test_normal_box_keeps_positive_dimensions():
    result = bounds_to_box(
        (0.0, 0.0, 10.0, 4.0),
        origin_x=0.0,
        origin_y=0.0,
        height_m=12.0,
        min_side_m=0.0,
    )
    assert result is not None
    assert result["half"][0] > 0
    assert result["half"][1] > 0
