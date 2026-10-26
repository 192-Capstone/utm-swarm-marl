"""Import OpenStreetMap building footprints into the project's obstacle layout.

The output layout is a plain YAML list compatible with envs/scenario_manager.py:
  - type: box
    center: [x_east_m, y_north_m]
    half: [half_width_m, half_depth_m]
    height: height_m

Buildings are approximated by a conservative set of oriented boxes. Near-rectangular
footprints use a minimum rotated rectangle; concave footprints are decomposed into
smaller pieces and merged where possible. Box ``yaw`` is counter-clockwise radians.
The layout remains a plain YAML list compatible with the project obstacle contract.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import math
import re
import sys
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import yaml


# Default query location: Electronic City Phase I, Bengaluru.
# Override with --center-lat/--center-lon for another site.
DEFAULT_CENTER_LAT = 12.84968
DEFAULT_CENTER_LON = 77.66498
DEFAULT_QUERY_DISTANCE_M = 300.0
DEFAULT_BUILDING_HEIGHT_M = 12.0
DEFAULT_FLOOR_HEIGHT_M = 3.0
DEFAULT_MIN_SIDE_M = 2.0


def _has_value(value: Any) -> bool:
    """Return False for None, blank strings, and NaN-like numeric values."""
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip()) and value.strip().lower() not in {"nan", "none", "null"}
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return True


def bounds_to_box(
    bounds: tuple[float, float, float, float],
    origin_x: float,
    origin_y: float,
    height_m: float,
    min_side_m: float = DEFAULT_MIN_SIDE_M,
) -> dict[str, Any] | None:
    """Convert projected bounds (minx, miny, maxx, maxy) to a local box.

    The chosen origin is subtracted so the selected map centre becomes (0, 0).
    Returns None for degenerate/tiny footprints that should be filtered out.
    """
    min_x, min_y, max_x, max_y = (float(v) for v in bounds)
    values = (min_x, min_y, max_x, max_y, origin_x, origin_y, height_m, min_side_m)
    if not all(math.isfinite(float(v)) for v in values):
        return None
    width = max_x - min_x
    depth = max_y - min_y
    if width <= 0 or depth <= 0 or height_m <= 0:
        return None
    if width < min_side_m or depth < min_side_m:
        return None

    centre_x = (min_x + max_x) / 2.0 - origin_x
    centre_y = (min_y + max_y) / 2.0 - origin_y
    # Round dimensions, then reject fragments that would become degenerate.
    # Very thin geometry slices can be positive before rounding but become 0.0.
    half_width = round(width / 2.0, 3)
    half_depth = round(depth / 2.0, 3)
    if half_width <= 0 or half_depth <= 0:
        return None

    return {
        "type": "box",
        "center": [round(centre_x, 3), round(centre_y, 3)],
        "half": [half_width, half_depth],
        "height": round(float(height_m), 2),
    }


def _download_buildings(center_lat: float, center_lon: float, distance_m: float):
    """Download OSM buildings and project them to a metre-based UTM CRS."""
    try:
        import geopandas as gpd
        import osmnx as ox
        from shapely.geometry import Point
    except ImportError as exc:
        raise RuntimeError(
            "OSM import extras (OSMnx, GeoPandas and Shapely) are required to regenerate map data. "
            "Install them with: python -m pip install -r requirements-optional-osm.txt"
        ) from exc

    print(
        f"Downloading building footprints near ({center_lat:.6f}, {center_lon:.6f}); "
        f"query distance={distance_m:g} m ...",
        flush=True,
    )
    try:
        buildings = ox.features.features_from_point(
            center_point=(center_lat, center_lon),
            tags={"building": True},
            dist=distance_m,
        )
    except Exception as exc:  # network/Overpass failures should be actionable
        raise RuntimeError(
            "OpenStreetMap query failed. Check your internet connection and try again; "
            f"the Overpass service may also be temporarily busy. Details: {exc}"
        ) from exc

    if buildings.empty:
        raise RuntimeError("No OSM building footprints were returned for this area.")

    projected = ox.projection.project_gdf(buildings)
    centre_series = gpd.GeoSeries(
        [Point(center_lon, center_lat)], crs="EPSG:4326"
    ).to_crs(projected.crs)
    centre_projected = centre_series.iloc[0]
    return projected, float(centre_projected.x), float(centre_projected.y), str(projected.crs)


def _polygon_components(geometry):
    """Yield polygon components from Polygon/MultiPolygon/GeometryCollection."""
    if geometry is None or geometry.is_empty:
        return
    if geometry.geom_type == "Polygon":
        yield geometry
    elif geometry.geom_type in {"MultiPolygon", "GeometryCollection"}:
        for part in geometry.geoms:
            yield from _polygon_components(part)


def _merged_vertical_bounds(bounds_list, tolerance: float = 1e-6):
    """Merge vertically adjacent rectangles when their x bounds match exactly."""
    ordered = sorted(bounds_list, key=lambda b: (round(b[0], 6), round(b[2], 6), b[1], b[3]))
    merged = []
    for min_x, min_y, max_x, max_y in ordered:
        if merged:
            prev = merged[-1]
            same_x = abs(prev[0] - min_x) <= tolerance and abs(prev[2] - max_x) <= tolerance
            touches = abs(prev[3] - min_y) <= tolerance
            if same_x and touches:
                merged[-1] = (prev[0], prev[1], prev[2], max_y)
                continue
        merged.append((min_x, min_y, max_x, max_y))
    return merged


def obstacle_to_polygon(obstacle):
    """Build a Shapely polygon for a box obstacle, respecting optional yaw.

    ``half`` are half-extents along the box's own local axes. ``yaw`` is a
    counter-clockwise rotation in radians about +Z; omitted yaw means 0.
    """
    from shapely.geometry import Polygon

    if obstacle.get("type") != "box":
        raise ValueError(f"Expected a box obstacle, got {obstacle.get('type')!r}")
    cx, cy = (float(v) for v in obstacle["center"])
    hx, hy = (float(v) for v in obstacle["half"])
    yaw = float(obstacle.get("yaw", 0.0))
    if not all(math.isfinite(v) for v in (cx, cy, hx, hy, yaw)) or hx <= 0 or hy <= 0:
        raise ValueError("Box centre, half-extents, and yaw must be finite; half-extents must be positive")
    c, s = math.cos(yaw), math.sin(yaw)
    corners = []
    for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
        lx, ly = sx * hx, sy * hy
        corners.append((cx + lx * c - ly * s, cy + lx * s + ly * c))
    return Polygon(corners)


def _normalise_yaw(yaw: float) -> float:
    """Normalise an undirected rectangle's yaw to [-pi/2, pi/2)."""
    return (float(yaw) + math.pi / 2.0) % math.pi - math.pi / 2.0


def _rectangle_to_obstacle(rectangle, origin_x: float, origin_y: float, height_m: float):
    """Convert a minimum-rotated rectangle into the project's box schema."""
    coords = list(rectangle.exterior.coords)[:-1]
    if len(coords) != 4:
        return None
    edges = []
    for idx in range(4):
        x0, y0 = coords[idx]
        x1, y1 = coords[(idx + 1) % 4]
        dx, dy = float(x1 - x0), float(y1 - y0)
        length = math.hypot(dx, dy)
        if length > 1e-9:
            edges.append((length, dx, dy))
    if len(edges) != 4:
        return None
    # The longer edge becomes the box's local X axis, so hx >= hy.
    length, dx, dy = max(edges, key=lambda edge: edge[0])
    other_length = min(edge[0] for edge in edges)
    if length <= 0 or other_length <= 0:
        return None
    yaw = _normalise_yaw(math.atan2(dy, dx))
    centre = rectangle.centroid
    # Small conservative padding prevents rounding from cutting off a footprint.
    hx = math.ceil((length / 2.0 + 0.001) * 1000.0 - 1e-9) / 1000.0
    hy = math.ceil((other_length / 2.0 + 0.001) * 1000.0 - 1e-9) / 1000.0
    obstacle = {
        "type": "box",
        "center": [round(float(centre.x) - origin_x, 4), round(float(centre.y) - origin_y, 4)],
        "half": [hx, hy],
        "height": round(float(height_m), 2),
    }
    # Omit yaw for ordinary axis-aligned rectangles to remain schema-compatible
    # with existing layouts and generated Stage 2 obstacles.
    if abs(yaw) > 1e-7:
        obstacle["yaw"] = round(yaw, 8)
    return obstacle


def _triangulate_polygon(polygon):
    """Triangulate a polygon without filling concavities or holes when possible."""
    try:
        from shapely import constrained_delaunay_triangles
        triangulation = constrained_delaunay_triangles(polygon)
        return [part for part in triangulation.geoms
                if part.geom_type == "Polygon" and part.area > 1e-8]
    except (ImportError, AttributeError):
        # Compatibility fallback for Shapely releases without constrained CDT.
        from shapely.ops import triangulate
        parts = []
        for triangle in triangulate(polygon):
            clipped = triangle.intersection(polygon)
            parts.extend(part for part in _polygon_components(clipped) if part.area > 1e-8)
        return parts


def _box_decomposition_polygons(polygon, max_area_ratio: float):
    """Partition a footprint into a compact set of shapes represented by boxes.

    Convex/near-rectangular footprints use one minimum-rotated rectangle. More
    concave footprints are constrained-triangulated, then adjacent pieces are
    greedily merged whenever replacing their two boxes with one box does not
    increase their combined covered area. This avoids the fixed horizontal
    strip staircase and handles holes/concavities without leaving coverage gaps.
    """
    minimum_rectangle = polygon.minimum_rotated_rectangle
    if polygon.area > 0 and minimum_rectangle.area / polygon.area <= max_area_ratio:
        return [polygon]

    pieces = _triangulate_polygon(polygon)
    if not pieces:
        return [polygon]

    # Greedy edge-adjacent merges. Comparing union areas of the box footprints
    # means a merge is accepted only when it does not add meaningful covered area.
    from shapely.ops import unary_union
    groups = list(pieces)
    tolerance = max(1e-7, polygon.area * 1e-9)
    while len(groups) > 1:
        rectangles = [group.minimum_rotated_rectangle for group in groups]
        best = None
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                if groups[i].boundary.intersection(groups[j].boundary).length <= 1e-8:
                    continue
                merged = groups[i].union(groups[j])
                if merged.geom_type != "Polygon" or merged.area <= 1e-8:
                    continue
                merged_rectangle = merged.minimum_rotated_rectangle
                old_cover_area = rectangles[i].union(rectangles[j]).area
                added_cover = merged_rectangle.area - old_cover_area
                if added_cover <= tolerance:
                    key = (added_cover, merged_rectangle.area / merged.area, i, j)
                    if best is None or key < best[0]:
                        best = (key, merged)
        if best is None:
            break
        (_, _, i, j), merged = best
        groups = [group for idx, group in enumerate(groups) if idx not in (i, j)]
        groups.append(merged)
    return groups


def _adaptive_band_cover_boxes(
    polygon,
    origin_x: float,
    origin_y: float,
    height_m: float,
    max_box_area_ratio: float,
    max_bands: int = 4096,
) -> list[dict[str, Any]] | None:
    """Cover a difficult footprint with progressively thinner rotated bands.

    This is a fallback for footprints whose triangulated/greedily merged boxes
    exceed the requested union-area ratio. Bands are aligned to the footprint's
    minimum-rotated-rectangle orientation, and each connected polygon part in a
    band is enclosed by its own oriented rectangle. The band count doubles until
    the exported box union covers the footprint and meets ``max_box_area_ratio``.
    """
    from shapely.affinity import rotate, translate
    from shapely.geometry import box as shapely_box
    from shapely.ops import unary_union

    minimum_rectangle = polygon.minimum_rotated_rectangle
    coordinates = list(minimum_rectangle.exterior.coords)[:-1]
    edges = []
    for index in range(len(coordinates)):
        x0, y0 = coordinates[index]
        x1, y1 = coordinates[(index + 1) % len(coordinates)]
        dx, dy = float(x1 - x0), float(y1 - y0)
        length = math.hypot(dx, dy)
        if length > 1e-9:
            edges.append((length, dx, dy))
    if len(edges) < 4:
        return None
    _, dx, dy = max(edges, key=lambda edge: edge[0])
    yaw = _normalise_yaw(math.atan2(dy, dx))
    yaw_degrees = math.degrees(yaw)
    origin = polygon.centroid
    aligned = rotate(polygon, -yaw_degrees, origin=origin)
    min_x, min_y, max_x, max_y = (float(v) for v in aligned.bounds)
    span_y = max_y - min_y
    if not math.isfinite(span_y) or span_y <= 1e-8:
        return None

    source_local = translate(polygon, xoff=-origin_x, yoff=-origin_y)
    gap_tolerance = max(1e-5, source_local.area * 1e-7)
    padding = max(1e-6, (max_x - min_x) * 1e-9)

    band_count = 2
    while band_count <= max_bands:
        band_height = span_y / band_count
        candidates: list[dict[str, Any]] = []
        for band_index in range(band_count):
            y0 = min_y + band_index * band_height
            y1 = max_y if band_index == band_count - 1 else min_y + (band_index + 1) * band_height
            slab = shapely_box(min_x - padding, y0, max_x + padding, y1)
            clipped = aligned.intersection(slab)
            for piece in _polygon_components(clipped):
                if piece.area <= 1e-9:
                    continue
                x0, py0, x1, py1 = (float(v) for v in piece.bounds)
                if x1 - x0 <= 1e-8 or py1 - py0 <= 1e-8:
                    continue
                aligned_rectangle = shapely_box(x0, py0, x1, py1)
                world_rectangle = rotate(aligned_rectangle, yaw_degrees, origin=origin)
                obstacle = _rectangle_to_obstacle(world_rectangle, origin_x, origin_y, height_m)
                if obstacle is not None:
                    candidates.append(obstacle)

        if candidates:
            covered = unary_union([obstacle_to_polygon(item) for item in candidates])
            uncovered_area = source_local.difference(covered).area
            area_ratio = covered.area / source_local.area if source_local.area > 0 else math.inf
            if uncovered_area <= gap_tolerance and area_ratio <= max_box_area_ratio + 1e-3:
                return candidates
        band_count *= 2
    return None


def geometry_to_boxes(
    geometry,
    origin_x: float,
    origin_y: float,
    height_m: float,
    min_side_m: float = DEFAULT_MIN_SIDE_M,
    max_box_area_ratio: float = 1.15,
) -> list[dict[str, Any]] | None:
    """Approximate a footprint by rotated boxes, using radians for ``yaw``.

    Rectangular/near-rectangular polygons use their minimum rotated rectangle.
    Concave shapes are split into adjacent pieces and merged where doing so
    doesn't materially increase covered area. The result is conservative: the
    union of exported boxes should cover the footprint, with extra area tracked
    as an approximation ratio.
    """
    try:
        from shapely.affinity import translate
        from shapely.ops import unary_union
    except ImportError as exc:
        raise RuntimeError("Shapely is required for footprint decomposition.") from exc

    numeric = (height_m, min_side_m, max_box_area_ratio, origin_x, origin_y)
    if (not all(math.isfinite(float(v)) for v in numeric) or height_m <= 0
            or min_side_m < 0 or max_box_area_ratio < 1.0):
        return None
    if geometry is None or geometry.is_empty:
        return None
    try:
        if not geometry.is_valid:
            geometry = geometry.buffer(0)
        if geometry is None or geometry.is_empty or geometry.geom_type not in {"Polygon", "MultiPolygon"}:
            return None

        boxes: list[dict[str, Any]] = []
        for polygon in _polygon_components(geometry):
            if polygon.area <= 1e-7:
                continue
            min_rect = polygon.minimum_rotated_rectangle
            if min_rect.geom_type != "Polygon" or len(min_rect.exterior.coords) < 5:
                continue
            rect_coords = list(min_rect.exterior.coords)
            edge_lengths = [math.hypot(rect_coords[i + 1][0] - rect_coords[i][0],
                                       rect_coords[i + 1][1] - rect_coords[i][1])
                            for i in range(4)]
            side_lengths = sorted(length for length in edge_lengths if length > 1e-8)
            if len(side_lengths) < 4 or side_lengths[0] < min_side_m:
                continue

            # First try the compact triangulate/merge decomposition. It is fast
            # and usually produces the fewest boxes.
            candidate_boxes = []
            for piece in _box_decomposition_polygons(polygon, max_box_area_ratio):
                rectangle = piece.minimum_rotated_rectangle
                item = _rectangle_to_obstacle(rectangle, origin_x, origin_y, height_m)
                if item is not None:
                    candidate_boxes.append(item)

            source_local = translate(polygon, xoff=-origin_x, yoff=-origin_y)
            gap_tolerance = max(1e-5, source_local.area * 1e-7)
            candidate_cover = (unary_union([obstacle_to_polygon(item) for item in candidate_boxes])
                               if candidate_boxes else None)
            candidate_gap = (source_local.difference(candidate_cover).area
                             if candidate_cover is not None else math.inf)
            candidate_ratio = (candidate_cover.area / source_local.area
                               if candidate_cover is not None and source_local.area > 0 else math.inf)

            # The previous greedy merge heuristic could exceed the requested
            # ratio on some real OSM footprints. Refine just those features by
            # using narrower oriented bands, keeping the ordinary case compact.
            if (candidate_cover is None or candidate_gap > gap_tolerance
                    or candidate_ratio > max_box_area_ratio + 1e-3):
                candidate_boxes = _adaptive_band_cover_boxes(
                    polygon, origin_x, origin_y, height_m, max_box_area_ratio
                ) or []
                candidate_cover = (unary_union([obstacle_to_polygon(item) for item in candidate_boxes])
                                   if candidate_boxes else None)
                candidate_gap = (source_local.difference(candidate_cover).area
                                 if candidate_cover is not None else math.inf)
                candidate_ratio = (candidate_cover.area / source_local.area
                                   if candidate_cover is not None and source_local.area > 0 else math.inf)

            if (not candidate_boxes or candidate_gap > gap_tolerance
                    or candidate_ratio > max_box_area_ratio + 1e-3):
                return None
            boxes.extend(candidate_boxes)

        if not boxes:
            return None

        # Final building-level check (including multi-polygons): fail closed if
        # rounding, overlaps between components, or decomposition leave a gap or
        # exceed the configured conservative coverage ratio.
        source_local = translate(geometry, xoff=-origin_x, yoff=-origin_y)
        covered = unary_union([obstacle_to_polygon(item) for item in boxes])
        if source_local.difference(covered).area > max(1e-5, source_local.area * 1e-7):
            return None
        if source_local.area <= 0 or covered.area / source_local.area > max_box_area_ratio + 1e-3:
            return None
        return boxes
    except (TypeError, ValueError, AttributeError, ImportError):
        return None


def parse_building_height_with_source(
    tags: Mapping[str, Any],
    default_height_m: float = DEFAULT_BUILDING_HEIGHT_M,
    floor_height_m: float = DEFAULT_FLOOR_HEIGHT_M,
) -> tuple[float, str]:
    """Resolve a building height and return its provenance for metadata."""
    raw_height = tags.get("height")
    if _has_value(raw_height):
        text = str(raw_height).strip().lower()
        match = re.search(r"[-+]?\d+(?:\.\d+)?", text)
        if match:
            height = float(match.group(0))
            if "ft" in text or "feet" in text or "foot" in text:
                height *= 0.3048
            if math.isfinite(height) and height > 0:
                return round(height, 2), "osm_height"

    raw_levels = tags.get("building:levels")
    if _has_value(raw_levels):
        match = re.search(r"\d+(?:\.\d+)?", str(raw_levels))
        if match:
            levels = float(match.group(0))
            height = levels * floor_height_m
            if math.isfinite(height) and height > 0:
                return round(height, 2), "building_levels_estimate"

    if not math.isfinite(default_height_m) or default_height_m <= 0:
        raise ValueError("default_height_m must be a positive finite number")
    return round(float(default_height_m), 2), "default_assumption"


def parse_building_height(
    tags: Mapping[str, Any],
    default_height_m: float = DEFAULT_BUILDING_HEIGHT_M,
    floor_height_m: float = DEFAULT_FLOOR_HEIGHT_M,
) -> float:
    """Resolve height from OSM ``height``, then ``building:levels``, else default."""
    return parse_building_height_with_source(tags, default_height_m, floor_height_m)[0]


def convert_buildings(
    projected_buildings,
    origin_x: float,
    origin_y: float,
    default_height_m: float = DEFAULT_BUILDING_HEIGHT_M,
    floor_height_m: float = DEFAULT_FLOOR_HEIGHT_M,
    min_side_m: float = DEFAULT_MIN_SIDE_M,
    max_box_area_ratio: float = 1.15,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Convert projected OSM footprints into yaw-aware simulation obstacle boxes."""
    from shapely.affinity import translate
    from shapely.ops import unary_union

    layout: list[dict[str, Any]] = []
    stats: dict[str, Any] = {
        "features_received": int(len(projected_buildings)), "buildings_exported": 0,
        "exported": 0, "skipped_geometry": 0, "skipped_too_small": 0,
        "height_source_counts": {"osm_height": 0, "building_levels_estimate": 0,
                               "default_assumption": 0},
        "building_height_sources": [], "building_coverage_ratios": [],
        "coverage_area_ratio_mean": None, "coverage_area_ratio_max": None,
        "buildings_over_area_ratio": 0,
    }
    area_ratios = []

    # Sort for stable output ordering between runs with the same OSM snapshot.
    for _, row in projected_buildings.sort_index().iterrows():
        geometry = row.geometry
        if geometry is None or geometry.is_empty:
            stats["skipped_geometry"] += 1
            continue
        try:
            if not geometry.is_valid:
                geometry = geometry.buffer(0)
            if geometry is None or geometry.is_empty or geometry.geom_type not in {"Polygon", "MultiPolygon"}:
                stats["skipped_geometry"] += 1
                continue
            height_m, height_source = parse_building_height_with_source(
                row, default_height_m=default_height_m, floor_height_m=floor_height_m
            )
            boxes = geometry_to_boxes(
                geometry, origin_x, origin_y, height_m,
                min_side_m=min_side_m, max_box_area_ratio=max_box_area_ratio,
            )
        except (TypeError, ValueError, AttributeError):
            stats["skipped_geometry"] += 1
            continue

        if not boxes:
            # Distinguish a genuinely undersized footprint from a geometry that
            # could not be safely represented within the requested area ratio.
            min_rect = geometry.minimum_rotated_rectangle
            coords = list(min_rect.exterior.coords)[:-1] if min_rect.geom_type == "Polygon" else []
            edges = [math.hypot(coords[(i + 1) % len(coords)][0] - coords[i][0],
                                coords[(i + 1) % len(coords)][1] - coords[i][1])
                     for i in range(len(coords))] if len(coords) >= 4 else []
            if edges and min(edges) < min_side_m:
                stats["skipped_too_small"] += 1
            else:
                stats["skipped_geometry"] += 1
            continue
        layout.extend(boxes)
        stats["buildings_exported"] += 1
        stats["height_source_counts"][height_source] += 1
        feature_id = row.name
        if isinstance(feature_id, tuple):
            feature_id = "/".join(str(part) for part in feature_id)
        stats["building_height_sources"].append({
            "feature_id": str(feature_id), "height_m": height_m, "source": height_source,
        })
        source_local = translate(geometry, xoff=-origin_x, yoff=-origin_y)
        covered = unary_union([obstacle_to_polygon(item) for item in boxes])
        ratio = float(covered.area / source_local.area) if source_local.area > 0 else math.inf
        area_ratios.append(ratio)
        stats["building_coverage_ratios"].append({
            "feature_id": str(feature_id),
            "area_ratio": round(ratio, 5),
            "obstacle_boxes": len(boxes),
            "footprint_area_m2": round(float(source_local.area), 3),
        })
        if ratio > max_box_area_ratio + 1e-3:
            stats["buildings_over_area_ratio"] += 1

    stats["exported"] = len(layout)
    if area_ratios:
        stats["coverage_area_ratio_mean"] = round(float(sum(area_ratios) / len(area_ratios)), 5)
        stats["coverage_area_ratio_max"] = round(float(max(area_ratios)), 5)
    return layout, stats


def _write_preview(projected_buildings, layout, origin_x, origin_y, output_path: Path, title: str):
    """Save a PNG preview of source footprints and their box approximations."""
    local_buildings = projected_buildings.copy()
    local_buildings.geometry = local_buildings.geometry.translate(
        xoff=-origin_x, yoff=-origin_y
    )

    fig, ax = plt.subplots(figsize=(10, 10))
    local_buildings.plot(ax=ax, facecolor="#dedede", edgecolor="#666666", linewidth=0.35)
    for obstacle in layout:
        cx, cy = obstacle["center"]
        hx, hy = obstacle["half"]
        ax.add_patch(Rectangle(
            (cx - hx, cy - hy), 2 * hx, 2 * hy,
            angle=math.degrees(float(obstacle.get("yaw", 0.0))),
            rotation_point="center", fill=False, edgecolor="#d62728", linewidth=0.65,
        ))
    ax.scatter([0], [0], marker="+", s=90, color="#1f77b4", label="Local origin / query centre")
    ax.set_title(f"{title}: OSM footprints and exported box approximations")
    ax.set_xlabel("Local X — East (m)")
    ax.set_ylabel("Local Y — North (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linewidth=0.25, alpha=0.4)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Import OSM building footprints as the UTM project's obstacle-layout YAML."
    )
    parser.add_argument("--center-lat", type=float, default=DEFAULT_CENTER_LAT,
                        help="query centre latitude (default: Electronic City Phase I, Bengaluru)")
    parser.add_argument("--center-lon", type=float, default=DEFAULT_CENTER_LON,
                        help="query centre longitude (default: Electronic City Phase I, Bengaluru)")
    parser.add_argument("--dist", type=float, default=DEFAULT_QUERY_DISTANCE_M,
                        help="OSM query distance in metres (default: 300)")
    parser.add_argument("--min-side-m", type=float, default=DEFAULT_MIN_SIDE_M,
                        help="skip buildings whose box width OR depth is below this (default: 2 m)")
    parser.add_argument("--max-box-area-ratio", type=float, default=1.15,
                        help="target max box-union area / footprint area ratio for decomposition (default: 1.15)")
    parser.add_argument("--default-height-m", type=float, default=DEFAULT_BUILDING_HEIGHT_M,
                        help="height used when OSM has no usable height/levels tag (default: 12 m)")
    parser.add_argument("--floor-height-m", type=float, default=DEFAULT_FLOOR_HEIGHT_M,
                        help="metres per building level when height is absent (default: 3 m)")
    parser.add_argument("--output-dir", type=Path, default=Path("data/osm_city"),
                        help="output directory (default: data/osm_city)")
    parser.add_argument("--name", default="bengaluru_ecity_phase1_rotated",
                        help="output file stem, letters/numbers/_/- only")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not -90 <= args.center_lat <= 90:
        parser.error("--center-lat must be between -90 and 90")
    if not -180 <= args.center_lon <= 180:
        parser.error("--center-lon must be between -180 and 180")
    if args.dist <= 0:
        parser.error("--dist must be greater than zero")
    if args.min_side_m < 0:
        parser.error("--min-side-m cannot be negative")
    if args.max_box_area_ratio < 1.0:
        parser.error("--max-box-area-ratio must be at least 1.0")
    if args.default_height_m <= 0 or args.floor_height_m <= 0:
        parser.error("height values must be greater than zero")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.name):
        parser.error("--name may contain only letters, numbers, underscores, and hyphens")

    try:
        projected, origin_x, origin_y, projected_crs = _download_buildings(
            args.center_lat, args.center_lon, args.dist
        )
        layout, stats = convert_buildings(
            projected, origin_x, origin_y,
            default_height_m=args.default_height_m,
            floor_height_m=args.floor_height_m,
            min_side_m=args.min_side_m,
            max_box_area_ratio=args.max_box_area_ratio,
        )
        if not layout:
            raise RuntimeError(
                "OSM returned buildings, but none survived geometry/size validation. "
                "Try a larger --dist or smaller --min-side-m."
            )

        args.output_dir.mkdir(parents=True, exist_ok=True)
        layout_path = args.output_dir / f"{args.name}_layout.yaml"
        preview_path = args.output_dir / f"{args.name}_preview.png"
        metadata_path = args.output_dir / f"{args.name}_metadata.yaml"

        # Keep the layout as a plain list so yaml.safe_load returns the exact
        # list expected by envs/scenario_manager.py and PyBulletAdapter.
        with layout_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(layout, handle, sort_keys=False, allow_unicode=True)

        _write_preview(projected, layout, origin_x, origin_y, preview_path, args.name)
        metadata = {
            "source": "OpenStreetMap building footprints via OSMnx",
            "attribution": "© OpenStreetMap contributors; data available under the Open Database License (ODbL).",
            "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
            "query_center_latitude": args.center_lat,
            "query_center_longitude": args.center_lon,
            "query_distance_m": args.dist,
            "projected_crs": projected_crs,
            "local_coordinate_frame": "X east, Y north, metres; query centre is (0, 0)",
            "footprint_approximation": "minimum rotated rectangles for near-rectangular footprints; concave shapes are decomposed and greedily merged into oriented boxes. Boxes may conservatively cover small areas outside footprints.",
            "box_orientation": "optional yaw in radians, counter-clockwise about +Z from the local east/X axis; omitted means zero yaw",
            "target_box_union_area_ratio": args.max_box_area_ratio,
            "default_height_m": args.default_height_m,
            "floor_height_m": args.floor_height_m,
            "minimum_box_side_m": args.min_side_m,
            "height_assignment": {
                "priority": ["OSM height tag (metres; ft/feet converted)", "building:levels multiplied by floor-height assumption", "default assumed height"],
                "floor_height_assumption_m": args.floor_height_m,
                "default_assumed_height_m": args.default_height_m,
                "source_counts": stats["height_source_counts"],
                "per_building_sources": stats["building_height_sources"],
                "note": "Records which source supplied each represented building's exported height; default_assumption means no usable OSM height/levels tag was available."
            },
            "stats": {k: v for k, v in stats.items() if k != "building_height_sources"},
            "layout_file": layout_path.name,
            "preview_file": preview_path.name,
        }
        with metadata_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(metadata, handle, sort_keys=False, allow_unicode=True)

    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print("Import complete.")
    print(f"Buildings returned: {stats['features_received']}")
    print(f"Buildings represented: {stats['buildings_exported']}")
    print(f"Obstacle boxes exported: {stats['exported']}")
    print(f"Skipped invalid/unsupported geometries: {stats['skipped_geometry']}")
    print(f"Skipped small footprints: {stats['skipped_too_small']}")
    print(f"Layout:   {layout_path}")
    print(f"Preview:  {preview_path}")
    print(f"Metadata: {metadata_path}")
    print(f"Height sources: {stats['height_source_counts']}")
    print(f"Mean/max footprint coverage ratio: {stats['coverage_area_ratio_mean']} / {stats['coverage_area_ratio_max']}")
    print("Note: inspect the preview and geometry coverage metadata before simulation use.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
