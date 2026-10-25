"""Crop an OSM-imported box layout into a local simulation region."""

from pathlib import Path
import argparse
import math
import re
import sys

import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]


def _polygon_components(geometry):
    if geometry is None or geometry.is_empty:
        return
    if geometry.geom_type == "Polygon":
        yield geometry
    elif geometry.geom_type in {"MultiPolygon", "GeometryCollection"}:
        for part in geometry.geoms:
            yield from _polygon_components(part)


def _axis_aligned_covering_boxes(polygon, crop_left, crop_bottom, crop_right, crop_top, center_x, center_y, height):
    """Cover a clipped boundary fragment with axis-aligned boxes kept in crop bounds.

    This is only used for a footprint that crosses the crop edge. Fully internal
    rotated boxes preserve their yaw. Restricting boundary fragments to the crop
    window avoids a rotated bounding rectangle protruding outside the crop.
    """
    from shapely.geometry import box as shapely_box
    result = []
    minx, miny, maxx, maxy = polygon.bounds
    y_levels = sorted(set([float(y) for _, y in polygon.exterior.coords] + [miny, maxy]))
    for low, high in zip(y_levels, y_levels[1:]):
        if high - low <= 1e-7:
            continue
        clipped_strip = polygon.intersection(shapely_box(crop_left, low, crop_right, high))
        for piece in _polygon_components(clipped_strip):
            if piece.area <= 1e-8:
                continue
            x0, y0, x1, y1 = piece.bounds
            hx, hy = (x1 - x0) / 2.0, (y1 - y0) / 2.0
            if hx <= 1e-5 or hy <= 1e-5:
                continue
            # Retain full floating precision for boundary fragments: rounding
            # the centre or padding the extents could push a box outside the crop.
            result.append({
                "type": "box",
                "center": [(x0 + x1) / 2 - center_x, (y0 + y1) / 2 - center_y],
                "half": [hx, hy],
                "height": round(float(height), 2),
            })
    return result


def crop_layout(layout, center_x, center_y, width, height):
    """Clip obstacles to a crop rectangle and re-centre it at local (0, 0).

    Oriented boxes fully inside the crop retain their yaw. A box crossing the
    crop boundary is converted through its true rotated footprint and clipped;
    only that boundary fragment is re-approximated with axis-aligned boxes so
    all exported boxes stay inside the selected crop window.
    """
    from shapely.geometry import box as shapely_box
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from scripts.import_osm_city import obstacle_to_polygon

    values = (center_x, center_y, width, height)
    if not all(math.isfinite(float(v)) for v in values):
        raise ValueError("Crop coordinates and dimensions must be finite.")
    if width <= 0 or height <= 0:
        raise ValueError("Crop width and height must be positive.")

    left, right = center_x - width / 2, center_x + width / 2
    bottom, top = center_y - height / 2, center_y + height / 2
    crop_polygon = shapely_box(left, bottom, right, top)
    result = []

    for item in layout:
        if item.get("type") != "box":
            raise ValueError(
                "This utility expects box obstacles from the OSM importer; "
                f"got {item.get('type')!r}."
            )
        # Validate fields first; obstacle_to_polygon validates dimensions and yaw.
        if "height" not in item or not math.isfinite(float(item["height"])) or float(item["height"]) <= 0:
            raise ValueError("Obstacle height must be positive and finite.")
        source_polygon = obstacle_to_polygon(item)
        clipped = source_polygon.intersection(crop_polygon)
        if clipped.is_empty or clipped.area <= 1e-8:
            continue

        # Preserve original oriented-box parameters when it lies entirely inside.
        if crop_polygon.covers(source_polygon):
            internal = dict(item)
            internal["center"] = [
                round(float(item["center"][0]) - center_x, 4),
                round(float(item["center"][1]) - center_y, 4),
            ]
            result.append(internal)
            continue

        # Crop boundary crossing: use a tight clipping geometry then axis-aligned
        # boxes. This may conservatively add area inside the crop but never extends
        # an obstacle beyond the crop bounds.
        result.extend(_axis_aligned_covering_boxes(
            clipped, left, bottom, right, top, center_x, center_y, item["height"]
        ))

    return result


def write_preview(layout, width, height, path, title):
    fig, ax = plt.subplots(figsize=(8, 8))
    for item in layout:
        x, y = item["center"]
        hx, hy = item["half"]
        ax.add_patch(Rectangle(
            (x - hx, y - hy), 2 * hx, 2 * hy,
            angle=math.degrees(float(item.get("yaw", 0.0))),
            rotation_point="center", facecolor="gray", edgecolor="black", linewidth=0.35,
        ))

    ax.set_xlim(-width / 2, width / 2)
    ax.set_ylim(-height / 2, height / 2)
    ax.set_aspect("equal")
    ax.set_xlabel("Local X (m)")
    ax.set_ylabel("Local Y (m)")
    ax.set_title(f"{title} — {len(layout)} obstacle boxes")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Crop an OSM obstacle layout for simulation or testing."
    )
    parser.add_argument(
        "--source",
        default="data/osm_city/bengaluru_ecity_phase1_rotated_v2_layout.yaml",
        help="Source layout path; relative paths are resolved from the project root.",
    )
    parser.add_argument("--center-x", type=float, default=-73.0)
    parser.add_argument("--center-y", type=float, default=230.0)
    parser.add_argument("--width", type=float, default=50.0)
    parser.add_argument("--height", type=float, default=50.0)
    parser.add_argument(
        "--name", default="bengaluru_ecity_phase1_rotated_crop50_cand1"
    )
    parser.add_argument("--output-dir", default="data/osm_city")
    args = parser.parse_args()

    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.name):
        parser.error("--name may contain only letters, numbers, underscores, and hyphens")

    source_path = Path(args.source)
    if not source_path.is_absolute():
        source_path = ROOT / source_path

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir

    if not source_path.is_file():
        raise FileNotFoundError(f"Source layout not found: {source_path}")

    layout = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    if not isinstance(layout, list) or not layout:
        raise ValueError("Source layout must be a non-empty YAML list.")

    cropped = crop_layout(
        layout, args.center_x, args.center_y, args.width, args.height
    )
    if not cropped:
        raise ValueError("Crop has no obstacles; try another centre or region size.")

    output_dir.mkdir(parents=True, exist_ok=True)
    layout_path = output_dir / f"{args.name}_layout.yaml"
    preview_path = output_dir / f"{args.name}_preview.png"
    metadata_path = output_dir / f"{args.name}_metadata.yaml"

    layout_path.write_text(
        yaml.safe_dump(cropped, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    write_preview(cropped, args.width, args.height, preview_path, args.name)

    source_display = (
        source_path.relative_to(ROOT).as_posix()
        if source_path.is_relative_to(ROOT)
        else source_path.as_posix()
    )
    metadata = {
        "source_layout": source_display,
        "crop_center_in_source_local_metres": [args.center_x, args.center_y],
        "crop_width_m": args.width,
        "crop_height_m": args.height,
        "output_coordinate_frame": "Crop centre is (0, 0); X east, Y north; metres",
        "source_obstacle_boxes": len(layout),
        "cropped_obstacle_boxes": len(cropped),
        "method": "Oriented footprints clipped to the crop rectangle; fully internal boxes preserve yaw; boundary fragments are conservatively re-boxed axis-aligned inside the crop",
    }
    metadata_path.write_text(
        yaml.safe_dump(metadata, sort_keys=False),
        encoding="utf-8",
    )

    print(f"Source obstacle boxes: {len(layout)}")
    print(f"Cropped obstacle boxes: {len(cropped)}")
    print(f"Layout:   {layout_path}")
    print(f"Preview:  {preview_path}")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
