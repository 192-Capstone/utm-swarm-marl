"""
airspace_zones.py
-----------------
D2 part 1: airspace structure for the UTM layer. No-fly zones and vertiports, a GeoJSON file
format, and the link to the E1 strategic planner (utm/strategic_planner.py), without editing E1.

    sky = map_airspace('ecity_crop50')                    # E1's Airspace for that map
    zones, ports, meta = load_zones('data/airspace/ecity_crop50_example.geojson', sky)
    planner = StrategicPlanner(layout, bounds, airspace=sky)
    index = apply_zones(planner, zones)                   # before any flight is submitted
    d = planner.submit(FlightPlan(...))

Zones follow the ideas of EUROCAE ED-269 (UAS geographical zones) and India's Digital Sky
red / yellow zones:
    restriction  'prohibited'    (red: nobody may fly there)
                 'authorisation' (yellow: only flights in `authorised`; see TODO in apply_zones)
    lower_m / upper_m            height band above ground (None = from the surface / no ceiling)
    active                       [[t0, t1], ...] sim-second windows; empty = permanent
    geometry                     polygon (local metres) or circle (centre + radius_m)

Zone -> E1 keys: zone_keys() gives the (H3 cell, layer) keys of a zone. Cells: conservative:
every cell that overlaps the zone, plus possibly a few that only come within a cell's corner radius
of it (works for any simple polygon, convex or not). Layers: those whose height band overlaps the
zone's band.

Zone -> E1: permanent zones go into planner.blocked, temporary zones are booked in the planner's
ledger as flight 'zone:<id>' over their windows, so E1 delays a flight until the zone ends exactly
as it would for another flight. Routes do not avoid zones yet (part 2, routing masks after D7).

Rules and limits:
  1. Order: zones must be applied before any flight is submitted (apply_zones raises otherwise;
     E1 is first come, first served). A zone that appears mid-flight is not handled (E4,
     contingency, later).
  2. Safety buffer (intended): a zone's key cells reach up to one cell corner radius (~0.6 m) past
     its edge, and on top of that E1 reserves every cell whose centre is within radius + one cell
     edge (2.1 + 0.58 m) of the drone. Together, drones stay about 2.2-3.3 m from a zone boundary:
     always blocked closer than ~2.2 m, never blocked beyond ~3.3 m; in between it depends on how
     the hexagons line up with the edge. This holds for E1's defaults (radius 2.1 m, H3 resolution
     15) and changes if those change. Measured on the example zones in data/airspace/: points 0-5 m
     outside each zone, blocked when planner.cells_near(point) shares a cell with the zone's keys
     (tests: test_safety_buffer_around_example_zones).
  3. Thin zones over-block: one row of ~1 m hexagons is wider than a thin strip, so a 0.5 m x 10 m
     strip blocks about 3.8x its area (3.3-4.1x depending on position and angle). Fine at our
     scale; noted for the report (tests: test_thin_strip_over_blocking_matches_docstring).
  4. Yellow ('authorisation') zones are treated as prohibited by apply_zones for now. The E1 change
     that lets the flights in zone.authorised through is coming in Aman's E1 PR.

GeoJSON (RFC 7946) FeatureCollection, with two extra top-level members:
    "frame": "wgs84" (default; coordinates [lng, lat]) or "local_m" (coordinates [x, y] metres)
    "anchor": [lat, lng] of local (0, 0); required for "local_m", must match the Airspace used
Features: properties.type = "zone" (Polygon, or Point + properties.radius_m for a circle) or
"vertiport" (Point). Polygons: one closed outer ring, no holes, not self-intersecting.
"""

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from envs.airspace import ANCHOR_ECITY, Airspace

RESTRICTIONS = ('prohibited', 'authorisation')
ZONE_PREFIX = 'zone:'
REPO = Path(__file__).resolve().parents[1]
CROP50_METADATA = REPO / 'data/osm_city/bengaluru_ecity_phase1_rotated_v2_crop50_cand1_metadata.yaml'
CIRCLE_SEGMENTS = 32
ANCHOR_TOL_DEG = 1e-7                      # ~1 cm


# ------------------------------------------------------------------ geometry helpers
def _area(poly):
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _orient(a, b, c):
    v = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    return 0 if abs(v) < 1e-12 else (1 if v > 0 else -1)


def _on_segment(a, b, p):
    return (min(a[0], b[0]) - 1e-12 <= p[0] <= max(a[0], b[0]) + 1e-12
            and min(a[1], b[1]) - 1e-12 <= p[1] <= max(a[1], b[1]) + 1e-12)


def _segments_meet(a, b, c, d):
    """True when segments ab and cd cross or touch."""
    o1, o2, o3, o4 = _orient(a, b, c), _orient(a, b, d), _orient(c, d, a), _orient(c, d, b)
    if o1 != o2 and o3 != o4:
        return True
    return ((o1 == 0 and _on_segment(a, b, c)) or (o2 == 0 and _on_segment(a, b, d))
            or (o3 == 0 and _on_segment(c, d, a)) or (o4 == 0 and _on_segment(c, d, b)))


def self_intersections(poly):
    """Pairs (i, j) of non-adjacent edges that cross or touch (edge i runs corner i -> i + 1)."""
    p = [tuple(map(float, q)) for q in poly]
    n, out = len(p), []
    for i in range(n):
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:                    # adjacent through the closing corner
                continue
            if _segments_meet(p[i], p[(i + 1) % n], p[j], p[(j + 1) % n]):
                out.append((i, j))
    return out


def convex_hull(pts):
    """Convex hull (counter-clockwise, Andrew's monotone chain) of an (N, 2) array."""
    p = sorted(set(map(tuple, np.asarray(pts, dtype=float).tolist())))
    if len(p) < 3:
        return np.asarray(p)

    def half(points):
        h = []
        for q in points:
            while len(h) >= 2 and _orient(h[-2], h[-1], q) <= 0:
                h.pop()
            h.append(q)
        return h
    lower, upper = half(p), half(p[::-1])
    return np.asarray(lower[:-1] + upper[:-1])


def _inside(poly, pts):
    """Even-odd point-in-polygon for (M, 2) points."""
    x, y = pts[:, 0:1], pts[:, 1:2]
    xi, yi = poly[:, 0][None, :], poly[:, 1][None, :]
    xj, yj = np.roll(poly[:, 0], 1)[None, :], np.roll(poly[:, 1], 1)[None, :]
    with np.errstate(divide='ignore', invalid='ignore'):
        cross = ((yi > y) != (yj > y)) & (x < (xj - xi) * (y - yi) / (yj - yi) + xi)
    return np.count_nonzero(cross, axis=1) % 2 == 1


def _boundary_distance(poly, pts):
    """Distance from each of (M, 2) points to the polygon's edges."""
    a = poly[None, :, :]
    b = np.roll(poly, -1, axis=0)[None, :, :]
    p = pts[:, None, :]
    ab = b - a
    t = np.clip(np.sum((p - a) * ab, axis=2) / np.maximum(np.sum(ab * ab, axis=2), 1e-18), 0.0, 1.0)
    return np.min(np.linalg.norm(p - (a + t[..., None] * ab), axis=2), axis=1)


def circle_polygon(centre, radius, segments=CIRCLE_SEGMENTS):
    """A polygon around a circle (circumscribed, so the whole circle is inside)."""
    a = np.linspace(0.0, 2.0 * math.pi, segments, endpoint=False)
    r = radius / math.cos(math.pi / segments)
    return np.column_stack([centre[0] + r * np.cos(a), centre[1] + r * np.sin(a)])


# ------------------------------------------------------------------ data
@dataclass
class Zone:
    id: str
    name: str
    restriction: str = 'prohibited'          # 'prohibited' (red) | 'authorisation' (yellow)
    reason: str = ''
    lower_m: float = None                    # None = from the surface
    upper_m: float = None                    # None = no ceiling
    active: list = field(default_factory=list)       # [[t0, t1], ...] s; empty = permanent
    authorised: list = field(default_factory=list)   # flight ids allowed into an 'authorisation' zone
    polygon: list = None                     # [(x, y), ...] local metres, not closed
    circle: tuple = None                     # ((x, y), radius_m)

    def __post_init__(self):
        self.validate()

    def _fail(self, msg):
        raise ValueError(f'zone {self.id!r}: {msg}')

    def validate(self):
        if not isinstance(self.id, str) or not self.id:
            raise ValueError(f'zone id must be a non-empty string, got {self.id!r}')
        if self.restriction not in RESTRICTIONS:
            self._fail(f'unknown restriction {self.restriction!r} (use one of {RESTRICTIONS})')
        for name in ('lower_m', 'upper_m'):
            v = getattr(self, name)
            if v is not None:
                if not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0:
                    self._fail(f'{name} must be a finite height >= 0 m or null, got {v!r}')
                setattr(self, name, float(v))
        if self.lower_m is not None and self.upper_m is not None and self.lower_m >= self.upper_m:
            self._fail(f'lower_m ({self.lower_m}) must be below upper_m ({self.upper_m})')
        windows = []
        for w in self.active:
            try:
                t0, t1 = (float(v) for v in w)
            except (TypeError, ValueError):
                self._fail(f'active window {w!r} must be [t0, t1]')
            if not (math.isfinite(t0) and math.isfinite(t1)) or t0 >= t1:
                self._fail(f'active window {w!r} must have finite t0 < t1')
            windows.append([t0, t1])
        self.active = windows
        self.authorised = [str(a) for a in self.authorised]
        if (self.polygon is None) == (self.circle is None):
            self._fail('needs exactly one geometry: polygon or circle')
        if self.circle is not None:
            (cx, cy), r = self.circle
            if not (math.isfinite(cx) and math.isfinite(cy) and math.isfinite(r)) or r <= 0:
                self._fail(f'circle needs a finite centre and radius_m > 0, got {self.circle!r}')
            self.circle = ((float(cx), float(cy)), float(r))
        else:
            pts = [tuple(map(float, p)) for p in self.polygon]
            if len(pts) >= 2 and pts[0] == pts[-1]:
                pts = pts[:-1]                           # accept a closed ring
            if len(pts) < 3:
                self._fail('polygon needs at least 3 distinct corners')
            if not all(math.isfinite(v) for p in pts for v in p):
                self._fail('polygon corners must be finite')
            if any(pts[i] == pts[(i + 1) % len(pts)] for i in range(len(pts))):
                self._fail('polygon repeats a corner')
            bad = self_intersections(pts)
            if bad:
                i, j = bad[0]
                self._fail(f'polygon is self-intersecting (edges {i} and {j} cross)')
            if abs(_area(np.asarray(pts))) < 1e-9:
                self._fail('polygon has zero area')
            self.polygon = pts

    @property
    def permanent(self):
        return not self.active

    def outline(self):
        """(N, 2) corners: the polygon, or a polygon around the circle."""
        if self.circle is not None:
            return circle_polygon(*self.circle)
        return np.asarray(self.polygon, dtype=float)

    def area(self):
        if self.circle is not None:
            return math.pi * self.circle[1] ** 2
        return abs(_area(np.asarray(self.polygon)))

    def distance(self, pts):
        """Distance (m) from each of (M, 2) local points to the zone, 0 inside."""
        pts = np.atleast_2d(np.asarray(pts, dtype=float))
        if self.circle is not None:
            (cx, cy), r = self.circle
            return np.maximum(np.hypot(pts[:, 0] - cx, pts[:, 1] - cy) - r, 0.0)
        poly = np.asarray(self.polygon)
        d = _boundary_distance(poly, pts)
        d[_inside(poly, pts)] = 0.0
        return d

    def layers(self, airspace, n_layers):
        """Altitude layers (0 .. n_layers - 1) whose height band overlaps the zone's band."""
        lo = -math.inf if self.lower_m is None else self.lower_m
        hi = math.inf if self.upper_m is None else self.upper_m
        out = []
        for k in range(n_layers):
            b, t = airspace.layer_range(k)
            if lo < t and b < hi:
                out.append(k)
        return out


@dataclass
class Vertiport:
    id: str
    x: float
    y: float
    kind: str = 'vertiport'                  # 'vertiport' | 'hub' | 'drop' | ...
    name: str = ''

    def pad(self):
        """(x, y, kind), the plain pad format for D4."""
        return (float(self.x), float(self.y), self.kind)


# ------------------------------------------------------------------ airspace for a map
def crop_anchor(metadata_path=CROP50_METADATA, source_anchor=ANCHOR_ECITY):
    """(lat, lng) of a cropped OSM map's (0, 0): the crop centre, given in the source map's
    local metres in its metadata file, converted through the source map's anchor."""
    meta = yaml.safe_load(Path(metadata_path).read_text(encoding='utf-8'))
    x, y = meta['crop_center_in_source_local_metres']
    lat, lng = Airspace(anchor=source_anchor).to_latlng(float(x), float(y))
    return float(lat), float(lng)


def planner_airspace(anchor=ANCHOR_ECITY, settings=None):
    """The Airspace E1 uses (its resolution and layers) at a given anchor; pass it to
    StrategicPlanner(..., airspace=...) and load_zones()."""
    if settings is None:
        from utm.strategic_planner import Settings
        settings = Settings()
    return Airspace(anchor=anchor, resolution=settings.resolution, layer_floor=settings.layer_floor,
                    layer_height=settings.layer_height)


def map_airspace(name, settings=None):
    """E1's Airspace for a known map: 'ecity' (full D3 map), 'ecity_crop50' (the 50 m crop, whose
    (0, 0) is 73 m west / 230 m north of the full map's), 'd1' (procedural cities, placed at the
    E-City anchor)."""
    anchors = {'ecity': lambda: ANCHOR_ECITY, 'd1': lambda: ANCHOR_ECITY, 'ecity_crop50': crop_anchor}
    if name not in anchors:
        raise ValueError(f'unknown map {name!r} (known: {sorted(anchors)})')
    return planner_airspace(anchors[name](), settings)


# ------------------------------------------------------------------ zones -> keys
def zone_cells(zone, airspace):
    """H3 cells (airspace resolution) for a zone, conservative: every cell that overlaps the zone,
    plus possibly a few that only come within a cell's corner radius of it. Rule: keep a cell when
    its centre is within its corner radius of the zone. Candidates come from B5's
    cells_in_polygon(cover=True) on the zone's convex hull (exact for convex shapes) plus one ring,
    so no overlapping cell is missed."""
    hull = convex_hull(zone.outline())
    cand = airspace.cells_in_polygon(hull, cover=True)
    cand.add(airspace.cell(*np.mean(hull, axis=0)))
    cand = set().union(*(airspace.disk(c, 1) for c in cand))
    cells = sorted(cand)
    centres = np.array([airspace.center(c) for c in cells])
    reach = np.array([max(math.hypot(px - cx, py - cy) for px, py in airspace.boundary(c))
                      for c, (cx, cy) in zip(cells, centres)])
    keep = zone.distance(centres) <= reach + 1e-9
    return {c for c, k in zip(cells, keep) if k}


def zone_keys(zone, airspace, n_layers):
    """The (H3 cell, layer) keys a zone covers, in E1's key format."""
    layers = zone.layers(airspace, n_layers)
    if not layers:
        return set()
    return {(c, k) for c in zone_cells(zone, airspace) for k in layers}


# ------------------------------------------------------------------ zones -> E1
def apply_zones(planner, zones):
    """Put zones into an E1 StrategicPlanner (call before any flight is submitted: E1 is first
    come, first served). Permanent zones -> planner.blocked; temporary zones -> booked in the
    ledger as 'zone:<id>' over their active windows. Returns {key: [zone ids]} for zones_hit().

    TODO(E1): 'authorisation' zones are treated as 'prohibited' for now. Letting the flights in
    zone.authorised through needs E1 to skip a zone's keys for those flight ids."""
    if planner.decisions:
        raise ValueError(f'apply_zones must run before any flight is submitted '
                         f'({len(planner.decisions)} already decided)')
    ids = [z.id for z in zones]
    if len(set(ids)) != len(ids):
        raise ValueError('zone ids must be unique')
    index = {}
    for z in zones:
        keys = zone_keys(z, planner.sky, planner.s.n_layers)
        for key in keys:
            index.setdefault(key, []).append(z.id)
        if z.permanent:
            planner.blocked |= keys
        else:
            windows = [tuple(w) for w in z.active]
            planner._book(ZONE_PREFIX + z.id, {key: windows for key in keys})
    return index


def zone_names(conflict_ids, zones):
    """E1 conflict ids -> readable zone names: 'zone:<id>' -> that zone's name, 'blocked' ->
    'permanent no-fly zone' (E1 does not say which; use zones_hit for that). Flight ids are left out."""
    by_id = {z.id: z for z in zones}
    out = set()
    for c in conflict_ids:
        if c == 'blocked':
            out.add('permanent no-fly zone')
        elif isinstance(c, str) and c.startswith(ZONE_PREFIX):
            z = by_id.get(c[len(ZONE_PREFIX):])
            out.add(z.name if z else c)
    return sorted(out)


def zones_hit(reservation, index, zones=None):
    """Zones (names if `zones` is given, else ids) that share a key with a reservation; ignores
    time, so it also names permanent zones."""
    hit = {zid for key in reservation for zid in index.get(key, ())}
    if zones is None:
        return sorted(hit)
    names = {z.id: z.name for z in zones}
    return sorted(names.get(h, h) for h in hit)


# ------------------------------------------------------------------ GeoJSON
def _bad(where, msg):
    raise ValueError(f'{where}: {msg}')


def load_zones(path, airspace):
    """Read a zone file. Returns (zones, vertiports, meta); all geometry in the airspace's local
    metres. Raises ValueError with the feature and the problem for anything invalid."""
    path = Path(path)
    try:
        doc = json.loads(path.read_text(encoding='utf-8'))
    except json.JSONDecodeError as exc:
        raise ValueError(f'{path}: not valid JSON ({exc})') from exc
    return parse_zones(doc, airspace, where=str(path))


def parse_zones(doc, airspace, where='zone file'):
    """Like load_zones, for an already-parsed GeoJSON dict."""
    if not isinstance(doc, dict) or doc.get('type') != 'FeatureCollection':
        _bad(where, 'must be a GeoJSON FeatureCollection')
    frame = doc.get('frame', 'wgs84')
    if frame not in ('wgs84', 'local_m'):
        _bad(where, f"unknown frame {frame!r} (use 'wgs84' or 'local_m')")
    if frame == 'local_m':
        anchor = doc.get('anchor')
        if not (isinstance(anchor, list) and len(anchor) == 2):
            _bad(where, "a 'local_m' file needs \"anchor\": [lat, lng]")
        if abs(anchor[0] - airspace.lat0) > ANCHOR_TOL_DEG or abs(anchor[1] - airspace.lng0) > ANCHOR_TOL_DEG:
            _bad(where, f'file anchor {anchor} differs from the airspace anchor '
                        f'[{airspace.lat0}, {airspace.lng0}]; use the Airspace for this map (map_airspace)')

    def local(coord, label):
        if not (isinstance(coord, (list, tuple)) and len(coord) >= 2
                and all(isinstance(v, (int, float)) and math.isfinite(v) for v in coord[:2])):
            _bad(label, f'bad coordinate {coord!r}')
        if frame == 'local_m':
            return float(coord[0]), float(coord[1])
        x, y = airspace.to_local(float(coord[1]), float(coord[0]))      # GeoJSON order: [lng, lat]
        return float(x), float(y)

    zones, ports, seen = [], [], set()
    features = doc.get('features')
    if not isinstance(features, list):
        _bad(where, "'features' must be a list")
    for i, f in enumerate(features):
        if not isinstance(f, dict) or f.get('type') != 'Feature':
            _bad(f'{where}, feature {i}', 'must be a GeoJSON Feature')
        props = f.get('properties') or {}
        fid = f.get('id', props.get('id'))
        label = f'{where}, feature {i}' + (f' ({fid!r})' if fid is not None else '')
        if not isinstance(fid, str) or not fid:
            _bad(label, 'needs a string id')
        if fid in seen:
            _bad(label, f'duplicate id {fid!r}')
        seen.add(fid)
        geom = f.get('geometry') or {}
        gtype = geom.get('type')
        kind = props.get('type')
        if kind == 'vertiport':
            if gtype != 'Point':
                _bad(label, 'a vertiport must be a Point')
            x, y = local(geom.get('coordinates'), label)
            ports.append(Vertiport(fid, x, y, str(props.get('kind', 'vertiport')), str(props.get('name', ''))))
            continue
        if kind != 'zone':
            _bad(label, f"properties.type must be 'zone' or 'vertiport', got {kind!r}")
        polygon = circle = None
        if gtype == 'Polygon':
            rings = geom.get('coordinates')
            if not isinstance(rings, list) or not rings:
                _bad(label, 'Polygon has no rings')
            if len(rings) > 1:
                _bad(label, 'Polygon holes are not supported (one outer ring only)')
            ring = rings[0]
            if not isinstance(ring, list) or len(ring) < 4:
                _bad(label, 'Polygon ring needs at least 4 positions (3 corners + closing)')
            if list(ring[0][:2]) != list(ring[-1][:2]):
                _bad(label, 'Polygon ring is not closed (first and last positions differ)')
            polygon = [local(c, label) for c in ring[:-1]]
        elif gtype == 'Point':
            if 'radius_m' not in props:
                _bad(label, 'a Point zone needs properties.radius_m (circle)')
            r = props['radius_m']
            if not isinstance(r, (int, float)):
                _bad(label, f'radius_m must be a number, got {r!r}')
            circle = (local(geom.get('coordinates'), label), float(r))
        else:
            _bad(label, f'zone geometry must be Polygon or Point (circle), got {gtype!r}')
        try:
            zones.append(Zone(fid, str(props.get('name', fid)), props.get('restriction', 'prohibited'),
                              str(props.get('reason', '')), props.get('lower_m'), props.get('upper_m'),
                              list(props.get('active') or []), list(props.get('authorised') or []),
                              polygon, circle))
        except ValueError as exc:
            _bad(label, str(exc))
    meta = {k: v for k, v in doc.items() if k not in ('type', 'features')}
    return zones, ports, meta


def to_geojson(zones, vertiports=(), airspace=None, frame='wgs84', meta=None):
    """GeoJSON dict for zones and vertiports (local metres in, `frame` out)."""
    if frame not in ('wgs84', 'local_m'):
        raise ValueError(f"unknown frame {frame!r} (use 'wgs84' or 'local_m')")
    if airspace is None:
        raise ValueError('to_geojson needs the airspace (for the anchor)')

    def out(x, y):
        if frame == 'local_m':
            return [round(float(x), 6), round(float(y), 6)]
        lat, lng = airspace.to_latlng(float(x), float(y))
        return [round(float(lng), 12), round(float(lat), 12)]          # ~0.1 micrometre

    feats = []
    for z in zones:
        props = {'type': 'zone', 'name': z.name, 'restriction': z.restriction, 'reason': z.reason,
                 'lower_m': z.lower_m, 'upper_m': z.upper_m, 'active': z.active, 'authorised': z.authorised}
        if z.circle is not None:
            geom = {'type': 'Point', 'coordinates': out(*z.circle[0])}
            props['radius_m'] = z.circle[1]
        else:
            ring = [out(x, y) for x, y in z.polygon]
            geom = {'type': 'Polygon', 'coordinates': [ring + [ring[0]]]}
        feats.append({'type': 'Feature', 'id': z.id, 'geometry': geom, 'properties': props})
    for v in vertiports:
        feats.append({'type': 'Feature', 'id': v.id, 'geometry': {'type': 'Point', 'coordinates': out(v.x, v.y)},
                      'properties': {'type': 'vertiport', 'kind': v.kind, 'name': v.name}})
    doc = {'type': 'FeatureCollection', 'frame': frame}
    if frame == 'local_m':
        doc['anchor'] = [airspace.lat0, airspace.lng0]
    doc.update(meta or {})
    doc['features'] = feats
    return doc


def save_zones(path, zones, vertiports=(), airspace=None, frame='wgs84', meta=None):
    """Write zones and vertiports to a GeoJSON file."""
    doc = to_geojson(zones, vertiports, airspace, frame, meta)
    Path(path).write_text(json.dumps(doc, indent=1) + '\n', encoding='utf-8')
    return doc


# ------------------------------------------------------------------ generator
def city_blocks(city):
    """Block rectangles (x0, x1, y0, y1) of a D1 city (envs/city_generator.generate_city)."""
    return [(x0, x1, y0, y1) for x0, x1 in city['blocks_x'] for y0, y1 in city['blocks_y']]


def tile_blocks(bounds, size):
    """Square tiles (x0, x1, y0, y1) over ((xmin, xmax), (ymin, ymax)), for maps without a
    block structure (e.g. the E-City crop). Edge tiles are clipped to the bounds."""
    (xmin, xmax), (ymin, ymax) = bounds
    xs = np.append(np.arange(xmin, xmax, size), xmax)
    ys = np.append(np.arange(ymin, ymax, size), ymax)
    return [(float(xs[i]), float(xs[i + 1]), float(ys[j]), float(ys[j + 1]))
            for i in range(len(xs) - 1) for j in range(len(ys) - 1) if xs[i + 1] - xs[i] > 1e-9
            and ys[j + 1] - ys[j] > 1e-9]


def random_block_zones(blocks, map_area, target, rng, restriction='prohibited', prefix='gen',
                       lower_m=None, upper_m=None):
    """Permanent zones over whole blocks covering about `target` (0-1) of map_area. Blocks are
    tried in a random order and kept when they bring the covered fraction closer to the target, so
    the result is within half a block of it. Same rng state -> same zones."""
    if not 0.0 <= target <= 1.0:
        raise ValueError(f'target must be a fraction in [0, 1], got {target}')
    order = rng.permutation(len(blocks))
    zones, covered = [], 0.0
    for i in order:
        x0, x1, y0, y1 = blocks[int(i)]
        a = (x1 - x0) * (y1 - y0) / map_area
        if abs(covered + a - target) < abs(covered - target):
            covered += a
            zones.append(Zone(f'{prefix}_{len(zones)}', f'generated block {len(zones)}', restriction,
                              'generated for experiments', lower_m, upper_m,
                              polygon=[(x0, y0), (x1, y0), (x1, y1), (x0, y1)]))
    return zones


def coverage(zones, map_area):
    """Fraction of the map area covered by zones (assumes they do not overlap, as generated)."""
    return sum(z.area() for z in zones) / map_area
