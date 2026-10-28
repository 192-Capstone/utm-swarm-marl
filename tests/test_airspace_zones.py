"""
test_airspace_zones.py
----------------------
D2 part 1: no-fly zones and vertiports (utm/airspace_zones.py), and their link to E1.

  A. GeoJSON: round trip, lat/lng and local metres give the same keys, bad files raise clear
     errors, the example files load (the crop file into E1's airspace for the crop).
  B. Keys: every point of a zone (convex, L-shape, circle) lies in one of its cells, and no cell is
     further than one cell from the zone; height bands pick the right layers.
  C. With E1 on an empty map: start pad in a permanent zone -> rejected; temporary zone -> delayed
     past it (rejected beyond max_delay); away from zones -> unaffected; conflict ids -> names;
     zones must be applied before any flight.
  D. Coverage generator: hits its target, deterministic.

Run: python -m pytest tests/test_airspace_zones.py -q
"""

import copy
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

pytest.importorskip('h3')

from envs.osm_scenario import load_osm_layout  # noqa: E402
from utm.airspace_zones import (Vertiport, Zone, apply_zones, city_blocks, coverage,  # noqa: E402
                                load_zones, map_airspace, parse_zones, planner_airspace,
                                random_block_zones, save_zones, tile_blocks, to_geojson,
                                zone_cells, zone_keys, zone_names, zones_hit)
from utm.strategic_planner import FlightPlan, Settings, StrategicPlanner  # noqa: E402

REPO = os.path.join(os.path.dirname(__file__), '..')
ECITY = os.path.join(REPO, 'data/airspace/ecity_crop50_example.geojson')
D1 = os.path.join(REPO, 'data/airspace/d1_medium_seed3_example.geojson')
CROP_LAYOUT = os.path.join(REPO, 'data/osm_city/bengaluru_ecity_phase1_rotated_v2_crop50_cand1_layout.yaml')
B = ((-25.0, 25.0), (-25.0, 25.0))


def square(cx, cy, h):
    return [(cx - h, cy - h), (cx + h, cy - h), (cx + h, cy + h), (cx - h, cy + h)]


SQUARE = Zone('sq', 'Square', polygon=square(3.3, -2.1, 2.7))
L_SHAPE = Zone('ell', 'L-shape', polygon=[(0, 0), (6, 0), (6, 2), (2, 2), (2, 7), (0, 7)])
CIRCLE = Zone('circ', 'Circle', circle=((-4.2, 5.5), 2.3))
SKY = planner_airspace()


def example_zones():
    return [SQUARE, L_SHAPE, CIRCLE,
            Zone('band', 'Band', 'authorisation', 'test', 3.0, 10.0, [[5.0, 9.0]], ['f1'],
                 polygon=square(-10, -10, 1.5))]


# ------------------------------------------------------------------ A. GeoJSON
@pytest.mark.parametrize('frame', ['wgs84', 'local_m'])
def test_round_trip(tmp_path, frame):
    zones = example_zones()
    ports = [Vertiport('vp', 1.5, -2.5, 'hub', 'Hub')]
    path = tmp_path / f'z_{frame}.geojson'
    save_zones(path, zones, ports, SKY, frame=frame, meta={'description': 'test'})
    z2, p2, meta = load_zones(path, SKY)
    assert meta['frame'] == frame and meta['description'] == 'test'
    (x, y, kind), = [p.pad() for p in p2]
    assert kind == 'hub' and p2[0].name == 'Hub' and abs(x - 1.5) < 1e-6 and abs(y + 2.5) < 1e-6
    for a, b in zip(zones, z2):
        assert (a.id, a.name, a.restriction, a.reason, a.lower_m, a.upper_m, a.active, a.authorised) == \
               (b.id, b.name, b.restriction, b.reason, b.lower_m, b.upper_m, b.active, b.authorised)
        assert np.allclose(a.outline(), b.outline(), atol=1e-6)


def test_latlng_and_local_files_give_identical_keys(tmp_path):
    zones = example_zones()
    save_zones(tmp_path / 'g.geojson', zones, (), SKY, frame='wgs84')
    save_zones(tmp_path / 'l.geojson', zones, (), SKY, frame='local_m')
    zg, _, _ = load_zones(tmp_path / 'g.geojson', SKY)
    zl, _, _ = load_zones(tmp_path / 'l.geojson', SKY)
    for a, b in zip(zg, zl):
        assert zone_keys(a, SKY, 2) == zone_keys(b, SKY, 2), a.id


def _doc():
    return to_geojson([SQUARE, CIRCLE], [Vertiport('vp', 0, 0)], SKY, frame='local_m')


def _poly(doc):
    return doc['features'][0]['geometry']['coordinates'][0]


def _props(doc):
    return doc['features'][0]['properties']


BAD = {
    'unclosed ring': (lambda d: _poly(d).pop(), 'not closed'),
    'hole': (lambda d: d['features'][0]['geometry']['coordinates'].append(_poly(d)), 'holes'),
    'unknown restriction': (lambda d: _props(d).update(restriction='orange'), 'unknown restriction'),
    'lower above upper': (lambda d: _props(d).update(lower_m=10, upper_m=5), 'must be below upper_m'),
    'negative height': (lambda d: _props(d).update(lower_m=-1), 'lower_m'),
    'bad window': (lambda d: _props(d).update(active=[[300, 120]]), 't0 < t1'),
    'window not a pair': (lambda d: _props(d).update(active=[[1, 2, 3]]), '[t0, t1]'),
    'zero radius': (lambda d: d['features'][1]['properties'].update(radius_m=0), 'radius_m > 0'),
    'circle without radius': (lambda d: d['features'][1]['properties'].pop('radius_m'), 'radius_m'),
    'unknown feature type': (lambda d: _props(d).update(type='corridor'), "'zone' or 'vertiport'"),
    'duplicate id': (lambda d: d['features'][1].update(id='sq'), 'duplicate id'),
    'unknown frame': (lambda d: d.update(frame='utm'), 'unknown frame'),
    'wrong anchor': (lambda d: d.update(anchor=[12.0, 77.0]), 'differs from the airspace anchor'),
    'self-intersecting': (lambda d: d['features'][0]['geometry'].update(
        coordinates=[[[0, 0], [4, 4], [4, 0], [0, 4], [0, 0]]]), 'self-intersecting'),
    'too few corners': (lambda d: d['features'][0]['geometry'].update(
        coordinates=[[[0, 0], [4, 4], [0, 0]]]), 'at least 4 positions'),
}


@pytest.mark.parametrize('case', sorted(BAD))
def test_bad_files_raise_clear_errors(case):
    edit, message = BAD[case]
    doc = copy.deepcopy(_doc())
    edit(doc)
    with pytest.raises(ValueError, match=message.replace('[', r'\[').replace(']', r'\]')):
        parse_zones(doc, SKY)


def test_bad_json_and_self_intersection_in_code(tmp_path):
    p = tmp_path / 'x.geojson'
    p.write_text('{not json', encoding='utf-8')
    with pytest.raises(ValueError, match='not valid JSON'):
        load_zones(p, SKY)
    with pytest.raises(ValueError, match='self-intersecting'):        # bow-tie
        Zone('bow', 'Bow', polygon=[(0, 0), (4, 4), (4, 0), (0, 4)])
    with pytest.raises(ValueError, match='self-intersecting'):        # edges touching at a corner
        Zone('pinch', 'Pinch', polygon=[(0, 0), (4, 0), (2, 2), (4, 4), (0, 4), (2, 2)])
    Zone('ok', 'Concave ok', polygon=L_SHAPE.polygon)                 # non-convex but simple is fine


def test_example_files_load():
    zones, ports, meta = load_zones(D1, map_airspace('d1'))
    assert {z.restriction for z in zones} == {'prohibited', 'authorisation'}
    assert any(not z.permanent for z in zones) and len(ports) >= 3
    assert 'NOT real' in meta['description']
    with pytest.raises(ValueError, match='anchor'):                    # the crop's anchor differs
        load_zones(D1, map_airspace('ecity_crop50'))


def test_crop_example_loads_into_e1_with_the_crop_airspace():
    """The crop's (0, 0) is not the default anchor: the helper gives E1 the right Airspace, and the
    zones end up in E1 as exactly zone_keys() of each zone."""
    sky = map_airspace('ecity_crop50')
    assert (sky.res, sky.layer_floor, sky.layer_height) == (15, 1.0, 2.0)
    zones, ports, meta = load_zones(ECITY, sky)
    assert meta['frame'] == 'wgs84' and len(ports) == 4
    layout = load_osm_layout(CROP_LAYOUT)
    planner = StrategicPlanner(layout, B, airspace=sky)
    apply_zones(planner, zones)
    red = set().union(*(zone_keys(z, sky, 2) for z in zones
                        if z.permanent and z.restriction == 'prohibited'))
    assert planner.blocked == red and red
    for z in zones:                                     # permanent yellow zones: booked forever
        if z.permanent and z.restriction == 'authorisation':
            for key in zone_keys(z, sky, 2):
                assert (-np.inf, np.inf, 'zone:' + z.id) in planner.ledger[key]
    temp = zones[[z.id for z in zones].index('ecity_temp_street')]
    for key in zone_keys(temp, sky, 2):
        assert (120.0, 300.0, 'zone:ecity_temp_street') in planner.ledger[key]
    # a lat/lng file is tied to the ground, not the anchor: read with the full-map anchor, the
    # same zone sits at (-73, 230) m + its crop position, and covers the same H3 cells
    full = planner_airspace()
    shifted, _, _ = load_zones(ECITY, full)
    assert np.allclose(shifted[0].outline(), zones[0].outline() + (-73.0, 230.0), atol=0.05)
    assert zone_cells(shifted[0], full) == zone_cells(zones[0], sky)
    # vertiports are clear of buildings and zones
    from envs.scenario_manager import surface_distance
    for v in ports:
        assert all(z.distance([(v.x, v.y)])[0] > 0 for z in zones), v.id
        assert all(surface_distance(ob, (v.x, v.y)) > 1.0 for ob in layout), v.id


def _hex_gap(sky, cell, p):
    """Distance (m) from point p to a cell's hexagon (0 inside)."""
    poly = np.asarray(sky.boundary(cell))
    a, b = poly, np.roll(poly, -1, axis=0)
    ab, ap = b - a, np.asarray(p, dtype=float) - a
    cross = ab[:, 0] * ap[:, 1] - ab[:, 1] * ap[:, 0]
    if np.all(cross >= 0) or np.all(cross <= 0):
        return 0.0
    t = np.clip(np.sum(ap * ab, axis=1) / np.sum(ab * ab, axis=1), 0.0, 1.0)
    return float(np.min(np.linalg.norm(ap - t[:, None] * ab, axis=1)))


def _example_map(name):
    if name == 'ecity_crop50':
        return ECITY, load_osm_layout(CROP_LAYOUT), map_airspace('ecity_crop50')
    from envs.city_generator import generate_city
    return D1, generate_city(np.random.RandomState(3), 'medium')['layout'], map_airspace('d1')


@pytest.mark.parametrize('name', ['ecity_crop50', 'd1_medium_seed3'])
def test_example_vertiports_are_clear_of_zones_and_buildings(name):
    """Every vertiport is >= 4 m from every zone's key cells (any layer; E1's pad radius is 2.1 m)
    and >= 1 m from every building."""
    from envs.scenario_manager import surface_distance
    path, layout, sky = _example_map(name)
    zones, ports, _ = load_zones(path, sky)
    assert ports
    for v in ports:
        for z in zones:
            cells = {c for c, _ in zone_keys(z, sky, 2)}
            gap = min(_hex_gap(sky, c, (v.x, v.y)) for c in cells)
            assert gap >= 4.0, (v.id, z.id, gap)
        bld = min(surface_distance(ob, (v.x, v.y)) for ob in layout)
        assert bld >= 1.0, (v.id, bld)


# ------------------------------------------------------------------ B. keys
def _inside_points(zone, rng, n=2000):
    pts = zone.outline()
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    cand = rng.uniform(lo, hi, size=(n * 4, 2))
    return cand[zone.distance(cand) == 0][:n]


@pytest.mark.parametrize('zone', [SQUARE, L_SHAPE, CIRCLE], ids=lambda z: z.id)
def test_every_point_of_a_zone_is_covered(zone):
    cells = zone_cells(zone, SKY)
    pts = _inside_points(zone, np.random.RandomState(0))
    assert len(pts) > 500
    missing = [p for p in pts if SKY.cell(*p) not in cells]
    assert not missing, missing[:3]


@pytest.mark.parametrize('zone', [SQUARE, L_SHAPE, CIRCLE], ids=lambda z: z.id)
def test_no_cell_lies_outside_the_zone_grown_by_one_cell(zone):
    """Each cell's centre is within one cell (its corner radius) of the zone; for the L-shape this
    includes the inner corner, where a convex-only cover would add cells."""
    cells = zone_cells(zone, SKY)
    centres = np.array([SKY.center(c) for c in cells])
    r = max(np.hypot(px - cx, py - cy) for c, (cx, cy) in zip(cells, centres) for px, py in SKY.boundary(c))
    assert zone.distance(centres).max() <= r + 1e-9
    if zone is L_SHAPE:                         # nothing in the notch (2..6, 2..7) away from the edges
        notch = Zone('n', 'n', polygon=[(2 + r, 2 + r), (6, 2 + r), (6, 7), (2 + r, 7)])
        assert not any(notch.distance([SKY.center(c)])[0] == 0 for c in cells)


def test_tiny_zone_still_has_its_cell():
    tiny = Zone('t', 'Tiny', circle=((1.0, 1.0), 0.05))
    assert SKY.cell(1.0, 1.0) in zone_cells(tiny, SKY)


def test_height_bands():
    """E1 layers: 0 = 1-3 m, 1 = 3-5 m."""
    poly = square(0, 0, 2)
    assert Zone('a', 'a', lower_m=3.0, upper_m=10.0, polygon=poly).layers(SKY, 2) == [1]
    assert Zone('b', 'b', polygon=poly).layers(SKY, 2) == [0, 1]
    assert Zone('c', 'c', upper_m=3.0, polygon=poly).layers(SKY, 2) == [0]
    assert Zone('d', 'd', lower_m=6.0, polygon=poly).layers(SKY, 2) == []    # above E1's layers
    assert Zone('e', 'e', upper_m=0.5, polygon=poly).layers(SKY, 2) == []    # under the pads
    keys = zone_keys(Zone('a', 'a', lower_m=3.0, upper_m=10.0, polygon=poly), SKY, 2)
    assert {layer for _, layer in keys} == {1}


def _points_outside(zone, d, rng, n):
    """n points exactly d m outside a zone's edge (circle, or convex polygon: an edge point moved
    out along that edge's normal)."""
    if zone.circle is not None:
        (cx, cy), r = zone.circle
        a = rng.uniform(0, 2 * np.pi, n)
        pts = np.column_stack([cx + (r + d) * np.cos(a), cy + (r + d) * np.sin(a)])
    else:
        poly = np.asarray(zone.polygon)
        if np.sum(poly[:, 0] * np.roll(poly[:, 1], -1) - np.roll(poly[:, 0], -1) * poly[:, 1]) < 0:
            poly = poly[::-1]                                   # counter-clockwise
        edges = np.roll(poly, -1, axis=0) - poly
        lengths = np.linalg.norm(edges, axis=1)
        k = rng.choice(len(poly), n, p=lengths / lengths.sum())
        t = rng.uniform(0, 1, n)[:, None]
        normals = np.column_stack([edges[:, 1], -edges[:, 0]]) / lengths[:, None]
        pts = poly[k] + t * edges[k] + d * normals[k]
    assert np.allclose(zone.distance(pts), d, atol=1e-6)        # really d m from the zone
    return pts


@pytest.mark.parametrize('name', ['ecity_crop50', 'd1_medium_seed3'])
def test_safety_buffer_around_example_zones(name):
    """Docstring rule 2, with E1's defaults (radius 2.1 m, resolution 15): a drone 2.0 m outside a
    zone's edge is always blocked (its cells_near shares a key cell), one 3.6 m outside never is."""
    path, _, sky = _example_map(name)
    zones, _, _ = load_zones(path, sky)
    planner = StrategicPlanner([], ((-60.0, 60.0), (-60.0, 60.0)), Settings(), airspace=sky)
    assert (planner.s.radius, sky.res) == (2.1, 15)
    rng = np.random.RandomState(4)
    for z in zones:
        cells = zone_cells(z, sky)
        for d, blocked in ((2.0, True), (3.6, False)):
            for x, y in _points_outside(z, d, rng, 200):
                hit = bool(cells.intersection(planner.cells_near(x, y)))
                assert hit == blocked, (z.id, d, (x, y))


def test_thin_strip_over_blocking_matches_docstring():
    """Docstring rule 3: a 0.5 m x 10 m strip's key cells cover about 3.8x its area (3.3-4.1x by
    position and angle); stays in 3.0-4.5x."""
    c = SKY.cell(0.0, 0.0)
    corners = np.asarray(SKY.boundary(c))
    hex_area = 0.5 * abs(np.sum(corners[:, 0] * np.roll(corners[:, 1], -1)
                                - np.roll(corners[:, 0], -1) * corners[:, 1]))
    rng = np.random.RandomState(5)
    ratios = []
    for _ in range(8):
        a = rng.uniform(0, np.pi)
        centre = rng.uniform(-5, 5, 2)
        u, v = np.array([np.cos(a), np.sin(a)]), np.array([-np.sin(a), np.cos(a)])
        strip = [tuple(centre + su * 5.0 * u + sv * 0.25 * v)
                 for su, sv in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
        ratios.append(len(zone_cells(Zone('strip', 'Strip', polygon=strip), SKY)) * hex_area / 5.0)
    assert all(3.0 <= r <= 4.5 for r in ratios), ratios
    assert 3.3 <= float(np.median(ratios)) <= 4.2, ratios


# ------------------------------------------------------------------ C. with E1
def test_start_pad_inside_a_permanent_zone_is_rejected():
    sky = StrategicPlanner([], B)
    apply_zones(sky, [Zone('pad', 'Pad zone', polygon=square(-15, 0, 2))])
    d = sky.submit(FlightPlan('A', (-15, 0), (15, 10)))
    assert d.status == 'rejected'


def _temporary(t0, t1):
    return Zone('tmp', 'Street closure', active=[[t0, t1]], polygon=square(0, 0, 2))


def _wall(active=()):
    """A closure across the whole map: no way around it."""
    return Zone('wall', 'Street closure', active=list(active),
                polygon=[(-2, -25), (2, -25), (2, 25), (-2, 25)])


def test_temporary_zone_delays_the_flight_until_it_ends():
    sky = StrategicPlanner([], B)
    zone = _wall([[0.0, 20.0]])
    apply_zones(sky, [zone])
    d = sky.submit(FlightPlan('A', (-15, 0), (15, 0)))
    assert d.status == 'approved' and d.delay > 0
    for key in zone_keys(zone, sky.sky, sky.s.n_layers):
        assert all(lo >= 20.0 for lo, hi in d.reservation.get(key, ()))     # only after the zone ends


def test_temporary_zone_beyond_max_delay_rejects():
    sky = StrategicPlanner([], B, Settings(max_delay=5.0))
    apply_zones(sky, [_wall([[0.0, 40.0]])])
    assert sky.submit(FlightPlan('A', (-15, 0), (15, 0))).status == 'rejected'


def test_flight_away_from_zones_is_unaffected():
    sky = StrategicPlanner([], B)
    apply_zones(sky, [_temporary(0.0, 300.0), Zone('perm', 'Perm', polygon=square(-15, -15, 3)),
                      Zone('high', 'High', lower_m=3.0, polygon=square(0, 15, 3))])
    d = sky.submit(FlightPlan('A', (-15, 15), (15, 15)))      # passes under the 'high' zone in layer 0
    assert (d.status, d.layer, d.delay) == ('approved', 0, 0.0)


def test_conflict_ids_map_back_to_zone_names():
    zones = [_temporary(0.0, 300.0), Zone('perm', 'Hospital', polygon=square(10, 0, 2))]
    free = StrategicPlanner([], B).submit(FlightPlan('A', (-15, 0), (15, 0)))
    sky = StrategicPlanner([], B)
    index = apply_zones(sky, zones)
    ids = sky.conflicts(free.reservation)
    assert ids == {'zone:tmp', 'blocked'}
    assert zone_names(ids | {'some_flight'}, zones) == ['Street closure', 'permanent no-fly zone']
    assert zones_hit(free.reservation, index, zones) == ['Hospital', 'Street closure']


def _yellow(active=()):
    return Zone('yel', 'Campus', 'authorisation', 'test', active=list(active), authorised=['ok'],
                polygon=square(0, 0, 2))


def test_authorised_flight_flies_through_a_permanent_yellow_zone():
    sky = StrategicPlanner([], B)
    apply_zones(sky, [_yellow()])
    ok = sky.submit(FlightPlan('ok', (-15, 0), (15, 0)))
    assert (ok.status, ok.delay) == ('approved', 0.0)
    other = sky.submit(FlightPlan('other', (-15, 1), (15, 1)))
    assert other.status == 'approved' and other.around_zones


def test_authorised_flight_is_not_delayed_by_a_temporary_yellow_zone():
    sky = StrategicPlanner([], B)
    apply_zones(sky, [_yellow(active=[[0.0, 20.0]])])
    assert sky.submit(FlightPlan('ok', (-15, 0), (15, 0))).delay == 0.0
    assert sky.submit(FlightPlan('other', (-15, 0), (15, 0), t_earliest=0.0)).delay > 0


def test_rejection_names_the_zone_in_the_way():
    zones = [_wall()]
    sky = StrategicPlanner([], B)
    apply_zones(sky, zones)
    d = sky.submit(FlightPlan('A', (-15, 0), (15, 0)))
    assert d.status == 'rejected' and d.blockers == ['blocked']
    clear = sky.submit(FlightPlan('B', (-15, 15), (-15, -15)))
    assert clear.status == 'approved' and clear.blockers == []


def test_apply_zones_before_any_flight_only():
    sky = StrategicPlanner([], B)
    sky.submit(FlightPlan('A', (-15, 0), (15, 0)))
    with pytest.raises(ValueError, match='before any flight'):
        apply_zones(sky, [SQUARE])
    with pytest.raises(ValueError, match='unique'):
        apply_zones(StrategicPlanner([], B), [SQUARE, SQUARE])


def _touches(d, zone, sky):
    keys = zone_keys(zone, sky.sky, sky.s.n_layers)
    return [key for key in d.reservation if key in keys]


def test_flight_detours_around_a_permanent_zone():
    sky = StrategicPlanner([], B)
    zone = Zone('perm', 'Perm', polygon=square(0, 0, 3))
    apply_zones(sky, [zone])
    d = sky.submit(FlightPlan('A', (-15, 0), (15, 0)))
    assert d.status == 'approved' and d.around_zones and d.delay == 0.0
    assert not _touches(d, zone, sky)
    assert max(abs(y) for _, y, _ in d.waypoints) > 3.0              # went around, not through


def test_short_closure_waits_long_closure_detours():
    """Cost decides: a closure ending soon is cheaper to wait out than a detour; a long one isn't."""
    def closure(t1):
        return Zone('tmp', 'Street closure', active=[[0.0, t1]], polygon=square(0, 0, 6))

    short, long_ = closure(10.0), closure(50.0)
    sky = StrategicPlanner([], B)
    apply_zones(sky, [short])
    wait = sky.submit(FlightPlan('A', (-20, 0), (20, 0)))
    assert wait.status == 'approved' and not wait.around_zones and wait.delay > 0
    sky = StrategicPlanner([], B)
    apply_zones(sky, [long_])
    go = sky.submit(FlightPlan('A', (-20, 0), (20, 0)))
    assert go.status == 'approved' and go.around_zones and go.delay < 50.0
    for key in zone_keys(long_, sky.sky, sky.s.n_layers):          # never inside while it is closed
        assert all(lo >= 50.0 for lo, hi in go.reservation.get(key, ()))


def test_authorised_flight_keeps_the_direct_route():
    sky = StrategicPlanner([], B)
    apply_zones(sky, [_yellow()])
    ok = sky.submit(FlightPlan('ok', (-15, 0), (15, 0)))
    assert not ok.around_zones and ok.delay == 0.0
    other = sky.submit(FlightPlan('other', (-15, 18), (15, -18)))
    assert other.status == 'approved' and other.around_zones


# ------------------------------------------------------------------ D. generator
@pytest.mark.parametrize('target', [0.10, 0.20, 0.30])
def test_generator_hits_its_target(target):
    from envs.city_generator import generate_city
    for seed in range(3):
        city = generate_city(np.random.RandomState(seed), 'medium')
        blocks = city_blocks(city)
        area = city['size'][0] * city['size'][1]
        zones = random_block_zones(blocks, area, target, np.random.RandomState(seed))
        tol = 0.5 * max((x1 - x0) * (y1 - y0) for x0, x1, y0, y1 in blocks) / area
        assert abs(coverage(zones, area) - target) <= tol + 1e-12, (seed, coverage(zones, area))
    tiles = tile_blocks(B, 5.0)
    zones = random_block_zones(tiles, 2500.0, target, np.random.RandomState(0))
    assert abs(coverage(zones, 2500.0) - target) <= 0.5 * 25.0 / 2500.0 + 1e-12


def test_generator_is_deterministic():
    blocks = tile_blocks(B, 5.0)
    a = random_block_zones(blocks, 2500.0, 0.2, np.random.RandomState(7))
    b = random_block_zones(blocks, 2500.0, 0.2, np.random.RandomState(7))
    c = random_block_zones(blocks, 2500.0, 0.2, np.random.RandomState(8))
    assert [z.polygon for z in a] == [z.polygon for z in b]
    assert [z.polygon for z in a] != [z.polygon for z in c]
    assert json.dumps(to_geojson(a, (), SKY)) == json.dumps(to_geojson(b, (), SKY))
