"""
test_webots_zones.py
--------------------
C7: D2 no-fly zones drawn in the Webots L2 multi world (envs/webots_zones.py, webots.draw_zones),
without Webots: the node strings are parsed back independently, and a fake supervisor records
what the drawer and the adapter ask for.

Run: python -m pytest tests/test_webots_zones.py -q
"""

import json
import math
import os
import re
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))

from envs import webots_zones as wz                                            # noqa: E402
from envs.webots_adapter import building_node, scan_lidar                      # noqa: E402
from envs.webots_l2_multi import DRONE_NODE, WebotsL2MultiAdapter              # noqa: E402
from utm.airspace_zones import Zone, load_zones, map_airspace                  # noqa: E402
from test_webots_l2_multi_city import _fake, _Root                             # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLES = [('data/airspace/d1_medium_seed3_example.geojson', 'd1'),
            ('data/airspace/ecity_crop50_example.geojson', 'ecity_crop50')]
SQUARE = [(0.0, 0.0), (4.0, 0.0), (4.0, 3.0), (0.0, 3.0)]
L_SHAPE = [(0.0, 0.0), (6.0, 0.0), (6.0, 2.0), (2.0, 2.0), (2.0, 5.0), (0.0, 5.0)]
STRIP = [(0.0, 0.0), (10.0, 0.0), (10.0, 0.5), (0.0, 0.5)]


# ------------------------------------------------------------------ independent parsers
def face_sets(s):
    """[(points (N, 3), faces [[i, ...]])] of every IndexedFaceSet in a node string."""
    out = []
    for pts, idx in re.findall(r'point \[ ([^\]]*) \] \} coordIndex \[ ([^\]]*) \]', s):
        p = np.array([[float(v) for v in q.split()] for q in pts.split(',')])
        faces, cur = [], []
        for v in idx.split():
            if v == '-1':
                faces.append(cur); cur = []
            else:
                cur.append(int(v))
        out.append((p, faces))
    return out


def area2d(pts):
    x, y = np.asarray(pts)[:, 0], np.asarray(pts)[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def inside(poly, p):
    poly = np.asarray(poly)
    c = False
    for i in range(len(poly)):
        (xi, yi), (xj, yj) = poly[i], poly[i - 1]
        if (yi > p[1]) != (yj > p[1]) and p[0] < (xj - xi) * (p[1] - yi) / (yj - yi) + xi:
            c = not c
    return c


def cap_triangles(points, faces, z):
    return [f for f in faces if len(f) == 3 and all(abs(points[i][2] - z) < 1e-9 for i in f)]


# ------------------------------------------------------------------ 1-3. node geometry
@pytest.mark.parametrize('lo,hi', [(None, None), (1.5, 4.0), (None, 3.0), (2.0, None)])
def test_circle_node(lo, hi):
    z = Zone('c', 'c', lower_m=lo, upper_m=hi, circle=((3.0, -2.0), 2.5))
    s = wz.zone_node(0, z, 5.0)
    b, t = (lo or 0.0), (5.0 if hi is None else hi)
    m = re.search(r'Pose \{ translation (\S+) (\S+) (\S+) children .*Cylinder \{ radius (\S+) height (\S+)', s)
    assert [float(v) for v in m.groups()] == pytest.approx([3.0, -2.0, (b + t) / 2, 2.5, t - b])
    assert 'baseColor 1.000000 0.000000 0.000000 transparency 0.45 ' in s
    assert s.startswith('DEF ZONE_0 Group')


@pytest.mark.parametrize('poly', [SQUARE, SQUARE[::-1], L_SHAPE, L_SHAPE[::-1], STRIP])
def test_polygon_prism(poly):
    z = Zone('p', 'p', lower_m=1.0, upper_m=4.0, polygon=poly)
    (pts, faces), _ = face_sets(wz.zone_node(3, z, 5.0))
    n = len(poly)
    bottom, top = pts[:n], pts[n:]
    assert np.allclose(bottom[:, 2], 1.0) and np.allclose(top[:, 2], 4.0)
    assert np.allclose(bottom[:, :2], top[:, :2])
    assert {tuple(p) for p in bottom[:, :2].round(6)} == {tuple(p) for p in np.round(poly, 6)}
    assert area2d(bottom) > 0                                             # counter-clockwise
    for zc in (1.0, 4.0):                                                 # caps cover the area exactly
        tris = cap_triangles(pts, faces, zc)
        assert sum(abs(area2d(pts[list(f)])) for f in tris) == pytest.approx(z.area())
        for f in tris:                                                    # and stay inside
            assert inside(poly, pts[list(f)][:, :2].mean(axis=0))
    assert sum(1 for f in faces if len(f) == 4) == n                      # one wall per edge


def test_triangulate_collinear_corner():
    poly = [(0.0, 0.0), (2.0, 0.0), (4.0, 0.0), (4.0, 3.0), (0.0, 3.0)]
    tris = wz.triangulate(wz.ccw(poly))
    assert sum(abs(area2d(np.array(poly)[list(t)])) for t in tris) == pytest.approx(12.0)


def test_outline_strips():
    for z in (Zone('p', 'p', polygon=L_SHAPE), Zone('c', 'c', circle=((1.0, 1.0), 3.0))):
        s = wz.zone_node(0, z, 5.0)
        pts, faces = face_sets(s)[-1]                                    # the outline comes last
        assert np.allclose(pts[:, 2], wz.OUTLINE_Z)
        assert s.count('emissiveColor 1.000000 0.000000 0.000000') == 1             # opaque outline
        assert s.count('transparency') == 1                                          # only the volume
    # polygon strip: each edge's rectangle is OUTLINE_W wide and centred on the edge
    pts, faces = wz.outline_strip(L_SHAPE)
    for f in faces:
        q = np.array([pts[i][:2] for i in f])
        assert np.linalg.norm(q[3] - q[0]) == pytest.approx(wz.OUTLINE_W)
        assert area2d(q) > 0
    # circle ring: radii r +- w/2
    pts, faces = wz.ring_strip((1.0, 1.0), 3.0)
    r = np.hypot(np.array(pts)[:, 0] - 1.0, np.array(pts)[:, 1] - 1.0)
    assert set(np.round(r, 9)) == {round(3.0 - wz.OUTLINE_W / 2, 9), round(3.0 + wz.OUTLINE_W / 2, 9)}
    assert all(area2d(np.array([pts[i][:2] for i in f])) > 0 for f in faces)


def test_amber_for_authorisation():
    s = wz.zone_node(0, Zone('a', 'a', 'authorisation', polygon=SQUARE), 5.0)
    assert 'baseColor 1.000000 0.500000 0.000000 transparency 0.6 ' in s
    assert 'emissiveColor 1.000000 0.500000 0.000000' in s


# ------------------------------------------------------------------ 4-5. visual only, heights
def test_zone_nodes_are_visual_only():
    for z in (Zone('p', 'p', polygon=L_SHAPE), Zone('c', 'c', 'authorisation', circle=((0, 0), 2.0))):
        s = wz.zone_node(0, z, 5.0)
        for bad in ('Solid', 'boundingObject', 'Physics', 'Robot', 'name '):
            assert bad not in s


def test_ceiling_and_band():
    assert wz.default_ceiling() == pytest.approx(5.0)
    z = Zone('p', 'p', polygon=SQUARE)
    assert wz.band(z, 5.0) == (0.0, 5.0) and wz.band(z, 7.5) == (0.0, 7.5)
    assert wz.band(Zone('p', 'p', lower_m=6.0, polygon=SQUARE), 5.0) is None       # above the ceiling
    assert wz.zone_node(0, Zone('p', 'p', lower_m=6.0, polygon=SQUARE), 5.0) is None
    assert wz.band(Zone('p', 'p', lower_m=6.0, upper_m=9.0, polygon=SQUARE), 5.0) == (6.0, 9.0)


# ------------------------------------------------------------------ 6-7. key, windows
def test_zones_key():
    z = Zone('p', 'p', polygon=SQUARE)
    base = wz.zones_key([z], 5.0)
    assert wz.zones_key([Zone('p', 'p', active=[[1.0, 2.0]], polygon=SQUARE)], 5.0) == base
    for other in (Zone('p', 'p', polygon=[(0, 0), (4, 0), (4, 3.001), (0, 3)]),
                  Zone('p', 'p', 'authorisation', polygon=SQUARE), Zone('p', 'p', upper_m=3.0, polygon=SQUARE),
                  Zone('p', 'p', lower_m=1.0, polygon=SQUARE), Zone('q', 'p', polygon=SQUARE)):
        assert wz.zones_key([other], 5.0) != base
    assert wz.zones_key([z], 6.0) != base


def test_active_at():
    z = Zone('p', 'p', active=[[2.0, 4.0], [6.0, 7.0]], polygon=SQUARE)
    assert [wz.active_at(z, t) for t in (0.0, 1.999, 2.0, 3.9, 4.0, 6.5, 7.0)] == \
        [False, False, True, True, False, True, False]
    assert wz.active_at(Zone('p', 'p', polygon=SQUARE), 1e9)


# ------------------------------------------------------------------ 8. fake supervisor
class _ZNode:
    def __init__(self, s, log):
        self.s, self.log = s, log

    def setVisibility(self, frm, vis):
        self.log.append(('vis', frm, re.match(r'DEF (\S+)', self.s).group(1), vis))


class _Children:
    def __init__(self, log):
        self.nodes, self.log = [], log

    def getCount(self):
        return len(self.nodes)

    def removeMF(self, k):
        self.nodes.pop(k)
        self.log.append(('remove',))

    def importMFNodeFromString(self, k, s):
        self.nodes.append(_ZNode(s, self.log))
        self.log.append(('import', s.split(' ')[1]))

    def getMFNode(self, k):
        return self.nodes[k]


class _RootNode:
    def __init__(self, sup):
        self.sup = sup

    def getField(self, name):
        return self

    def importMFNodeFromString(self, k, s):
        assert s == wz.ZONES_GROUP
        self.sup.log.append(('root_import', s))
        self.sup.zones = _Children(self.sup.log)


class _Def:
    def __init__(self, f):
        self.f = f

    def getField(self, name):
        assert name == 'children'
        return self.f


class _ZSup:
    def __init__(self):
        self.log, self.zones = [], None

    def getRoot(self):
        return _RootNode(self)

    def getFromDef(self, name):
        assert name == 'ZONES' and self.zones is not None
        return _Def(self.zones)


def _drawer(reuse=True):
    sup = _ZSup()
    return sup, wz.ZoneDrawer(sup, 'VIEW', None, reuse)


ZONES = [Zone('red', 'red', polygon=L_SHAPE),
         Zone('temp', 'temp', active=[[2.0, 4.0]], circle=((10.0, 0.0), 2.0)),
         Zone('high', 'high', lower_m=7.0, polygon=SQUARE)]                  # above the ceiling: not drawn


def test_drawer_builds_once_and_reuses():
    sup, d = _drawer()
    d.build(ZONES)
    assert [e for e in sup.log if e[0] == 'root_import'] == [('root_import', wz.ZONES_GROUP)]
    assert d.names == ['red', 'temp'] and sup.zones.getCount() == 2 and d.builds == 1
    assert ('vis', 'VIEW', 'ZONE_1', False) in sup.log                   # inactive at t = 0: hidden
    d.build([Zone(z.id, z.name, z.restriction, z.reason, z.lower_m, z.upper_m, z.active, z.authorised,
                  z.polygon, z.circle) for z in ZONES])                   # equal zones: kept
    assert d.builds == 1 and sum(1 for e in sup.log if e[0] == 'import') == 2
    d.build(ZONES[:1])                                                    # changed: rebuilt
    assert d.builds == 2 and sup.zones.getCount() == 1
    assert sum(1 for e in sup.log if e[0] == 'root_import') == 1         # group imported once


def test_drawer_reuse_off_rebuilds():
    sup, d = _drawer(reuse=False)
    d.build(ZONES); d.build(ZONES)
    assert d.builds == 2 and sup.zones.getCount() == 2


def test_visibility_only_on_window_edges():
    sup, d = _drawer()
    d.build(ZONES)
    sup.log.clear()
    for k in range(200):
        d.update(k * 0.032)
    assert [e[1:] for e in sup.log] == [('VIEW', 'ZONE_1', True), ('VIEW', 'ZONE_1', False)]
    n = len(sup.log)
    d.build(ZONES)                                                        # next episode, reused, t = 0:
    assert len(sup.log) == n and d.toggles == 3                           # already hidden, nothing sent


def test_reused_nodes_follow_new_windows():
    sup, d = _drawer()
    d.build(ZONES)
    moved = [ZONES[0], Zone('temp', 'temp', active=[[0.0, 1.0]], circle=((10.0, 0.0), 2.0)), ZONES[2]]
    d.build(moved)                                                        # same geometry: no rebuild
    assert d.builds == 1 and ('vis', 'VIEW', 'ZONE_1', True) == sup.log[-1]


# adapter: flag off touches nothing, flag on builds at reset and updates per step
def _adapter(drawer=None):
    ad = _fake(2)
    ad.real, ad.reuse_layout, ad._built_key = True, True, None
    ad.layout_builds, ad.layout_build_s = 0, 0.0
    ad._obstacle_root = _Root()
    ad.ep, ad.spinup_hold_s, ad.goals, ad.control_dt = 0, 0.008, {}, 0.032
    ad.reimport_log, ad._reimported_now = [], []
    if drawer is not None:
        ad._zones, ad._zone_list = drawer, list(ZONES)
    return ad


BOX = {'type': 'box', 'center': (1.0, 2.0), 'half': (0.5, 1.5), 'height': 8.0}
SP = {'drone_0': [0.0, 0.0, 1.0], 'drone_1': [3.0, 0.0, 1.0]}


def test_flag_off_adapter_unchanged():
    ad = _adapter()
    assert ad._zones is None                                              # class default
    ad.set_zones(ZONES)                                                   # stored, not drawn
    ad.reset_world(SP, SP, [BOX])                                         # _Sup has no getRoot: would raise
    ad.apply_actions({k: [0.0] * 4 for k in SP}); ad.step_simulation()
    assert ad._obstacle_root.nodes == [building_node(0, BOX)]
    assert not hasattr(ad, '_ep_step')


def test_flag_on_adapter_builds_and_updates():
    sup, d = _drawer()
    ad = _adapter(d)
    ad.reset_world(SP, SP, [BOX])
    assert d.builds == 1 and ad._ep_step == 0
    for _ in range(70):                                                   # 2.24 s: temp zone shown
        ad.apply_actions({k: [0.0] * 4 for k in SP}); ad.step_simulation()
    assert ad._ep_step == 70 and ('vis', 'VIEW', 'ZONE_1', True) in sup.log
    ad.reset_world(SP, SP, [BOX])
    assert d.builds == 1 and ad._ep_step == 0 and sup.log[-1] == ('vis', 'VIEW', 'ZONE_1', False)
    assert ad._obstacle_root.nodes == [building_node(0, BOX)]             # buildings unchanged


def test_node_strings_unchanged():
    """The strings the world is made of with the flag off (same as before C7)."""
    assert DRONE_NODE.format(i=1, x=2.0) == ('DEF DRONE_1 Crazyflie {\n  translation 2.0 0 0.015\n'
                                             '  name "drone_1"\n  controller "cf_velocity"\n}')
    assert building_node(0, BOX).startswith('SimpleBuilding { translation 1.0 2.0 0 name "obstacle_0" ')


# ------------------------------------------------------------------ 9. GeoJSON
@pytest.mark.parametrize('path,name', EXAMPLES)
def test_geojson_examples_drawn_where_the_file_says(path, name):
    sky = map_airspace(name)
    doc = json.load(open(os.path.join(ROOT, path)))
    zones = wz.zones_from_config({'zones_file': os.path.join(ROOT, path), 'zones_map': name})
    by_id = {z.id: z for z in zones}
    feats = [f for f in doc['features'] if f['properties']['type'] == 'zone']
    assert sorted(by_id) == sorted(f['id'] for f in feats)

    def local(c):                                       # straight from the file
        if doc.get('frame', 'wgs84') == 'local_m':
            return float(c[0]), float(c[1])
        return tuple(float(v) for v in sky.to_local(c[1], c[0]))

    for f in feats:
        z, p = by_id[f['id']], f['properties']
        s = wz.zone_node(0, z, 5.0)
        lo = p['lower_m'] or 0.0
        hi = 5.0 if p['upper_m'] is None else p['upper_m']
        if f['geometry']['type'] == 'Point':
            cx, cy = local(f['geometry']['coordinates'])
            m = re.search(r'translation (\S+) (\S+) (\S+) .*radius (\S+) height (\S+)', s)
            assert [float(v) for v in m.groups()] == pytest.approx([cx, cy, (lo + hi) / 2, p['radius_m'], hi - lo], abs=1e-6)
        else:
            ring = [local(c) for c in f['geometry']['coordinates'][0][:-1]]
            pts, _ = face_sets(s)[0]
            got = {tuple(np.round(q, 5)) for q in pts[:len(ring), :2]}
            assert got == {tuple(np.round(q, 5)) for q in ring}
            assert np.allclose(pts[:len(ring), 2], lo) and np.allclose(pts[len(ring):, 2], hi)


def test_config_errors():
    with pytest.raises(ValueError, match='zones_map'):
        wz.zones_from_config({'zones_file': EXAMPLES[0][0]})
    with pytest.raises(ValueError, match='zones_file'):
        wz.zones_from_config({'zones_map': 'd1'})
    with pytest.raises(ValueError, match='unknown map'):
        wz.zones_from_config({'zones_file': os.path.join(ROOT, EXAMPLES[0][0]), 'zones_map': 'mars'})
    assert wz.zones_from_config({}) == []


# ------------------------------------------------------------------ 10. LiDAR does not see zones
def test_lidar_ignores_zones():
    """The analytic LiDAR reads the layout only; the drawer never touches it."""
    from envs import lidar as lidar_mod
    import yaml
    cfg = yaml.safe_load(open(os.path.join(ROOT, 'configs/default_config.yaml')))
    lcfg = dict(lidar_mod.lidar_config(cfg), enabled=True)
    pos = np.array([[0.0, 0.0, 2.0]])
    before = scan_lidar(pos, [0.0], [0.0], [BOX], lcfg, 14)
    assert np.asarray(before[0]).min() < lcfg['range']                    # the scan does see the box
    ad = _adapter(_drawer()[1])
    ad.reset_world({'drone_0': [0, 0, 1], 'drone_1': [3, 0, 1]}, SP, [BOX])
    assert ad.layout == [BOX]
    after = scan_lidar(pos, [0.0], [0.0], ad.layout, lcfg, 14)
    assert np.array_equal(before[0], after[0]) and before[1] == after[1]
