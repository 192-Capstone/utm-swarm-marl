"""
webots_zones.py
---------------
C7: D2's no-fly zones (utm/airspace_zones.py) drawn in the Webots city worlds
(envs/webots_l2_multi.py, webots.draw_zones). Visual only: the nodes are plain Pose / Shape
nodes (no Solid, no boundingObject, no Physics), so ODE, the analytic LiDAR and the crash checks
never see them, and the drone controllers only read the IMU, GPS and gyro.

    circle    a semi-transparent Cylinder (the true circle, not E1's circumscribed polygon)
    polygon   the outline extruded upward: an IndexedFaceSet with side walls and top / bottom
              caps (ear clipping, so concave zones are right)
    outline   an opaque flat strip, OUTLINE_W wide and centred on the zone's edge, OUTLINE_Z above
              the ground (one rectangle per edge, overlapping at the corners; a ring for circles)
    colour    'prohibited' red (transparency 0.45), 'authorisation' deep amber 1 / 0.5 / 0 (0.6)
    height    lower_m .. upper_m; null lower = the ground, null upper = the airspace ceiling
              (webots.zones_ceiling_m, default E1's top: layer_floor + n_layers * layer_height).
              A zone whose band ends at or below the ground (or bottom >= top) is not drawn.

Show / hide: zones with active windows are shown only while active ([t0, t1), episode seconds,
the clock SwarmEnv uses for E1 departures). ZoneDrawer.update(t) calls Node.setVisibility(VIEW,
visible) on a zone's node only when its state changes: a rendering-only switch, no field of the
scene tree changes and nothing is imported or removed mid-episode.

Nodes are rebuilt only when zones_key changes (geometry, restriction, heights, ceiling; not the
windows, which only change visibility), as the buildings follow webots.reuse_layout.
"""

import math
import time

import numpy as np

COLOURS = {'prohibited': (1.0, 0.0, 0.0), 'authorisation': (1.0, 0.5, 0.0)}
# volume transparency: red less transparent so it stays red over the grass in the render
TRANSPARENCY = {'prohibited': 0.45, 'authorisation': 0.6}
OUTLINE_W = 0.12                 # m
OUTLINE_Z = 0.02                 # m above the ground (above the Floor at z = 0)
CIRCLE_SUBDIVISION = 48
ZONES_GROUP = 'DEF ZONES Group { }'


def default_ceiling():
    """E1's airspace top: layer_floor + n_layers * layer_height (5 m with its defaults)."""
    from utm.strategic_planner import Settings
    s = Settings()
    return float(s.layer_floor + s.n_layers * s.layer_height)


def band(zone, ceiling_m):
    """(bottom, top) heights a zone is drawn over, or None when there is nothing to draw."""
    lo = 0.0 if zone.lower_m is None else float(zone.lower_m)
    hi = float(ceiling_m) if zone.upper_m is None else float(zone.upper_m)
    return (lo, hi) if hi > lo else None


def active_at(zone, t):
    """Permanent zones always; else t0 <= t < t1 for one of its windows."""
    return not zone.active or any(t0 <= t < t1 for t0, t1 in zone.active)


def zones_key(zones, ceiling_m):
    """Exact, hashable form of what the zone nodes are made from (not the active windows)."""
    out = []
    for z in zones:
        geom = (('circle', tuple(z.circle[0]), z.circle[1]) if z.circle is not None
                else ('polygon', tuple(tuple(p) for p in z.polygon)))
        out.append((z.id, z.restriction, z.lower_m, z.upper_m, geom))
    return float(ceiling_m), tuple(out)


# ------------------------------------------------------------------ geometry
def _signed_area(pts):
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _cross(o, a, b):
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _in_triangle(p, a, b, c):
    """p inside or on the edge of the counter-clockwise triangle abc."""
    return _cross(a, b, p) >= -1e-12 and _cross(b, c, p) >= -1e-12 and _cross(c, a, p) >= -1e-12


def ccw(poly):
    """The polygon's corners, counter-clockwise."""
    pts = np.asarray(poly, dtype=float)
    return pts if _signed_area(pts) > 0 else pts[::-1].copy()


def triangulate(poly):
    """Ear clipping of a simple counter-clockwise polygon (no holes). Returns index triples
    (counter-clockwise) into poly; collinear corners give no triangle of their own."""
    pts = [tuple(p) for p in np.asarray(poly, dtype=float)]
    idx = list(range(len(pts)))
    tris = []
    while len(idx) > 3:
        n = len(idx)
        for k in range(n):
            i0, i1, i2 = idx[k - 1], idx[k], idx[(k + 1) % n]
            a, b, c = pts[i0], pts[i1], pts[i2]
            if _cross(a, b, c) <= 1e-12:
                continue                                    # reflex or collinear: not an ear
            if any(_in_triangle(pts[j], a, b, c) for j in idx if j not in (i0, i1, i2)):
                continue
            tris.append((i0, i1, i2))
            del idx[k]
            break
        else:                                               # no ear: drop a collinear corner
            for k in range(n):
                if abs(_cross(pts[idx[k - 1]], pts[idx[k]], pts[idx[(k + 1) % n]])) <= 1e-12:
                    del idx[k]
                    break
            else:
                raise ValueError('triangulate: polygon is not simple')
    if len(idx) == 3 and _cross(*(pts[i] for i in idx)) > 1e-12:
        tris.append(tuple(idx))
    return tris


def _f(v):
    return f'{v:.6f}'


def _vec(p):
    return ' '.join(_f(v) for v in p)


def _appearance(colour, transparency):
    r, g, b = (_f(v) for v in colour)
    if transparency:
        return f'PBRAppearance {{ baseColor {r} {g} {b} transparency {transparency} roughness 1 metalness 0 }}'
    return f'PBRAppearance {{ baseColor {r} {g} {b} emissiveColor {r} {g} {b} roughness 1 metalness 0 }}'


def _face_set(points, faces, appearance):
    pts = ', '.join(_vec(p) for p in points)
    idx = ' '.join(' '.join(str(i) for i in f) + ' -1' for f in faces)
    return (f'Shape {{ appearance {appearance} geometry IndexedFaceSet {{ coord Coordinate {{ point [ {pts} ] }} '
            f'coordIndex [ {idx} ] solid FALSE creaseAngle 0 }} }}')


def prism(poly, lo, hi):
    """Points and faces of a polygon extruded from lo to hi: bottom ring (0 .. n-1) at lo, top
    ring (n .. 2n-1) at hi; bottom cap, top cap, side quads."""
    p = ccw(poly)
    n = len(p)
    points = [(x, y, lo) for x, y in p] + [(x, y, hi) for x, y in p]
    tris = triangulate(p)
    faces = [(c, b, a) for a, b, c in tris]                           # bottom: facing down
    faces += [(a + n, b + n, c + n) for a, b, c in tris]              # top: facing up
    faces += [(i, (i + 1) % n, (i + 1) % n + n, i + n) for i in range(n)]
    return points, faces


def outline_strip(poly, width=OUTLINE_W, z=OUTLINE_Z):
    """One flat rectangle per edge, `width` wide, centred on the edge and extended by width / 2
    at both ends so the corners are covered."""
    p = ccw(poly)
    h = width / 2.0
    points, faces = [], []
    for i in range(len(p)):
        a, b = p[i], p[(i + 1) % len(p)]
        d = (b - a) / np.linalg.norm(b - a)
        nrm = np.array([-d[1], d[0]])
        a2, b2 = a - d * h, b + d * h
        k = len(points)
        points += [(*(a2 - nrm * h), z), (*(b2 - nrm * h), z), (*(b2 + nrm * h), z), (*(a2 + nrm * h), z)]
        faces.append((k, k + 1, k + 2, k + 3))
    return points, faces


def ring_strip(centre, radius, width=OUTLINE_W, z=OUTLINE_Z, segments=CIRCLE_SUBDIVISION):
    """A flat annulus `width` wide centred on the circle."""
    cx, cy = centre
    r0, r1 = max(radius - width / 2.0, 0.0), radius + width / 2.0
    a = np.linspace(0.0, 2.0 * math.pi, segments, endpoint=False)
    points = [(cx + r * math.cos(t), cy + r * math.sin(t), z) for t in a for r in (r0, r1)]
    faces = [(2 * i, 2 * i + 1, (2 * i + 3) % (2 * segments), (2 * i + 2) % (2 * segments))
             for i in range(segments)]                      # counter-clockwise seen from above
    return points, faces


def zone_node(k, zone, ceiling_m):
    """The node string for one zone (DEF ZONE_<k>), or None when there is nothing to draw."""
    b = band(zone, ceiling_m)
    if b is None:
        return None
    lo, hi = b
    colour = COLOURS[zone.restriction]
    vol_app, line_app = _appearance(colour, TRANSPARENCY[zone.restriction]), _appearance(colour, 0)
    if zone.circle is not None:
        (cx, cy), r = zone.circle
        volume = (f'Pose {{ translation {_f(cx)} {_f(cy)} {_f((lo + hi) / 2)} children [ Shape {{ '
                  f'appearance {vol_app} geometry Cylinder {{ radius {_f(r)} height {_f(hi - lo)} '
                  f'subdivision {CIRCLE_SUBDIVISION} }} }} ] }}')
        outline = _face_set(*ring_strip((cx, cy), r), line_app)
    else:
        volume = _face_set(*prism(zone.polygon, lo, hi), vol_app)
        outline = _face_set(*outline_strip(zone.polygon), line_app)
    return f'DEF ZONE_{k} Group {{ children [ {volume} {outline} ] }}'


def zones_from_config(wcfg):
    """Zones from webots.zones_file / webots.zones_map ([] when no file is set)."""
    path, name = wcfg.get('zones_file'), wcfg.get('zones_map')
    if not path:
        if name:
            raise ValueError('webots.zones_map is set but webots.zones_file is not')
        return []
    if not name:
        raise ValueError("webots.zones_file needs webots.zones_map (the map whose airspace anchors it: "
                         "'ecity', 'ecity_crop50' or 'd1')")
    from utm.airspace_zones import load_zones, map_airspace
    zones, _, _ = load_zones(path, map_airspace(name))
    return zones


# ------------------------------------------------------------------ drawer
class ZoneDrawer:
    """Holds the ZONES group of a running Webots: build(zones) at every reset, update(t) after
    every control step. The group is imported into the scene root at the first build."""

    def __init__(self, sup, viewpoint, ceiling_m=None, reuse=True):
        self.sup, self.viewpoint, self.reuse = sup, viewpoint, bool(reuse)
        self.ceiling_m = default_ceiling() if ceiling_m is None else float(ceiling_m)
        self._children = None
        self._key = None
        self.zones, self.names, self._drawn = [], [], []     # _drawn: [(zone, node, visible)]
        self.builds, self.build_s, self.toggles = 0, 0.0, 0

    def build(self, zones):
        """Make the nodes for these zones (only when they changed, if reuse) and show the ones
        active at t = 0."""
        zones = list(zones)
        key = zones_key(zones, self.ceiling_m)
        if not (self.reuse and key == self._key):
            t0 = time.time()
            if self._children is None:
                self.sup.getRoot().getField('children').importMFNodeFromString(-1, ZONES_GROUP)
                self._children = self.sup.getFromDef('ZONES').getField('children')
            for _ in range(self._children.getCount()):
                self._children.removeMF(-1)
            self._drawn, self.names = [], []
            for k, z in enumerate(zones):
                s = zone_node(k, z, self.ceiling_m)
                if s is None:
                    continue
                self._children.importMFNodeFromString(-1, s)
                self._drawn.append([z, self._children.getMFNode(-1), True])    # imported visible
                self.names.append(z.id)
            self._key = key
            self.builds += 1
            self.build_s += time.time() - t0
        else:                                               # same nodes; windows may have changed
            by_id = {z.id: z for z in zones}
            for d in self._drawn:
                d[0] = by_id[d[0].id]
        self.zones = zones
        self.update(0.0)

    def update(self, t):
        """Show the zones active at episode time t, hide the others (only changes are sent)."""
        for d in self._drawn:
            vis = active_at(d[0], t)
            if vis != d[2]:
                d[1].setVisibility(self.viewpoint, vis)
                d[2] = vis
                self.toggles += 1
