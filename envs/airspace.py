"""
airspace.py
-----------
H3-indexed airspace (task B5): one standard cell system for the whole UTM layer.

Our simulations use local metres (x = east, y = north). H3 divides the Earth into
hexagonal cells addressed by latitude / longitude. An Airspace ties the two
together at an anchor point: by default the real point the Electronic City OSM
map (D3) was built around, so a simulated position maps to the H3 cell a real
system would use for that spot.

    sky = Airspace()                              # anchor 12.84968 N, 77.66498 E, resolution 12
    c = sky.cell(3.0, -7.5)                       # H3 cell id of a local position
    x, y = sky.center(c)                          # back to local metres
    ring = sky.disk(c, 1)                         # the cell and its 6 neighbours
    poly = sky.boundary(c)                        # 6 corners, local metres (dashboard)
    zone = sky.cells_for_obstacle(box)            # cells covering an obstacle / no-fly zone
    slot = sky.cell3d(3.0, -7.5, 42.0)            # (cell, altitude layer)

Uses:
    neighbour search (B5)           cells around a drone -> candidate neighbours
    no-fly zones, corridors (D2)    areas -> sets of cells
    altitude layers (D2)            (cell, layer) = a 3D airspace volume
    reservations (E1)               (cell, layer, time window) slots
    conformance (E3)                is the drone inside its reserved cells?
    traffic demand (D4), dashboard (E5): counts and heat maps per cell

Resolutions (average hexagon edge / area): 11 = 28.7 m / 2,150 m2, 12 = 10.8 m / 307 m2,
13 = 4.1 m / 44 m2. H3 is 2D; altitude is a separate layer index.

Local <-> lat/lng uses an equirectangular projection around the anchor (sub-mm
round trip, distance error < 0.01% within a few km), plenty for city-scale maps.
"""

import math

import h3
import numpy as np

# The query centre of the Electronic City OSM import (data/osm_city/*_metadata.yaml)
ANCHOR_ECITY = (12.84968, 77.66498)
EARTH_RADIUS_M = 6371007.180918475          # H3's authalic Earth radius
DEFAULT_RESOLUTION = 12


def _offset_convex(pts, g):
    """Grow a convex polygon (N, 2) outward by distance g: every edge moves out by g, and each
    corner moves to where its two shifted edges meet."""
    area2 = np.sum(pts[:, 0] * np.roll(pts[:, 1], -1) - np.roll(pts[:, 0], -1) * pts[:, 1])
    if area2 < 0:                                   # make the order counter-clockwise
        pts = pts[::-1]
    edges = np.roll(pts, -1, axis=0) - pts
    normals = np.column_stack([edges[:, 1], -edges[:, 0]])          # outward for CCW order
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    n_in, n_out = np.roll(normals, 1, axis=0), normals              # edges meeting at each corner
    bis = n_in + n_out
    scale = g / np.maximum(1.0 + np.sum(n_in * n_out, axis=1, keepdims=True), 1e-9)
    return pts + bis * scale


class Airspace:
    def __init__(self, anchor=ANCHOR_ECITY, resolution=DEFAULT_RESOLUTION,
                 layer_floor=0.0, layer_height=5.0):
        """anchor: (lat, lng) in degrees for local (0, 0). resolution: H3 resolution for cells.
        Altitude layers: layer k covers [layer_floor + k * layer_height, + layer_height)."""
        self.lat0, self.lng0 = float(anchor[0]), float(anchor[1])
        self.res = int(resolution)
        self.layer_floor = float(layer_floor)
        self.layer_height = float(layer_height)
        self._m_per_deg_lat = math.radians(1.0) * EARTH_RADIUS_M
        self._m_per_deg_lng = self._m_per_deg_lat * math.cos(math.radians(self.lat0))

    # ------------------------------------------------------------ coordinates
    def to_latlng(self, x, y):
        """Local metres -> (lat, lng) degrees. Works on scalars or numpy arrays."""
        return self.lat0 + np.asarray(y) / self._m_per_deg_lat, self.lng0 + np.asarray(x) / self._m_per_deg_lng

    def to_local(self, lat, lng):
        """(lat, lng) degrees -> local metres (x, y)."""
        return ((np.asarray(lng) - self.lng0) * self._m_per_deg_lng,
                (np.asarray(lat) - self.lat0) * self._m_per_deg_lat)

    # ------------------------------------------------------------ cells
    def cell(self, x, y, res=None):
        """H3 cell id containing the local point (x, y)."""
        lat, lng = self.to_latlng(x, y)
        return h3.latlng_to_cell(float(lat), float(lng), self.res if res is None else res)

    def cells(self, xy, res=None):
        """Cell ids for an (N, 2+) array of local points."""
        xy = np.asarray(xy, dtype=float)
        lat, lng = self.to_latlng(xy[:, 0], xy[:, 1])
        r = self.res if res is None else res
        return [h3.latlng_to_cell(a, b, r) for a, b in zip(lat.tolist(), lng.tolist())]

    def center(self, cell):
        """Local (x, y) of a cell's centre."""
        lat, lng = h3.cell_to_latlng(cell)
        x, y = self.to_local(lat, lng)
        return float(x), float(y)

    def boundary(self, cell):
        """The cell's corners as local (x, y) points (6 for a hexagon)."""
        return [tuple(map(float, self.to_local(lat, lng))) for lat, lng in h3.cell_to_boundary(cell)]

    def disk(self, cell, k):
        """The cell and every cell within k steps of it."""
        return set(h3.grid_disk(cell, k))

    def edge_length(self, res=None):
        """Average hexagon edge length (m) at a resolution."""
        return h3.average_hexagon_edge_length(self.res if res is None else res, 'm')

    def k_for_radius(self, radius, res=None):
        """Smallest disk size k such that every point within `radius` of any point inside a
        cell lies in grid_disk(cell, k). Hexagon centres are sqrt(3) * edge apart; one extra
        ring covers the cell's own extent and the variation of cell size across the globe."""
        e = self.edge_length(res)
        return int(math.ceil(radius / (math.sqrt(3.0) * e * 0.9))) + 1

    # ------------------------------------------------------------ areas
    def cells_in_polygon(self, xy, res=None, cover=False):
        """Cells for a polygon given as local (x, y) corners. cover=False: cells whose centre
        is inside (H3's rule). cover=True: also cells that only partly overlap it (the polygon
        is grown by one cell radius first; exact for convex shapes such as our boxes)."""
        r = self.res if res is None else res
        pts = np.asarray(xy, dtype=float)
        if cover:
            pts = _offset_convex(pts, self.edge_length(r) * 1.05)
        lat, lng = self.to_latlng(pts[:, 0], pts[:, 1])
        poly = h3.LatLngPoly(list(zip(lat.tolist(), lng.tolist())))
        return set(h3.polygon_to_cells(poly, r))

    def cells_for_obstacle(self, ob, res=None, cover=True, segments=16):
        """Cells covering one obstacle in the standard layout format (pillar, or box with
        optional yaw), e.g. to turn buildings or no-fly zones into blocked cells."""
        cx, cy = ob['center']
        if ob['type'] == 'pillar':
            a = np.linspace(0.0, 2.0 * math.pi, segments, endpoint=False)
            # a polygon around the circle (circumscribed, so the circle is fully inside)
            r = ob['radius'] / math.cos(math.pi / segments)
            corners = np.column_stack([cx + r * np.cos(a), cy + r * np.sin(a)])
        else:
            hx, hy = ob['half']
            yaw = ob.get('yaw', 0.0)
            c, s = math.cos(yaw), math.sin(yaw)
            corners = np.array([(cx + c * u - s * v, cy + s * u + c * v)
                                for u, v in ((-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy))])
        found = self.cells_in_polygon(corners, res, cover=cover)
        if not found:                                   # smaller than one cell: its own cell
            found = {self.cell(cx, cy, res)}
        return found

    # ------------------------------------------------------------ 3D
    def layer(self, z):
        """Altitude layer index of a height (m)."""
        return int(math.floor((float(z) - self.layer_floor) / self.layer_height))

    def layer_range(self, k):
        """(bottom, top) heights (m) of layer k."""
        lo = self.layer_floor + k * self.layer_height
        return lo, lo + self.layer_height

    def cell3d(self, x, y, z, res=None):
        """(cell, layer): a 3D airspace volume."""
        return self.cell(x, y, res), self.layer(z)
