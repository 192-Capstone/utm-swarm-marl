"""
route_planner.py
----------------
Shared global route planner for Stage 3 (task D7), used by D1 (procedural cities), D3 (OSM map)
and, later, the environment (carrot targets for the policy).

    grid = build_grid(layout, bounds=((-25, 25), (-25, 25)), res=0.25, inflate=0.5)
    path = plan(grid, start_xy, goal_xy)          # list of (x, y) cell centres, or None
    pts  = simplify(grid, path)                   # a few waypoints, every segment collision-free
    L    = path_length(pts)                       # metres
    follower = RouteFollower(pts3d, lookahead=4.5)
    target = follower.target(drone_xyz)           # where the drone should head right now

Obstacles use the standard layout format (envs/scenario_manager.py): pillars and boxes, boxes with
an optional 'yaw' (radians, counter-clockwise). A cell is blocked when a drone centre there would
be closer than `inflate` to an obstacle surface. Only obstacles taller than `min_height` count
(for flying above low buildings later; 0 = every obstacle).
"""

import heapq
import math

import numpy as np

SQRT2 = math.sqrt(2.0)


class Grid:
    """Occupancy grid over a rectangle. occ[iy, ix] is True when blocked.
    Cell (iy, ix) has its centre at (xs[ix], ys[iy])."""

    def __init__(self, occ, xs, ys, res):
        self.occ, self.xs, self.ys, self.res = occ, xs, ys, float(res)

    @property
    def shape(self):
        return self.occ.shape

    def to_cell(self, xy):
        ix = int(round((xy[0] - self.xs[0]) / self.res))
        iy = int(round((xy[1] - self.ys[0]) / self.res))
        return (min(max(iy, 0), len(self.ys) - 1), min(max(ix, 0), len(self.xs) - 1))

    def to_world(self, cell):
        iy, ix = cell
        return (float(self.xs[ix]), float(self.ys[iy]))

    def inside(self, xy):
        h = self.res / 2
        return self.xs[0] - h <= xy[0] <= self.xs[-1] + h and self.ys[0] - h <= xy[1] <= self.ys[-1] + h

    def free(self, cell):
        return not self.occ[cell]


def _surface_distance_grid(ob, gx, gy):
    """Horizontal distance from every grid point to one obstacle surface (negative inside).
    Same maths as scenario_manager.surface_distance, vectorised."""
    px, py = gx - ob['center'][0], gy - ob['center'][1]
    yaw = ob.get('yaw', 0.0)
    if yaw:
        c, s = math.cos(yaw), math.sin(yaw)
        px, py = c * px + s * py, -s * px + c * py
    if ob['type'] == 'pillar':
        return np.hypot(px, py) - ob['radius']
    dx, dy = np.abs(px) - ob['half'][0], np.abs(py) - ob['half'][1]
    return np.hypot(np.maximum(dx, 0), np.maximum(dy, 0)) + np.minimum(np.maximum(dx, dy), 0)


def _footprint_radius(ob):
    return ob['radius'] if ob['type'] == 'pillar' else math.hypot(*ob['half'])


def build_grid(layout, bounds, res=0.25, inflate=0.5, min_height=0.0):
    """bounds = ((xmin, xmax), (ymin, ymax)). Each obstacle only updates the cells inside its own
    bounding square (+ inflate), so big maps stay fast."""
    (x0, x1), (y0, y1) = bounds
    xs = np.arange(x0 + res / 2, x1, res)
    ys = np.arange(y0 + res / 2, y1, res)
    occ = np.zeros((len(ys), len(xs)), dtype=bool)
    for ob in layout:
        if ob.get('height', np.inf) <= min_height:
            continue
        r = _footprint_radius(ob) + inflate
        cx, cy = ob['center']
        ix0, ix1 = np.searchsorted(xs, cx - r), np.searchsorted(xs, cx + r, side='right')
        iy0, iy1 = np.searchsorted(ys, cy - r), np.searchsorted(ys, cy + r, side='right')
        if ix0 >= ix1 or iy0 >= iy1:
            continue
        gx, gy = np.meshgrid(xs[ix0:ix1], ys[iy0:iy1])
        occ[iy0:iy1, ix0:ix1] |= _surface_distance_grid(ob, gx, gy) < inflate
    return Grid(occ, xs, ys, res)


def snap(grid, xy, max_dist=1.0):
    """The free cell nearest to xy within max_dist metres, or None."""
    c = grid.to_cell(xy)
    if grid.free(c):
        return c
    k = int(math.ceil(max_dist / grid.res))
    iy, ix = c
    best, bd = None, np.inf
    for jy in range(max(0, iy - k), min(grid.shape[0], iy + k + 1)):
        for jx in range(max(0, ix - k), min(grid.shape[1], ix + k + 1)):
            if not grid.occ[jy, jx]:
                d = math.hypot(grid.xs[jx] - xy[0], grid.ys[jy] - xy[1])
                if d < bd and d <= max_dist:
                    best, bd = (jy, jx), d
    return best


STEPS = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
         (-1, -1, SQRT2), (-1, 1, SQRT2), (1, -1, SQRT2), (1, 1, SQRT2)]


def astar(grid, start, goal):
    """8-connected A* on cells, no corner cutting (a diagonal step needs both side cells free).
    Returns a list of cells from start to goal, or None."""
    occ, (ny, nx) = grid.occ, grid.shape
    if occ[start] or occ[goal]:
        return None

    def h(c):                                   # octile distance: exact on an empty 8-connected grid
        dy, dx = abs(c[0] - goal[0]), abs(c[1] - goal[1])
        return (dx + dy) + (SQRT2 - 2) * min(dx, dy)

    g = {start: 0.0}
    parent = {start: None}
    heap = [(h(start), 0, start)]
    tie = 0
    closed = set()
    while heap:
        _, _, cur = heapq.heappop(heap)
        if cur in closed:
            continue
        if cur == goal:
            path = []
            while cur is not None:
                path.append(cur)
                cur = parent[cur]
            return path[::-1]
        closed.add(cur)
        cy, cx = cur
        for dy, dx, w in STEPS:
            ny_, nx_ = cy + dy, cx + dx
            if not (0 <= ny_ < ny and 0 <= nx_ < nx) or occ[ny_, nx_]:
                continue
            if dy and dx and (occ[cy + dy, cx] or occ[cy, cx + dx]):
                continue                        # no squeezing between two blocked cells
            nxt = (ny_, nx_)
            cand = g[cur] + w
            if cand < g.get(nxt, np.inf):
                g[nxt] = cand
                parent[nxt] = cur
                tie += 1
                heapq.heappush(heap, (cand + h(nxt), tie, nxt))
    return None


def segment_free(grid, a, b):
    """True when the straight segment a -> b (world xy) crosses only free cells.
    Samples the segment every res/4."""
    n = max(2, int(math.ceil(math.hypot(b[0] - a[0], b[1] - a[1]) / (grid.res / 4))) + 1)
    for t in np.linspace(0.0, 1.0, n):
        if not grid.free(grid.to_cell((a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])))):
            return False
    return True


def simplify(grid, points):
    """Greedy line-of-sight shortcutting: from each kept point, jump to the farthest later point
    that is still visible. Keeps the first and last point."""
    if not points or len(points) <= 2:
        return list(points or [])
    out = [points[0]]
    i = 0
    while i < len(points) - 1:
        j = len(points) - 1
        while j > i + 1 and not segment_free(grid, points[i], points[j]):
            j -= 1
        out.append(points[j])
        i = j
    return out


def path_length(points):
    return float(sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(points, points[1:])))


def plan(grid, start_xy, goal_xy, snap_dist=1.0):
    """World-coordinate path from start to goal through free cells: [start, cell centres..., goal],
    or None when no path exists (or start/goal can't be snapped to a free cell)."""
    s, g = snap(grid, start_xy, snap_dist), snap(grid, goal_xy, snap_dist)
    if s is None or g is None:
        return None
    cells = astar(grid, s, g)
    if cells is None:
        return None
    return [tuple(map(float, start_xy[:2]))] + [grid.to_world(c) for c in cells[1:-1]] + \
           [tuple(map(float, goal_xy[:2]))]


class RouteFollower:
    """Carrot following: the target is `lookahead` metres further along the route than the drone's
    projection onto it, never moving backwards; within `lookahead` of the end the target is the goal.
    waypoints: list of (x, y, z)."""

    def __init__(self, waypoints, lookahead=4.5):
        self.w = np.asarray(waypoints, dtype=float)
        seg = np.diff(self.w[:, :2], axis=0)
        self.seg_len = np.linalg.norm(seg, axis=1)
        self.cum = np.concatenate([[0.0], np.cumsum(self.seg_len)])
        self.total = float(self.cum[-1])
        self.lookahead = float(lookahead)
        self.progress = 0.0                     # arc length already reached (monotone)

    def _project(self, xy):
        best_s, best_d = 0.0, np.inf
        for k, L in enumerate(self.seg_len):
            if L < 1e-9:
                continue
            a = self.w[k, :2]
            t = float(np.clip(np.dot(xy - a, self.w[k + 1, :2] - a) / L ** 2, 0.0, 1.0))
            d = float(np.linalg.norm(a + t * (self.w[k + 1, :2] - a) - xy))
            if d < best_d:
                best_d, best_s = d, self.cum[k] + t * L
        return best_s

    def point_at(self, s):
        s = float(np.clip(s, 0.0, self.total))
        k = int(np.searchsorted(self.cum, s, side='right') - 1)
        k = min(k, len(self.seg_len) - 1)
        L = self.seg_len[k]
        t = 0.0 if L < 1e-9 else (s - self.cum[k]) / L
        return self.w[k] + t * (self.w[k + 1] - self.w[k])

    def target(self, pos):
        self.progress = max(self.progress, self._project(np.asarray(pos, dtype=float)[:2]))
        s = self.progress + self.lookahead
        if s >= self.total:
            return self.w[-1].copy()
        return self.point_at(s)

    def visible_target(self, pos, grid, min_dist=1.0, step=0.25):
        """target(), pulled back along the route until the straight line from the drone to it is
        free on `grid` (the carrot behind a corner would steer the drone into the building), but
        never closer than min_dist to the drone (a goal that close would look like arrival).
        If nothing between the two is visible, the first point along the route at least min_dist
        away (in a straight line)."""
        pos = np.asarray(pos, dtype=float)
        far = self.target(pos)
        s_far = self.total if np.allclose(far, self.w[-1]) else self.progress + self.lookahead
        s = s_far
        while s > self.progress:
            p = self.point_at(s)
            if np.hypot(*(p[:2] - pos[:2])) < min_dist:
                break
            if segment_free(grid, pos[:2], p[:2]):
                return p
            s -= step
        s = self.progress + min_dist                    # straight-line distance, also at a bend
        while s < self.total and np.hypot(*(self.point_at(s)[:2] - pos[:2])) < min_dist:
            s += step
        return self.point_at(min(s, self.total))
