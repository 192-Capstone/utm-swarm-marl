"""
city_generator.py
-----------------
Procedural city districts for Stage 3, in the same obstacle layout format as
scenario_manager.py (axis-aligned 'box' dicts), so the PyBullet adapter, a
Webots adapter, the teacher and the evaluation can use them unchanged.

A district is a street grid: blocks of varied size separated by streets, with
a wider avenue every few blocks. Each block is split into lots (1, 2 or 4) with
small setbacks and alleys between buildings, some lots stay open (parks), and
some blocks are courtyards: a ring of buildings with one narrow entrance, the
kind of dead-end pocket a purely reactive controller can get trapped in.

Scale is drone-sized (50 x 50 m district, streets 2.5-7 m, blocks 5-12 m): our drones are 0.3 m,
fly 6-20 m trips at 1-5 m altitude and see 6-10 m with the LiDAR. Buildings are
6-20 m tall, so drones must go around them. Real-map districts are D3.

Levels (DIFFICULTY): easy (wide streets, sparse), medium, hard (narrow streets,
dense, more courtyards and alleys). Any field can be overridden.

    city = generate_city(np.random.RandomState(0), level='medium')
    spawns, goals = sample_flights(np.random.RandomState(1), city, n=4)
    # city['layout'] is the obstacle list for SwarmEnv.start_episode(spawns, goals, obstacles=...)

Flights start and end in streets, clear of buildings, and are only accepted
when a path exists on an occupancy grid (inflated by the drone radius plus a
margin), so no episode is impossible.
"""

import numpy as np

from envs.route_planner import build_grid, plan, path_length
from envs.scenario_manager import DRONE_RADIUS, surface_distance

DIFFICULTY = {
    #                  street width  block size   avenue every / width x  park   courtyard  alley   lots (1,2,4)
    'easy':   dict(street=(5.0, 7.0), block=(8.0, 12.0), avenue_every=3, avenue_scale=1.4, park=0.30,
                   courtyard=0.00, alley=(2.5, 3.0), lots=(0.6, 0.3, 0.1)),
    'medium': dict(street=(3.5, 5.0), block=(6.0, 10.0), avenue_every=3, avenue_scale=1.4, park=0.15,
                   courtyard=0.15, alley=(1.8, 2.5), lots=(0.35, 0.4, 0.25)),
    'hard':   dict(street=(2.5, 3.5), block=(5.0, 8.0), avenue_every=4, avenue_scale=1.3, park=0.05,
                   courtyard=0.30, alley=(1.4, 2.0), lots=(0.2, 0.4, 0.4)),
}
COMMON = dict(size=(50.0, 50.0), height=(6.0, 20.0), setback=(0.0, 0.4), wall=(0.6, 1.0),
              entrance=(1.6, 2.4), grid_res=0.25, route_margin=0.35)


def _cuts(rng, length, cfg):
    """Alternating street / block intervals along one axis, starting and ending with a street.
    Returns (blocks, streets) as lists of (lo, hi), coordinates centred on 0."""
    # as many blocks as fit on average, then everything scaled by a small factor to fill the length
    s_mean, b_mean = np.mean(cfg['street']), np.mean(cfg['block'])
    n = max(1, int(round((length - s_mean) / (b_mean + s_mean))))
    sizes = [rng.uniform(*cfg['block']) for _ in range(n)]
    widths = [rng.uniform(*cfg['street']) * (cfg['avenue_scale'] if k % cfg['avenue_every'] == 0 and 0 < k < n else 1.0)
              for k in range(n + 1)]
    f = length / (sum(sizes) + sum(widths))
    sizes, widths = [b * f for b in sizes], [w * f for w in widths]
    pos, blocks, streets = -length / 2, [], []
    for k, w in enumerate(widths):
        streets.append((pos, pos + w))
        pos += w
        if k < len(sizes):
            blocks.append((pos, pos + sizes[k]))
            pos += sizes[k]
    return blocks, streets


def _box(x0, x1, y0, y1, h):
    return {'type': 'box', 'center': (float((x0 + x1) / 2), float((y0 + y1) / 2)),
            'half': (float((x1 - x0) / 2), float((y1 - y0) / 2)), 'height': float(h)}


def _fill_block(rng, x0, x1, y0, y1, cfg, out):
    """Buildings for one block (appended to out)."""
    h = lambda: rng.uniform(*cfg['height'])
    if rng.uniform() < cfg['park']:
        return
    sb = rng.uniform(*cfg['setback'])
    x0, x1, y0, y1 = x0 + sb, x1 - sb, y0 + sb, y1 - sb
    if rng.uniform() < cfg['courtyard'] and min(x1 - x0, y1 - y0) > 4.5:
        # ring of 4 walls with one entrance on a random side
        t = rng.uniform(*cfg['wall'])
        e = rng.uniform(*cfg['entrance'])
        side = rng.randint(4)
        hh = h()
        walls = {0: (x0, x1, y0, y0 + t), 1: (x0, x1, y1 - t, y1),      # south, north
                 2: (x0, x0 + t, y0 + t, y1 - t), 3: (x1 - t, x1, y0 + t, y1 - t)}   # west, east
        for s, (a, b, c, d) in walls.items():
            if s != side:
                out.append(_box(a, b, c, d, hh))
                continue
            if s < 2:                                   # split a horizontal wall around the entrance
                m = rng.uniform(a + 0.3 * (b - a), b - 0.3 * (b - a))
                for lo, hi in ((a, m - e / 2), (m + e / 2, b)):
                    if hi - lo > 0.3:
                        out.append(_box(lo, hi, c, d, hh))
            else:
                m = rng.uniform(c + 0.3 * (d - c), d - 0.3 * (d - c))
                for lo, hi in ((c, m - e / 2), (m + e / 2, d)):
                    if hi - lo > 0.3:
                        out.append(_box(a, b, lo, hi, hh))
        return
    n = rng.choice([1, 2, 4], p=cfg['lots'])
    a = rng.uniform(*cfg['alley'])
    if n == 1:
        out.append(_box(x0, x1, y0, y1, h()))
    elif n == 2:
        if (x1 - x0) >= (y1 - y0):
            m = rng.uniform(x0 + 0.4 * (x1 - x0), x1 - 0.4 * (x1 - x0))
            out += [_box(x0, m - a / 2, y0, y1, h()), _box(m + a / 2, x1, y0, y1, h())]
        else:
            m = rng.uniform(y0 + 0.4 * (y1 - y0), y1 - 0.4 * (y1 - y0))
            out += [_box(x0, x1, y0, m - a / 2, h()), _box(x0, x1, m + a / 2, y1, h())]
    else:
        mx = rng.uniform(x0 + 0.4 * (x1 - x0), x1 - 0.4 * (x1 - x0))
        my = rng.uniform(y0 + 0.4 * (y1 - y0), y1 - 0.4 * (y1 - y0))
        for xa, xb in ((x0, mx - a / 2), (mx + a / 2, x1)):
            for ya, yb in ((y0, my - a / 2), (my + a / 2, y1)):
                if xb - xa > 0.8 and yb - ya > 0.8:
                    out.append(_box(xa, xb, ya, yb, h()))

def route_length(city, a, b):
    """Shortest path length in metres, or None if unreachable."""
    grid = city.get('grid')
    if grid is None:
        # Support city dictionaries created before the shared grid was added.
        W, H = city['size']
        grid = build_grid(
            city['layout'],
            ((-W / 2, W / 2), (-H / 2, H / 2)),
            city['res'],
            DRONE_RADIUS + COMMON['route_margin'],
        )

    path = plan(grid, a, b, snap_dist=0.0)
    return None if path is None else path_length(path)

def generate_city(rng, level='medium', **overrides):
    """Generate a city and its reusable shared-planner grid.

    Retains the existing city dictionary keys and adds ``grid`` for route queries.
    ``occ`` marks cells a drone centre cannot use, inflated by drone radius + margin.
    """
    cfg = dict(COMMON, **DIFFICULTY[level], **overrides)
    W, H = cfg['size']
    bx, sx = _cuts(rng, W, cfg)
    by, sy = _cuts(rng, H, cfg)
    layout = []
    for x0, x1 in bx:
        for y0, y1 in by:
            _fill_block(rng, x0, x1, y0, y1, cfg, layout)
    grid = build_grid(
        layout,
        ((-W / 2, W / 2), (-H / 2, H / 2)),
        cfg['grid_res'],
        DRONE_RADIUS + cfg['route_margin'],
    )
    occ, xs, ys = grid.occ, grid.xs, grid.ys
    return {
        'layout': layout,
        'streets_x': sx,
        'streets_y': sy,
        'blocks_x': bx,
        'blocks_y': by,
        'size': (W, H),
        'level': level,
        'occ': occ,
        'xs': xs,
        'ys': ys,
        'res': cfg['grid_res'],
        'grid': grid,
    }


def street_point(rng, city, clearance=1.0, edge=1.0, tries=200):
    """A random point in a street, at least `clearance` m from every building and `edge` m
    inside the district."""
    W, H = city['size']
    for _ in range(tries):
        if rng.uniform() < 0.5:
            a, b = city['streets_x'][rng.randint(len(city['streets_x']))]
            p = (rng.uniform(a, b), rng.uniform(-H / 2, H / 2))
        else:
            a, b = city['streets_y'][rng.randint(len(city['streets_y']))]
            p = (rng.uniform(-W / 2, W / 2), rng.uniform(a, b))
        if abs(p[0]) > W / 2 - edge or abs(p[1]) > H / 2 - edge:
            continue
        if all(surface_distance(ob, p) >= clearance for ob in city['layout']):
            return p
    raise RuntimeError('no street point with that clearance')


def sample_flights(rng, city, n, dist=(8.0, 20.0), clearance=1.0, separation=2.0, z=(2.0, 3.0), tries=400):
    """n drones: spawns and goals in streets, start-goal straight distance within `dist`, spawns
    (and goals) `separation` m apart, and a free path for every flight. Returns (spawns, goals)."""
    spawns, goals = {}, {}
    for i in range(n):
        for _ in range(tries):
            a = street_point(rng, city, clearance)
            b = street_point(rng, city, clearance)
            d = float(np.hypot(a[0] - b[0], a[1] - b[1]))
            if not dist[0] <= d <= dist[1]:
                continue
            if any(np.hypot(a[0] - s[0], a[1] - s[1]) < separation for s in spawns.values()) or \
                    any(np.hypot(b[0] - g[0], b[1] - g[1]) < separation for g in goals.values()):
                continue
            if route_length(city, a, b) is None:
                continue
            h = float(rng.uniform(*z))
            spawns[f'drone_{i}'] = [float(a[0]), float(a[1]), h]
            goals[f'drone_{i}'] = [float(b[0]), float(b[1]), h]
            break
        else:
            raise RuntimeError(f'could not place drone {i}')
    return spawns, goals


def city_scenario(level):
    """A scenario function with the COURSES signature: (rng, n_agents) -> spawns, goals, layout."""
    def make(rng, n_agents):
        city = generate_city(rng, level)
        spawns, goals = sample_flights(rng, city, n_agents)
        return spawns, goals, city['layout']
    make.__name__ = f'city_{level}'
    return make


CITIES = {f'city_{lv}': city_scenario(lv) for lv in DIFFICULTY}
