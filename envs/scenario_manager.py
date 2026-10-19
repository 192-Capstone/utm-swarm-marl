"""
scenario_manager.py
-------------------
Static obstacle layouts for Stage 2, independent of the simulator. The layout
is a plain list of dicts, so the PyBullet adapter and a Webots adapter can
both build the same world from it, and the environment and the teacher can
reason about it without asking the simulator.

Two obstacle types, both tall enough that drones (flying at 1-5 m) can't go
over them and have to go around:
    pillar : vertical cylinder   {'type': 'pillar', 'center': (x, y), 'radius': r,      'height': h}
    box    : axis-aligned block  {'type': 'box',    'center': (x, y), 'half': (hx, hy), 'height': h}

Placement: obstacles that never stand in anyone's way teach nothing, so a share
of them is placed ON a drone's straight path (between 30% and 70% of the way
from spawn to goal, with a small sideways jitter). The rest are scattered
around the area the drones use. Every obstacle keeps a clearance from all
spawns and goals, and obstacles don't overlap each other.

Config (block 'obstacles'; disabled when missing):
    enabled          true / false
    count            [min, max] obstacles per episode            (3, 8)
    on_path_share    share of obstacles placed on drone paths    (0.6)
    pillar_share     share of pillars vs boxes                   (0.6)
    pillar_radius    [min, max] m                                (0.3, 0.8)
    box_half         [min, max] half-size of a box side, m       (0.4, 1.2)
    height           m                                           (8.0)
    spawn_clearance  min surface distance to any spawn, m        (1.5)
    goal_clearance   min surface distance to any goal, m         (1.5)
    gap              min surface distance between obstacles, m   (1.0)
"""

import numpy as np

DRONE_RADIUS = 0.15     # half-size of the drone body used for collision checks


def obstacle_config(config):
    """The 'obstacles' block with defaults filled in, or None when disabled."""
    o = config.get('obstacles') or {}
    if not o.get('enabled', False):
        return None
    return {
        'count':           tuple(o.get('count', (3, 8))),
        'on_path_share':   float(o.get('on_path_share', 0.6)),
        'pillar_share':    float(o.get('pillar_share', 0.6)),
        'pillar_radius':   tuple(o.get('pillar_radius', (0.3, 0.8))),
        'box_half':        tuple(o.get('box_half', (0.4, 1.2))),
        'height':          float(o.get('height', 8.0)),
        'spawn_clearance': float(o.get('spawn_clearance', 1.5)),
        'goal_clearance':  float(o.get('goal_clearance', 1.5)),
        'gap':             float(o.get('gap', 1.0)),
    }


# --------------------------------------------------------------- geometry ---
def surface_distance(ob, point):
    """Horizontal distance from a point to the obstacle's surface (negative
    inside). Height is handled separately (obstacles start at the ground)."""
    px, py = point[0] - ob['center'][0], point[1] - ob['center'][1]
    if ob['type'] == 'pillar':
        return float(np.hypot(px, py) - ob['radius'])
    hx, hy = ob['half']
    dx, dy = abs(px) - hx, abs(py) - hy
    outside = np.hypot(max(dx, 0.0), max(dy, 0.0))
    inside = min(max(dx, dy), 0.0)
    return float(outside + inside)


def footprint_radius(ob):
    """Radius of a circle that contains the obstacle's footprint."""
    return ob['radius'] if ob['type'] == 'pillar' else float(np.hypot(*ob['half']))


def distance_to_obstacles(layout, point):
    """Smallest 3D clearance from a point to any obstacle (inf when none).
    Above an obstacle's top, the vertical gap counts too."""
    best = np.inf
    for ob in layout:
        horiz = surface_distance(ob, point)
        above = point[2] - ob['height']
        if above > 0:
            d = np.hypot(max(horiz, 0.0), above)
        else:
            d = horiz
        best = min(best, d)
    return float(best)


def collides(layout, point, drone_radius=DRONE_RADIUS):
    """True when a drone body centred at point touches any obstacle."""
    return distance_to_obstacles(layout, point) < drone_radius


# --------------------------------------------------------------- layouts ---
def _random_shape(rng, cfg):
    if rng.uniform() < cfg['pillar_share']:
        return {'type': 'pillar', 'radius': float(rng.uniform(*cfg['pillar_radius']))}
    return {'type': 'box', 'half': (float(rng.uniform(*cfg['box_half'])),
                                    float(rng.uniform(*cfg['box_half'])))}


def generate_layout(rng, spawns, goals, cfg, tries_per_obstacle=60):
    """Place obstacles for one episode.

    rng    : np.random.RandomState or the np.random module
    spawns, goals : {'drone_i': [x, y, z]}
    cfg    : output of obstacle_config()
    Returns a list of obstacle dicts (may hold fewer than requested if the
    area is too crowded to keep every clearance).
    """
    keys = sorted(spawns)
    S = np.array([spawns[k] for k in keys], dtype=float)
    G = np.array([goals[k] for k in keys], dtype=float)
    lo = np.minimum(S.min(0), G.min(0))[:2] - 3.0
    hi = np.maximum(S.max(0), G.max(0))[:2] + 3.0
    n_target = int(rng.randint(cfg['count'][0], cfg['count'][1] + 1))
    layout = []

    def fits(ob):
        for p in S:
            if surface_distance(ob, p) < cfg['spawn_clearance']:
                return False
        for p in G:
            if surface_distance(ob, p) < cfg['goal_clearance']:
                return False
        for other in layout:
            d = np.hypot(*(np.subtract(ob['center'], other['center'])))
            if d - footprint_radius(ob) - footprint_radius(other) < cfg['gap']:
                return False
        return True

    for k in range(n_target):
        want_on_path = rng.uniform() < cfg['on_path_share']
        for attempt in range(tries_per_obstacle):
            ob = _random_shape(rng, cfg)
            ob['height'] = cfg['height']
            # If no on-path spot fits within half the tries (short paths in the
            # 2 m random task, or the hovering drone in fly-through), place the
            # obstacle elsewhere instead of dropping it.
            on_path = want_on_path and attempt < tries_per_obstacle // 2
            if on_path:
                i = int(rng.randint(len(keys)))
                seg = G[i, :2] - S[i, :2]
                length = np.linalg.norm(seg)
                if length < 1e-6:
                    continue                               # hovering drone: no path to block
                along = rng.uniform(0.3, 0.7)
                side = np.array([-seg[1], seg[0]]) / length
                c = S[i, :2] + along * seg + side * rng.uniform(-0.4, 0.4)
            else:
                c = np.array([rng.uniform(lo[0], hi[0]), rng.uniform(lo[1], hi[1])])
            ob['center'] = (float(c[0]), float(c[1]))
            if fits(ob):
                ob['on_path'] = bool(on_path)
                layout.append(ob)
                break
    return layout
