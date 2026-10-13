"""
lidar.py
--------
The 14-ray LiDAR layout, shared by every simulator adapter (PyBullet now,
Webots later) so all of them produce the same readings.

Ray order (matches ObservationProcessor.lidar_quadrants):
     0 front    1 back     2 left     3 right    4 up    5 down
     6 front-right, tilted up 45 deg     10 front-right, tilted down 45 deg
     7 back-left,   tilted up 45 deg     11 back-left,   tilted down 45 deg
     8 front-left,  tilted up 45 deg     12 front-left,  tilted down 45 deg
     9 back-right,  tilted up 45 deg     13 back-right,  tilted down 45 deg

"Front" is the direction from the drone to its goal (the fan is rotated to
that heading), not a fixed world axis. With world-fixed rays there are 45-deg
gaps, and an obstacle straight ahead on a diagonal path can sit in a gap and
stay invisible until it is very close; pointing ray 0 at the goal means the
ray that matters most always checks the path the drone is about to fly.

Raw readings are distances in metres, equal to the range when nothing is hit.
ObservationProcessor turns them into proximity = 1 - distance / range
(0 = nothing seen, 1 = touching).

Config (block 'lidar'):
    enabled      false -> every ray reports "nothing seen" (Stage-1 behaviour)
    range        metres (6.0); keep it the same across stages
    sees_drones  other drones block rays too (true)
"""

import numpy as np

# (azimuth from front, counter-clockwise towards the left; elevation), degrees
RAYS = [(0, 0), (180, 0), (90, 0), (-90, 0), (0, 90), (0, -90),
        (-45, 45), (135, 45), (45, 45), (-135, 45),
        (-45, -45), (135, -45), (45, -45), (-135, -45)]
N_RAYS = len(RAYS)


def lidar_config(config):
    l = config.get('lidar') or {}
    return {
        'enabled':     bool(l.get('enabled', False)),
        'range':       float(l.get('range', 6.0)),
        'sees_drones': bool(l.get('sees_drones', True)),
    }


def heading_to_goal(position, goal, fallback=0.0):
    """Yaw (rad) of the horizontal direction from position to goal; fallback
    when the drone is (almost) exactly above or on its goal."""
    dx, dy = goal[0] - position[0], goal[1] - position[1]
    if dx * dx + dy * dy < 1e-4:
        return fallback
    return float(np.arctan2(dy, dx))


def ray_directions(yaw):
    """(14, 3) unit vectors of the fan rotated to the given heading."""
    out = np.empty((N_RAYS, 3))
    for k, (az, el) in enumerate(RAYS):
        a, e = np.radians(az) + yaw, np.radians(el)
        out[k] = (np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e))
    return out


def proximity(distances, max_range):
    """Distances (m) -> proximity in [0, 1]: 0 = nothing within range, 1 = touching.
    A reading of 0 or less means "no data" (the old Stage-1 placeholder was
    14 zeros); a real hit can't be at 0 because rays start outside the drone."""
    d = np.asarray(distances, dtype=np.float32)
    p = np.clip(1.0 - d / max_range, 0.0, 1.0)
    return np.where(d <= 0.0, 0.0, p).astype(np.float32)
