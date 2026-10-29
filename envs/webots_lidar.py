"""
webots_lidar.py
---------------
C3: an optional Webots Lidar on every drone of webots_l2_multi, to check the analytic LiDAR
(envs/webots_adapter.py scan_lidar) the policy was trained on. Off by default.

    webots.lidar: {enabled: false, use_for_policy: false, horizontal_res: 1440, layers: 5, log: null}

    enabled         every drone gets a Lidar (and an Emitter for its scans) in its extensionSlot and
                    controllerArgs "--lidar"; the supervisor enables its Receiver "lidar_rx" (in the
                    world file, inert when off). Off: drone nodes, commands and devices as before.
    use_for_policy  false ("compare"): the policy still gets the analytic scan, the Lidar scan is only
                    logged next to it; true: the policy gets the Lidar scan
    log             .npz path for the per-step log (both scans, poses), written by close()

The device (step 0, 2026-10-29: probe in Webots R2025a):
    fixed, cylindrical, fieldOfView 2 pi, verticalFieldOfView 3.14159 (3.141593 is rejected),
    numberOfLayers 5 -> layers at +89.875, +45.06, 0, -45.06, -89.875 deg (layer 0 up),
    horizontal_res 1440 -> column c looks at 180 - (c + 0.5) * 0.25 deg (body frame, clockwise),
    near = minRange = 0.3 m (= RAY_START): geometry closer than 0.3 m is clipped, so the drone's own
    body is invisible and what is behind it is seen, as the analytic rays that start 0.3 m out
    (minRange alone would read inf AND hide what is behind), maxRange = lidar.range.
    It sees what is drawn: buildings without bounding objects, the floor, Crazyflies, and
    semi-transparent shapes (transparency 0.45 and 0.6 both read as solid) -> it refuses
    webots.draw_zones (C7 no-fly zone volumes would block it).

Picking the analytic rays (pick_rays): every ray of the fan (14) and ring (32), turned to the goal
heading in the world, is turned into the drone's body frame by the YAW of the scan's own IMU only,
then the nearest layer and column are read. Roll and pitch are not compensated: a body-mounted
Lidar tilts with the drone, and the error that makes is part of what C3 measures. Ground: a hit
whose point (from the full scan pose) is below GROUND_EPS is "nothing seen" (the analytic scan
ignores the ground). inf or beyond range -> range.
"""

import json
import math

import numpy as np

from envs import lidar as lidar_mod
from envs.lidar_msg import LIDAR_CHANNEL, ARG

DEFAULTS = {'enabled': False, 'use_for_policy': False, 'horizontal_res': 1440, 'layers': 5, 'log': None}
VFOV = 3.14159
NEAR = 0.3                      # = webots_adapter.RAY_START
GROUND_EPS = 0.05               # m: a hit below this height is the ground


def lidar_webots_config(webots_cfg):
    """webots.lidar from the config, with defaults; checks the values."""
    lc = dict(DEFAULTS)
    lc.update((webots_cfg or {}).get('lidar') or {})
    unknown = set(lc) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"webots.lidar: unknown keys {sorted(unknown)}")
    lc['enabled'], lc['use_for_policy'] = bool(lc['enabled']), bool(lc['use_for_policy'])
    lc['horizontal_res'], lc['layers'] = int(lc['horizontal_res']), int(lc['layers'])
    if lc['horizontal_res'] < 8 or lc['layers'] < 2:
        raise ValueError('webots.lidar: horizontal_res must be >= 8 and layers >= 2')
    if lc['use_for_policy'] and not lc['enabled']:
        raise ValueError('webots.lidar: use_for_policy needs enabled')
    if lc['enabled'] and (webots_cfg or {}).get('draw_zones'):
        raise ValueError('webots.lidar cannot be used with webots.draw_zones: the Lidar sees the '
                         'semi-transparent zone volumes as solid walls (C3 step 0)')
    return lc


def device_nodes(lc, max_range):
    """The Lidar and its Emitter, for a drone's extensionSlot."""
    return [f'Lidar {{\n      name "lidar"\n      type "fixed"\n      projection "cylindrical"\n'
            f'      fieldOfView 6.283185\n      verticalFieldOfView {VFOV}\n'
            f'      numberOfLayers {lc["layers"]}\n      horizontalResolution {lc["horizontal_res"]}\n'
            f'      near {NEAR}\n      minRange {NEAR}\n      maxRange {float(max_range)}\n    }}',
            f'Emitter {{\n      name "lidar_tx"\n      channel {LIDAR_CHANNEL}\n      range -1\n    }}']


def controller_args():
    return [ARG]


# ---------------------------------------------------------------- rays
def layer_elevations(layers):
    return VFOV / 2 - np.arange(layers) * VFOV / (layers - 1)


def column_azimuths(res):
    return math.pi - (np.arange(res) + 0.5) * 2 * math.pi / res


def rot_rpy(rpy):
    """World-from-body rotation, R = Rz(yaw) Ry(pitch) Rx(roll) (as webots_adapter._euler_from_matrix)."""
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def ray_index(dirs, yaw, layers, res):
    """World ray directions (R, 3) -> (layer, column) of the nearest Lidar ray, the body turned
    by yaw only."""
    az = np.arctan2(dirs[:, 1], dirs[:, 0]) - yaw
    el = np.arcsin(np.clip(dirs[:, 2], -1.0, 1.0))
    col = np.floor(np.mod(math.pi - az, 2 * math.pi) / (2 * math.pi / res)).astype(int) % res
    lay = np.clip(np.rint((VFOV / 2 - el) / (VFOV / (layers - 1))).astype(int), 0, layers - 1)
    return lay, col


def pick_rays(img, dirs, pos, rpy, max_range):
    """img: (layers, res) range image; dirs: (R, 3) world directions of the analytic rays;
    pos, rpy: the scan's pose. Returns (R,) distances with the analytic conventions."""
    layers, res = img.shape
    lay, col = ray_index(dirs, rpy[2], layers, res)
    d = img[lay, col].astype(np.float64)
    el, az = layer_elevations(layers)[lay], column_azimuths(res)[col]
    body = np.stack([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)], axis=1)
    wz = body @ rot_rpy(rpy)[2]
    with np.errstate(invalid='ignore'):
        ground = np.isfinite(d) & (pos[2] + d * wz < GROUND_EPS)
    ok = np.isfinite(d) & ~ground & (d <= max_range)
    return np.where(ok, d, max_range)


def analytic_dirs(heading, n_ring):
    d = lidar_mod.ray_directions(heading)
    if n_ring:
        d = np.vstack([d, lidar_mod.ring_directions(heading, n_ring)])
    return d


# ---------------------------------------------------------------- log
class ScanLog:
    """Rows of (episode, control step, drone): both scans and the poses; written to .npz."""

    FIELDS = ('ep', 'k', 'drone', 't_state', 't_scan', 'pos', 'orn', 'heading', 'scan_pos', 'scan_rpy',
              'lidar', 'analytic', 'all_pos', 'all_yaw', 'cpu_send', 'parked')

    def __init__(self, path):
        self.path = path
        self.rows = {f: [] for f in self.FIELDS}
        self.layouts = {}                       # ep -> layout
        self.stats = {'missing': 0, 'stale': 0, 'wrong_time': 0, 'recv_s': 0.0, 'pick_s': 0.0,
                      'scans': 0, 'bytes': 0}

    def add(self, **kw):
        for f in self.FIELDS:
            self.rows[f].append(kw[f])

    def save(self):
        if not self.path:
            return
        arr = {f: np.asarray(v) for f, v in self.rows.items()}
        arr['layouts_json'] = np.array(json.dumps({str(k): v for k, v in self.layouts.items()}))
        arr['stats_json'] = np.array(json.dumps(self.stats))
        np.savez_compressed(self.path, **arr)
