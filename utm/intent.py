"""
intent.py
---------
E2: intent. Each drone shares its planned path for the next few seconds with the UTM (in its
telemetry report, utm/telemetry.py); the UTM compares the drones' paths to see conflicts coming.

    route_intent(follower, pos, speed, t)      # the drone's side: its next ~10 s along its route
    predicted_path(report, now)                # the UTM's side: (times, positions) for [now, now + h]
    closest_approach(path_a, path_b)           # (min distance, time) of two predicted paths
    first_conflict(path_a, path_b, radius, vertical_sep)   # first time they get too close, or None

A drone's intent: where it will be along its own route (envs/route_planner.RouteFollower) if it
keeps its current speed, every `step` s for `horizon` s, at cruise height: so a drone shares no
intent while it climbs from its pad or lands (the caller sends [] then), and the UTM predicts it
from its position and velocity instead, which follow the vertical motion. The UTM uses the intent
while it covers the time asked for, and past its last point continues at the last segment's
velocity (a late report); with no intent (an intruder, a climbing / landing drone) or an intent
that is already over, it extrapolates the last reported position at the last reported velocity.

A conflict is judged as E1 separates drones: closer than `radius` horizontally AND closer than
`vertical_sep` vertically (default half E1's layer gap: 1 m), a flat disc, not a ball. Drones on
different layers (2 m apart) crossing one above the other are separated by design, not a conflict.
Downwash is not modelled: at Crazyflie scale it reaches ~0.3 m below a drone, well inside 1 m;
larger drones would need a larger, vertically elongated separation.

Drones do not use each other's intent (that would be F4, a policy change); E2 is drones -> UTM.
"""

import numpy as np

HORIZON = 10.0                    # s of intent shared / predicted
STEP = 2.0                        # s between intent points
GRID = 0.5                        # s between the points of a predicted path
VERTICAL_SEP = 1.0                # m: half E1's 2 m layer gap


def route_intent(follower, pos, speed, t, horizon=HORIZON, step=STEP, min_speed=0.3):
    """[[t_k, x, y, z], ...] for t_k = t + step .. t + horizon: the drone's position along its
    route at its current horizontal speed (at least min_speed: a drone about to move off)."""
    s0 = max(follower.progress, follower._project(np.asarray(pos, dtype=float)[:2]))
    v = max(float(speed), min_speed)
    out = []
    for k in range(1, int(round(horizon / step)) + 1):
        p = follower.point_at(min(s0 + v * k * step, follower.total))
        out.append([t + k * step, float(p[0]), float(p[1]), float(p[2])])
    return out


def predicted_path(report, now, horizon=HORIZON, grid=GRID):
    """(times, positions [k, 3]) of the drone from now to now + horizon, every grid s."""
    times = now + np.arange(0.0, horizon + 1e-9, grid)
    pos0, t0 = np.asarray(report.pos, dtype=float), float(report.t_sent)
    if report.intent and report.intent[-1][0] >= now:
        kt = np.array([t0] + [q[0] for q in report.intent])
        kp = np.vstack([pos0, np.array([q[1:] for q in report.intent])])
        path = np.column_stack([np.interp(times, kt, kp[:, j]) for j in range(3)])
        after = times > kt[-1]                    # past the intent (a late report): keep going
        if after.any():
            v = (kp[-1] - kp[-2]) / max(kt[-1] - kt[-2], 1e-9)
            path[after] = kp[-1] + np.outer(times[after] - kt[-1], v)
    else:
        path = pos0 + np.outer(times - t0, np.asarray(report.vel, dtype=float))
    return times, path


def first_conflict(a, b, radius, vertical_sep=VERTICAL_SEP):
    """First time two predicted paths (same time grid) are within `radius` horizontally and
    `vertical_sep` vertically, or None."""
    (ta, pa), (_, pb) = a, b
    d = pa - pb
    hit = (np.hypot(d[:, 0], d[:, 1]) < radius) & (np.abs(d[:, 2]) < vertical_sep)
    return float(ta[np.argmax(hit)]) if hit.any() else None


def closest_approach(a, b):
    """(min distance, time) of two predicted paths on the same time grid."""
    (ta, pa), (_, pb) = a, b
    d = np.linalg.norm(pa - pb, axis=1)
    k = int(np.argmin(d))
    return float(d[k]), float(ta[k])
