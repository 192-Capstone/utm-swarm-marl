"""
intent.py
---------
E2: intent. Each drone shares its planned path for the next few seconds with the UTM (in its
telemetry report, utm/telemetry.py); the UTM compares the drones' paths to see conflicts coming.

    route_intent(follower, pos, speed, t)      # the drone's side: its next ~10 s along its route
    profile_intent(follower, pos, speed, t, phase, z_pad, ...)   # ... with its climb and landing too
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

profile_intent (E2 follow-up, option D; eval_layered --intent-profile) shares the whole E1 profile,
at the vertical speeds the drone really flies (INTENT_VZ, not E1's booking speed): a climbing drone,
straight up to cruise height, then along its route; a cruising one, along its route and then straight
down onto its pad; a landing one, straight down. route_intent stops at the end of the route at
cruise height, so a drone about to land showed no descent and the UTM pictured it staying up (E2 v2:
city seed 22, f8 landed through f3's layer 1.0 m from it, warned by the booking 4.7 s ahead and not
at all by intent; with D, 6.7 s ahead). Timing needs no correction: an intent is rebuilt at every
report from the drone's real position and speed, so a late or slow drone shares its real timing.

Drones do not use each other's intent (that would be F4, a policy change); E2 is drones -> UTM.
"""

import numpy as np

HORIZON = 10.0                    # s of intent shared / predicted
STEP = 2.0                        # s between intent points
GRID = 0.5                        # s between the points of a predicted path
VERTICAL_SEP = 1.0                # m: half E1's 2 m layer gap
INTENT_VZ = 1.0                   # m/s: the climb and descent the intent assumes. Measured on our runs
                                  # (climb median 1.06 m/s, E6 rooftops; descent ~1 m/s without a landing
                                  # cap, E2 v2 city seed 22) and not E1's booking speed of 0.5, at which
                                  # the intent pictured f8 too high to warn. A landing cap (--land-vz-max)
                                  # is only a limit: when on, it replaces this value for the descent.


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


def profile_intent(follower, pos, speed, t, phase, z_pad, v_cruise, v_up, v_down, horizon=HORIZON, step=STEP,
                   min_speed=0.3):
    """[[t_k, x, y, z], ...] for t_k = t + step .. t + horizon along the drone's E1 profile from where
    it is: phase 'climb' (straight up at v_up to the route's cruise height, then along the route at
    v_cruise), 'cruise' (along the route at its current horizontal speed, at least min_speed) or
    'land' (no more route); after the route, straight down at v_down to z_pad over its end."""
    pos = np.asarray(pos, dtype=float)
    zc = float(follower.w[0, 2])                      # the route's cruise height
    s0 = max(follower.progress, follower._project(pos[:2]))
    v = float(v_cruise) if phase == 'climb' else max(float(speed), min_speed)
    t_up = max(0.0, zc - pos[2]) / v_up if phase == 'climb' else 0.0
    t_route = 0.0 if phase == 'land' else (follower.total - s0) / v
    z_top, end = (pos[2], pos[:2]) if phase == 'land' else (zc, follower.w[-1, :2])
    out = []
    for k in range(1, int(round(horizon / step)) + 1):
        u = k * step
        if u < t_up:
            p = (pos[0], pos[1], pos[2] + v_up * u)
        elif u < t_up + t_route:
            p = follower.point_at(s0 + v * (u - t_up))
        else:
            p = (end[0], end[1], max(float(z_pad), z_top - v_down * (u - t_up - t_route)))
        out.append([t + u, float(p[0]), float(p[1]), float(p[2])])
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
