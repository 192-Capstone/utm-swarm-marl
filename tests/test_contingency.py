"""
test_contingency.py
-------------------
E4 contingency handling (utm/contingency.py) and the E1 pieces it uses (release, replan from
mid-air, emergency priority), on hand-made plans (nothing is flown).

  A. release cuts a booking at a time; the profile can start mid-air; replan books from the
     drone's position, keeps the next `grace` s unchecked, and keeps the old booking if nothing fits.
  B. Emergencies: a flight still on the ground is filed again (delayed); an airborne one is
     re-planned from its position; zones never give way.
  C. A zone added mid-flight re-plans exactly the flights it clashes with, around the zone.
  D. An intruder gets a protective volume; flights about to meet it are re-planned.
  E. Alert rules: intrusion at once; late only after it lasted `persist` s; early never; cooldown.

Run: python -m pytest tests/test_contingency.py -q
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

pytest.importorskip('h3')

from utm.airspace_zones import Zone, zone_keys  # noqa: E402
from utm.conformance import ConformanceMonitor  # noqa: E402
from utm.contingency import ContingencyManager  # noqa: E402
from utm.strategic_planner import FlightPlan, StrategicPlanner  # noqa: E402

B = ((-25.0, 25.0), (-25.0, 25.0))


def at(planner, d, t):
    """Where decision d plans the drone at absolute time t (x, y, z)."""
    route = [(x, y) for x, y, _ in d.waypoints[1:-1]]
    prof, _ = planner.profile(route, d.layer, d.waypoints[0][2])
    tau = t - d.t_dep
    best = min(prof, key=lambda s: abs(s[0] - tau))
    return best[1:]


def overlap(a, b, after):
    """Keys where bookings a and b overlap in time, counting only from `after` on."""
    out = []
    for k, ivs in a.items():
        for lo, hi in ivs:
            for lo2, hi2 in b.get(k, ()):
                if max(lo, lo2, after) < min(hi, hi2):
                    out.append(k)
    return out


def owned(planner, fid):
    return {k: [(lo, hi) for lo, hi, o in rows if o == fid] for k, rows in planner.ledger.items()
            if any(o == fid for _, _, o in rows)}


# ------------------------------------------------------------------ A. E1 pieces
def test_release_cuts_the_booking_at_t():
    sky = StrategicPlanner([], B)
    sky.submit(FlightPlan('A', (-15, 0), (15, 0)))
    sky.release('A', 8.0)
    ivs = [iv for ivs in owned(sky, 'A').values() for iv in ivs]
    assert ivs and max(hi for _, hi in ivs) <= 8.0


def test_profile_can_start_mid_air():
    sky = StrategicPlanner([], B)
    r = [(-10.0, 0.0), (10.0, 0.0)]
    _, from_pad = sky.profile(r, 1)                       # 1 m -> 4 m: 6 s up, 6 s down
    _, from_2m = sky.profile(r, 1, z_start=2.0)           # 2 m -> 4 m: 4 s up
    assert from_pad - from_2m == pytest.approx(2.0)
    assert sky.profile(r, 0, z_start=2.0)[0][0][3] == pytest.approx(2.0)


def test_replan_from_mid_air_books_from_the_drone_on():
    sky = StrategicPlanner([], B)
    a = sky.submit(FlightPlan('A', (-15, 0), (15, 0)))
    sky.submit(FlightPlan('C', (0, -15), (0, 15), t_earliest=a.t_dep))
    pos = at(sky, a, 6.0)
    d = sky.replan('A', pos, 6.0)
    assert d.status == 'approved' and d.t_dep >= 6.0
    assert d.waypoints[0] == pytest.approx((pos[0], pos[1], pos[2]))
    assert not overlap(owned(sky, 'A'), owned(sky, 'C'), after=6.0 + 2.0)


def test_replan_mid_air_leaves_now():
    """A flying drone can't wait on a pad: a mid-air re-plan never delays its departure."""
    sky = StrategicPlanner([], B)
    a = sky.submit(FlightPlan('A', (-15, 0), (15, 0)))
    for k in range(5):                                     # crowd the air around A's route
        sky.submit(FlightPlan(f'C{k}', (-15 + 6 * k, -15), (-15 + 6 * k, 15), t_earliest=4.0 + k))
    d = sky.replan('A', at(sky, a, 6.0), 6.0)
    assert d.status == 'rejected' or d.t_dep == 6.0


def test_failed_replan_keeps_the_old_booking():
    sky = StrategicPlanner([], B)
    sky.submit(FlightPlan('A', (-15, 0), (15, 0)))
    for pad in ((15, 0), (-15, 0)):                       # neither the goal nor the base is usable
        sky.blocked |= {(c, 0) for c in sky.cells_near(*pad)}
    before = {k: list(v) for k, v in sky.ledger.items()}
    d = sky.replan('A', (-12.0, 0.0, 2.0), 5.0)
    assert d.status == 'rejected' and sky.ledger == before


# ------------------------------------------------------------------ B. emergencies
def test_emergency_on_the_ground_delays_the_normal_flight():
    sky = StrategicPlanner([], B)
    a = sky.submit(FlightPlan('A', (-15, 0), (15, 0)))
    e = sky.submit(FlightPlan('E', (0, -15), (0, 15), priority=1), now=0.0)
    assert e.status == 'approved' and e.delay == 0.0 and e.displaced == []
    a2 = sky.current['A']
    assert a2 is not a and a2.delay > 0
    assert not overlap(owned(sky, 'A'), owned(sky, 'E'), after=-1e9)


def test_emergency_displaces_an_airborne_flight_which_is_replanned():
    sky = StrategicPlanner([], B)
    a = sky.submit(FlightPlan('A', (-15, 0), (15, 0)))
    e4 = ContingencyManager(sky)
    pos = {'A': at(sky, a, 3.0)}
    acts = e4.emergency(FlightPlan('E', (0, -15), (0, 15), t_earliest=3.0, priority=1), 3.0, pos)
    assert acts[0]['decision'].status == 'approved' and acts[0]['decision'].displaced == ['A']
    re = [x for x in acts if x['flight'] == 'A']
    assert re and re[0]['decision'].status == 'approved'
    assert not overlap(owned(sky, 'A'), owned(sky, 'E'), after=3.0 + 2.0)


def test_zones_never_give_way_to_an_emergency():
    sky = StrategicPlanner([], B)
    sky.blocked |= zone_keys(Zone('wall', 'Wall', polygon=[(-2, -25), (2, -25), (2, 25), (-2, 25)]),
                             sky.sky, sky.s.n_layers)
    assert sky.submit(FlightPlan('E', (-15, 0), (15, 0), priority=5)).status == 'rejected'


# ------------------------------------------------------------------ C. zone mid-flight
def test_zone_added_mid_flight_replans_only_the_flights_it_touches():
    sky = StrategicPlanner([], B)
    a = sky.submit(FlightPlan('A', (-15, 0), (15, 0)))
    far = sky.submit(FlightPlan('F', (-15, 18), (15, 18)))
    e4 = ContingencyManager(sky)
    zone = Zone('crash', 'Crash site', polygon=[(-3, -3), (3, -3), (3, 3), (-3, 3)])
    pos = {'A': at(sky, a, 3.0), 'F': at(sky, far, 3.0)}
    acts = e4.add_zone(zone, 3.0, pos)
    assert [x['flight'] for x in acts] == ['A']
    d = acts[0]['decision']
    assert d.status == 'approved' and d.around_zones
    zk = set(zone_keys(zone, sky.sky, sky.s.n_layers))
    assert not [k for k in owned(sky, 'A') if k in zk and max(hi for _, hi in owned(sky, 'A')[k]) > 3.0 + 2.0]
    assert sky.current['F'] is far


def test_drone_next_to_a_new_zone_still_gets_around_it():
    """The zone appears with the drone already inside its margin (< 2.7 m from it): the re-plan
    starts from the nearest free air and goes around."""
    sky = StrategicPlanner([], B)
    sky.submit(FlightPlan('A', (-15, 0), (15, 0)))
    e4 = ContingencyManager(sky)
    zone = Zone('z', 'Closure', polygon=[(-2, -3), (4, -3), (4, 3), (-2, 3)])
    acts = e4.add_zone(zone, 9.0, {'A': (-4.0, 0.0, 2.0)})   # 2 m from the zone's edge
    assert acts and acts[0]['decision'].status == 'approved'


def test_goal_inside_a_new_zone_returns_to_base():
    sky = StrategicPlanner([], B)
    sky.submit(FlightPlan('A', (-15, 0), (15, 0)))
    e4 = ContingencyManager(sky)
    zone = Zone('z', 'Closure at the goal', polygon=[(12, -3), (18, -3), (18, 3), (12, 3)])
    acts = e4.add_zone(zone, 5.0, {'A': (-11.0, 0.0, 2.0)})
    d = acts[0]['decision']
    assert d.status == 'approved' and d.reason == 'return to base'
    assert d.waypoints[-1][:2] == pytest.approx((-15.0, 0.0))


def test_traffic_alone_never_sends_a_drone_home():
    sky = StrategicPlanner([], B)
    a = sky.submit(FlightPlan('A', (-15, 0), (15, 0)))
    for k in range(5):                                     # crowd the air ahead of A
        sky.submit(FlightPlan(f'C{k}', (-15 + 6 * k, -15), (-15 + 6 * k, 15), t_earliest=4.0 + k))
    d = sky.replan('A', at(sky, a, 6.0), 6.0)
    assert d.reason != 'return to base'


def test_ground_flight_that_cannot_fly_is_cancelled_and_landed_ones_are_left_alone():
    sky = StrategicPlanner([], B)
    sky.submit(FlightPlan('W', (-15, 0), (15, 0), t_earliest=30.0))   # still waiting at t = 5
    done = sky.submit(FlightPlan('L', (-15, 10), (15, 10)))            # landed by t = 40
    e4 = ContingencyManager(sky)
    goal_zone = Zone('z', 'Closure', polygon=[(12, -3), (18, -3), (18, 3), (12, 3)])
    acts = e4.add_zone(goal_zone, 5.0, {})
    assert [x['why'] for x in acts] == ['zone z -> cancelled']
    assert 'W' not in sky.current and not owned(sky, 'W')
    later = Zone('z2', 'Closure 2', polygon=[(12, 7), (18, 7), (18, 13), (12, 13)])
    assert e4.add_zone(later, done.arrival + 20.0, {}) == []


# ------------------------------------------------------------------ D. intruder
def test_flight_about_to_meet_an_intruder_is_replanned():
    sky = StrategicPlanner([], B)
    a = sky.submit(FlightPlan('A', (-15, 0), (15, 0)))
    mon = ConformanceMonitor(sky)
    e4 = ContingencyManager(sky, mon)
    acts = e4.step(4.0, {'A': at(sky, a, 4.0), 'rogue': (0.0, 0.0, 2.0)})
    assert [x['flight'] for x in acts] == ['A'] and acts[0]['why'] == 'intruder rogue'
    assert acts[0]['decision'].status == 'approved'
    assert not overlap(owned(sky, 'A'), owned(sky, 'intruder:rogue'), after=4.0 + 2.0)


# ------------------------------------------------------------------ E. alert rules
def _late_run(persist, traffic=True):
    """A is 6 s behind its plan; with traffic, B was planned across A's route after A."""
    sky = StrategicPlanner([], B)
    a = sky.submit(FlightPlan('A', (-15, 0), (15, 0)))
    if traffic:
        sky.submit(FlightPlan('B', (0, -15), (0, 15)))
    mon = ConformanceMonitor(sky)
    e4 = ContingencyManager(sky, mon, persist=persist)
    acts, t = [], 0.0
    while t < 20.0:
        if t >= 6.0:
            acts += e4.step(t, {'A': at(sky, a, t - 6.0)})
        t += 0.25
    return mon, acts


def test_late_with_nobody_around_is_left_alone():
    mon, acts = _late_run(persist=3.0, traffic=False)
    assert any(e['state'] == 'late' for e in mon.events) and acts == []


def test_late_into_traffic_is_replanned_once_high_risk_for_long_enough():
    mon, acts = _late_run(persist=3.0)
    late = [e for e in mon.events if e['flight'] == 'A' and e['state'] == 'late']
    high = [e['t'] for e in late if e['risk'] == 'high']
    mine = [x for x in acts if x['flight'] == 'A' and x['why'] == 'late']
    assert high and mine
    assert mine[0]['t'] >= late[0]['t'] + 3.0 and mine[0]['t'] >= high[0]


def test_cooldown_limits_replans():
    _, acts = _late_run(persist=0.0)
    times = [x['t'] for x in acts if x['flight'] == 'A']
    assert times and all(b - a >= 5.0 for a, b in zip(times, times[1:]))


def test_intrusion_replans_the_other_flight_first():
    """A drifts into B's booked air: E4 moves B (on its plan, more options) and leaves A."""
    sky = StrategicPlanner([], B)
    a = sky.submit(FlightPlan('A', (-15, 0), (15, 0)))
    b = sky.submit(FlightPlan('B', (-15, 12), (15, 12), t_earliest=20.0))
    mon = ConformanceMonitor(sky)
    e4 = ContingencyManager(sky, mon)
    acts = []
    for k in range(40):                                     # A flies along B's route at B's time
        t = b.t_dep + 2.0 + 0.25 * k
        acts += e4.step(t, {'A': at(sky, b, t)})
        if acts:
            break
    assert acts and acts[0]['flight'] == 'B' and acts[0]['why'] == 'intrusion by A'
    assert acts[0]['decision'].status == 'approved'


def test_early_drone_is_left_alone():
    sky = StrategicPlanner([], B)
    a = sky.submit(FlightPlan('A', (-15, 0), (15, 0)))
    mon = ConformanceMonitor(sky)
    e4 = ContingencyManager(sky, mon)
    acts = []
    for k in range(60):
        t = k * 0.25
        acts += e4.step(t, {'A': at(sky, a, t + 5.0)})     # 5 s ahead of its plan
    assert any(e['state'] == 'early' for e in mon.events) and acts == []
