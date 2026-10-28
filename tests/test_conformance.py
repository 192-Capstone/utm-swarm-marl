"""
test_conformance.py
-------------------
E3 conformance monitor (utm/conformance.py) on hand-made E1 plans (nothing is flown).

  A. A drone flying its plan exactly raises no alert (no false alarms).
  B. Late, off-route, intrusion into another flight's booking or an active zone, overstay at the
     goal pad, and an unplanned drone are each reported, with the other flight / zone named.
  C. Debounce: a short blip (< 1 s) raises nothing.

Run: python -m pytest tests/test_conformance.py -q
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

pytest.importorskip('h3')

from utm.airspace_zones import Zone, apply_zones  # noqa: E402
from utm.conformance import ConformanceMonitor  # noqa: E402
from utm.strategic_planner import FlightPlan, StrategicPlanner  # noqa: E402

B = ((-25.0, 25.0), (-25.0, 25.0))
DT = 0.25


def track(planner, d, shift=0.0, offset=(0.0, 0.0)):
    """(t, (x, y, z)) along a decision's profile, `shift` s late, moved by `offset` metres."""
    route = [(x, y) for x, y, _ in d.waypoints[1:-1]]
    prof, _ = planner.profile(route, d.layer)
    return [(d.t_dep + shift + tau, (x + offset[0], y + offset[1], z)) for tau, x, y, z in prof]


def fly(mon, fid, samples):
    for t, pos in samples:
        mon.check(t, {fid: pos})
    return [e for e in mon.events if e['flight'] == fid]


def test_flying_the_plan_raises_nothing():
    sky = StrategicPlanner([], B)
    ds = [sky.submit(FlightPlan(f, a, b, t_earliest=t)) for f, a, b, t in
          (('A', (-15, 0), (15, 0), 0.0), ('B', (0, -15), (0, 15), 0.0), ('C', (-15, 10), (15, -10), 3.0))]
    mon = ConformanceMonitor(sky)
    for d in ds:
        assert fly(mon, d.id, track(sky, d)) == []


def test_late_drone():
    sky = StrategicPlanner([], B)
    d = sky.submit(FlightPlan('A', (-15, 0), (15, 0)))
    events = fly(ConformanceMonitor(sky), 'A', track(sky, d, shift=6.0))
    assert events and events[0]['state'] == 'late'


def test_off_route_drone():
    sky = StrategicPlanner([], B)
    d = sky.submit(FlightPlan('A', (-15, 0), (15, 0)))
    events = fly(ConformanceMonitor(sky), 'A', track(sky, d, offset=(0.0, 6.0)))
    assert 'off_route' in {e['state'] for e in events}


def test_intrusion_names_the_other_flight():
    sky = StrategicPlanner([], B)
    a = sky.submit(FlightPlan('A', (-15, 0), (15, 0)))
    sky.submit(FlightPlan('B', (-15, 12), (15, 12)))
    mon = ConformanceMonitor(sky)
    # B's drone flies along A's corridor at A's time (B never booked it)
    for t, pos in track(sky, a):
        mon.check(t, {'B': pos})
    hits = [e for e in mon.events if e['state'] == 'intrusion']
    assert hits and hits[0]['other'] == 'A'


def test_intrusion_into_an_active_zone():
    sky = StrategicPlanner([], B)
    apply_zones(sky, [Zone('closed', 'Street closure', active=[[0.0, 100.0]],
                           polygon=[(-2, 8), (2, 8), (2, 12), (-2, 12)])])
    d = sky.submit(FlightPlan('A', (-15, 0), (15, 0)))
    events = fly(ConformanceMonitor(sky), 'A', track(sky, d, offset=(0.0, 10.0)))
    assert any(e['state'] == 'intrusion' and e['other'] == 'zone:closed' for e in events)


def test_overstay_over_the_goal_pad():
    sky = StrategicPlanner([], B)
    d = sky.submit(FlightPlan('A', (-15, 0), (15, 0)))
    samples = track(sky, d)
    t_end, (gx, gy, gz) = samples[-1]
    samples += [(t_end + k * DT, (gx, gy, 1.5)) for k in range(1, 80)]      # hovers 20 s over the pad
    events = fly(ConformanceMonitor(sky), 'A', samples)
    assert events and events[-1]['state'] == 'overstay'


def test_unplanned_drone_is_unknown():
    sky = StrategicPlanner([], B)
    sky.submit(FlightPlan('A', (-15, 0), (15, 0)))
    events = fly(ConformanceMonitor(sky), 'rogue', [(k * DT, (0.0, 5.0, 2.0)) for k in range(8)])
    assert [e['state'] for e in events] == ['unknown']


def test_late_with_nobody_around_is_low_risk():
    sky = StrategicPlanner([], B)
    d = sky.submit(FlightPlan('A', (-15, 0), (15, 0)))
    events = fly(ConformanceMonitor(sky), 'A', track(sky, d, shift=6.0))
    assert events and all(e['risk'] == 'low' for e in events if e['state'] == 'late')


def test_late_into_someone_elses_air_escalates_to_high_risk():
    """A is 6 s late; B was planned to cross A's route after A had passed. While A is far from
    the crossing its alert is low risk; nearing B's booked air it becomes high, naming B."""
    sky = StrategicPlanner([], B)
    a = sky.submit(FlightPlan('A', (-15, 0), (15, 0)))
    sky.submit(FlightPlan('B', (0, -15), (0, 15)))          # E1 delays B behind A at the crossing
    events = [e for e in fly(ConformanceMonitor(sky), 'A', track(sky, a, shift=6.0)) if e['state'] == 'late']
    risks = [e['risk'] for e in events]
    assert risks[0] == 'low' and 'high' in risks
    assert [e['other'] for e in events if e['risk'] == 'high'][0] == 'B'


def test_intrusion_is_always_high_risk():
    sky = StrategicPlanner([], B)
    a = sky.submit(FlightPlan('A', (-15, 0), (15, 0)))
    sky.submit(FlightPlan('B', (-15, 12), (15, 12)))
    mon = ConformanceMonitor(sky)
    for t, pos in track(sky, a):
        mon.check(t, {'B': pos})
    hits = [e for e in mon.events if e['state'] == 'intrusion']
    assert hits and all(e['risk'] == 'high' for e in hits)
    assert mon.summary()['high_risk'] >= len(hits)


def test_a_short_blip_is_ignored():
    sky = StrategicPlanner([], B)
    d = sky.submit(FlightPlan('A', (-15, 0), (15, 0)))
    samples = track(sky, d)
    mid = len(samples) // 2
    for k in range(mid, mid + 3):                                           # 0.5-0.75 s off route
        t, (x, y, z) = samples[k]
        samples[k] = (t, (x, y + 6.0, z))
    assert fly(ConformanceMonitor(sky), 'A', samples) == []
