"""E2: telemetry reports (utm/telemetry.py), intent and predicted paths (utm/intent.py), and E3's
intent-based risk (utm/conformance.py, risk='intent')."""

import numpy as np
import pytest

from envs.route_planner import RouteFollower
from utm.conformance import ConformanceMonitor
from utm.intent import closest_approach, first_conflict, predicted_path, profile_intent, route_intent
from utm.strategic_planner import FlightPlan, StrategicPlanner
from utm.telemetry import Report, TelemetryModel

B = ((-20, 20), (-20, 20))


# ------------------------------------------------------------------ telemetry
def test_perfect_link_delivers_the_true_state_at_once():
    tm = TelemetryModel()
    tm.send(1.0, {'a': ((1, 2, 3), (0.5, 0, 0), [[3.0, 2, 2, 3]])})
    got = tm.receive(1.0)
    assert len(got) == 1 and np.allclose(got[0].pos, (1, 2, 3)) and got[0].intent == [[3.0, 2, 2, 3]]
    assert tm.latest['a'].t_sent == 1.0


def test_delay_loss_and_noise():
    tm = TelemetryModel(noise_m=1.0, delay_s=0.5, loss=0.2, seed=0)
    for k in range(500):
        tm.send(float(k), {'a': ((0, 0, 2), (0, 0, 0), [])})
    early = tm.receive(10.4)
    assert early and 'a' in tm.latest and tm.latest['a'].t_sent <= 9.9               # nothing newer yet
    got = early + tm.receive(1000.0)
    assert 0.15 < tm.lost / tm.sent < 0.25
    err = np.array([r.pos for r in got]) - (0, 0, 2)
    assert 0.9 < err.std() < 1.1
    seqs = sorted(r.seq for r in got)
    assert seqs == sorted(set(seqs)) and 500 - len(seqs) == tm.lost and max(seqs) <= 499    # gaps = lost
    assert all(r.t_received == pytest.approx(r.t_sent + 0.5) for r in got)


def test_bad_settings_are_refused():
    with pytest.raises(ValueError):
        TelemetryModel(loss=1.0)


# ------------------------------------------------------------------ intent
def test_route_intent_follows_the_route_at_the_drones_speed():
    f = RouteFollower([(0, 0, 2), (10, 0, 2), (10, 10, 2)])
    it = route_intent(f, (0, 0, 2), speed=1.0, t=5.0)
    assert [q[0] for q in it] == [7.0, 9.0, 11.0, 13.0, 15.0]
    assert np.allclose(it[0][1:], (2, 0, 2)) and np.allclose(it[-1][1:], (10, 0, 2))
    it = route_intent(f, (0, 0, 2), speed=2.0, t=0.0)
    assert np.allclose(it[-1][1:], (10, 10, 2))               # past the end: stays at the goal


def test_predicted_path_uses_intent_else_velocity():
    with_intent = Report('a', 0.0, np.array([0.0, 0, 2]), np.array([1.0, 0, 0]), [[10.0, 0, 10, 2]])
    t, p = predicted_path(with_intent, 0.0)
    assert np.allclose(p[-1], (0, 10, 2))                      # turns as planned, not straight on
    no_intent = Report('b', 0.0, np.array([0.0, 0, 2]), np.array([1.0, 0, 0]))
    t, p = predicted_path(no_intent, 2.0)
    assert np.allclose(p[0], (2, 0, 2)) and np.allclose(p[-1], (12, 0, 2))


def test_closest_approach_of_two_crossing_drones():
    a = Report('a', 0.0, np.array([-5.0, 0, 2]), np.array([1.0, 0, 0]))
    b = Report('b', 0.0, np.array([0.0, -5, 2]), np.array([0.0, 1, 0]))
    d, t = closest_approach(predicted_path(a, 0.0), predicted_path(b, 0.0))
    assert d == pytest.approx(0.0) and t == pytest.approx(5.0)


def test_a_late_report_keeps_moving_past_its_last_intent_point():
    late = Report('a', 0.0, np.array([0.0, 0, 2]), np.zeros(3), [[2.0, 2, 0, 2], [4.0, 4, 0, 2]])
    t, p = predicted_path(late, 3.0)                           # intent ends at 4 s, path runs to 13 s
    assert np.allclose(p[0], (3, 0, 2)) and np.allclose(p[-1], (13, 0, 2))   # keeps 1 m/s along x


def test_a_conflict_is_a_flat_disc_as_e1_separates_drones():
    low = Report('a', 0.0, np.array([-5.0, 0, 2.0]), np.array([1.0, 0, 0]))
    high = Report('b', 0.0, np.array([0.0, -5, 4.0]), np.array([0.0, 1, 0]))   # the layer above
    same = Report('c', 0.0, np.array([0.0, -5, 2.3]), np.array([0.0, 1, 0]))   # same layer
    a, b, c = (predicted_path(r, 0.0) for r in (low, high, same))
    assert closest_approach(a, b)[0] == pytest.approx(2.0)    # < 2.1 m in 3D, but...
    assert first_conflict(a, b, 2.1) is None                  # ...2 m apart in height: separated
    assert first_conflict(a, c, 2.1) == pytest.approx(4.0)    # same layer: within 2.1 m from 3.52 s


# ------------------------------------------------------------------ E3 with intent risk
def _sky():
    sky = StrategicPlanner([], B)
    sky.submit(FlightPlan('A', (-15, 0), (15, 0), t_earliest=0.0))
    sky.submit(FlightPlan('B', (-15, 10), (15, 10), t_earliest=0.0))
    return sky


def test_intent_risk_is_low_when_nobody_is_heading_the_drones_way():
    sky = _sky()
    mon = ConformanceMonitor(sky, risk='intent')
    off = (-10.0, 5.0, sky.layer_z(0))                         # A wandered off its route
    paths = {'A': predicted_path(Report('A', 0.0, np.array(off), np.zeros(3)), 0.0)}
    for t in (0.0, 1.0, 2.0):
        mon.check(t, {'A': off}, paths)
    assert mon.state['A'] == 'off_route' and mon.risk['A'] == 'low'


def test_intent_risk_is_high_when_two_paths_meet():
    sky = _sky()
    mon = ConformanceMonitor(sky, risk='intent')
    z = sky.layer_z(0)
    a = Report('A', 0.0, np.array([-10.0, 5.0, z]), np.array([0.0, 1.0, 0]))    # off route, heading for B
    b = Report('B', 0.0, np.array([-10.0, 10.0, z]), np.zeros(3))
    for t in (0.0, 1.0, 2.0):
        mon.check(t, {'A': a.pos, 'B': b.pos}, {'A': predicted_path(a, t), 'B': predicted_path(b, t)})
    assert mon.state['A'] == 'off_route' and mon.risk['A'] == 'high'
    assert [e['other'] for e in mon.events if e['flight'] == 'A'] == ['B']


def test_intent_risk_ignores_a_drone_crossing_on_the_other_layer():
    sky = _sky()
    mon = ConformanceMonitor(sky, risk='intent')
    a = Report('A', 0.0, np.array([-10.0, 5.0, sky.layer_z(0)]), np.array([0.0, 1.0, 0]))
    b = Report('B', 0.0, np.array([-10.0, 10.0, sky.layer_z(1)]), np.zeros(3))   # 2 m above A's path
    for t in (0.0, 1.0, 2.0):
        mon.check(t, {'A': a.pos, 'B': b.pos}, {'A': predicted_path(a, t), 'B': predicted_path(b, t)})
    assert mon.state['A'] == 'off_route' and mon.risk['A'] == 'low'


def test_booking_risk_is_still_the_default():
    assert ConformanceMonitor(_sky()).risk_mode == 'booking'
    with pytest.raises(ValueError):
        ConformanceMonitor(_sky(), risk='guess')


def test_utm_drops_a_drone_only_when_its_reports_stop():
    """#37 review (1.7): the UTM forgets a drone once it has been silent for LINK_TIMEOUT_S, never
    from the simulator's truth (a landed, crashed or comms-lost drone just goes quiet)."""
    from utm.telemetry import LINK_TIMEOUT_S, TelemetryModel, silent
    tm = TelemetryModel()
    for t in range(0, 6):                                  # f1 reports every second; f2 stops after t = 2
        tm.send(t, {'f1': ((0, 0, 2), (0, 0, 0), [])} if t > 2 else
                {'f1': ((0, 0, 2), (0, 0, 0), []), 'f2': ((5, 0, 2), (0, 0, 0), [])})
        tm.receive(t)
    assert silent(tm.latest, 5) == []                              # f2 quiet for 3 s: not yet
    assert silent(tm.latest, 2 + LINK_TIMEOUT_S + 0.5) == ['f2']
    tm.forget('f2')
    assert set(tm.latest) == {'f1'}


def _route(zc=4.0):
    return RouteFollower([(0.0, 0.0, zc), (20.0, 0.0, zc)])     # cruise at 4 m, pad at (20, 0)


def test_profile_intent_climbs_then_flies_the_route():
    """E2 option D: a drone still climbing shares the climb, then the route."""
    out = profile_intent(_route(), (0.0, 0.0, 1.0), 0.0, 10.0, 'climb', 1.0, v_cruise=1.5, v_up=0.5, v_down=0.5)
    z = [q[3] for q in out]
    assert out[0][0] == 12.0 and z[0] == pytest.approx(2.0)           # 1 m up after 2 s at 0.5 m/s
    assert z[2] == pytest.approx(4.0) and out[-1][1] > 0.0            # at cruise height, then moving off


def test_profile_intent_shows_the_landing_ahead():
    """E2 v2, city seed 22: f8 cruising 2 m short of its pad was pictured staying at cruise height and
    landed through f3's layer unwarned. With D its intent goes down onto the pad."""
    f = _route()
    old = route_intent(f, (18.0, 0.0, 4.0), 1.0, 0.0)
    new = profile_intent(f, (18.0, 0.0, 4.0), 1.0, 0.0, 'cruise', 1.0, v_cruise=1.5, v_up=0.5, v_down=0.5)
    assert min(q[3] for q in old) == pytest.approx(4.0)               # route_intent: never comes down
    assert [q[3] for q in new][:3] == pytest.approx([4.0, 3.0, 2.0])  # at the pad by 2 s, then 0.5 m/s down
    assert new[-1][1:] == pytest.approx([20.0, 0.0, 1.0])             # on the pad


def test_profile_intent_landing_goes_straight_down():
    out = profile_intent(_route(), (19.8, 0.1, 3.0), 0.0, 0.0, 'land', 1.0, v_cruise=1.5, v_up=0.5, v_down=0.5)
    assert all(q[1:3] == [19.8, 0.1] for q in out)
    assert [q[3] for q in out][:3] == pytest.approx([2.0, 1.0, 1.0])


def test_a_late_drone_shares_its_real_timing():
    """Option A's time shift is not needed: the intent starts where the drone is, at its speed, so a
    drone 5 m behind schedule at 0.5 m/s shares exactly that."""
    out = profile_intent(_route(), (5.0, 0.0, 4.0), 0.5, 30.0, 'cruise', 1.0, v_cruise=1.5, v_up=0.5, v_down=0.5)
    assert out[0][:2] == pytest.approx([32.0, 6.0])


def test_blind_margin_widens_only_pairs_with_a_drone_that_shares_no_intent():
    """E2 option B (small): 2.6 m apart is clear at the 2.1 m radius, a risk with a 1 m margin, and
    only when one of the two shares no intent."""
    sky = _sky()
    z = sky.layer_z(0)
    a = Report('A', 0.0, np.array([-10.0, 5.0, z]), np.zeros(3))
    b = Report('B', 0.0, np.array([-10.0, 7.6, z]), np.zeros(3))
    for margin, blind, risk in ((0.0, {'B'}, 'low'), (1.0, set(), 'low'), (1.0, {'B'}, 'high')):
        mon = ConformanceMonitor(sky, risk='intent', blind_margin=margin)
        mon.blind = blind
        mon.plans = {}                                                 # both unplanned: risk only
        for t in (0.0, 1.0, 2.0):
            mon.check(t, {'A': a.pos, 'B': b.pos}, {'A': predicted_path(a, t), 'B': predicted_path(b, t)})
        assert mon.risk.get('A') == risk, (margin, blind)


def test_flags_off_leave_the_monitor_unchanged():
    """Without --blind-margin (the default 0) the monitor ignores `blind`: the same close pair gives the
    same risk whoever is marked blind, so fly() without the E2 follow-up flags behaves as before."""
    sky = _sky()
    z = sky.layer_z(0)
    a = Report('A', 0.0, np.array([-10.0, 5.0, z]), np.zeros(3))
    b = Report('B', 0.0, np.array([-10.0, 6.5, z]), np.zeros(3))      # 1.5 m apart: inside the 2.1 m radius
    risks = []
    for blind in (set(), {'A'}, {'A', 'B'}):
        mon = ConformanceMonitor(sky, risk='intent')
        mon.blind = blind
        mon.plans = {}
        for t in (0.0, 1.0, 2.0):
            mon.check(t, {'A': a.pos, 'B': b.pos}, {'A': predicted_path(a, t), 'B': predicted_path(b, t)})
        risks.append(mon.risk.get('A'))
    assert risks == ['high'] * 3
