"""
test_conformance_failures.py
----------------------------
E3 drone-failure detection (utm/conformance.py, ConformanceMonitor(failures=True)): lost_link,
down and emergency_landing, judged only from reports (status, vel, time sent), on hand-made
plans (nothing is flown).

  A. Options off: the output is as before, whatever reports are passed.
  B. booked_at / the closing rule: the planner's CURRENT booking, cached per decision.
  C. lost_link: the timeout is its debounce; judged at the booked position; a drone dropped from
     tm.latest still counts; an arrived drone going quiet does not; reports resuming clear it.
  D. down: status 'emergency' or vz below -2.5 m/s, at once.
  E. emergency_landing: status 'landing', or descending below its booking away from its goal pad;
     a re-plan that moves the drone down a layer (14 m -> 8 m, faster than booked) is not one.
  F. Precedence and stickiness.
  G. booked(): only the planner's newest approved decision.
  H. No false alarms: 12 healthy drones through the real telemetry noise; drones that land and go
     quiet are not lost; drones that go quiet mid-route are.

Run: python -m pytest tests/test_conformance_failures.py -q
"""

import os
import sys
from dataclasses import replace

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

pytest.importorskip('h3')

from utm.conformance import (CLIMB_DONE_M, DOWN_VZ, FAILURE_STATES, FINAL_DESCENT_M,  # noqa: E402
                             HEIGHT_MARGIN_M,
                             LANDING_DEBOUNCE_S, LANDING_VZ, STATES, ConformanceMonitor)
from utm.strategic_planner import Decision, FlightPlan, Settings, StrategicPlanner  # noqa: E402
from utm.telemetry import LINK_TIMEOUT_S, Report, TelemetryModel, silent  # noqa: E402

B = ((-25.0, 25.0), (-25.0, 25.0))


def rep(fid, t, pos, vz=0.0, status='ok', vxy=(0.0, 0.0)):
    return Report(fid, float(t), np.asarray(pos, dtype=float), np.array([vxy[0], vxy[1], vz]), status=status)


def sky_and_flight(layers=2, layer=0, **kw):
    """A planner with one approved flight A, (-15, 0) -> (15, 0), cruising on `layer`."""
    sky = StrategicPlanner([], B, Settings(n_layers=layers, layer_penalty=100.0, **kw))
    d = sky.submit(FlightPlan('A', (-15, 0), (15, 0), layer=layer))
    assert d.status == 'approved' and d.layer == layer
    return sky, d


def monitor(sky, **kw):
    return ConformanceMonitor(sky, failures=True, **kw)


def states(mon, fid='A'):
    return [e['state'] for e in mon.events if e['flight'] == fid]


def mid(d):
    """A time well inside the cruise."""
    return d.t_dep + 0.5 * (d.arrival - d.t_dep)


# ------------------------------------------------------------------ A. options off
def test_options_off_ignore_reports_and_keep_the_old_output():
    plain = ConformanceMonitor(StrategicPlanner([], B))
    assert plain.failures is False
    sky = StrategicPlanner([], B)
    d = sky.submit(FlightPlan('A', (-15, 0), (15, 0)))
    route = [(x, y) for x, y, _ in d.waypoints[1:-1]]
    prof, _ = sky.profile(route, d.layer)
    a, b = ConformanceMonitor(sky), ConformanceMonitor(sky)
    for tau, x, y, z in prof:
        t = d.t_dep + 6.0 + tau
        a.check(t, {'A': (x, y, z)})
        b.check(t, {'A': (x, y, z)}, reports={'A': rep('A', t, (x, y, z), vz=-5.0, status='emergency')})
    assert a.events == b.events and a.events and a.state == b.state
    assert a.summary() == b.summary()
    assert set(a.summary()['by_state']) == set(STATES) - {'ok'}        # no new keys when off
    assert not set(FAILURE_STATES) & {e['state'] for e in b.events}


def test_options_on_without_reports_is_the_old_output_too():
    sky = StrategicPlanner([], B)
    d = sky.submit(FlightPlan('A', (-15, 0), (15, 0)))
    route = [(x, y) for x, y, _ in d.waypoints[1:-1]]
    prof, _ = sky.profile(route, d.layer)
    off, on = ConformanceMonitor(sky), ConformanceMonitor(sky, failures=True)
    for tau, x, y, z in prof:
        t = d.t_dep + 6.0 + tau
        off.check(t, {'A': (x, y, z)})
        on.check(t, {'A': (x, y, z)})
    assert off.events == on.events and off.events
    assert set(on.summary()['by_state']) == (set(STATES) | set(FAILURE_STATES)) - {'ok'}


def test_simulator_climb_done_constant_is_mirrored():
    pytest.importorskip('torch')
    swarm_env = pytest.importorskip('envs.swarm_env')
    assert CLIMB_DONE_M == swarm_env.CLIMB_DONE_M


# ------------------------------------------------------------------ B. booked_at and closing
def test_booked_at_follows_the_profile_and_clamps():
    sky, d = sky_and_flight()
    mon = monitor(sky)
    x, y, z = mon.booked_at('A', mid(d))
    assert z == pytest.approx(sky.layer_z(0)) and -15 < x < 15 and y == pytest.approx(0.0)
    assert mon.booked_at('A', d.t_dep - 10) == pytest.approx((-15.0, 0.0, 1.0), abs=1e-6)
    assert mon.booked_at('A', d.arrival + 10) == pytest.approx((15.0, 0.0, 1.0), abs=1e-6)
    assert mon.booked_at('nobody', 0.0) is None


def test_booked_at_uses_the_planners_current_booking_not_a_copy_only_e4_refreshes():
    sky, d = sky_and_flight()
    mon = monitor(sky)                                  # mon.plans[A] is the first decision
    t = mid(d)
    first = mon.booked_at('A', t)
    d2 = sky.replan('A', (first[0], first[1], first[2]), t)
    assert d2.status == 'approved' and sky.current['A'] is d2 and mon.plans['A'] is d
    assert mon.booked('A') is d2                      # the monitor's own copy is not consulted
    assert mon.booked_at('A', d2.arrival + 5) == pytest.approx((15.0, 0.0, 1.0), abs=1e-6)
    assert mon.booked_at('A', d2.arrival + 5) != mon.booked_at('A', d.arrival + 5) or d2.arrival == d.arrival


def test_booked_at_profile_is_cached_per_decision_object():
    sky, d = sky_and_flight()
    mon = monitor(sky)
    mon.booked_at('A', mid(d))
    cached = mon._profiles['A']
    mon.booked_at('A', mid(d) + 1.0)
    assert mon._profiles['A'] is cached and cached[0] is d
    t = mid(d)
    x, y, z = mon.booked_at('A', t)
    d2 = sky.replan('A', (x, y, z), t)
    mon.booked_at('A', t + 5.0)
    assert mon._profiles['A'][0] is d2 and mon._profiles['A'] is not cached


# ------------------------------------------------------------------ C. lost_link
def test_lost_link_after_the_timeout_and_at_the_booked_position():
    sky, d = sky_and_flight()
    mon = monitor(sky)
    t0 = mid(d)
    last = rep('A', t0, mon.booked_at('A', t0))
    mon.check(t0, {}, reports={'A': last})
    mon.check(t0 + LINK_TIMEOUT_S, {}, reports={'A': last})            # exactly the timeout: not yet
    assert states(mon) == []
    t1 = t0 + LINK_TIMEOUT_S + 0.5
    ev = mon.check(t1, {}, reports={'A': last})
    assert [e['state'] for e in ev] == ['lost_link'] and mon.state['A'] == 'lost_link'
    assert ev[0]['key'] == mon.key_of(mon.booked_at('A', t1))         # where the booking has it now
    assert ev[0]['key'] != mon.key_of(last.pos)                       # not where it was last heard
    assert mon.risk['A'] in ('high', 'low')
    assert mon.check(t1 + 1, {}, reports={'A': last}) == []           # no repeated event


def test_a_drone_dropped_from_tm_latest_is_lost_link_not_gone():
    sky, d = sky_and_flight()
    mon = monitor(sky)
    tm = TelemetryModel()
    t0 = int(mid(d))
    for t in range(t0, t0 + 3):                         # reports for 3 s, the eval checks each second
        tm.send(t, {'A': (mon.booked_at('A', t), (1.5, 0, 0), [])})
        tm.receive(t)
        mon.check(t, {}, reports=tm.latest)
    for t in range(t0 + 3, t0 + 12):                    # then silence; the eval forgets A after the timeout
        for fid in silent(tm.latest, t):
            tm.forget(fid)
        mon.check(t, {}, reports=tm.latest)
    assert 'A' not in tm.latest                         # dropped, as eval_layered does
    assert mon.state['A'] == 'lost_link'


def test_an_arrived_drone_going_quiet_is_not_lost_link():
    sky, d = sky_and_flight()
    mon = monitor(sky)
    goal = d.waypoints[-1]
    last = rep('A', d.arrival - 1.0, (goal[0] + 0.5, goal[1], goal[2] + 0.2))       # on its goal pad, low
    mon.check(d.arrival - 1.0, {}, reports={'A': last})
    for t in (d.arrival + 5, d.arrival + 60):
        assert mon.check(t, {}, reports={'A': last}) == []
    assert 'A' not in mon.state


def test_a_drone_that_went_quiet_in_the_air_over_its_goal_is_lost_link_until_closed():
    """Over the goal but high and before the booked arrival: not arrived yet; close() ends doubt."""
    sky, d = sky_and_flight()
    mon = monitor(sky, noise_m=0.0)                     # exact telemetry: 3 m over the pad is not on it
    goal = d.waypoints[-1]
    last = rep('A', d.arrival - 6.0, (goal[0], goal[1], goal[2] + 3.0))
    mon.check(d.arrival - 6.0, {}, reports={'A': last})
    assert states(mon) == []
    mon.check(d.arrival - 1.0, {}, reports={'A': last})
    assert mon.state['A'] == 'lost_link'
    sky2, d2 = sky_and_flight()
    mon2 = monitor(sky2, noise_m=0.0)
    mon2.check(d2.arrival - 6.0, {}, reports={'A': last})
    mon2.close('A')
    mon2.check(d2.arrival - 1.0, {}, reports={'A': last})
    assert 'A' not in mon2.state


def test_lost_link_clears_when_the_reports_resume():
    sky, d = sky_and_flight()
    mon = monitor(sky)
    t0 = mid(d)
    last = rep('A', t0, mon.booked_at('A', t0))
    mon.check(t0, {}, reports={'A': last})
    mon.check(t0 + 5, {}, reports={'A': last})
    assert mon.state['A'] == 'lost_link'
    back = rep('A', t0 + 6, mon.booked_at('A', t0 + 6))
    ev = mon.check(t0 + 6.2, {}, reports={'A': back})
    assert [e['state'] for e in ev] == ['ok'] and mon.state['A'] == 'ok' and mon.risk['A'] == ''


def test_flights_without_a_booking_or_never_heard_are_not_judged():
    sky, d = sky_and_flight()
    mon = monitor(sky)
    mon.check(100.0, {}, reports={'intruder': rep('intruder', 90.0, (0, 0, 2), status='emergency')})
    mon.check(200.0, {})                                # A was never heard (still on its pad)
    assert mon.events == [] and mon.state == {}


# ------------------------------------------------------------------ D. down
def test_down_on_an_emergency_status_at_once():
    sky, d = sky_and_flight()
    mon = monitor(sky)
    t = mid(d)
    mon.check(t, {}, reports={'A': rep('A', t, mon.booked_at('A', t))})
    ev = mon.check(t + 1, {}, reports={'A': rep('A', t + 1, mon.booked_at('A', t + 1), status='emergency')})
    assert [e['state'] for e in ev] == ['down'] and ev[0]['t'] == t + 1 and ev[0]['risk'] == 'high'


def test_down_on_a_fall_faster_than_a_healthy_drone_can_descend():
    sky, d = sky_and_flight(layers=4, layer=2)
    mon = monitor(sky)
    t = mid(d)
    p = mon.booked_at('A', t)
    mon.check(t, {}, reports={'A': rep('A', t, p, vz=-2.0)})                    # healthy limit: 2 m/s
    assert mon.state.get('A') is None
    mon.check(t + 1, {}, reports={'A': rep('A', t + 1, p, vz=DOWN_VZ - 0.1)})
    assert mon.state['A'] == 'down'


# ------------------------------------------------------------------ E. emergency_landing
def test_landing_on_a_landing_status():
    sky, d = sky_and_flight()
    mon = monitor(sky)
    t = mid(d)
    ev = mon.check(t, {}, reports={'A': rep('A', t, mon.booked_at('A', t), status='landing')})
    assert [e['state'] for e in ev] == ['emergency_landing']


def test_landing_on_a_descent_below_the_booking_after_the_debounce():
    sky, d = sky_and_flight(layers=4, layer=2)                                   # cruises at 6 m, floor 6 - 3 = 3 m
    mon = monitor(sky)
    t = mid(d)
    x, y, _ = mon.booked_at('A', t)
    for k, z in enumerate((4.0, 2.5, 2.0, 1.5, 1.0)):
        mon.check(t + k, {}, reports={'A': rep('A', t + k, (x, y, z), vz=-0.5)})
        if k < 3:
            assert mon.state.get('A') is None                  # 2.5 m is the first below the floor (k = 1)
    assert LANDING_DEBOUNCE_S == 2.0
    assert mon.state['A'] == 'emergency_landing'
    assert mon.events[-1]['t'] == t + 1                        # since the first report below the booking


def test_a_slow_drift_or_a_stopped_descent_is_not_a_landing():
    sky, d = sky_and_flight(layers=4, layer=2)
    mon = monitor(sky)
    t = mid(d)
    x, y, _ = mon.booked_at('A', t)
    for k, (z, vz) in enumerate(((2.5, -0.5), (2.4, -0.1), (2.3, -0.5), (2.2, -0.1), (2.2, 0.0))):
        mon.check(t + k, {}, reports={'A': rep('A', t + k, (x, y, z), vz=vz)})    # run broken by vz > -0.2
    assert mon.state.get('A') is None


def test_a_planned_layer_change_is_not_a_landing():
    """E4 re-plans a drone from 14 m down to 8 m (the layer-change path: StrategicPlanner.replan).
    It descends at 2 m/s, faster than the 0.5 m/s the booking ramps down at, so for several
    seconds it is far below the booked height: the floor is the new layer, not the ramp."""
    sky, d = sky_and_flight(layers=7, layer=6)
    assert sky.layer_z(6) == 14.0 and sky.layer_z(3) == 8.0
    mon = monitor(sky)
    t0 = mid(d)
    x, y, z = mon.booked_at('A', t0)
    assert z == pytest.approx(14.0)
    sky.plans['A'] = replace(sky.plans['A'], layer=3)                    # prefers 8 m now
    d2 = sky.replan('A', (x, y, 14.0), t0)
    assert d2.status == 'approved' and d2.layer == 3 and sky.current['A'] is d2
    for k in range(0, 9):                                                # 14 -> 8 in 3 s, then level
        z = max(8.0, 14.0 - 2.0 * k)
        booked_z = mon.booked_at('A', t0 + k)[2]
        mon.check(t0 + k, {}, reports={'A': rep('A', t0 + k, (x, y, z), vz=-2.0 if z > 8.0 else 0.0)})
        if k == 3:
            assert booked_z - z > 3.0                                    # naive "below the booked height" would fire
    assert mon.state.get('A') is None and mon.events == []


def test_diving_below_the_new_layer_is_still_a_landing():
    sky, d = sky_and_flight(layers=7, layer=6)
    mon = monitor(sky)
    t0 = mid(d)
    x, y, _ = mon.booked_at('A', t0)
    sky.plans['A'] = replace(sky.plans['A'], layer=3)
    assert sky.replan('A', (x, y, 14.0), t0).layer == 3
    for k, z in enumerate((14.0, 12.0, 10.0, 8.0, 6.0, 4.0, 2.0, 0.5)):  # does not stop at 8 m (floor 5 m)
        mon.check(t0 + k, {}, reports={'A': rep('A', t0 + k, (x, y, z), vz=-2.0)})
    assert mon.state['A'] == 'emergency_landing'


def test_descending_onto_its_own_goal_pad_is_not_a_landing():
    sky, d = sky_and_flight(layers=4, layer=2)
    mon = monitor(sky)
    gx, gy, gz = d.waypoints[-1]
    t = d.arrival - 6.0
    for k, z in enumerate((6.0, 4.0, 2.0, 1.2, 1.0, 1.0)):               # the booked landing, over the goal
        mon.check(t + k, {}, reports={'A': rep('A', t + k, (gx + 0.5, gy, z), vz=-0.5)})
    assert mon.state.get('A') is None and mon.events == []


# ------------------------------------------------------------------ F. precedence and stickiness
def test_precedence_down_over_landing_over_lost_link_and_sticky():
    sky, d = sky_and_flight()
    mon = monitor(sky)
    t = mid(d)
    p = mon.booked_at('A', t)
    mon.check(t, {}, reports={'A': rep('A', t, p)})
    mon.check(t + 5, {}, reports={'A': rep('A', t, p)})                       # silent
    assert mon.state['A'] == 'lost_link'
    land = rep('A', t + 6, p, status='landing')
    mon.check(t + 6, {}, reports={'A': land})
    assert mon.state['A'] == 'emergency_landing'
    mon.check(t + 30, {}, reports={'A': land})                                # then quiet: keeps its state
    assert mon.state['A'] == 'emergency_landing'
    down = rep('A', t + 31, p, status='emergency')
    mon.check(t + 31, {}, reports={'A': down})
    assert mon.state['A'] == 'down'
    mon.check(t + 32, {}, reports={'A': rep('A', t + 32, p, status='landing')})   # a later landing report
    mon.check(t + 60, {}, reports={'A': down})
    assert mon.state['A'] == 'down'
    assert states(mon) == ['lost_link', 'emergency_landing', 'down']


def test_a_resumed_report_does_not_clear_down_or_landing():
    sky, d = sky_and_flight()
    mon = monitor(sky)
    t = mid(d)
    p = mon.booked_at('A', t)
    mon.check(t, {}, reports={'A': rep('A', t, p, status='landing')})
    mon.check(t + 1, {}, reports={'A': rep('A', t + 1, p)})                   # status back to ok
    assert mon.state['A'] == 'emergency_landing'


def test_a_failed_drone_is_not_judged_by_its_plan():
    sky, d = sky_and_flight()
    mon = monitor(sky)
    t = mid(d)
    mon.check(t, {'A': (0.0, 20.0, 2.0)}, reports={'A': rep('A', t, (0.0, 20.0, 2.0), status='landing')})
    mon.check(t + 3, {'A': (0.0, 20.0, 2.0)})                                 # off_route, but it is landing
    assert states(mon) == ['emergency_landing']


# ------------------------------------------------------------------ G. booked(): the newest approved decision
def test_booked_is_the_planners_current_approved_decision_only():
    sky, d = sky_and_flight()
    mon = monitor(sky)
    assert mon.booked('A') is d and mon.booked('nobody') is None
    assert mon.last_report('A') is None
    r = rep('A', 10.0, (0, 0, 2.0))
    mon.check(10.0, {}, reports={'A': r})
    assert mon.last_report('A') is r


def test_a_rejected_replan_leaves_the_old_approved_booking_current():
    sky, d = sky_and_flight()
    mon = monitor(sky)
    for pad in ((15, 0), (-15, 0)):                       # neither the goal nor the base is usable
        sky.blocked |= {(c, 0) for c in sky.cells_near(*pad)}
    rejected = sky.replan('A', (-12.0, 0.0, 2.0), 5.0)
    assert rejected.status == 'rejected' and sky.decisions[-1] is rejected
    assert mon.booked('A') is d and mon.booked_at('A', mid(d)) is not None       # not the rejected one


def test_a_cancelled_flight_has_no_booking_not_a_stale_one():
    from utm.airspace_zones import Zone
    from utm.contingency import ContingencyManager
    sky = StrategicPlanner([], B)
    sky.submit(FlightPlan('W', (-15, 0), (15, 0), t_earliest=30.0))
    mon = monitor(sky)
    e4 = ContingencyManager(sky, mon, failures=True)
    old = mon.booked('W')
    assert old is not None and mon.plans['W'] is old
    acts = e4.add_zone(Zone('z', 'Closure', polygon=[(12, -3), (18, -3), (18, 3), (12, 3)]), 5.0, {})
    assert [x['why'] for x in acts] == ['zone z -> cancelled']
    assert 'W' not in sky.current and mon.booked('W') is None and mon.booked_at('W', 40.0) is None
    assert mon.plans['W'] is old                          # the monitor's own copy is stale; it is not used


def test_a_non_approved_decision_in_current_is_ignored():
    sky, d = sky_and_flight()
    mon = monitor(sky)
    sky.current['A'] = Decision('A', 'rejected', reason='x')
    assert mon.booked('A') is None


def test_booked_time_at_is_the_time_the_booking_is_nearest_to_a_position():
    sky, d = sky_and_flight()
    mon = monitor(sky)
    t = mid(d)
    assert mon.booked_time_at('A', mon.booked_at('A', t)) == pytest.approx(t, abs=sky.s.dt)
    assert mon.booked_time_at('nobody', (0, 0, 0)) is None


# ------------------------------------------------------------------ H. no false alarms on healthy drones
def _healthy_fleet(seed=20, noise_m=1.0, delay_s=0.5, loss=0.05, observe=300.0, drones=12):
    """12 healthy drones on synthetic bookings through TelemetryModel's real noise: ids f0..f11, the
    first four cruise at 14 m and, 60 s in, are re-planned to 8 m and descend at 2 m/s (faster than
    the 0.5 m/s the booking ramps at); the others cruise on layers 0-5, all with climb and cruise,
    for `observe` s. Returns the monitor and the margins to each threshold."""
    sky = StrategicPlanner([], ((-300, 300), (-300, 300)), Settings(n_layers=7))
    tm = TelemetryModel(noise_m=noise_m, delay_s=delay_s, loss=loss, seed=seed)
    mon = monitor(sky, noise_m=tm.noise)
    truth = {}
    for k in range(drones):
        fid, layer, y = f'f{k}', (6 if k < 4 else k % 6), -55.0 + 10 * k
        zl = sky.layer_z(layer)
        x0 = -200.0
        x1 = x0 + 1.5 * (observe - 2 * (zl - 1.0) / 0.5 + 40.0)          # still flying at the end
        prof, total = sky.profile([(x0, y), (x1, y)], layer, 1.0)
        prof = np.asarray(prof)
        sky.current[fid] = Decision(fid, 'approved', layer, 0.0, 0.0, total, 0.0,
                                    [(x0, y, 1.0), (x0, y, zl), (x1, y, zl), (x1, y, 1.0)])

        def fly(t, prof=prof):
            return np.array([np.interp(t, prof[:, 0], prof[:, j]) for j in (1, 2, 3)])
        replan = None
        if k < 4:
            t0 = 60.0
            p0 = fly(t0)
            prof2, total2 = sky.profile([(p0[0], y), (x1, y)], 3, 14.0)
            prof2 = np.asarray(prof2)
            d2 = Decision(fid, 'approved', 3, t0, 0.0, t0 + total2, 0.0,
                          [(p0[0], y, 14.0), (p0[0], y, 8.0), (x1, y, 8.0), (x1, y, 1.0)])
            replan = (t0, d2)

            def fly(t, f=fly, prof2=prof2, t0=t0):
                if t < t0:
                    return f(t)
                q = np.array([np.interp(t - t0, prof2[:, 0], prof2[:, j]) for j in (1, 2, 3)])
                q[2] = min(q[2], max(8.0, 14.0 - 2.0 * (t - t0)))               # dives to 8 m at 2 m/s
                return q
        truth[fid] = (fly, replan)
    m = {'min_vz': np.inf, 'min_height_over_floor': np.inf, 'longest_descent_run_s': 0.0, 'max_silence_s': 0.0}
    run_from, heard, gaps = {}, {}, {}          # gaps: (flight, t_sent before the gap) -> reports lost in a row
    t = 0.0
    while t < observe:
        t += 1.0
        states = {}
        for fid, (fly, replan) in truth.items():
            if replan is not None and t >= replan[0]:
                sky.current[fid] = replan[1]
            states[fid] = (fly(t), (fly(t + 0.05) - fly(t - 0.05)) / 0.1, [])
        tm.send(t, states)
        for r in tm.receive(t):
            if r.id in heard and r.seq > heard[r.id].seq + 1:
                gaps[r.id, heard[r.id].t_sent] = r.seq - heard[r.id].seq - 1
            heard[r.id] = r
            d = mon.booked(r.id)
            floor = min(mon.booked_at(r.id, r.t_sent)[2], sky.layer_z(d.layer)) - HEIGHT_MARGIN_M
            m['min_vz'] = min(m['min_vz'], float(r.vel[2]))
            m['min_height_over_floor'] = min(m['min_height_over_floor'], float(r.pos[2]) - floor)
            if r.vel[2] < LANDING_VZ and r.pos[2] < floor:                    # the descent rule's two conditions
                run_from.setdefault(r.id, r.t_sent)
                m['longest_descent_run_s'] = max(m['longest_descent_run_s'], r.t_sent - run_from[r.id])
            else:
                run_from.pop(r.id, None)
        mon.check(t, {}, reports=tm.latest)
        for r in tm.latest.values():
            m['max_silence_s'] = max(m['max_silence_s'], t - r.t_sent)
    m['gaps'] = gaps
    return mon, m


def test_twelve_healthy_drones_through_the_telemetry_noise_raise_no_down_or_landing_event():
    """Seeded (20): GPS noise 1 m (velocity 0.2 m/s), delay 0.5 s, 5% loss; cruising, climbing and a
    14 m -> 8 m layer change at 2 m/s. down and emergency_landing: none, with margin (run with -s):
    single reports do fall below the height floor (3 sd), but never for LANDING_DEBOUNCE_S, and vz
    stays clear of DOWN_VZ. lost_link is separate (next test): it depends on the loss, not the noise."""
    mon, m = _healthy_fleet()
    print('margins:', {k: round(v, 3) for k, v in m.items() if k != 'gaps'},
          '| to DOWN_VZ', round(m['min_vz'] - DOWN_VZ, 3), '| debounce left',
          round(LANDING_DEBOUNCE_S - m['longest_descent_run_s'], 3))
    assert [e for e in mon.events if e['state'] in ('down', 'emergency_landing')] == []
    assert m['min_vz'] > DOWN_VZ                                     # a real fall is far past DOWN_VZ
    assert m['longest_descent_run_s'] < LANDING_DEBOUNCE_S


@pytest.mark.parametrize('seed', [20, 21, 22, 23, 24])
def test_lost_link_events_on_healthy_drones_are_exactly_runs_of_three_lost_reports(seed):
    """A healthy drone is silent for more than LINK_TIMEOUT_S exactly when 3 reports in a row are lost
    (5% loss: 1.25e-4 per report, ~0.45 events per 12 drones x 300 s, so seed 20 may have none: this
    runs 25% loss, where they are common). Each run raises lost_link and the next report clears it;
    nothing else is raised, whatever the seed."""
    mon, m = _healthy_fleet(seed=seed, loss=0.25)
    runs = {k: n for k, n in m['gaps'].items() if n >= 3}
    lost = [(e['flight'], e['t']) for e in mon.events if e['state'] == 'lost_link']
    print(f'seed {seed}, 25% loss: lost_link events {len(lost)}, runs of 3+ lost reports {len(runs)}')
    assert lost and sorted(lost) == sorted(runs)                     # cause: a run of >= 3 lost reports
    assert {e['state'] for e in mon.events} <= {'lost_link', 'ok'}   # no down, no landing, no plan alarm
    for fid in {f for f, _ in lost}:                                 # each cleared by the next report
        ev = [e['state'] for e in mon.events if e['flight'] == fid]
        assert ev[::2] == ['lost_link'] * len(ev[::2]) and ev[1::2] == ['ok'] * len(ev[1::2])


# arrival: a drone that lands on its goal pad and stops reporting
def _landings(seed, cls=ConformanceMonitor, n=40, noise_m=1.0, loss=0.05, settle=1.0):
    """n drones land on their goal pads (layer 1, 4 m), each reporting through the landing and for
    `settle` s on the pad, then quiet; returns the flights still flagged lost_link at the end."""
    sky = StrategicPlanner([], ((-60, 60), (-300, 300)), Settings(n_layers=3))
    tm = TelemetryModel(noise_m=noise_m, delay_s=0.5, loss=loss, seed=seed)
    mon = cls(sky, failures=True, noise_m=tm.noise)
    truth = {}
    for k in range(n):
        fid, y = f'f{k}', -290.0 + 14 * k
        prof, total = sky.profile([(-40.0, y), (40.0, y)], 1, 1.0)
        prof = np.asarray(prof)
        sky.current[fid] = Decision(fid, 'approved', 1, 0.0, 0.0, total, 0.0,
                                    [(-40, y, 1.0), (-40, y, 4.0), (40, y, 4.0), (40, y, 1.0)])
        truth[fid] = (lambda t, prof=prof: np.array([np.interp(t, prof[:, 0], prof[:, j]) for j in (1, 2, 3)]), total)
    t = 0.0
    while t < 200.0:
        t += 1.0
        tm.send(t, {f: (g(t), (g(t + 0.05) - g(t - 0.05)) / 0.1, []) for f, (g, total) in truth.items()
                    if t <= total + settle})
        tm.receive(t)
        mon.check(t, {}, reports=tm.latest)
    return [f for f in truth if mon.state.get(f) == 'lost_link']


class _SingleReportMonitor(ConformanceMonitor):
    """The first closing rule (one report), to show what the mean of three fixes."""

    def _closed_flight(self, fid, d):
        r = self.last_report(fid)
        goal = d.waypoints[-1]
        if np.hypot(*(np.asarray(r.pos[:2], dtype=float) - np.asarray(goal[:2]))) > self.p.s.radius:
            return False
        return float(r.pos[2]) <= goal[2] + CLIMB_DONE_M or r.t_sent >= d.arrival


@pytest.mark.parametrize('seed', [20, 21, 22, 23, 24])
def test_drones_that_land_and_go_quiet_are_not_lost_link_under_gps_noise(seed):
    assert _landings(seed) == []                                     # 40 landings per seed, 200 in all


def test_the_single_report_rule_would_have_flagged_landed_drones():
    flagged = sum(len(_landings(seed, cls=_SingleReportMonitor)) for seed in range(20, 25))
    print('single-report rule: arrived drones flagged lost_link:', flagged, 'of 200')
    assert flagged > 0


def test_a_drone_going_quiet_mid_route_high_and_well_before_arrival_is_still_lost_link():
    sky = StrategicPlanner([], ((-60, 60), (-60, 60)), Settings(n_layers=4, layer_penalty=100.0))
    d = sky.submit(FlightPlan('A', (-50, 0), (50, 0), layer=2))
    tm = TelemetryModel(noise_m=1.0, delay_s=0.5, loss=0.0, seed=20)
    mon = monitor(sky, noise_m=tm.noise)
    t0 = d.t_dep + 0.35 * (d.arrival - d.t_dep)                      # cruising at 6 m, far from the goal pad
    for k in range(4):
        t = float(int(t0) + k)
        tm.send(t, {'A': (mon.booked_at('A', t), (1.5, 0.0, 0.0), [])})
        tm.receive(t)
        mon.check(t, {}, reports=tm.latest)
    assert mon.state.get('A') is None
    quiet = t + 5.0
    mon.check(quiet, {}, reports=tm.latest)
    assert mon.state['A'] == 'lost_link'


def test_the_arrival_test_averages_the_last_three_reports_and_widens_with_the_noise():
    sky, d = sky_and_flight()
    goal = d.waypoints[-1]
    # the last report 6.6 m off the pad (an outlier: alone it fails even with noise_m = 1: 6.6 > 2.1 + 3),
    # the two before on it: the mean is 2.2 m off, outside the radius 2.1 but inside 2.1 + 3 / sqrt(3)
    off = [rep('A', d.arrival - 3.0 + k, (goal[0] + (6.6 if k == 2 else 0.0), goal[1], goal[2] + 0.2)) for k in range(3)]
    mon = monitor(sky, noise_m=1.0)
    for r in off:
        mon.check(r.t_sent, {}, reports={'A': r})
    mon.check(d.arrival + 20.0, {}, reports={'A': off[-1]})
    assert mon.state.get('A') is None                                # mean 2.2 m off, tolerance 3.8: on the pad
    exact = monitor(sky, noise_m=0.0)                                # same reports, exact telemetry (no tolerance): off the pad
    for r in off:
        exact.check(r.t_sent, {}, reports={'A': r})
    exact.check(d.arrival + 20.0, {}, reports={'A': off[-1]})
    assert exact.state['A'] == 'lost_link'


# ------------------------------------------------------------------ I. early landings and the 'landed' status
@pytest.mark.parametrize('early', [10.0, 20.0])
@pytest.mark.parametrize('with_landed', [False, True])
def test_a_drone_landing_early_with_exact_telemetry_is_not_lost_link(early, with_landed):
    """noise_m = 0: the last reports 2.5 / 2.0 / 1.5 m over the pad (mean 2.0, above goal z + CLIMB_DONE_M)
    must not make a landed drone lost_link 3 s later, with or without a final 'landed' report."""
    sky, d = sky_and_flight()
    mon = monitor(sky, noise_m=0.0)
    goal = d.waypoints[-1]
    t0 = d.arrival - early
    for k, h in enumerate((2.5, 2.0, 1.5)):
        r = rep('A', t0 + k, (goal[0], goal[1], goal[2] + h), vz=-0.5)
        mon.check(r.t_sent, {}, reports={'A': r})
    if with_landed:
        r = rep('A', t0 + 3, (goal[0], goal[1], goal[2]), status='landed')
        mon.check(r.t_sent, {}, reports={'A': r})
    for t in (t0 + 6, t0 + 10, d.arrival, d.arrival + 60):
        assert mon.check(t, {}, reports={'A': r}) == []
    assert 'A' not in mon.state and states(mon) == []


def test_final_descent_margin_is_what_covers_the_lost_landed_report():
    sky, d = sky_and_flight()
    goal = d.waypoints[-1]
    top = goal[2] + CLIMB_DONE_M + FINAL_DESCENT_M
    for h, lost in ((top - 0.05, False), (top + 0.2, True)):
        mon = monitor(sky, noise_m=0.0)
        r = rep('A', d.arrival - 20.0, (goal[0], goal[1], h))
        mon.check(r.t_sent, {}, reports={'A': r})
        mon.check(r.t_sent + 10.0, {}, reports={'A': r})
        assert (mon.state.get('A') == 'lost_link') is lost


def test_landed_at_the_goal_closes_the_flight_at_once():
    sky, d = sky_and_flight()
    mon = monitor(sky, noise_m=0.0)
    goal = d.waypoints[-1]
    r = rep('A', d.arrival - 30.0, (goal[0] + 0.5, goal[1], goal[2] + 5.0), status='landed')   # z is not judged
    mon.check(r.t_sent, {}, reports={'A': r})
    assert 'A' in mon._closed
    assert mon.check(r.t_sent + 60.0, {}, reports={'A': r}) == [] and 'A' not in mon.state


def test_landed_away_from_the_goal_latches_emergency_landing_and_is_never_lost_link():
    sky, d = sky_and_flight()
    mon = monitor(sky, noise_m=0.0)
    t = mid(d)
    x, y, _ = mon.booked_at('A', t)
    r = rep('A', t, (x, y, 0.0), status='landed')
    mon.check(t, {}, reports={'A': r})
    assert mon.state['A'] == 'emergency_landing' and 'A' not in mon._closed
    mon.check(t + 60.0, {}, reports={'A': r})
    assert states(mon) == ['emergency_landing']


def test_landed_away_keeps_down():
    sky, d = sky_and_flight()
    mon = monitor(sky, noise_m=0.0)
    t = mid(d)
    x, y, _ = mon.booked_at('A', t)
    mon.check(t, {}, reports={'A': rep('A', t, (x, y, 2.0), status='emergency')})
    mon.check(t + 1, {}, reports={'A': rep('A', t + 1, (x, y, 0.0), status='landed')})
    mon.check(t + 60, {}, reports={'A': rep('A', t + 1, (x, y, 0.0), status='landed')})
    assert states(mon) == ['down'] and mon.state['A'] == 'down'


def test_landed_does_nothing_with_failures_off():
    sky, d = sky_and_flight()
    mon = ConformanceMonitor(sky)
    t = mid(d)
    x, y, _ = mon.booked_at('A', t)
    assert mon.check(t, {}, reports={'A': rep('A', t, (x, y, 0.0), status='landed')}) == []
    assert mon.state == {} and not mon._closed and not mon._latched


@pytest.mark.parametrize('noise', [0.0, 1.0])
def test_a_drone_going_quiet_mid_route_at_cruise_height_is_lost_link_whatever_the_noise(noise):
    sky = StrategicPlanner([], ((-60, 60), (-60, 60)), Settings(n_layers=4, layer_penalty=100.0))
    d = sky.submit(FlightPlan('A', (-50, 0), (50, 0), layer=2))
    tm = TelemetryModel(noise_m=noise, delay_s=0.0, loss=0.0, seed=20)
    mon = monitor(sky, noise_m=noise)
    t0 = d.t_dep + 0.5 * (d.arrival - d.t_dep)
    for k in range(3):
        t = float(int(t0) + k)
        tm.send(t, {'A': (mon.booked_at('A', t), (1.5, 0.0, 0.0), [])})
        tm.receive(t)
        mon.check(t, {}, reports=tm.latest)
    mon.check(t + 5.0, {}, reports=tm.latest)
    assert mon.state['A'] == 'lost_link'
