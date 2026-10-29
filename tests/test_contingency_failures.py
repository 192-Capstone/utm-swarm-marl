"""
test_contingency_failures.py
----------------------------
E4 reactions to drone failures (utm/contingency.py, ContingencyManager(failures=True)) with E3's
failure states (ConformanceMonitor(failures=True)), on hand-made plans (nothing is flown).

  A. The option: ValueError without a failures monitor; off, behaviour is as before.
  B. lost_link: the booking is kept, a volume 'lostlink:<id>' is booked ONCE at onset along the
     whole remaining route (wide in time by the drone's lateness; covers the last heard position and
     the layers it passes between); clashing flights are re-planned then, once; released after
     booked arrival + margin, and when the link is restored.
  C. down: the booking and any volumes are released, and it is logged.
  D. emergency_landing: the route is released, a column 'landing:<id>' is held down to the floor
     for the descent time + hold; clashing flights are re-planned.
  E. Transitions lost_link -> landing -> down leave no orphan volumes.
  F. The guard: a drone in a failure state is never re-planned; _intrusion moves the other flight
     (and neither if both have failed).
  G. Flight ids of any type; log-only actions are marked and are not rejections.

Run: python -m pytest tests/test_contingency_failures.py -q
"""

import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

pytest.importorskip('h3')

from utm.conformance import ConformanceMonitor  # noqa: E402
from utm.contingency import (LANDING_HOLD_S, LOSTLINK_RADIUS_M, LOSTLINK_RELEASE_MARGIN_S,  # noqa: E402
                             LOSTLINK_TIME_SLACK_S, ContingencyManager, cells_within)
from utm.strategic_planner import FlightPlan, Settings, StrategicPlanner  # noqa: E402
from utm.telemetry import Report  # noqa: E402

B_ = ((-25.0, 25.0), (-25.0, 25.0))


def rep(fid, t, pos, vz=0.0, status='ok'):
    return Report(fid, float(t), np.asarray(pos, dtype=float), np.array([0.0, 0.0, vz]), status=status)


def world(layers=2, failures=True, monitor_failures=True):
    """A crosses B's route (E1 times B just after A at the crossing); both on layer 0."""
    sky = StrategicPlanner([], B_, Settings(n_layers=layers, layer_penalty=100.0))
    a = sky.submit(FlightPlan('A', (-15, 0), (15, 0)))
    b = sky.submit(FlightPlan('B', (0, -15), (0, 15)))
    assert a.status == b.status == 'approved'
    mon = ConformanceMonitor(sky, failures=monitor_failures)
    e4 = ContingencyManager(sky, mon, failures=failures)
    return sky, a, b, mon, e4


def mid(d):
    return d.t_dep + 0.5 * (d.arrival - d.t_dep)


def go_quiet(mon, e4, a, positions=None):
    """A is heard once, 4 s before the lost-link check at t = its mid-flight; returns (t, last report)."""
    t = mid(a)
    last = rep('A', t - 4.0, mon.booked_at('A', t - 4.0))
    e4.step(t - 4.0, {}, reports={'A': last})
    return t, last


def owned(sky, owner):
    return {k: [(lo, hi) for lo, hi, o in rows if o == owner] for k, rows in sky.ledger.items()
            if any(o == owner for _, _, o in rows)}


def volumes(sky):
    """The lostlink: / landing: owners that have anything in the ledger."""
    return {o for rows in sky.ledger.values() for _, _, o in rows if str(o).startswith(('lostlink:', 'landing:'))}


def overlap(a, b, after):
    return [k for k, ivs in a.items() for lo, hi in ivs for lo2, hi2 in b.get(k, ())
            if max(lo, lo2, after) < min(hi, hi2)]


def covered(sky, owner, x, y, layer, t):
    """Is the cell of (x, y) on `layer` in the owner's volume at time t?"""
    cell = sky.sky.cell(x, y)
    return any(lo <= t < hi for (c, ly), ivs in owned(sky, owner).items() if ly == layer
               and c == cell for lo, hi in ivs)


# ------------------------------------------------------------------ A. the option
def test_failures_needs_a_failures_monitor():
    sky = StrategicPlanner([], B_)
    with pytest.raises(ValueError):
        ContingencyManager(sky, ConformanceMonitor(sky), failures=True)
    with pytest.raises(ValueError):
        ContingencyManager(sky, None, failures=True)
    ContingencyManager(sky, ConformanceMonitor(sky, failures=True), failures=True)       # fine
    ContingencyManager(sky, ConformanceMonitor(sky))                                      # off: any monitor


def test_off_a_lost_drone_changes_nothing_in_the_ledger():
    sky, a, b, mon, e4 = world(failures=False, monitor_failures=False)
    before = {k: list(v) for k, v in sky.ledger.items()}
    t, last = go_quiet(mon, e4, a)
    assert e4.step(t, {'B': mon.booked_at('B', t)}, reports={'A': last}) == []
    assert sky.ledger == before and volumes(sky) == set() and e4.failures is False


def test_off_with_a_failures_monitor_the_manager_still_does_nothing():
    sky, a, b, mon, e4 = world(failures=False, monitor_failures=True)
    before = {k: list(v) for k, v in sky.ledger.items()}
    t = mid(a)
    last = rep('A', t - 4.0, mon.booked_at('A', t - 4.0))
    mon.check(t - 4.0, {}, reports={'A': last})
    mon.check(t, {}, reports={'A': last})               # E3 (fed directly: step() does not forward reports)
    assert mon.state['A'] == 'lost_link'
    assert e4.step(t, {'B': mon.booked_at('B', t)}) == []
    assert sky.ledger == before                          # E4 (off) does not react
    assert e4._replan('A', t, {'A': (0, 0, 2.0)}, 'x') is not None       # and no guard: as before


# ------------------------------------------------------------------ B. lost_link
def test_lost_link_keeps_the_booking_holds_a_volume_and_replans_the_clash():
    sky, a, b, mon, e4 = world()
    t, last = go_quiet(mon, e4, a)
    a_before, b_before = owned(sky, 'A'), sky.current['B']
    acts = e4.step(t, {'B': mon.booked_at('B', t)}, reports={'A': last})
    assert mon.state['A'] == 'lost_link'
    assert owned(sky, 'A') == a_before and sky.current['A'] is a                 # continues as filed
    assert not [x for x in acts if x['flight'] == 'A' and x['decision'].status == 'approved']   # never re-planned
    vol = owned(sky, 'lostlink:A')
    assert vol and volumes(sky) == {'lostlink:A'}
    moved = [x for x in acts if x['flight'] == 'B']
    assert moved and moved[0]['why'] == 'lost link A' and sky.current['B'] is not b_before
    assert overlap(owned(sky, 'B'), vol, after=t + e4.grace) == []                # B now stays clear of it


def test_the_volume_surrounds_the_booked_route_and_the_last_heard_position():
    sky, a, b, mon, e4 = world()
    t, last = go_quiet(mon, e4, a)
    e4.step(t, {}, reports={'A': last})
    x, y, z = mon.booked_at('A', t + 3.0)
    assert covered(sky, 'lostlink:A', x, y, 0, t + 3.0)
    assert covered(sky, 'lostlink:A', x + 3.0, y, 0, t + 3.0)                     # within LOSTLINK_RADIUS_M
    assert covered(sky, 'lostlink:A', x, y + 3.0, 0, t + 3.0)
    assert not covered(sky, 'lostlink:A', x, y + LOSTLINK_RADIUS_M + 3.0, 0, t + 3.0)     # off the route
    lx, ly, lz = last.pos
    assert covered(sky, 'lostlink:A', lx, ly, 0, t + 8.0)                         # where it was last heard
    end = a.arrival + LOSTLINK_RELEASE_MARGIN_S
    spans = [iv for ivs in owned(sky, 'lostlink:A').values() for iv in ivs]
    assert min(lo for lo, _ in spans) >= a.t_dep - LOSTLINK_TIME_SLACK_S - 1e-9
    assert max(hi for _, hi in spans) <= end + LOSTLINK_TIME_SLACK_S + 1e-9
    assert cells_within(sky, 0.0, 0.0, LOSTLINK_RADIUS_M) and \
        set(sky.cells_near(0.0, 0.0)) <= set(cells_within(sky, 0.0, 0.0, LOSTLINK_RADIUS_M))


def test_the_volume_is_booked_once_at_onset_and_covers_the_whole_remaining_route():
    sky, a, b, mon, e4 = world()
    t, last = go_quiet(mon, e4, a)
    e4.step(t, {}, reports={'A': last})
    first = owned(sky, 'lostlink:A')
    for tau in np.arange(t, a.arrival + LOSTLINK_RELEASE_MARGIN_S, 2.0):          # the route to its end
        x, y, _ = mon.booked_at('A', tau)
        assert covered(sky, 'lostlink:A', x, y, 0, tau)
    for k in (1, 2, 5, 9):
        e4.step(t + k, {}, reports={'A': last})
        assert owned(sky, 'lostlink:A') == first and volumes(sky) == {'lostlink:A'}   # not re-booked, not moved


def test_a_flight_crossing_the_route_30_s_ahead_is_replanned_once_at_onset():
    sky = StrategicPlanner([], ((-100.0, 100.0), (-100.0, 100.0)), Settings(grid_res=0.5))
    a = sky.submit(FlightPlan('A', (-90, 0), (90, 0)))
    b = sky.submit(FlightPlan('B', (75, -15), (75, 15), t_earliest=100.0))       # crosses at x = 75
    mon = ConformanceMonitor(sky, failures=True)
    e4 = ContingencyManager(sky, mon, failures=True)
    cross = next(tau for tau in np.arange(b.t_dep, b.arrival, 0.25) if abs(mon.booked_at('B', tau)[1]) < 0.3)
    t = cross - 30.0                                                              # onset: B crosses 30 s later
    last = rep('A', t - 4.0, mon.booked_at('A', t - 4.0))
    e4.step(t - 4.0, {}, reports={'A': last})
    old_b = sky.current['B']
    got = []
    for k in range(0, 40):                                                        # and on, to well past the crossing
        got += [x for x in e4.step(t + k, {}, reports={'A': last}) if x['flight'] == 'B']
    assert mon.state['A'] == 'lost_link'
    assert len(got) == 1 and got[0]['why'] == 'lost link A' and got[0]['t'] == t  # exactly once, at onset
    assert sky.current['B'] is not old_b and sky.current['B'].status == 'approved'
    assert overlap(owned(sky, 'B'), owned(sky, 'lostlink:A'), after=t) == []


def test_a_drone_5_s_late_at_link_loss_is_inside_the_volume_at_its_real_position():
    sky, a, b, mon, e4 = world()
    t = mid(a)
    t_last = t - 4.0
    last = rep('A', t_last, mon.booked_at('A', t_last - 5.0))                     # 5 s behind its booking
    assert t_last - mon.booked_time_at('A', last.pos) == pytest.approx(5.0, abs=0.3)
    e4.step(t_last, {}, reports={'A': last})
    e4.step(t, {}, reports={'A': last})
    for tau in (t, t + 2.0, t + 6.0, t + 12.0):                                   # it keeps flying 5 s behind
        x, y, z = mon.booked_at('A', tau - 5.0)
        assert covered(sky, 'lostlink:A', x, y, sky.layer_of(z), tau), tau
    # and a drone 12 s late is not assumed to be where the 5 s slack reaches: the slack follows the lateness
    sky2, a2, _, mon2, e42 = world()
    late = rep('A', t_last, mon2.booked_at('A', t_last - 12.0))
    e42.step(t_last, {}, reports={'A': late})
    e42.step(t, {}, reports={'A': late})
    x, y, z = mon2.booked_at('A', t + 3.0 - 12.0)
    assert covered(sky2, 'lostlink:A', x, y, sky2.layer_of(z), t + 3.0)


def test_the_volume_covers_the_layers_the_booking_passes_between():
    sky = StrategicPlanner([], B_, Settings(n_layers=3, layer_penalty=100.0))
    a = sky.submit(FlightPlan('A', (-15, 0), (15, 0), layer=1))      # climbs 1 m -> 4 m at 0.5 m/s
    assert a.layer == 1
    mon = ConformanceMonitor(sky, failures=True)
    e4 = ContingencyManager(sky, mon, failures=True)
    t = a.t_dep + 4.0
    last = rep('A', t - 4.0, mon.booked_at('A', t - 4.0))
    e4.step(t - 4.0, {}, reports={'A': last})
    e4.step(t, {}, reports={'A': last})
    layers = {ly for (_, ly) in owned(sky, 'lostlink:A')}
    assert layers == {0, 1}                                            # climbing from layer 0 into 1, never 2


def test_the_volume_is_released_after_the_booked_arrival_plus_the_margin():
    sky, a, b, mon, e4 = world()
    t, last = go_quiet(mon, e4, a)
    e4.step(t, {}, reports={'A': last})
    late = a.arrival + 1.0                                # booking ends soon: the volume stops at arrival + margin
    e4.step(late, {}, reports={'A': last})
    assert volumes(sky) == {'lostlink:A'}
    assert max(hi for ivs in owned(sky, 'lostlink:A').values() for _, hi in ivs) <= \
        a.arrival + LOSTLINK_RELEASE_MARGIN_S + LOSTLINK_TIME_SLACK_S + 1e-9
    done = a.arrival + LOSTLINK_RELEASE_MARGIN_S + 0.5
    acts = e4.step(done, {}, reports={'A': last})
    assert volumes(sky) == set() and [x['why'] for x in acts] == ['lost link: released after its booked arrival']
    assert e4.step(done + 5, {}, reports={'A': last}) == [] and volumes(sky) == set()   # and not re-booked


def test_the_volume_is_released_when_the_link_is_restored():
    sky, a, b, mon, e4 = world()
    t, last = go_quiet(mon, e4, a)
    e4.step(t, {}, reports={'A': last})
    assert volumes(sky) == {'lostlink:A'}
    back = rep('A', t + 2.0, mon.booked_at('A', t + 2.0))
    e4.step(t + 2.1, {}, reports={'A': back})
    assert mon.state['A'] == 'ok' and volumes(sky) == set()
    assert owned(sky, 'A')                                 # its own booking was never touched


# ------------------------------------------------------------------ C. down
def test_down_releases_the_remaining_booking_and_logs_it():
    sky, a, b, mon, e4 = world()
    t = mid(a)
    pos = mon.booked_at('A', t)
    acts = e4.step(t, {'B': mon.booked_at('B', t)}, reports={'A': rep('A', t, pos, status='emergency')})
    assert mon.state['A'] == 'down'
    assert [(x['flight'], x['why']) for x in acts] == [('A', 'down: booking released')]
    assert all(hi <= t + 1e-9 for ivs in owned(sky, 'A').values() for _, hi in ivs)     # nothing left after t
    assert volumes(sky) == set()
    assert e4.step(t + 1, {}, reports={'A': rep('A', t, pos, status='emergency')}) == []   # logged once


def test_down_after_lost_link_drops_its_volume():
    sky, a, b, mon, e4 = world()
    t, last = go_quiet(mon, e4, a)
    e4.step(t, {}, reports={'A': last})
    assert volumes(sky) == {'lostlink:A'}
    e4.step(t + 1, {}, reports={'A': rep('A', t + 1, last.pos, status='emergency')})
    assert mon.state['A'] == 'down' and volumes(sky) == set()


# ------------------------------------------------------------------ D. emergency_landing
def _landing_world():
    sky = StrategicPlanner([], B_, Settings(n_layers=3, layer_penalty=100.0))
    a = sky.submit(FlightPlan('A', (-15, 0), (15, 0), layer=1))
    b = sky.submit(FlightPlan('B', (0, -15), (0, 15), layer=1))
    mon = ConformanceMonitor(sky, failures=True)
    return sky, a, b, mon, ContingencyManager(sky, mon, failures=True)


def test_landing_releases_the_route_and_holds_a_column_down_to_the_floor():
    sky, a, b, mon, e4 = _landing_world()
    t = mid(a)
    x, y, z = mon.booked_at('A', t)
    assert z == pytest.approx(4.0)
    acts = e4.step(t, {'B': mon.booked_at('B', t)}, reports={'A': rep('A', t, (x, y, z), status='landing')})
    assert mon.state['A'] == 'emergency_landing'
    assert all(hi <= t + 1e-9 for ivs in owned(sky, 'A').values() for _, hi in ivs)     # route released
    col = owned(sky, 'landing:A')
    assert {ly for (_, ly) in col} == {0, 1} and volumes(sky) == {'landing:A'}          # layer 1 down to the floor
    end = t + (z - sky.s.layer_floor) / min(sky.s.v_climb, e4.landing_descent_ms) + LANDING_HOLD_S
    assert end == pytest.approx(t + 6.0 + 7.0)
    assert max(hi for ivs in col.values() for _, hi in ivs) == pytest.approx(end)
    assert all(lo == pytest.approx(t) for ivs in col.values() for lo, _ in ivs)
    assert covered(sky, 'landing:A', x, y, 0, t + 1) and covered(sky, 'landing:A', x, y, 1, t + 12)
    assert not covered(sky, 'landing:A', x, y, 1, end + 0.1)
    assert acts[0]['why'].startswith('emergency landing')
    assert not [q for q in acts if q['flight'] == 'A' and q['decision'].status == 'approved']
    for q in acts:                                          # whoever was re-planned stays out of the column
        if q['flight'] == 'B':
            assert q['why'] == 'landing A'
            assert overlap(owned(sky, 'B'), col, after=t + e4.grace) == []


def test_the_landing_column_ends_with_its_time():
    sky, a, b, mon, e4 = _landing_world()
    t = mid(a)
    pos = mon.booked_at('A', t)
    last = rep('A', t, pos, status='landing')
    e4.step(t, {}, reports={'A': last})
    assert volumes(sky) == {'landing:A'}
    e4.step(t + 13.5, {}, reports={'A': rep('A', t + 13.5, pos, status='landing')})
    assert volumes(sky) == set() and mon.state['A'] == 'emergency_landing'


@pytest.mark.parametrize('speed, per_s', [(0.5, 0.5), (0.25, 0.25), (1.0, 0.5)])
def test_the_descent_time_uses_the_slower_of_v_climb_and_landing_descent_ms(speed, per_s):
    sky, a, b, mon, _ = _landing_world()
    e4 = ContingencyManager(sky, mon, failures=True, landing_descent_ms=speed)
    assert e4.landing_descent_ms == speed
    t = mid(a)
    e4.step(t, {}, reports={'A': rep('A', t, mon.booked_at('A', t), status='landing')})
    assert math.isclose(max(hi for ivs in owned(sky, 'landing:A').values() for _, hi in ivs),
                        t + (4.0 - sky.s.layer_floor) / per_s + LANDING_HOLD_S)


def test_landing_descent_ms_must_be_positive():
    sky = StrategicPlanner([], B_)
    with pytest.raises(ValueError):
        ContingencyManager(sky, landing_descent_ms=0.0)


# ------------------------------------------------------------------ E. transitions
def test_lost_link_to_landing_to_down_leaves_no_orphan_volume():
    sky, a, b, mon, e4 = _landing_world()
    t = mid(a)
    pos = mon.booked_at('A', t)
    last = rep('A', t - 4.0, mon.booked_at('A', t - 4.0))
    e4.step(t - 4.0, {}, reports={'A': last})
    e4.step(t, {}, reports={'A': last})
    assert mon.state['A'] == 'lost_link' and volumes(sky) == {'lostlink:A'}
    e4.step(t + 1, {}, reports={'A': rep('A', t + 1, pos, status='landing')})
    assert mon.state['A'] == 'emergency_landing' and volumes(sky) == {'landing:A'}
    e4.step(t + 2, {}, reports={'A': rep('A', t + 2, pos, status='emergency')})
    assert mon.state['A'] == 'down' and volumes(sky) == set()
    e4.step(t + 3, {}, reports={'A': rep('A', t + 2, pos, status='emergency')})
    assert volumes(sky) == set()
    assert [x['why'].split(':')[0] for x in e4.actions if x['flight'] == 'A'
            and x.get('kind') == 'log'][:3] == ['lost link', 'emergency landing', 'down']


# ------------------------------------------------------------------ F. the guard and _intrusion
def test_replan_refuses_flights_in_a_failure_state():
    sky, a, b, mon, e4 = world()
    pos = mon.booked_at('A', mid(a))
    for state in ('lost_link', 'down', 'emergency_landing'):
        mon.state['A'] = state
        assert e4._replan('A', mid(a), {'A': pos}, 'x') is None
    assert sky.current['A'] is a and e4.actions == []
    mon.state['A'] = 'ok'
    assert e4._replan('A', mid(a), {'A': pos}, 'x') is not None         # the guard is only for failures


def _intrusion_world(a_state=None, b_state=None):
    sky, a, b, mon, e4 = world()
    mon.events.append({'t': 1.0, 'flight': 'A', 'state': 'intrusion', 'risk': 'high', 'key': None, 'other': 'B'})
    if a_state:
        mon.state['A'] = a_state
    else:
        mon.state['A'] = 'intrusion'
    if b_state:
        mon.state['B'] = b_state
    return sky, a, b, mon, e4


def test_intrusion_by_a_failed_drone_replans_the_other_flight():
    sky, a, b, mon, e4 = _intrusion_world(a_state='lost_link')
    t = mid(b)
    acts = e4._intrusion('A', t, {'A': mon.booked_at('A', t), 'B': mon.booked_at('B', t)})
    assert [(x['flight'], x['why']) for x in acts] == [('B', 'intrusion by A')]
    assert acts[0]['decision'].status == 'approved'
    assert sky.current['A'] is a


def test_intrusion_with_both_flights_failed_replans_neither():
    sky, a, b, mon, e4 = _intrusion_world(a_state='lost_link', b_state='emergency_landing')
    t = mid(b)
    acts = e4._intrusion('A', t, {'A': mon.booked_at('A', t), 'B': mon.booked_at('B', t)})
    assert len(acts) == 1 and acts[0]['kind'] == 'log' and 'neither' in acts[0]['why']
    assert sky.current['A'] is a and sky.current['B'] is b


def test_intrusion_by_a_healthy_drone_into_a_failed_ones_air_replans_the_intruder():
    """The other flight can't move (failed): the old fallback re-plans the intruder itself."""
    sky, a, b, mon, e4 = _intrusion_world(b_state='lost_link')
    t = mid(b)
    acts = e4._intrusion('A', t, {'A': mon.booked_at('A', t), 'B': mon.booked_at('B', t)})
    assert [x['flight'] for x in acts] == ['A'] and acts[0]['why'] == 'intrusion'


# ------------------------------------------------------------------ G. flight ids, log entries
def test_flight_ids_of_any_type_lost_link_landing_and_down_together():
    """Ids 1, 2, 3 are ints, 'B' a str: nothing sorts, parses or compares ids, owners use str(id)."""
    sky = StrategicPlanner([], B_, Settings(n_layers=3, layer_penalty=100.0))
    for fid, y in ((1, 0), (2, 10), (3, -10)):
        assert sky.submit(FlightPlan(fid, (-15, y), (15, y), layer=1)).status == 'approved'
    assert sky.submit(FlightPlan('B', (0, -20), (0, 20), layer=1)).status == 'approved'
    mon = ConformanceMonitor(sky, failures=True)
    e4 = ContingencyManager(sky, mon, failures=True)
    t = mid(sky.current[1])
    heard = {f: rep(f, t - 4.0, mon.booked_at(f, t - 4.0)) for f in (1, 2, 3)}
    e4.step(t - 4.0, {}, reports=heard)
    now = {1: heard[1],                                                           # silent
           2: rep(2, t, mon.booked_at(2, t), status='landing'),
           3: rep(3, t, mon.booked_at(3, t), status='emergency')}
    acts = e4.step(t, {'B': mon.booked_at('B', t)}, reports=now)
    assert {f: mon.state[f] for f in (1, 2, 3)} == {1: 'lost_link', 2: 'emergency_landing', 3: 'down'}
    assert e4._owners == {'lostlink:1': 1, 'landing:2': 2}                        # no owner for the fallen one
    assert volumes(sky) == {'lostlink:1', 'landing:2'}
    logged = {x['flight']: x['why'].split(':')[0] for x in acts if x.get('kind') == 'log'}
    assert logged == {1: 'lost link', 2: 'emergency landing', 3: 'down'}
    assert sky.current[1].status == 'approved' and not any(x['flight'] in (1, 2, 3) and 'kind' not in x for x in acts)
    back = {1: rep(1, t + 2.0, mon.booked_at(1, t + 2.0))}
    e4.step(t + 2.1, {}, reports={**now, **back})                                 # link of 1 restored
    assert mon.state[1] == 'ok' and e4._owners == {'landing:2': 2}
    e4.step(t + 3.0, {}, reports={**now, 3: rep(3, t + 2.5, (0, -10, 0.0), status='emergency'),
                                  2: rep(2, t + 2.5, mon.booked_at(2, t), status='emergency')})
    assert mon.state[2] == 'down' and e4._owners == {} and volumes(sky) == set()


def test_log_entries_are_marked_and_are_not_rejections():
    sky, a, b, mon, e4 = world()
    t, last = go_quiet(mon, e4, a)
    acts = e4.step(t, {'B': mon.booked_at('B', t)}, reports={'A': last})
    logs = [x for x in acts if x.get('kind') == 'log']
    real = [x for x in acts if x.get('kind') is None]
    assert logs and real                                    # the note, and B's re-plan
    assert all(x['decision'].status == 'logged' for x in logs)
    assert all(x['decision'].status in ('approved', 'rejected') for x in real)
    assert not [x for x in acts if x['decision'].status == 'rejected' and x.get('kind') == 'log']
