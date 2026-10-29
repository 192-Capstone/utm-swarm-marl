"""
test_strategic_planner.py
-------------------------
E1 strategic planner (utm/strategic_planner.py), plan-only (nothing is flown).

  A. Crossing flights in the same layer at the same time: the second is delayed or moved, and
     afterwards no key is booked twice with overlapping times.
  B. The same crossing in different layers, or far apart in time: both approved on time.
  C. The reservation covers the whole profile and the pads: a short window around take-off and
     landing ('window', default), or the whole episode ('episode': drones hover on their pads).
  D. Blocked keys, max_delay and rejection; buildings shape the route; deterministic.

Run: python -m pytest tests/test_strategic_planner.py -q
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

pytest.importorskip('h3')

from utm.strategic_planner import FlightPlan, Settings, StrategicPlanner  # noqa: E402

B = ((-25.0, 25.0), (-25.0, 25.0))
WALL = {'type': 'box', 'center': (0.0, 0.0), 'half': (0.5, 8.0), 'height': 10.0}


def no_double_booking(sky):
    for key, ivs in sky.ledger.items():
        for i, (lo1, hi1, f1) in enumerate(ivs):
            for lo2, hi2, f2 in ivs[i + 1:]:
                if f1 != f2:
                    assert not (lo1 < hi2 and lo2 < hi1), (key, f1, f2)


def test_crossing_same_layer_same_time_is_resolved():
    sky = StrategicPlanner([], B)
    a = sky.submit(FlightPlan('A', (-15, 0), (15, 0)))
    b = sky.submit(FlightPlan('B', (0, -15), (0, 15)))
    assert a.status == b.status == 'approved'
    assert a.delay == 0 and a.layer == 0
    assert b.delay > 0 or b.layer != 0
    no_double_booking(sky)


def test_different_layers_or_times_need_no_change():
    sky = StrategicPlanner([], B)
    sky.submit(FlightPlan('A', (-15, 0), (15, 0), layer=0))
    b = sky.submit(FlightPlan('B', (0, -15), (0, 15), layer=1))
    assert (b.status, b.layer, b.delay) == ('approved', 1, 0.0)
    c = sky.submit(FlightPlan('C', (-15, 5), (15, 5), layer=0, t_earliest=200.0))
    assert (c.status, c.layer, c.delay) == ('approved', 0, 0.0)
    no_double_booking(sky)


def test_cheaper_option_wins_delay_vs_layer():
    """Layer change costs its extra climb time plus the penalty; a big penalty means waiting,
    no penalty means the cheaper of the two."""
    wait = StrategicPlanner([], B, Settings(layer_penalty=1e6))
    wait.submit(FlightPlan('A', (-15, 0), (15, 0)))
    assert wait.submit(FlightPlan('B', (0, -15), (0, 15))).layer == 0
    move = StrategicPlanner([], B, Settings(layer_penalty=0.0, max_delay=0.0))
    move.submit(FlightPlan('A', (-15, 0), (15, 0)))
    b = move.submit(FlightPlan('B', (0, -15), (0, 15)))
    assert (b.layer, b.delay) == (1, 0.0)


def test_reservation_covers_the_profile_and_the_pads():
    sky = StrategicPlanner([], B)
    a = sky.submit(FlightPlan('A', (-15, 0), (15, 0), t_earliest=5.0))
    r = sky.route((-15, 0), (15, 0), a.layer)
    prof, total = sky.profile(r, a.layer)
    for tau, x, y, z in prof:
        key = (sky.sky.cell(x, y), sky.sky.layer(z))
        assert any(lo <= a.t_dep + tau <= hi for lo, hi in a.reservation[key])
    pad = sky.sky.layer(sky.s.start_z)
    start_key = (sky.sky.cell(-15, 0), pad)
    goal_key = (sky.sky.cell(15, 0), pad)
    assert any(lo <= a.t_dep - sky.s.pad_before + 1e-9 for lo, hi in a.reservation[start_key])
    assert any(hi >= a.arrival + sky.s.pad_after - 1e-9 and hi < float('inf')
               for lo, hi in a.reservation[goal_key])                                  # landed, then gone


def test_pad_modes():
    """'window': a drone is on the ground before and after its flight, so a later flight can pass
    over its landed goal in the same layer, on time. 'episode': it hovers there for good, so the
    later flight can't wait it out and must change layer."""
    for mode, layer in (('window', 0), ('episode', 1)):
        sky = StrategicPlanner([], B, Settings(pad_mode=mode))
        sky.submit(FlightPlan('A', (-15, 0), (0, 0)))
        b = sky.submit(FlightPlan('B', (0, -15), (0, 15), t_earliest=60.0))
        assert (b.status, b.layer, b.delay) == ('approved', layer, 0.0), mode
        no_double_booking(sky)
    with pytest.raises(ValueError):
        StrategicPlanner([], B, Settings(pad_mode='hover'))


def test_every_key_is_in_a_real_layer():
    """Climb and descent start and end exactly on the floor (start_z); rounding must not book
    anything in layer -1 or above the top layer, where nobody else would check it. (The descent
    used to dip a hair under start_z through rounding; it now ends on it exactly.)"""
    import numpy as np
    sky = StrategicPlanner([], B)
    rng = np.random.RandomState(0)
    for i in range(6):
        a, b = tuple(rng.uniform(-20, 20, 2)), tuple(rng.uniform(-20, 20, 2))
        prof, _ = sky.profile([a, b], 1)
        assert min(z for *_, z in prof) >= sky.s.start_z - 1e-9
        assert sky.submit(FlightPlan(f'd{i}', a, b, layer=1, t_earliest=60.0 * i)).status == 'approved'
    layers = {layer for (_, layer) in sky.ledger}
    assert layers <= set(range(sky.s.n_layers)), layers


def test_blocked_keys_and_max_delay_reject():
    sky = StrategicPlanner([], B)
    goal_cells = sky.cells_near(15, 0)
    sky.blocked = {(c, layer) for c in goal_cells for layer in range(sky.s.n_layers)}
    assert sky.submit(FlightPlan('A', (-15, 0), (15, 0))).status == 'rejected'
    tight = StrategicPlanner([], B, Settings(n_layers=1, max_delay=0.0))
    tight.submit(FlightPlan('A', (-15, 0), (15, 0)))
    assert tight.submit(FlightPlan('B', (0, -15), (0, 15))).status == 'rejected'
    assert tight.summary()['approved'] == 1


def test_buildings_shape_the_route():
    sky = StrategicPlanner([WALL], B)
    a = sky.submit(FlightPlan('A', (-8, 0), (8, 0)))
    assert a.status == 'approved'
    assert len(a.waypoints) > 4                       # pad, climb, a detour around the wall, goal


def test_deterministic():
    def run():
        sky = StrategicPlanner([WALL], B)
        out = [sky.submit(FlightPlan(f'f{i}', (-15, -10 + 4 * i), (15, 10 - 4 * i))) for i in range(6)]
        return [(d.status, d.layer, d.t_dep) for d in out]
    assert run() == run()


def test_a_mid_air_replan_never_steps_down_onto_a_roof():
    """Over a 12 m building at 15 m, a re-plan may not drop the drone to a layer below the roof
    there (the drop is straight down); in the open it may."""
    roof = {'type': 'box', 'center': (0.0, 0.0), 'half': (4.0, 4.0), 'height': 12.0, 'yaw': 0.0}
    sky = StrategicPlanner([roof], ((-30, 30), (-30, 30)),
                           Settings(layer_floor=1.0, n_layers=8, layers_open=(2, 7), layer_penalty=0.0))
    assert not sky._clear_below((0.0, 0.0), 2) and sky._clear_below((20.0, 0.0), 2)
    assert sky._clear_below((0.0, 0.0), 7)                         # 16 m: over the roof
    over = sky._submit(FlightPlan('a', (0.0, 0.0), (25.0, 25.0), layer=2), z_start=15.0, pad_start=False)
    assert over.status == 'approved' and over.layer == 7


def test_traffic_detour_swerves_on_the_same_layer_instead_of_changing_layer():
    """A crossing flight that may not wait (as mid-air) gets a same-layer detour around the air the
    first one has booked when it passes, rather than another layer; off, it changes layer."""
    for detour in (False, True):
        sky = StrategicPlanner([], B, Settings(traffic_detour=detour, layer_penalty=30.0))
        a = sky.submit(FlightPlan('A', (-15, 0), (15, 0)))
        b = sky._submit(FlightPlan('B', (0, -15), (0, 15)), max_delay=0.0)
        assert a.status == b.status == 'approved'
        if detour:
            assert (b.layer, b.delay, b.around_traffic) == (0, 0.0, True)
            assert max(abs(x) for x, _, _ in b.waypoints) > 2.0            # it really went round
        else:
            assert b.layer == 1 and not b.around_traffic
        no_double_booking(sky)


def test_traffic_detour_ignores_air_booked_at_other_times():
    """A cell someone else uses long before or after doesn't count: the route stays straight."""
    sky = StrategicPlanner([], B, Settings(traffic_detour=True))
    sky.submit(FlightPlan('A', (-15, 0), (15, 0)))
    b = sky.submit(FlightPlan('B', (0, -15), (0, 15), t_earliest=300.0))
    assert (b.layer, b.delay, b.around_traffic) == (0, 0.0, False)
