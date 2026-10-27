"""
test_demand.py
--------------
D4 traffic demand (utm/demand.py):

  A. Departures: Poisson count at rate x area / 2,500 m2 per minute, a peak adds flights near
     its centre, times sorted inside [0, duration).
  B. Places: random requests lie in the largest connected free region, trips >= min_trip
     (<= max_trip when set); hubs mode sends hub_share of the flights out of the depots.
  C. Routability: unroutable requests are dropped and counted, or kept with keep_unroutable.
  D. Operators follow their shares; same seed -> the same demand; save / load round trip.
  E. It feeds E1, and random_requests (the E1 preview's generator) still works from here.
  F. The full E-City map.

Run: python -m pytest tests/test_demand.py -q
"""

import json
import os
import sys

import numpy as np
import pytest

pytest.importorskip('h3')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from envs.city_generator import generate_city                                      # noqa: E402
from envs.route_planner import plan                                                # noqa: E402
from utm.demand import (DemandSettings, Pad, Region, departure_times,  # noqa: E402
                        generate_demand, layer0_grid, load_demand, random_requests, save_demand)
from utm.strategic_planner import FlightPlan, StrategicPlanner                     # noqa: E402

B = ((-25.0, 25.0), (-25.0, 25.0))


def city(level='medium', seed=0):
    c = generate_city(np.random.RandomState(seed), level)
    W, H = c['size']
    return c['layout'], ((-W / 2, W / 2), (-H / 2, H / 2))


def walled():
    """An empty 50 m map with a closed box (walls 1 m thick) around (15, 15): its inside is free
    but cut off, so the largest region is everything outside the box."""
    walls = [{'type': 'box', 'center': (15.0, 20.0), 'half': (5.0, 0.5), 'height': 10.0},
             {'type': 'box', 'center': (15.0, 10.0), 'half': (5.0, 0.5), 'height': 10.0},
             {'type': 'box', 'center': (10.0, 15.0), 'half': (0.5, 5.0), 'height': 10.0},
             {'type': 'box', 'center': (20.0, 15.0), 'half': (0.5, 5.0), 'height': 10.0}]
    return walls, B


# ----------------------------------------------------------------------------- A. time
def test_poisson_count_scales_with_rate_and_area():
    rng = np.random.RandomState(0)
    n = [len(departure_times(rng, 20.0, 120.0)) for _ in range(400)]
    assert abs(np.mean(n) - 40.0) < 1.0                         # 20 / min x 2 min
    assert abs(np.var(n) - 40.0) < 8.0                          # Poisson: variance = mean
    t = departure_times(rng, 20.0, 120.0)
    assert np.all(np.diff(t) >= 0) and t.min() >= 0 and t.max() < 120.0
    # the map area sets the per-map rate: 100 x 50 m = 2 districts
    d1 = generate_demand([], B, DemandSettings(rate=10, duration_s=60), seed=1)
    d2 = generate_demand([], ((-50.0, 50.0), (-25.0, 25.0)), DemandSettings(rate=10, duration_s=60), seed=1)
    assert d1.report['districts'] == pytest.approx(1.0) and d2.report['districts'] == pytest.approx(2.0)
    assert d2.report['rate_per_min_map'] == pytest.approx(2 * d1.report['rate_per_min_map'])


def test_peak_adds_flights_near_its_centre():
    rng = np.random.RandomState(1)
    flat = np.concatenate([departure_times(rng, 20.0, 120.0) for _ in range(200)])
    peak = np.concatenate([departure_times(rng, 20.0, 120.0, (60.0, 10.0, 3.0)) for _ in range(200)])
    near = lambda t: np.mean((t > 50) & (t < 70)) * len(t) / 200   # flights per run near the centre
    far = lambda t: np.mean(t < 20) * len(t) / 200
    assert near(peak) > 2.2 * near(flat)                        # factor 3 at the centre
    assert abs(far(peak) - far(flat)) < 0.6                     # unchanged far from it
    with pytest.raises(ValueError):
        departure_times(rng, 20.0, 120.0, (60.0, 0.0, 3.0))


# ----------------------------------------------------------------------------- B. places
def test_region_is_the_largest_connected_part():
    layout, bounds = walled()
    g = layer0_grid(layout, bounds)
    r = Region(g)
    assert r.n_cells < r.free_cells                             # the box's inside is left out
    assert not r.contains((15.0, 15.0)) and r.contains((-15.0, -15.0))
    assert not r.contains((100.0, 0.0))
    rng = np.random.RandomState(0)
    for _ in range(200):
        assert r.contains(r.sample(rng))
    # uniform over the region: on an empty map, half the samples fall in the left half
    e = Region(layer0_grid([], B))
    xs = np.array([e.sample(rng)[0] for _ in range(4000)])
    assert e.n_cells == e.free_cells and abs(np.mean(xs < 0) - 0.5) < 0.03


def test_random_requests_are_connected_and_long_enough():
    layout, bounds = city('hard', 3)
    s = DemandSettings(rate=20, duration_s=60)
    d = generate_demand(layout, bounds, s, seed=4)
    g = layer0_grid(layout, bounds)
    r = Region(g)
    assert d.report['kept'] == len(d.plans) > 5
    for fp in d.plans:
        assert isinstance(fp, FlightPlan)
        assert r.contains(fp.start) and r.contains(fp.goal)
        assert np.hypot(fp.goal[0] - fp.start[0], fp.goal[1] - fp.start[1]) >= s.min_trip
        assert plan(g, fp.start, fp.goal) is not None
    assert [fp.t_earliest for fp in d.plans] == sorted(fp.t_earliest for fp in d.plans)
    assert len({fp.id for fp in d.plans}) == len(d.plans)


def test_max_trip_and_pad_separation():
    layout, bounds = city('easy', 1)
    s = DemandSettings(rate=40, duration_s=60, max_trip=15.0)
    d = generate_demand(layout, bounds, s, seed=2)
    L = [np.hypot(fp.goal[0] - fp.start[0], fp.goal[1] - fp.start[1]) for fp in d.plans]
    assert len(L) > 10 and min(L) >= s.min_trip and max(L) <= s.max_trip + 0.25
    for i, a in enumerate(d.plans):                             # pads of flights close in time apart
        for b in d.plans[i + 1:]:
            if b.t_earliest - a.t_earliest >= s.pad_window_s:
                break
            for p in (a.start, a.goal):
                for q in (b.start, b.goal):
                    assert np.hypot(p[0] - q[0], p[1] - q[1]) >= s.pad_sep


def test_hubs_mode():
    layout, bounds = city('medium', 2)
    s = DemandSettings(rate=80, duration_s=120, mode='hubs', n_hubs=2, n_drops=12, hub_share=0.8)
    d = generate_demand(layout, bounds, s, seed=5)
    hubs = {(p.x, p.y) for p in d.pads if p.kind == 'hub'}
    drops = {(p.x, p.y) for p in d.pads if p.kind == 'drop'}
    assert len(hubs) == 2 and len(drops) == 12
    out = sum(fp.start in hubs and fp.goal in drops for fp in d.plans)
    back = sum(fp.start in drops and fp.goal in hubs for fp in d.plans)
    assert out + back == len(d.plans) > 100
    assert abs(out / len(d.plans) - 0.8) < 0.1
    # given pads: kind 'hub' are the depots
    pads = [(-20.0, -20.0, 'hub'), Pad(18.0, 18.0, 'drop'), (18.0, -18.0)]
    with pytest.raises(ValueError):
        generate_demand([], B, DemandSettings(mode='hubs'), pads=[(0.0, 0.0), (10.0, 0.0)])
    d = generate_demand([], B, DemandSettings(rate=20, duration_s=60, mode='hubs'), seed=1, pads=pads)
    assert all((-20.0, -20.0) in (fp.start, fp.goal) for fp in d.plans)


# ----------------------------------------------------------------------------- C. routability
def test_unroutable_requests_are_dropped_and_counted():
    layout, bounds = walled()
    pads = [(-20.0, -20.0), (20.0, -20.0), (15.0, 15.0)]       # (15, 15) is inside the closed box
    s = DemandSettings(rate=30, duration_s=60)
    d = generate_demand(layout, bounds, s, seed=0, pads=pads)
    assert d.report['dropped_unroutable'] > 0
    assert d.report['kept'] == len(d.plans) == d.report['requested'] - d.report['dropped_unroutable']
    assert all((15.0, 15.0) not in (fp.start, fp.goal) for fp in d.plans)
    raw = generate_demand(layout, bounds, DemandSettings(rate=30, duration_s=60, keep_unroutable=True),
                          seed=0, pads=pads)
    assert raw.report['dropped_unroutable'] == 0 and raw.report['unroutable_kept'] == d.report['dropped_unroutable']
    assert len(raw.plans) == raw.report['requested']


# ----------------------------------------------------------------------------- D. operators, seeds, files
def test_operators_follow_their_shares():
    s = DemandSettings(rate=80, duration_s=600, operators={'a': 0.7, 'b': 0.2, 'c': 0.1})
    d = generate_demand([], B, s, seed=3)
    n = len(d.plans)
    for k, share in s.operators.items():
        assert abs(sum(fp.operator == k for fp in d.plans) / n - share) < 0.05
    with pytest.raises(ValueError):
        generate_demand([], B, DemandSettings(operators={'a': -1.0, 'b': 2.0}))


def test_same_seed_same_demand():
    layout, bounds = city('medium', 0)
    s = DemandSettings(rate=20, duration_s=60, peak=(30.0, 10.0, 2.0))
    a = generate_demand(layout, bounds, s, seed=7)
    b = generate_demand(layout, bounds, s, seed=7)
    c = generate_demand(layout, bounds, s, seed=8)
    assert a.plans == b.plans and a.report == b.report
    assert a.plans != c.plans


def test_save_load_round_trip(tmp_path):
    layout, bounds = city('easy', 0)
    d = generate_demand(layout, bounds, DemandSettings(rate=20, duration_s=60, mode='hubs',
                                                       peak=(30.0, 10.0, 2.0)), seed=1)
    p = tmp_path / 'demand.json'
    save_demand(p, d)
    raw = json.loads(p.read_text())
    assert raw['format'] == 'utm-demand' and raw['flights'][0]['operator'].startswith('op')
    e = load_demand(p)
    assert e.plans == d.plans and e.pads == d.pads and e.settings == d.settings
    assert e.report == d.report and e.seed == 1 and e.bounds == d.bounds
    raw['format'] = 'something else'
    p.write_text(json.dumps(raw))
    with pytest.raises(ValueError):
        load_demand(p)


# ----------------------------------------------------------------------------- E. with E1
def test_feeds_the_strategic_planner():
    layout, bounds = city('medium', 1)
    d = generate_demand(layout, bounds, DemandSettings(rate=10, duration_s=60), seed=2)
    sky = StrategicPlanner(layout, bounds)
    for fp in d.plans:
        sky.submit(fp)
    s = sky.summary()
    assert s['requests'] == len(d.plans) and s['approved'] >= 0.8 * len(d.plans)
    assert all(dec.reason != 'no route on any layer' for dec in sky.decisions)


def test_random_requests_still_works_for_the_preview():
    layout, bounds = city('easy', 0)
    sky = StrategicPlanner(layout, bounds)
    out = random_requests(np.random.RandomState(0), sky, 10)
    assert len(out) == 10 and all(0.0 <= fp.t_earliest < 30.0 for fp in out)
    assert out == random_requests(np.random.RandomState(0), StrategicPlanner(layout, bounds), 10)


# ----------------------------------------------------------------------------- F. full map
def test_full_ecity_map():
    from envs.osm_scenario import DEFAULT_LAYOUT_PATH, layout_bounds, load_osm_layout
    layout = load_osm_layout(DEFAULT_LAYOUT_PATH)
    x0, y0, x1, y1 = layout_bounds(layout)
    bounds = ((x0, x1), (y0, y1))
    s = DemandSettings(rate=5, duration_s=2.0, max_trip=60.0)
    d = generate_demand(layout, bounds, s, seed=0)
    assert d.report['districts'] > 100
    assert d.report['kept'] > 10 and d.report['no_valid_pair'] <= 2
    assert all(np.hypot(fp.goal[0] - fp.start[0], fp.goal[1] - fp.start[1]) <= 60.25 for fp in d.plans)
