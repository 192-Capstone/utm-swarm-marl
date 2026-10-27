"""
test_demand_invariants.py
-------------------------
D4 demand (utm/demand.py): invariants that must hold for every demand, checked by
utm/demand_checks.check_demand over every map x mode x level x a few seeds:

  pads in the connected region; every request routable (real A* on each one, so this also tests
  that skipping A* for region pads is safe); min / max trip; pad spacing between flights close in
  time; departure times sorted inside [0, duration); unique ids; report counts adding up; hubs-mode
  flights have exactly one hub end; no more pads placed than requested.

Maps: D1 easy / medium / hard (city seed 0) and the E-City crop. Modes: random, random with
max_trip (short trips), hubs. Levels: LEVELS (5 .. 80 flights / min / district), seeds 0 and 1,
20 s of departures to keep it fast (scripts/demand_stats.py --invariants runs the full 120 s).

Run: python -m pytest tests/test_demand_invariants.py -q
"""

import os
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip('h3')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from envs.city_generator import generate_city                                      # noqa: E402
from envs.osm_scenario import load_osm_layout                                      # noqa: E402
from utm.demand import LEVELS, DemandSettings, Region, generate_demand, layer0_grid  # noqa: E402
from utm.demand_checks import check_demand, check_seeds                            # noqa: E402

REPO = Path(__file__).resolve().parents[1]
CROP = REPO / 'data/osm_city/bengaluru_ecity_phase1_rotated_v2_crop50_cand1_layout.yaml'
SEEDS = check_seeds([0, 1])
MODES = {
    'random': dict(mode='random'),
    'random-short': dict(mode='random', max_trip=20.0),
    'hubs': dict(mode='hubs'),
}


def load(name):
    if name == 'crop':
        return load_osm_layout(CROP), ((-25.0, 25.0), (-25.0, 25.0))
    c = generate_city(np.random.RandomState(0), name)
    W, H = c['size']
    return c['layout'], ((-W / 2, W / 2), (-H / 2, H / 2))


@pytest.fixture(scope='module')
def maps():
    out = {}
    for name in ('easy', 'medium', 'hard', 'crop'):
        layout, bounds = load(name)
        out[name] = (layout, bounds, Region(layer0_grid(layout, bounds)))
    return out


@pytest.mark.parametrize('mode', list(MODES))
@pytest.mark.parametrize('name', ['easy', 'medium', 'hard', 'crop'])
def test_invariants_hold_everywhere(maps, name, mode):
    layout, bounds, region = maps[name]
    kept = 0
    for level in LEVELS:
        for seed in SEEDS:
            s = DemandSettings(rate=level, duration_s=20.0, **MODES[mode])
            d = generate_demand(layout, bounds, s, seed=seed, region=region)
            kept += len(d.plans)
            problems = check_demand(d, region, route=True)
            assert problems == [], f"{name} {mode} level {level} seed {seed}: {problems[:5]}"
    assert kept > 50                            # ~103 expected (155 / min x 20 s x 2 seeds): real work done


def test_check_demand_catches_broken_demand(maps):
    """The checker itself: each kind of damage is reported."""
    layout, bounds, region = maps['medium']
    d = generate_demand(layout, bounds, DemandSettings(rate=40, duration_s=30), seed=1, region=region)
    assert check_demand(d, region) == [] and len(d.plans) > 5
    p = d.plans

    def broken(**kw):
        return check_demand(replace(d, **kw), region, route=False)

    assert any('sorted' in m for m in broken(plans=[p[1], p[0]] + p[2:]))
    assert any('unique' in m for m in broken(plans=[p[0], replace(p[1], id=p[0].id)] + p[2:]))
    assert any('trip' in m for m in broken(plans=[replace(p[0], goal=p[0].start)] + p[1:]))
    assert any('kept' in m for m in broken(report={**d.report, 'kept': len(p) + 1}))
    blocked = tuple(map(float, region.grid.to_world(tuple(np.argwhere(region.grid.occ)[0]))))
    assert any('outside the connected region' in m for m in broken(plans=[replace(p[0], goal=blocked)] + p[1:]))
    close = replace(p[1], t_earliest=p[0].t_earliest, start=(p[0].start[0] + 0.5, p[0].start[1]))
    assert any('apart' in m for m in broken(plans=[p[0], close] + p[2:]))
    # a free cell cut off from the connected region (the crop has some): A* finds no route
    layout, bounds, crop = maps['crop']
    g = crop.grid
    cut_off = next(g.to_world((int(iy), int(ix))) for iy, ix in np.argwhere(~g.occ)
                   if not crop.contains(g.to_world((int(iy), int(ix)))))
    c = generate_demand(layout, bounds, DemandSettings(rate=20, duration_s=30), seed=1, region=crop)
    lost = replace(c, plans=[replace(c.plans[0], goal=tuple(map(float, cut_off)))])
    assert any('A* route' in m for m in check_demand(lost, crop))


def test_reserved_seeds_are_refused():
    with pytest.raises(ValueError, match='reserved'):
        check_seeds([0, 12])
