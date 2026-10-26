"""
test_city_generator.py
----------------------
envs/city_generator.py: procedural city districts in the obstacle layout format.

  A. Same seed, same district; different seeds, different districts.
  B. Buildings stay inside the district, never overlap, never stand in a street.
  C. Harder levels: narrower streets, more buildings.
  D. Sampled flights start/end in streets, clear of buildings, apart from each other,
     and a free path exists for every one.
  E. SwarmEnv accepts the city scenarios in its scenario mix and steps through them.

Run: python -m pytest tests/test_city_generator.py -q
"""

import os
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from envs.city_generator import DIFFICULTY, generate_city, sample_flights, route_length
from envs.scenario_manager import surface_distance


def boxes_overlap(a, b):
    (ax, ay), (ahx, ahy) = a['center'], a['half']
    (bx, by), (bhx, bhy) = b['center'], b['half']
    return abs(ax - bx) < ahx + bhx - 1e-9 and abs(ay - by) < ahy + bhy - 1e-9


def test_deterministic():
    a = generate_city(np.random.RandomState(5), 'medium')['layout']
    b = generate_city(np.random.RandomState(5), 'medium')['layout']
    c = generate_city(np.random.RandomState(6), 'medium')['layout']
    assert a == b and a != c


def test_buildings_valid():
    for level in DIFFICULTY:
        for seed in range(5):
            city = generate_city(np.random.RandomState(seed), level)
            W, H = city['size']
            lay = city['layout']
            for ob in lay:
                assert ob['type'] == 'box' and 6.0 <= ob['height'] <= 20.0
                (cx, cy), (hx, hy) = ob['center'], ob['half']
                assert hx > 0 and hy > 0
                assert -W / 2 <= cx - hx and cx + hx <= W / 2 and -H / 2 <= cy - hy and cy + hy <= H / 2
            for i in range(len(lay)):
                for j in range(i + 1, len(lay)):
                    assert not boxes_overlap(lay[i], lay[j])
            for a, b in city['streets_x']:                     # street centre lines are free
                x = (a + b) / 2
                for y in np.linspace(-H / 2, H / 2, 40):
                    assert all(surface_distance(ob, (x, y)) > 0 for ob in lay)


def test_levels_get_harder():
    def stats(level):
        cities = [generate_city(np.random.RandomState(s), level) for s in range(6)]
        street = np.mean([b - a for c in cities for a, b in c['streets_x'] + c['streets_y']])
        count = np.mean([len(c['layout']) for c in cities])
        return street, count
    (se, ne), (sm, nm), (sh, nh) = stats('easy'), stats('medium'), stats('hard')
    assert se > sm > sh
    assert ne < nm < nh


def test_flights_valid():
    for level in DIFFICULTY:
        rng = np.random.RandomState(3)
        city = generate_city(rng, level)
        spawns, goals = sample_flights(rng, city, 8)
        assert len(spawns) == len(goals) == 8
        pts_s = [np.array(v[:2]) for v in spawns.values()]
        pts_g = [np.array(v[:2]) for v in goals.values()]
        for k in spawns:
            a, b = spawns[k], goals[k]
            assert 8.0 <= np.hypot(a[0] - b[0], a[1] - b[1]) <= 20.0
            assert a[2] == b[2] and 2.0 <= a[2] <= 3.0
            for p in (a, b):
                assert all(surface_distance(ob, p) >= 1.0 for ob in city['layout'])
            assert route_length(city, a, b) is not None
        for pts in (pts_s, pts_g):
            for i in range(len(pts)):
                for j in range(i + 1, len(pts)):
                    assert np.linalg.norm(pts[i] - pts[j]) >= 2.0


def test_env_runs_city_scenarios():
    from envs.swarm_env import SwarmEnv
    config = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), '..', 'configs',
                                              'config_stage2_ring_wider_teacher.yaml'), encoding='utf-8'))
    config['environment']['n_agents'] = 3
    config['scenario_mix'] = {'city_easy': 1.0, 'city_medium': 1.0, 'city_hard': 1.0}
    env = SwarmEnv(config)
    try:
        np.random.seed(0)
        for _ in range(3):
            obs = env.reset()
            assert env.current_scenario.startswith('city_') and len(env.layout) > 0
            for _ in range(5):
                obs, _, done, _ = env.step({f'drone_{i}': [0.0, 0.0, 0.0, 0.0] for i in range(3)})
                assert not done
    finally:
        env.close()
