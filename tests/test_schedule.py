"""
test_schedule.py
----------------
Scheduled departures (envs/swarm_env.py environment.schedule) and 3D routes for the carrot.

  A. A drone is parked out of the world until its departure time, then appears on its pad.
  B. A drone that has settled at its goal lands (is parked again); the episode ends when every
     drone has settled.
  C. Routes with heights: the carrot carries the route's height (E1 layer profile).
  D. Off by default: departures are refused.

Run: python -m pytest tests/test_schedule.py -q
"""

import os
import sys

import numpy as np
import pytest
import torch
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from envs.swarm_env import SwarmEnv  # noqa: E402

CONFIG = os.path.join(os.path.dirname(__file__), '..', 'configs', 'config_stage2_round7_teacher.yaml')
SPAWNS = {'drone_0': [-4.0, 0.0, 1.0], 'drone_1': [4.0, 6.0, 1.0]}
GOALS = {'drone_0': [-4.0, 4.0, 1.0], 'drone_1': [4.0, 2.0, 1.0]}


def make_env(schedule=True, route=False):
    cfg = yaml.safe_load(open(CONFIG, encoding='utf-8'))
    cfg.pop('scenario_mix', None)
    cfg['environment'].update(n_agents=2, collision_physics=False, crash_mode='remove')
    cfg['environment'].pop('sensor_noise', None)
    cfg.setdefault('obstacles', {})['enabled'] = False
    if schedule:
        cfg['environment']['schedule'] = {'enabled': True}
    if route:
        cfg['environment']['route_follow'] = {'enabled': True}
    return SwarmEnv(cfg, device=torch.device('cpu'))


def fly_to_goals(env, steps):
    """Simple P-controller towards each drone's real goal; returns (step, info) when done."""
    info = {}
    for t in range(1, steps + 1):
        pos = env.adapter.get_positions()
        v = np.clip(1.5 * (env.goals - pos), -1.0, 1.0)
        _, _, done, info = env.step({f'drone_{i}': [*v[i], 0.0] for i in range(2)})
        if done:
            return t, info
    return steps, info


def test_waits_off_map_then_takes_off_at_its_time():
    env = make_env()
    env.start_episode(SPAWNS, GOALS, obstacles=[], departures={'drone_1': 1.0})
    dt = env.control_dt
    dep_step = int(np.ceil(1.0 / dt - 1e-9))
    assert env.waiting.tolist() == [False, True]
    assert np.abs(env.adapter.get_positions()[1, :2]).min() > 1e4          # parked
    for _ in range(dep_step - 1):
        env.step({'drone_0': [0.0] * 4, 'drone_1': [0.0] * 4})
        assert np.abs(env.adapter.get_positions()[1, :2]).min() > 1e4      # still waiting
    env.step({'drone_0': [0.0] * 4, 'drone_1': [0.0] * 4})
    assert not env.waiting[1]
    assert np.linalg.norm(env.adapter.get_positions()[1] - SPAWNS['drone_1']) < 0.05   # on its pad
    env.close()


def test_settled_drones_land_and_the_episode_ends_when_all_have():
    env = make_env()
    env.start_episode(SPAWNS, GOALS, obstacles=[], departures={'drone_1': 2.0})
    steps, info = fly_to_goals(env, 2000)
    assert info['success'] and info['drone_reached_goal'] == [True, True]
    assert env.landed.all()
    assert steps * env.control_dt > 2.0                    # the late drone did fly after 2 s
    env.close()


def test_layer_profile_climbs_and_lands_vertically():
    """A route with heights (E1): the carrot is straight above the pad until the drone is near
    the cruise height, then along the route at that height, then the goal once above it, so the
    climb and the landing stay over the pads, where E1 books them."""
    env = make_env(route=True)
    spawns = {'drone_0': [-8.0, 0.0, 1.0], 'drone_1': [8.0, 6.0, 1.0]}
    goals = {'drone_0': [8.0, 0.0, 1.0], 'drone_1': [8.0, 2.0, 1.0]}
    route = {'drone_0': [(-8.0, 0.0, 1.0), (-8.0, 0.0, 4.0), (8.0, 0.0, 4.0), (8.0, 0.0, 1.0)]}
    env.start_episode(spawns, goals, obstacles=[], routes=route)
    c = env.adapter.goals['drone_0']
    assert np.allclose(c, [-8.0, 0.0, 4.0])                 # climb: straight up
    xs_low = []
    for _ in range(1500):
        pos = env.adapter.get_positions()
        c = env.adapter.goals['drone_0']
        v = np.clip(1.5 * (c - pos[0]), -1.0, 1.0)
        _, _, done, info = env.step({'drone_0': [*v, 0.0], 'drone_1': [0.0] * 4})
        p0 = env.adapter.get_positions()[0]
        if not env.landed[0] and p0[2] < 3.0 and abs(p0[1]) < 1e3:
            xs_low.append(p0[0])                            # below the layer: only near the pads
        if info['drone_reached_goal'][0]:
            break
    assert info['drone_reached_goal'][0]
    assert all(x < -7.0 or x > 7.0 for x in xs_low)          # climbed and landed over the pads
    env.close()


def test_lands_even_if_the_drone_stops_short_of_the_carrot():
    """In flown E1 runs the policy settled ~0.9-1.0 m short of the carrot above the goal (at cruise
    height); the landing must still start (with LAND_RADIUS_M 0.6 such drones hovered there for up
    to a minute). Stand-in controller: stops 0.9 m short of a carrot above the goal, flies straight
    to the real goal."""
    env = make_env(route=True)
    spawns = {'drone_0': [-8.0, 0.0, 1.0], 'drone_1': [8.0, 6.0, 1.0]}
    goals = {'drone_0': [8.0, 0.0, 1.0], 'drone_1': [8.0, 2.0, 1.0]}
    route = {'drone_0': [(-8.0, 0.0, 1.0), (-8.0, 0.0, 4.0), (8.0, 0.0, 4.0), (8.0, 0.0, 1.0)]}
    env.start_episode(spawns, goals, obstacles=[], routes=route)
    info = {}
    for _ in range(1500):
        pos = env.adapter.get_positions()[0]
        c = env.adapter.goals['drone_0']
        d = c - pos
        h = np.hypot(d[0], d[1])
        vxy = 1.5 * d[:2]
        if c[2] > 2.0 and h > 1e-9:                         # the carrot above the goal: stop short
            vxy = d[:2] * (1.5 * max(h - 0.9, 0.0) / h)
        v = np.clip([*vxy, 1.5 * d[2]], -1.0, 1.0)
        _, _, done, info = env.step({'drone_0': [*v, 0.0], 'drone_1': [0.0] * 4})
        if info['drone_reached_goal'][0]:
            break
    assert info['drone_reached_goal'][0]
    env.close()


def test_off_by_default():
    env = make_env(schedule=False)
    with pytest.raises(ValueError):
        env.start_episode(SPAWNS, GOALS, obstacles=[], departures={'drone_1': 1.0})
    env.close()
