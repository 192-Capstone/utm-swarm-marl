"""
C5: E1 landing options of envs/swarm_env.py (environment.schedule; both off by default).

  land_latch_m  once a drone with an E1 layer profile comes within LAND_RADIUS_M of its goal, its
                carrot stays the goal until it has landed, unless it drifts beyond land_latch_m.
  land_vz_max   while landing (latched the same way; without a latch: within LAND_RADIUS_M), the
                commanded vz is clipped to +-land_vz_max.

The drone is moved with adapter.place (teleport, still) so each landing state is set exactly.
"""
import os
import sys

import numpy as np
import pytest
import torch
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from envs.swarm_env import SwarmEnv, LAND_RADIUS_M  # noqa: E402

CONFIG = os.path.join(os.path.dirname(__file__), '..', 'configs', 'config_stage2_round7_teacher.yaml')
SPAWNS = {'drone_0': [-8.0, 0.0, 1.0], 'drone_1': [8.0, 6.0, 1.0]}
GOALS = {'drone_0': [8.0, 0.0, 1.0], 'drone_1': [8.0, 2.0, 1.0]}
ROUTE = {'drone_0': [(-8.0, 0.0, 1.0), (-8.0, 0.0, 4.0), (8.0, 0.0, 4.0), (8.0, 0.0, 1.0)]}
ZERO = {'drone_0': [0.0] * 4, 'drone_1': [0.0] * 4}


def make_env(**sched):
    cfg = yaml.safe_load(open(CONFIG, encoding='utf-8'))
    cfg.pop('scenario_mix', None)
    cfg['environment'].update(n_agents=2, collision_physics=False, crash_mode='remove')
    cfg['environment'].pop('sensor_noise', None)
    cfg.setdefault('obstacles', {})['enabled'] = False
    cfg['environment']['schedule'] = {'enabled': True, **sched}
    cfg['environment']['route_follow'] = {'enabled': True}
    env = SwarmEnv(cfg, device=torch.device('cpu'))
    env.start_episode(SPAWNS, GOALS, obstacles=[], routes=ROUTE)
    return env


def at(env, dx, actions=ZERO):
    """Put drone_0 at cruise height dx m before its goal (still) and take one step."""
    env.adapter.place(0, [GOALS['drone_0'][0] - dx, 0.0, 4.0])
    env.step(actions)
    return np.allclose(env.adapter.goals['drone_0'], GOALS['drone_0'])      # carrot = goal?


def test_default_carrot_jumps_back_when_the_drone_drifts_just_outside():
    env = make_env()
    assert at(env, 5.0) is False                       # cruise: route carrot (climbed on the way)
    assert at(env, 0.5) is True                        # within LAND_RADIUS_M: the goal
    assert at(env, 1.8) is False                       # just outside: back to the route at cruise
    assert not env._landing.any()                      # no landing state kept when off
    env.close()


def test_latch_keeps_the_goal_until_the_drone_drifts_beyond_land_latch_m():
    env = make_env(land_latch_m=2.4)
    assert at(env, 1.8) is False                       # not landing yet: 1.8 m > LAND_RADIUS_M
    assert at(env, 0.5) is True and env._landing[0]    # landing starts
    assert at(env, 1.8) is True                        # latched: still the goal
    assert at(env, 2.3) is True
    assert at(env, 3.0) is False and not env._landing[0]   # beyond the latch: route again
    assert at(env, 1.8) is False                       # and landing starts afresh at LAND_RADIUS_M
    assert at(env, 1.0) is True
    env.close()


def _spy(env):
    seen = []
    orig = env.adapter.apply_actions
    env.adapter.apply_actions = lambda a: (seen.append({k: np.array(v, float) for k, v in a.items()}), orig(a))[1]
    return seen


def test_vz_limit_only_while_landing_and_only_vz():
    env = make_env(land_latch_m=2.4, land_vz_max=0.5)
    seen = _spy(env)
    down = {'drone_0': [0.7, -0.3, -1.5, 0.0], 'drone_1': [0.0, 0.0, -1.5, 0.0]}
    at(env, 5.0)
    env.step(down)                                     # not landing: untouched
    assert seen[-1]['drone_0'].tolist() == pytest.approx([0.7, -0.3, -1.5, 0.0])
    at(env, 0.5)                                       # landing (and latched)
    env.step(down)
    assert seen[-1]['drone_0'].tolist() == pytest.approx([0.7, -0.3, -0.5, 0.0])
    assert seen[-1]['drone_1'].tolist() == pytest.approx([0.0, 0.0, -1.5, 0.0])   # other drone free
    assert env.prev_actions[0].tolist() == pytest.approx([0.7, -0.3, -0.5, 0.0])  # recorded as applied
    at(env, 1.8)                                       # latched: still limited, upward too
    env.step({**down, 'drone_0': [0.0, 0.0, 1.4, 0.0]})
    assert seen[-1]['drone_0'][2] == pytest.approx(0.5)
    assert down['drone_0'][2] == -1.5                  # the caller's dict is not changed
    env.close()


def test_vz_limit_without_latch_follows_land_radius():
    env = make_env(land_vz_max=0.5)
    seen = _spy(env)
    assert at(env, 0.5) is True
    env.step({'drone_0': [0.0, 0.0, -1.5, 0.0], 'drone_1': [0.0] * 4})
    assert seen[-1]['drone_0'][2] == pytest.approx(-0.5)
    assert at(env, 1.8) is False                       # no latch: outside LAND_RADIUS_M, not landing
    env.step({'drone_0': [0.0, 0.0, -1.5, 0.0], 'drone_1': [0.0] * 4})
    assert seen[-1]['drone_0'][2] == pytest.approx(-1.5)
    env.close()


def test_options_off_fly_exactly_as_before():
    """Keys absent and keys None give the same flight, bit for bit; with the options on a full
    flight still climbs, cruises and lands."""
    tracks = []
    for sched in ({}, {'land_latch_m': None, 'land_vz_max': None}, {'land_latch_m': 2.4, 'land_vz_max': 0.5}):
        np.random.seed(0)                              # the PyBullet lag model draws from np.random
        env = make_env(**sched)
        tr, info = [], {}
        for _ in range(1500):
            pos = env.adapter.get_positions()
            v = np.clip(1.5 * (env.adapter.goals['drone_0'] - pos[0]), -1.0, 1.0)
            _, _, done, info = env.step({'drone_0': [*v, 0.0], 'drone_1': [0.0] * 4})
            tr.append(env.adapter.get_positions()[0].copy())
            if info['drone_reached_goal'][0]:
                break
        assert info['drone_reached_goal'][0]
        tracks.append(np.array(tr))
        env.close()
    assert tracks[0].shape == tracks[1].shape and np.array_equal(tracks[0], tracks[1])


def test_set_route_clears_the_landing_latch():
    """E4 re-plans a drone mid-landing (set_route): it climbs and follows the new route instead of
    staying latched (no vz limit on the re-climb, no carrot pulled straight to the new goal)."""
    env = make_env(land_latch_m=2.4, land_vz_max=0.5)
    seen = _spy(env)
    assert at(env, 0.5) is True and env._landing[0]               # landing, latched
    px, py, _ = env.adapter.get_positions()[0]
    gx = px - 2.0                                                  # new pad 2 m away: inside the latch
    env.set_route(0, [(px, py, 1.0), (px, py, 4.0), (gx, py, 4.0), (gx, py, 1.0)])
    assert not env._landing[0]
    env.step({'drone_0': [0.0, 0.0, 1.5, 0.0], 'drone_1': [0.0] * 4})
    assert seen[-1]['drone_0'][2] == pytest.approx(1.5)            # re-climb not limited
    env.adapter.place(0, [px, py, 4.0])                            # climbed, 2 m from the new goal
    env.step(ZERO)
    env.step(ZERO)
    assert not env._landing[0]
    assert env.adapter.goals['drone_0'][2] == pytest.approx(4.0)   # route at cruise, not the goal
    env.close()


def test_bad_values_are_refused():
    with pytest.raises(ValueError):
        make_env(land_latch_m=LAND_RADIUS_M - 0.1)
    with pytest.raises(ValueError):
        make_env(land_vz_max=-0.1)
