"""
test_sensor_noise.py
--------------------
environment.sensor_noise perturbs only what the policy sees.

  A. Off (block missing): observations identical to a second env on the same seed.
  B. On: observations change, but the true state (adapter) and the teacher's
     action do not.
  C. LiDAR: rays that see nothing stay at full range; hits stay inside [0, range].
  D. The noise has the requested standard deviation.

Run: python -m pytest tests/test_sensor_noise.py -q
"""

import copy
import os
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from envs.swarm_env import SwarmEnv
from baselines.repulsion_controller import RepulsionController

CONFIG = os.path.join(os.path.dirname(__file__), '..', 'configs', 'config_stage2_ring_lag_teacher.yaml')
NOISE = {'position': 0.05, 'velocity': 0.05, 'orientation': 0.01, 'lidar': 0.03, 'seed': 7}


def make_env(noise=None, n_agents=3):
    config = yaml.safe_load(open(CONFIG, encoding='utf-8'))
    config['environment']['n_agents'] = n_agents
    config['environment'].get('dynamics', {})['lag'] = False      # instant response keeps the test simple
    if noise:
        config['environment']['sensor_noise'] = dict(noise)
    else:
        config['environment'].pop('sensor_noise', None)
    return SwarmEnv(config), config


def reset(env, seed=3):
    np.random.seed(seed)
    return env.reset()


def test_off_matches_a_second_env():
    a, _ = make_env()
    b, _ = make_env()
    try:
        oa, ob = reset(a), reset(b)
        for k in ('local_fixed', 'neighbor_states'):
            assert np.array_equal(np.asarray(oa[k]), np.asarray(ob[k]))
        assert not a.noise_on
    finally:
        a.close(); b.close()


def test_on_changes_obs_but_not_truth_or_teacher():
    clean, config = make_env()
    noisy, _ = make_env(NOISE)
    try:
        oc, on = reset(clean), reset(noisy)
        assert noisy.noise_on
        assert not np.allclose(np.asarray(oc['local_fixed']), np.asarray(on['local_fixed']))
        rc, rn = clean.adapter.get_raw_state(), noisy.adapter.get_raw_state()
        for i in range(clean.n_agents):
            assert np.allclose(rc[f'drone_{i}']['position'], rn[f'drone_{i}']['position'])
        teacher = RepulsionController(config)
        ac = teacher.act(oc, clean, clean.n_agents)
        an = teacher.act(on, noisy, noisy.n_agents)
        for i in range(clean.n_agents):
            assert np.allclose(ac[f'drone_{i}'], an[f'drone_{i}'])
    finally:
        clean.close(); noisy.close()


def test_lidar_misses_stay_at_range():
    env, _ = make_env(NOISE, n_agents=1)
    try:
        R = env.lidar_range
        raw = {'goals': {'drone_0': [0, 0, 1]},
               'drone_0': {'position': [0, 0, 1], 'velocity': [0, 0, 0], 'orientation': [0, 0, 0],
                           'lidar': [R] * 7 + [0.01, 1.0, R - 0.001, 2.0, 3.0, 4.0, 5.0],
                           'lidar_ring': [R] * 16 + [1.5] * 16}}
        for _ in range(50):
            out = env._noisy(copy.deepcopy(raw))['drone_0']
            lid, ring = np.asarray(out['lidar']), np.asarray(out['lidar_ring'])
            assert np.all(lid[:7] == R) and np.all(ring[:16] == R)
            assert np.all((lid >= 0) & (lid <= R)) and np.all((ring >= 0) & (ring <= R))
    finally:
        env.close()


def test_noise_has_the_requested_std():
    env, _ = make_env({'position': 0.04, 'velocity': 0.0, 'lidar': 0.0}, n_agents=1)
    try:
        raw = {'goals': {'drone_0': [0, 0, 1]},
               'drone_0': {'position': [1.0, 2.0, 3.0], 'velocity': [0, 0, 0], 'orientation': [0, 0, 0],
                           'lidar': [6.0] * 14}}
        d = np.array([env._noisy(raw)['drone_0']['position'] for _ in range(4000)]) - [1.0, 2.0, 3.0]
        assert abs(d.std() - 0.04) < 0.004 and abs(d.mean()) < 0.004
        assert env._noisy(raw)['drone_0']['velocity'] == [0, 0, 0]
    finally:
        env.close()
