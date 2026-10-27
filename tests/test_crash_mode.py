"""
test_crash_mode.py
------------------
environment.crash_mode (envs/swarm_env.py).

4 drones: drone_0 and drone_1 start 0.1 m apart (a crash on the first step), drone_2 and
drone_3 fly 2 m to their goals under a simple P-controller.
  end (default): the crash ends the episode on step 1, as always.
  remove: drones 0 and 1 are parked out of the world, drones 2 and 3 still settle, and the
          episode then ends with reason 'collision' (one crash still fails the episode).

Run: python -m pytest tests/test_crash_mode.py -q
"""

import os
import sys

import numpy as np
import pytest
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from envs.swarm_env import SwarmEnv                               # noqa: E402
from record_rollout import disable_all_collision_physics          # noqa: E402

CONFIG = os.path.join(os.path.dirname(__file__), '..', 'configs', 'config_stage2_round7_teacher.yaml')
SPAWNS = {'drone_0': [0.0, 0.0, 2.0], 'drone_1': [0.1, 0.0, 2.0],
          'drone_2': [5.0, 0.0, 2.0], 'drone_3': [-5.0, 0.0, 2.0]}
GOALS = {'drone_0': [0.0, 3.0, 2.0], 'drone_1': [0.1, -3.0, 2.0],
         'drone_2': [5.0, 2.0, 2.0], 'drone_3': [-5.0, -2.0, 2.0]}


def make_env(mode):
    cfg = yaml.safe_load(open(CONFIG, encoding='utf-8'))
    cfg.pop('scenario_mix', None)
    cfg['environment'].update(n_agents=4, collision_physics=False, crash_mode=mode)
    cfg.setdefault('obstacles', {})['enabled'] = False
    cfg['environment'].pop('sensor_noise', None)
    env = SwarmEnv(cfg, device='cpu')
    disable_all_collision_physics(env, 4)
    env.start_episode(SPAWNS, GOALS, obstacles=[])
    return env


def fly(env, max_steps=2000):
    for t in range(1, max_steps + 1):
        pos = env.adapter.get_positions()
        v = np.clip(1.5 * (env.goals - pos), -1.0, 1.0)
        _, _, done, info = env.step({f'drone_{i}': [*v[i], 0.0] for i in range(4)})
        if done:
            return t, info
    return max_steps, info


def test_end_is_default_and_unchanged():
    env = make_env('end')
    steps, info = fly(env)
    assert steps == 1 and info['reason'] == 'collision'
    assert not any(info['drone_crashed'])
    env.close()


def test_remove_parks_crashed_drones_and_the_rest_keep_flying():
    env = make_env('remove')
    steps, info = fly(env)
    assert steps > 1 and info['reason'] == 'collision' and not info['success']
    assert info['drone_crashed'] == [True, True, False, False]
    assert info['drone_reached_goal'] == [False, False, True, True]
    pos = env.adapter.get_positions()
    assert np.all(np.abs(pos[:2, :2]) > 1e4)                     # parked, out of every range
    assert np.linalg.norm(pos[0] - pos[1]) > 50                  # and apart from each other
    env.close()


def test_unknown_mode_refused():
    with pytest.raises(ValueError):
        make_env('bounce')
