"""
schedule.pad_queue_m (envs/swarm_env.py, off by default): drones bound for one pad line up for it.

The drones are moved with adapter.place (teleport, still) so each situation is set exactly.
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
PAD, Z = (20.0, 0.0), 4.0


def make_env(goal_1=PAD, **sched):
    """Two drones on straight routes along x to PAD (drone_1's may end elsewhere), both at cruise height."""
    cfg = yaml.safe_load(open(CONFIG, encoding='utf-8'))
    cfg.pop('scenario_mix', None)
    cfg['environment'].update(n_agents=2, collision_physics=False, crash_mode='remove')
    cfg['environment'].pop('sensor_noise', None)
    cfg.setdefault('obstacles', {})['enabled'] = False
    cfg['environment']['schedule'] = {'enabled': True, **sched}
    cfg['environment']['route_follow'] = {'enabled': True}
    env = SwarmEnv(cfg, device=torch.device('cpu'))
    spawns = {'drone_0': [-20.0, 0.0, 1.0], 'drone_1': [-20.0, 3.0, 1.0]}
    goals = {'drone_0': [*PAD, 1.0], 'drone_1': [*goal_1, 1.0]}
    routes = {k: [tuple(spawns[k]), (*spawns[k][:2], Z), (*goals[k][:2], Z), tuple(goals[k])] for k in spawns}
    env.start_episode(spawns, goals, obstacles=[], routes=routes)
    return env


def carrots(env, x0, x1):
    """Both drones placed x0 / x1 m before the pad (on drone_0's line), still; how far each carrot leads its drone."""
    for _ in range(2):                                    # a place in the queue is taken on the next step
        env.adapter.place(0, [PAD[0] - x0, 0.0, Z])
        env.adapter.place(1, [PAD[0] - x1, 1.0, Z])
        env.step({'drone_0': [0.0] * 4, 'drone_1': [0.0] * 4})
    g = [np.asarray(env.adapter.goals[f'drone_{i}'], dtype=float) for i in range(2)]
    return g[0][0] - (PAD[0] - x0), g[1][0] - (PAD[0] - x1)


def test_the_one_behind_holds_until_the_one_ahead_is_pad_queue_m_away():
    env = make_env(pad_queue_m=6.0)
    ahead, behind = carrots(env, 15.0, 16.0)
    assert ahead > 4.0 and abs(behind) < 0.7              # the front one flies on, the other holds where it is
    ahead, behind = carrots(env, 8.0, 16.0)
    assert 1.0 < behind < 2.5                             # 8 m apart: it may close up by 2 m
    ahead, behind = carrots(env, 5.0, 16.0)
    assert behind > 4.0                                   # far enough: at full speed
    env.close()


def test_it_waits_pad_queue_m_from_the_pad_until_the_one_ahead_has_landed():
    env = make_env(pad_queue_m=6.0)
    carrots(env, 1.0, 12.0)                               # drone_0 takes the first place
    assert abs(carrots(env, 0.2, 6.0)[1]) < 0.7           # drone_0 over the pad: drone_1 waits 6 m away
    env.landed[0] = True
    assert carrots(env, 0.2, 6.0)[1] > 4.0                # landed: drone_1 goes
    env.close()


def test_the_place_is_kept_and_the_front_one_is_never_held():
    env = make_env(pad_queue_m=6.0)
    carrots(env, 15.0, 16.0)
    assert carrots(env, 14.0, 13.0)[0] > 4.0              # drone_1 slipped past it: drone_0 still goes first
    assert list(env._pad_queue.values()) == [[0, 1]]
    env.close()


def test_a_drone_sent_elsewhere_leaves_the_queue():
    env = make_env(pad_queue_m=6.0)
    carrots(env, 15.0, 16.0)
    env.set_route(0, [(5.0, 0.0, Z), (-20.0, 0.0, Z), (-20.0, 0.0, 1.0)])
    assert carrots(env, 15.0, 16.0)[1] > 4.0
    env.close()


def test_other_pads_do_not_queue_and_it_is_off_by_default():
    env = make_env(goal_1=(20.0, 3.0), pad_queue_m=6.0)
    assert carrots(env, 15.0, 16.0)[1] > 4.0
    env.close()
    env = make_env()
    assert env.pad_queue_m is None and carrots(env, 15.0, 16.0)[1] > 4.0
    env.close()


def test_it_must_be_positive():
    with pytest.raises(ValueError):
        make_env(pad_queue_m=0)
