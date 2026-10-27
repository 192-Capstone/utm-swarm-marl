"""
test_route_follow.py
--------------------
D7 environment hook: the policy chases a carrot along its route (envs/swarm_env.py
environment.route_follow, envs/route_planner.py RouteFollower.visible_target).

  A. The carrot stays in line of sight around a corner, never closer than min_dist except at
     the goal, and ends on the goal.
  B. In the env: the adapter's goal (observation + LiDAR heading) is the carrot, the env's goal
     (success, settling, rewards) is still the real goal; the observation's goal vector points
     at the carrot.
  C. Off by default: routes are refused, and nothing changes.

Run: python -m pytest tests/test_route_follow.py -q
"""

import os
import sys

import numpy as np
import pytest
import torch
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from envs.route_planner import RouteFollower, build_grid, plan, segment_free, simplify  # noqa: E402
from envs.swarm_env import SwarmEnv                                                        # noqa: E402

CONFIG = os.path.join(os.path.dirname(__file__), '..', 'configs', 'config_stage2_round7_teacher.yaml')
B = ((-15.0, 15.0), (-15.0, 15.0))
# a wall the route has to go around: from (-6, 0) to (6, 0) past the wall's end at y = 6
WALL = {'type': 'box', 'center': (0.0, -2.0), 'half': (0.4, 8.0), 'height': 10.0}


def corner_route():
    """(line-of-sight grid: walls grown by the drone radius, as the env uses; the route)."""
    g = build_grid([WALL], B, res=0.25, inflate=0.5)
    return build_grid([WALL], B, res=0.25, inflate=0.15), simplify(g, plan(g, (-6.0, 0.0), (6.0, 0.0)))


def test_carrot_stays_visible_and_away_from_the_drone():
    g, route = corner_route()
    f = RouteFollower([(x, y, 2.0) for x, y in route], lookahead=4.5)
    plain = RouteFollower([(x, y, 2.0) for x, y in route], lookahead=4.5)
    hidden = 0
    for s in np.arange(0.0, f.total, 0.25):            # the drone exactly on its route
        pos = f.point_at(s)
        t = f.visible_target(pos, g, min_dist=1.0)
        hidden += not segment_free(g, pos[:2], plain.target(pos)[:2])
        assert segment_free(g, pos[:2], t[:2])
        if not np.allclose(t, f.w[-1]):
            assert np.hypot(*(t[:2] - pos[:2])) >= 1.0 - 1e-9
    assert hidden > 0                                   # the plain carrot does go behind the wall
    assert np.allclose(f.visible_target(f.w[-1], g), f.w[-1])


def make_env(enabled):
    cfg = yaml.safe_load(open(CONFIG, encoding='utf-8'))
    cfg.pop('scenario_mix', None)
    cfg['environment'].update(n_agents=1, collision_physics=False)
    cfg['environment'].pop('sensor_noise', None)
    cfg.setdefault('obstacles', {})['enabled'] = False
    if enabled:
        cfg['environment']['route_follow'] = {'enabled': True}
    return SwarmEnv(cfg, device=torch.device('cpu'))


def test_env_uses_the_carrot_for_the_policy_and_the_real_goal_for_scoring():
    _, route = corner_route()
    env = make_env(True)
    spawns, goals = {'drone_0': [-6.0, 0.0, 2.0]}, {'drone_0': [6.0, 0.0, 2.0]}
    obs = env.start_episode(spawns, goals, obstacles=[WALL], routes={'drone_0': route})
    carrot = env.adapter.goals['drone_0']
    assert np.allclose(env.goals[0], goals['drone_0'])                     # scoring: the real goal
    assert not np.allclose(carrot, goals['drone_0'])                       # policy: the carrot
    pos = env.adapter.get_positions()[0]
    lf = np.asarray(obs['local_fixed'])[0]
    proc = env.obs_processor
    want = carrot - pos
    want = want * min(1.0, proc.goal_vector_cap / np.linalg.norm(want)) / proc.goal_vector_cap
    assert np.allclose(lf[9:12], want, atol=1e-4)                          # obs goal vector -> carrot
    for _ in range(20):
        obs, _, done, info = env.step({'drone_0': [0.5, 0.5, 0.0, 0.0]})
    assert np.allclose(env.goals[0], goals['drone_0'])
    env.close()


def test_off_by_default():
    env = make_env(False)
    spawns, goals = {'drone_0': [-6.0, 0.0, 2.0]}, {'drone_0': [6.0, 0.0, 2.0]}
    with pytest.raises(ValueError):
        env.start_episode(spawns, goals, obstacles=[WALL], routes={'drone_0': [(-6, 0), (6, 0)]})
    env.start_episode(spawns, goals, obstacles=[WALL])
    assert env.followers is None
    assert np.allclose(env.adapter.goals['drone_0'], goals['drone_0'])
    env.close()
