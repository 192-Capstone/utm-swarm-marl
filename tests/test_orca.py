"""
test_orca.py
------------
B1 baseline: ORCA (baselines/orca.py).

Core (the RVO2 port, plain 2D maths):
  A. A head-on pair (slightly offset) passes at the safety distance and both
     reach their goals.
  B. A perfectly symmetric scene gives perfectly symmetric moves.
  C. A drone flying at a wall of obstacle points stops before it.

The policy (real observation layout, scripted MockAdapter, no noise):
  D. Bad settings are refused.
  E. Alone, it flies straight at the goal at pref_speed; height is a capped
     P-controller.
  F. LiDAR ring: a hit straight ahead blocks forward motion, a hit to the side
     doesn't (checks the ring directions: ray 0 = towards the goal).
  G. Two drones head-on through SwarmEnv: they avoid each other using only
     the neighbour slots, and both settle at their goals.
  H. ORCA-map flies around a pillar (true shapes) without touching it.

Run with:
    python -m pytest tests/test_orca.py -s -v
"""

import copy
import os
import sys

import numpy as np
import pytest
import yaml

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, ROOT)

from baselines.orca import OrcaPolicy, orca_line, orca_velocity
from envs.scenario_manager import DRONE_RADIUS
from envs.swarm_env import SwarmEnv
from tests.test_env_validation import MockAdapter

ROUND7 = os.path.join(ROOT, 'configs', 'config_stage2_round7_teacher.yaml')
DT = 0.032


def print_test_header(test_num, test_name, purpose):
    """Print formatted test header (same style as the other test files)."""
    print("\n" + "=" * 60)
    print(f"TEST {test_num}: {test_name}")
    print("=" * 60)
    print(f"Purpose: {purpose}")


def config(n_agents=1, **orca):
    with open(ROUND7, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    cfg = copy.deepcopy(cfg)
    cfg['environment']['n_agents'] = n_agents
    sn = cfg['environment'].setdefault('sensor_noise', {})
    for k in ('position', 'velocity', 'orientation', 'lidar'):
        sn[k] = 0.0
    if orca:
        cfg['orca'] = orca
    return cfg


def simulate(pos, goals, steps, obstacle_points=(), radius=0.25, v_max=2.0):
    """Drones that fly exactly the ORCA velocity (no lag), for the core tests."""
    pos = np.array(pos, dtype=float)
    goals = np.array(goals, dtype=float)
    vel = np.zeros_like(pos)
    traj, seps = [pos.copy()], []
    for _ in range(steps):
        new = []
        for i in range(len(pos)):
            g = goals[i] - pos[i]
            d = np.linalg.norm(g)
            pref = g / d * min(v_max, 1.2 * d) if d > 1e-9 else np.zeros(2)
            agents = [orca_line(pos[j] - pos[i], vel[i] - vel[j], vel[i], 2 * radius, 2.0, DT, 0.5)
                      for j in range(len(pos)) if j != i]
            obst = [orca_line(np.asarray(p) - pos[i], vel[i], vel[i], 0.3, 1.0, DT, 1.0)
                    for p in obstacle_points if np.linalg.norm(np.asarray(p) - pos[i]) <= 3.0]
            new.append(orca_velocity(pref, obst, agents, v_max))
        vel = np.array(new)
        pos = pos + vel * DT
        traj.append(pos.copy())
        if len(pos) > 1:
            seps.append(min(np.linalg.norm(pos[a] - pos[b])
                            for a in range(len(pos)) for b in range(a + 1, len(pos))))
    return pos, vel, np.array(traj), seps


# ---------------------------------------------------------------- the core
def test_head_on_pair_avoids():
    print_test_header('A', 'head-on pair avoids',
                      "two drones flying at each other pass at the safety distance and reach their goals")
    pos, _, _, seps = simulate([[-3.0, 0.05], [3.0, -0.05]], [[3.0, 0.05], [-3.0, -0.05]], 800)
    assert min(seps) >= 0.5 - 0.01, f"closest approach {min(seps):.3f} m"
    assert np.allclose(pos, [[3.0, 0.05], [-3.0, -0.05]], atol=0.02)


def test_symmetric_scene_symmetric_moves():
    print_test_header('B', 'symmetric moves',
                      "a mirror-symmetric head-on scene stays mirror-symmetric at every step")
    _, _, traj, _ = simulate([[-3.0, 0.0], [3.0, 0.0]], [[3.0, 0.0], [-3.0, 0.0]], 300)
    assert np.allclose(traj[:, 0], -traj[:, 1], atol=1e-9)


def test_stops_before_a_wall():
    print_test_header('C', 'stops before a wall',
                      "a drone whose goal is behind a wall stops before it, never inside the safety distance")
    wall = [(1.0, y) for y in np.arange(-3.0, 3.01, 0.2)]
    pos, vel, traj, _ = simulate([[-2.0, 0.0]], [[3.0, 0.0]], 400, obstacle_points=wall)
    closest = min(np.linalg.norm(np.array(wall) - p, axis=1).min() for p in traj[:, 0])
    assert closest >= 0.3 - 0.03, f"came within {closest:.3f} m of the wall"
    assert np.linalg.norm(vel[0]) < 0.05


# --------------------------------------------------------------- the policy
def test_bad_settings_refused():
    print_test_header('D', 'settings checks', "unknown keys, too-high speed and old schemas raise")
    with pytest.raises(ValueError, match='unknown'):
        OrcaPolicy(config(radious=0.3))
    with pytest.raises(ValueError, match='pref_speed'):
        OrcaPolicy(config(pref_speed=3.0))
    cfg = config()
    cfg['observation']['schema'] = 'v2'
    with pytest.raises(ValueError, match='schema'):
        OrcaPolicy(cfg)


def single_drone(goal, spawn=(0.0, 0.0, 2.0), **orca):
    cfg = config(**orca)
    env = SwarmEnv(cfg, adapter=MockAdapter(1))
    obs = env.start_episode({'drone_0': list(spawn)}, {'drone_0': list(goal)}, obstacles=[])
    return cfg, env, obs


def test_alone_flies_at_the_goal():
    print_test_header('E', 'alone', "straight at the goal at pref_speed; height by a capped P-controller")
    cfg, env, obs = single_drone((5.0, 0.0, 2.0))
    a = OrcaPolicy(cfg).act(obs, env, 1)['drone_0']
    assert np.allclose(a, [2.0, 0.0, 0.0, 0.0], atol=1e-4)
    cfg, env, obs = single_drone((0.0, 3.0, 4.0))           # 2 m up: vz capped at vz_max
    a = OrcaPolicy(cfg).act(obs, env, 1)['drone_0']
    assert a[2] == pytest.approx(1.0, abs=1e-4) and a[1] > 0.5
    cfg, env, obs = single_drone((0.0, 1.0, 2.3))           # 0.3 m up: vz = k_z * 0.3
    a = OrcaPolicy(cfg).act(obs, env, 1)['drone_0']
    assert a[2] == pytest.approx(0.3, abs=1e-3)


def test_ring_directions():
    print_test_header('F', 'LiDAR ring', "a hit ahead blocks, a hit to the side doesn't (ray 0 = towards the goal)")
    cfg, env, _ = single_drone((5.0, 0.0, 2.0))
    policy = OrcaPolicy(cfg)
    rays = env.obs_processor.ring_rays

    def with_hit(ray, d):
        raw = env.adapter.get_raw_state()
        ring = [env.lidar_range] * rays
        ring[ray] = d
        raw['drone_0']['lidar_ring'] = ring
        return env._build_obs(raw)

    ahead = policy.act(with_hit(0, 0.5), env, 1)['drone_0']            # straight at the goal
    assert ahead[0] <= 0.25, f"still flying {ahead[0]:.2f} m/s into the obstacle"
    left = policy.act(with_hit(rays // 4, 0.5), env, 1)['drone_0']     # 90 deg to the left
    assert left[0] >= 1.9 and abs(left[1]) <= 0.25


def test_two_drones_head_on_in_the_env():
    print_test_header('G', 'head-on in SwarmEnv',
                      "avoidance from the neighbour slots only; both drones settle at their goals")
    cfg = config(n_agents=2)
    env = SwarmEnv(cfg, adapter=MockAdapter(2))
    obs = env.start_episode({'drone_0': [-3.0, 0.05, 2.0], 'drone_1': [3.0, -0.05, 2.0]},
                            {'drone_0': [3.0, 0.05, 2.0], 'drone_1': [-3.0, -0.05, 2.0]}, obstacles=[])
    policy = OrcaPolicy(cfg)
    closest, info = np.inf, {}
    for _ in range(900):
        obs, _, done, info = env.step(policy.act(obs, env, 2))
        closest = min(closest, info['min_inter_drone_distance'])
        if done:
            break
    assert closest >= 0.45, f"closest approach {closest:.3f} m"
    assert info['reason'] == 'success', info['reason']


def test_orca_map_flies_around_a_pillar():
    print_test_header('H', 'ORCA-map pillar', "true obstacle shapes: around the pillar without touching it")
    cfg = config()
    env = SwarmEnv(cfg, adapter=MockAdapter(1))
    pillar = {'type': 'pillar', 'center': (0.0, 0.15), 'radius': 0.4, 'height': 5.0}
    obs = env.start_episode({'drone_0': [-3.0, 0.0, 2.0]}, {'drone_0': [3.0, 0.0, 2.0]},
                            obstacles=[pillar])
    policy = OrcaPolicy(cfg, use_map=True)
    gap, info = np.inf, {}
    for _ in range(900):
        obs, _, done, info = env.step(policy.act(obs, env, 1))
        gap = min(gap, info['obstacle_distance'][0])
        if done:
            break
    assert gap > DRONE_RADIUS, f"touched the pillar (gap {gap:.3f} m)"
    assert info['reason'] == 'success', info['reason']
