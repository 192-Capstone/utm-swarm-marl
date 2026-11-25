"""
test_birds.py
-------------
Birds (envs/birds.py, environment.birds): moving, non-cooperative obstacles.

  A. Config: off by default, bad settings refused.
  B. Motion is deterministic per seed; bird i does not depend on the bird count.
  C. Timing: a bird is at its aim point (on a drone's path) at the aim time, and stays inside the
     bounds and the altitude band.
  D. Birds off: no new info keys, and with a bird far away the drones fly exactly as without birds.
  E. A bird on a drone's path is hit by the LiDAR rays (ring and fan), and seen in info.
  F. A contact is a bird crash: reason 'bird' (end), crashed + bird_crashed (remove).
  G. Goal-hold hands control back to the policy when a bird is within the ring's 0.5 m.
  H. Scenario key: validated.

Run: python -m pytest tests/test_birds.py -q
"""

import os
import sys

import numpy as np
import pytest
import yaml

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, ROOT)

from envs import birds as B                                        # noqa: E402
from envs.swarm_env import SwarmEnv                                # noqa: E402
from modules.goal_hold import GoalHoldPolicy                       # noqa: E402
from record_rollout import disable_all_collision_physics           # noqa: E402
from utm import scenario as sc_mod                                 # noqa: E402

CONFIG = os.path.join(ROOT, 'configs', 'config_stage2_round7_eval_goalhold.yaml')
DT = 0.032


def make_env(n=1, birds=None, mode='end', noise=True):
    cfg = yaml.safe_load(open(CONFIG, encoding='utf-8'))
    cfg.pop('scenario_mix', None)
    cfg['environment'].update(n_agents=n, collision_physics=False, crash_mode=mode)
    cfg.setdefault('obstacles', {})['enabled'] = False
    if not noise:
        cfg['environment'].pop('sensor_noise', None)
    if birds is not None:
        cfg['environment']['birds'] = birds
    env = SwarmEnv(cfg, device='cpu')
    disable_all_collision_physics(env, n)
    return env, cfg


SP = {'drone_0': [0.0, 0.0, 2.0]}
GL = {'drone_0': [8.0, 0.0, 2.0]}
PATHS = [np.array([[0.0, 0.0, 2.0], [8.0, 0.0, 2.0]])]
STILL = [0.0, 0.0, 0.0, 0.0]


def gen(seed, count=3, **kw):
    cfg = B.validate_block({'enabled': True, 'count': count, 'seed': seed, **kw})
    return B.BirdField.generate(cfg, PATHS, DT)


# ---------------------------------------------------------------- A -------
def test_off_by_default_and_bad_settings():
    assert B.config({'environment': {}}) is None
    assert B.config({'environment': {'birds': {'enabled': False, 'count': 3}}}) is None
    assert B.config({'environment': {'birds': {'enabled': True, 'count': 0}}}) is None
    assert B.config({'environment': {'birds': {'enabled': True, 'count': 2}}})['count'] == 2
    for bad in ({'count': -1}, {'speed': [3, 1]}, {'radius': 0}, {'modes': ['dive']}, {'bogus': 1}):
        with pytest.raises(ValueError):
            B.validate_block(bad)


# ---------------------------------------------------------------- B -------
def test_deterministic_per_seed():
    a, b, c = gen(0), gen(0), gen(1)
    for k in (0, 10, 500, 3000):                      # 3000 > the first table: extension is deterministic too
        assert np.array_equal(a.positions(k), b.positions(k))
    assert not np.allclose(a.positions(100), c.positions(100))


def test_bird_does_not_depend_on_the_count():
    one, three = gen(4, count=1), gen(4, count=3)
    assert np.array_equal(one.positions(50)[0], three.positions(50)[0])


# ---------------------------------------------------------------- C -------
@pytest.mark.parametrize('mode', B.MODES)
def test_bird_is_at_its_aim_point_at_the_aim_time(mode):
    for seed in range(10):
        f = gen(seed, count=2, modes=[mode], jitter_s=0.0)
        for i, bird in enumerate(f.birds):
            k = int(round(bird.t_aim / DT))
            assert np.allclose(f.positions(k)[i], bird.aim, atol=1e-6)
            assert 0.0 <= bird.aim[0] <= 8.0 and abs(bird.aim[1]) < 1e-9        # on the drone's path
            assert abs(bird.t_aim - bird.aim[0] / 2.0) < max(DT, 0.5)            # when a 2 m/s drone is there


def test_bounds_and_altitude_band():
    f = gen(3, count=4, bounds=[[-5, 12], [-6, 6]], z_band=[1.5, 3.0])
    pos = np.array([f.positions(k) for k in range(0, 4000, 7)])
    assert pos[..., 0].min() >= -5 - 1e-9 and pos[..., 0].max() <= 12 + 1e-9
    assert pos[..., 1].min() >= -6 - 1e-9 and pos[..., 1].max() <= 6 + 1e-9
    assert pos[..., 2].min() >= 1.5 - 1e-9 and pos[..., 2].max() <= 3.0 + 1e-9


def test_most_episodes_have_an_encounter():
    close = 0
    for seed in range(40):
        f = gen(seed, count=1, speed=[1.0, 1.0])
        assert abs(np.linalg.norm(f.positions(1)[0] - f.positions(0)[0]) / DT - 1.0) < 0.35   # ~1 m/s (+ vz)
        drone = lambda k: np.array([min(2.0 * k * DT, 8.0), 0.0, 2.0])
        close += min(np.linalg.norm(f.positions(k)[0] - drone(k)) for k in range(0, 400)) < 1.5
    assert close >= 30                                  # an encounter in at least 75 % of the episodes


def test_routed_flights_aim_at_the_route():
    route = np.array([[0.0, 0.0, 2.0], [0.0, 10.0, 2.0], [10.0, 10.0, 2.0]])
    cfg = B.validate_block({'enabled': True, 'count': 6, 'seed': 2})
    f = B.BirdField.generate(cfg, [route], DT)
    for b in f.birds:
        on = (abs(b.aim[0]) < 1e-6 and 0 <= b.aim[1] <= 10) or (abs(b.aim[1] - 10) < 1e-6 and 0 <= b.aim[0] <= 10)
        assert on


# ---------------------------------------------------------------- D -------
def test_birds_off_has_no_bird_keys():
    env, _ = make_env()
    env.start_episode(SP, GL, obstacles=[])
    _, _, _, info = env.step({'drone_0': [1.0, 0.0, 0.0, 0.0]})
    assert env.birds is None and not any(k.startswith('bird') for k in info)
    env.close()


def run_straight(env, steps=60):
    out = []
    for _ in range(steps):
        obs, _, done, info = env.step({'drone_0': [1.0, 0.0, 0.0, 0.0]})
        out.append(np.concatenate([env.adapter.get_positions().ravel(), obs['local_fixed'].numpy().ravel()]))
        if done:
            break
    return np.array(out)


def test_far_bird_changes_nothing():
    env0, _ = make_env()
    np.random.seed(7)                                  # the lag parameters are drawn from np.random
    env0.start_episode(SP, GL, obstacles=[])
    ref = run_straight(env0)
    env0.close()
    env1, _ = make_env()
    far = B.BirdField([B.Bird.straight([40.0, 40.0, 2.0], [0.0, 1.0, 0.0], 0.12, DT)], DT)
    np.random.seed(7)
    env1.start_episode(SP, GL, obstacles=[], birds=far)
    got = run_straight(env1)
    env1.close()
    assert np.array_equal(ref, got)                    # same drone, same noise, same lag, bit for bit


# ---------------------------------------------------------------- E -------
def test_bird_on_the_path_is_seen_by_the_lidar():
    env, _ = make_env(noise=False)
    still = B.BirdField([B.Bird.straight([2.0, 0.0, 2.0], [0.0, 0.0, 0.0], 0.12, DT)], DT)
    env.start_episode(SP, GL, obstacles=[], birds=still)
    for _ in range(3):
        obs, _, _, info = env.step({'drone_0': STILL})
    assert info['bird_seen'] == [[True]]
    assert abs(info['bird_distance'][0][0] - 2.0) < 1e-3
    lf = obs['local_fixed'].numpy()[0]
    ring = lf[42:42 + env.obs_processor.ring_rays]                  # proximity: 1 - d / range, 0 = nothing
    assert ring.max() > 0 and abs((1 - ring.max()) * env.lidar_range - (2.0 - 0.12)) < 0.1
    env.close()


# ---------------------------------------------------------------- F -------
def toward_drone():
    return B.BirdField([B.Bird.straight([3.0, 0.0, 2.0], [-2.0, 0.0, 0.0], 0.12, DT)], DT)


def test_contact_is_a_bird_crash_end_mode():
    env, _ = make_env(noise=False)
    env.start_episode(SP, GL, obstacles=[], birds=toward_drone())
    for _ in range(200):
        _, _, done, info = env.step({'drone_0': STILL})
        if done:
            break
    assert done and info['reason'] == 'bird' and not info['success']
    assert info['bird_collision'] == [True]
    assert env.episode_manager.stats['collisions'] == 1
    env.close()


def test_contact_is_a_bird_crash_remove_mode():
    env, _ = make_env(noise=False, mode='remove')
    env.start_episode(SP, GL, obstacles=[], birds=toward_drone())
    for _ in range(200):
        _, _, done, info = env.step({'drone_0': STILL})
        if done:
            break
    assert done and info['reason'] == 'bird' and info['drone_crashed'] == [True]
    assert info['bird_crashed'] == [True]
    env.close()


def test_a_near_miss_is_no_crash():
    env, _ = make_env(noise=False)
    f = B.BirdField([B.Bird.straight([3.0, 0.5, 2.0], [-2.0, 0.0, 0.0], 0.12, DT)], DT)    # passes 0.5 m beside
    env.start_episode(SP, GL, obstacles=[], birds=f)
    for _ in range(150):
        _, _, _, info = env.step({'drone_0': STILL})
        assert not info['bird_collision'][0]
    env.close()


# ---------------------------------------------------------------- G -------
class Zero:
    def act(self, obs, env, n):
        return {f'drone_{i}': list(STILL) for i in range(n)}


def test_goal_hold_hands_back_when_a_bird_is_inside_the_ring_distance():
    sp = {'drone_0': [0.0, 0.0, 2.0]}
    goal = {'drone_0': [0.2, 0.0, 2.0]}                 # 0.2 m from the goal: holding, would command +x
    cmd = {}
    for name in ('clear', 'bird'):
        env, cfg = make_env(noise=False)
        bird = (B.BirdField([B.Bird.straight([-0.45, 0.0, 2.0], [0, 0, 0], 0.12, DT)], DT) if name == 'bird' else None)
        obs = env.start_episode(sp, goal, obstacles=[], birds=bird)
        pol = GoalHoldPolicy(Zero(), cfg)
        for _ in range(3):
            act = pol.act(obs, env, 1)
            obs, _, _, _ = env.step(act)
        cmd[name] = np.array(act['drone_0'][:3])
        if bird is not None:
            assert pol.guarded[0]
        env.close()
    assert np.linalg.norm(cmd['clear']) > 0.05          # goal-hold drove the drone
    assert np.allclose(cmd['bird'], 0.0)                # the policy (here: zeros) kept control


# ---------------------------------------------------------------- H -------
def test_scenario_key_is_validated():
    base = {'name': 'x', 'seed': 20, 'map': {'map': 'city', 'level': 'medium'}, 'demand': {'flights': 3, 'window': 10}}
    ok = sc_mod.load({**base, 'birds': {'count': 2, 'speed': [1, 2]}})
    assert ok['birds']['count'] == 2
    assert 'birds' not in sc_mod.load(base)             # absent stays absent
    for bad in ({'count': -2}, {'modes': ['dive']}, 3):
        with pytest.raises(ValueError):
            sc_mod.load({**base, 'birds': bad})
