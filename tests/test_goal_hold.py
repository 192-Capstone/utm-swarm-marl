"""
test_goal_hold.py
-----------------
A1 part 3: goal-hold handover (modules/goal_hold.py) and the optional
per-episode sensor-noise seed (environment.sensor_noise.per_episode_seed).

  A. Off by default; bad settings are refused.
  B. Inside the goal zone the command points at the goal, capped at max_speed,
     with yaw 0; far from the goal the wrapped policy's action is untouched.
     Uses the real observation layout (goal vector at local_fixed[9:12]).
  C. Hysteresis: holding starts under engage_dist, continues to release_dist.
  D. A drone that has settled keeps being held (the policy would push it away).
  E. Safety guard: a close neighbour or obstacle hands control back to the
     policy; holding resumes only once they are beyond the resume distances.
  F. A new episode clears the hold state (also when the goals repeat);
     other attributes pass through.
  G. Noise default unchanged: without per_episode_seed one generator runs on
     across episodes, exactly as before.
  H. With per_episode_seed the same scenario always gets the same noise,
     whatever ran before it; different scenarios get different noise.

Run with:
    python -m pytest tests/test_goal_hold.py -s -v
"""

import copy
import os
import sys

import numpy as np
import pytest
import yaml

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, ROOT)

from envs.swarm_env import SwarmEnv
from modules.goal_hold import GoalHoldPolicy, goal_hold_enabled
from tests.test_env_validation import MockAdapter

ROUND7 = os.path.join(ROOT, 'configs', 'config_stage2_round7_teacher.yaml')
GOAL = [0.0, 0.0, 2.0]
FAR = [0.0, 10.0, 2.0]          # a second drone's spot, far from everything


def print_test_header(test_num, test_name, purpose):
    """Print formatted test header (same style as the other test files)."""
    print("\n" + "=" * 60)
    print(f"TEST {test_num}: {test_name}")
    print("=" * 60)
    print(f"Purpose: {purpose}")


def load_round7():
    with open(ROUND7, encoding='utf-8') as f:
        return yaml.safe_load(f)


def config(n_agents=1, noise=False, per_episode=None, goal_hold=True, **gh):
    cfg = copy.deepcopy(load_round7())
    cfg['environment']['n_agents'] = n_agents
    sn = cfg['environment'].setdefault('sensor_noise', {})
    if not noise:
        for k in ('position', 'velocity', 'orientation', 'lidar'):
            sn[k] = 0.0
    if per_episode is not None:
        sn['per_episode_seed'] = per_episode
    if goal_hold:
        cfg['goal_hold'] = {'enabled': True, **gh}
    return cfg


def make_env(cfg, spawns, goals):
    """SwarmEnv on the scripted MockAdapter (no PyBullet), no obstacles."""
    n = cfg['environment']['n_agents']
    env = SwarmEnv(cfg, adapter=MockAdapter(n))
    obs = env.start_episode({f'drone_{i}': spawns[i] for i in range(n)},
                            {f'drone_{i}': goals[i] for i in range(n)}, obstacles=[])
    return env, obs


class ConstantPolicy:
    """Stand-in for the learned policy: the same action for every drone."""
    def __init__(self, action=(1.0, 1.0, 1.0, 0.5)):
        self.action = list(action)
        self.ckpt = {'episode': 123}

    def act(self, obs, env, n_agents):
        return {f'drone_{i}': list(self.action) for i in range(n_agents)}


def place(env, positions):
    """Move the mock drones and return the observation they now see."""
    env.adapter.positions = np.array(positions, dtype=np.float32)
    return env._build_obs(env.adapter.get_raw_state())


def offset(d):
    return [GOAL[0] + d, GOAL[1], GOAL[2]]


# ---------------------------------------------------------------- A. config
def test_off_by_default_and_bad_settings_refused():
    print_test_header('A', 'off by default, bad settings refused',
                      "no goal_hold block = off; wrong settings raise")
    assert not goal_hold_enabled(load_round7())
    assert not goal_hold_enabled({'goal_hold': {'enabled': False}})
    assert goal_hold_enabled({'goal_hold': {'enabled': True}})
    inner = ConstantPolicy()
    with pytest.raises(ValueError, match='unknown'):
        GoalHoldPolicy(inner, config(gian=1.0))
    with pytest.raises(ValueError, match='engage_dist'):
        GoalHoldPolicy(inner, config(engage_dist=0.8, release_dist=0.7))
    with pytest.raises(ValueError, match='neighbour_release'):
        GoalHoldPolicy(inner, config(neighbour_release=1.5))
    with pytest.raises(ValueError, match='positive'):
        GoalHoldPolicy(inner, config(gain=0.0))
    cfg = config()
    cfg['observation']['schema'] = 'v1'
    with pytest.raises(ValueError, match='schema'):
        GoalHoldPolicy(inner, cfg)


# ------------------------------------------------------------ B. the command
def test_command_inside_zone_and_untouched_outside():
    print_test_header('B', 'hold command / untouched far away',
                      "v = gain * goal vector, capped at max_speed, yaw 0, from the real obs layout")
    cfg = config()
    env, obs = make_env(cfg, [offset(0.4)], [GOAL])
    gh = GoalHoldPolicy(ConstantPolicy(), cfg)

    a = gh.act(obs, env, 1)['drone_0']            # 0.4 m away: 1.0 * 0.4 = 0.4 -> capped 0.3
    assert np.allclose(a, [-0.3, 0.0, 0.0, 0.0], atol=1e-5)

    obs = place(env, [offset(0.2)])               # below the cap: exactly gain * distance
    a = gh.act(obs, env, 1)['drone_0']
    assert np.allclose(a, [-0.2, 0.0, 0.0, 0.0], atol=1e-5)

    env2, obs2 = make_env(cfg, [offset(2.0)], [GOAL])
    gh2 = GoalHoldPolicy(ConstantPolicy(), cfg)
    assert gh2.act(obs2, env2, 1)['drone_0'] == [1.0, 1.0, 1.0, 0.5]


# -------------------------------------------------------------- C. hysteresis
def test_hysteresis():
    print_test_header('C', 'hysteresis',
                      "starts under engage_dist (0.5), continues until release_dist (0.7)")
    cfg = config()
    env, obs = make_env(cfg, [offset(0.6)], [GOAL])
    gh = GoalHoldPolicy(ConstantPolicy(), cfg)
    assert gh.act(obs, env, 1)['drone_0'] == [1.0, 1.0, 1.0, 0.5]       # 0.6: not started yet
    assert gh.act(place(env, [offset(0.45)]), env, 1)['drone_0'][3] == 0.0   # 0.45: holding
    assert gh.act(place(env, [offset(0.6)]), env, 1)['drone_0'][3] == 0.0    # 0.6: still holding
    assert gh.act(place(env, [offset(0.75)]), env, 1)['drone_0'] == [1.0, 1.0, 1.0, 0.5]  # released


# ------------------------------------------------- D. settled drones are held
def test_settled_drone_keeps_being_held():
    print_test_header('D', 'held after settling',
                      "the policy keeps flying settled drones; goal-hold must keep holding them")
    cfg = config(n_agents=2)
    env, obs = make_env(cfg, [offset(0.3), FAR], [GOAL, [0.0, 20.0, 2.0]])
    # the wrapped policy would push every drone off at 0.5 m/s
    gh = GoalHoldPolicy(ConstantPolicy((0.5, 0.0, 0.0, 0.0)), cfg)
    settled_at = None
    for step in range(200):
        obs, _, done, info = env.step(gh.act(obs, env, 2))
        assert not done
        if settled_at is None and info['drone_reached_goal'][0]:
            settled_at = step
    assert settled_at is not None, "drone_0 never settled"
    assert settled_at < 140, f"settled at step {settled_at}"
    dist = np.linalg.norm(env.adapter.get_positions()[0] - np.array(GOAL))
    assert dist < 0.05 and gh.holding[0], f"drifted to {dist:.3f} m after settling"


# ------------------------------------------------------------- E. the guard
def test_guard_close_neighbour():
    print_test_header('E1', 'guard: neighbour',
                      "neighbour < 0.9 m hands control back; holding resumes beyond 1.4 m")
    cfg = config(n_agents=2)
    env, _ = make_env(cfg, [offset(0.3), FAR], [GOAL, FAR])
    gh = GoalHoldPolicy(ConstantPolicy(), cfg)
    me = offset(0.3)
    near = lambda d: [me[0], me[1] + d, me[2]]
    policy_action = [1.0, 1.0, 1.0, 0.5]
    assert gh.act(place(env, [me, near(0.8)]), env, 2)['drone_0'] == policy_action   # too close
    assert gh.act(place(env, [me, near(1.2)]), env, 2)['drone_0'] == policy_action   # not clear yet
    assert gh.act(place(env, [me, near(1.5)]), env, 2)['drone_0'][3] == 0.0          # clear: holding
    assert gh.stats['guard_steps'] == 2


def test_guard_close_obstacle():
    print_test_header('E2', 'guard: obstacle',
                      "LiDAR ring hit < 0.5 m hands control back; holding resumes beyond 0.7 m")
    cfg = config()
    env, _ = make_env(cfg, [offset(0.3)], [GOAL])
    gh = GoalHoldPolicy(ConstantPolicy(), cfg)
    rays = env.obs_processor.ring_rays

    def with_ring_hit(d):
        raw = env.adapter.get_raw_state()
        ring = [env.lidar_range] * rays
        ring[0] = d
        raw['drone_0']['lidar_ring'] = ring
        return env._build_obs(raw)

    policy_action = [1.0, 1.0, 1.0, 0.5]
    assert gh.act(with_ring_hit(0.3), env, 1)['drone_0'] == policy_action
    assert gh.act(with_ring_hit(0.6), env, 1)['drone_0'] == policy_action
    assert gh.act(with_ring_hit(0.8), env, 1)['drone_0'][3] == 0.0


# ------------------------------------------------ F. episodes, pass-through
def test_new_episode_resets_and_attributes_pass_through():
    print_test_header('F', 'new episode / pass-through',
                      "hold state is cleared for a new episode; ckpt etc. come from the wrapped policy")
    cfg = config()
    env, obs = make_env(cfg, [offset(0.6)], [GOAL])
    gh = GoalHoldPolicy(ConstantPolicy(), cfg)
    gh.act(place(env, [offset(0.4)]), env, 1)
    assert gh.holding[0]
    other_goal = [5.0, 5.0, 2.0]
    obs = env.start_episode({'drone_0': [5.6, 5.0, 2.0]}, {'drone_0': other_goal}, obstacles=[])
    assert gh.act(obs, env, 1)['drone_0'] == [1.0, 1.0, 1.0, 0.5]   # 0.6 m, new episode: not holding
    assert gh.ckpt == {'episode': 123}


def test_new_episode_with_the_same_goals_resets():
    print_test_header('F2', 'new episode, same goals',
                      "the episode counter clears hold and guard state even when the goals repeat")
    cfg = config(n_agents=2)
    env, _ = make_env(cfg, [offset(0.3), FAR], [GOAL, FAR])
    gh = GoalHoldPolicy(ConstantPolicy(), cfg)
    me = offset(0.3)
    gh.act(place(env, [me, [me[0], me[1] + 0.8, me[2]]]), env, 2)      # neighbour close: guarded
    assert gh.guarded[0]
    obs = env.start_episode({'drone_0': offset(0.6), 'drone_1': FAR},
                            {'drone_0': GOAL, 'drone_1': FAR}, obstacles=[])  # same goals
    assert gh.act(obs, env, 2)['drone_0'] == [1.0, 1.0, 1.0, 0.5]     # 0.6 m, fresh: not holding
    assert not gh.guarded[0] and not gh.holding[0]


# ------------------------------------------------------------ G/H. noise seed
def first_obs(env, spawn, goal, steps=0):
    """Start an episode, fly `steps` steps at hover, return all observations."""
    obs = env.start_episode({'drone_0': spawn}, {'drone_0': goal}, obstacles=[])
    out = [obs['local_fixed'].clone()]
    for _ in range(steps):
        obs, _, _, _ = env.step({'drone_0': [0.0, 0.0, 0.0, 0.0]})
        out.append(obs['local_fixed'].clone())
    return out


def test_noise_default_unchanged():
    print_test_header('G', 'noise default unchanged',
                      "no per_episode_seed: one generator runs on across episodes, as before")
    spawn, goal = [3.0, 0.0, 2.0], GOAL
    env_a = SwarmEnv(config(noise=True, goal_hold=False), adapter=MockAdapter(1))
    env_b = SwarmEnv(config(noise=True, per_episode=False, goal_hold=False), adapter=MockAdapter(1))
    assert env_a.noise_on and not env_a.noise_per_episode
    rng = env_a._noise_rng
    run_a = first_obs(env_a, spawn, goal, 5) + first_obs(env_a, spawn, goal, 5)
    run_b = first_obs(env_b, spawn, goal, 5) + first_obs(env_b, spawn, goal, 5)
    assert env_a._noise_rng is rng                      # never replaced
    for x, y in zip(run_a, run_b):
        assert np.array_equal(x.numpy(), y.numpy())
    # one stream: the same scenario run twice in a row sees different noise
    assert not np.array_equal(run_a[0].numpy(), run_a[6].numpy())


def test_per_episode_seed_same_scenario_same_noise():
    print_test_header('H', 'per-episode noise seed',
                      "same scenario = same noise, whatever ran before; other scenarios differ")
    spawn, goal = [3.0, 0.0, 2.0], GOAL
    env = SwarmEnv(config(noise=True, per_episode=True, goal_hold=False), adapter=MockAdapter(1))
    first = first_obs(env, spawn, goal, 5)
    first_obs(env, [0.0, 3.0, 2.0], goal, 17)            # a different, longer episode in between
    again = first_obs(env, spawn, goal, 5)
    for x, y in zip(first, again):
        assert np.array_equal(x.numpy(), y.numpy())
    # a different scenario gets different noise (compared with the noise-free view)
    clean = SwarmEnv(config(noise=False, goal_hold=False), adapter=MockAdapter(1))
    other_spawn = [3.0, 0.5, 2.0]
    noise_first = first[0].numpy() - first_obs(clean, spawn, goal)[0].numpy()
    noise_other = (first_obs(env, other_spawn, goal)[0].numpy()
                   - first_obs(clean, other_spawn, goal)[0].numpy())
    assert np.abs(noise_first).max() > 0
    assert not np.allclose(noise_first, noise_other)
