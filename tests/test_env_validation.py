"""
test_env_validation.py
----------------------
Bottom-up validation ladder for the SwarmEnv before any training runs.

Tests:
  A. Action direction — positive command moves drone toward positive-direction goal
  B. Settlement logic — counter increments, latches, resets correctly
  C. Reward ordering — settle > controlled entry > flythrough > partial > no-movement
  D. Heuristic P-controller (mock adapter) — proportional controller achieves 100%
  E. Heuristic P-controller (real PyBullet) — same test through production path
  F. Boundary discontinuity — entering goal zone must not cause reward cliff

Run: conda run -n utm-swarm-marl python tests/test_env_validation.py
"""

import copy
import numpy as np
import yaml
import os
import sys
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from envs.swarm_env import SwarmEnv
from modules.reward_engine import RewardEngine


def load_config():
    path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'default_config.yaml')
    with open(path) as f:
        return yaml.safe_load(f)


class MockAdapter:
    """Adapter that lets us script exact positions and velocities."""

    def __init__(self, n_agents):
        self.n_agents = n_agents
        self.positions = np.zeros((n_agents, 3), dtype=np.float32)
        self.velocities = np.zeros((n_agents, 3), dtype=np.float32)
        self.goals = {}

    def reset_world(self, spawn_positions, goal_positions):
        for i in range(self.n_agents):
            key = f'drone_{i}'
            self.positions[i] = spawn_positions[key]
            self.velocities[i] = [0, 0, 0]
            self.goals[key] = np.array(goal_positions[key], dtype=np.float32)

    def apply_actions(self, actions):
        for i in range(self.n_agents):
            key = f'drone_{i}'
            a = np.array(actions[key][:3], dtype=np.float32)
            self.velocities[i] = np.clip(a, -2.0, 2.0)

    def step_simulation(self):
        dt = 0.032
        self.positions = self.positions + self.velocities * dt

    def get_raw_state(self):
        state = {'goals': {}}
        for i in range(self.n_agents):
            key = f'drone_{i}'
            state[key] = {
                'position': self.positions[i].tolist(),
                'velocity': self.velocities[i].tolist(),
                'orientation': [0.0, 0.0, 0.0],
                'lidar': [0.0] * 14,
            }
            state['goals'][key] = self.goals[key].tolist()
        return state

    def get_positions(self):
        return self.positions.copy()

    def close(self):
        pass


def make_env_with_mock(config, n_agents=1):
    cfg = copy.deepcopy(config)
    cfg['environment']['n_agents'] = n_agents
    adapter = MockAdapter(n_agents)
    env = SwarmEnv(cfg, adapter=adapter)
    return env, adapter


def run_scripted_episode(config, velocity_fn, max_steps=200, n_agents=1):
    """Run a scripted episode through full env.step() and return total reward + info."""
    cfg = copy.deepcopy(config)
    cfg['environment']['n_agents'] = n_agents
    cfg['environment']['max_steps'] = max_steps

    adapter = MockAdapter(n_agents)
    env = SwarmEnv(cfg, adapter=adapter)

    spawn = {f'drone_{i}': [0.0, float(i * 3), 2.0] for i in range(n_agents)}
    goal = {f'drone_{i}': [1.0, float(i * 3), 2.0] for i in range(n_agents)}
    adapter.reset_world(spawn, goal)
    for i in range(n_agents):
        adapter.goals[f'drone_{i}'] = np.array(goal[f'drone_{i}'], dtype=np.float32)
    env.goals = np.array([goal[f'drone_{i}'] for i in range(n_agents)], dtype=np.float32)
    env.prev_dist = np.linalg.norm(
        np.array([spawn[f'drone_{i}'] for i in range(n_agents)]) -
        np.array([goal[f'drone_{i}'] for i in range(n_agents)]), axis=1
    ).astype(np.float32)
    env.prev_actions = np.zeros((n_agents, 4), dtype=np.float32)
    env.drone_reached_goal = [False] * n_agents
    env.settle_counter = np.zeros(n_agents, dtype=np.int64)

    env.best_settle_counter = np.zeros(n_agents, dtype=np.int64)
    env.ever_entered_goal = [False] * n_agents
    env.initial_dist = env.prev_dist.copy()
    env.min_dist_ever = env.prev_dist.copy()
    env.episode_manager.reset()

    total_reward = 0.0
    component_totals = {}
    last_info = {}
    for step in range(max_steps):
        actions = {}
        for i in range(n_agents):
            vel = velocity_fn(step, adapter.positions[i], env.goals[i])
            actions[f'drone_{i}'] = list(vel) + [0.0]
        _, rewards, done, info = env.step(actions)
        step_reward = sum(rewards.values())
        total_reward += step_reward
        for k, v in info.get('reward_components', {}).items():
            component_totals[k] = component_totals.get(k, 0.0) + v
        last_info = info
        if done:
            break

    env.close()
    return total_reward, last_info.get('success', False), component_totals, step + 1


# ============================================================
# TEST A: Action direction
# ============================================================

class TestActionDirection:

    def test_all_three_axes(self):
        config = load_config()
        for axis, goal_pos in [(0, [1, 0, 2]), (1, [0, 1, 2]), (2, [0, 0, 3])]:
            env, adapter = make_env_with_mock(config, n_agents=1)

            spawn = {'drone_0': [0.0, 0.0, 2.0]}
            goal = {'drone_0': goal_pos}
            adapter.reset_world(spawn, goal)
            adapter.goals = {'drone_0': np.array(goal_pos, dtype=np.float32)}
            env.goals = np.array([goal_pos], dtype=np.float32)
            env.prev_dist = np.linalg.norm(np.array([[0, 0, 2.0]]) - np.array([goal_pos]), axis=1).astype(np.float32)
            env.prev_actions = np.zeros((1, 4), dtype=np.float32)
            env.drone_reached_goal = [False]
            env.settle_counter = np.zeros(1, dtype=np.int64)

            env.best_settle_counter = np.zeros(1, dtype=np.int64)
            env.ever_entered_goal = [False]
            env.initial_dist = env.prev_dist.copy()
            env.min_dist_ever = env.prev_dist.copy()
            env.episode_manager.reset()

            action = [0.0, 0.0, 0.0, 0.0]
            action[axis] = 1.0
            _, rewards, _, _ = env.step({'drone_0': action})

            new_dist = np.linalg.norm(adapter.positions[0] - np.array(goal_pos))
            init = env.initial_dist[0]
            assert new_dist < init + 0.001, \
                f"Axis {axis}: distance should decrease. Was {init}, now {new_dist}"
            env.close()

    def test_goal_vector_sign(self):
        config = load_config()
        env, adapter = make_env_with_mock(config, n_agents=1)

        spawn = {'drone_0': [0.0, 0.0, 2.0]}
        goal = {'drone_0': [2.0, 0.0, 2.0]}
        adapter.reset_world(spawn, goal)
        adapter.goals = {'drone_0': np.array([2.0, 0.0, 2.0], dtype=np.float32)}
        env.goals = np.array([[2.0, 0.0, 2.0]], dtype=np.float32)

        raw = adapter.get_raw_state()
        obs = env._build_obs(raw)
        lf = obs['local_fixed'].cpu().numpy()[0]
        rel_goal = lf[9:12]
        assert rel_goal[0] > 0, f"Goal vector x should be positive, got {rel_goal}"
        env.close()


# ============================================================
# TEST B: Settlement logic
# ============================================================

class TestSettlement:

    def _make_env_at_goal(self):
        config = load_config()
        env, adapter = make_env_with_mock(config, n_agents=1)

        spawn = {'drone_0': [5.0, 5.0, 2.0]}
        goal = {'drone_0': [5.0, 5.0, 2.0]}
        adapter.reset_world(spawn, goal)
        adapter.goals = {'drone_0': np.array([5.0, 5.0, 2.0], dtype=np.float32)}
        env.goals = np.array([[5.0, 5.0, 2.0]], dtype=np.float32)
        env.prev_dist = np.array([0.0], dtype=np.float32)
        env.prev_actions = np.zeros((1, 4), dtype=np.float32)
        env.drone_reached_goal = [False]
        env.settle_counter = np.zeros(1, dtype=np.int64)

        env.best_settle_counter = np.zeros(1, dtype=np.int64)
        env.ever_entered_goal = [False]
        env.initial_dist = np.array([0.0], dtype=np.float32)
        env.min_dist_ever = np.array([0.0], dtype=np.float32)
        env.episode_manager.reset()
        return env, adapter

    def test_counter_increments(self):
        env, adapter = self._make_env_at_goal()
        zero_action = {'drone_0': [0.0, 0.0, 0.0, 0.0]}
        for step in range(30):
            env.step(zero_action)
            assert env.settle_counter[0] == step + 1
            assert not env.drone_reached_goal[0]
        env.close()

    def test_latch_at_31(self):
        env, adapter = self._make_env_at_goal()
        zero_action = {'drone_0': [0.0, 0.0, 0.0, 0.0]}
        for step in range(31):
            _, _, done, info = env.step(zero_action)
        assert env.drone_reached_goal[0]
        assert info['success']
        assert done
        env.close()

    def test_counter_resets_on_speed(self):
        env, adapter = self._make_env_at_goal()
        zero_action = {'drone_0': [0.0, 0.0, 0.0, 0.0]}
        for _ in range(20):
            env.step(zero_action)
        assert env.settle_counter[0] == 20
        env.step({'drone_0': [1.0, 0.0, 0.0, 0.0]})
        assert env.settle_counter[0] == 0
        env.close()

    def test_counter_resets_on_leaving_radius(self):
        env, adapter = self._make_env_at_goal()
        zero_action = {'drone_0': [0.0, 0.0, 0.0, 0.0]}
        for _ in range(15):
            env.step(zero_action)
        adapter.positions[0] = np.array([5.0, 5.0, 3.0])
        env.step(zero_action)
        assert env.settle_counter[0] == 0
        env.close()

    def test_goal_reached_bonus_fires_once(self):
        env, adapter = self._make_env_at_goal()
        zero_action = {'drone_0': [0.0, 0.0, 0.0, 0.0]}
        bonuses = []
        for step in range(35):
            _, rewards, done, _ = env.step(zero_action)
            bonuses.append(rewards['drone_0'])
            if done:
                break
        goal_bonus = env.reward_engine.goal_reached_bonus
        bonus_steps = [i for i, r in enumerate(bonuses) if r > goal_bonus * 0.5]
        assert len(bonus_steps) == 1, f"Bonus should fire once, fired at: {bonus_steps}"
        assert bonus_steps[0] == 30
        env.close()

    def test_three_drones_all_settle(self):
        config = load_config()
        cfg = copy.deepcopy(config)
        cfg['environment']['n_agents'] = 3
        adapter = MockAdapter(3)
        env = SwarmEnv(cfg, adapter=adapter)

        spawns = {f'drone_{i}': [float(i * 3), 0.0, 2.0] for i in range(3)}
        goals = {f'drone_{i}': [float(i * 3), 0.0, 2.0] for i in range(3)}
        adapter.reset_world(spawns, goals)
        for i in range(3):
            adapter.goals[f'drone_{i}'] = np.array(goals[f'drone_{i}'], dtype=np.float32)
        env.goals = np.array([goals[f'drone_{i}'] for i in range(3)], dtype=np.float32)
        env.prev_dist = np.zeros(3, dtype=np.float32)
        env.prev_actions = np.zeros((3, 4), dtype=np.float32)
        env.drone_reached_goal = [False] * 3
        env.settle_counter = np.zeros(3, dtype=np.int64)

        env.best_settle_counter = np.zeros(3, dtype=np.int64)
        env.ever_entered_goal = [False] * 3
        env.initial_dist = np.zeros(3, dtype=np.float32)
        env.min_dist_ever = np.zeros(3, dtype=np.float32)
        env.episode_manager.reset()

        zero_actions = {f'drone_{i}': [0.0, 0.0, 0.0, 0.0] for i in range(3)}
        for step in range(31):
            _, _, done, info = env.step(zero_actions)
        assert done and info['success'] and all(env.drone_reached_goal)
        env.close()


# ============================================================
# TEST C: Reward ordering (strict)
# ============================================================

class TestRewardOrdering:

    def test_reward_ordering_strict(self):
        config = load_config()

        def settle_controller(step, pos, goal):
            err = goal - pos
            dist = np.linalg.norm(err)
            if dist < 0.01:
                return [0.0, 0.0, 0.0]
            direction = err / dist
            speed = min(1.5, dist * 3.0)
            return (direction * speed).tolist()

        def enter_and_hold(step, pos, goal):
            """Slower approach than settle_controller, but also stops and holds
            inside the goal — this ALSO settles, just via a different speed
            profile. Confirms settle reward isn't sensitive to approach speed."""
            err = goal - pos
            dist = np.linalg.norm(err)
            if dist < 0.3:
                return [0.0, 0.0, 0.0]
            if dist < 0.01:
                return [0.0, 0.0, 0.0]
            direction = err / dist
            speed = min(1.0, dist * 2.0)
            return (direction * speed).tolist()

        def enter_and_leave(step, pos, goal):
            """Enter the goal zone, then drift back out before 31 steps
            elapse — genuinely never settles. Tests that partial entry
            without holding is worth less than a full settle."""
            err = goal - pos
            dist = np.linalg.norm(err)
            if dist < 0.05:
                # Reverse direction once inside — drift back out
                return (-err / (dist + 1e-8) * 0.3).tolist() if dist > 1e-6 else [0.3, 0.0, 0.0]
            direction = err / dist
            speed = min(1.0, dist * 2.0)
            return (direction * speed).tolist()

        def flythrough(step, pos, goal):
            err = goal - pos
            dist = np.linalg.norm(err)
            if dist < 0.001:
                return [0.5, 0.0, 0.0]
            direction = err / dist
            return (direction * 1.5).tolist()

        def partial(step, pos, goal):
            if step < 10:
                err = goal - pos
                dist = np.linalg.norm(err)
                if dist < 0.001:
                    return [0.0, 0.0, 0.0]
                direction = err / dist
                return (direction * 0.5).tolist()
            return [0.0, 0.0, 0.0]

        def no_movement(step, pos, goal):
            return [0.0, 0.0, 0.0]

        r_settle, s_settle, c_settle, steps_settle = run_scripted_episode(config, settle_controller, max_steps=200)
        r_hold, s_hold, c_hold, steps_hold = run_scripted_episode(config, enter_and_hold, max_steps=200)
        r_leave, s_leave, c_leave, steps_leave = run_scripted_episode(config, enter_and_leave, max_steps=200)
        r_fly, s_fly, c_fly, steps_fly = run_scripted_episode(config, flythrough, max_steps=200)
        r_partial, _, c_partial, steps_partial = run_scripted_episode(config, partial, max_steps=200)
        r_none, _, c_none, steps_none = run_scripted_episode(config, no_movement, max_steps=200)

        print(f"\n  Settle (fast approach):    reward={r_settle:+8.2f}  success={s_settle}  steps={steps_settle}")
        print(f"  Enter+hold (slow approach):reward={r_hold:+8.2f}  success={s_hold}  steps={steps_hold}")
        print(f"  Enter+leave (no settle):   reward={r_leave:+8.2f}  success={s_leave}  steps={steps_leave}")
        print(f"  Flythrough:                reward={r_fly:+8.2f}  success={s_fly}  steps={steps_fly}")
        print(f"  Partial:                   reward={r_partial:+8.2f}  steps={steps_partial}")
        print(f"  No movement:               reward={r_none:+8.2f}  steps={steps_none}")

        print(f"\n  Settle components:     {c_settle}")
        print(f"  Enter+hold components: {c_hold}")
        print(f"  Enter+leave components:{c_leave}")
        print(f"  Flythrough components: {c_fly}")
        print(f"  Partial components:    {c_partial}")
        print(f"  No-move components:    {c_none}")

        # Both settle and enter_and_hold actually settle (stop-and-stay), just
        # at different approach speeds — reward should be approach-speed-insensitive
        assert s_settle, "Settle controller must achieve success"
        assert s_hold, "Enter+hold must also settle (it stops inside the goal)"
        assert abs(r_settle - r_hold) < 5.0, \
            f"Both settling controllers should get similar reward regardless of approach speed: " \
            f"settle={r_settle:.1f} vs hold={r_hold:.1f}"

        # enter_and_leave genuinely never settles — must get entry bonus but not goal_reached
        assert not s_leave, "Enter+leave must NOT count as success (never held 31 steps)"
        assert c_leave.get('goal_entry', 0.0) > 0, "Enter+leave should still get the one-time entry bonus"
        assert c_leave.get('goal_reached', 0.0) == 0.0, "Enter+leave must NOT get the terminal goal_reached bonus"

        # Full ordering: settling beats non-settling entry beats flythrough beats passivity
        assert r_settle > r_leave, \
            f"Settle ({r_settle:.1f}) must beat enter+leave ({r_leave:.1f})"
        assert r_leave > r_fly, \
            f"Enter+leave ({r_leave:.1f}) must beat flythrough ({r_fly:.1f})"
        assert r_fly >= r_partial, \
            f"Flythrough ({r_fly:.1f}) must beat partial ({r_partial:.1f})"
        assert r_partial >= r_none, \
            f"Partial ({r_partial:.1f}) must beat no-movement ({r_none:.1f})"

    def test_full_horizon_3_agents(self):
        """Test reward ordering at production scale: 3 agents, 1000 steps."""
        config = load_config()

        def settle_controller(step, pos, goal):
            err = goal - pos
            dist = np.linalg.norm(err)
            if dist < 0.01:
                return [0.0, 0.0, 0.0]
            direction = err / dist
            speed = min(1.5, dist * 3.0)
            return (direction * speed).tolist()

        def no_movement(step, pos, goal):
            return [0.0, 0.0, 0.0]

        r_settle, s_settle, _, _ = run_scripted_episode(config, settle_controller, max_steps=1000, n_agents=3)
        r_none, _, _, _ = run_scripted_episode(config, no_movement, max_steps=1000, n_agents=3)

        print(f"\n  3-agent settle: reward={r_settle:+.2f}  success={s_settle}")
        print(f"  3-agent idle:   reward={r_none:+.2f}")

        assert s_settle, "3-agent settle must succeed"
        assert r_settle > r_none, \
            f"3-agent settle ({r_settle:.1f}) must beat idle ({r_none:.1f})"
        assert r_settle > 0, \
            f"Successful 3-agent episode must have positive reward, got {r_settle:.1f}"


# ============================================================
# TEST D: Heuristic P-controller (mock adapter)
# ============================================================

class TestHeuristicControllerMock:

    def test_p_controller_single_drone(self):
        config = load_config()
        cfg = copy.deepcopy(config)
        cfg['environment']['n_agents'] = 1

        successes = 0
        n_trials = 20
        for trial in range(n_trials):
            np.random.seed(trial)
            adapter = MockAdapter(1)
            env = SwarmEnv(cfg, adapter=adapter)
            env.reset()

            for step in range(1000):
                pos = adapter.get_positions()[0]
                goal = env.goals[0]
                err = goal - pos
                dist = np.linalg.norm(err)
                if dist < 0.01:
                    vel = np.zeros(3)
                else:
                    direction = err / dist
                    speed = min(1.5, dist * 3.0)
                    vel = direction * speed
                _, _, done, info = env.step({'drone_0': vel.tolist() + [0.0]})
                if done:
                    break
            if info.get('success', False):
                successes += 1
            env.close()

        sr = successes / n_trials
        print(f"\n  Mock P-controller (1 drone): {successes}/{n_trials} = {sr:.0%}")
        assert sr >= 0.95, f"Should succeed ≥95%, got {sr:.0%}"


# ============================================================
# TEST E: Heuristic P-controller (real PyBullet adapter)
# ============================================================

class TestHeuristicControllerReal:

    def test_p_controller_real_single(self):
        config = load_config()
        cfg = copy.deepcopy(config)
        cfg['environment']['n_agents'] = 1

        successes = 0
        n_trials = 20
        for trial in range(n_trials):
            np.random.seed(trial)
            env = SwarmEnv(cfg)
            env.reset()

            for step in range(1000):
                pos = env.adapter.get_positions()[0]
                goal = env.goals[0]
                err = goal - pos
                dist = np.linalg.norm(err)
                if dist < 0.01:
                    vel = np.zeros(3)
                else:
                    direction = err / dist
                    speed = min(1.5, dist * 3.0)
                    vel = direction * speed
                _, _, done, info = env.step({'drone_0': vel.tolist() + [0.0]})
                if done:
                    break
            if info.get('success', False):
                successes += 1
            env.close()

        sr = successes / n_trials
        print(f"\n  Real P-controller (1 drone): {successes}/{n_trials} = {sr:.0%}")
        assert sr >= 0.90, f"Should succeed ≥90%, got {sr:.0%}"

    def test_p_controller_real_three_drones(self):
        config = load_config()
        cfg = copy.deepcopy(config)
        cfg['environment']['n_agents'] = 3

        successes = 0
        n_trials = 20
        for trial in range(n_trials):
            np.random.seed(trial + 100)
            env = SwarmEnv(cfg)
            env.reset()

            for step in range(1000):
                actions = {}
                positions = env.adapter.get_positions()
                for i in range(3):
                    err = env.goals[i] - positions[i]
                    dist = np.linalg.norm(err)
                    if dist < 0.01:
                        vel = np.zeros(3)
                    else:
                        direction = err / dist
                        speed = min(1.5, dist * 3.0)
                        vel = direction * speed
                    actions[f'drone_{i}'] = vel.tolist() + [0.0]
                _, _, done, info = env.step(actions)
                if done:
                    break
            if info.get('success', False):
                successes += 1
            env.close()

        sr = successes / n_trials
        print(f"\n  Real P-controller (3 drones): {successes}/{n_trials} = {sr:.0%}")
        assert sr >= 0.90, f"Should succeed ≥90%, got {sr:.0%}"

    def test_p_controller_from_observation(self):
        """P-controller using observation goal vector, not env.goals directly."""
        config = load_config()
        cfg = copy.deepcopy(config)
        cfg['environment']['n_agents'] = 1

        successes = 0
        n_trials = 20
        for trial in range(n_trials):
            np.random.seed(trial + 200)
            env = SwarmEnv(cfg)
            obs = env.reset()

            for step in range(1000):
                lf = obs['local_fixed'].cpu().numpy()[0]
                # obs layout: [0:3]=pos, [3:6]=vel, [6:9]=ori, [9:12]=rel_goal, [12]=dist
                rel_goal = lf[9:12]
                goal_dist = lf[12]

                if goal_dist < 0.01:
                    vel = np.zeros(3)
                else:
                    direction = rel_goal / (np.linalg.norm(rel_goal) + 1e-8)
                    speed = min(1.5, goal_dist * 3.0)
                    vel = direction * speed

                obs, _, done, info = env.step({'drone_0': vel.tolist() + [0.0]})
                if done:
                    break
            if info.get('success', False):
                successes += 1
            env.close()

        sr = successes / n_trials
        print(f"\n  Obs-based P-controller (1 drone): {successes}/{n_trials} = {sr:.0%}")
        assert sr >= 0.90, f"Should succeed ≥90%, got {sr:.0%}"


# ============================================================
# TEST F: Boundary discontinuity
# ============================================================

class TestBoundaryDiscontinuity:
    """Crossing into goal zone must not cause a large reward drop."""

    def test_no_reward_cliff_at_boundary(self):
        config = load_config()

        for test_speed in [0.0, 0.1, 0.15, 0.5, 1.0, 2.0]:
            def make_controller(spd):
                def ctrl(step, pos, goal):
                    err = goal - pos
                    dist = np.linalg.norm(err)
                    if dist < 0.001:
                        return [0.0, 0.0, 0.0]
                    direction = err / dist
                    return (direction * spd).tolist()
                return ctrl

            # 10-step episode starting at 0.55m (just outside) and 0.45m (just inside)
            for start_dist, label in [(0.55, "outside"), (0.45, "inside")]:
                cfg = copy.deepcopy(config)
                cfg['environment']['n_agents'] = 1
                cfg['environment']['max_steps'] = 10

                adapter = MockAdapter(1)
                env = SwarmEnv(cfg, adapter=adapter)

                spawn = {'drone_0': [start_dist, 0.0, 2.0]}
                goal = {'drone_0': [0.0, 0.0, 2.0]}
                adapter.reset_world(spawn, goal)
                adapter.goals = {'drone_0': np.array([0.0, 0.0, 2.0], dtype=np.float32)}
                env.goals = np.array([[0.0, 0.0, 2.0]], dtype=np.float32)
                env.prev_dist = np.array([start_dist], dtype=np.float32)
                env.prev_actions = np.zeros((1, 4), dtype=np.float32)
                env.drone_reached_goal = [False]
                env.settle_counter = np.zeros(1, dtype=np.int64)

                env.best_settle_counter = np.zeros(1, dtype=np.int64)
                env.ever_entered_goal = [False]
                env.initial_dist = np.array([start_dist], dtype=np.float32)
                env.min_dist_ever = np.array([start_dist], dtype=np.float32)
                env.episode_manager.reset()

                total = 0.0
                for step in range(10):
                    pos = adapter.positions[0]
                    goal_arr = env.goals[0]
                    vel = make_controller(test_speed)(step, pos, goal_arr)
                    _, rewards, done, _ = env.step({'drone_0': vel + [0.0]})
                    total += rewards['drone_0']
                    if done:
                        break
                env.close()

                if label == "outside":
                    r_outside = total
                else:
                    r_inside = total

            # Being inside should not be punished relative to outside
            assert r_inside >= r_outside - 1.0, \
                f"Speed {test_speed}: inside reward ({r_inside:.2f}) must not cliff vs outside ({r_outside:.2f})"
            print(f"  Speed {test_speed:4.1f}: outside={r_outside:+.3f}  inside={r_inside:+.3f}  OK")


if __name__ == '__main__':
    test_classes = [
        TestActionDirection,
        TestSettlement,
        TestRewardOrdering,
        TestHeuristicControllerMock,
        TestHeuristicControllerReal,
        TestBoundaryDiscontinuity,
    ]
    passed, failed = 0, 0
    for cls in test_classes:
        print(f"\n{'='*60}")
        print(f"  {cls.__name__}")
        print(f"{'='*60}")
        obj = cls()
        methods = sorted([m for m in dir(obj) if m.startswith('test_')])
        for m in methods:
            name = f"{cls.__name__}.{m}"
            try:
                getattr(obj, m)()
                print(f"  PASS  {name}")
                passed += 1
            except Exception as e:
                print(f"  FAIL  {name}: {e}")
                traceback.print_exc()
                failed += 1
    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed")
    if failed:
        sys.exit(1)
