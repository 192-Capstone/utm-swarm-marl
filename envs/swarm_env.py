"""
swarm_env.py
------------
SwarmEnv - the conductor. It wires the simulator adapter, ObservationProcessor,
RewardEngine and EpisodeManager into the reset()/step() interface MAPPOTrainer
expects. This is the keystone that connects the MAPPO engine to a simulator.

Interface used by MAPPOTrainer:
    reset() -> obs_dict {'local_fixed','neighbor_states','neighbor_mask','global_state'}
    step(actions_dict) -> (obs_dict, rewards_dict, done, info)
        actions_dict : {'drone_i': [vx, vy, vz, yaw_rate]}
        rewards_dict : {'drone_i': float}      (per-drone total reward)
        done         : bool
        info         : {'success': bool, 'reason': str|None}
    set_curriculum_stage(stage)

The env is simulator-agnostic: swap PyBulletAdapter for a WebotsAdapter that
emits the same raw-state dict and nothing else here changes.
"""

import numpy as np
import torch

from modules.observation_processor import ObservationProcessor
from modules.reward_engine import RewardEngine
from envs.episode_manager import EpisodeManager
from envs.pybullet_adapter import PyBulletAdapter


class SwarmEnv:
    def __init__(self, config, device=None, gui: bool = False, adapter=None):
        self.config   = config
        self.device   = device or torch.device('cpu')
        self.n_agents = config['environment']['n_agents']
        self.world_size       = config['environment'].get('world_size', 50)
        self.goal_threshold   = config['environment'].get('goal_threshold', 0.5)
        # collision_radius falls back to 0.2 to match RewardEngine's default
        self.collision_radius = config['environment'].get('collision_radius', 0.2)
        self.min_spawn_sep    = config.get('scenario_manager', {}).get('min_spawn_separation', 5.0)
        self.curriculum_stage = 1

        self.adapter        = adapter or PyBulletAdapter(config, gui=gui)
        self.obs_processor  = ObservationProcessor(config, self.device)
        self.reward_engine  = RewardEngine(config)
        self.episode_manager = EpisodeManager(config)

        self.goals        = None
        self.prev_dist    = None
        self.prev_actions = None

        # Goal distance curriculum: start close, widen as success rate improves
        goal_dist_cfg = config.get('goal_curriculum', {})
        self.goal_dist_min   = goal_dist_cfg.get('start_dist', 3.0)
        self.goal_dist_max   = goal_dist_cfg.get('end_dist', 20.0)
        self.goal_dist_current = self.goal_dist_min
        self.goal_dist_step  = goal_dist_cfg.get('step_dist', 1.0)
        self.goal_dist_threshold = goal_dist_cfg.get('advance_sr', 0.3)
        self.goal_dist_window = goal_dist_cfg.get('window', 50)
        self._recent_successes = []

    # ---- world layout (a tiny Stage-1 scenario generator) -------------------
    def _sample_positions(self):
        """Random spawns + goals inside the central area; spawns kept separated."""
        reach = (self.world_size / 2.0) * 0.4     # keep away from the far walls

        def rand_xyz():
            return np.array([np.random.uniform(-reach, reach),
                             np.random.uniform(-reach, reach),
                             np.random.uniform(1.0, 5.0)], dtype=np.float32)

        def goal_near(spawn):
            """Sample a goal within goal_dist_current of the spawn."""
            for _ in range(200):
                g = rand_xyz()
                d = np.linalg.norm(g - spawn)
                if d <= self.goal_dist_current and d >= 1.0:
                    return g
            # Fallback: place goal exactly goal_dist_current away in a random direction
            direction = np.random.randn(3).astype(np.float32)
            direction[2] = abs(direction[2])  # keep altitude positive
            direction /= (np.linalg.norm(direction) + 1e-8)
            g = spawn + direction * self.goal_dist_current
            g[2] = np.clip(g[2], 1.0, 5.0)
            return g

        spawns, placed = {}, []
        for i in range(self.n_agents):
            cand = rand_xyz()
            for _ in range(100):
                if all(np.linalg.norm(cand - q) >= self.min_spawn_sep for q in placed):
                    break
                cand = rand_xyz()
            placed.append(cand)
            spawns[f'drone_{i}'] = cand.tolist()

        goals = {f'drone_{i}': goal_near(placed[i]).tolist()
                 for i in range(self.n_agents)}
        return spawns, goals

    def update_goal_curriculum(self, success: bool):
        """Call after each episode to potentially widen goal distance."""
        self._recent_successes.append(float(success))
        if len(self._recent_successes) > self.goal_dist_window:
            self._recent_successes.pop(0)
        if len(self._recent_successes) >= self.goal_dist_window:
            sr = np.mean(self._recent_successes)
            if sr >= self.goal_dist_threshold and self.goal_dist_current < self.goal_dist_max:
                old = self.goal_dist_current
                self.goal_dist_current = min(
                    self.goal_dist_current + self.goal_dist_step,
                    self.goal_dist_max,
                )
                self._recent_successes.clear()
                print(f"  [GoalCurriculum] Distance {old:.1f}m -> {self.goal_dist_current:.1f}m")

    # ---- RL interface -------------------------------------------------------
    def reset(self):
        spawns, goals = self._sample_positions()
        self.adapter.reset_world(spawns, goals)
        self.goals = np.array([goals[f'drone_{i}'] for i in range(self.n_agents)],
                              dtype=np.float32)
        self.episode_manager.reset()

        raw       = self.adapter.get_raw_state()
        positions = self.adapter.get_positions()
        self.prev_dist    = np.linalg.norm(positions - self.goals, axis=1)
        self.prev_actions = np.zeros((self.n_agents, 4), dtype=np.float32)
        self.drone_reached_goal = [False] * self.n_agents

        # Track closest approach to goal — diagnoses "almost succeeded" vs
        # "never made progress" for episodes that don't reach the goal
        self.initial_dist   = self.prev_dist.copy()
        self.min_dist_ever  = self.prev_dist.copy()

        return self._build_obs(raw)

    def step(self, actions_dict):
        # 1. apply actions + advance the simulator
        self.adapter.apply_actions(actions_dict)
        self.adapter.step_simulation()
        self.episode_manager.step()

        # 2. read the new world state
        raw       = self.adapter.get_raw_state()
        positions = self.adapter.get_positions()
        curr_dist = np.linalg.norm(positions - self.goals, axis=1)
        self.min_dist_ever = np.minimum(self.min_dist_ever, curr_dist)

        # 3. pairwise drone distances (feed proximity + drone-collision reward)
        drone_distances = [
            [float(np.linalg.norm(positions[i] - positions[j]))
             for j in range(self.n_agents) if j != i]
            for i in range(self.n_agents)
        ]
        # drone-drone collision flags (used for termination)
        drone_collision = [bool(d and min(d) < self.collision_radius)
                           for d in drone_distances]
        # obstacle collisions: none in Stage 1 (empty world)
        obstacle_collision = [False] * self.n_agents

        actions_arr = np.array([actions_dict[f'drone_{i}'] for i in range(self.n_agents)],
                               dtype=np.float32)

        # 4. track which drones JUST reached their goal (one-time flag)
        goal_just_reached = [False] * self.n_agents
        for i in range(self.n_agents):
            if not self.drone_reached_goal[i] and curr_dist[i] < self.goal_threshold:
                goal_just_reached[i] = True
                self.drone_reached_goal[i] = True

        reward_data = {
            'prev_dist':         self.prev_dist,
            'curr_dist':         curr_dist,
            'collision':         obstacle_collision,
            'drone_distances':   drone_distances,
            'action':            actions_arr,
            'prev_action':       self.prev_actions,
            'goal_just_reached': goal_just_reached,
            'drone_reached_goal': list(self.drone_reached_goal),
        }
        rewards = self.reward_engine.compute_rewards(reward_data)
        rewards_dict = {f'drone_{i}': float(rewards['total'][i])
                        for i in range(self.n_agents)}

        # 5. termination (collision / all-goals-reached / timeout)
        # Use latched drone_reached_goal flags: a drone that reached its
        # goal at ANY point counts, not just drones inside the radius
        # right now. This avoids penalising successful navigation that
        # overshoots or drifts after arrival.
        done, reason = self.episode_manager.check_done(
            positions.tolist(), self.goals.tolist(), drone_collision)
        if not done and all(self.drone_reached_goal):
            done, reason = True, 'success'
        if done:
            self.episode_manager.record_episode_result(reason or 'timeout')

        # 6. book-keeping for next step
        self.prev_dist    = curr_dist
        self.prev_actions = actions_arr

        drones_at_goal = sum(self.drone_reached_goal)
        info = {'success': reason == 'success', 'reason': reason,
                'drones_at_goal': drones_at_goal,
                'min_dist_to_goal': float(self.min_dist_ever.mean()),
                'initial_dist_to_goal': float(self.initial_dist.mean())}
        return self._build_obs(raw), rewards_dict, bool(done), info

    def _build_obs(self, raw):
        lf, ns, nm, gs = self.obs_processor.process(raw, self.curriculum_stage)
        return {
            'local_fixed':     lf,
            'neighbor_states': ns,
            'neighbor_mask':   nm,
            'global_state':    gs,
        }

    def set_curriculum_stage(self, stage: int):
        self.curriculum_stage = stage

    def close(self):
        self.adapter.close()
