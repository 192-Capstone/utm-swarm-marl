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

import hashlib

import numpy as np
import torch

from modules.observation_processor import ObservationProcessor
from modules.reward_engine import RewardEngine
from envs.episode_manager import EpisodeManager
from envs.pybullet_adapter import PyBulletAdapter
from envs import scenarios
from envs import scenario_manager
from envs.lidar import lidar_config


class SwarmEnv:
    def __init__(self, config, device=None, gui: bool = False, adapter=None):
        self.config   = config
        self.device   = device or torch.device('cpu')
        self.n_agents = config['environment']['n_agents']
        self.world_size       = config['environment'].get('world_size', 50)
        self.goal_threshold   = config['environment'].get('goal_threshold', 0.5)
        # collision_radius falls back to 0.2 to match RewardEngine's default
        self.collision_radius = config['environment'].get('collision_radius', 0.2)
        # Settling: a drone must be inside goal_threshold AND slower than
        # settle_speed for settle_steps CONSECUTIVE steps before it latches
        # as "reached" — prevents learning to fly through the goal at speed.
        self.settle_speed     = config['environment'].get('settle_speed', 0.15)
        self.settle_steps     = config['environment'].get('settle_steps', 31)
        self.min_spawn_sep    = config.get('scenario_manager', {}).get('min_spawn_separation', 5.0)
        # What a crash does (B5 scale test). 'end' (default): any crash ends the episode, as in
        # every training and evaluation run so far. 'remove': the crashed drones are parked out
        # of the world (adapter.park) and the rest keep flying; the episode ends when every
        # drone has settled or crashed (reason 'collision' if any crashed) or at the timeout.
        self.crash_mode = config['environment'].get('crash_mode', 'end')
        if self.crash_mode not in ('end', 'remove'):
            raise ValueError(f"environment.crash_mode must be 'end' or 'remove', got {self.crash_mode!r}")
        self.crashed = np.zeros(self.n_agents, dtype=bool)
        # Route following (D7, opt-in; environment.route_follow: {enabled, lookahead, min_target}).
        # With start_episode(..., routes=...), every step the policy's goal (the adapter's goal:
        # observation and LiDAR heading) is a carrot `lookahead` m ahead along the drone's route,
        # pulled back to stay in line of sight but never closer than min_target m (envs/
        # route_planner.py RouteFollower.visible_target). Success, settling, rewards and crashes
        # still use the real goal. Off: nothing changes.
        rf = config['environment'].get('route_follow') or {}
        self.route_follow = bool(rf.get('enabled', False))
        self.route_lookahead = float(rf.get('lookahead', 4.5))
        self.route_min_target = float(rf.get('min_target', 1.0))
        self.followers = None
        # Safety cost for constrained RL (MAPPO-Lagrangian), kept separate from
        # the reward: 1 per step a drone is closer than cost_distance to any
        # other drone, plus collision_cost on the step it crashes (a crash ends
        # the episode, so without the extra it would count as one cheap step).
        safety = config.get('safety', {})
        self.cost_distance  = safety.get('cost_distance', 0.5)
        self.collision_cost = safety.get('collision_cost', 0.0)
        # Obstacles count too: 1 per step closer than obstacle_cost_distance
        # (drone centre to obstacle surface), plus collision_cost on a hit.
        self.obstacle_cost_distance = safety.get('obstacle_cost_distance', 0.5)
        self.curriculum_stage = int(config.get('training', {}).get('initial_curriculum_stage', 1))
        if not 1 <= self.curriculum_stage <= 4:
            raise ValueError("training.initial_curriculum_stage must be between 1 and 4")

        # Stage-2 static obstacles (config block 'obstacles'; None = empty world)
        self.obstacle_cfg = scenario_manager.obstacle_config(config)
        self.layout = []

        # Sensor noise (environment.sensor_noise, standard deviations): only what
        # the policy SEES is perturbed. The simulator, collisions, the settle
        # check, rewards and the teacher all keep the true state. Off when the
        # block is missing or all zero, and then no random numbers are drawn,
        # so older configs and seeds behave exactly as before.
        sn = config['environment'].get('sensor_noise') or {}
        self.noise_std = {k: float(sn.get(k, 0.0))
                          for k in ('position', 'velocity', 'orientation', 'lidar')}
        self.noise_on = any(v > 0 for v in self.noise_std.values())
        self._noise_rng = np.random.default_rng(int(sn.get('seed', 0)))
        # Optional (sensor_noise.per_episode_seed, default off): reseed the
        # noise at the start of every episode from that episode's spawns and
        # goals, so the same scenario always gets the same noise, whatever
        # ran before it (paired comparisons stay in step). Off: one generator
        # runs on across episodes, exactly as before.
        self._noise_seed = int(sn.get('seed', 0))
        self.noise_per_episode = bool(sn.get('per_episode_seed', False))
        self.lidar_range = lidar_config(config)['range']

        # environment.simulator: pybullet (default) | webots | webots_l2 | webots_l2_multi
        if adapter is None:
            sim = config['environment'].get('simulator', 'pybullet')
            if sim == 'webots':                  # Level 1: velocity-controlled bodies
                from envs.webots_adapter import WebotsAdapter
                adapter = WebotsAdapter(config, gui=gui)
            elif sim == 'webots_l2':             # Level 2: quadrotor dynamics + velocity PID
                from envs.webots_l2 import WebotsL2Adapter
                adapter = WebotsL2Adapter(config, gui=gui)
            elif sim == 'webots_l2_multi':       # Level 2, n drones: one onboard PID controller each
                from envs.webots_l2_multi import WebotsL2MultiAdapter
                adapter = WebotsL2MultiAdapter(config, gui=gui)
            else:
                adapter = PyBulletAdapter(config, gui=gui)
        self.adapter        = adapter
        self.obs_processor  = ObservationProcessor(config, self.device)
        self.reward_engine  = RewardEngine(config)
        self.episode_manager = EpisodeManager(config)

        self.goals        = None
        self.episode_index = 0      # +1 at every start_episode(); lets wrappers notice a new episode
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

        # Scenario mix: which kind of episode reset() produces, e.g.
        # {random: 0.5, headon: 0.2, crossing: 0.2, swap: 0.1}. Without it,
        # every episode is the plain random task and no extra RNG is drawn,
        # so existing configs and seeds behave exactly as before.
        mix = config.get('scenario_mix') or {'random': 1.0}
        unknown = set(mix) - {'random', 'osm_ecity', *scenarios.GENERATORS, *scenarios.COURSES,
                                 *(f'{k}_open' for k in scenarios.GENERATORS)}
        if unknown:
            raise ValueError(f"scenario_mix has unknown scenarios: {sorted(unknown)}")
        self.scenario_names = list(mix)
        weights = np.array([mix[k] for k in self.scenario_names], dtype=np.float64)
        self.scenario_probs = weights / weights.sum()
        self._use_mix = self.scenario_names != ['random']
        self.current_scenario = 'random'
        self._osm_layout_cache = None
        self._osm_layout_cache_path = None

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
        scenario = 'random'
        if self._use_mix:
            scenario = str(np.random.choice(self.scenario_names, p=self.scenario_probs))
        self.current_scenario = scenario
        if scenario == 'random':
            spawns, goals = self._sample_positions()
        elif scenario == 'osm_ecity':
            from pathlib import Path
            from envs.osm_scenario import DEFAULT_LAYOUT_PATH, load_osm_layout, sample_flights
            osm_cfg = self.config.get('osm_scenario', {})
            raw_path = Path(osm_cfg.get('layout_path', str(DEFAULT_LAYOUT_PATH)))
            layout_path = raw_path if raw_path.is_absolute() else Path(__file__).resolve().parents[1] / raw_path
            if self._osm_layout_cache is None or self._osm_layout_cache_path != layout_path:
                self._osm_layout_cache = load_osm_layout(layout_path)
                self._osm_layout_cache_path = layout_path
            spawns, goals = sample_flights(
                self._osm_layout_cache, self.n_agents, rng=np.random,
                min_goal_distance_m=float(osm_cfg.get('min_goal_distance_m', 15.0)),
                max_goal_distance_m=float(osm_cfg.get('max_goal_distance_m', 45.0)),
                obstacle_clearance_m=float(osm_cfg.get('obstacle_clearance_m', 2.0)),
                min_point_separation_m=float(osm_cfg.get('min_point_separation_m', self.min_spawn_sep)),
                boundary_margin_m=float(osm_cfg.get('boundary_margin_m', 2.0)),
                min_altitude_m=float(osm_cfg.get('min_altitude_m', 2.0)),
                max_altitude_m=float(osm_cfg.get('max_altitude_m', 4.0)),
                max_sampling_attempts=int(osm_cfg.get('max_sampling_attempts', 1000)),
            )
            return self.start_episode(spawns, goals, obstacles=self._osm_layout_cache)
        elif scenario in scenarios.COURSES:          # comes with its own obstacle layout
            spawns, goals, layout = scenarios.COURSES[scenario](np.random, self.n_agents)
            return self.start_episode(spawns, goals, obstacles=layout)
        elif scenario.endswith('_open'):             # e.g. headon_open: no obstacles, pure crowd
            spawns, goals = scenarios.GENERATORS[scenario[:-5]](np.random, self.n_agents)
            return self.start_episode(spawns, goals, obstacles=[])
        else:
            spawns, goals = scenarios.GENERATORS[scenario](np.random, self.n_agents)
        return self.start_episode(spawns, goals)

    def start_episode(self, spawns, goals, rng=None, obstacles='auto', routes=None):
        """Begin an episode with the given spawns and goals ({'drone_i': [x, y, z]}).
        reset() uses this, and so do the evaluation and analysis scripts that
        build their own scenarios, so nothing (obstacles included) is skipped.

        rng       : random generator for the obstacle layout (default np.random)
        obstacles : 'auto' = generate a layout if obstacles are enabled,
                    or an explicit layout list ([] for an empty world)
        routes    : {'drone_i': [(x, y), ...]} simplified D7 routes from spawn to goal
                    (needs environment.route_follow.enabled); drones without a route fly
                    straight at their goal
        """
        if routes is not None and not self.route_follow:
            raise ValueError("start_episode(routes=...) needs environment.route_follow.enabled")
        self.episode_index += 1
        rng = np.random if rng is None else rng
        if obstacles == 'auto':
            obstacles = (scenario_manager.generate_layout(rng, spawns, goals, self.obstacle_cfg)
                         if self.obstacle_cfg else [])
        self.layout = list(obstacles)
        self.adapter.reset_world(spawns, goals, self.layout)
        self.goals = np.array([goals[f'drone_{i}'] for i in range(self.n_agents)],
                              dtype=np.float32)
        if self.noise_per_episode:
            self._noise_rng = np.random.default_rng(self._episode_noise_seed(spawns))
        self.episode_manager.reset()
        self.followers = self._make_followers(spawns, goals, routes) if routes else None
        self._update_carrots()

        raw       = self.adapter.get_raw_state()
        positions = self.adapter.get_positions()
        self.prev_dist    = np.linalg.norm(positions - self.goals, axis=1)
        self.prev_actions = np.zeros((self.n_agents, 4), dtype=np.float32)
        self.drone_reached_goal = [False] * self.n_agents
        self.settle_counter = np.zeros(self.n_agents, dtype=np.int64)
        self.best_settle_counter = np.zeros(self.n_agents, dtype=np.int64)
        self.ever_entered_goal = [False] * self.n_agents
        self.crashed = np.zeros(self.n_agents, dtype=bool)

        # Track closest approach to goal — diagnoses "almost succeeded" vs
        # "never made progress" for episodes that don't reach the goal
        self.initial_dist   = self.prev_dist.copy()
        self.min_dist_ever  = self.prev_dist.copy()

        self.obs_processor.reset_history()
        return self._build_obs(raw)

    def _make_followers(self, spawns, goals, routes):
        """A RouteFollower per routed drone (3D waypoints: height interpolated from spawn to goal
        along the route) and the line-of-sight grid for this layout."""
        from envs.route_planner import RouteFollower, build_grid
        followers = {}
        pts = []
        for key, route in routes.items():
            xy = np.asarray([p[:2] for p in route], dtype=float)
            z0, z1 = float(spawns[key][2]), float(goals[key][2])
            cum = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))])
            frac = cum / cum[-1] if cum[-1] > 0 else np.zeros_like(cum)
            followers[key] = RouteFollower(np.column_stack([xy, z0 + frac * (z1 - z0)]),
                                           lookahead=self.route_lookahead)
            pts.append(xy)
        pts = np.vstack(pts)
        for ob in self.layout:
            pts = np.vstack([pts, np.asarray(ob['center'], dtype=float)[None, :2]])
        lo, hi = pts.min(axis=0) - 10.0, pts.max(axis=0) + 10.0
        # line of sight = not hidden behind a building: walls grown by the drone's own radius only
        # (the planner's extra 0.35 m margin is for choosing routes, not for seeing)
        self._los_grid = build_grid(self.layout, ((lo[0], hi[0]), (lo[1], hi[1])), 0.25,
                                    scenario_manager.DRONE_RADIUS)
        return followers

    def _update_carrots(self):
        """Point each routed drone's adapter goal (observation + LiDAR heading) at its carrot."""
        if not self.followers:
            return
        positions = self.adapter.get_positions()
        for key, f in self.followers.items():
            i = int(key.split('_')[1])
            carrot = f.visible_target(positions[i], self._los_grid, min_dist=self.route_min_target)
            self.adapter.goals[key] = np.asarray(carrot, dtype=np.float32)

    def step(self, actions_dict):
        # 1. apply actions + advance the simulator
        self.adapter.apply_actions(actions_dict)
        self.adapter.step_simulation()
        if self.crashed.any():                      # crash_mode remove: keep them parked
            self.adapter.park(np.flatnonzero(self.crashed))
        self.episode_manager.step()
        self._update_carrots()                    # route following (D7): new carrots first

        # 2. read the new world state
        raw       = self.adapter.get_raw_state()
        positions = self.adapter.get_positions()
        curr_dist = np.linalg.norm(positions - self.goals, axis=1)
        self.min_dist_ever = np.minimum(self.min_dist_ever, curr_dist)

        # 3. pairwise drone distances (feed proximity + drone-collision reward)
        # (one batched product instead of a Python loop per pair; for the float32
        # positions this matches the old per-pair np.linalg.norm bit for bit,
        # which an axis-wise norm or sum does not)
        diff = positions[:, None, :] - positions[None, :, :]
        rows = np.sqrt((diff[..., None, :] @ diff[..., :, None])[..., 0, 0]).tolist()
        drone_distances = [row[:i] + row[i + 1:] for i, row in enumerate(rows)]
        # drone-drone collision flags (used for termination)
        drone_collision = [bool(d and min(d) < self.collision_radius)
                           for d in drone_distances]
        # obstacle collisions: geometric check against this episode's layout
        # (empty in Stage 1), so it works with contact physics off
        obstacle_gap = [scenario_manager.distance_to_obstacles(self.layout, positions[i])
                        for i in range(self.n_agents)]
        obstacle_collision = [g < scenario_manager.DRONE_RADIUS for g in obstacle_gap]
        if self.crash_mode == 'remove':
            new = (np.array(drone_collision) | np.array(obstacle_collision)) & ~self.crashed
            if new.any():
                # park them now, so no other drone sees or hits them from the next observation on
                self.crashed |= new
                self.adapter.park(np.flatnonzero(new))
                raw = self.adapter.get_raw_state()
        cost = [float(bool(d) and min(d) < self.cost_distance)
                + self.collision_cost * float(drone_collision[i])
                + float(obstacle_gap[i] < self.obstacle_cost_distance)
                + self.collision_cost * float(obstacle_collision[i])
                for i, d in enumerate(drone_distances)]

        actions_arr = np.array([actions_dict[f'drone_{i}'] for i in range(self.n_agents)],
                               dtype=np.float32)

        # 4. Settling: a drone must be inside goal_threshold AND slower than
        # settle_speed for settle_steps CONSECUTIVE steps before it latches
        # as "reached". Leaving the radius or speeding back up resets its
        # counter — this is what forces the policy to actually stop instead
        # of flying through the goal region once and moving on.
        speed = np.array([np.linalg.norm(raw[f'drone_{i}']['velocity'])
                          for i in range(self.n_agents)], dtype=np.float32)
        settled_now = (curr_dist < self.goal_threshold) & (speed < self.settle_speed)
        self.settle_counter = np.where(settled_now, self.settle_counter + 1, 0)

        goal_just_reached = [False] * self.n_agents
        for i in range(self.n_agents):
            if not self.drone_reached_goal[i] and self.settle_counter[i] >= self.settle_steps:
                goal_just_reached[i] = True
                self.drone_reached_goal[i] = True

        # Detect first-time goal entry
        goal_entry_just = [False] * self.n_agents
        for i in range(self.n_agents):
            if not self.ever_entered_goal[i] and curr_dist[i] < self.goal_threshold:
                goal_entry_just[i] = True
                self.ever_entered_goal[i] = True

        # Settle-counter progress uses a highwater mark (never decreases within
        # an episode) so the reward is bounded at settle_counter_scale ×
        # settle_steps per drone. Without this, a drone that repeatedly climbs
        # partway up the counter and resets (e.g. oscillating at the goal
        # boundary) farms reward every climb — the same class of exploit as
        # the earlier per-step settle_shaping bug.
        capped_counter = np.minimum(self.settle_counter, self.settle_steps)
        new_best = np.maximum(self.best_settle_counter, capped_counter)
        settle_counter_delta = new_best - self.best_settle_counter
        self.best_settle_counter = new_best

        reward_data = {
            'prev_dist':             self.prev_dist,
            'curr_dist':             curr_dist,
            'collision':             obstacle_collision,
            'drone_distances':       drone_distances,
            'action':                actions_arr,
            'prev_action':           self.prev_actions,
            'speed':                 speed,
            'goal_just_reached':     goal_just_reached,
            'drone_reached_goal':    list(self.drone_reached_goal),
            'goal_entry_just':       goal_entry_just,
            'settle_counter_delta':  settle_counter_delta,
            'obstacle_clearance':    [g - scenario_manager.DRONE_RADIUS for g in obstacle_gap],
        }
        rewards = self.reward_engine.compute_rewards(reward_data)
        rewards_dict = {f'drone_{i}': float(rewards['total'][i])
                        for i in range(self.n_agents)}

        # 5. termination (collision / all-goals-reached / timeout)
        # Use latched drone_reached_goal flags: a drone counts once it has
        # SETTLED (inside goal_threshold, under settle_speed, held for
        # settle_steps — see step 4 above), not on first contact. This is
        # what keeps a drone from "succeeding" by flying through the goal
        # at speed and drifting off afterward.
        removing = self.crash_mode == 'remove'
        done, reason = self.episode_manager.check_done(
            positions.tolist(), self.goals.tolist(),
            [False] * self.n_agents if removing
            else [a or b for a, b in zip(drone_collision, obstacle_collision)])
        # EpisodeManager's own "success" path checks instantaneous position
        # only (no speed/settling) — it would let a swarm that happens to be
        # simultaneously inside every goal radius at full speed count as a
        # success, bypassing the settle logic above entirely. Only accept
        # collision/timeout from it; success is decided by the settle latch.
        if done and reason == 'success':
            done, reason = False, None
        if removing:
            if not done and all(r or c for r, c in zip(self.drone_reached_goal, self.crashed)):
                done, reason = True, 'collision' if self.crashed.any() else 'success'
            elif done and self.crashed.any():
                reason = 'collision'            # timed out after a crash: still a crash episode
        elif not done and all(self.drone_reached_goal):
            done, reason = True, 'success'
        if done:
            self.episode_manager.record_episode_result(reason or 'timeout')

        # 6. book-keeping for next step
        self.prev_dist    = curr_dist
        self.prev_actions = actions_arr

        drones_at_goal = sum(self.drone_reached_goal)
        drones_inside = int(np.sum(curr_dist < self.goal_threshold))

        # Minimum inter-drone separation this step — None (not inf/0) for a
        # single agent, where the concept doesn't apply. drone_distances[i]
        # is empty when n_agents==1, so min() over all pairs would error;
        # guard explicitly rather than let evaluators improvise a sentinel.
        if self.n_agents >= 2:
            all_pairwise = [d for dd in drone_distances for d in dd]
            min_inter_drone_distance = float(min(all_pairwise)) if all_pairwise else None
        else:
            min_inter_drone_distance = None

        info = {'success': reason == 'success', 'reason': reason,
                'scenario': self.current_scenario,
                'drones_at_goal': drones_at_goal,
                # Per-agent LATCHED success (settled for settle_steps), not
                # merely "inside goal_threshold on this step" — this is what
                # per-agent success rate must be computed from.
                'drone_reached_goal':       list(self.drone_reached_goal),
                'drone_collision':          list(drone_collision),
                'obstacle_collision':       list(obstacle_collision),
                'drone_crashed':            self.crashed.tolist(),   # crash_mode remove only
                'obstacle_distance':        [float(g) for g in obstacle_gap],
                'cost':                     cost,
                'min_dist_to_goal':         float(self.min_dist_ever.mean()),
                'per_agent_min_dist':       self.min_dist_ever.tolist(),
                'initial_dist_to_goal':     float(self.initial_dist.mean()),
                'drones_inside_goal':       drones_inside,
                'mean_speed':               float(speed.mean()),
                'per_agent_speed':          speed.tolist(),
                'max_settle_counter':       int(self.settle_counter.max()),
                'per_agent_settle_counter': self.settle_counter.tolist(),
                # Clamped to [0, 1] per agent — settle_counter keeps
                # incrementing past settle_steps for an already-latched
                # agent waiting on its teammates (the counter update above
                # doesn't freeze once a drone reaches drone_reached_goal),
                # so an unclamped fraction could exceed 1.0 (observed up to
                # 1.41 in spring-vortex-58's 3-agent eval logs). Collective
                # metric uses MIN across agents, not max: the team's
                # progress toward full settlement is bottlenecked by its
                # least-progressed agent, not led by its most-progressed one.
                'settle_fraction':          float(np.min(np.minimum(
                    self.settle_counter / self.settle_steps, 1.0))),
                'per_agent_settle_fraction': np.minimum(
                    self.settle_counter / self.settle_steps, 1.0).tolist(),
                'min_inter_drone_distance': min_inter_drone_distance,
                'reward_components': {
                    k: float(v.sum()) for k, v in rewards.items() if k != 'total'
                }}
        return self._build_obs(raw), rewards_dict, bool(done), info

    def _episode_noise_seed(self, spawns):
        """Seed for one episode's sensor noise: the base seed plus a hash of
        the spawns and goals (rounded to mm, like eval_suite's scenario_id)."""
        sp = np.round(np.array([spawns[f'drone_{i}'] for i in range(self.n_agents)],
                               dtype=np.float64), 3)
        gl = np.round(self.goals.astype(np.float64), 3)
        digest = hashlib.sha1(np.concatenate([sp.ravel(), gl.ravel()]).tobytes()).digest()
        return [self._noise_seed, int.from_bytes(digest[:8], 'little')]

    def _noisy(self, raw):
        """Copy of raw with sensor noise added. Positions double as the
        neighbour broadcasts, so those get the same noise. LiDAR noise only
        touches rays that hit something: a ray that sees nothing stays at
        full range (a real LiDAR doesn't invent returns)."""
        rng, s = self._noise_rng, self.noise_std
        out = {'goals': raw['goals']}
        for key, d in raw.items():
            if key == 'goals':
                continue
            d = dict(d)
            for name, std in (('position', s['position']), ('velocity', s['velocity']),
                              ('orientation', s['orientation'])):
                if std > 0:
                    d[name] = (np.asarray(d[name], dtype=np.float64)
                               + rng.normal(0.0, std, 3)).tolist()
            if s['lidar'] > 0:
                for name in ('lidar', 'lidar_ring'):
                    if name in d:
                        r = np.asarray(d[name], dtype=np.float64)
                        hit = r < self.lidar_range - 1e-6
                        r = np.where(hit, np.clip(r + rng.normal(0.0, s['lidar'], r.shape),
                                                  0.0, self.lidar_range), r)
                        d[name] = r.tolist()
            out[key] = d
        return out

    def _build_obs(self, raw):
        if self.noise_on:
            raw = self._noisy(raw)
        extra = None
        if self.obs_processor.critic_version >= 2:
            # critic.version 2: the critic also sees each drone's settle state
            extra = {'settle_progress': np.minimum(self.settle_counter / self.settle_steps, 1.0),
                     'reached': np.asarray(self.drone_reached_goal, dtype=np.float32)}
        lf, ns, nm, gs = self.obs_processor.process(raw, self.curriculum_stage,
                                                    critic_extra=extra)
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
