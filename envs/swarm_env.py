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
from envs import disturbances


# E1 flight profile for routes given with heights (SwarmEnv._update_carrots): the carrot stays
# above the pad until the drone is within CLIMB_DONE_M of the cruise height (more than goal-hold's
# 0.5 m zone, so the hold doesn't engage on the climb), and becomes the goal once the drone is
# within LAND_RADIUS_M of it horizontally, so climb and landing stay over the pads, as E1 books
# them. 1.2 m: the policy can stop ~1 m short of the carrot above the goal, and at 0.6 m such a
# drone hovered beside the goal until it drifted close enough to descend (E1 holds the goal pad
# within its 2.1 m radius, so a landing that starts 1.2 m off-centre stays inside).
CLIMB_DONE_M = 0.6
LAND_RADIUS_M = 1.2
ROOF_CLEAR_M = 1.0         # line of sight: a roof this far below a drone's cruise height hides nothing from it


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
        # Scheduled departures (E1, opt-in; environment.schedule: {enabled}). With
        # start_episode(..., departures={'drone_i': t_dep_s}) a drone is parked out of the world
        # (adapter.park) until its departure time, then placed at its spawn; once it has settled
        # at its goal it is parked again (landed). So a drone is only in the air during its own
        # flight, as E1 books it. Use with crash_mode remove. Off: nothing changes.
        self.schedule = bool((config['environment'].get('schedule') or {}).get('enabled', False))
        # E1 landing options (C5; both off by default, None). Landing starts when a drone with an
        # E1 layer profile first comes within LAND_RADIUS_M of its goal (horizontally).
        #   land_latch_m: from then on its carrot stays the goal until it has landed, unless it
        #                 drifts more than land_latch_m from the goal (then: route carrot again,
        #                 and landing starts afresh at LAND_RADIUS_M). Off: the carrot is the goal
        #                 only while the drone is within LAND_RADIUS_M (the carrot jumps back up to
        #                 cruise height when it drifts just outside).
        #   land_vz_max:  while landing (latched as above, or within LAND_RADIUS_M without a
        #                 latch), its commanded vz is clipped to +-land_vz_max m/s; the clipped
        #                 action is the one applied and recorded.
        #   land_wait_clear_m: a drone does not descend onto its pad while another drone is low
        #                 over it (closer than this horizontally to the pad and more than 0.5 m
        #                 below cruise height and below it: still landing there, or taking off; of two
        #                 drones dipping to one pad the lower lands first): it holds at
        #                 cruise height above the pad until it is clear (E6; found in Webots,
        #                 where landings take ~2 s longer than in PyBullet and a drone came down
        #                 on one still settling).
        sch = config['environment'].get('schedule') or {}
        self.land_latch_m = None if sch.get('land_latch_m') is None else float(sch['land_latch_m'])
        self.land_vz_max = None if sch.get('land_vz_max') is None else float(sch['land_vz_max'])
        # land_vz_high (m/s, off by default): while landing more than land_slow_m over the pad, vz is
        # clipped to this instead of land_vz_max: only the last metres are slow (the gentle touchdown
        # C5 needs), so a landing from a high layer does not crawl down the whole way
        self.land_vz_high = None if sch.get('land_vz_high') is None else float(sch['land_vz_high'])
        self.land_slow_m = 3.0 if sch.get('land_slow_m') is None else float(sch['land_slow_m'])
        # los_below_cruise (off by default): a drone's line of sight ignores roofs more than
        # ROOF_CLEAR_M below its own cruise height (one flying over a roof is not hidden behind it).
        # Off, every building blocks it, as before (E1-flown runs unchanged).
        self.los_below_cruise = bool(sch.get('los_below_cruise', False))
        self.land_wait_clear_m = None if sch.get('land_wait_clear_m') is None else float(sch['land_wait_clear_m'])
        # cruise_z_bias: the carrot's height during the climb and the cruise is raised by this much. The
        # policy settles ~0.8 m below a height target at every height tested (2-18 m), so aiming that
        # much higher makes it fly at the height E1 booked; without it a drone whose sag exceeds
        # CLIMB_DONE_M never counts as climbed and hovers over its pad (seen at 14 m and 18 m).
        self.cruise_z_bias = float(sch.get('cruise_z_bias') or 0.0)
        # land_step_m: while landing, the carrot sits at most this far below the drone (over the pad),
        # sliding down with it, instead of on the pad itself. From a high cruise the pad is far straight
        # below, which the policy never saw in training (landings from 3-4 m): it drifted sideways out of
        # the landing radius and climbed back, over and over (14 m and 18 m in scripts/altitude_test.py).
        self.land_step_m = None if sch.get('land_step_m') is None else float(sch['land_step_m'])
        if self.land_latch_m is not None and self.land_latch_m < LAND_RADIUS_M:
            raise ValueError(f"schedule.land_latch_m ({self.land_latch_m}) must be >= LAND_RADIUS_M ({LAND_RADIUS_M})")
        if self.land_vz_max is not None and self.land_vz_max < 0:
            raise ValueError("schedule.land_vz_max must be >= 0")
        self._landing = np.zeros(self.n_agents, dtype=bool)
        self.control_dt = float(config['environment'].get('control_timestep', 0.032))
        self.waiting = np.zeros(self.n_agents, dtype=bool)
        self.landed = np.zeros(self.n_agents, dtype=bool)
        self.cancelled = np.zeros(self.n_agents, dtype=bool)     # E4: flight called off on its pad
        self.departures = None
        self._t_step = 0
        # D5 drone failures (envs/disturbances.py; off by default: then nothing below runs, no
        # random numbers are drawn and info has no new keys).
        #   environment.failures start_episode(failures=[{'drone', 't', 'type'}]): at time t the
        #                        drone, if it is flying, has a
        #       motor   failure: it ignores its commands and falls; on the ground or a building it
        #               is crashed (crashed + failed 'motor')
        #       land    forced landing where it is: its carrot goes straight down to land_z above
        #               the ground or the roof under it, |vz| <= land_vz, the policy flies it there;
        #               settled there it has landed (landed + forced_down, not arrived)
        #       comms   comms loss: it flies on as before, but set_route no longer reaches it (the
        #               eval stops its reports: utm/telemetry.py TelemetryModel.mute)
        #   Needs crash_mode remove. A failure whose drone is not flying at t (still on its pad,
        #   landed or crashed) is skipped (failures_skipped). set_route refuses failed drones.
        self.failure_cfg = disturbances.failure_config(config)
        if self.failure_cfg is not None and self.crash_mode != 'remove':
            raise ValueError("environment.failures needs environment.crash_mode: remove")
        self._reset_failures()
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
        if self.failure_cfg is not None and not hasattr(adapter, 'set_falling'):
            raise ValueError(f"environment.failures is not supported by {type(adapter).__name__}")
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

    def start_episode(self, spawns, goals, rng=None, obstacles='auto', routes=None, departures=None,
                      failures=None):
        """Begin an episode with the given spawns and goals ({'drone_i': [x, y, z]}).
        reset() uses this, and so do the evaluation and analysis scripts that
        build their own scenarios, so nothing (obstacles included) is skipped.

        rng       : random generator for the obstacle layout (default np.random)
        obstacles : 'auto' = generate a layout if obstacles are enabled,
                    or an explicit layout list ([] for an empty world)
        routes    : {'drone_i': [(x, y), ...] or [(x, y, z), ...]} simplified D7 routes from
                    spawn to goal (needs environment.route_follow.enabled); with (x, y) the
                    height goes from spawn to goal along the route, with (x, y, z) it is given
                    (E1: climb to the layer, cruise, descend); drones without a route fly
                    straight at their goal
        departures: {'drone_i': t_dep_s} take-off times (needs environment.schedule.enabled);
                    missing drones take off at t = 0
        failures  : [{'drone': 'drone_i', 't': s, 'type': 'motor' | 'land' | 'comms'}] (needs
                    environment.failures.enabled; envs/disturbances.py draw_failures)
        """
        if failures is not None and self.failure_cfg is None:
            raise ValueError("start_episode(failures=...) needs environment.failures.enabled")
        if routes is not None and not self.route_follow:
            raise ValueError("start_episode(routes=...) needs environment.route_follow.enabled")
        if departures is not None and not self.schedule:
            raise ValueError("start_episode(departures=...) needs environment.schedule.enabled")
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
        self.crashed = np.zeros(self.n_agents, dtype=bool)
        self.landed = np.zeros(self.n_agents, dtype=bool)
        self.cancelled = np.zeros(self.n_agents, dtype=bool)
        self._t_step = 0
        self._spawns = np.array([spawns[f'drone_{i}'] for i in range(self.n_agents)], dtype=np.float32)
        if departures:
            self.departures = np.array([float(departures.get(f'drone_{i}', 0.0))
                                        for i in range(self.n_agents)])
            self.waiting = self.departures > 0.0
            if self.waiting.any():
                self.adapter.park(np.flatnonzero(self.waiting))
        else:
            self.departures = None
            self.waiting = np.zeros(self.n_agents, dtype=bool)
        self._landing = np.zeros(self.n_agents, dtype=bool)
        self._reset_failures(failures)
        self.followers = self._make_followers(spawns, goals, routes) if routes else None
        self._update_carrots()

        raw       = self.adapter.get_raw_state()
        positions = self.adapter.get_positions()
        if self.waiting.any():               # distances of waiting drones from their pads
            positions = np.where(self.waiting[:, None], self._spawns, positions)
        self.prev_dist    = np.linalg.norm(positions - self.goals, axis=1)
        self.prev_actions = np.zeros((self.n_agents, 4), dtype=np.float32)
        self.drone_reached_goal = [False] * self.n_agents
        self.settle_counter = np.zeros(self.n_agents, dtype=np.int64)
        self.best_settle_counter = np.zeros(self.n_agents, dtype=np.int64)
        self.ever_entered_goal = [False] * self.n_agents

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
        self._cruise = {}                         # key -> cruise height (routes given with heights)
        self._climbed = {}
        for key, route in routes.items():
            xy = np.asarray([p[:2] for p in route], dtype=float)
            if len(route[0]) >= 3:                # E1 layer profile: climb, cruise, land
                # cruise height = the middle waypoints' (a re-planned route may start higher, E4)
                self._cruise[key] = float(route[1][2] if len(route) > 2 else max(p[2] for p in route))
                self._climbed[key] = False
                z = np.full(len(xy), self._cruise[key])
            else:
                z0, z1 = float(spawns[key][2]), float(goals[key][2])
                cum = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))])
                frac = cum / cum[-1] if cum[-1] > 0 else np.zeros_like(cum)
                z = z0 + frac * (z1 - z0)
            followers[key] = RouteFollower(np.column_stack([xy, z]), lookahead=self.route_lookahead)
            pts.append(xy)
        pts = np.vstack(pts)
        for ob in self.layout:
            pts = np.vstack([pts, np.asarray(ob['center'], dtype=float)[None, :2]])
        lo, hi = pts.min(axis=0) - 10.0, pts.max(axis=0) + 10.0
        # line of sight = not hidden behind a building: walls grown by the drone's own radius only
        # (the planner's extra 0.35 m margin is for choosing routes, not for seeing). Per drone, by
        # its cruise height: a building over 1 m below it hides nothing (a drone held behind a roof
        # it flies over waits for ever), so high and low flights each see their own buildings
        # (schedule.los_below_cruise; off, every drone sees every building, as before)
        self._los_bounds, self._los_cache = ((lo[0], hi[0]), (lo[1], hi[1])), {}
        self._los_grid = self._los(None)
        self._los_for = {key: self._los(self._cruise.get(key)) for key in followers}
        return followers

    def _los(self, zc):
        """Line-of-sight grid for a drone cruising at zc (None: every building), cached by the
        set of buildings it keeps."""
        from envs.route_planner import build_grid
        low = zc is not None and self.los_below_cruise
        seen = tuple(k for k, ob in enumerate(self.layout) if not low or ob['height'] > zc - ROOF_CLEAR_M)
        if seen not in self._los_cache:
            self._los_cache[seen] = build_grid([self.layout[k] for k in seen], self._los_bounds, 0.25,
                                               scenario_manager.DRONE_RADIUS)
        return self._los_cache[seen]

    def set_route(self, i, route, departure=None):
        """Give drone i a new route (3D waypoints, E1 profile) mid-episode (E4 re-planning): a
        flying drone starts following it from its first waypoint (its current position); a drone
        still waiting on its pad also gets the new departure time. The route's end becomes the
        drone's goal (a re-plan may send it back to its start pad).
        Returns False (and changes nothing) for a drone with a failure (D5): it can't follow a new
        route (motor, forced landing) or doesn't hear it (comms loss); True otherwise."""
        from envs.route_planner import RouteFollower
        if self.failed[i]:
            return False
        key = f'drone_{i}'
        xy = np.asarray([p[:2] for p in route], dtype=float)
        self._cruise[key] = float(route[1][2] if len(route) > 2 else max(p[2] for p in route))
        self._climbed[key] = False
        self._landing[i] = False                  # C5: a re-routed drone is not latched onto its old goal
        self.followers[key] = RouteFollower(np.column_stack([xy, np.full(len(xy), self._cruise[key])]),
                                            lookahead=self.route_lookahead)
        if getattr(self, '_los_cache', None) is not None:      # a re-plan may change its layer
            self._los_for[key] = self._los(self._cruise[key])
        self.goals[i, :2] = xy[-1]
        if departure is not None and self.departures is not None and self.waiting[i]:
            self.departures[i] = float(departure)
        return True

    def cancel(self, i):
        """Call off drone i's flight while it is still waiting on its pad (E4): it never takes off,
        and the episode doesn't wait for it (it does not count as arrived)."""
        if self.departures is None or not self.waiting[i]:
            raise ValueError(f'drone_{i} is not waiting on its pad')
        self.departures[i] = np.inf
        self.cancelled[i] = True

    def refly(self, i, spawn, route, departure):
        """Give drone slot i a new flight (the live UTM, scripts/utm_server.py: a fixed set of drones
        serves the bookings as they come): it waits parked until `departure` (s, episode time), is then
        placed on `spawn` (its pad) and flies `route` (3D waypoints, E1 profile) to its end. The slot
        must not be flying or crashed (a fresh one waiting with departure inf, a landed or a cancelled
        one); its arrival and settling are reset. Needs schedule and route_follow."""
        if self.departures is None or self.followers is None:
            raise ValueError('refly needs environment.schedule and a routed episode')
        if self.crashed[i] or not (self.waiting[i] or self.landed[i] or self.cancelled[i]):
            raise ValueError(f'drone_{i} is flying or crashed')
        self._spawns[i] = np.asarray(spawn, dtype=np.float32)
        self.waiting[i], self.landed[i], self.cancelled[i] = True, False, False
        self.drone_reached_goal[i] = self.ever_entered_goal[i] = False
        self.settle_counter[i] = self.best_settle_counter[i] = 0
        self.set_route(i, route, departure)
        self.goals[i, 2] = float(route[-1][2])

    def _update_carrots(self):
        """Point each routed drone's adapter goal (observation + LiDAR heading) at its carrot."""
        if not self.followers:
            return
        positions = self.adapter.get_positions()
        for key, f in self.followers.items():
            i = int(key.split('_')[1])
            if self.waiting[i] or self.landed[i] or self.crashed[i]:
                self._landing[i] = False          # (all False unless a landing option is on)
                continue                          # parked: not flying
            carrot = f.visible_target(positions[i], self._los_for.get(key, self._los_grid), min_dist=self.route_min_target)
            if key in self._cruise:               # E1 profile: vertical climb and landing at the pads
                zc, goal = self._cruise[key], self.goals[i]
                if not self._climbed[key] and positions[i][2] >= zc - CLIMB_DONE_M:
                    self._climbed[key] = True
                if not self._climbed[key]:
                    carrot = np.array([f.w[0, 0], f.w[0, 1], zc + self.cruise_z_bias])
                elif self._pad_busy(i, goal, zc, positions):   # wait above until the pad is clear
                    self._landing[i] = False
                    carrot = np.array([goal[0], goal[1], zc + self.cruise_z_bias])
                elif self.land_latch_m is None and self.land_vz_max is None:   # default (unchanged)
                    if np.hypot(*(positions[i][:2] - goal[:2])) < LAND_RADIUS_M:
                        carrot = goal
                else:                             # landing options: track the landing state
                    d = np.hypot(*(positions[i][:2] - goal[:2]))
                    if d < LAND_RADIUS_M:
                        self._landing[i] = True
                    elif self.land_latch_m is None or d > self.land_latch_m:
                        self._landing[i] = False
                    if self._landing[i]:
                        carrot = goal
                if self.land_step_m is not None and carrot is goal and positions[i][2] - goal[2] > self.land_step_m:
                    carrot = np.array([goal[0], goal[1], positions[i][2] - self.land_step_m])   # step down to the pad
                if self.cruise_z_bias and self._climbed[key] and abs(float(carrot[2]) - zc) < 1e-6:
                    carrot = np.array([carrot[0], carrot[1], zc + self.cruise_z_bias])   # cruising: aim higher
            self.adapter.goals[key] = np.asarray(carrot, dtype=np.float32)

    def _pad_busy(self, i, goal, zc, positions):
        """land_wait_clear_m: another flying drone is low over drone i's pad (landing or taking off)."""
        if self.land_wait_clear_m is None or np.hypot(*(positions[i][:2] - goal[:2])) >= 2.0 * LAND_RADIUS_M:
            return False                          # off, or not near its pad yet
        busy = ~(self.waiting | self.landed | self.crashed)
        busy[i] = False
        near = np.hypot(positions[:, 0] - goal[0], positions[:, 1] - goal[1]) < self.land_wait_clear_m
        low = (positions[:, 2] < zc - 0.5) & (positions[:, 2] < positions[i][2])   # lower one lands first
        return bool(np.any(busy & near & low))

    def step(self, actions_dict):
        if self.land_vz_max is not None and self._landing.any():   # E1 landing: limit vz
            actions_dict = dict(actions_dict)
            z = self.adapter.get_positions()[:, 2] if self.land_vz_high is not None else None
            for i in np.flatnonzero(self._landing):
                a = np.array(actions_dict[f'drone_{i}'], dtype=np.float32)
                cap = (self.land_vz_high if z is not None and z[i] - self.goals[i][2] > self.land_slow_m
                       else self.land_vz_max)
                a[2] = np.clip(a[2], -cap, cap)
                actions_dict[f'drone_{i}'] = a
        # 1. apply actions + advance the simulator
        if self._forced.any():                    # D5 forced landing: limit vz
            actions_dict = self._forced_vz_limit(actions_dict)
        self.adapter.apply_actions(actions_dict)
        self.adapter.step_simulation()
        self._t_step += 1
        if self.departures is not None:            # schedule: take off when it's time
            go = self.waiting & (self.departures <= self._t_step * self.control_dt + 1e-9)
            for i in np.flatnonzero(go):
                self.adapter.place(i, self._spawns[i])
            self.waiting &= ~go
        if self._fail_queue:
            self._trigger_failures()
        parked = self.crashed | self.waiting | self.landed
        if parked.any():                            # crashed / not departed / landed: off the map
            self.adapter.park(np.flatnonzero(parked))
        self.episode_manager.step()
        self._update_carrots()                    # route following (D7): new carrots first
        if self._forced.any():
            self._failure_carrots()

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
            if self._falling.any():                 # motor failure: down on the ground
                new |= self._falling & (positions[:, 2] <= scenario_manager.DRONE_RADIUS + 0.05) & ~self.crashed
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
        if self._forced.any():
            self._forced_landing_done(positions, speed)

        goal_just_reached = [False] * self.n_agents
        for i in range(self.n_agents):
            if not self.drone_reached_goal[i] and self.settle_counter[i] >= self.settle_steps:
                goal_just_reached[i] = True
                self.drone_reached_goal[i] = True
        if self.departures is not None:            # schedule: a drone that has settled lands
            self.landed |= np.asarray(self.drone_reached_goal, dtype=bool)

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
            if not done and all(r or c or x or f for r, c, x, f in zip(self.drone_reached_goal, self.crashed,
                                                                      self.cancelled, self.forced_down)):
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
        if self.failure_cfg is not None:
            info['drone_failed'] = list(self.failed)
            info['drone_forced_down'] = self.forced_down.tolist()
            info['drone_comms_lost'] = self.comms_lost.tolist()
        return self._build_obs(raw), rewards_dict, bool(done), info

    # ---- D5 drone failures (envs/disturbances.py) ---------------------------
    def _reset_failures(self, failures=None):
        """Failure state for a new episode (all clear when failures are off)."""
        n = self.n_agents
        self.failed = [''] * n                    # failure type per drone ('' = none)
        self.failure_t = np.full(n, np.nan)       # when it happened (s)
        self.comms_lost = np.zeros(n, dtype=bool)
        self.forced_down = np.zeros(n, dtype=bool)
        self.emergency_report = np.zeros(n, dtype=bool)   # motor: its emergency report gets out
        self.failures_skipped = []                # scheduled, but the drone wasn't flying then
        self._falling = np.zeros(n, dtype=bool)
        self._forced = np.zeros(n, dtype=bool)    # forced landing under way
        self._forced_target = np.zeros((n, 3), dtype=np.float32)
        self._forced_counter = np.zeros(n, dtype=np.int64)
        self._fail_queue = disturbances.check_failures(failures or [], n)
        if self.failure_cfg is not None:
            self._fail_rng = np.random.default_rng([int(self.failure_cfg['seed']), getattr(self, 'episode_index', 0)])

    def _forced_vz_limit(self, actions_dict):
        """A forced-landing drone's commanded vz, clipped to +-land_vz (the action applied)."""
        actions_dict = dict(actions_dict)
        lim = self.failure_cfg['land_vz']
        for i in np.flatnonzero(self._forced):
            a = np.array(actions_dict[f'drone_{i}'], dtype=np.float32)
            a[2] = np.clip(a[2], -lim, lim)
            actions_dict[f'drone_{i}'] = a
        return actions_dict

    def _trigger_failures(self):
        """Start the scheduled failures whose time has come."""
        now = self._t_step * self.control_dt + 1e-9
        while self._fail_queue and self._fail_queue[0]['t'] <= now:
            f = self._fail_queue.pop(0)
            i = int(f['drone'].split('_')[1])
            if self.waiting[i] or self.landed[i] or self.crashed[i] or self.cancelled[i]:
                self.failures_skipped.append(f)
                continue
            self.failed[i], self.failure_t[i] = f['type'], f['t']
            if f['type'] == 'motor':
                self._falling[i] = True
                self.emergency_report[i] = self._fail_rng.uniform() < self.failure_cfg['p_emergency_report']
                self.adapter.set_falling(i)
            elif f['type'] == 'land':
                x, y, _ = self.adapter.get_positions()[i]
                roof = disturbances.roof_under(self.layout, x, y)
                self._forced[i] = True
                self._forced_target[i] = (x, y, roof + self.failure_cfg['land_z'])
                self._landing[i] = False
            else:
                self.comms_lost[i] = True

    def _failure_carrots(self):
        """A forced-landing drone's carrot (observation + LiDAR heading): the point under it."""
        for i in np.flatnonzero(self._forced):
            self.adapter.goals[f'drone_{i}'] = self._forced_target[i].copy()

    def _forced_landing_done(self, positions, speed):
        """A forced-landing drone that has settled at its landing point (land_tol, settle_speed,
        settle_steps) has landed: it is parked from the next step on."""
        near = np.linalg.norm(positions - self._forced_target, axis=1) < self.failure_cfg['land_tol']
        ok = self._forced & ~self.crashed & near & (speed < self.settle_speed)
        self._forced_counter = np.where(ok, self._forced_counter + 1, 0)
        done = self._forced & (self._forced_counter >= self.settle_steps)
        if done.any():
            self.forced_down |= done
            self.landed |= done
            self._forced &= ~done
        self._forced &= ~self.crashed

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
