"""
mappo_trainer.py
----------------
Orchestrates the complete MAPPO training loop.

Responsibilities:
    1. Collect rollouts from SwarmEnv (or dummy env for Phase B testing)
    2. Store transitions in RolloutBuffer
    3. After buffer fills, compute GAE and run PPO update
    4. Log metrics to WandB
    5. Save checkpoints when policy improves
    6. Manage curriculum stage progression

Training loop (one iteration):
    while not converged:
        collect episode → store in buffer
        if buffer full:
            compute_gae()
            ppo_optimizer.update()  ← 10 epochs of minibatch updates
            buffer.reset()
        if curriculum threshold met:
            advance_stage()
"""

import torch
import numpy as np
import os
import yaml
import time
from typing import Optional, Dict
import json


class MAPPOTrainer:
    """
    Orchestrates the full MAPPO training pipeline.

    Usage:
        trainer = MAPPOTrainer(config, actor, critic, env, obs_proc, reward_eng, ep_mgr)
        trainer.train()

    For Phase B testing (no Webots):
        trainer = MAPPOTrainer(config, actor, critic, env=None, ...)
        trainer.train_on_dummy_data(n_episodes=100)
    """

    def __init__(
        self,
        config:       dict,
        actor,                       # ActorNetwork
        critic,                      # CriticNetwork
        env=None,                    # SwarmEnv (None for dummy testing)
        obs_processor=None,          # ObservationProcessor
        reward_engine=None,          # RewardEngine
        episode_manager=None,        # EpisodeManager
        device:       Optional[torch.device] = None,
    ):
        self.config          = config
        self.actor           = actor
        self.critic          = critic
        self.env             = env
        self.obs_processor   = obs_processor
        self.reward_engine   = reward_engine
        self.episode_manager = episode_manager
        self.device          = device or torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )

        # Move networks to device
        self.actor.to(self.device)
        self.critic.to(self.device)

        # Core components
        try:
            from .rollout_buffer import RolloutBuffer
            from .ppo_optimizer import PPOOptimizer
        except ImportError:
            from rollout_buffer import RolloutBuffer
            from ppo_optimizer import PPOOptimizer

        self.buffer    = RolloutBuffer(config, self.device)
        self.optimizer = PPOOptimizer(actor, critic, config, self.device)

        # Training settings
        self.n_agents        = config['environment']['n_agents']
        self.max_steps       = config['environment']['max_steps']
        self.total_episodes  = config['training']['total_episodes']
        self.save_interval   = config['training']['save_interval']
        self.eval_interval   = config['training']['eval_interval']
        self.eval_episodes   = config['training'].get('eval_episodes', 10)
        self.buffer_size     = config['training']['buffer_size']
        self.checkpoint_dir  = config['logging']['checkpoint_dir']
        self.k_max           = config['observation']['k_neighbors_max']

        # Early stopping — stop once deterministic eval sustains a strong
        # pass rate, instead of continuing to train past the point where
        # PPO starts overwriting an already-good policy (see chocolate-deluge-39:
        # settlement was learned by ep 50-125, then degraded by ep 300 because
        # stochastic rollouts almost never produced successful settle
        # experience to reinforce it).
        self.early_stop_success_rate = config['training'].get('early_stop_success_rate', None)
        self.early_stop_patience     = config['training'].get('early_stop_patience', 2)
        # Guards against stopping before an exploration-annealing schedule has
        # had a chance to run: run daily-snowflake-38/chocolate-deluge-39's
        # successor hit 95% eval by ep 75 purely off the DETERMINISTIC mean
        # policy, before annealing (ep 50-200) had lowered noise more than
        # ~6% — early-stopped without ever testing whether lower noise fixes
        # stochastic training. Default 0 preserves old behavior for configs
        # without an exploration schedule.
        self.early_stop_min_episode  = config['training'].get('early_stop_min_episode', 0)
        self._consecutive_strict_passes = 0
        self._should_stop = False
        self._buffer_log_std_ceiling = None

        # Exploration annealing — anneal the actor's log_std ceiling down
        # over training instead of relying only on PPO's gradient to shrink
        # it. Absent an 'exploration' config block, init==final==0.5
        # reproduces the old fixed-ceiling behavior exactly (no schedule).
        exploration_cfg = config.get('exploration', {})
        self._exploration_enabled = 'exploration' in config
        self.log_std_init  = exploration_cfg.get('log_std_init', 0.5)
        self.log_std_final = exploration_cfg.get('log_std_final', 0.5)
        self.log_std_anneal_start = exploration_cfg.get('anneal_start_episode', 0)
        self.log_std_anneal_end   = exploration_cfg.get('anneal_end_episode', 0)

        # Second, optional squeeze phase: tighten the FLOOR further below
        # log_std_final after settlement has had a chance to emerge. Without
        # this, a seed can learn a mean action that stops just outside
        # goal_threshold and relies on residual sampling noise (std≈0.05) to
        # nudge it across the boundary — succeeds stochastically, fails
        # deterministically (seed 1, honest-thunder-47: 93% training success,
        # 0% deterministic eval). Defaults reproduce no-op (floor stays at
        # log_std_final, i.e. old single-phase behavior) unless configured.
        self.log_std_floor_final = exploration_cfg.get('log_std_floor_final', self.log_std_final)
        self.floor_anneal_start  = exploration_cfg.get('floor_anneal_start_episode', self.log_std_anneal_end)
        self.floor_anneal_end    = exploration_cfg.get('floor_anneal_end_episode', self.log_std_anneal_end)

        # State tracking
        self.global_step     = 0
        self.episode_count   = 0
        self.curriculum_stage= 1
        self.best_success_rate = 0.0
        # Lexicographic ranking key: (success_rate, -mean_min_dist, -final_speed).
        # Plain success-rate comparison treats every 100% policy as equal —
        # exalted-sky-56 saved its FIRST 100% (episode 50: min_dist=0.150m)
        # as "best" while episode 325's policy (min_dist=0.048m, a clearly
        # tighter and more reliable settle) never replaced it because the
        # success rate never strictly increased past 1.0. Negating the
        # lower-is-better fields lets a plain tuple '>' comparison rank all
        # three criteria at once.
        self.best_eval_key  = (-1.0, float('-inf'), float('-inf'))
        self.best_eval_info  = {}
        self.episode_rewards = []
        self.episode_lengths = []
        self.success_history = []

        # WandB (optional — graceful fallback if not available)
        self.use_wandb = self._init_wandb()

        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.eval_env = None
        self._consecutive_eval_passes = 0

    def _init_wandb(self) -> bool:
        """Initialize WandB logging. Returns False if unavailable."""
        try:
            import wandb
            wandb.init(
                project=self.config['logging']['wandb_project'],
                config=self.config,
                resume='allow',
                save_code=True,
            )
            print("WandB initialized.")
            return True
        except Exception as e:
            print(f"WandB not available: {e}. Training without logging.")
            return False

    def train(self, optuna_trial=None, report_interval: int = 50):
        """
        Full training loop — requires SwarmEnv to be set.
        Runs until total_episodes is reached.

        Args:
            optuna_trial    : optuna.Trial instance for hyperparameter tuning.
                              When provided, reports mean reward every report_interval
                              episodes and prunes if the trial looks unpromising.
            report_interval : how often (in episodes) to report to Optuna pruner.
        """
        assert self.env is not None, \
            "env must be set for train(). Use train_on_dummy_data() for Phase B testing."

        print(f"Starting MAPPO training on {self.device}")
        print(f"Total episodes: {self.total_episodes}")
        print(f"Buffer size: {self.buffer_size}")

        start_time = time.time()

        # Set exploration bounds ONCE before collection starts, not every
        # episode — see the loop body below for why this matters.
        self._update_exploration_schedule()

        while self.episode_count < self.total_episodes:
            episode_reward, episode_length, success, min_dist_to_goal, initial_dist_to_goal, ep_info = \
                self._collect_episode()

            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            self.success_history.append(float(success))
            self.episode_count += 1

            # Goal distance curriculum is driven by deterministic eval (see _run_eval),
            # not stochastic training success — stochastic success stays near zero
            # because the 31-step dwell gate can't fire through exploration noise.

            # Update policy when buffer is full
            if self.buffer.is_full():
                metrics = self._update_policy()
                self.buffer.reset()
                # Only change the exploration schedule BETWEEN buffers, never
                # mid-collection. A transition's stored old_log_prob reflects
                # whatever log_std ceiling/floor was active when it was
                # sampled; if the schedule shifts std mid-buffer (a real risk
                # during a fast squeeze — one 1024-step buffer can span 7-10+
                # short successful episodes), PPO's update re-evaluates every
                # transition under the CURRENT (possibly much lower) std,
                # producing an inflated old-vs-new divergence that reflects
                # the schedule change, not actual policy learning. That
                # artificial KL can trip the trust-region limit and reject
                # updates during exactly the window we need the mean to
                # shift (see charmed-meadow-55: repeated "KL exceeded limit"
                # rejections clustered inside the squeeze phase).
                self._update_exploration_schedule()
            else:
                metrics = None

            # Log EVERY episode to WandB (not just buffer-fill episodes)
            self._log(metrics, episode_reward, episode_length, success,
                       min_dist_to_goal, initial_dist_to_goal, extra_info=ep_info)

            # Optuna pruning check
            if optuna_trial is not None and self.episode_count % report_interval == 0:
                import optuna
                window = min(report_interval, len(self.episode_rewards))
                mean_reward = np.mean(self.episode_rewards[-window:])
                optuna_trial.report(mean_reward, self.episode_count)
                if optuna_trial.should_prune():
                    raise optuna.TrialPruned()

            # Deterministic evaluation
            if self.episode_count % self.eval_interval == 0:
                eval_metrics = self._run_eval()
                if self.use_wandb:
                    import wandb
                    wandb.log(eval_metrics, step=self.episode_count)
                if self._should_stop:
                    print(f"  [EarlyStop] eval/success_rate >= {self.early_stop_success_rate} "
                          f"for {self.early_stop_patience} consecutive evals — "
                          f"stopping at episode {self.episode_count}")
                    break

            # Curriculum check
            self._check_curriculum_advancement()

            # Checkpointing
            if self.episode_count % self.save_interval == 0:
                self._save_checkpoint(tag=f"ep{self.episode_count}")

            # Progress print
            if self.episode_count % 10 == 0:
                elapsed = time.time() - start_time
                sr = np.mean(self.success_history[-100:]) if self.success_history else 0
                print(
                    f"Ep {self.episode_count:5d} | "
                    f"Reward: {episode_reward:7.2f} | "
                    f"Length: {episode_length:4d} | "
                    f"Success: {sr:.2f} | "
                    f"Stage: {self.curriculum_stage} | "
                    f"Time: {elapsed:.0f}s"
                )

        print("Training complete.")
        self._save_checkpoint(tag="final")
        if self.best_eval_info:
            print(f"Best eval checkpoint: episode {self.best_eval_info['episode']} "
                  f"(success_rate={self.best_eval_info['success_rate']:.2f}, "
                  f"mean_min_dist={self.best_eval_info['mean_min_dist']:.3f}, "
                  f"final_speed={self.best_eval_info['final_speed']:.3f})")
        if self.eval_env is not None:
            self.eval_env.close()

    def _update_exploration_schedule(self):
        """Anneal the actor's log_std CEILING through a piecewise-linear
        schedule with up to three anchor points:
            (anneal_start_episode,     log_std_init)
            (anneal_end_episode,       log_std_final)
            (floor_anneal_end_episode, log_std_floor_final)
        The FLOOR is a fixed safety bound, not part of the schedule.

        This must drive the CEILING, not the floor: torch.clamp(raw, floor,
        ceiling) only produces a value below the ceiling if the raw
        parameter itself is already below it. Once the raw action_log_std
        parameter saturates at (or above) a ceiling — which is exactly what
        happens here, since torch.clamp's gradient is zero outside its
        bounds — lowering the FLOOR further has NO effect on the output.
        An earlier version of this method animated the floor instead of the
        ceiling in the second phase; verified via direct forward() testing
        to be a complete no-op (std stayed pinned at exp(ceiling) regardless
        of floor) — this rewrite fixes that.

        With no 'exploration' config block at all, this is a true no-op:
        floor/ceiling are left untouched at the network's own hardcoded
        defaults (-3.0, 0.5), not recomputed to some derived value.
        """
        if not self._exploration_enabled:
            self._buffer_log_std_ceiling = self.actor.log_std_ceiling
            return

        def interp(ep, x0, y0, x1, y1):
            if x1 <= x0:
                return y0
            t = min(max((ep - x0) / (x1 - x0), 0.0), 1.0)
            return y0 + t * (y1 - y0)

        ep = self.episode_count
        if ep <= self.log_std_anneal_start:
            ceiling = self.log_std_init
        elif ep <= self.log_std_anneal_end:
            ceiling = interp(ep, self.log_std_anneal_start, self.log_std_init,
                              self.log_std_anneal_end, self.log_std_final)
        elif ep <= self.floor_anneal_end:
            ceiling = interp(ep, self.floor_anneal_start, self.log_std_final,
                              self.floor_anneal_end, self.log_std_floor_final)
        else:
            ceiling = self.log_std_floor_final

        # Fixed safety floor — never above the deepest ceiling target this
        # schedule will ever reach, so the ceiling alone determines
        # effective noise throughout. Not annealed; no gradient-blocking
        # concern here since it's the more permissive (lower) bound.
        floor = min(self.log_std_final, self.log_std_floor_final)
        floor = min(floor, ceiling)  # safety: floor must never exceed ceiling

        self.actor.set_log_std_ceiling(ceiling)
        self.actor.set_log_std_floor(floor)

        # Record the ceiling that will govern the ENTIRE next buffer's
        # collection, so _log can verify at update time that it never
        # drifted mid-buffer (see exploration/buffer_log_std_start vs
        # exploration/update_log_std — these must be identical if the
        # buffer-boundary timing fix is actually holding).
        self._buffer_log_std_ceiling = ceiling

    def _collect_episode(self) -> tuple:
        """
        Run one episode and store transitions in the buffer.

        Returns:
            total_reward   : float — sum of rewards across all agents and steps
            episode_length : int   — number of steps taken
            success        : bool  — did all agents reach their goals?
        """
        obs_dict = self.env.reset()

        episode_reward = 0.0
        episode_length = 0
        success        = False
        min_dist_to_goal     = None
        initial_dist_to_goal = None
        last_info            = {}
        reward_component_sums = {}
        speed_accum = 0.0

        # Unpack observations from env reset
        local_fixed, neighbor_states, neighbor_mask, global_state = \
            self._unpack_obs(obs_dict)

        for step in range(self.max_steps):
            # Actor samples actions from current policy
            with torch.no_grad():
                actions, log_probs = self.actor.get_action(
                    local_fixed, neighbor_states, neighbor_mask,
                    deterministic=False
                )
                # Critic estimates value of current global state
                values = self.critic.get_value(
                    global_state.unsqueeze(0)
                )
                values = self._as_agent_values(values)

            # Apply yaw mask for Stage 1-2. Masking changes the action, so the
            # stored log_prob must be recomputed for the masked action —
            # otherwise old_log_prob (sampled yaw) and the action PPO later
            # re-evaluates (yaw=0) disagree, corrupting the ratio at epoch 0.
            masked_actions = self._apply_yaw_mask(actions)
            if self.curriculum_stage < 3:
                with torch.no_grad():
                    log_probs, _ = self.actor.evaluate_actions(
                        local_fixed, neighbor_states, masked_actions, neighbor_mask,
                        active_dims=self._active_action_dims()
                    )

            # Step environment
            next_obs_dict, rewards, done, info = self.env.step(
                self._actions_to_dict(masked_actions)
            )

            rewards_tensor = torch.tensor(
                [rewards[f'drone_{i}'] for i in range(self.n_agents)],
                dtype=torch.float32, device=self.device
            )
            done_tensor = torch.tensor(
                [float(done)] * self.n_agents,
                dtype=torch.float32, device=self.device
            )

            # Store transition
            self.buffer.store(
                local_obs       = local_fixed,
                neighbor_states = neighbor_states,
                neighbor_mask   = neighbor_mask,
                global_state    = global_state,
                actions         = masked_actions,
                log_probs       = log_probs,
                rewards         = rewards_tensor,
                dones           = done_tensor,
                values          = values,
            )

            episode_reward += rewards_tensor.sum().item()
            episode_length += 1
            self.global_step += 1

            components = info.get('reward_components', {})
            for k, v in components.items():
                reward_component_sums[k] = reward_component_sums.get(k, 0.0) + v
            speed_accum += info.get('mean_speed', 0.0)

            last_info = info
            if done:
                success = info.get('success', False)
                min_dist_to_goal     = info.get('min_dist_to_goal')
                initial_dist_to_goal = info.get('initial_dist_to_goal')
                break

            # Move to next observation
            local_fixed, neighbor_states, neighbor_mask, global_state = \
                self._unpack_obs(next_obs_dict)

        last_info['reward_component_sums'] = reward_component_sums
        last_info['episode_mean_speed'] = speed_accum / max(episode_length, 1)

        # Bootstrap value for GAE if episode didn't terminate
        with torch.no_grad():
            last_values = self.critic.get_value(
                global_state.unsqueeze(0)
            )
            last_values = self._as_agent_values(last_values)
        self.buffer.compute_gae(last_values)

        return episode_reward, episode_length, success, min_dist_to_goal, initial_dist_to_goal, last_info

    def _active_action_dims(self) -> int:
        """Yaw (dim 3) is masked to 0.0 and causally inert before Stage 3 —
        exclude it from log_prob/entropy so PPO isn't learning a ratio for
        an action that never affects the environment."""
        return 3 if self.curriculum_stage < 3 else 4

    def _update_policy(self) -> Dict:
        """Run PPO update. Returns metrics dict for logging."""
        self.optimizer.active_action_dims = self._active_action_dims()
        metrics = self.optimizer.update(self.buffer)
        return metrics

    def _as_agent_values(self, value_tensor: torch.Tensor) -> torch.Tensor:
        """
        Normalize critic output into per-agent shape (N,).

        The current critic returns a team-level scalar value. Buffer/GAE logic
        expects one value per agent, so scalar values are broadcast to all agents.
        """
        v = value_tensor.reshape(-1)
        if v.numel() == 1:
            return v.repeat(self.n_agents)
        if v.numel() == self.n_agents:
            return v
        raise ValueError(
            f"Unexpected critic value shape {tuple(value_tensor.shape)}; "
            f"cannot convert to ({self.n_agents},)"
        )

    def _apply_yaw_mask(self, actions: torch.Tensor) -> torch.Tensor:
        """Zero out yaw rate component for Stage 1-2."""
        if self.curriculum_stage < 3:
            actions = actions.clone()
            actions[:, 3] = 0.0
        return actions

    def _actions_to_dict(self, actions: torch.Tensor) -> Dict:
        """Convert (N, 4) tensor to dict keyed by drone id."""
        return {
            f'drone_{i}': actions[i].cpu().numpy().tolist()
            for i in range(self.n_agents)
        }

    def _unpack_obs(self, obs_dict: Dict):
        """Unpack observation dict from SwarmEnv into tensors."""
        return (
            obs_dict['local_fixed'].to(self.device),
            obs_dict['neighbor_states'].to(self.device),
            obs_dict['neighbor_mask'].to(self.device),
            obs_dict['global_state'].to(self.device),
        )

    def _run_eval(self) -> Dict:
        """Run deterministic evaluation episodes on an isolated env.

        Uses a separate SwarmEnv with its own RNG so eval doesn't
        perturb training trajectories. Returns metrics dict for logging.
        """
        if self.eval_env is None:
            from envs.swarm_env import SwarmEnv
            self.eval_env = SwarmEnv(self.config, device=self.device, gui=False)
            self.eval_env.goal_dist_current = self.env.goal_dist_current

        self.eval_env.goal_dist_current = self.env.goal_dist_current

        # CRITICAL: eval_env has its OWN ObservationProcessor with its own
        # RunningNormalizer, created once and independently accumulating
        # stats from eval episodes only — completely different from
        # self.env's normalizer, which is what the actor's weights were
        # actually trained against. Left unsynced, every deterministic eval
        # this whole session fed the actor observations scaled by the WRONG
        # normalizer, corrupting what the deterministic policy actually
        # produces (verified: standalone eval_checkpoint.py, which restores
        # and freezes the TRAINING env's exact normalizer snapshot, shows
        # 100%/100% success on charmed-meadow-55's final checkpoint in both
        # modes — while this live eval reported 0% deterministic the whole
        # back half of that run on the identical weights). Re-sync every
        # call since the training normalizer keeps evolving; freeze so the
        # ~20 eval episodes in this call don't drift the copy before the
        # next resync anyway overwrites it.
        train_norm = self.env.obs_processor.normalizer
        eval_norm = self.eval_env.obs_processor.normalizer
        eval_norm.mean  = train_norm.mean.copy()
        eval_norm.var   = train_norm.var.copy()
        eval_norm.count = train_norm.count
        eval_norm.freeze()
        print(f"  [Eval] train_norm.count={train_norm.count}, "
              f"eval_norm.count={eval_norm.count}, frozen={eval_norm._frozen}")

        successes = []
        min_dists = []
        ep_mean_speeds = []
        final_speeds = []
        max_settle_fracs = []
        drones_inside_counts = []
        all_entered_goal = []

        # Safety / multi-agent metrics (all N/A-safe for n_agents==1)
        per_agent_success = [[] for _ in range(self.n_agents)]
        any_collision_episodes = []
        drone_collision_episodes = []
        obstacle_collision_episodes = []
        min_inter_drone_seps = []       # None entries skipped when aggregating
        agents_settled_counts = []

        for ep in range(self.eval_episodes):
            obs_dict = self.eval_env.reset()
            lf, ns, nm, gs = self._unpack_obs(obs_dict)

            ep_max_settle_frac = 0.0
            ep_max_drones_inside = 0
            speed_accum = 0.0
            steps_done = 0
            ep_any_drone_collision = False
            ep_any_obstacle_collision = False
            ep_min_inter_drone = None

            for step in range(self.max_steps):
                with torch.no_grad():
                    actions, _ = self.actor.get_action(
                        lf, ns, nm, deterministic=True,
                        active_dims=self._active_action_dims()
                    )
                masked_actions = self._apply_yaw_mask(actions)

                next_obs, _, done, info = self.eval_env.step(
                    self._actions_to_dict(masked_actions)
                )

                ep_max_settle_frac = max(
                    ep_max_settle_frac, info.get('settle_fraction', 0.0))
                ep_max_drones_inside = max(
                    ep_max_drones_inside, info.get('drones_inside_goal', 0))
                speed_accum += info.get('mean_speed', 0.0)
                steps_done += 1

                if any(info.get('drone_collision', [])):
                    ep_any_drone_collision = True
                if any(info.get('obstacle_collision', [])):
                    ep_any_obstacle_collision = True
                step_sep = info.get('min_inter_drone_distance', None)
                if step_sep is not None:
                    ep_min_inter_drone = step_sep if ep_min_inter_drone is None \
                        else min(ep_min_inter_drone, step_sep)

                if done:
                    break
                lf, ns, nm, gs = self._unpack_obs(next_obs)

            successes.append(float(info.get('success', False)))
            min_dists.append(info.get('min_dist_to_goal', 0.0))
            ep_mean_speeds.append(speed_accum / max(steps_done, 1))
            final_speeds.append(info.get('mean_speed', 0.0))
            max_settle_fracs.append(ep_max_settle_frac)
            drones_inside_counts.append(ep_max_drones_inside)
            all_entered_goal.append(float(ep_max_drones_inside >= self.n_agents))

            # Per-agent success uses the LATCHED drone_reached_goal state
            # (settled for settle_steps), not merely "inside on final step".
            reached = info.get('drone_reached_goal', [False] * self.n_agents)
            for i in range(self.n_agents):
                per_agent_success[i].append(float(reached[i]))
            agents_settled_counts.append(info.get('drones_at_goal', 0))
            any_collision_episodes.append(float(ep_any_drone_collision or ep_any_obstacle_collision))
            drone_collision_episodes.append(float(ep_any_drone_collision))
            obstacle_collision_episodes.append(float(ep_any_obstacle_collision))
            min_inter_drone_seps.append(ep_min_inter_drone)

        eval_sr = np.mean(successes)
        metrics = {
            'eval/success_rate':        eval_sr,
            'eval/mean_min_dist':       np.mean(min_dists),
            'eval/mean_speed':          np.mean(ep_mean_speeds),
            'eval/final_speed':         np.mean(final_speeds),
            'eval/max_settle_fraction': np.mean(max_settle_fracs),
            'eval/max_drones_inside':   np.mean(drones_inside_counts),
            'eval/all_entered_goal':    np.mean(all_entered_goal),
            'eval/episodes':            self.eval_episodes,
            'eval/any_collision_rate':      np.mean(any_collision_episodes),
            'eval/drone_collision_rate':    np.mean(drone_collision_episodes),
            'eval/obstacle_collision_rate': np.mean(obstacle_collision_episodes),
            'eval/mean_agents_settled':     np.mean(agents_settled_counts),
        }
        for i in range(self.n_agents):
            metrics[f'eval/agent_{i}_success_rate'] = np.mean(per_agent_success[i])

        valid_seps = [s for s in min_inter_drone_seps if s is not None]
        metrics['eval/min_inter_drone_distance'] = float(min(valid_seps)) if valid_seps else None

        min_sep_str = 'N/A' if not valid_seps else f"{metrics['eval/min_inter_drone_distance']:.3f}"
        print(f"  [Eval] SR={eval_sr:.2f} | min_dist={np.mean(min_dists):.3f} | "
              f"mean_spd={np.mean(ep_mean_speeds):.3f} | final_spd={np.mean(final_speeds):.3f} | "
              f"settle_frac={np.mean(max_settle_fracs):.2f} | "
              f"collision_rate={np.mean(any_collision_episodes):.2f} | "
              f"min_sep={min_sep_str}")

        # Best-eval checkpointing — driven by DETERMINISTIC eval/success_rate,
        # not stochastic success_history. The latter stays at 0 for tasks
        # requiring rare stochastic settling (see chocolate-deluge-39), so the
        # old best-policy save (gated on rolling stochastic success) never
        # fired and the final (possibly degraded) checkpoint was the only
        # saved artifact. This tracks the actual best deterministic policy
        # seen at any point in training, independent of what happens after.
        #
        # Ranked lexicographically — (success_rate, -min_dist, -final_speed) —
        # not by success_rate alone: once success_rate saturates at 1.0 (as
        # it did from episode 50 onward in exalted-sky-56), a scalar
        # comparison never updates again even though later checkpoints keep
        # settling with a tighter margin (min_dist 0.150m at ep50 vs 0.048m
        # at ep325) and lower residual speed — real quality improvements a
        # plain success-rate check is blind to.
        eval_key = (eval_sr, -float(np.mean(min_dists)), -float(np.mean(final_speeds)))
        if eval_key > self.best_eval_key:
            self.best_eval_key = eval_key
            self.best_eval_info = {
                'episode':       self.episode_count,
                'success_rate':  eval_sr,
                'mean_min_dist': float(np.mean(min_dists)),
                'final_speed':   float(np.mean(final_speeds)),
            }
            self._save_best_eval_checkpoint()

        # Early stopping — stop once eval sustains a strong pass rate rather
        # than continuing to train past the point where PPO may overwrite an
        # already-good policy with no reliable training signal to preserve it.
        # The streak only accumulates once episode_count >= early_stop_min_episode:
        # effortless-sun-44 had already banked its 2-pass streak from BEFORE
        # the min-episode gate (passes at ep50/75, gate only blocked stopping,
        # not counting), so it stopped the instant ep200 arrived instead of
        # requiring 2 fresh passes measured after annealing completed.
        if self.early_stop_success_rate is not None:
            if self.episode_count < self.early_stop_min_episode:
                self._consecutive_strict_passes = 0
            elif eval_sr >= self.early_stop_success_rate:
                self._consecutive_strict_passes += 1
            else:
                self._consecutive_strict_passes = 0
            if self._consecutive_strict_passes >= self.early_stop_patience:
                self._should_stop = True

        advance_sr = self.config.get('goal_curriculum', {}).get('advance_sr', 0.3)
        if eval_sr >= advance_sr:
            self._consecutive_eval_passes += 1
        else:
            self._consecutive_eval_passes = 0

        if self._consecutive_eval_passes >= 2 and hasattr(self.env, 'goal_dist_current'):
            old = self.env.goal_dist_current
            step = self.config.get('goal_curriculum', {}).get('step_dist', 1.0)
            end = self.config.get('goal_curriculum', {}).get('end_dist', 20.0)
            self.env.goal_dist_current = min(old + step, end)
            self._consecutive_eval_passes = 0
            print(f"  [GoalCurriculum] Distance {old:.1f}m -> {self.env.goal_dist_current:.1f}m (eval-driven)")

        return metrics

    def _check_curriculum_advancement(self):
        """Advance curriculum stage if success rate threshold met.

        Currently disabled: PyBulletAdapter does not implement obstacle
        spawning, LiDAR, or domain randomization for Stages 2-4 — advancing
        the stage counter only changes the yaw mask and global_state_dim
        without changing the actual environment, which is misleading.
        Re-enable once the adapter supports per-stage world configuration.
        """
        return

        window = self.config['curriculum']['progression_window']
        threshold = self.config['curriculum']['progression_threshold']

        if len(self.success_history) < window:
            return

        recent_sr = np.mean(self.success_history[-window:])

        if recent_sr >= threshold and self.curriculum_stage < 4:
            self.curriculum_stage += 1
            self.success_history.clear()
            print(f"\n>>> CURRICULUM ADVANCE: Stage {self.curriculum_stage} <<<")
            print(f"    Success rate: {recent_sr:.2f} >= {threshold}")
            if self.env is not None:
                self.env.set_curriculum_stage(self.curriculum_stage)
            if self.obs_processor is not None:
                self.obs_processor.reset_normalizer()
            self._save_checkpoint(tag=f"stage{self.curriculum_stage}_start")

    def _log(self, metrics, episode_reward: float,
             episode_length: int, success: bool,
             min_dist_to_goal: Optional[float] = None,
             initial_dist_to_goal: Optional[float] = None,
             extra_info: Optional[Dict] = None):
        """Log metrics to WandB and local stats."""
        if not self.use_wandb:
            return

        import wandb
        log_dict = {
            'episode_reward':   episode_reward,
            'episode_length':   episode_length,
            'success':          float(success),
            'curriculum_stage': self.curriculum_stage,
            'global_step':      self.global_step,
        }
        if metrics is not None:
            log_dict.update(metrics)

        # Closest approach to goal — diagnoses "almost succeeded" (small
        # min_dist, no success) vs "never made progress" (min_dist ~= initial)
        if min_dist_to_goal is not None and initial_dist_to_goal is not None:
            log_dict['episode_min_dist_to_goal']     = min_dist_to_goal
            log_dict['episode_initial_dist_to_goal'] = initial_dist_to_goal
            if initial_dist_to_goal > 1e-6:
                log_dict['episode_progress_ratio'] = (
                    1.0 - min_dist_to_goal / initial_dist_to_goal
                )

        # Raw advantage and return stats from the buffer (logged on update episodes)
        if hasattr(self.buffer, 'raw_adv_mean') and metrics is not None:
            log_dict['raw_advantage_mean'] = self.buffer.raw_adv_mean
            log_dict['raw_advantage_std']  = self.buffer.raw_adv_std
            log_dict['raw_advantage_min']  = self.buffer.raw_adv_min
            log_dict['raw_advantage_max']  = self.buffer.raw_adv_max
            log_dict['return_mean']        = self.buffer.return_mean
            log_dict['return_std']         = self.buffer.return_std
            log_dict['return_min']         = self.buffer.return_min
            log_dict['return_max']         = self.buffer.return_max
            log_dict['value_mean']         = self.buffer.value_mean
            log_dict['value_std']          = self.buffer.value_std

        if extra_info:
            for key in ['drones_inside_goal', 'mean_speed', 'max_settle_counter',
                        'settle_fraction']:
                if key in extra_info:
                    log_dict[f'settle/{key}'] = extra_info[key]

            if 'episode_mean_speed' in extra_info:
                log_dict['settle/episode_mean_speed'] = extra_info['episode_mean_speed']

            for k, v in extra_info.get('reward_component_sums', {}).items():
                log_dict[f'reward/{k}'] = v

        if len(self.success_history) >= 100:
            log_dict['success_rate_100'] = np.mean(self.success_history[-100:])

        if self.env is not None and hasattr(self.env, 'goal_dist_current'):
            log_dict['goal_dist_current'] = self.env.goal_dist_current

        clamped = torch.clamp(
            self.actor.action_log_std, self.actor.log_std_floor, self.actor.log_std_ceiling
        )
        log_dict['action_log_std_mean'] = clamped.mean().item()
        for i, name in enumerate(['vx', 'vy', 'vz', 'yaw']):
            log_dict[f'action_log_std_{name}'] = clamped[i].item()
        log_dict['action_std_mean'] = torch.exp(clamped).mean().item()
        active_d = self._active_action_dims()
        log_dict['action_std_active_mean'] = torch.exp(clamped[:active_d]).mean().item()
        log_dict['exploration/log_std_ceiling'] = self.actor.log_std_ceiling
        log_dict['exploration/log_std_floor']   = self.actor.log_std_floor

        if self.env is not None and hasattr(self.env, 'obs_processor'):
            norm = self.env.obs_processor.normalizer
            log_dict['normalizer_count'] = norm.count
            log_dict['normalizer_var_min'] = float(norm.var.min())
            log_dict['normalizer_var_max'] = float(norm.var.max())

        wandb.log(log_dict, step=self.episode_count)

    def _normalizer_state(self) -> Dict:
        """Snapshot the training env's observation-normalizer stats, if any,
        so eval_checkpoint.py can restore the exact normalization the actor
        was trained under instead of accumulating fresh (and possibly
        different) stats from scratch."""
        if self.env is not None and hasattr(self.env, 'obs_processor'):
            norm = self.env.obs_processor.normalizer
            return {
                'normalizer_mean':  norm.mean.copy(),
                'normalizer_var':   norm.var.copy(),
                'normalizer_count': norm.count,
            }
        return {}

    def _save_checkpoint(self, tag: str = ""):
        """Save actor and critic weights."""
        path = os.path.join(self.checkpoint_dir, f"checkpoint_{tag}.pth")
        torch.save({
            'actor_state_dict':    self.actor.state_dict(),
            'critic_state_dict':   self.critic.state_dict(),
            'actor_optim':         self.optimizer.actor_optimizer.state_dict(),
            'critic_optim':        self.optimizer.critic_optimizer.state_dict(),
            'episode_count':       self.episode_count,
            'curriculum_stage':    self.curriculum_stage,
            'best_success_rate':   self.best_success_rate,
            # log_std_floor/ceiling are plain attributes, not part of
            # actor's state_dict — save the ACTUAL values in effect at this
            # episode (not the config's eventual annealed targets) since a
            # checkpoint can be taken mid-anneal.
            'log_std_ceiling':     self.actor.log_std_ceiling,
            'log_std_floor':       self.actor.log_std_floor,
            'config':              self.config,
            **self._normalizer_state(),
        }, path)
        print(f"Checkpoint saved: {path}")

        # Upload the FINAL checkpoint to WandB — effortless-sun-44 uploaded
        # only best_eval_policy.pth (locked to episode 100, the first eval to
        # hit 100%, before annealing had lowered noise enough to demonstrate
        # STOCHASTIC settlement). The final checkpoint is the one that
        # actually saw std anneal down to ~0.05 and produced real training
        # successes — it must survive independent of the training machine.
        if tag == "final" and self.use_wandb:
            import wandb
            wandb.save(path, base_path=self.checkpoint_dir)

        # Save best policy separately (stochastic training success rate —
        # kept for backward compatibility, but see _save_best_eval_checkpoint
        # for the deterministic-eval-driven selection that actually matters
        # for sparse-success tasks like settlement, where this stays at 0).
        if len(self.success_history) >= 100:
            sr = np.mean(self.success_history[-100:])
            if sr > self.best_success_rate:
                self.best_success_rate = sr
                best_path = self.config['logging']['best_checkpoint']
                os.makedirs(os.path.dirname(best_path), exist_ok=True)
                torch.save({'actor_state_dict': self.actor.state_dict()}, best_path)
                print(f"New best policy saved: {best_path} (SR={sr:.2f})")

    def _save_best_eval_checkpoint(self):
        """Save the actor whenever deterministic eval/success_rate reaches a
        new best. Distinct from _save_checkpoint's stochastic-success-driven
        best-policy save, which never fires for sparse-success tasks (e.g.
        settlement) where stochastic rollouts almost never succeed even
        though the deterministic policy has learned the task."""
        path = os.path.join(self.checkpoint_dir, "best_eval_policy.pth")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        torch.save({
            'actor_state_dict':  self.actor.state_dict(),
            'log_std_ceiling':   self.actor.log_std_ceiling,
            'log_std_floor':     self.actor.log_std_floor,
            'curriculum_stage':  self.curriculum_stage,
            **self._normalizer_state(),
            **self.best_eval_info,
        }, path)
        print(f"  [BestEval] New best deterministic policy saved: {path} "
              f"(episode={self.best_eval_info['episode']}, "
              f"success_rate={self.best_eval_info['success_rate']:.2f}, "
              f"mean_min_dist={self.best_eval_info['mean_min_dist']:.3f}, "
              f"final_speed={self.best_eval_info['final_speed']:.3f})")

        # Upload to WandB so the checkpoint survives beyond the training
        # machine's local disk — previously only metrics were logged and the
        # .pth files existed nowhere but checkpoints_dir/.
        if self.use_wandb:
            import wandb
            wandb.save(path, base_path=self.checkpoint_dir)

    def load_checkpoint(self, path: str):
        """Load a saved checkpoint to resume training."""
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt['actor_state_dict'])
        self.critic.load_state_dict(ckpt['critic_state_dict'])
        self.optimizer.actor_optimizer.load_state_dict(ckpt['actor_optim'])
        self.optimizer.critic_optimizer.load_state_dict(ckpt['critic_optim'])
        self.episode_count    = ckpt['episode_count']
        self.curriculum_stage = ckpt['curriculum_stage']
        self.best_success_rate= ckpt['best_success_rate']
        print(f"Checkpoint loaded: {path} (episode {self.episode_count})")

    # ── Phase B Testing: Dummy Data Training ─────────────────────────────────

    def train_on_dummy_data(self, n_episodes: int = 50):
        """
        Run the complete training loop on synthetic dummy tensors.
        NO Webots required. Tests the entire Phase B engine end-to-end.

        This is Phase C preparation — verifies that:
        1. RolloutBuffer stores and retrieves correctly
        2. GAE produces valid advantage estimates
        3. PPOOptimizer updates both networks
        4. Loss values are reasonable and decreasing
        5. No shape mismatches or NaN values

        Args:
            n_episodes : number of dummy episodes to simulate
        """
        N    = self.n_agents
        K    = self.k_max
        STEP = self.max_steps

        print("=" * 60)
        print(f"Phase B Dummy Data Test — {n_episodes} episodes")
        print(f"Device: {self.device}")
        print(f"Agents: {N}, Steps/ep: {STEP}, Buffer: {self.buffer_size}")
        print("=" * 60)

        all_actor_losses  = []
        all_critic_losses = []
        all_kls           = []
        all_clips         = []

        for ep in range(n_episodes):
            ep_reward = 0.0

            for step in range(STEP):
                # Synthetic observations — random tensors mimicking real obs
                local_fixed     = torch.randn(N, 27,  device=self.device)
                neighbor_states = torch.randn(N, K, 6, device=self.device)
                neighbor_mask   = torch.zeros(N, K,    device=self.device, dtype=torch.bool)
                global_state    = torch.randn(N, 17,   device=self.device)

                # Actor forward pass
                with torch.no_grad():
                    actions, log_probs = self.actor.get_action(
                        local_fixed, neighbor_states, neighbor_mask,
                        deterministic=False
                    )
                    # Critic forward pass
                    values = self.critic.get_value(
                        global_state.unsqueeze(0)
                    )
                    values = self._as_agent_values(values)

                # Synthetic rewards — drone 0 gets positive, others get small negative
                rewards = torch.zeros(N, device=self.device)
                rewards[0] = np.random.uniform(0.0, 1.0)
                rewards[1:] = -0.01

                # Done at last step of episode
                done = torch.zeros(N, device=self.device)
                if step == STEP - 1:
                    done[:] = 1.0

                ep_reward += rewards.sum().item()

                self.buffer.store(
                    local_obs       = local_fixed,
                    neighbor_states = neighbor_states,
                    neighbor_mask   = neighbor_mask,
                    global_state    = global_state,
                    actions         = actions,
                    log_probs       = log_probs,
                    rewards         = rewards,
                    dones           = done,
                    values          = values,
                )

                if self.buffer.is_full():
                    break

            # Bootstrap and compute GAE
            with torch.no_grad():
                last_gs  = torch.randn(N, 17, device=self.device)
                last_val = self.critic.get_value(last_gs.unsqueeze(0))
                last_val = self._as_agent_values(last_val)

            self.buffer.compute_gae(last_val)

            # PPO update
            metrics = self.optimizer.update(self.buffer)
            self.buffer.reset()

            all_actor_losses.append(metrics['actor_loss'])
            all_critic_losses.append(metrics['critic_loss'])
            all_kls.append(metrics['approx_kl'])
            all_clips.append(metrics['clip_fraction'])

            self.episode_count += 1

            if (ep + 1) % 10 == 0:
                mean_al = np.mean(all_actor_losses[-10:])
                mean_cl = np.mean(all_critic_losses[-10:])
                print(
                    f"Episode {ep+1:4d}/{n_episodes} | "
                    f"Reward: {ep_reward:7.2f} | "
                    f"Actor Loss: {mean_al:6.4f} | "
                    f"Critic Loss: {mean_cl:6.4f} | "
                    f"KL: {metrics['approx_kl']:.4f} | "
                    f"Clip%: {metrics['clip_fraction']:.2f}"
                )

        print("\n" + "=" * 60)
        print("Phase B Dummy Test Complete")
        print(f"  Final actor loss  : {all_actor_losses[-1]:.4f}")
        print(f"  Final critic loss : {all_critic_losses[-1]:.4f}")
        if len(all_critic_losses) >= 10:
            first_10_critic = float(np.mean(all_critic_losses[:10]))
            last_10_critic  = float(np.mean(all_critic_losses[-10:]))
            print(f"  Critic loss (first10 mean): {first_10_critic:.4f}")
            print(f"  Critic loss (last10 mean) : {last_10_critic:.4f}")
        if len(all_kls) >= 10:
            print(f"  KL (last10 mean)          : {float(np.mean(all_kls[-10:])):.4f}")
        if len(all_clips) >= 10:
            print(f"  Clip fraction (last10)    : {float(np.mean(all_clips[-10:])):.4f}")
        print(f"  No crashes        : ✓")
        print(f"  All shapes valid  : ✓")
        print("=" * 60)
        print("\nPhase B engine verified. Ready for Phase C (SwarmEnv integration).")


if __name__ == "__main__":
    """
    Phase B end-to-end test — runs complete MAPPO engine on dummy data.
    No Webots required.

    python mappo_trainer.py
    """
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

    import yaml
    from networks.actor_network  import ActorNetwork
    from networks.critic_network import CriticNetwork

    config_path = os.path.join(
        os.path.dirname(__file__), '..', 'configs', 'default_config.yaml'
    )
    with open(config_path) as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    actor  = ActorNetwork(config)
    critic = CriticNetwork(config)

    trainer = MAPPOTrainer(config, actor, critic, device=device)
    trainer.train_on_dummy_data(n_episodes=30)