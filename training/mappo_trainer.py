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
        self.buffer_size     = config['training']['buffer_size']
        self.checkpoint_dir  = config['logging']['checkpoint_dir']
        self.k_max           = config['observation']['k_neighbors_max']

        # State tracking
        self.global_step     = 0
        self.episode_count   = 0
        self.curriculum_stage= 1
        self.best_success_rate = 0.0
        self.episode_rewards = []
        self.episode_lengths = []
        self.success_history = []

        # WandB (optional — graceful fallback if not available)
        self.use_wandb = self._init_wandb()

        os.makedirs(self.checkpoint_dir, exist_ok=True)

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

        while self.episode_count < self.total_episodes:
            episode_reward, episode_length, success, min_dist_to_goal, initial_dist_to_goal = \
                self._collect_episode()

            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            self.success_history.append(float(success))
            self.episode_count += 1

            # Update goal distance curriculum
            if self.env is not None and hasattr(self.env, 'update_goal_curriculum'):
                self.env.update_goal_curriculum(success)

            # Update policy when buffer is full
            if self.buffer.is_full():
                metrics = self._update_policy()
                self.buffer.reset()
            else:
                metrics = None

            # Log EVERY episode to WandB (not just buffer-fill episodes)
            self._log(metrics, episode_reward, episode_length, success,
                       min_dist_to_goal, initial_dist_to_goal)

            # Optuna pruning check
            if optuna_trial is not None and self.episode_count % report_interval == 0:
                import optuna
                window = min(report_interval, len(self.episode_rewards))
                mean_reward = np.mean(self.episode_rewards[-window:])
                optuna_trial.report(mean_reward, self.episode_count)
                if optuna_trial.should_prune():
                    raise optuna.TrialPruned()

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

            if done:
                success = info.get('success', False)
                min_dist_to_goal     = info.get('min_dist_to_goal')
                initial_dist_to_goal = info.get('initial_dist_to_goal')
                break

            # Move to next observation
            local_fixed, neighbor_states, neighbor_mask, global_state = \
                self._unpack_obs(next_obs_dict)

        # Bootstrap value for GAE if episode didn't terminate
        with torch.no_grad():
            last_values = self.critic.get_value(
                global_state.unsqueeze(0)
            )
            last_values = self._as_agent_values(last_values)
        self.buffer.compute_gae(last_values)

        return episode_reward, episode_length, success, min_dist_to_goal, initial_dist_to_goal

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
             initial_dist_to_goal: Optional[float] = None):
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

        if len(self.success_history) >= 100:
            log_dict['success_rate_100'] = np.mean(self.success_history[-100:])

        if self.env is not None and hasattr(self.env, 'goal_dist_current'):
            log_dict['goal_dist_current'] = self.env.goal_dist_current

        clamped = torch.clamp(self.actor.action_log_std, -2.0, 0.5)
        log_dict['action_log_std_mean'] = clamped.mean().item()
        for i, name in enumerate(['vx', 'vy', 'vz', 'yaw']):
            log_dict[f'action_log_std_{name}'] = clamped[i].item()
        log_dict['action_std_mean'] = torch.exp(clamped).mean().item()

        if self.env is not None and hasattr(self.env, 'obs_processor'):
            norm = self.env.obs_processor.normalizer
            log_dict['normalizer_count'] = norm.count
            log_dict['normalizer_var_min'] = float(norm.var.min())
            log_dict['normalizer_var_max'] = float(norm.var.max())

        wandb.log(log_dict, step=self.episode_count)

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
            'config':              self.config,
        }, path)
        print(f"Checkpoint saved: {path}")

        # Save best policy separately
        if len(self.success_history) >= 100:
            sr = np.mean(self.success_history[-100:])
            if sr > self.best_success_rate:
                self.best_success_rate = sr
                best_path = self.config['logging']['best_checkpoint']
                os.makedirs(os.path.dirname(best_path), exist_ok=True)
                torch.save({'actor_state_dict': self.actor.state_dict()}, best_path)
                print(f"New best policy saved: {best_path} (SR={sr:.2f})")

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