"""
ppo_optimizer.py
----------------
Implements the PPO update step for both Actor and Critic networks.

Takes minibatches from RolloutBuffer and performs:
    1. Actor update  — clipped surrogate loss + entropy bonus
    2. Critic update — mean squared error value loss
    3. Combined      — single backward pass through total loss

Key PPO mechanisms:
    - Probability ratio r(theta) = pi_new(a|s) / pi_old(a|s)
    - Clipped surrogate: min(r * A, clip(r, 1-eps, 1+eps) * A)
    - Clip epsilon = 0.2 → policy can change at most 20% per update
    - Entropy bonus — prevents premature policy collapse
    - Gradient clipping — prevents exploding gradients
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Tuple
import yaml
import os


class PPOOptimizer:
    """
    PPO optimizer for the MAPPO actor-critic system.

    Manages the optimizers for both Actor and Critic and implements
    the full PPO loss computation and parameter update.

    The total loss combines three terms:
        L_total = -L_clip + c1 * L_value - c2 * L_entropy

    where:
        L_clip    = clipped surrogate objective (maximize)
        L_value   = critic MSE loss (minimize)
        L_entropy = policy entropy bonus (maximize, encourages exploration)
    """

    def __init__(self, actor, critic, config: dict, device: torch.device):
        """
        Args:
            actor   : ActorNetwork instance
            critic  : CriticNetwork instance
            config  : loaded default_config.yaml
            device  : torch.device
        """
        self.actor  = actor
        self.critic = critic
        self.device = device

        # PPO hyperparameters
        self.clip_epsilon    = config['training']['clip_epsilon']      # 0.2
        self.n_epochs        = config['training']['n_epochs']          # 10
        self.batch_size      = config['training']['batch_size']        # 256
        self.value_loss_coef = config['training']['value_loss_coef']   # 0.5
        self.entropy_coef    = config['training']['entropy_coef']      # 0.01
        self.max_grad_norm   = config['training']['max_grad_norm']     # 0.5
        self.target_kl       = config['training'].get('target_kl', 0.02)

        # Number of action dims PPO should evaluate log_prob/entropy over.
        # Yaw (dim 3) is masked to 0.0 and has no causal effect on reward in
        # Stage 1-2, so it must be excluded from the ratio/KL/entropy — set
        # by MAPPOTrainer before each update() call based on curriculum stage.
        self.active_action_dims = None

        # Separate optimizers for actor and critic
        # Allows different learning rates if needed
        self.actor_optimizer = optim.Adam(
            actor.parameters(),
            lr=config['training']['actor_lr']    # 3e-4
        )
        self.critic_optimizer = optim.Adam(
            critic.parameters(),
            lr=config['training']['critic_lr']   # 3e-4
        )

        # Running stats for logging
        self.stats = {
            'actor_loss':     [],
            'critic_loss':    [],
            'entropy':        [],
            'clip_fraction':  [],
            'approx_kl':      [],
            'actor_grad_norm':  [],
            'critic_grad_norm': [],
            'advantage_mean':   [],
            'advantage_std':    [],
        }

    def update(self, buffer) -> Dict[str, float]:
        """
        Run N_epochs of PPO updates over the rollout buffer.

        Called by MAPPOTrainer after each episode (or buffer fill).
        Iterates over the buffer multiple times with shuffled minibatches.

        Args:
            buffer : RolloutBuffer — must have compute_gae() called first

        Returns:
            metrics : dict of mean loss values for WandB logging
        """
        epoch_stats = {k: [] for k in self.stats}
        epochs_completed = 0
        kl_limit = 1.5 * self.target_kl

        for epoch in range(self.n_epochs):
            stop_update = False

            for batch in buffer.get_minibatches(self.batch_size):

                actor_loss, critic_loss, entropy, clip_frac, approx_kl = \
                    self._compute_losses(batch)

                if torch.isnan(actor_loss) or torch.isnan(critic_loss):
                    print(f"  [PPO] NaN in loss at epoch {epoch}, skipping batch")
                    continue

                # Check KL BEFORE stepping — once a policy drifts past the
                # trust region, applying this batch's update only makes it
                # worse. Checking after the fact (old behavior: abort only
                # past KL 10.0) let several bad minibatches compound within
                # a single epoch before anything stopped them.
                if not torch.isfinite(approx_kl) or approx_kl.item() > kl_limit:
                    print(f"  [PPO] KL {approx_kl.item():.4f} exceeded limit "
                          f"({kl_limit:.4f}) at epoch {epoch}, stopping update "
                          f"before applying this batch")
                    stop_update = True
                    break

                total_loss = (
                    - actor_loss
                    + self.value_loss_coef * critic_loss
                    - self.entropy_coef    * entropy
                )

                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()

                total_loss.backward()

                actor_gn  = nn.utils.clip_grad_norm_(self.actor.parameters(),  self.max_grad_norm)
                critic_gn = nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)

                self.actor_optimizer.step()
                self.critic_optimizer.step()

                epoch_stats['actor_loss'].append(actor_loss.item())
                epoch_stats['critic_loss'].append(critic_loss.item())
                epoch_stats['entropy'].append(entropy.item())
                epoch_stats['clip_fraction'].append(clip_frac.item())
                epoch_stats['approx_kl'].append(approx_kl.item())
                epoch_stats['actor_grad_norm'].append(actor_gn.item())
                epoch_stats['critic_grad_norm'].append(critic_gn.item())
                epoch_stats['advantage_mean'].append(batch['advantages'].mean().item())
                epoch_stats['advantage_std'].append(batch['advantages'].std().item())

            epochs_completed = epoch + 1

            if stop_update:
                break

        metrics = {k: float(torch.tensor(v).mean()) if v else 0.0
                   for k, v in epoch_stats.items()}
        metrics['epochs_used'] = epochs_completed

        # Explained variance: how well does the critic predict returns?
        # 1.0 = perfect, 0.0 = no better than mean, <0 = worse than mean
        with torch.no_grad():
            T = buffer.ptr
            val_flat = buffer.values[:T].reshape(-1)
            ret_flat = buffer.returns[:T].reshape(-1)
            var_ret = ret_flat.var()
            if var_ret < 1e-8:
                metrics['explained_variance'] = 0.0
            else:
                metrics['explained_variance'] = (
                    1.0 - (ret_flat - val_flat).var() / var_ret
                ).item()

        return metrics

    def _compute_losses(self, batch: Dict) -> Tuple[torch.Tensor, ...]:
        """
        Compute PPO actor loss, critic loss, and entropy for one minibatch.

        Actor Loss (Clipped Surrogate):
            ratio    = exp(log_prob_new - log_prob_old)
            surr1    = ratio * advantage
            surr2    = clip(ratio, 1-eps, 1+eps) * advantage
            L_actor  = mean(min(surr1, surr2))

        The min(surr1, surr2) ensures we never make the policy change
        too dramatically in one update — the lower of the two surrogate
        objectives acts as a constraint.

        Critic Loss (Mean Squared Error):
            L_critic = mean((V(s) - returns)^2)

        Entropy Bonus:
            L_entropy = mean(H[pi(.|s)])
            High entropy = more random policy = more exploration
        """
        # Unpack batch
        local_obs       = batch['local_obs']        # (B, 27)
        neighbor_states = batch['neighbor_states']  # (B, K, 6)
        neighbor_mask   = batch['neighbor_mask']    # (B, K)
        global_state    = batch['global_state']     # (B, N, 17) — full team block per sample
        actions         = batch['actions']          # (B, 4)
        old_log_probs   = batch['old_log_probs']    # (B,)
        advantages      = batch['advantages']       # (B,)
        returns         = batch['returns']          # (B,)

        # ── Actor Loss ────────────────────────────────────────────────────────

        # Get log prob of the STORED actions under the CURRENT policy
        # (not sampling new actions — evaluating old ones under new distribution)
        new_log_probs, entropy = self.actor.evaluate_actions(
            local_obs, neighbor_states, actions, neighbor_mask,
            active_dims=self.active_action_dims
        )
        # new_log_probs : (B,)
        # entropy       : (B,)

        # Probability ratio r(theta) = pi_new / pi_old
        # In log space: log(r) = log_prob_new - log_prob_old
        # Then: r = exp(log_prob_new - log_prob_old)
        log_ratio = new_log_probs - old_log_probs
        ratio     = torch.exp(log_ratio)            # (B,)

        # Clipped surrogate objective
        # surr1: unclipped — ratio * advantage
        # surr2: clipped   — clamp ratio to [1-eps, 1+eps] * advantage
        surr1 = ratio * advantages                  # (B,)
        surr2 = torch.clamp(
            ratio,
            1.0 - self.clip_epsilon,
            1.0 + self.clip_epsilon
        ) * advantages                              # (B,)

        # Take the minimum — this is the "pessimistic" surrogate
        # If advantage > 0: we want to increase prob but not too much
        # If advantage < 0: we want to decrease prob but not too much
        actor_loss = torch.min(surr1, surr2).mean() # scalar

        # ── Critic Loss ───────────────────────────────────────────────────────

        # Critic is centralized: global_state is already (B, N, 17) — the full
        # team's state block for each sample's timestep (see RolloutBuffer.
        # get_minibatches). This matches exactly what the critic saw live
        # during rollout, so update-time and rollout-time predictions agree.
        values = self.critic(global_state).squeeze(-1)  # (B,)

        critic_loss = nn.functional.mse_loss(values, returns)        # scalar

        # ── Diagnostics ───────────────────────────────────────────────────────

        # Clip fraction — fraction of ratios that were clipped
        # High clip fraction means the policy is trying to change too fast
        with torch.no_grad():
            clip_frac = ((ratio - 1.0).abs() > self.clip_epsilon).float().mean()

            # Approximate KL divergence — monitors how much policy changed
            # KL ≈ mean((ratio - 1) - log_ratio), always non-negative
            approx_kl = ((ratio - 1) - log_ratio).mean()

        return actor_loss, critic_loss, entropy.mean(), clip_frac, approx_kl

    def get_lr(self) -> Dict[str, float]:
        """Return current learning rates for logging."""
        return {
            'actor_lr':  self.actor_optimizer.param_groups[0]['lr'],
            'critic_lr': self.critic_optimizer.param_groups[0]['lr'],
        }
