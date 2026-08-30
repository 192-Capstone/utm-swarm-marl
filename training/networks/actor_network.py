"""
actor_network.py
----------------
Decentralized Actor Network for MAPPO drone swarm UTM system.

Architecture:
    1. Attention Module  — encodes variable K neighbors into fixed 32-dim context vector
    2. MLP Policy Head   — takes concat(local_obs, context) → action distribution

Input:
    local_obs       : (batch_size, 27)    — fixed local observation per drone
    neighbor_states : (batch_size, K, 6)  — K nearest neighbors, each with 6 values
                                            (relative pos (3) + relative vel (3))

Output:
    action_mean     : (batch_size, 4)     — mean of Gaussian action distribution
    action_log_std  : (batch_size, 4)     — log std of Gaussian action distribution

Action dimensions (always 4 — yaw masked externally by SwarmEnv in Stage 1-2):
    [0] Vx       — East-West velocity
    [1] Vy       — North-South velocity
    [2] Vz       — Altitude velocity
    [3] Yaw Rate — Rotation rate (active Stage 3+, masked to 0.0 in Stage 1-2)

This network is what gets saved to best_policy.pth and deployed to each drone.
At deployment, only this network runs — no other components needed for inference.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import os


class NeighborAttention(nn.Module):
    """
    Single-head dot-product attention module for neighbor encoding.

    Takes variable K neighbor states and produces a fixed-size context vector.
    This is what allows the architecture to scale from small swarms (KNN)
    to large swarms (H3 indexing) without any architectural changes —
    regardless of how many neighbors are found, output is always context_dim.

    How it works:
        1. Project own state into a query vector  — "what am I looking for?"
        2. Project each neighbor into a key vector — "what does this neighbor represent?"
        3. Score = dot product(query, key_i)      — how relevant is neighbor i?
        4. Weights = softmax(scores)              — normalize to sum to 1
        5. Context = weighted sum of value projections of neighbors

    The three projection matrices (W_q, W_k, W_v) are learned during training.
    Nobody manually assigns weights — the network discovers that nearby drones
    heading toward you deserve more attention than far away drones moving away.
    """

    def __init__(self, neighbor_obs_dim: int, context_dim: int):
        """
        Args:
            neighbor_obs_dim : dimension of each neighbor's state vector (6)
            context_dim      : output dimension of attention module (32)
        """
        super(NeighborAttention, self).__init__()

        self.context_dim = context_dim

        # Query projection — maps own state context to query space
        # We use context_dim as the query/key dimension
        self.query_proj = nn.Linear(context_dim, context_dim)

        # Key projection — maps each neighbor state to key space
        self.key_proj = nn.Linear(neighbor_obs_dim, context_dim)

        # Value projection — maps each neighbor state to value space
        self.value_proj = nn.Linear(neighbor_obs_dim, context_dim)

        # Scale factor for dot product attention — prevents vanishing gradients
        # with large context_dim (standard in transformer attention)
        self.scale = context_dim ** 0.5

    def forward(self, query: torch.Tensor, neighbor_states: torch.Tensor,
                neighbor_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            query           : (batch_size, context_dim)  — projected own state as query
            neighbor_states : (batch_size, K, 6)         — K neighbor states
            neighbor_mask   : (batch_size, K)            — True for padded/empty neighbors
                              If fewer than K_max neighbors exist, padded slots are masked
                              so they don't contribute to the context vector

        Returns:
            context         : (batch_size, context_dim=32) — fixed size neighbor summary
        """
        batch_size, K, _ = neighbor_states.shape

        # Project query: (batch_size, context_dim)
        q = self.query_proj(query)                          # (batch_size, context_dim)
        q = q.unsqueeze(1)                                  # (batch_size, 1, context_dim)

        # Project keys and values for all neighbors
        k = self.key_proj(neighbor_states)                  # (batch_size, K, context_dim)
        v = self.value_proj(neighbor_states)                # (batch_size, K, context_dim)

        # Compute attention scores: dot product of query with each key
        scores = torch.bmm(q, k.transpose(1, 2)) / self.scale  # (batch_size, 1, K)
        scores = scores.squeeze(1)                              # (batch_size, K)

        # Apply mask for padded neighbors (if any slots are empty)
        # Masked slots get -inf so softmax gives them weight 0
        if neighbor_mask is not None:
            scores = scores.masked_fill(neighbor_mask, float('-inf'))

        # Normalize scores to weights that sum to 1
        weights = F.softmax(scores, dim=-1)                 # (batch_size, K)

        # Handle edge case: if ALL neighbors are masked (no neighbors at all)
        # softmax of all -inf produces NaN — replace with zeros
        weights = torch.nan_to_num(weights, nan=0.0)

        # Weighted sum of values → context vector
        weights = weights.unsqueeze(1)                      # (batch_size, 1, K)
        context = torch.bmm(weights, v)                     # (batch_size, 1, context_dim)
        context = context.squeeze(1)                        # (batch_size, context_dim)

        return context


class ActorNetwork(nn.Module):
    """
    Decentralized Actor Network — runs on each drone independently at deployment.

    Full pipeline:
        neighbor_states (K×6) → NeighborAttention → context (32)
        local_obs (27) + context (32) → concat (59) → MLP → action_mean (4), action_log_std (4)

    This network is self-contained — it handles its own neighbor encoding internally.
    At deployment, feed raw sensor data directly to this network.
    No external preprocessing needed for inference.
    """

    def __init__(self, config: dict):
        """
        Args:
            config : loaded default_config.yaml as dict
        """
        super(ActorNetwork, self).__init__()

        # --- Load dimensions from config ---
        self.local_obs_dim      = config['observation']['local_obs_dim']        # 27
        self.neighbor_obs_dim   = config['observation']['neighbor_obs_dim']     # 6
        self.neighbor_context_dim = config['observation']['neighbor_context_dim']  # 32
        self.total_obs_dim      = config['observation']['total_obs_dim']        # 59
        self.action_dim         = config['network']['action_dim']               # 4
        self.hidden_dim         = config['network']['hidden_dim']               # 256
        self.n_hidden_layers    = config['network']['n_hidden_layers']          # 2

        # --- Attention Module ---
        # Encodes variable K neighbors → fixed 32-dim context
        # Query is derived from a projection of local_obs
        self.local_proj = nn.Linear(self.local_obs_dim, self.neighbor_context_dim)
        self.attention  = NeighborAttention(
            neighbor_obs_dim=self.neighbor_obs_dim,
            context_dim=self.neighbor_context_dim
        )

        # --- MLP Policy Head ---
        # Input: concat(local_obs, context) = 27 + 32 = 59
        # Hidden: 256 → 256
        # Output: action_mean (4)
        layers = []
        input_dim = self.total_obs_dim                      # 59

        for _ in range(self.n_hidden_layers):
            layers.append(nn.Linear(input_dim, self.hidden_dim))
            layers.append(nn.Tanh())                        # tanh for bounded continuous control
            input_dim = self.hidden_dim

        self.mlp = nn.Sequential(*layers)

        # Output head — produces action mean
        self.action_mean_head = nn.Linear(self.hidden_dim, self.action_dim)

        # Log standard deviation — learned parameter, not a network output
        # One log_std value per action dimension, shared across all states
        # Initialized to 0 → std = exp(0) = 1.0 → reasonable initial exploration
        self.action_log_std = nn.Parameter(torch.zeros(self.action_dim))

        # Clamp bounds for log_std. Both bounds are mutable — MAPPOTrainer
        # can anneal them over training via set_log_std_ceiling/_floor to
        # force exploration noise down on a schedule, independent of what
        # the learned parameter's own gradient does. Without this, log_std
        # can sit near 0 (std≈1.0) for the whole run because reducing
        # entropy doesn't reliably reduce PPO's loss when successful settle
        # rollouts are rare — the sole gradient signal for shrinking log_std
        # is too weak to compete with high-noise runs.
        #
        # Default floor -3.0 (std≈0.05) is the level at which 31-step
        # settling first becomes probabilistically feasible during
        # stochastic training (see forward() docstring). But a second,
        # optional squeeze BELOW -3.0 matters too: with std≈0.05, a seed can
        # learn a mean action that stops just OUTSIDE goal_threshold and
        # relies on residual sampling noise to occasionally nudge it across
        # the boundary — a policy that only works stochastically, not
        # deterministically (see run honest-thunder-47, seed 1: 93% training
        # success, 0% deterministic eval, mean_min_dist≈0.53m). Squeezing
        # the floor further after settlement first emerges removes that
        # noise crutch and forces PPO to place the MEAN action inside the
        # goal with real margin.
        self.log_std_floor   = -3.0
        self.log_std_ceiling = 0.5

        # --- Initialize weights ---
        self._initialize_weights()

    def set_log_std_floor(self, floor: float):
        """External hook for a second exploration-tightening phase (see
        MAPPOTrainer._update_exploration_schedule). Symmetric to
        set_log_std_ceiling — only changes the clamp floor applied in
        forward(), never the learned parameter itself."""
        self.log_std_floor = float(floor)

    def set_log_std_ceiling(self, ceiling: float):
        """External hook for an exploration-annealing schedule (see
        MAPPOTrainer._update_exploration_schedule). Does not touch the
        learned action_log_std parameter itself — only the clamp ceiling
        applied to it in forward(), so annealing is fully reversible and
        never destroys the learned value outright."""
        self.log_std_ceiling = float(ceiling)

    def _initialize_weights(self):
        """
        Orthogonal initialization — standard for PPO networks.
        Prevents vanishing/exploding gradients at the start of training.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)
                nn.init.zeros_(module.bias)

        # Output layer uses smaller gain — keeps initial actions close to zero
        nn.init.orthogonal_(self.action_mean_head.weight, gain=0.01)

    def forward(self, local_obs: torch.Tensor,
                neighbor_states: torch.Tensor,
                neighbor_mask: torch.Tensor = None):
        """
        Forward pass — produces action distribution parameters.

        Args:
            local_obs       : (batch_size, 27)   — own state + goal + lidar
            neighbor_states : (batch_size, K, 6) — K neighbor states from KNN/H3
            neighbor_mask   : (batch_size, K)    — True for padded empty neighbor slots

        Returns:
            action_mean     : (batch_size, 4)    — mean of action Gaussian
            action_log_std  : (batch_size, 4)    — log std of action Gaussian
                              (same log_std for all states — learned scalar per dim)

        Note on action_log_std:
            During training  → sample action from N(mean, exp(log_std)) for exploration
            During deployment → use mean directly for deterministic behavior
        """
        batch_size = local_obs.shape[0]

        # --- Step 1: Project local_obs to query space for attention ---
        # The query represents "what kind of neighbor information am I looking for
        # given my current state?" — e.g. if I'm moving fast, I care more about
        # neighbors directly in my path
        query = torch.tanh(self.local_proj(local_obs))     # (batch_size, 32)

        # --- Step 2: Attention over neighbors → fixed context vector ---
        # Regardless of K (2 neighbors or 10 neighbors), output is always (batch_size, 32)
        context = self.attention(query, neighbor_states, neighbor_mask)
                                                            # (batch_size, 32)

        # --- Step 3: Concatenate local obs with neighbor context ---
        combined = torch.cat([local_obs, context], dim=-1) # (batch_size, 59)

        # --- Step 4: MLP policy head ---
        features = self.mlp(combined)                      # (batch_size, 256)

        # --- Step 5: Output action mean ---
        action_mean = self.action_mean_head(features)      # (batch_size, 4)

        # Clamp log_std to prevent numerical instability. Floor fixed at
        # -3.0 so 3D speed with zero-mean action ≈ 0.097 m/s — at this floor
        # P(speed<0.15) ≈ 97%, making 31-step settling ≈ 41% probable per
        # window (previous floor of -2.0 made this effectively impossible).
        # Ceiling is mutable (see set_log_std_ceiling) so training can anneal
        # exploration down over time instead of relying solely on the
        # learned parameter's own gradient, which is too weak a signal when
        # successful settle rollouts are rare during early training.
        clamped_log_std = torch.clamp(
            self.action_log_std, self.log_std_floor, self.log_std_ceiling
        )
        action_log_std = clamped_log_std.expand_as(action_mean)
                                                            # (batch_size, 4)

        return action_mean, action_log_std

    def get_action(self, local_obs: torch.Tensor,
                   neighbor_states: torch.Tensor,
                   neighbor_mask: torch.Tensor = None,
                   deterministic: bool = False,
                   active_dims: int = None):
        """
        Sample an action from the policy distribution.

        Args:
            local_obs       : (batch_size, 27)
            neighbor_states : (batch_size, K, 6)
            neighbor_mask   : (batch_size, K) optional
            deterministic   : if True, return mean (used at deployment/evaluation)
                              if False, sample from distribution (used during training)
            active_dims     : number of leading action dims to include in log_prob
                              (yaw is dim 3 — masked to 0.0 and causally inert in
                              Stage 1-2, so it must be excluded from the ratio/KL).
                              None = all action_dim dims.

        Returns:
            action          : (batch_size, 4)
            log_prob        : (batch_size,)   — log probability of sampled action
                              needed by PPO for computing the probability ratio r(θ)
        """
        action_mean, action_log_std = self.forward(
            local_obs, neighbor_states, neighbor_mask
        )

        action_std = torch.exp(action_log_std)

        # Create Gaussian distribution
        dist = torch.distributions.Normal(action_mean, action_std)

        if deterministic:
            # Deployment — use mean directly, no randomness
            action = action_mean
        else:
            # Training — sample for exploration
            action = dist.sample()

        # Log probability of this action — needed for PPO ratio computation
        # Sum across action dimensions (independent Gaussians)
        d = self.action_dim if active_dims is None else active_dims
        log_prob = dist.log_prob(action)[:, :d].sum(dim=-1)  # (batch_size,)

        return action, log_prob

    def evaluate_actions(self, local_obs: torch.Tensor,
                         neighbor_states: torch.Tensor,
                         actions: torch.Tensor,
                         neighbor_mask: torch.Tensor = None,
                         active_dims: int = None):
        """
        Evaluate log probability and entropy of given actions under current policy.
        Called by PPOOptimizer during the update step.

        During PPO update we need:
        - log_prob of the SAME actions under the NEW policy (for ratio r(θ))
        - entropy for the entropy bonus (encourages exploration)

        Args:
            local_obs       : (batch_size, 27)
            neighbor_states : (batch_size, K, 6)
            actions         : (batch_size, 4)  — actions that were taken during rollout
            neighbor_mask   : (batch_size, K) optional
            active_dims     : number of leading action dims to include (see get_action).
                              None = all action_dim dims.

        Returns:
            log_prob        : (batch_size,)    — log prob of actions under current policy
            entropy         : (batch_size,)    — entropy of current policy distribution
        """
        action_mean, action_log_std = self.forward(
            local_obs, neighbor_states, neighbor_mask
        )

        action_std = torch.exp(action_log_std)
        dist = torch.distributions.Normal(action_mean, action_std)

        d = self.action_dim if active_dims is None else active_dims
        log_prob = dist.log_prob(actions)[:, :d].sum(dim=-1)  # (batch_size,)
        entropy  = dist.entropy()[:, :d].sum(dim=-1)          # (batch_size,)

        return log_prob, entropy


def load_actor(config_path: str, checkpoint_path: str = None) -> ActorNetwork:
    """
    Utility function to load Actor from config and optionally a checkpoint.
    Used by ClientNode at deployment time.

    Args:
        config_path     : path to default_config.yaml
        checkpoint_path : path to saved .pth file (optional)

    Returns:
        actor           : ActorNetwork ready for inference
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    actor = ActorNetwork(config)

    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        # checkpoint may contain full training state — extract actor weights only
        if 'actor_state_dict' in checkpoint:
            actor.load_state_dict(checkpoint['actor_state_dict'])
        else:
            actor.load_state_dict(checkpoint)
        print(f"Actor loaded from checkpoint: {checkpoint_path}")

    return actor
