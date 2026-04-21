"""
rollout_buffer.py
-----------------
Stores transitions from all agents during episode rollout and computes
Generalized Advantage Estimation (GAE) after episode completion.

The buffer collects experience from ALL agents simultaneously.
After an episode ends, GAE is computed using stored value estimates
and actual rewards, producing per-agent advantages for PPO updates.

Key design:
    - Stores (local_obs, neighbor_states, neighbor_mask, actions,
               log_probs, rewards, dones, values) per timestep per agent
    - GAE computed post-episode: A_t = delta_t + gamma*lambda*A_{t+1}
    - Returns minibatches for PPO's multiple update epochs
"""

import torch
import numpy as np
from typing import Dict, List, Optional


class RolloutBuffer:
    """
    Experience replay buffer for MAPPO.

    Stores complete episode transitions for all N agents simultaneously.
    After episode completion, computes GAE advantages and returns
    shuffled minibatches for PPO optimization epochs.

    Buffer layout (per timestep):
        local_obs       : (N, 27)       fixed local observation per agent
        neighbor_states : (N, K, 6)     neighbor state matrix per agent
        neighbor_mask   : (N, K)        padding mask per agent
        global_state    : (N, 17)       full swarm state for critic
        actions         : (N, 4)        actions taken
        log_probs       : (N,)          log probability of actions taken
        rewards         : (N,)          per-agent rewards
        dones           : (N,)          episode termination flags
        values          : (N,)          critic value estimates C_est
    """

    def __init__(self, config: dict, device: torch.device):
        self.config     = config
        self.device     = device

        # Dimensions from config
        self.n_agents        = config['environment']['n_agents']           # 3
        self.buffer_size     = config['training']['buffer_size']           # 2048
        self.local_obs_dim   = config['observation']['local_obs_dim']      # 27
        self.neighbor_obs_dim= config['observation']['neighbor_obs_dim']   # 6
        self.k_neighbors_max = config['observation']['k_neighbors_max']    # 5
        self.action_dim      = config['network']['action_dim']             # 4
        self.gamma           = config['training']['gamma']                 # 0.99
        self.lambda_gae      = config['training']['lambda_gae']            # 0.95

        # Global state dim depends on curriculum stage
        # Stage 1: N*12, Stage 2+: N*17 — stored per drone (17 always, zeros in Stage 1)
        self.drone_state_dim = 17

        self._init_storage()

    def _init_storage(self):
        """Pre-allocate all storage tensors on device."""
        B = self.buffer_size
        N = self.n_agents
        K = self.k_neighbors_max

        # Per-step storage — pre-allocated for efficiency
        self.local_obs       = torch.zeros(B, N, self.local_obs_dim,    device=self.device)
        self.neighbor_states = torch.zeros(B, N, K, self.neighbor_obs_dim, device=self.device)
        self.neighbor_mask   = torch.ones( B, N, K, dtype=torch.bool,   device=self.device)
        self.global_state    = torch.zeros(B, N, self.drone_state_dim,  device=self.device)
        self.actions         = torch.zeros(B, N, self.action_dim,       device=self.device)
        self.log_probs       = torch.zeros(B, N,                        device=self.device)
        self.rewards         = torch.zeros(B, N,                        device=self.device)
        self.dones           = torch.zeros(B, N,                        device=self.device)
        self.values          = torch.zeros(B, N,                        device=self.device)

        # Computed post-episode
        self.advantages      = torch.zeros(B, N,                        device=self.device)
        self.returns         = torch.zeros(B, N,                        device=self.device)

        self.ptr   = 0       # current write position
        self.size  = 0       # number of valid entries

    def store(
        self,
        local_obs:       torch.Tensor,   # (N, 27)
        neighbor_states: torch.Tensor,   # (N, K, 6)
        neighbor_mask:   torch.Tensor,   # (N, K) bool
        global_state:    torch.Tensor,   # (N, 17)
        actions:         torch.Tensor,   # (N, 4)
        log_probs:       torch.Tensor,   # (N,)
        rewards:         torch.Tensor,   # (N,)
        dones:           torch.Tensor,   # (N,)
        values:          torch.Tensor,   # (N,)
    ):
        """
        Store one timestep of experience for all N agents.
        Called by MAPPOTrainer on every environment step.
        """
        assert self.ptr < self.buffer_size, \
            f"Buffer overflow at ptr={self.ptr}. Call compute_gae() before storing more."

        # Fail fast on shape contract violations to avoid hard-to-debug training errors.
        assert local_obs.shape == (self.n_agents, self.local_obs_dim), \
            f"local_obs shape {tuple(local_obs.shape)} != ({self.n_agents}, {self.local_obs_dim})"
        assert neighbor_states.shape == (self.n_agents, self.k_neighbors_max, self.neighbor_obs_dim), \
            f"neighbor_states shape {tuple(neighbor_states.shape)} != ({self.n_agents}, {self.k_neighbors_max}, {self.neighbor_obs_dim})"
        assert neighbor_mask.shape == (self.n_agents, self.k_neighbors_max), \
            f"neighbor_mask shape {tuple(neighbor_mask.shape)} != ({self.n_agents}, {self.k_neighbors_max})"
        assert global_state.shape == (self.n_agents, self.drone_state_dim), \
            f"global_state shape {tuple(global_state.shape)} != ({self.n_agents}, {self.drone_state_dim})"
        assert actions.shape == (self.n_agents, self.action_dim), \
            f"actions shape {tuple(actions.shape)} != ({self.n_agents}, {self.action_dim})"
        assert log_probs.shape == (self.n_agents,), \
            f"log_probs shape {tuple(log_probs.shape)} != ({self.n_agents},)"
        assert rewards.shape == (self.n_agents,), \
            f"rewards shape {tuple(rewards.shape)} != ({self.n_agents},)"
        assert dones.shape == (self.n_agents,), \
            f"dones shape {tuple(dones.shape)} != ({self.n_agents},)"
        assert values.shape == (self.n_agents,), \
            f"values shape {tuple(values.shape)} != ({self.n_agents},)"

        idx = self.ptr

        self.local_obs[idx]       = local_obs.detach()
        self.neighbor_states[idx] = neighbor_states.detach()
        self.neighbor_mask[idx]   = neighbor_mask.detach()
        self.global_state[idx]    = global_state.detach()
        self.actions[idx]         = actions.detach()
        self.log_probs[idx]       = log_probs.detach()
        self.rewards[idx]         = rewards.detach()
        self.dones[idx]           = dones.detach()
        self.values[idx]          = values.detach()

        self.ptr  += 1
        self.size  = min(self.size + 1, self.buffer_size)

    def compute_gae(self, last_values: torch.Tensor):
        """
        Compute Generalized Advantage Estimation after episode completion.

        GAE formula (backwards pass):
            delta_t = r_t + gamma * V(s_{t+1}) * (1 - done_t) - V(s_t)
            A_t     = delta_t + gamma * lambda * (1 - done_t) * A_{t+1}

        Args:
            last_values : (N,) — critic value estimate for the final state
                          (V(s_T) — used as bootstrap value for incomplete episodes)

        The backwards computation is key:
            - We start from the last timestep and work backwards
            - done flags zero out future advantage at episode boundaries
            - GAE exponentially discounts future TD errors (lambda controls decay)
        """
        T = self.ptr   # number of valid timesteps

        gae = torch.zeros(self.n_agents, device=self.device)  # running GAE accumulator

        for t in reversed(range(T)):
            # Bootstrap: use last_values for the step after the buffer ends
            if t == T - 1:
                next_values = last_values                          # (N,)
                next_done   = self.dones[t]                        # (N,)
            else:
                next_values = self.values[t + 1]                   # (N,)
                next_done   = self.dones[t]                        # (N,)

            # One-step TD error
            # delta = r_t + gamma * V(s_{t+1}) * (1 - done) - V(s_t)
            delta = (
                self.rewards[t]
                + self.gamma * next_values * (1.0 - next_done)
                - self.values[t]
            )

            # Recursive GAE accumulation
            # A_t = delta_t + gamma * lambda * (1 - done_t) * A_{t+1}
            gae = delta + self.gamma * self.lambda_gae * (1.0 - next_done) * gae

            self.advantages[t] = gae

        # Returns = advantages + value estimates (used for critic loss target)
        self.returns[:T] = self.advantages[:T] + self.values[:T]

        # Normalize advantages across the buffer for training stability
        # Zero mean, unit variance — standard PPO practice
        adv = self.advantages[:T].reshape(-1)     # flatten across time and agents
        self.advantages[:T] = (
            (self.advantages[:T] - adv.mean()) / (adv.std() + 1e-8)
        )

    def get_minibatches(self, batch_size: int):
        """
        Yield shuffled minibatches for PPO's multiple update epochs.
        Called by PPOOptimizer for each training epoch.

        Yields dicts with:
            local_obs, neighbor_states, neighbor_mask, global_state,
            actions, old_log_probs, advantages, returns
        """
        T = self.ptr
        N = self.n_agents

        # Flatten time and agent dimensions: (T, N, ...) -> (T*N, ...)
        lo  = self.local_obs[:T].reshape(T * N, -1)
        ns  = self.neighbor_states[:T].reshape(T * N, self.k_neighbors_max, self.neighbor_obs_dim)
        nm  = self.neighbor_mask[:T].reshape(T * N, self.k_neighbors_max)
        gs  = self.global_state[:T].reshape(T * N, self.drone_state_dim)
        act = self.actions[:T].reshape(T * N, self.action_dim)
        lp  = self.log_probs[:T].reshape(T * N)
        adv = self.advantages[:T].reshape(T * N)
        ret = self.returns[:T].reshape(T * N)

        total = T * N
        indices = torch.randperm(total, device=self.device)

        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            idx = indices[start:end]

            yield {
                'local_obs':       lo[idx],
                'neighbor_states': ns[idx],
                'neighbor_mask':   nm[idx],
                'global_state':    gs[idx],
                'actions':         act[idx],
                'old_log_probs':   lp[idx],
                'advantages':      adv[idx],
                'returns':         ret[idx],
            }

    def reset(self):
        """Clear buffer after PPO update. Called by MAPPOTrainer after each update."""
        self.ptr  = 0
        self.size = 0
        # Zero out storage (not strictly necessary but prevents stale data bugs)
        self._init_storage()

    def is_full(self) -> bool:
        """True when buffer has collected buffer_size timesteps."""
        return self.ptr >= self.buffer_size

    def __len__(self) -> int:
        return self.size


if __name__ == "__main__":
    """Unit test — run directly: python rollout_buffer.py"""
    import yaml, os

    config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'default_config.yaml')
    with open(config_path) as f:
        config = yaml.safe_load(f)

    device = torch.device('cpu')
    buf    = RolloutBuffer(config, device)

    N  = config['environment']['n_agents']     # 3
    K  = config['observation']['k_neighbors_max']  # 5
    BS = config['training']['buffer_size']     # 2048

    print("=" * 60)
    print("RolloutBuffer Unit Test")
    print("=" * 60)

    # Fill buffer with random transitions
    print(f"\nFilling buffer with {BS} timesteps for {N} agents...")
    for t in range(BS):
        buf.store(
            local_obs       = torch.randn(N, config['observation']['local_obs_dim']),
            neighbor_states = torch.randn(N, K, 6),
            neighbor_mask   = torch.zeros(N, K, dtype=torch.bool),
            global_state    = torch.randn(N, 17),
            actions         = torch.randn(N, 4),
            log_probs       = torch.randn(N),
            rewards         = torch.rand(N),
            dones           = torch.zeros(N),
            values          = torch.randn(N),
        )

    print(f"Buffer size: {len(buf)} / {BS}  ✓")
    assert buf.is_full(), "Buffer should be full"
    print(f"is_full(): {buf.is_full()}  ✓")

    # Compute GAE
    last_values = torch.randn(N)
    buf.compute_gae(last_values)
    print(f"\nGAE computed:")
    print(f"  advantages shape : {buf.advantages[:BS].shape}  expected: ({BS}, {N})")
    print(f"  returns shape    : {buf.returns[:BS].shape}  expected: ({BS}, {N})")
    assert not torch.isnan(buf.advantages).any(), "NaN in advantages"
    assert not torch.isnan(buf.returns).any(),    "NaN in returns"
    print(f"  no NaN in advantages  ✓")
    print(f"  no NaN in returns     ✓")

    # Advantage normalization
    adv_flat = buf.advantages[:BS].reshape(-1)
    print(f"  advantage mean   : {adv_flat.mean():.4f}  expected: ~0.0")
    print(f"  advantage std    : {adv_flat.std():.4f}   expected: ~1.0")

    # Minibatch generation
    batch_size = config['training']['batch_size']  # 256
    batches = list(buf.get_minibatches(batch_size))
    print(f"\nMinibatch generation:")
    print(f"  batch_size          : {batch_size}")
    print(f"  total samples       : {BS * N}")
    print(f"  num batches         : {len(batches)}")
    b = batches[0]
    print(f"  first batch shapes  :")
    for k, v in b.items():
        print(f"    {k:20s}: {v.shape}")

    # Reset
    buf.reset()
    assert len(buf) == 0, "Buffer should be empty after reset"
    print(f"\nBuffer reset: len={len(buf)}  ✓")

    print("\n" + "=" * 60)
    print("All tests passed ✓")
    print("=" * 60)