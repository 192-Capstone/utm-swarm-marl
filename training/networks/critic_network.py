"""
critic_network.py
-----------------
Centralized Critic Network for MAPPO drone swarm UTM system.

Used ONLY during training. Completely discarded at deployment.
Its sole job is to estimate V(s) — how good is the overall team situation —
giving the Actor better gradient signal than it could get from local info alone.

Architecture:
    1. Per-Drone Encoder (shared weights) — encodes each drone's 17-dim state
       into a 128-dim embedding independently
       Linear(17, 128) → Tanh → Linear(128, 128) → Tanh

    2. Mean Pooling — aggregates all N drone embeddings into one 128-dim vector
       Makes the critic permutation-invariant and N-agnostic

    3. MLP Value Head — takes pooled embedding → scalar value estimate
       Linear(128, 256) → Tanh → Linear(256, 256) → Tanh → Linear(256, 1)

Input:
    global_state : (batch_size, N, 17) — per-drone state for all N drones
                   Each drone's 17 values:
                     absolute position    (3) — x, y, z
                     absolute velocity    (3) — vx, vy, vz
                     orientation          (3) — roll, pitch, yaw
                     goal position        (3) — gx, gy, gz
                     5-quadrant LiDAR     (5) — front, back, left, right, vertical
                   Stage 1: LiDAR slots (last 5) are zero-padded — no obstacles exist

Output:
    value        : (batch_size, 1) — scalar V(s) estimate

Why per-drone encoding + mean pooling over flat vector:
    - Input layer is always Linear(17, 128) regardless of N
      No rebuilding needed when scaling from 3 to 20+ drones
    - Permutation invariant — drone ordering doesn't matter
      mean([drone1, drone2, drone3]) == mean([drone3, drone1, drone2])
    - Parameter efficient — encoder sees one drone at a time
      Linear(17,128) vs Linear(N*17, 256) which explodes with N
    - Separation of concerns:
      Encoder = what does one drone's state mean?
      MLP     = what does the overall team situation mean?

Why discarded at deployment:
    - Requires global state (all drone positions) — doesn't exist at runtime
    - Runtime actors use only local observations
    - Critic's job ends when training ends
"""

import torch
import torch.nn as nn
import yaml
import os


class DroneStateEncoder(nn.Module):
    """
    Shared per-drone encoder. Same weights applied to every drone's state.

    Two-stage encoding:
        17 → 128 → 128

    First layer learns basic drone state features (position, velocity patterns).
    Second layer refines those into a rich 128-dim embedding.

    Gradual expansion is more stable than jumping 17 → 256 directly.
    """

    def __init__(self, drone_state_dim: int, embed_dim: int):
        """
        Args:
            drone_state_dim : dimension of one drone's state vector (17)
            embed_dim       : output embedding dimension (128)
        """
        super(DroneStateEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(drone_state_dim, embed_dim),   # 17 → 128
            nn.Tanh(),
            nn.Linear(embed_dim, embed_dim),          # 128 → 128
            nn.Tanh()
        )

    def forward(self, drone_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            drone_states : (batch_size, N, 17) — all drone states

        Returns:
            embeddings   : (batch_size, N, 128) — per-drone embeddings
        """
        batch_size, N, state_dim = drone_states.shape

        # Flatten batch and drone dims for efficient batched linear
        flat      = drone_states.reshape(batch_size * N, state_dim)  # (B*N, 17)
        encoded   = self.encoder(flat)                                # (B*N, 128)
        embeddings = encoded.reshape(batch_size, N, -1)              # (B, N, 128)

        return embeddings


class CriticNetwork(nn.Module):
    """
    Centralized Critic — estimates V(s) from global swarm state.

    Full pipeline:
        global_state (B, N, 17)
            ↓
        DroneStateEncoder (shared weights) → (B, N, 128)
            ↓
        Mean Pool over N dim → (B, 128)
            ↓
        MLP: 128 → 256 → 256 → 1
            ↓
        value estimate (B, 1)

    Training only — never deployed to drones.
    """

    def __init__(self, config: dict):
        """
        Args:
            config : loaded default_config.yaml as dict
        """
        super(CriticNetwork, self).__init__()

        # --- Load dimensions from config ---
        self.n_agents        = config['environment']['n_agents']      # 3
        self.drone_state_dim = 17      # per-drone global state dim
                                       # Stage 1: last 5 (LiDAR) are zero-padded
        self.embed_dim       = 128     # per-drone embedding dimension
        self.hidden_dim      = config['network']['hidden_dim']        # 256
        self.n_hidden_layers = config['network']['n_hidden_layers']   # 2

        # --- Per-Drone Encoder ---
        # Shared weights — same encoder runs on every drone independently
        self.drone_encoder = DroneStateEncoder(
            drone_state_dim=self.drone_state_dim,
            embed_dim=self.embed_dim
        )

        # --- MLP Value Head ---
        # Input: mean-pooled 128-dim team embedding
        # Cross-drone reasoning happens here after pooling
        # 128 → 256 → 256 → 1
        layers    = []
        input_dim = self.embed_dim                 # 128 after mean pooling

        for _ in range(self.n_hidden_layers):
            layers.append(nn.Linear(input_dim, self.hidden_dim))
            layers.append(nn.Tanh())
            input_dim = self.hidden_dim

        self.mlp = nn.Sequential(*layers)

        # Value output — single scalar
        self.value_head = nn.Linear(self.hidden_dim, 1)

        # --- Initialize weights ---
        self._initialize_weights()

    def _initialize_weights(self):
        """
        Orthogonal initialization — standard for PPO networks.
        Prevents vanishing/exploding gradients at training start.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)
                nn.init.zeros_(module.bias)

        # Value head — standard gain
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)

    def forward(self, global_state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass — estimates how good the overall team situation is.

        Args:
            global_state : (batch_size, N, 17) — per-drone states for all N drones
                           Caller reshapes from flat (B, N*17) to (B, N, 17)
                           before passing in — or pass shaped directly.

                           Stage 1: last 5 values per drone (LiDAR) = 0.0
                           Stage 2+: all 17 values populated

        Returns:
            value        : (batch_size, 1) — scalar value estimate V(s)
        """

        # --- Step 1: Encode each drone independently (shared weights) ---
        embeddings = self.drone_encoder(global_state)    # (B, N, 128)

        # --- Step 2: Mean pool over all drone embeddings ---
        # Permutation invariant — drone ordering has no effect
        # N-agnostic — works for any number of drones
        team_embedding = embeddings.mean(dim=1)          # (B, 128)

        # --- Step 3: MLP value head ---
        features = self.mlp(team_embedding)              # (B, 256)
        value    = self.value_head(features)             # (B, 1)

        return value

    def get_value(self, global_state: torch.Tensor) -> torch.Tensor:
        """
        Convenience method — same as forward().
        Named explicitly for clarity when called by MAPPOTrainer.

        Args:
            global_state : (batch_size, N, 17)

        Returns:
            value        : (batch_size, 1)
        """
        return self.forward(global_state)


# def build_global_state_tensor(raw_states: dict,
#                                n_agents: int,
#                                curriculum_stage: int) -> torch.Tensor:
#     """
#     Utility — builds the (1, N, 17) global state tensor from raw Webots state.
#     Called by ObservationProcessor.build_global_state().

#     Args:
#         raw_states       : dict from WebotsAdapter.get_raw_state()
#         n_agents         : number of drones
#         curriculum_stage : 1 = zero-pad LiDAR, 2+ = full 17 values

#     Returns:
#         global_state     : (1, N, 17) tensor ready for CriticNetwork
#     """
#     drone_states = []

#     for i in range(n_agents):
#         drone = raw_states[f'drone_{i}']

#         state_vec = (
#             drone['position'] +           # [x, y, z]          3 values
#             drone['velocity'] +           # [vx, vy, vz]        3 values
#             drone['orientation'] +        # [roll, pitch, yaw]  3 values
#             drone['goal_position']        # [gx, gy, gz]        3 values
#         )                                 # 12 values

#         # LiDAR quadrant summary — zero in Stage 1, populated Stage 2+
#         if curriculum_stage >= 2:
#             lidar = drone['lidar_quadrants']    # [F, B, L, R, V] 5 values
#         else:
#             lidar = [0.0, 0.0, 0.0, 0.0, 0.0]  # Stage 1 zero-pad

#         state_vec += lidar                      # 17 values total

#         assert len(state_vec) == 17, \
#             f"Drone state has {len(state_vec)} values, expected 17"

#         drone_states.append(state_vec)

#     global_state = torch.tensor(drone_states, dtype=torch.float32)
#     global_state = global_state.unsqueeze(0)    # (1, N, 17)

#     return global_state


def load_critic(config_path: str,
                checkpoint_path: str = None) -> CriticNetwork:
    """
    Utility — loads Critic from config and optionally a checkpoint.
    Used by MAPPOTrainer at training start.

    Note: Critic is NEVER used at deployment — training only.

    Args:
        config_path     : path to default_config.yaml
        checkpoint_path : path to .pth checkpoint (optional)

    Returns:
        critic          : CriticNetwork ready for training
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    critic = CriticNetwork(config)

    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'critic_state_dict' in checkpoint:
            critic.load_state_dict(checkpoint['critic_state_dict'])
        else:
            critic.load_state_dict(checkpoint)
        print(f"Critic loaded from checkpoint: {checkpoint_path}")

    return critic


if __name__ == "__main__":
    """
    Unit test — run directly to verify network works correctly.
    python critic_network.py
    """
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        'configs', 'default_config.yaml'
    )
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    print("=" * 60)
    print("CriticNetwork Unit Test")
    print("=" * 60)

    # --- Build network ---
    critic = CriticNetwork(config)
    print(f"\nNetwork architecture:\n{critic}")

    # Parameter count breakdown
    total_params   = sum(p.numel() for p in critic.parameters())
    encoder_params = sum(p.numel() for p in critic.drone_encoder.parameters())
    mlp_params     = (sum(p.numel() for p in critic.mlp.parameters()) +
                      sum(p.numel() for p in critic.value_head.parameters()))

    print(f"\nParameter breakdown:")
    print(f"  DroneStateEncoder : {encoder_params:,}")
    print(f"  MLP + Value Head  : {mlp_params:,}")
    print(f"  Total             : {total_params:,}")

    batch_size = 4
    N = config['environment']['n_agents']   # 3

    # --- Test 1: Basic forward pass ---
    global_state = torch.randn(batch_size, N, 17)
    value = critic.forward(global_state)

    print(f"\nTest 1 — Basic forward pass:")
    print(f"  global_state shape : {global_state.shape}  "
          f"expected: ({batch_size}, {N}, 17)")
    print(f"  value shape        : {value.shape}  "
          f"expected: ({batch_size}, 1)")
    assert value.shape == (batch_size, 1), "value shape mismatch"
    print(f"  shape check        : ✓")

    # --- Test 2: Stage 1 zero-padded LiDAR ---
    gs_stage1 = torch.randn(batch_size, N, 17)
    gs_stage1[:, :, 12:] = 0.0        # zero out last 5 (LiDAR quadrants)
    val_stage1 = critic.forward(gs_stage1)
    assert not torch.isnan(val_stage1).any(), "NaN with zero-padded LiDAR"
    print(f"\nTest 2 — Stage 1 zero-padded LiDAR:")
    print(f"  value shape     : {val_stage1.shape}  ✓")
    print(f"  no NaN detected : ✓")

    # --- Test 3: Permutation invariance ---
    gs_orig    = torch.randn(1, N, 17)
    idx        = torch.randperm(N)
    gs_shuf    = gs_orig[:, idx, :]
    val_orig   = critic.forward(gs_orig)
    val_shuf   = critic.forward(gs_shuf)
    assert torch.allclose(val_orig, val_shuf, atol=1e-5), \
        "Permutation invariance violated"
    print(f"\nTest 3 — Permutation invariance:")
    print(f"  original value  : {val_orig.item():.6f}")
    print(f"  shuffled value  : {val_shuf.item():.6f}")
    print(f"  values match    : ✓ (mean pooling is order-independent)")

    # --- Test 4: N-agnostic ---
    print(f"\nTest 4 — N-agnostic (same network, different swarm sizes):")
    for n in [2, 3, 5, 10, 20]:
        gs = torch.randn(batch_size, n, 17)
        v  = critic.forward(gs)
        assert v.shape == (batch_size, 1), f"Failed for N={n}"
        print(f"  N={n:2d} → value shape: {v.shape}  ✓")

    # --- Test 5: get_value matches forward ---
    val_v2 = critic.get_value(global_state)
    assert torch.allclose(value, val_v2), "get_value != forward()"
    print(f"\nTest 5 — get_value == forward()  : ✓")

    # --- Test 6: Gradient flow ---
    gs_grad = torch.randn(batch_size, N, 17, requires_grad=True)
    v_grad  = critic.forward(gs_grad)
    v_grad.sum().backward()
    assert gs_grad.grad is not None, "No gradient flowing to input"
    print(f"Test 6 — Gradient flows to input : ✓")

    print("\n" + "=" * 60)
    print("All tests passed ✓")
    print("=" * 60)

    print(f"\nArchitecture summary:")
    print(f"  Input   : (batch, N, 17) — N drones, 17 values each")
    print(f"  Encode  : Linear(17,128) → Tanh → Linear(128,128) → Tanh  [shared]")
    print(f"  Pool    : mean over N → (batch, 128)")
    print(f"  MLP     : 128 → 256 → 256 → 1")
    print(f"  Output  : (batch, 1) — scalar value estimate V(s)")
    print(f"\n  N-agnostic     : ✓")
    print(f"  Perm-invariant : ✓")
    print(f"  Training only  : critic discarded at deployment")