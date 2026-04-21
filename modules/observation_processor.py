"""
observation_processor.py
------------------------
Builds local observation tensors (59-dim per agent) and global state
tensors (N x 17) from raw Webots state dictionaries.

Responsibilities:
    1. Feature extraction  — parse raw_state dict into numpy arrays
    2. Neighbor selection  — brute-force KNN to find K nearest neighbors
    3. Local obs assembly  — [own_state(27)] + [attention_context(32)]
                             NOTE: attention context is computed by ActorNetwork
                             ObservationProcessor outputs the raw split:
                               local_fixed (N, 27) and neighbor_states (N, K, 6)
                             The Actor internally runs attention to produce (N, 59)
    4. Global state build  — (N, 17) per-drone tensor for CriticNetwork
    5. Normalization       — running mean/std on local_fixed components

Raw state format from WebotsAdapter.get_raw_state():
    {
      'drone_0': {
        'position':    [x, y, z],
        'velocity':    [vx, vy, vz],
        'orientation': [roll, pitch, yaw],
        'lidar':       [14 ray distances],
      },
      'drone_1': { ... },
      ...
      'goals': {
        'drone_0': [gx, gy, gz],
        'drone_1': [gx, gy, gz],
        ...
      }
    }
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
import yaml
import os


class RunningNormalizer:
    """
    Online running mean and standard deviation normalizer.
    Updates incrementally without storing all past data.
    Applied to local_fixed observations for training stability.
    """

    def __init__(self, dim: int, epsilon: float = 1e-8):
        self.mean    = np.zeros(dim, dtype=np.float32)
        self.var     = np.ones(dim,  dtype=np.float32)
        self.count   = 0
        self.epsilon = epsilon

    def update(self, x: np.ndarray):
        """Update running stats with a batch of observations (B, dim)."""
        batch_mean  = x.mean(axis=0)
        batch_var   = x.var(axis=0)
        batch_count = x.shape[0]

        total = self.count + batch_count
        delta = batch_mean - self.mean

        self.mean  = self.mean + delta * batch_count / total
        self.var   = (
            (self.var * self.count + batch_var * batch_count
             + delta ** 2 * self.count * batch_count / total)
            / total
        )
        self.count = total

    def normalize(self, x: np.ndarray) -> np.ndarray:
        """Normalize x using running statistics."""
        return (x - self.mean) / (np.sqrt(self.var) + self.epsilon)


class ObservationProcessor:
    """
    Converts raw Webots state to PyTorch tensors for Actor and Critic.

    Output per call to process():
        local_fixed     : (N, 27)    — fixed local obs per drone
        neighbor_states : (N, K, 6)  — K nearest neighbor states per drone
        neighbor_mask   : (N, K)     — True for empty/padded neighbor slots
        global_state    : (N, 17)    — per-drone global state for critic

    The Actor receives (local_fixed, neighbor_states, neighbor_mask)
    and internally runs attention to produce the full 59-dim observation.
    The Critic receives global_state (N, 17).
    """

    def __init__(self, config: dict, device: torch.device):
        self.config  = config
        self.device  = device

        self.n_agents        = config['environment']['n_agents']          # 3
        self.k_neighbors_max = config['observation']['k_neighbors_max']   # 5
        self.n_lidar_rays    = config['observation']['n_lidar_rays']      # 14
        self.n_lidar_quad    = config['observation']['n_lidar_quadrants'] # 5
        self.neighbor_obs_dim= config['observation']['neighbor_obs_dim']  # 6
        self.local_obs_dim   = config['observation']['local_obs_dim']     # 27
        self.drone_state_dim = 17   # per-drone global state dimension

        # Running normalizer for local fixed observations
        self.normalizer = RunningNormalizer(self.local_obs_dim)

        # LiDAR quadrant ray assignments
        # Front: rays 0 (F), 6 (NE_up), 10 (NE_down)
        # Back:  rays 1 (B), 7 (SW_up), 11 (SW_down)
        # Left:  rays 2 (L), 8 (NW_up), 12 (NW_down)
        # Right: rays 3 (R), 9 (SE_up), 13 (SE_down)
        # Vertical: rays 4 (up), 5 (down)
        self.lidar_quadrants = {
            'front':    [0, 6, 10],   # front cardinal + NE upper + NE lower
            'back':     [1, 7, 11],   # back cardinal  + SW upper + SW lower
            'left':     [2, 8, 12],   # left cardinal  + NW upper + NW lower
            'right':    [3, 9, 13],   # right cardinal + SE upper + SE lower
            'vertical': [4, 5],       # straight up + straight down
        }

    def process(
        self,
        raw_state:        Dict,
        curriculum_stage: int,
        drone_ids:        Optional[List[str]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Main processing function. Converts raw_state to all required tensors.

        Args:
            raw_state        : dict from WebotsAdapter.get_raw_state()
            curriculum_stage : 1 = zero LiDAR in global state, 2+ = full
            drone_ids        : ordered list of drone keys e.g. ['drone_0','drone_1',...]
                               If None, auto-detected from raw_state keys

        Returns:
            local_fixed     : (N, 27)     tensor — own state per drone
            neighbor_states : (N, K, 6)   tensor — KNN neighbor states
            neighbor_mask   : (N, K) bool — True for padded empty slots
            global_state    : (N, 17)     tensor — per-drone global state
        """
        if drone_ids is None:
            drone_ids = sorted([k for k in raw_state.keys() if k != 'goals'])

        N = len(drone_ids)

        # Extract per-drone arrays
        positions    = np.array([raw_state[d]['position']    for d in drone_ids], dtype=np.float32)
        velocities   = np.array([raw_state[d]['velocity']    for d in drone_ids], dtype=np.float32)
        orientations = np.array([raw_state[d]['orientation'] for d in drone_ids], dtype=np.float32)
        lidars       = np.array([raw_state[d]['lidar']       for d in drone_ids], dtype=np.float32)
        goals        = np.array([raw_state['goals'][d]       for d in drone_ids], dtype=np.float32)

        # ── 1. Build local fixed observations (N, 27) ──────────────────────
        local_fixed = self._build_local_fixed(
            positions, velocities, orientations, lidars, goals
        )

        # ── 2. KNN neighbor encoding ────────────────────────────────────────
        neighbor_states, neighbor_mask = self._build_neighbor_states(
            positions, velocities
        )

        # ── 3. Build global state (N, 17) for critic ────────────────────────
        global_state = self._build_global_state(
            positions, velocities, orientations, goals, lidars, curriculum_stage
        )

        # Convert to tensors
        lf = torch.tensor(local_fixed,     dtype=torch.float32, device=self.device)
        ns = torch.tensor(neighbor_states, dtype=torch.float32, device=self.device)
        nm = torch.tensor(neighbor_mask,   dtype=torch.bool,    device=self.device)
        gs = torch.tensor(global_state,    dtype=torch.float32, device=self.device)

        return lf, ns, nm, gs

    def _build_local_fixed(
        self,
        positions:    np.ndarray,   # (N, 3)
        velocities:   np.ndarray,   # (N, 3)
        orientations: np.ndarray,   # (N, 3)
        lidars:       np.ndarray,   # (N, 14)
        goals:        np.ndarray,   # (N, 3)
    ) -> np.ndarray:
        """
        Builds the 27-dimensional fixed local observation per drone.

        Layout:
            [0:3]   own position          (3)
            [3:6]   own velocity          (3)
            [6:9]   own orientation       (3)
            [9:12]  relative goal vector  (3)  = goal - position
            [12]    goal distance scalar  (1)
            [13:27] 14 LiDAR ray values   (14)

        Total: 27
        """
        N = positions.shape[0]
        obs = np.zeros((N, 27), dtype=np.float32)

        obs[:, 0:3]  = positions
        obs[:, 3:6]  = velocities
        obs[:, 6:9]  = orientations

        # Relative goal vector (translation invariant)
        rel_goal          = goals - positions              # (N, 3)
        obs[:, 9:12]      = rel_goal

        # Goal distance scalar
        goal_dist         = np.linalg.norm(rel_goal, axis=1, keepdims=True)  # (N, 1)
        obs[:, 12:13]     = goal_dist

        # 14 LiDAR rays
        obs[:, 13:27]     = lidars

        # Update running normalizer and normalize
        self.normalizer.update(obs)
        obs = self.normalizer.normalize(obs)

        return obs    # (N, 27)

    def _build_neighbor_states(
        self,
        positions:  np.ndarray,   # (N, 3)
        velocities: np.ndarray,   # (N, 3)
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Builds neighbor state matrix using brute-force KNN.
        Returns relative positions and velocities for K nearest neighbors.

        Each neighbor contributes 6 values:
            [rel_px, rel_py, rel_pz, rel_vx, rel_vy, rel_vz]

        Relative (not absolute) for translation invariance.

        Returns:
            neighbor_states : (N, K_max, 6)  — padded neighbor matrix
            neighbor_mask   : (N, K_max) bool — True = padded empty slot
        """
        N = positions.shape[0]
        K = self.k_neighbors_max

        neighbor_states = np.zeros((N, K, 6), dtype=np.float32)
        neighbor_mask   = np.ones( (N, K),    dtype=bool)   # True = masked (empty)

        for i in range(N):
            # Compute distances to all other drones
            others = [j for j in range(N) if j != i]
            if not others:
                continue   # single drone — no neighbors

            dists = np.linalg.norm(positions[others] - positions[i], axis=1)

            # Sort by distance — K nearest neighbors
            k_actual = min(len(others), K)
            sorted_idx = np.argsort(dists)[:k_actual]
            neighbor_indices = [others[s] for s in sorted_idx]

            for slot, j in enumerate(neighbor_indices):
                rel_pos  = positions[j]  - positions[i]    # relative position
                rel_vel  = velocities[j] - velocities[i]   # relative velocity
                neighbor_states[i, slot, :3] = rel_pos
                neighbor_states[i, slot, 3:] = rel_vel
                neighbor_mask[i, slot]        = False       # slot is populated

        return neighbor_states, neighbor_mask   # (N, K, 6), (N, K)

    def _build_global_state(
        self,
        positions:        np.ndarray,   # (N, 3)
        velocities:       np.ndarray,   # (N, 3)
        orientations:     np.ndarray,   # (N, 3)
        goals:            np.ndarray,   # (N, 3)
        lidars:           np.ndarray,   # (N, 14)
        curriculum_stage: int,
    ) -> np.ndarray:
        """
        Builds global state tensor (N, 17) for the centralized critic.

        Per-drone layout (17 values):
            [0:3]   absolute position     (3)   — critic needs absolute, not relative
            [3:6]   absolute velocity     (3)
            [6:9]   orientation           (3)
            [9:12]  goal position         (3)   — critic understands each drone's intent
            [12:17] 5-quadrant LiDAR      (5)   — zero in Stage 1 (no obstacles)

        NOTE: Global state uses ABSOLUTE positions (not relative).
        The critic needs the full spatial picture of the swarm.
        """
        N = positions.shape[0]
        gs = np.zeros((N, 17), dtype=np.float32)

        gs[:, 0:3]  = positions
        gs[:, 3:6]  = velocities
        gs[:, 6:9]  = orientations
        gs[:, 9:12] = goals

        # LiDAR quadrant compression — Stage 2+ only
        if curriculum_stage >= 2:
            quads = self._compress_lidar_to_quadrants(lidars)   # (N, 5)
            gs[:, 12:17] = quads
        # else: Stage 1 — LiDAR slots stay zero (no obstacles exist)

        return gs   # (N, 17)

    def _compress_lidar_to_quadrants(self, lidars: np.ndarray) -> np.ndarray:
        """
        Compress 14 LiDAR rays into 5 directional quadrant minimums.

        Each quadrant value = minimum ray distance in that quadrant.
        Minimum represents the closest threat — conservative and safety-appropriate.

        Quadrant assignments:
            Front    : rays [0, 6, 10]   — front cardinal + NE upper/lower
            Back     : rays [1, 7, 11]   — back cardinal  + SW upper/lower
            Left     : rays [2, 8, 12]   — left cardinal  + NW upper/lower
            Right    : rays [3, 9, 13]   — right cardinal + SE upper/lower
            Vertical : rays [4, 5]       — straight up + straight down

        Returns: (N, 5)
        """
        N = lidars.shape[0]
        quads = np.zeros((N, 5), dtype=np.float32)

        quads[:, 0] = lidars[:, self.lidar_quadrants['front']].min(axis=1)
        quads[:, 1] = lidars[:, self.lidar_quadrants['back']].min(axis=1)
        quads[:, 2] = lidars[:, self.lidar_quadrants['left']].min(axis=1)
        quads[:, 3] = lidars[:, self.lidar_quadrants['right']].min(axis=1)
        quads[:, 4] = lidars[:, self.lidar_quadrants['vertical']].min(axis=1)

        return quads   # (N, 5)

    def reset_normalizer(self):
        """Reset running normalizer — call at start of each curriculum stage."""
        self.normalizer = RunningNormalizer(self.local_obs_dim)


def build_dummy_raw_state(n_agents: int, lidar_range: float = 10.0) -> Dict:
    """
    Build a dummy raw_state dict for testing without Webots.
    Simulates the output of WebotsAdapter.get_raw_state().
    """
    state = {'goals': {}}
    for i in range(n_agents):
        key = f'drone_{i}'
        state[key] = {
            'position':    np.random.uniform(-10, 10, 3).tolist(),
            'velocity':    np.random.uniform(-2, 2, 3).tolist(),
            'orientation': np.random.uniform(-0.5, 0.5, 3).tolist(),
            'lidar':       np.random.uniform(0.5, lidar_range, 14).tolist(),
        }
        state['goals'][key] = np.random.uniform(-10, 10, 3).tolist()
    return state


if __name__ == "__main__":
    """Unit test — python observation_processor.py"""
    import yaml
    config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'default_config.yaml')
    with open(config_path) as f:
        config = yaml.safe_load(f)

    device = torch.device('cpu')
    proc   = ObservationProcessor(config, device)
    N      = config['environment']['n_agents']
    K      = config['observation']['k_neighbors_max']

    print("=" * 60)
    print("ObservationProcessor Unit Test")
    print("=" * 60)

    raw = build_dummy_raw_state(N)

    # Stage 1
    lf, ns, nm, gs = proc.process(raw, curriculum_stage=1)
    print(f"\nStage 1 (no LiDAR in global state):")
    print(f"  local_fixed shape     : {lf.shape}   expected: ({N}, 27)")
    print(f"  neighbor_states shape : {ns.shape}  expected: ({N}, {K}, 6)")
    print(f"  neighbor_mask shape   : {nm.shape}   expected: ({N}, {K})")
    print(f"  global_state shape    : {gs.shape}   expected: ({N}, 17)")
    print(f"  global_state LiDAR slots (last 5) zero: {(gs[:, 12:] == 0).all()}  ✓")

    # Stage 2
    lf2, ns2, nm2, gs2 = proc.process(raw, curriculum_stage=2)
    print(f"\nStage 2 (LiDAR quadrants active):")
    print(f"  global_state shape    : {gs2.shape}  expected: ({N}, 17)")
    print(f"  LiDAR slots populated : {(gs2[:, 12:] != 0).any()}  ✓")

    # Shape assertions
    assert lf.shape  == (N, 27),    f"local_fixed shape mismatch: {lf.shape}"
    assert ns.shape  == (N, K, 6),  f"neighbor_states shape mismatch: {ns.shape}"
    assert nm.shape  == (N, K),     f"neighbor_mask shape mismatch: {nm.shape}"
    assert gs.shape  == (N, 17),    f"global_state shape mismatch: {gs.shape}"

    # No NaN
    assert not torch.isnan(lf).any(),  "NaN in local_fixed"
    assert not torch.isnan(ns).any(),  "NaN in neighbor_states"
    assert not torch.isnan(gs2).any(), "NaN in global_state (Stage 2)"
    print(f"\n  no NaN in any tensor  ✓")

    # Neighbor mask — at least some slots should be populated for N>1
    populated = (~nm).any().item()
    print(f"  neighbor slots populated: {populated}  ✓")

    print("\n" + "=" * 60)
    print("All tests passed ✓")
    print("=" * 60)