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

from modules.neighbour_source import TruthNeighbourSource
from modules.neighbour_index import NeighbourIndex
from envs import lidar as lidar_mod
from training.networks.critic_network import critic_version, CRITIC_STATE_DIM


class RunningNormalizer:
    """
    Online running mean and standard deviation normalizer.
    Updates incrementally without storing all past data.
    Applied to local_fixed observations for training stability.

    Warm-up: returns raw observations until min_count samples seen,
    so statistics are stable before they drive normalization.
    Clip: normalized output clamped to [-clip, clip] to prevent
    extreme values from saturating tanh layers.
    """

    def __init__(self, dim: int, epsilon: float = 1e-8,
                 min_count: int = 1000, clip: float = 10.0):
        self.mean    = np.zeros(dim, dtype=np.float32)
        self.var     = np.ones(dim,  dtype=np.float32)
        self.count   = 0
        self.epsilon = epsilon
        self.min_count = min_count
        self.clip    = clip
        self._frozen = False

    def freeze(self):
        """Stop accumulating new statistics. Use when evaluating a trained
        checkpoint whose normalizer stats were restored from training — the
        actor's weights expect a specific fixed scale, and letting eval
        observations keep updating mean/var drifts away from it, changing
        the actor's effective inputs mid-evaluation."""
        self._frozen = True

    def unfreeze(self):
        self._frozen = False

    def update(self, x: np.ndarray):
        """Update running stats with a batch of observations (B, dim).
        No-op while frozen (see freeze())."""
        if self._frozen:
            return
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
        """Normalize using running stats. Returns raw obs during warm-up."""
        if self.count < self.min_count:
            return x
        normed = (x - self.mean) / (np.sqrt(self.var) + self.epsilon)
        return np.clip(normed, -self.clip, self.clip)


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
        # per-drone global state dimension: 17 (critic.version 1) or 32 (version 2)
        self.critic_version  = critic_version(config)
        self.drone_state_dim = CRITIC_STATE_DIM[self.critic_version]
        self.goal_threshold  = config['environment'].get('goal_threshold', 0.5)

        # Running normalizer for local fixed observations
        self.normalizer = RunningNormalizer(self.local_obs_dim)

        # Neighbour information (see docs/neighbour_information_design.md).
        #   sensing_radius     : drones further away than this are invisible
        #                        (null = no limit, the old behaviour)
        #   neighbor_pos_scale : relative positions are divided by this
        #   neighbor_vel_scale : relative velocities are divided by this
        # Defaults reproduce the old raw, unlimited neighbour inputs exactly,
        # so existing configs and checkpoints are unaffected.
        obs_cfg = config['observation']
        self.sensing_radius     = obs_cfg.get('sensing_radius', None)
        self.neighbor_pos_scale = float(obs_cfg.get('neighbor_pos_scale', 1.0))
        self.neighbor_vel_scale = float(obs_cfg.get('neighbor_vel_scale', 1.0))
        self.neighbour_source   = TruthNeighbourSource()
        #   neighbor_index     : brute (default, every other drone) | grid | h3; the
        #                        index only narrows the candidates, the picked
        #                        neighbours are identical (modules/neighbour_index.py)
        self.neighbour_index    = NeighbourIndex(obs_cfg.get('neighbor_index', 'brute'),
                                                 self.sensing_radius, obs_cfg.get('h3_resolution'))

        # LiDAR uses a fixed scale instead of the running normalizer:
        # proximity = 1 - distance / range (0 = nothing seen, 1 = touching).
        # In Stage 1 the LiDAR inputs were always 0, so a normalizer frozen
        # then has zero variance for them and would blow any real reading up
        # to the clip limit; a fixed formula also can't drift and means the
        # same in every simulator. Nothing seen gives 0, exactly the value
        # Stage-1 checkpoints were trained on.
        self.lidar_range = lidar_mod.lidar_config(config)['range']

        # Observation schema (scripts/convert_checkpoint_inputs.py turns v1 actors into v2):
        #   v1 : 27 inputs, everything but LiDAR through the running normalizer
        #   v2 : 42 inputs, every input on a fixed, documented scale (no
        #        normalizer, so nothing can drift), plus the previous LiDAR
        #        reading (14) and a log goal distance (1)
        #   v3 : v2 + the horizontal LiDAR ring (lidar.ring_rays proximities,
        #        appended at [42:]); 42 + 32 = 74 for the Stage-2 ring
        self.schema = obs_cfg.get('schema', 'v1')
        self.ring_rays = lidar_mod.lidar_config(config)['ring_rays']
        expected_dim = {'v1': 27, 'v2': 42, 'v3': 42 + self.ring_rays}[self.schema]
        if self.local_obs_dim != expected_dim:
            raise ValueError(f"observation.schema {self.schema} needs local_obs_dim "
                             f"{expected_dim}, config has {self.local_obs_dim}")
        env_cfg = config['environment']
        self.position_scale = np.array([env_cfg.get('world_size', 50) / 2.0] * 2
                                       + [obs_cfg.get('altitude_scale', 10.0)], dtype=np.float32)
        self.velocity_scale = float(env_cfg.get('v_max', 2.0))
        # Caps chosen where the v1 normalizer clipped (goal axes ~2.4-2.9 m,
        # distance ~4 m), so converted v1 actors see the same values below
        # them. Beyond the cap the goal vector keeps its direction (v1 clipped
        # each axis separately, which bent the direction of far goals).
        # altitude_cap (schemas v2/v3, off by default): the drone's own height input is read as at most
        # this many metres. The policy was trained near 1-4 m; above ~8 m its unseen height input made it
        # settle ~2 m short of its goal and never land (scripts/altitude_test.py). Only the own-height
        # feature is capped: the goal vector and neighbours' relative positions keep the true heights.
        cap = obs_cfg.get('altitude_cap')
        self.altitude_cap = None if cap is None else float(cap)
        # position_origin (schemas v2/v3, off by default): 'goal' reads the drone's own x, y input
        # relative to its current goal (the route carrot) instead of the map centre. The policy was
        # trained in a 50 m arena; on a bigger map an own position many arenas out made it stop short
        # of its goal. The goal vector, LiDAR and neighbours are relative already and do not change.
        self.position_origin = obs_cfg.get('position_origin')
        if self.position_origin not in (None, 'goal'):
            raise ValueError(f"observation.position_origin must be 'goal' or unset, not {self.position_origin!r}")
        self.goal_vector_cap   = float(obs_cfg.get('goal_vector_cap', 2.5))
        self.goal_distance_cap = float(obs_cfg.get('goal_distance_cap', 4.0))
        self._prev_prox = None    # v2: last step's LiDAR proximity, per drone

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
        critic_extra:     Optional[Dict] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Main processing function. Converts raw_state to all required tensors.

        Args:
            raw_state        : dict from WebotsAdapter.get_raw_state()
            curriculum_stage : 1 = zero LiDAR in global state, 2+ = full
            drone_ids        : ordered list of drone keys e.g. ['drone_0','drone_1',...]
                               If None, auto-detected from raw_state keys
            critic_extra     : critic.version 2 only, from SwarmEnv:
                               {'settle_progress': (N,) in [0, 1], 'reached': (N,) bool}

        Returns:
            local_fixed     : (N, 27)     tensor — own state per drone
            neighbor_states : (N, K, 6)   tensor — KNN neighbor states
            neighbor_mask   : (N, K) bool — True for padded empty slots
            global_state    : (N, 17)     tensor — per-drone global state
        """
        if drone_ids is None:
            # numeric order (drone_2 before drone_10): row i must be drone_i, which a plain
            # text sort breaks from 11 drones on
            drone_ids = sorted([k for k in raw_state.keys() if k != 'goals'], key=lambda k: (len(k), k))

        N = len(drone_ids)

        # Extract per-drone arrays
        positions    = np.array([raw_state[d]['position']    for d in drone_ids], dtype=np.float32)
        velocities   = np.array([raw_state[d]['velocity']    for d in drone_ids], dtype=np.float32)
        orientations = np.array([raw_state[d]['orientation'] for d in drone_ids], dtype=np.float32)
        lidars       = np.array([raw_state[d]['lidar']       for d in drone_ids], dtype=np.float32)
        rings        = (np.array([raw_state[d].get('lidar_ring', [self.lidar_range] * self.ring_rays)
                                  for d in drone_ids], dtype=np.float32).reshape(len(drone_ids), -1)
                        if self.schema == 'v3' else None)
        goals        = np.array([raw_state['goals'][d]       for d in drone_ids], dtype=np.float32)

        # ── 1. Build local fixed observations (N, 27) ──────────────────────
        local_fixed = self._build_local_fixed(
            positions, velocities, orientations, lidars, goals, rings
        )

        # ── 2. KNN neighbor encoding ────────────────────────────────────────
        neighbor_states, neighbor_mask = self._build_neighbor_states(
            positions, velocities
        )

        # ── 3. Build global state (N, 17) for critic ────────────────────────
        global_state = self._build_global_state(
            positions, velocities, orientations, goals, lidars, curriculum_stage
        )
        if self.critic_version >= 2:
            global_state = np.concatenate(
                [global_state, self._critic_extra_state(positions, goals, rings, critic_extra)],
                axis=1)                                                    # (N, 32)
            # metre-valued slots (position, goal, goal - position, goal distance)
            # / 10, so they sit on the same ~[-1, 1] scale as the rest. Held-out
            # fit of BCA's returns (scripts/critic_fit_check.py): 0.37 -> 0.42
            global_state[:, self.CRITIC_METRE_SLOTS] /= 10.0

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
        rings:        Optional[np.ndarray] = None,   # (N, ring_rays), schema v3 only
    ) -> np.ndarray:
        """
        Builds the 27-dimensional fixed local observation per drone.

        Layout:
            [0:3]   own position          (3)
            [3:6]   own velocity          (3)
            [6:9]   own orientation       (3)
            [9:12]  relative goal vector  (3)  = goal - position
            [12]    goal distance scalar  (1)
            [13:27] 14 LiDAR proximities  (14)  = 1 - distance / range, not normalized

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

        # 14 LiDAR rays as proximity on a fixed scale
        prox              = lidar_mod.proximity(lidars, self.lidar_range)
        obs[:, 13:27]     = prox

        if self.schema == 'v2':
            return self._fixed_scale(obs, prox)
        if self.schema == 'v3':
            ring = lidar_mod.proximity(rings, self.lidar_range)
            return np.concatenate([self._fixed_scale(obs, prox), ring], axis=1)   # (N, 42 + ring)

        # Update running normalizer and normalize; LiDAR bypasses it
        self.normalizer.update(obs)
        obs = self.normalizer.normalize(obs)
        obs[:, 13:27]     = prox

        return obs    # (N, 27)

    def _fixed_scale(self, raw: np.ndarray, prox: np.ndarray) -> np.ndarray:
        """
        Schema v2: the 27 v1 inputs on fixed scales, plus 15 new ones.

            [0:3]   position / (world_size/2, world_size/2, altitude_scale)
            [3:6]   velocity / v_max
            [6:9]   orientation / pi
            [9:12]  goal vector, length capped at goal_vector_cap, / cap
            [12]    goal distance, capped at goal_distance_cap, / cap
            [13:27] LiDAR proximity               (as v1)
            [27:41] previous step's LiDAR proximity (approaching vs receding)
            [41]    log(1 + goal distance) / 4     (far goals stay distinguishable)
        """
        N = raw.shape[0]
        obs = np.zeros((N, 42), dtype=np.float32)
        obs[:, 0:3]   = raw[:, 0:3] / self.position_scale
        if self.altitude_cap is not None:
            obs[:, 2] = np.minimum(raw[:, 2], self.altitude_cap) / self.position_scale[2]
        if self.position_origin == 'goal':                    # raw[:, 9:12] = goal - position
            obs[:, 0:2] = -raw[:, 9:11] / self.position_scale[:2]
        obs[:, 3:6]   = raw[:, 3:6] / self.velocity_scale
        obs[:, 6:9]   = raw[:, 6:9] / np.pi
        dist          = raw[:, 12:13]
        shrink        = np.minimum(1.0, self.goal_vector_cap / np.maximum(dist, 1e-6))
        obs[:, 9:12]  = raw[:, 9:12] * shrink / self.goal_vector_cap
        obs[:, 12:13] = np.minimum(dist, self.goal_distance_cap) / self.goal_distance_cap
        obs[:, 13:27] = prox
        # First step of an episode (or a change in drone count): no history
        # yet, so "previous" = current, i.e. nothing approaching.
        if self._prev_prox is None or self._prev_prox.shape != prox.shape:
            self._prev_prox = prox
        obs[:, 27:41] = self._prev_prox
        obs[:, 41:42] = np.log1p(dist) / 4.0
        self._prev_prox = prox.copy()
        return obs    # (N, 42)

    def reset_history(self):
        """Forget the previous LiDAR reading — call at the start of every episode."""
        self._prev_prox = None

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

        Steps, per observing drone i:
            1. ask the neighbour source what i knows about the others
            2. drop drones beyond the sensing radius (if one is set)
            3. keep the K nearest, scale them, mask the empty slots

        Returns:
            neighbor_states : (N, K_max, 6)  — padded neighbor matrix
            neighbor_mask   : (N, K_max) bool — True = padded empty slot
        """
        N = positions.shape[0]
        K = self.k_neighbors_max

        neighbor_states = np.zeros((N, K, 6), dtype=np.float32)
        neighbor_mask   = np.ones( (N, K),    dtype=bool)   # True = masked (empty)

        known_pos, known_vel, known = self.neighbour_source.observe(positions, velocities)

        # The truth source broadcasts the same positions to everyone, so one index
        # over them serves every observer.
        candidates = self.neighbour_index.candidates(known_pos[0])

        for i in range(N):
            others = [j for j in candidates[i].tolist() if known[i, j]]
            if not others:
                continue   # single drone, or nothing known about anyone

            dists = np.linalg.norm(known_pos[i, others] - positions[i], axis=1)

            # Sensing radius: anything further away is invisible
            if self.sensing_radius is not None:
                in_range = dists <= self.sensing_radius
                others = [j for j, ok in zip(others, in_range) if ok]
                dists = dists[in_range]
                if not others:
                    continue

            # Sort by distance — K nearest neighbors
            k_actual = min(len(others), K)
            sorted_idx = np.argsort(dists)[:k_actual]
            neighbor_indices = [others[s] for s in sorted_idx]

            for slot, j in enumerate(neighbor_indices):
                rel_pos  = known_pos[i, j] - positions[i]    # relative position
                rel_vel  = known_vel[i, j] - velocities[i]   # relative velocity
                neighbor_states[i, slot, :3] = rel_pos / self.neighbor_pos_scale
                neighbor_states[i, slot, 3:] = rel_vel / self.neighbor_vel_scale
                neighbor_mask[i, slot]        = False         # slot is populated

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
            [12:17] 5-quadrant LiDAR      (5)   — closest proximity per quadrant
                                                   (0 when nothing is seen, as in Stage 1)

        NOTE: Global state uses ABSOLUTE positions (not relative).
        The critic needs the full spatial picture of the swarm.

        The LiDAR slots no longer depend on curriculum_stage (the stage counter
        never advances while curriculum advancement is disabled); with LiDAR
        off or nothing in range they are exactly 0, as before.
        """
        N = positions.shape[0]
        gs = np.zeros((N, 17), dtype=np.float32)

        gs[:, 0:3]  = positions
        gs[:, 3:6]  = velocities
        gs[:, 6:9]  = orientations
        gs[:, 9:12] = goals
        gs[:, 12:17] = self._compress_lidar_to_quadrants(
            lidar_mod.proximity(lidars, self.lidar_range))      # (N, 5)

        return gs   # (N, 17)

    def _compress_lidar_to_quadrants(self, lidars: np.ndarray) -> np.ndarray:
        """
        Compress 14 LiDAR proximities into 5 directional quadrant maximums
        (the closest thing seen in each direction).

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

        quads[:, 0] = lidars[:, self.lidar_quadrants['front']].max(axis=1)
        quads[:, 1] = lidars[:, self.lidar_quadrants['back']].max(axis=1)
        quads[:, 2] = lidars[:, self.lidar_quadrants['left']].max(axis=1)
        quads[:, 3] = lidars[:, self.lidar_quadrants['right']].max(axis=1)
        quads[:, 4] = lidars[:, self.lidar_quadrants['vertical']].max(axis=1)

        return quads   # (N, 5)

    CRITIC_METRE_SLOTS = [0, 1, 2, 9, 10, 11, 17, 18, 19, 20]   # critic.version 2

    def _critic_extra_state(
        self,
        positions:    np.ndarray,            # (N, 3)
        goals:        np.ndarray,            # (N, 3)
        rings:        Optional[np.ndarray],  # (N, ring_rays) or None
        critic_extra: Optional[Dict],
    ) -> np.ndarray:
        """
        The 15 values critic.version 2 adds to each drone's 17 (-> 32):

            [17:20] goal - position     (3) m   "how far, which way" directly
            [20]    goal distance       (1) m
            [21]    inside goal radius  (1) 0/1
            [22]    settle progress     (1) settle counter / settle_steps, in [0, 1]
            [23]    reached (latched)   (1) 0/1
            [24:32] ring, 8 sectors     (8) closest proximity per 45-degree sector
                                            of the horizontal ring (0 without one)

        Without the settle state, a drone one step from the +100 settle reward
        and one thirty steps away look the same to the critic; without the
        ring it sees obstacles only as 5 coarse quadrant values.
        """
        N = positions.shape[0]
        rel = goals - positions
        dist = np.linalg.norm(rel, axis=1, keepdims=True)
        extra = critic_extra or {}
        settle = np.asarray(extra.get('settle_progress', np.zeros(N)), dtype=np.float32).reshape(N, 1)
        reached = np.asarray(extra.get('reached', np.zeros(N)), dtype=np.float32).reshape(N, 1)
        sectors = np.zeros((N, 8), dtype=np.float32)
        if rings is not None and rings.shape[1] >= 8 and rings.shape[1] % 8 == 0:
            prox = lidar_mod.proximity(rings, self.lidar_range)
            sectors = prox.reshape(N, 8, -1).max(axis=2)
        return np.concatenate([rel, dist, (dist < self.goal_threshold).astype(np.float32),
                               settle, reached, sectors], axis=1).astype(np.float32)   # (N, 15)

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