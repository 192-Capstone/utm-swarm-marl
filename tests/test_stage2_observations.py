"""
test_stage2_observations.py
---------------------------
Unit tests for the Stage-2 observation layouts (schemas v2 and v3) and the
LiDAR geometry they depend on.

Purpose:
    - Verify that ObservationProcessor produces the shapes the networks expect
      (42 inputs per drone for schema v2, 74 for schema v3 with the LiDAR ring)
    - Verify the documented layout of those inputs (goal vector and distance
      caps, "previous LiDAR" slots, LiDAR proximity slots)
    - Verify the neighbour inputs (shape, padding mask, scaling)
    - Verify the LiDAR helper functions in envs/lidar.py (proximity formula,
      fan and ring ray directions)
    - Verify that a config whose schema and local_obs_dim disagree is rejected

Why this matters:
    A model trained on one input layout silently breaks if the code starts
    producing another. These tests fail loudly the moment the layout changes.

No simulator and no trained model are needed: a small hand-made world state
(three drones in a row) is fed straight to ObservationProcessor.

Run with:
    python tests/test_stage2_observations.py
    python -m pytest tests/test_stage2_observations.py -q -s

Use -s (show prints) to see what each test is doing and the values it checks.
Without -s pytest hides the printed output and only shows pass / fail.
"""

import copy
import os
import sys

import numpy as np
import pytest
import torch
import yaml

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs import lidar as lidar_mod
from modules.observation_processor import ObservationProcessor
from training.networks.critic_network import CRITIC_STATE_DIM, critic_version


# ============================================================================
# Helper Functions
# ============================================================================

# (config file, observation schema, expected local_obs_dim)
STAGE2_SCHEMAS = [
    ("config_stage2_obstacles.yaml", "v2", 42),
    ("config_stage2_ring.yaml", "v3", 74),
]


def load_config(name):
    """Load one of the real configuration files from configs/."""
    config_path = os.path.join(
        os.path.dirname(__file__), '..', 'configs', name
    )
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def print_test_header(test_num, test_name, purpose, label=None):
    """Print formatted test header (label = the schema the test is running for)."""
    print("\n" + "=" * 60)
    suffix = f"  [{label}]" if label else ""
    print(f"TEST {test_num}: {test_name}{suffix}")
    print("=" * 60)
    print(f"Purpose: {purpose}")


def make_processor(config):
    """Create an ObservationProcessor on the CPU from a config dict."""
    return ObservationProcessor(config, torch.device("cpu"))


def make_raw_state(config, lidar_distance=None, ring_distance=None):
    """
    Build a raw state dict for three drones in a row, in the same format
    PyBulletAdapter.get_raw_state() returns.

    World (all drones hovering, so every velocity is zero):
        drone_0 at x=0 m, goal 10 m ahead   -> goal further than the 2.5 m / 4 m caps
        drone_1 at x=2 m, goal 1 m ahead    -> goal inside the caps
        drone_2 at x=4 m, goal 1 m above it -> goal inside the caps

    Args:
        config          : config dict (gives the LiDAR range and ring size)
        lidar_distance  : distance (m) that every one of the 14 fan rays reports
                          (default = the LiDAR range, i.e. "nothing seen")
        ring_distance   : same for the ring rays (default = the LiDAR range)
    """
    lidar_cfg = lidar_mod.lidar_config(config)
    lidar_range = lidar_cfg['range']
    ring_rays = lidar_cfg['ring_rays']
    lidar_distance = lidar_range if lidar_distance is None else lidar_distance
    ring_distance = lidar_range if ring_distance is None else ring_distance

    positions = [[0.0, 0.0, 2.0], [2.0, 0.0, 2.0], [4.0, 0.0, 2.0]]
    goals = [[10.0, 0.0, 2.0], [3.0, 0.0, 2.0], [4.0, 0.0, 3.0]]

    raw = {'goals': {}}
    for i in range(3):
        key = f'drone_{i}'
        raw[key] = {
            'position': positions[i],
            'velocity': [0.0, 0.0, 0.0],
            'orientation': [0.0, 0.0, 0.0],
            'lidar': [lidar_distance] * lidar_mod.N_RAYS,
        }
        if ring_rays > 0:
            raw[key]['lidar_ring'] = [ring_distance] * ring_rays
        raw['goals'][key] = goals[i]
    return raw


def process(processor, config, raw):
    """
    Run ObservationProcessor.process() and return the four outputs as numpy
    arrays: local_fixed, neighbor_states, neighbor_mask, global_state.

    critic_extra is only read when critic.version is 2; it is passed in
    every case so the same helper works for both critic versions.
    """
    critic_extra = {
        'settle_progress': np.zeros(3, dtype=np.float32),
        'reached': np.zeros(3, dtype=bool),
    }
    lf, ns, nm, gs = processor.process(raw, curriculum_stage=2, critic_extra=critic_extra)
    return lf.numpy(), ns.numpy(), nm.numpy(), gs.numpy()


@pytest.fixture(params=STAGE2_SCHEMAS, ids=["v2-obstacles", "v3-ring"])
def setup(request):
    """
    Yield (config, processor, schema, expected_dim) once for each Stage-2
    schema, so every test below runs for both v2 and v3.
    """
    name, schema, expected_dim = request.param
    config = load_config(name)
    processor = make_processor(config)
    return config, processor, schema, expected_dim


# ============================================================================
# TEST 1: Local observation shape
# ============================================================================

def test_local_obs_shape(setup):
    """
    Test that the local observation has the right size for the schema.

    What this tests:
        - Schema v2 gives 42 inputs per drone, schema v3 gives 74
        - The config's observation.schema and local_obs_dim agree
        - Every value is a finite float32 (no NaN / inf)
    """
    config, processor, schema, expected_dim = setup
    print_test_header(1, "Local Observation Shape",
                      "Verify each drone gets the right number of inputs", label=schema)
    lf, _, _, _ = process(processor, config, make_raw_state(config))

    print(f"\n  config schema:        {config['observation']['schema']}")
    print(f"  config local_obs_dim: {config['observation']['local_obs_dim']} (expected {expected_dim})")
    print(f"  local_fixed shape:    {lf.shape}  (drones, inputs per drone)")
    print(f"  dtype: {lf.dtype}, all values finite: {bool(np.isfinite(lf).all())}")

    assert config['observation']['schema'] == schema
    assert config['observation']['local_obs_dim'] == expected_dim, \
        "config local_obs_dim does not match the schema"
    assert lf.shape == (3, expected_dim), \
        f"local_fixed shape mismatch: got {lf.shape}, expected (3, {expected_dim})"
    assert lf.dtype == np.float32
    assert np.isfinite(lf).all(), "local_fixed contains NaN or inf"
    print(f"\n  \u2713 Observation has {expected_dim} inputs per drone")


# ============================================================================
# TEST 2: Neighbour inputs
# ============================================================================

def test_neighbour_shapes_and_mask(setup):
    """
    Test the neighbour inputs: shape, padding mask and scaling.

    What this tests:
        - neighbor_states is (N, K_max, 6) and neighbor_mask is (N, K_max)
        - With three drones every drone has exactly two real neighbours, so
          the first two slots are real (mask False) and the rest padded (True)
        - The nearest neighbour of drone_0 is drone_1, 2 m ahead, and its
          relative position is divided by observation.neighbor_pos_scale
    """
    config, processor, schema, _ = setup
    print_test_header(2, "Neighbour Inputs",
                      "Verify neighbour shape, padding mask and scaling", label=schema)
    _, ns, nm, _ = process(processor, config, make_raw_state(config))

    k_max = config['observation']['k_neighbors_max']
    pos_scale = float(config['observation'].get('neighbor_pos_scale', 1.0))
    print(f"\n  neighbor_states shape: {ns.shape}  (drones, K_max={k_max}, 6 values each)")
    print(f"  neighbor_mask shape:   {nm.shape}")
    print(f"  mask for drone_0:      {nm[0].tolist()}  (True = empty slot)")
    print(f"  drone_0 nearest neighbour, relative x: {ns[0, 0, 0]:.4f}  "
          f"(2 m / pos_scale {pos_scale} = {2.0 / pos_scale:.4f})")

    assert ns.shape == (3, k_max, 6), f"neighbor_states shape: {ns.shape}"
    assert nm.shape == (3, k_max), f"neighbor_mask shape: {nm.shape}"

    assert not nm[:, :2].any(), "the first two slots should hold real neighbours"
    assert nm[:, 2:].all(), "the remaining slots should be padding"

    assert np.isclose(ns[0, 0, 0], 2.0 / pos_scale), \
        f"nearest neighbour of drone_0 should be 2 m / {pos_scale} ahead, got {ns[0, 0, 0]}"
    print("\n  \u2713 Two real neighbours per drone, the rest padded, scaling correct")


# ============================================================================
# TEST 3: Critic global state
# ============================================================================

def test_global_state_shape(setup):
    """
    Test that the critic's per-drone global state has the right size.

    What this tests:
        - 17 values per drone for critic.version 1, 32 for version 2
          (read from the config, so the test follows whichever the config uses)
    """
    config, processor, schema, _ = setup
    print_test_header(3, "Critic Global State",
                      "Verify the critic gets the right number of values per drone", label=schema)
    _, _, _, gs = process(processor, config, make_raw_state(config))

    expected = CRITIC_STATE_DIM[critic_version(config)]
    print(f"\n  critic.version: {critic_version(config)}")
    print(f"  global_state shape: {gs.shape}  (expected (3, {expected}))")

    assert gs.shape == (3, expected), \
        f"global_state shape mismatch: got {gs.shape}, expected (3, {expected})"
    assert np.isfinite(gs).all(), "global_state contains NaN or inf"
    print(f"\n  \u2713 Global state has {expected} values per drone, all finite")


# ============================================================================
# TEST 4: LiDAR slots in the observation
# ============================================================================

def test_nothing_seen_gives_zero_proximity(setup):
    """
    Test that "nothing in range" becomes proximity 0 in the observation.

    What this tests:
        - The 14 fan proximities live at [13:27] and are 0 when every ray
          reports the full range
        - For schema v3, the ring proximities at [42:] are also 0
    """
    config, processor, schema, expected_dim = setup
    print_test_header(4, "Nothing Seen Gives Zero Proximity",
                      "Verify empty LiDAR readings become 0 in the observation", label=schema)
    lf, _, _, _ = process(processor, config, make_raw_state(config))

    print(f"\n  every LiDAR ray reports the full range (nothing in sight)")
    print(f"  fan proximity   [13:27] max: {lf[:, 13:27].max():.4f}  (expected 0)")
    if schema == 'v3':
        print(f"  ring proximity  [42:]   max: {lf[:, 42:].max():.4f}  (expected 0)")

    assert np.allclose(lf[:, 13:27], 0.0), "fan proximity should be 0 when nothing is seen"
    if schema == 'v3':
        assert np.allclose(lf[:, 42:], 0.0), "ring proximity should be 0 when nothing is seen"
    print("\n  \u2713 No obstacle gives proximity 0")


def test_obstacle_raises_proximity(setup):
    """
    Test that a LiDAR hit shows up in the right slots.

    What this tests:
        - Every fan ray reporting half the range gives proximity 0.5
          (proximity = 1 - distance / range) at [13:27]
        - For schema v3, every ring ray at half the range gives 0.5 at [42:]
    """
    config, processor, schema, _ = setup
    print_test_header(5, "Obstacle Raises Proximity",
                      "Verify a LiDAR hit shows up in the right input slots", label=schema)
    half = lidar_mod.lidar_config(config)['range'] / 2.0
    raw = make_raw_state(config, lidar_distance=half, ring_distance=half)
    lf, _, _, _ = process(processor, config, raw)

    print(f"\n  every LiDAR ray reports {half} m (half of the range)")
    print(f"  fan proximity   [13:27] mean: {lf[:, 13:27].mean():.4f}  (expected 0.5)")
    if schema == 'v3':
        print(f"  ring proximity  [42:]   mean: {lf[:, 42:].mean():.4f}  (expected 0.5)")

    assert np.allclose(lf[:, 13:27], 0.5), "fan proximity should be 0.5 at half range"
    if schema == 'v3':
        assert np.allclose(lf[:, 42:], 0.5), "ring proximity should be 0.5 at half range"
    print("\n  \u2713 Half range gives proximity 0.5")


# ============================================================================
# TEST 5: "Previous LiDAR" slots
# ============================================================================

def test_previous_lidar_slots(setup):
    """
    Test the [27:41] slots, which hold the previous step's LiDAR reading.

    What this tests:
        - On the first step of an episode there is no history, so the
          "previous" slots equal the current reading (nothing approaching)
        - On the next step they hold the reading from the step before
        - reset_history() forgets the history again (called every episode)
    """
    config, processor, schema, _ = setup
    print_test_header(6, "Previous LiDAR Slots",
                      "Verify the previous-step LiDAR memory and its reset", label=schema)
    half = lidar_mod.lidar_config(config)['range'] / 2.0

    # Step 1: nothing seen -> previous == current == 0
    lf1, _, _, _ = process(processor, config, make_raw_state(config))
    print(f"\n  step 1 (nothing seen):   current={lf1[0, 13]:.2f}  previous={lf1[0, 27]:.2f}"
          f"  (no history yet, so previous = current)")
    assert np.allclose(lf1[:, 27:41], lf1[:, 13:27]), "first step: previous should equal current"

    # Step 2: obstacle appears -> current 0.5, previous still 0
    lf2, _, _, _ = process(processor, config, make_raw_state(config, lidar_distance=half))
    print(f"  step 2 (obstacle appears): current={lf2[0, 13]:.2f}  previous={lf2[0, 27]:.2f}"
          f"  (previous = step-1 reading)")
    assert np.allclose(lf2[:, 13:27], 0.5)
    assert np.allclose(lf2[:, 27:41], 0.0), "second step: previous should be the step-1 reading"

    # After reset_history() the next call has no history again
    processor.reset_history()
    lf3, _, _, _ = process(processor, config, make_raw_state(config, lidar_distance=half))
    print(f"  step 3 (after reset):      current={lf3[0, 13]:.2f}  previous={lf3[0, 27]:.2f}"
          f"  (history forgotten, previous = current)")
    assert np.allclose(lf3[:, 27:41], lf3[:, 13:27]), "after reset: previous should equal current"
    print("\n  \u2713 Previous-LiDAR memory works and resets between episodes")


# ============================================================================
# TEST 6: Goal vector and goal distance caps
# ============================================================================

def test_goal_vector_and_distance_caps(setup):
    """
    Test the fixed scaling of the goal inputs (schema v2 and v3).

    What this tests:
        - A far goal (10 m) keeps its direction but its vector is shrunk to
          length 1 (capped at goal_vector_cap, then divided by it), and its
          distance slot is capped at 1.0
        - A near goal (1 m) is simply divided by the caps:
          vector length 1 / 2.5 and distance 1 / 4
    """
    config, processor, schema, _ = setup
    print_test_header(7, "Goal Vector and Distance Caps",
                      "Verify far and near goals are scaled as documented", label=schema)
    obs_cfg = config['observation']
    vec_cap = float(obs_cfg.get('goal_vector_cap', 2.5))
    dist_cap = float(obs_cfg.get('goal_distance_cap', 4.0))

    lf, _, _, _ = process(processor, config, make_raw_state(config))

    print(f"\n  caps: goal vector {vec_cap} m, goal distance {dist_cap} m")
    print(f"  drone_0 (goal 10 m away): vector length {np.linalg.norm(lf[0, 9:12]):.3f} (expected 1.0),"
          f" distance slot {lf[0, 12]:.3f} (expected 1.0)")
    print(f"  drone_1 (goal 1 m away):  vector length {np.linalg.norm(lf[1, 9:12]):.3f}"
          f" (expected {1.0 / vec_cap:.3f}), distance slot {lf[1, 12]:.3f} (expected {1.0 / dist_cap:.3f})")

    # drone_0: goal 10 m straight ahead (+x), beyond both caps
    assert np.isclose(np.linalg.norm(lf[0, 9:12]), 1.0, atol=1e-5)
    assert lf[0, 9] > 0 and np.isclose(lf[0, 10], 0.0) and np.isclose(lf[0, 11], 0.0), \
        "capped goal vector must keep pointing along +x"
    assert np.isclose(lf[0, 12], 1.0), "goal distance should be capped at 1.0"

    # drone_1: goal 1 m ahead, inside both caps
    assert np.isclose(np.linalg.norm(lf[1, 9:12]), 1.0 / vec_cap, atol=1e-5)
    assert np.isclose(lf[1, 12], 1.0 / dist_cap, atol=1e-5)
    print("\n  \u2713 Far goals are capped (direction kept), near goals scaled by the caps")


# ============================================================================
# TEST 7: Schema / local_obs_dim mismatch is rejected
# ============================================================================

def test_schema_dim_mismatch_raises():
    """
    Test that a config whose schema and local_obs_dim disagree is rejected.

    What this tests:
        - schema v2 with local_obs_dim 74 raises ValueError instead of
          silently building observations of the wrong size
    """
    print_test_header(8, "Schema / Size Mismatch Is Rejected",
                      "Verify an inconsistent config raises an error")
    config = copy.deepcopy(load_config("config_stage2_obstacles.yaml"))
    config['observation']['local_obs_dim'] = 74   # v2 needs 42

    print("\n  config: schema v2 but local_obs_dim = 74 (v2 needs 42)")
    with pytest.raises(ValueError) as error:
        make_processor(config)
    print(f"  raised ValueError: {error.value}")
    print("\n  \u2713 The inconsistent config was rejected")


# ============================================================================
# TEST 8: LiDAR proximity formula
# ============================================================================

def test_proximity_formula():
    """
    Test lidar.proximity(): distance in metres -> proximity in [0, 1].

    What this tests:
        - Full range (nothing seen) -> 0
        - Half range -> 0.5
        - Very close (about to touch) -> close to 1
        - A reading of 0 means "no data" and also gives 0
        - Readings beyond the range are clipped to 0
    """
    print_test_header(9, "LiDAR Proximity Formula",
                      "Verify distance (m) converts to proximity in [0, 1]")
    d = np.array([6.0, 3.0, 0.01, 0.0, 9.0], dtype=np.float32)
    p = lidar_mod.proximity(d, 6.0)

    print("\n  range = 6.0 m")
    for dist, prox in zip(d, p):
        print(f"  distance {dist:5.2f} m -> proximity {prox:.4f}")

    assert np.isclose(p[0], 0.0)
    assert np.isclose(p[1], 0.5)
    assert p[2] > 0.99
    assert p[3] == 0.0, "a reading of 0 means no data, not touching"
    assert p[4] == 0.0, "a reading beyond the range must clip to 0"
    print("\n  \u2713 Full range 0, half range 0.5, touching ~1, no data 0, beyond range 0")


# ============================================================================
# TEST 9: LiDAR ray geometry
# ============================================================================

def test_fan_ray_directions():
    """
    Test the 14-ray fan (lidar.ray_directions).

    What this tests:
        - 14 unit vectors
        - Ray 0 points along the heading to the goal (here yaw = 30 degrees)
        - Ray 4 points straight up and ray 5 straight down
    """
    print_test_header(10, "Fan Ray Directions",
                      "Verify the 14-ray fan is rotated to the goal heading")
    yaw = np.radians(30.0)
    dirs = lidar_mod.ray_directions(yaw)

    print(f"\n  heading to goal: 30 degrees")
    print(f"  fan shape: {dirs.shape}")
    print(f"  ray 0 (front): {np.round(dirs[0], 3).tolist()}")
    print(f"  ray 4 (up):    {np.round(dirs[4], 3).tolist()}")
    print(f"  ray 5 (down):  {np.round(dirs[5], 3).tolist()}")

    assert dirs.shape == (14, 3)
    assert np.allclose(np.linalg.norm(dirs, axis=1), 1.0), "every ray must be a unit vector"
    assert np.allclose(dirs[0], [np.cos(yaw), np.sin(yaw), 0.0]), "ray 0 must follow the heading"
    assert np.allclose(dirs[4], [0.0, 0.0, 1.0]), "ray 4 must point straight up"
    assert np.allclose(dirs[5], [0.0, 0.0, -1.0]), "ray 5 must point straight down"
    print("\n  \u2713 14 unit rays, front ray follows the heading, up and down are vertical")


def test_ring_ray_directions():
    """
    Test the horizontal ring (lidar.ring_directions) for the Stage-2 setup.

    What this tests:
        - 32 unit vectors, all horizontal (z = 0)
        - Ray 0 points along the heading
        - Consecutive rays are 360 / 32 = 11.25 degrees apart
    """
    print_test_header(11, "Ring Ray Directions",
                      "Verify the 32-ray horizontal ring")
    yaw = np.radians(30.0)
    ring = lidar_mod.ring_directions(yaw, 32)

    cosines = np.sum(ring[:-1] * ring[1:], axis=1)
    spacing = np.degrees(np.arccos(np.clip(cosines, -1.0, 1.0)))
    print(f"\n  ring shape: {ring.shape}")
    print(f"  ring ray 0: {np.round(ring[0], 3).tolist()}  (follows the 30 degree heading)")
    print(f"  largest height (z) of any ring ray: {np.abs(ring[:, 2]).max():.4f}  (expected 0)")
    print(f"  angle between neighbouring rays: {spacing.min():.2f} to {spacing.max():.2f} degrees"
          f"  (expected 11.25)")

    assert ring.shape == (32, 3)
    assert np.allclose(np.linalg.norm(ring, axis=1), 1.0), "every ring ray must be a unit vector"
    assert np.allclose(ring[:, 2], 0.0), "ring rays must be horizontal"
    assert np.allclose(ring[0], [np.cos(yaw), np.sin(yaw), 0.0]), "ring ray 0 must follow the heading"

    assert np.allclose(cosines, np.cos(np.radians(11.25)), atol=1e-6), \
        "neighbouring ring rays must be 11.25 degrees apart"
    print("\n  \u2713 32 horizontal unit rays, 11.25 degrees apart, ray 0 follows the heading")


def test_heading_to_goal():
    """
    Test lidar.heading_to_goal().

    What this tests:
        - The yaw points from the drone towards the goal in the x-y plane
        - A drone directly above or on its goal returns the fallback heading
    """
    print_test_header(12, "Heading To Goal",
                      "Verify the yaw towards the goal and the fallback")
    heading = lidar_mod.heading_to_goal([0, 0, 2], [0, 5, 2])
    above = lidar_mod.heading_to_goal([1, 1, 2], [1, 1, 5], fallback=0.7)
    print(f"\n  goal due +y from the origin: {np.degrees(heading):.1f} degrees  (expected 90)")
    print(f"  goal directly above the drone: {above}  (expected the fallback 0.7)")

    assert np.isclose(heading, np.pi / 2)
    assert above == 0.7
    print("\n  \u2713 Heading points at the goal; falls back when directly above it")


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    """
    Entry point when script is run directly.

    Usage:
        python tests/test_stage2_observations.py

    These tests use pytest fixtures, so running the file directly simply
    hands it to pytest (with -s so the printed progress is shown).
    """
    sys.exit(pytest.main([__file__, "-q", "-s"]))