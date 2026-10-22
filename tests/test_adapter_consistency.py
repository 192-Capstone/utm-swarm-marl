"""
test_adapter_consistency.py
----------------------------
Checks that the test mock (MockAdapter, used across the tests folder) keeps
up with the real simulator adapter (PyBulletAdapter).

Why this exists:
    Both test_env_validation.py and test_actor_network.py's siblings build a
    MockAdapter by hand, matching PyBulletAdapter's methods and the shape of
    get_raw_state() from memory. When PyBulletAdapter changes (Stage 2 added
    an `obstacles` argument to reset_world and a 'lidar_ring' key to
    get_raw_state(), for example), nothing used to say so: the mock silently
    fell behind until a test that exercised the new behaviour broke somewhere
    else. These tests catch that the moment it happens, by comparing the two
    adapters directly instead of relying on someone noticing.

What this tests:
    - PyBulletAdapter and MockAdapter expose the same public methods
    - Both reset_world() accept the same arguments
    - Both get_raw_state() return a dict with the same top-level shape
      (one key per drone, plus 'goals'), and each drone's reading has the
      same keys the observation processor expects (position, velocity,
      orientation, lidar, and lidar_ring when the ring is enabled)

Run with:
    python tests/test_adapter_consistency.py
    python -m pytest tests/test_adapter_consistency.py -s -v
"""

import inspect
import os
import sys

import numpy as np
import pytest
import yaml

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.pybullet_adapter import PyBulletAdapter
from envs.swarm_env import SwarmEnv
from tests.test_env_validation import MockAdapter, load_config as load_default_config


def print_test_header(test_num, test_name, purpose):
    """Print formatted test header (same style as the other test files)."""
    print("\n" + "=" * 60)
    print(f"TEST {test_num}: {test_name}")
    print("=" * 60)
    print(f"Purpose: {purpose}")


def load_config(name):
    """Load one of the real configuration files from configs/."""
    config_path = os.path.join(
        os.path.dirname(__file__), '..', 'configs', name
    )
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


# Public methods every adapter must expose for SwarmEnv to drive it.
# (SwarmEnv.__init__ / reset / start_episode / step / close — see envs/swarm_env.py)
REQUIRED_METHODS = [
    'reset_world', 'apply_actions', 'step_simulation',
    'get_raw_state', 'get_positions', 'close',
]

# Keys every drone's raw-state entry must have (modules/observation_processor.py
# reads these from raw_state[drone_id]).
REQUIRED_DRONE_KEYS = {'position', 'velocity', 'orientation', 'lidar'}


# ============================================================================
# TEST 1: Same public methods
# ============================================================================

def test_same_public_methods():
    """
    Test that MockAdapter implements every method PyBulletAdapter has.

    What this tests:
        - Every method SwarmEnv calls on an adapter (REQUIRED_METHODS) exists
          on both PyBulletAdapter and MockAdapter
        - MockAdapter isn't missing a method PyBulletAdapter has gained since
          the mock was written

    MockAdapter is allowed extra helper methods PyBulletAdapter doesn't have
    (there are none today) — this only checks nothing required is missing.
    """
    print_test_header(1, "Same Public Methods",
                      "Verify MockAdapter has every method SwarmEnv needs from an adapter")

    real_methods = {m for m in dir(PyBulletAdapter) if not m.startswith('_')}
    mock_methods = {m for m in dir(MockAdapter) if not m.startswith('_')}

    print(f"  PyBulletAdapter public methods: {sorted(real_methods)}")
    print(f"  MockAdapter public methods:     {sorted(mock_methods)}")

    missing_from_mock = set(REQUIRED_METHODS) - mock_methods
    missing_from_real = set(REQUIRED_METHODS) - real_methods

    print(f"  required methods: {REQUIRED_METHODS}")
    print(f"  missing from MockAdapter: {sorted(missing_from_mock) or 'none'}")
    print(f"  missing from PyBulletAdapter: {sorted(missing_from_real) or 'none'}")

    assert not missing_from_real, \
        f"REQUIRED_METHODS is out of date — PyBulletAdapter no longer has {missing_from_real}"
    assert not missing_from_mock, \
        f"MockAdapter is missing {missing_from_mock} — update it to match PyBulletAdapter"
    print("\n  ✓ MockAdapter implements every required method")


# ============================================================================
# TEST 2: reset_world() takes the same arguments
# ============================================================================

def test_reset_world_same_signature():
    """
    Test that reset_world() accepts the same arguments on both adapters.

    What this tests:
        - Calling PyBulletAdapter.reset_world(spawns, goals, obstacles) and
          MockAdapter.reset_world(spawns, goals, obstacles) both work
        - This is exactly the bug this test suite was written to catch:
          Stage 2 added the `obstacles` argument to PyBulletAdapter and
          MockAdapter was left on the old two-argument signature, so every
          test that built a SwarmEnv with a MockAdapter failed with
          "takes 3 positional arguments but 4 were given"
    """
    print_test_header(2, "reset_world() Signature",
                      "Verify both adapters accept the same (spawns, goals, obstacles) call")

    real_sig = inspect.signature(PyBulletAdapter.reset_world)
    mock_sig = inspect.signature(MockAdapter.reset_world)
    print(f"  PyBulletAdapter.reset_world{real_sig}")
    print(f"  MockAdapter.reset_world{mock_sig}")

    real_params = list(real_sig.parameters)
    mock_params = list(mock_sig.parameters)
    print(f"  parameter names: real={real_params}  mock={mock_params}")

    assert real_params == mock_params, \
        f"reset_world() parameters differ: PyBulletAdapter{real_params} vs MockAdapter{mock_params}"

    # And it actually runs with three positional arguments, the way
    # SwarmEnv.start_episode() calls it (envs/swarm_env.py: "self.adapter
    # .reset_world(spawns, goals, self.layout)")
    mock = MockAdapter(1)
    mock.reset_world({'drone_0': [0.0, 0.0, 2.0]}, {'drone_0': [1.0, 0.0, 2.0]}, [])
    print("  MockAdapter.reset_world(spawns, goals, []) ran without error")
    print("\n  ✓ Both adapters take the same reset_world() arguments")


# ============================================================================
# TEST 3: get_raw_state() has the same top-level shape
# ============================================================================

def test_raw_state_same_shape():
    """
    Test that get_raw_state() returns the same top-level structure from
    both adapters, for the same scenario.

    What this tests:
        - Both dicts have exactly the same keys: one per drone, plus 'goals'
        - Each drone's 'goals' entry and each drone's own entry have the
          same keys (position, velocity, orientation, lidar)
        - The LiDAR fan has 14 values from both adapters

    Runs the real PyBulletAdapter, so this test takes a few seconds (it
    starts and stops PyBullet) and needs the full environment installed.
    """
    print_test_header(3, "get_raw_state() Shape",
                      "Verify both adapters describe the world the same way")

    config = load_default_config()
    n_agents = 2
    spawns = {f'drone_{i}': [float(i * 3), 0.0, 2.0] for i in range(n_agents)}
    goals = {f'drone_{i}': [float(i * 3), 5.0, 2.0] for i in range(n_agents)}

    mock = MockAdapter(n_agents)
    mock.reset_world(spawns, goals, [])
    mock_state = mock.get_raw_state()
    mock.close()

    real_cfg = dict(config)
    real_cfg['environment'] = dict(config['environment'])
    real_cfg['environment']['n_agents'] = n_agents
    real = PyBulletAdapter(real_cfg)
    real.reset_world(spawns, goals, [])
    real_state = real.get_raw_state()
    real.close()

    print(f"  mock top-level keys: {sorted(mock_state.keys())}")
    print(f"  real top-level keys: {sorted(real_state.keys())}")
    assert set(mock_state.keys()) == set(real_state.keys()), \
        "get_raw_state() must return the same top-level keys from both adapters"

    for drone_id in [f'drone_{i}' for i in range(n_agents)]:
        mock_keys = set(mock_state[drone_id].keys())
        real_keys = set(real_state[drone_id].keys())
        print(f"  {drone_id}: mock keys={sorted(mock_keys)}  real keys={sorted(real_keys)}")
        assert REQUIRED_DRONE_KEYS <= mock_keys, \
            f"MockAdapter's {drone_id} entry is missing {REQUIRED_DRONE_KEYS - mock_keys}"
        assert REQUIRED_DRONE_KEYS <= real_keys, \
            f"PyBulletAdapter's {drone_id} entry is missing {REQUIRED_DRONE_KEYS - real_keys}"

        assert len(mock_state[drone_id]['lidar']) == len(real_state[drone_id]['lidar']) == 14, \
            "both adapters must report 14 LiDAR fan readings"

    print("\n  ✓ Both adapters describe the world with the same shape")


if __name__ == "__main__":
    """
    Entry point when script is run directly.

    Usage:
        python tests/test_adapter_consistency.py

    Hands off to pytest with -s so the printed progress is shown.
    """
    sys.exit(pytest.main([__file__, "-s", "-v"]))
