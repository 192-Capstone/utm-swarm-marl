"""
test_checkpoint_conversion.py
-------------------------------
Unit test for scripts/convert_checkpoint_inputs.py's v2 -> v3 conversion
(pad_v2_to_v3): turning a checkpoint trained on the 42-input schema v2 into
one that runs on the 74-input schema v3 (the 32-ray LiDAR ring), without
retraining.

Why this works (see the docstring of scripts/convert_checkpoint_inputs.py):
    The 32 new ring inputs are appended after the 42 v2 inputs with zero
    weights in local_proj and the first MLP layer, so the network's output
    is mathematically unchanged by their presence until training adjusts
    those weights. This test checks that directly: build a small v2 actor,
    run it through the real pad_v2_to_v3() function, and confirm the
    resulting v3 actor gives the same action as the original v2 actor on
    the same situations, whatever the (made-up) ring reports.

    pad_v2_to_v3() already runs this exact check itself every time it
    converts a real checkpoint (sys.exit'ing if it fails) — this test runs
    the real function, not a re-implementation of its logic, so it is
    exercised on every test run rather than only when someone happens to
    run the conversion script by hand on a real checkpoint.

pad_v2_to_v3() writes its result to args.out as a side effect (that is how
the real script is used), so this test gives it a throwaway path under
pytest's tmp_path and reads the actor weights back out of it.

Run with:
    python tests/test_checkpoint_conversion.py
    python -m pytest tests/test_checkpoint_conversion.py -s -v
"""

import copy
import os
import sys

import numpy as np
import pytest
import torch
import yaml

# Add parent directory to path, then scripts/ (convert_checkpoint_inputs.py
# is a script, not a package, so it needs its own directory on sys.path)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'scripts'))

from training.networks.actor_network import ActorNetwork
from convert_checkpoint_inputs import V2_DIM, pad_v2_to_v3


def print_test_header(test_num, test_name, purpose):
    """Print formatted test header (same style as the other test files)."""
    print("\n" + "=" * 60)
    print(f"TEST {test_num}: {test_name}")
    print("=" * 60)
    print(f"Purpose: {purpose}")


def config_path(name):
    """Path to one of the real configuration files under configs/."""
    return os.path.join(os.path.dirname(__file__), '..', 'configs', name)


def load_config(name):
    """Load one of the real configuration files from configs/."""
    with open(config_path(name), 'r') as f:
        return yaml.safe_load(f)


class FakeArgs:
    """
    Stand-in for the real script's parsed command-line arguments.

    pad_v2_to_v3() reads:
        args.config     : re-loaded from disk to build the v2 actor
        args.checkpoint : only copied into the output dict's 'converted_from'
        args.out        : where it writes the converted checkpoint (torch.save)
    """
    def __init__(self, config, checkpoint, out):
        self.config = config
        self.checkpoint = checkpoint
        self.out = str(out)


# ============================================================================
# TEST 1: v2 -> v3 conversion preserves actions exactly
# ============================================================================

def test_v2_to_v3_conversion_preserves_actions(tmp_path):
    """
    Test that converting a v2 checkpoint to v3 (the real pad_v2_to_v3)
    doesn't change what the actor does, even though it now has 32 extra
    ring inputs.

    What this tests:
        - pad_v2_to_v3() runs end to end on a v2 checkpoint (an untrained
          actor stands in for a real one — it only needs valid weights,
          not a trained policy) and writes a v3 checkpoint
        - Its own internal check (comparing old vs new on 4096 random
          states) passes, i.e. it does not sys.exit("conversion check
          failed, ...")
        - Loading the written checkpoint's actor and feeding it v2-style
          inputs plus arbitrary values in the new ring slots gives exactly
          the same action as the original v2 actor on the same 42 values,
          across many random ring readings (confirming the ring's weights
          really are zero, not just zero in one lucky case)
    """
    print_test_header(1, "v2 -> v3 Conversion Preserves Actions",
                      "Verify the ring's new inputs start at zero weight, so v2 behaviour is kept")

    config_v3 = load_config("config_stage2_ring.yaml")
    config_v2 = copy.deepcopy(config_v3)
    config_v2['observation'].update(
        schema='v2', local_obs_dim=V2_DIM,
        total_obs_dim=V2_DIM + config_v3['observation']['neighbor_context_dim'],
    )

    # A small untrained v2 actor stands in for a real checkpoint: the
    # conversion only depends on the weights, not on how they were trained.
    v2_actor = ActorNetwork(config_v2)
    ckpt = {'actor_state_dict': v2_actor.state_dict()}
    args = FakeArgs(config=config_path("config_stage2_ring.yaml"),
                    checkpoint="<in-memory v2 actor, no real checkpoint file>",
                    out=tmp_path / "converted_v3.pth")

    pad_v2_to_v3(ckpt, config_v3, args)   # writes args.out; raises/exits on failure
    assert os.path.exists(args.out), "pad_v2_to_v3 should have written the converted checkpoint"

    converted = torch.load(args.out, map_location='cpu', weights_only=False)
    print(f"  converted checkpoint observation_schema: {converted.get('observation_schema')}")
    print(f"  converted checkpoint converted_from:      {converted.get('converted_from')}")
    assert converted['observation_schema'] == 'v3'

    v3_actor = ActorNetwork(config_v3)
    v3_actor.load_state_dict(converted['actor_state_dict'])

    rng = np.random.default_rng(1)
    n, k = 256, 5
    ring_dim = config_v3['observation']['local_obs_dim'] - V2_DIM
    v2_obs = torch.tensor(rng.uniform(-1, 1, (n, V2_DIM)), dtype=torch.float32)
    ring_obs = torch.tensor(rng.uniform(0, 1, (n, ring_dim)), dtype=torch.float32)
    v3_obs = torch.cat([v2_obs, ring_obs], dim=1)
    neighbor_states = torch.tensor(rng.normal(size=(n, k, 6)), dtype=torch.float32)
    neighbor_mask = torch.tensor(rng.uniform(size=(n, k)) < 0.4)

    with torch.no_grad():
        action_v2, _ = v2_actor(v2_obs, neighbor_states, neighbor_mask)
        action_v3, _ = v3_actor(v3_obs, neighbor_states, neighbor_mask)

    max_diff = (action_v2 - action_v3).abs().max().item()
    print(f"  {n} random states, {ring_dim} random ring readings per state")
    print(f"  max action difference (original v2 actor vs converted v3 actor): {max_diff:.2e}")

    assert max_diff < 1e-5, \
        f"converted v3 actor should give the same action as the original v2 actor, max diff {max_diff:.2e}"
    print("\n  ✓ The converted actor behaves exactly like the original, whatever the ring sees")


# ============================================================================
# TEST 2: the target config is consistent with a v3 (ring) checkpoint
# ============================================================================

def test_target_config_is_consistent_v3():
    """
    Test that config_stage2_ring.yaml, the config pad_v2_to_v3() converts
    towards, is internally consistent.

    What this tests:
        - observation.schema is 'v3'
        - local_obs_dim is exactly V2_DIM (42) + 32 ring rays
          (ObservationProcessor.__init__ already enforces this at runtime;
          this test catches the same mistake earlier, without constructing
          a processor)
    """
    print_test_header(2, "Target Config Is a Valid v3 Config",
                      "Verify config_stage2_ring.yaml matches the v3 (ring) schema")

    config_v3 = load_config("config_stage2_ring.yaml")
    print(f"  observation.schema:        {config_v3['observation']['schema']}")
    print(f"  observation.local_obs_dim: {config_v3['observation']['local_obs_dim']}"
          f"  (expected {V2_DIM} + 32 = {V2_DIM + 32})")

    assert config_v3['observation']['schema'] == 'v3'
    assert config_v3['observation']['local_obs_dim'] == V2_DIM + 32
    print("\n  ✓ The target config is a valid v3 (ring) config")


if __name__ == "__main__":
    """
    Entry point when script is run directly.

    Usage:
        python tests/test_checkpoint_conversion.py

    Hands off to pytest with -s so the printed progress is shown.
    """
    sys.exit(pytest.main([__file__, "-s", "-v"]))
