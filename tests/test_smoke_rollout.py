"""
test_smoke_rollout.py
----------------------
A short end-to-end smoke test: real PyBullet environment -> observation
processor -> actor network -> action -> environment, for a handful of
episodes, with no training and no saved checkpoint.

Why this exists:
    The other test files check one piece at a time (the actor network on
    made-up tensors, the environment's rewards with a scripted controller,
    the Stage-2 observation layout on a hand-built world state). None of
    them actually wire the real pieces together and run the policy inside
    the real simulator. This test does that: it is the one check that would
    catch a problem that only appears when everything runs together (a
    shape the other unit tests didn't happen to try, a NaN that only shows
    up after a few real physics steps, and so on).

    It does not check whether the policy flies well — an untrained network
    picks essentially random actions. It only checks that nothing crashes
    and that no value turns into NaN or inf along the way.

Run with:
    python tests/test_smoke_rollout.py
    python -m pytest tests/test_smoke_rollout.py -s -v

This test uses the real PyBulletAdapter (no mock), so it is slower than the
other test files — a few seconds per config, not a fraction of a second.
"""

import os
import sys

import numpy as np
import pytest
import torch
import yaml

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.swarm_env import SwarmEnv
from training.networks.actor_network import ActorNetwork

# (config file, label) — Stage 1 (no obstacles) and both Stage-2 schemas,
# so the smoke run covers every observation layout currently in use.
SMOKE_CONFIGS = [
    ("default_config.yaml", "stage1-v1"),
    ("config_stage2_obstacles.yaml", "stage2-v2"),
    ("config_stage2_ring.yaml", "stage2-v3-ring"),
]

N_STEPS = 20       # a few dozen control steps, not a full episode
N_AGENTS = 3


def print_test_header(test_num, test_name, purpose, label=None):
    """Print formatted test header (same style as the other test files)."""
    print("\n" + "=" * 60)
    suffix = f"  [{label}]" if label else ""
    print(f"TEST {test_num}: {test_name}{suffix}")
    print("=" * 60)
    print(f"Purpose: {purpose}")


def load_config(name):
    """Load one of the real configuration files from configs/, forced to
    N_AGENTS drones and a short max_steps so the smoke run is quick."""
    config_path = os.path.join(
        os.path.dirname(__file__), '..', 'configs', name
    )
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    config['environment']['n_agents'] = N_AGENTS
    config['environment']['max_steps'] = max(N_STEPS + 5, config['environment'].get('max_steps', 0))
    return config


def assert_finite(name, tensor):
    """Fail with a clear message if a tensor contains NaN or inf."""
    arr = tensor.detach().cpu().numpy() if torch.is_tensor(tensor) else np.asarray(tensor)
    assert np.isfinite(arr).all(), f"{name} contains NaN or inf: {arr}"


@pytest.fixture(params=SMOKE_CONFIGS, ids=[label for _, label in SMOKE_CONFIGS])
def smoke_setup(request):
    """
    Build one real SwarmEnv (PyBullet, headless) and one matching
    ActorNetwork for a Stage-1 or Stage-2 config, closing the environment
    afterwards even if the test fails.
    """
    name, label = request.param
    config = load_config(name)
    device = torch.device("cpu")
    env = SwarmEnv(config, device=device, gui=False)
    actor = ActorNetwork(config)
    try:
        yield config, label, env, actor, device
    finally:
        env.close()


# ============================================================================
# TEST 1: A few real episodes run without crashing
# ============================================================================

def test_rollout_runs_without_crashing(smoke_setup):
    """
    Test that the full chain (env.reset -> actor -> env.step, repeated) runs
    for a few steps without raising and without producing NaN or inf, for
    every observation schema currently in use.

    What this tests:
        - SwarmEnv.reset() and SwarmEnv.step() run against real PyBullet
        - The observation shapes SwarmEnv._build_obs() produces match what
          ActorNetwork expects (otherwise this raises a shape error
          immediately, the same way it would in real training)
        - The actor's sampled actions are finite and the environment accepts
          them back without error
        - Rewards and the done/info values stay finite and sane across
          several steps, not just the first one
    """
    config, label, env, actor, device = smoke_setup
    print_test_header(1, "Rollout Runs Without Crashing",
                      "Run env -> actor -> env for a few steps with no training", label)

    obs = env.reset()
    print(f"  n_agents: {config['environment']['n_agents']}, steps: {N_STEPS}")
    print(f"  local_fixed shape:     {tuple(obs['local_fixed'].shape)}")
    print(f"  neighbor_states shape: {tuple(obs['neighbor_states'].shape)}")
    print(f"  global_state shape:    {tuple(obs['global_state'].shape)}")
    assert_finite("local_fixed (reset)", obs['local_fixed'])
    assert_finite("global_state (reset)", obs['global_state'])

    total_reward = 0.0
    for step in range(N_STEPS):
        with torch.no_grad():
            action, log_prob = actor.get_action(
                obs['local_fixed'], obs['neighbor_states'],
                obs.get('neighbor_mask'), deterministic=False,
            )
        assert_finite(f"action (step {step})", action)
        assert_finite(f"log_prob (step {step})", log_prob)

        actions_dict = {
            f'drone_{i}': action[i].tolist() for i in range(config['environment']['n_agents'])
        }
        obs, rewards, done, info = env.step(actions_dict)

        assert_finite(f"local_fixed (step {step})", obs['local_fixed'])
        assert_finite(f"global_state (step {step})", obs['global_state'])
        for drone_id, r in rewards.items():
            assert np.isfinite(r), f"reward for {drone_id} at step {step} is not finite: {r}"
            total_reward += r

        if done:
            print(f"  episode ended at step {step}: {info.get('reason', info)}")
            break

    print(f"  ran {step + 1} step(s), total reward: {total_reward:+.2f}")
    print("\n  ✓ Full env -> actor -> env chain ran with no crash and no NaN/inf")


if __name__ == "__main__":
    """
    Entry point when script is run directly.

    Usage:
        python tests/test_smoke_rollout.py

    Hands off to pytest with -s so the printed progress is shown.
    """
    sys.exit(pytest.main([__file__, "-s", "-v"]))
