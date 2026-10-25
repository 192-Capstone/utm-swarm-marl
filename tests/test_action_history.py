"""
test_action_history.py
-----------------------
Observation schema v4: the policy also sees its own last K velocity commands
(observation.action_history), because a real quadrotor reacts 0.1-0.3 s late
and the commands still taking effect are invisible in the drone's own state.

  A. Config: v4 needs action_history >= 1 and local_obs_dim = v3 + 3K.
  B. The first inputs of v4 are exactly the v3 inputs; the history is appended.
  C. In SwarmEnv the most recent command is in the first history slot, older
     ones shift back one slot per step.
  D. What is recorded is the command actually sent (clipped to v_max like the
     adapter does) and sensor noise never touches it.
  E. The history is cleared at the start of every episode.
  F. v3 environments are unchanged (no history kept or passed on).
  G. Converting a v3 actor to v4 gives exactly the same actions, whatever the
     history inputs contain, for both an actor-only and a full checkpoint.
  H. configs/config_stage2_round7_v4.yaml is the round-7 config with only the
     observation block changed to v4.

Run with:
    python -m pytest tests/test_action_history.py -s -v
"""

import copy
import os
import sys

import numpy as np
import pytest
import torch
import yaml

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

from envs.swarm_env import SwarmEnv
from modules.observation_processor import ObservationProcessor, build_dummy_raw_state
from training.networks.actor_network import ActorNetwork
from tests.test_env_validation import MockAdapter
from convert_checkpoint_inputs import pad_v3_to_v4

ROUND7 = os.path.join(ROOT, 'configs', 'config_stage2_round7_teacher.yaml')
ROUND7_V4 = os.path.join(ROOT, 'configs', 'config_stage2_round7_v4.yaml')
K = 8           # commands in the history for these tests
V_MAX = 2.0


def print_test_header(test_num, test_name, purpose):
    """Print formatted test header (same style as the other test files)."""
    print("\n" + "=" * 60)
    print(f"TEST {test_num}: {test_name}")
    print("=" * 60)
    print(f"Purpose: {purpose}")


def load(path):
    with open(path, encoding='utf-8') as f:
        return yaml.safe_load(f)


def v3_config(n_agents=2):
    config = load(ROUND7)
    config['environment']['n_agents'] = n_agents
    return config


def v4_config(n_agents=2, k=K):
    """The round-7 (v3) config switched to schema v4 with k commands."""
    config = v3_config(n_agents)
    obs = config['observation']
    v3_dim = obs['local_obs_dim']
    obs.update(schema='v4', action_history=k, local_obs_dim=v3_dim + 3 * k,
               total_obs_dim=v3_dim + 3 * k + obs['neighbor_context_dim'])
    return config


def make_env(config):
    """SwarmEnv on the scripted MockAdapter (no PyBullet), two drones, no obstacles."""
    n = config['environment']['n_agents']
    env = SwarmEnv(config, adapter=MockAdapter(n))
    spawns = {f'drone_{i}': [5.0 * i, 0.0, 2.0] for i in range(n)}
    goals = {f'drone_{i}': [5.0 * i, 10.0, 2.0] for i in range(n)}
    obs = env.start_episode(spawns, goals, obstacles=[])
    return env, obs, spawns, goals


def history_slots(obs, v3_dim):
    """(N, K, 3) view of the history inputs in a v4 observation."""
    lf = obs['local_fixed'].numpy()
    return lf[:, v3_dim:].reshape(lf.shape[0], -1, 3)


# ============================================================================
# TEST A: config checks
# ============================================================================

def test_v4_config_checks():
    print_test_header('A', "v4 Config Checks",
                      "v4 needs action_history >= 1 and local_obs_dim = v3 + 3K")
    cpu = torch.device('cpu')
    proc = ObservationProcessor(v4_config(), cpu)
    print(f"  v4 with K={K}: local_obs_dim {proc.local_obs_dim}, action_history {proc.action_history}")
    assert proc.action_history == K
    assert proc.local_obs_dim == 74 + 3 * K

    bad = v4_config()
    bad['observation']['local_obs_dim'] += 1
    with pytest.raises(ValueError):
        ObservationProcessor(bad, cpu)
    no_k = v4_config()
    del no_k['observation']['action_history']
    with pytest.raises(ValueError):
        ObservationProcessor(no_k, cpu)
    print("  wrong local_obs_dim -> ValueError, missing action_history -> ValueError")

    assert ObservationProcessor(v3_config(), cpu).action_history == 0
    print("  v3 processor: action_history 0")


# ============================================================================
# TEST B: v4 = v3 inputs + history
# ============================================================================

def test_v4_starts_with_the_v3_inputs():
    print_test_header('B', "v4 = v3 Inputs + History",
                      "The first 74 v4 inputs equal the v3 inputs; the history is appended")
    cpu = torch.device('cpu')
    np.random.seed(0)
    raw = build_dummy_raw_state(2)
    p3 = ObservationProcessor(v3_config(), cpu)
    p4 = ObservationProcessor(v4_config(), cpu)
    lf3 = p3.process(raw, 1)[0].numpy()
    lf4 = p4.process(raw, 1)[0].numpy()
    print(f"  v3 shape {lf3.shape}, v4 shape {lf4.shape}")
    assert lf4.shape == (2, 74 + 3 * K)
    assert np.array_equal(lf4[:, :74], lf3)
    assert np.all(lf4[:, 74:] == 0.0), "no history given -> zeros"

    hist = np.random.uniform(-V_MAX, V_MAX, (2, K, 3)).astype(np.float32)
    p4.reset_history()
    lf4h = p4.process(raw, 1, action_history=hist)[0].numpy()
    assert np.allclose(lf4h[:, 74:], hist.reshape(2, -1) / V_MAX)
    print("  first 74 inputs identical to v3; history = commands / v_max, most recent first")

    with pytest.raises(ValueError):
        p4.process(raw, 1, action_history=hist[:, :K - 1])


# ============================================================================
# TEST C: most recent command first, older ones shift back
# ============================================================================

def test_history_most_recent_first():
    print_test_header('C', "History Order",
                      "The latest command is in slot 0 and moves back one slot per step")
    env, obs, _, _ = make_env(v4_config())
    a = np.array([[0.5, -0.2, 0.1], [1.0, 0.0, -0.4]], dtype=np.float32)
    b = np.array([[-0.3, 0.6, 0.0], [0.2, 0.2, 0.2]], dtype=np.float32)
    act = lambda cmd: {f'drone_{i}': cmd[i].tolist() + [0.0] for i in range(2)}

    obs, _, _, _ = env.step(act(a))
    h = history_slots(obs, 74)
    print(f"  after 1 step, slot 0 = {h[0, 0]}  (sent {a[0]} / {V_MAX})")
    assert np.allclose(h[:, 0], a / V_MAX) and np.all(h[:, 1:] == 0.0)

    obs, _, _, _ = env.step(act(b))
    h = history_slots(obs, 74)
    print(f"  after 2 steps, slot 0 = {h[0, 0]}, slot 1 = {h[0, 1]}")
    assert np.allclose(h[:, 0], b / V_MAX) and np.allclose(h[:, 1], a / V_MAX)
    assert np.all(h[:, 2:] == 0.0)

    for _ in range(K):                          # older commands drop off the end
        obs, _, _, _ = env.step(act(b))
    assert np.allclose(history_slots(obs, 74), (b / V_MAX)[:, None, :])
    print(f"  after {K} more steps the first command has dropped off the end")
    env.close()


# ============================================================================
# TEST D: recorded = command actually sent, no sensor noise
# ============================================================================

def test_history_is_clipped_and_noise_free():
    print_test_header('D', "Clipped and Noise-Free",
                      "Commands are recorded as sent (clipped to v_max), sensor noise never touches them")
    config = v4_config()
    assert any(float(v) > 0 for k, v in config['environment']['sensor_noise'].items() if k != 'seed'), \
        "the round-7 config has sensor noise on, which is what this test relies on"
    env, obs, _, _ = make_env(config)
    cmd = np.array([[5.0, -5.0, 0.5], [-3.0, 0.4, 9.0]], dtype=np.float32)
    obs, _, _, _ = env.step({f'drone_{i}': cmd[i].tolist() + [0.0] for i in range(2)})
    h = history_slots(obs, 74)
    expected = np.clip(cmd, -V_MAX, V_MAX) / V_MAX
    print(f"  sent {cmd[0]} -> recorded {h[0, 0]} (expected {expected[0]})")
    assert np.array_equal(h[:, 0], expected), "exact match: clipped, and no noise added"
    assert np.all(np.abs(h) <= 1.0)
    env.close()


# ============================================================================
# TEST E: cleared every episode
# ============================================================================

def test_history_resets_each_episode():
    print_test_header('E', "Reset Each Episode",
                      "start_episode() clears the history")
    env, obs, spawns, goals = make_env(v4_config())
    for _ in range(3):
        obs, _, _, _ = env.step({f'drone_{i}': [1.0, 1.0, 0.0, 0.0] for i in range(2)})
    assert np.any(history_slots(obs, 74) != 0.0)
    obs = env.start_episode(spawns, goals, obstacles=[])
    print(f"  after 3 steps and a new episode: max |history| = {np.abs(history_slots(obs, 74)).max()}")
    assert np.all(history_slots(obs, 74) == 0.0)
    env.close()


# ============================================================================
# TEST F: v3 unchanged
# ============================================================================

def test_v3_env_keeps_no_history():
    print_test_header('F', "v3 Unchanged",
                      "A v3 environment keeps no history and still gives 74 inputs")
    env, obs, _, _ = make_env(v3_config())
    obs, _, _, _ = env.step({f'drone_{i}': [1.0, 0.0, 0.0, 0.0] for i in range(2)})
    print(f"  history_len {env.history_len}, local_fixed width {obs['local_fixed'].shape[1]}")
    assert env.history_len == 0 and obs['local_fixed'].shape[1] == 74
    env.close()


# ============================================================================
# TEST G: v3 -> v4 conversion keeps the actions
# ============================================================================

class FakeArgs:
    """The attributes pad_v3_to_v4 reads from the script's command-line arguments."""
    def __init__(self, config, checkpoint, out):
        self.config, self.checkpoint, self.out = str(config), checkpoint, str(out)


@pytest.mark.parametrize('kind', ['policy', 'resume'])
def test_v3_to_v4_conversion_preserves_actions(tmp_path, kind):
    print_test_header('G', f"v3 -> v4 Conversion ({kind})",
                      "The converted v4 actor gives the same actions, whatever the history holds")
    cfg3, cfg4 = v3_config(), v4_config()
    cfg4_path = tmp_path / 'v4.yaml'
    with open(cfg4_path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(cfg4, f)

    torch.manual_seed(0)
    actor3 = ActorNetwork(cfg3)
    ckpt = {'actor_state_dict': actor3.state_dict(), 'observation_schema': 'v3'}
    if kind == 'resume':                         # a full training checkpoint carries more
        ckpt.update(critic_state_dict={'w': torch.ones(3)}, actor_optim={'dropped': True},
                    episode_count=500)
    args = FakeArgs(cfg4_path, f'<in-memory v3 {kind}>', tmp_path / 'v4.pth')
    pad_v3_to_v4(ckpt, cfg4, args)               # writes args.out; exits if its own check fails

    out = torch.load(args.out, map_location='cpu', weights_only=False)
    assert out['observation_schema'] == 'v4'
    if kind == 'resume':
        assert torch.equal(out['critic_state_dict']['w'], torch.ones(3)) and out['episode_count'] == 500
        assert 'actor_optim' not in out, "the actor optimizer no longer fits and is dropped"
    actor4 = ActorNetwork(cfg4)
    actor4.load_state_dict(out['actor_state_dict'])

    rng = np.random.default_rng(1)
    n = 256
    f3 = torch.tensor(rng.uniform(-1, 1, (n, 74)), dtype=torch.float32)
    f4 = torch.cat([f3, torch.tensor(rng.uniform(-1, 1, (n, 3 * K)), dtype=torch.float32)], 1)
    ns = torch.tensor(rng.normal(size=(n, 5, 6)), dtype=torch.float32)
    nm = torch.tensor(rng.uniform(size=(n, 5)) < 0.4)
    with torch.no_grad():
        diff = (actor3(f3, ns, nm)[0] - actor4(f4, ns, nm)[0]).abs().max().item()
    print(f"  {n} random states with random histories: max action difference {diff:.2e}")
    assert diff < 1e-5


# ============================================================================
# TEST H: the v4 config file
# ============================================================================

def test_round7_v4_config_file():
    print_test_header('H', "Round-7 v4 Config",
                      "config_stage2_round7_v4.yaml = round-7 config with only the observation block changed")
    base, v4 = load(ROUND7), load(ROUND7_V4)
    obs = v4['observation']
    k = obs['action_history']
    print(f"  schema {obs['schema']}, action_history {k}, local_obs_dim {obs['local_obs_dim']}, "
          f"total_obs_dim {obs['total_obs_dim']}")
    assert obs['schema'] == 'v4' and k >= 1
    assert obs['local_obs_dim'] == base['observation']['local_obs_dim'] + 3 * k
    assert obs['total_obs_dim'] == obs['local_obs_dim'] + obs['neighbor_context_dim']
    assert {key: val for key, val in v4.items() if key != 'observation'} == \
           {key: val for key, val in base.items() if key != 'observation'}
    changed = {key for key in set(obs) | set(base['observation'])
               if obs.get(key) != base['observation'].get(key)}
    print(f"  changed observation keys: {sorted(changed)}")
    assert changed == {'schema', 'local_obs_dim', 'total_obs_dim', 'action_history'}
    ObservationProcessor(v4, torch.device('cpu'))      # builds without errors


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-s", "-v"]))
