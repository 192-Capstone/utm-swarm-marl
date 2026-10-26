"""
test_checkpoint_loading.py
--------------------------
Checkpoints stay loadable on torch >= 2.6, where torch.load defaults to
weights_only=True and rejects the numpy scalars our older checkpoints hold
(best_success_rate / success_rate saved as numpy.float64).

  A. The evaluation loader (eval_suite.MappoPolicy) loads such a checkpoint
     even when torch.load's default is the strict one.
  B. The network loaders (ActorNetwork / CriticNetwork.from_checkpoint-style
     helpers) pass weights_only=False explicitly, so they do too.

Run: python -m pytest tests/test_checkpoint_loading.py -q
"""

import os
import sys

import numpy as np
import pytest
import torch
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from training.networks.actor_network import ActorNetwork

CONFIG = os.path.join(os.path.dirname(__file__), '..', 'configs', 'config_stage2_round7_teacher.yaml')


@pytest.fixture
def strict_torch_load(monkeypatch):
    """Make torch.load behave like torch >= 2.6: weights_only=True unless the caller says otherwise."""
    real = torch.load

    def load(*args, **kwargs):
        kwargs.setdefault('weights_only', True)
        return real(*args, **kwargs)

    monkeypatch.setattr(torch, 'load', load)
    return real


def numpy_checkpoint(tmp_path):
    config = yaml.safe_load(open(CONFIG, encoding='utf-8'))
    actor = ActorNetwork(config)
    path = tmp_path / 'ckpt.pth'
    torch.save({'actor_state_dict': actor.state_dict(), 'curriculum_stage': 1,
                'observation_schema': config['observation']['schema'],
                'best_success_rate': np.float64(0.93)}, path)
    return config, path


def test_strict_default_really_rejects_numpy_scalars(tmp_path, strict_torch_load):
    _, path = numpy_checkpoint(tmp_path)
    with pytest.raises(Exception):
        torch.load(path, map_location='cpu')          # what torch >= 2.6 does by default


def test_eval_loader_handles_numpy_scalars(tmp_path, strict_torch_load):
    from eval_suite import MappoPolicy
    config, path = numpy_checkpoint(tmp_path)
    policy = MappoPolicy(config, str(path), 'cpu', deterministic=True)
    assert float(policy.ckpt['best_success_rate']) == pytest.approx(0.93)


def test_trainer_and_network_loaders_pass_weights_only_false():
    """The other loaders of our own checkpoints must not rely on torch.load's default."""
    root = os.path.join(os.path.dirname(__file__), '..')
    for rel in ('training/mappo_trainer.py', 'training/networks/actor_network.py',
                'training/networks/critic_network.py', 'eval_checkpoint.py'):
        src = open(os.path.join(root, rel), encoding='utf-8').read()
        calls = [line for line in src.splitlines() if 'torch.load(' in line]
        assert calls, rel
        assert all('weights_only=False' in line for line in calls), (rel, calls)
