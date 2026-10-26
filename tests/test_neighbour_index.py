"""
test_neighbour_index.py
-----------------------
Neighbour search through a spatial index (modules/neighbour_index.py).

  A. Every index returns a superset: each drone within the sensing radius is a candidate.
  B. Through the real ObservationProcessor, grid and h3 pick exactly the same K neighbours
     (same slots, same values, same mask) as brute force, from 2 to 300 drones, with clusters.
  C. Brute force stays the default; grid / h3 without a sensing radius are refused.

Run: python -m pytest tests/test_neighbour_index.py -q
"""

import importlib.util
import os
import sys

import numpy as np
import pytest
import torch
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from modules.neighbour_index import NeighbourIndex                # noqa: E402
from modules.observation_processor import ObservationProcessor    # noqa: E402

CONFIG = os.path.join(os.path.dirname(__file__), '..', 'configs', 'config_stage2_round7_teacher.yaml')
HAS_H3 = importlib.util.find_spec('h3') is not None
KINDS = ['grid', pytest.param('h3', marks=pytest.mark.skipif(not HAS_H3, reason='h3 not installed'))]


def swarm(rng, n, cluster):
    side = 30 * np.sqrt(n / 8) * rng.uniform(0.5, 3)
    p = np.column_stack([rng.uniform(-side, side, (n, 2)), rng.uniform(1, 5, n)]).astype(np.float32)
    if cluster:
        p[:n // 2, :2] = rng.normal(0, 5, (n // 2, 2))
    return p


@pytest.mark.parametrize('kind', KINDS)
def test_candidates_contain_every_drone_in_range(kind):
    rng = np.random.default_rng(0)
    index = NeighbourIndex(kind, 30.0)
    for n in (2, 20, 200):
        for cluster in (False, True):
            p = swarm(rng, n, cluster)
            cands = index.candidates(p)
            d = np.linalg.norm(p[:, None, :] - p[None, :, :], axis=2)
            for i in range(n):
                inside = set(np.flatnonzero((d[i] <= 30.0) & (np.arange(n) != i)).tolist())
                assert inside <= set(cands[i].tolist())
                assert i not in cands[i]


@pytest.mark.parametrize('kind', KINDS)
def test_same_neighbours_as_brute_force(kind):
    cfg = yaml.safe_load(open(CONFIG, encoding='utf-8'))
    brute = ObservationProcessor(cfg, torch.device('cpu'))
    other = ObservationProcessor({**cfg, 'observation': {**cfg['observation'], 'neighbor_index': kind}},
                                 torch.device('cpu'))
    rng = np.random.default_rng(1)
    for n in (2, 8, 50, 300):
        for cluster in (False, True):
            p = swarm(rng, n, cluster)
            v = rng.normal(0, 1, (n, 3)).astype(np.float32)
            a, b = brute._build_neighbor_states(p, v), other._build_neighbor_states(p, v)
            assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


def test_defaults_and_errors():
    cfg = yaml.safe_load(open(CONFIG, encoding='utf-8'))
    assert ObservationProcessor(cfg, torch.device('cpu')).neighbour_index.kind == 'brute'
    with pytest.raises(ValueError):
        NeighbourIndex('grid', None)
    with pytest.raises(ValueError):
        NeighbourIndex('kdtree', 30.0)
