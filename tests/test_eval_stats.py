"""
test_eval_stats.py
------------------
Checks the statistics helpers in eval_stats.py against known values.

Run: python -m pytest tests/test_eval_stats.py -q
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from eval_stats import wilson, mcnemar_exact, holm, spread


def test_wilson_known_values():
    lo, hi = wilson(97, 100)
    assert abs(lo - 0.9155) < 1e-3 and abs(hi - 0.9897) < 1e-3
    lo, hi = wilson(0, 50)
    assert lo == 0.0 and 0.05 < hi < 0.08          # never a zero-width interval at k = 0
    lo, hi = wilson(50, 50)
    assert hi == 1.0 and 0.92 < lo < 0.95
    assert all(math.isnan(x) for x in wilson(0, 0))


def test_mcnemar_exact():
    assert mcnemar_exact(0, 0) == 1.0
    assert abs(mcnemar_exact(0, 6) - 0.03125) < 1e-12     # 2 * 0.5**6
    assert mcnemar_exact(3, 3) == 1.0
    assert mcnemar_exact(10, 2) == mcnemar_exact(2, 10)   # symmetric
    assert 0.03 < mcnemar_exact(10, 2) < 0.04             # 2 * P(X <= 2), X ~ Bin(12, 0.5) = 0.0386


def test_holm():
    assert holm([0.01, 0.04, 0.03]) == [0.03, 0.06, 0.06]
    assert holm([0.5]) == [0.5]
    assert holm([0.6, 0.6]) == [1.0, 1.0]                 # capped at 1


def test_spread():
    mean, sd, lo, hi = spread([90.0, 92.0, 94.0])
    assert mean == 92.0 and abs(sd - 2.0) < 1e-12 and lo == 90.0 and hi == 94.0
    assert spread([93.8])[1] == 0.0
