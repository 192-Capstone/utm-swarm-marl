"""
test_c5_compare.py
------------------
C5 part 4 (PyBullet vs Webots, paired): the statistics and pairing of scripts/c5_compare.py and
the new pieces of the evaluation scripts (scripts/eval_city_routes.py, scripts/eval_e1_flown.py).

  A. Wilcoxon signed-rank: exact p equals brute-force enumeration (ties, zeros), known values.
  B. City routes: episode-level McNemar counts, Holm over the maps, pooled row, scenario check,
     Wilcoxon on cross-track p95.
  C. E1: per-flight pairing by (seed, condition, flight id) whatever the row order, episode level,
     approvals check, late_s only where both settled.
  D. eval_e1_flown.per_flight and eval_city_routes.scenario_id; seeds 10-19 refused (also for a
     worker); unknown simulators refused.

Run: python -m pytest tests/test_c5_compare.py -q
"""
import itertools
import json
import os
import subprocess
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

import c5_compare as cc                                            # noqa: E402
from eval_stats import mcnemar_exact, holm                         # noqa: E402


# ======================================================== A. Wilcoxon
def brute_wilcoxon(d):
    """Two-sided p by enumerating all 2^n sign patterns (zeros dropped, average ranks)."""
    d = [x for x in d if x != 0]
    n = len(d)
    a = np.abs(d)
    srt = sorted(a)
    ranks = np.array([np.mean([j + 1 for j in range(n) if srt[j] == v]) for v in a])   # average ranks
    w = ranks[np.array(d) > 0].sum()
    ws = [sum(r for r, s in zip(ranks, signs) if s) for signs in itertools.product([0, 1], repeat=n)]
    lo = np.mean([x <= w + 1e-9 for x in ws])
    hi = np.mean([x >= w - 1e-9 for x in ws])
    return min(1.0, 2 * min(lo, hi))


@pytest.mark.parametrize('d', [[1, 2, 3, 4, 5, 6], [-1.5, 2, 0, 3, -3, 0.5, 2, 4],
                               [0.1, 0.1, -0.1, 0.2, 0.2, -0.3, 0.05], [3, -1, -2, 0, 0, 5, -5, 1, 1]])
def test_wilcoxon_matches_enumeration(d):
    p, n, _ = cc.wilcoxon_signed_rank(d)
    assert n == sum(1 for x in d if x != 0)
    assert p == pytest.approx(brute_wilcoxon(d), abs=1e-12)


def test_wilcoxon_random_samples_match_enumeration():
    rng = np.random.RandomState(20)
    for _ in range(20):
        d = np.round(rng.normal(0.2, 1.0, rng.randint(3, 11)), 1).tolist()   # rounded: ties and zeros
        assert cc.wilcoxon_signed_rank(d)[0] == pytest.approx(brute_wilcoxon(d), abs=1e-12)


def test_wilcoxon_known_values_and_edge_cases():
    assert cc.wilcoxon_signed_rank([1, 2, 3, 4, 5, 6])[0] == pytest.approx(2 / 64)
    assert cc.wilcoxon_signed_rank([0, 0, 0]) == (1.0, 0, 0.0)
    p_pos = cc.wilcoxon_signed_rank([0.5, -0.2, 1.0, 0.3])[0]
    assert cc.wilcoxon_signed_rank([-0.5, 0.2, -1.0, -0.3])[0] == pytest.approx(p_pos)   # symmetric
    big = cc.wilcoxon_signed_rank(np.random.RandomState(21).normal(0.3, 1.0, 400))
    assert big[1] == 400 and 0.0 <= big[0] < 1e-3                     # large n runs, shift detected


# ======================================================== B. city routes
def _route_rows(outcomes, seed=20, xt=0.3, sid='abc'):
    """outcomes: list over episodes of list over drones of (settled, crash, stuck)."""
    rows = []
    for ep, drones in enumerate(outcomes):
        for i, (s, c, st) in enumerate(drones):
            rows.append({'seed': seed, 'episode': ep, 'drone': i, 'settled': s, 'crashed': c,
                         'episode_crash': c, 'stuck': st, 'xt_p95': xt + 0.01 * i, 'scenario_id': f'{sid}{ep}'})
    return rows


def _write(root, name, obj):
    with open(os.path.join(root, name), 'w') as f:
        json.dump(obj, f)


OK, CR, STK = (True, False, False), (False, True, False), (False, False, True)


def test_route_episode_mcnemar_holm_and_pooled(tmp_path):
    # medium: Webots settles every episode, PyBullet fails 0 and 1 -> only_wb 2
    wb_m = _route_rows([[OK, OK]] * 4, xt=0.40)
    pb_m = _route_rows([[OK, STK], [CR, OK], [OK, OK], [OK, OK]], xt=0.30)
    # hard: PyBullet better in episode 3
    wb_h = _route_rows([[OK, OK], [OK, OK], [OK, OK], [STK, OK]], xt=0.5)
    pb_h = _route_rows([[OK, OK]] * 4, xt=0.5)
    for sim, m, rows in (('wb', 'medium', wb_m), ('pb', 'medium', pb_m), ('wb', 'hard', wb_h), ('pb', 'hard', pb_h)):
        _write(tmp_path, f'routes_{sim}_{m}.route.rows.json', {'rows': rows})
    out = cc.compare_routes(str(tmp_path))
    settled = [r for r in out['episode'] if r['metric'] == 'all_settled']
    med, hard, tot = (next(r for r in settled if r['test'] == t) for t in ('medium', 'hard', 'ALL'))
    assert (med['n'], med['wb'], med['pb'], med['only_wb'], med['only_pb']) == (4, 4, 2, 2, 0)
    assert (hard['only_wb'], hard['only_pb']) == (0, 1)
    raw = [mcnemar_exact(2, 0), mcnemar_exact(0, 1)]
    assert [med['p'], hard['p']] == pytest.approx(holm(raw))
    assert (tot['only_wb'], tot['only_pb'], tot['p_kind']) == (2, 1, 'raw')
    assert tot['p'] == pytest.approx(mcnemar_exact(2, 1))
    assert out['scenario_mismatch'] == {'medium': 0, 'hard': 0}
    xt = next(r for r in out['xt_p95'] if r['test'] == 'medium')
    assert xt['n'] == 8 and xt['median_diff'] == pytest.approx(0.10)
    xt_h = next(r for r in out['xt_p95'] if r['test'] == 'hard')
    assert xt_h['n_nonzero'] == 0 and xt_h['p'] is None
    md = cc.write_report(str(tmp_path), out, cc.compare_e1(str(tmp_path)))
    assert 'City routes' in md and os.path.exists(tmp_path / 'c5_compare.csv')


def test_route_scenario_mismatch_is_counted(tmp_path):
    _write(tmp_path, 'routes_wb_easy.route.rows.json', {'rows': _route_rows([[OK]] * 3, sid='x')})
    _write(tmp_path, 'routes_pb_easy.route.rows.json', {'rows': _route_rows([[OK]] * 3, sid='y')})
    assert cc.compare_routes(str(tmp_path))['scenario_mismatch'] == {'easy': 3}


# ======================================================== C. E1
def _flight(seed, mode, fid, arrived=True, crashed=False, zone=False, c2=False, c14=False, late=1.0):
    return {'seed': seed, 'mode': mode, 'id': fid, 'arrived': arrived, 'crashed': crashed,
            'zone_entered': zone, 'close_2.0': c2, 'close_1.4': c14, 'late_s': late}


def test_e1_pairs_by_flight_id_and_reports_both_levels(tmp_path):
    wb = [_flight(20, 'no_e1', 'f1', arrived=False, crashed=True, c2=True, late=None),
          _flight(20, 'no_e1', 'f2', late=2.0), _flight(20, 'e1', 'f1', late=0.5), _flight(20, 'e1', 'f2', late=1.5),
          _flight(21, 'e1', 'f7', late=3.0)]
    pb = [_flight(21, 'e1', 'f7', late=2.0), _flight(20, 'e1', 'f2', late=1.0), _flight(20, 'e1', 'f1', late=0.4),
          _flight(20, 'no_e1', 'f2', late=1.0), _flight(20, 'no_e1', 'f1', c2=True, late=1.0)]   # shuffled
    _write(tmp_path, 'e1_wb_medium.rows.json', {'rows': wb})
    _write(tmp_path, 'e1_pb_medium.rows.json', {'rows': pb})
    trials = {'trials': [{'seed': 20, 'approved': 2, 'requested': 3}, {'seed': 21, 'approved': 1, 'requested': 3}]}
    _write(tmp_path, 'e1_wb_medium.json', trials)
    _write(tmp_path, 'e1_pb_medium.json', trials)
    out = cc.compare_e1(str(tmp_path))
    assert out['approvals']['medium']['identical']
    arr = {r['test']: r for r in out['flight'] if r['metric'] == 'arrived'}
    assert (arr['medium/no_e1']['n'], arr['medium/no_e1']['only_pb'], arr['medium/no_e1']['only_wb']) == (2, 1, 0)
    assert arr['medium/e1']['n'] == 3 and arr['medium/e1']['p_raw'] is None
    close = {r['test']: r for r in out['flight'] if r['metric'] == 'close_2.0'}
    assert close['medium/no_e1']['only_wb'] == 0 and close['medium/no_e1']['wb'] == 1   # same flight, both
    ep = {r['test']: r for r in out['episode'] if r['metric'] == 'all_arrived'}
    assert ep['medium/no_e1']['n'] == 1 and ep['medium/no_e1']['only_pb'] == 1
    assert ep['medium/e1']['n'] == 2
    late = {r['test']: r for r in out['late_s']}
    assert late['medium/no_e1']['n'] == 1                                   # f1 did not settle in Webots
    assert late['medium/e1']['n'] == 3 and late['medium/e1']['median_diff'] == pytest.approx(0.5)


def test_e1_approval_difference_is_flagged(tmp_path):
    _write(tmp_path, 'e1_wb_crop.rows.json', {'rows': [_flight(20, 'e1', 'a')]})
    _write(tmp_path, 'e1_pb_crop.rows.json', {'rows': [_flight(20, 'e1', 'a')]})
    _write(tmp_path, 'e1_wb_crop.json', {'trials': [{'seed': 20, 'approved': 5, 'requested': 6}]})
    _write(tmp_path, 'e1_pb_crop.json', {'trials': [{'seed': 20, 'approved': 4, 'requested': 6}]})
    assert not cc.compare_e1(str(tmp_path))['approvals']['crop']['identical']


# ======================================================== D. evaluation script pieces
def test_per_flight_rows():
    import eval_e1_flown as ef
    flights = [{'id': f'f{i}', 'no_e1': (float(i), []), 'arrival': {'no_e1': 10.0 + i}} for i in range(3)]
    close = {2.0: {(0, 2)}, 1.4: set()}
    rows = ef.per_flight(flights, 'no_e1', reached=np.array([True, False, True]), crashed=np.array([False, True, False]),
                         entered={1}, close=close, settled=np.array([12.5, np.nan, 11.0]),
                         stats={'max_roll_deg': [1.0, 70.0, 2.0], 'max_pitch_deg': [3.0, 0.0, 1.0],
                                'max_speed': [1.0, 4.0, 1.2]})
    assert [r['id'] for r in rows] == ['f0', 'f1', 'f2']
    assert [r['arrived'] for r in rows] == [True, False, True]
    assert [r['crashed'] for r in rows] == [False, True, False]
    assert [r['zone_entered'] for r in rows] == [False, True, False]
    assert [r['close_2.0'] for r in rows] == [True, False, True]
    assert [r['close_1.4'] for r in rows] == [False, False, False]
    assert [r['late_s'] for r in rows] == [2.5, None, -1.0]
    assert [r['max_tilt_deg'] for r in rows] == [3.0, 70.0, 2.0]
    assert 'max_tilt_deg' not in ef.per_flight(flights, 'no_e1', [True] * 3, [False] * 3, set(), close,
                                               np.array([1.0, 1.0, 1.0]))[0]


def test_scenario_id():
    import eval_city_routes as ecr
    sp = {'drone_0': [1.0, 2.0, 2.0], 'drone_1': [3.0, 4.0, 2.5]}
    gl = {'drone_0': [5.0, 6.0, 2.0], 'drone_1': [7.0, 8.0, 2.5]}
    sid = ecr.scenario_id(sp, gl)
    assert sid == ecr.scenario_id(dict(reversed(list(sp.items()))), gl)          # key order irrelevant
    assert sid == ecr.scenario_id({k: [x + 1e-4 for x in v] for k, v in sp.items()}, gl)   # below 1 mm
    assert sid != ecr.scenario_id({**sp, 'drone_1': [3.0, 4.002, 2.5]}, gl)


def _cli(script, *extra):
    return subprocess.run([sys.executable, os.path.join(ROOT, 'scripts', script), '--config', 'x.yaml',
                           '--checkpoint', 'none.pth', *extra], capture_output=True, text=True, cwd=ROOT)


@pytest.mark.parametrize('script', ['eval_city_routes.py', 'eval_e1_flown.py'])
def test_cli_refuses_test_seeds_and_unknown_simulators(script):
    assert _cli(script, '--seeds', '20', '12').returncode == 2
    assert _cli(script, '--simulator', 'webots').returncode == 2
    if script == 'eval_e1_flown.py':
        p = _cli(script, '--seeds', '20', '--worker-seed', '15', '--worker-mode', 'e1', '--worker-out', '/dev/null')
        assert p.returncode == 2 and 'reserved' in p.stderr
