"""
c5_compare.py
-------------
C5: the city worlds flown in Webots Level 2 (envs/webots_l2_multi.py) vs PyBullet, paired.
Reads finished results only (validation seeds 20-24, config_stage2_round7_eval_goalhold.yaml,
same scenarios and per-episode sensor noise in both simulators):

  City routes (scripts/eval_city_routes.py --modes route; tests = maps easy, medium, hard, crop)
      <root>/routes_{pb,wb}_<map>.route.rows.json
    Episode level (pairs: (seed, episode); scenario_id is a cross-check, mismatches counted):
      all settled, any crash, any stuck: exact McNemar per map, Holm over the maps; pooled row
      raw. Per flight (descriptive): settled / crashed / stuck rates, lost control (Webots,
      tilt > 60 deg).
    Cross-track p95 per flight (pairs: (seed, episode, drone)): Wilcoxon signed-rank (exact),
      Webots - PyBullet, per map, Holm over the maps.

  E1-flown (scripts/eval_e1_flown.py; tests = map x condition: medium / crop x no_e1 / e1)
      <root>/e1_{pb,wb}_<map>.json and .rows.json
    E1's plan is the same in both simulators (checked: approved / requested per seed).
    Main test, per flight (pairs: (seed, condition, flight id)): arrived, crashed, entered an
      active zone, in a close pass below 2.0 m / 1.4 m: exact McNemar per test, Holm over the
      tests. Flights of one trial share the air, so they are NOT fully independent: the p-values
      are optimistic; the episode-level tests below do not have that problem but have few pairs.
    Episode level (pairs: (seed, condition)): all arrived, any crash, any close pass < 2.0 m,
      any zone entry: exact McNemar per test, Holm.
    late_s (settled minus planned arrival) per flight, both settled: Wilcoxon signed-rank, Holm.

Wilcoxon: zero differences dropped (Wilcoxon's rule), tied |differences| get average ranks, the
p-value is exact (the full null distribution of W+ given those ranks), two-sided.
Writes <root>/c5_compare.md and <root>/c5_compare.csv.
Usage (repo root): python scripts/c5_compare.py --root eval_results/c5_val
"""
import argparse
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval_stats import wilson, mcnemar_exact, holm                  # noqa: E402

ROUTE_MAPS = ['easy', 'medium', 'hard', 'crop']
E1_MAPS = ['medium', 'crop']
E1_MODES = ['no_e1', 'e1']
SIMS = {'wb': 'Webots', 'pb': 'PyBullet'}
LOST_TILT_DEG = 60.0


# ------------------------------------------------------------------ statistics
def wilcoxon_signed_rank(diffs):
    """Two-sided exact Wilcoxon signed-rank test on paired differences.
    Returns (p, n_used, w_plus)."""
    d = np.asarray([x for x in diffs if x != 0], dtype=np.float64)
    n = len(d)
    if n == 0:
        return 1.0, 0, 0.0
    a = np.abs(d)
    order = np.argsort(a, kind='mergesort')
    ranks = np.empty(n)
    k = 0
    while k < n:                                         # average ranks over ties
        j = k
        while j + 1 < n and a[order[j + 1]] == a[order[k]]:
            j += 1
        ranks[order[k:j + 1]] = (k + j) / 2.0 + 1.0
        k = j + 1
    r2 = np.rint(2 * ranks).astype(np.int64)            # doubled ranks are integers
    w2 = int(r2[d > 0].sum())
    dist = np.zeros(int(r2.sum()) + 1)
    dist[0] = 1.0
    for r in r2:                                         # each rank + or - with probability 1/2
        new = 0.5 * dist
        new[r:] += 0.5 * dist[:-r]
        dist = new
    p = 2.0 * min(dist[:w2 + 1].sum(), dist[w2:].sum())
    return float(min(1.0, p)), n, w2 / 2.0


def paired_binary(a, b, keys, metric):
    """a, b: {key: row}; McNemar counts on metric over keys. b10 = only a true, b01 = only b."""
    b10 = sum(1 for k in keys if a[k][metric] and not b[k][metric])
    b01 = sum(1 for k in keys if b[k][metric] and not a[k][metric])
    na = sum(1 for k in keys if a[k][metric])
    nb = sum(1 for k in keys if b[k][metric])
    return na, nb, b10, b01


# ------------------------------------------------------------------ loading
def load_json(path):
    with open(path) as f:
        return json.load(f)


def route_rows(root, sim, m):
    path = os.path.join(root, f'routes_{sim}_{m}.route.rows.json')
    return load_json(path)['rows'] if os.path.exists(path) else None


def route_episodes(rows):
    """(seed, episode) -> episode outcome."""
    eps = {}
    for r in rows:
        eps.setdefault((r['seed'], r['episode']), []).append(r)
    out = {}
    for k, rs in eps.items():
        out[k] = {'all_settled': all(r['settled'] for r in rs),
                  'any_crash': any(r['episode_crash'] for r in rs),
                  'any_stuck': any(r['stuck'] for r in rs),
                  'scenario_id': rs[0].get('scenario_id')}
    return out


def e1_rows(root, sim, m):
    path = os.path.join(root, f'e1_{sim}_{m}.rows.json')
    return load_json(path)['rows'] if os.path.exists(path) else None


def e1_approvals(root, sim, m):
    path = os.path.join(root, f'e1_{sim}_{m}.json')
    if not os.path.exists(path):
        return None
    return {t['seed']: (t['approved'], t['requested']) for t in load_json(path)['trials']}


def e1_episodes(rows):
    """(seed, mode) -> episode outcome."""
    eps = {}
    for r in rows:
        eps.setdefault((r['seed'], r['mode']), []).append(r)
    return {k: {'all_arrived': all(r['arrived'] for r in rs), 'any_crash': any(r['crashed'] for r in rs),
                'any_close_2.0': any(r['close_2.0'] for r in rs),
                'any_zone_entered': any(r['zone_entered'] for r in rs)} for k, rs in eps.items()}


# ------------------------------------------------------------------ comparisons
def mcnemar_family(tests, metric, lines_key):
    """tests: [(name, a_rows_by_key, b_rows_by_key)] -> result rows with Holm over the tests,
    plus a pooled row (raw p)."""
    res = []
    for name, a, b in tests:
        keys = sorted(set(a) & set(b))
        na, nb, b10, b01 = paired_binary(a, b, keys, metric)
        res.append({'family': lines_key, 'metric': metric, 'test': name, 'n': len(keys),
                    'wb': na, 'pb': nb, 'only_wb': b10, 'only_pb': b01,
                    'p_raw': mcnemar_exact(b10, b01) if b10 + b01 else None,
                    'unpaired': len(set(a) ^ set(b))})
    ps = [r['p_raw'] for r in res if r['p_raw'] is not None]
    adj = iter(holm(ps)) if ps else iter(())
    for r in res:
        r['p'], r['p_kind'] = (next(adj), 'holm') if r['p_raw'] is not None else (None, 'holm')
    if len(res) > 1:
        tot = {k: sum(r[k] for r in res) for k in ('n', 'wb', 'pb', 'only_wb', 'only_pb', 'unpaired')}
        p = mcnemar_exact(tot['only_wb'], tot['only_pb']) if tot['only_wb'] + tot['only_pb'] else None
        res.append({'family': lines_key, 'metric': metric, 'test': 'ALL', **tot, 'p_raw': p, 'p': p,
                    'p_kind': 'raw'})
    return res


def wilcoxon_family(tests, field, lines_key):
    """tests: [(name, a_rows_by_key, b_rows_by_key)]; pairs where both have field not None.
    Difference = a - b (Webots - PyBullet)."""
    res = []
    for name, a, b in tests:
        keys = sorted(k for k in set(a) & set(b) if a[k].get(field) is not None and b[k].get(field) is not None)
        d = [float(a[k][field]) - float(b[k][field]) for k in keys]
        p, n_used, w = wilcoxon_signed_rank(d)
        res.append({'family': lines_key, 'metric': field, 'test': name, 'n': len(keys), 'n_nonzero': n_used,
                    'median_wb': float(np.median([a[k][field] for k in keys])) if keys else None,
                    'median_pb': float(np.median([b[k][field] for k in keys])) if keys else None,
                    'median_diff': float(np.median(d)) if d else None, 'w_plus': w,
                    'p_raw': p if n_used else None})
    ps = [r['p_raw'] for r in res if r['p_raw'] is not None]
    adj = iter(holm(ps)) if ps else iter(())
    for r in res:
        r['p'] = next(adj) if r['p_raw'] is not None else None
        r['p_kind'] = 'holm'
    return res


def compare_routes(root):
    out, scen = {'episode': [], 'xt_p95': [], 'flight_rates': [], 'scenario_mismatch': {}}, []
    tests_ep, tests_fl = [], []
    for m in ROUTE_MAPS:
        a, b = route_rows(root, 'wb', m), route_rows(root, 'pb', m)
        if a is None or b is None:
            continue
        ea, eb = route_episodes(a), route_episodes(b)
        out['scenario_mismatch'][m] = sum(1 for k in set(ea) & set(eb)
                                          if ea[k]['scenario_id'] != eb[k]['scenario_id'])
        tests_ep.append((m, ea, eb))
        fa = {(r['seed'], r['episode'], r['drone']): r for r in a}
        fb = {(r['seed'], r['episode'], r['drone']): r for r in b}
        tests_fl.append((m, fa, fb))
        for sim, rows in (('wb', a), ('pb', b)):
            n = len(rows)
            rate = {k: sum(1 for r in rows if r[k]) for k in ('settled', 'crashed', 'stuck')}
            lost = (sum(1 for r in rows if r.get('max_tilt_deg', 0.0) > LOST_TILT_DEG)
                    if any('max_tilt_deg' in r for r in rows) else None)
            out['flight_rates'].append({'test': m, 'sim': sim, 'flights': n, **rate, 'lost_control': lost})
    for metric in ('all_settled', 'any_crash', 'any_stuck'):
        out['episode'] += mcnemar_family(tests_ep, metric, 'routes/episode')
    out['xt_p95'] = wilcoxon_family(tests_fl, 'xt_p95', 'routes/flight')
    return out


def compare_e1(root):
    out = {'flight': [], 'episode': [], 'late_s': [], 'approvals': {}, 'flight_rates': []}
    tests_fl, tests_ep = [], []
    for m in E1_MAPS:
        a, b = e1_rows(root, 'wb', m), e1_rows(root, 'pb', m)
        if a is None or b is None:
            continue
        apa, apb = e1_approvals(root, 'wb', m), e1_approvals(root, 'pb', m)
        out['approvals'][m] = {'wb': apa, 'pb': apb, 'identical': apa == apb}
        ea, eb = e1_episodes(a), e1_episodes(b)
        for mode in E1_MODES:
            name = f'{m}/{mode}'
            fa = {(r['seed'], r['mode'], r['id']): r for r in a if r['mode'] == mode}
            fb = {(r['seed'], r['mode'], r['id']): r for r in b if r['mode'] == mode}
            tests_fl.append((name, fa, fb))
            tests_ep.append((name, {k: v for k, v in ea.items() if k[1] == mode},
                             {k: v for k, v in eb.items() if k[1] == mode}))
            for sim, rows in (('wb', fa), ('pb', fb)):
                rs = list(rows.values())
                lost = (sum(1 for r in rs if r.get('max_tilt_deg', 0.0) > LOST_TILT_DEG)
                        if any('max_tilt_deg' in r for r in rs) else None)
                out['flight_rates'].append({'test': name, 'sim': sim, 'flights': len(rs),
                                            **{k: sum(1 for r in rs if r[k])
                                               for k in ('arrived', 'crashed', 'zone_entered', 'close_2.0', 'close_1.4')},
                                            'lost_control': lost})
    for metric in ('arrived', 'crashed', 'zone_entered', 'close_2.0', 'close_1.4'):
        out['flight'] += mcnemar_family(tests_fl, metric, 'e1/flight')
    for metric in ('all_arrived', 'any_crash', 'any_close_2.0', 'any_zone_entered'):
        out['episode'] += mcnemar_family(tests_ep, metric, 'e1/episode')
    out['late_s'] = wilcoxon_family(tests_fl, 'late_s', 'e1/flight')
    return out


# ------------------------------------------------------------------ report
def fmt_p(p):
    return '-' if p is None else (f'{p:.4f}' if p >= 1e-4 else f'{p:.1e}')


def pct(k, n):
    if not n:
        return '-'
    lo, hi = wilson(k, n)
    return f'{100 * k / n:.1f}% [{100 * lo:.0f}-{100 * hi:.0f}]'


def md_mcnemar(rows, title):
    out = [f'### {title}', '', '| metric | test | pairs | Webots | PyBullet | only W | only P | p | |',
           '|---|---|---|---|---|---|---|---|---|']
    for r in rows:
        out.append(f"| {r['metric']} | {r['test']} | {r['n']} | {pct(r['wb'], r['n'])} | {pct(r['pb'], r['n'])} | "
                   f"{r['only_wb']} | {r['only_pb']} | {fmt_p(r['p'])} | {r['p_kind']} |")
    return out + ['']


def md_wilcoxon(rows, title, unit):
    out = [f'### {title}', '', f'| test | pairs | non-zero | median W ({unit}) | median P ({unit}) | '
           f'median W-P ({unit}) | p (Holm) |', '|---|---|---|---|---|---|---|']
    f = lambda x: '-' if x is None else f'{x:.3f}'                          # noqa: E731
    for r in rows:
        out.append(f"| {r['test']} | {r['n']} | {r['n_nonzero']} | {f(r['median_wb'])} | {f(r['median_pb'])} | "
                   f"{f(r['median_diff'])} | {fmt_p(r['p'])} |")
    return out + ['']


def write_report(root, routes, e1):
    md = ['# C5: Webots Level 2 vs PyBullet in the city worlds', '',
          'W = Webots (webots_l2_multi), P = PyBullet. Rates with 95% Wilson intervals. McNemar exact, '
          'Holm over the tests of each metric; ALL rows raw.', '']
    if routes['episode']:
        md += ['## City routes (route mode, 4 drones)', '',
               f"Scenario-id mismatches per map: {routes['scenario_mismatch']}", '']
        md += md_mcnemar(routes['episode'], 'Episode level')
        md += ['### Per flight (descriptive)', '', '| test | sim | flights | settled | crashed | stuck | lost control |',
               '|---|---|---|---|---|---|---|']
        for r in routes['flight_rates']:
            md.append(f"| {r['test']} | {SIMS[r['sim']]} | {r['flights']} | {pct(r['settled'], r['flights'])} | "
                      f"{pct(r['crashed'], r['flights'])} | {pct(r['stuck'], r['flights'])} | "
                      f"{'-' if r['lost_control'] is None else r['lost_control']} |")
        md += ['']
        md += md_wilcoxon(routes['xt_p95'], 'Cross-track error, 95th percentile per flight (Wilcoxon)', 'm')
    if e1['flight']:
        md += ['## E1-flown traffic', '', f"E1 approvals identical in both: "
               f"{ {m: v['identical'] for m, v in e1['approvals'].items()} }", '',
               'Main test: per flight. Flights of one trial share the air and are not fully independent, '
               'so these p-values are optimistic; the episode-level tests have no such dependence but few pairs.', '']
        md += md_mcnemar(e1['flight'], 'Per flight (main)')
        md += md_mcnemar(e1['episode'], 'Episode level (trial x condition)')
        md += md_wilcoxon(e1['late_s'], 'Lateness late_s, both settled (Wilcoxon)', 's')
        md += ['### Per flight rates', '', '| test | sim | flights | arrived | crashed | zone entered | '
               'close < 2.0 m | close < 1.4 m | lost control |', '|---|---|---|---|---|---|---|---|---|']
        for r in e1['flight_rates']:
            n = r['flights']
            md.append(f"| {r['test']} | {SIMS[r['sim']]} | {n} | {pct(r['arrived'], n)} | {pct(r['crashed'], n)} | "
                      f"{pct(r['zone_entered'], n)} | {pct(r['close_2.0'], n)} | {pct(r['close_1.4'], n)} | "
                      f"{'-' if r['lost_control'] is None else r['lost_control']} |")
        md += ['']
    with open(os.path.join(root, 'c5_compare.md'), 'w') as f:
        f.write('\n'.join(md))
    rows = routes['episode'] + routes['xt_p95'] + e1['flight'] + e1['episode'] + e1['late_s']
    cols = ['family', 'metric', 'test', 'n', 'wb', 'pb', 'only_wb', 'only_pb', 'n_nonzero', 'median_wb',
            'median_pb', 'median_diff', 'p_raw', 'p', 'p_kind', 'unpaired']
    with open(os.path.join(root, 'c5_compare.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)
    return '\n'.join(md)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', default='eval_results/c5_val')
    a = ap.parse_args()
    routes, e1 = compare_routes(a.root), compare_e1(a.root)
    if not routes['episode'] and not e1['flight']:
        sys.exit(f'no paired results under {a.root}')
    print(write_report(a.root, routes, e1))


if __name__ == '__main__':
    main()
