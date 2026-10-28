"""
c5_compare.py
-------------
C5: the city worlds flown in Webots Level 2 (envs/webots_l2_multi.py) vs PyBullet, paired.
Reads finished results only (validation seeds 20-24, config_stage2_round7_eval_goalhold.yaml,
same scenarios and per-episode sensor noise in both simulators):

  City routes (scripts/eval_city_routes.py --modes route; tests = maps easy, medium, hard, crop)
      <root>/routes_{pb,wb}_<map>.route.rows.json
    Episode level (pairs: (seed, episode); scenario_id is a cross-check, mismatches counted):
      all settled, any crash or lost control, any stuck, any lost control: exact McNemar per
      map, Holm over the maps; pooled row raw. Per flight (descriptive): settled / crashed /
      stuck rates, lost control.
    Cross-track p95 per flight (pairs: (seed, episode, drone)): Wilcoxon signed-rank (exact),
      Webots - PyBullet, per map, Holm over the maps.

  E1-flown (scripts/eval_e1_flown.py; tests = map x condition: medium / crop x no_e1 / e1)
      <root>/e1_{pb,wb}_<map>.json and .rows.json
    E1's plan is the same in both simulators (checked: approved / requested per seed).
    Main test, per flight (pairs: (seed, condition, flight id)): arrived, crashed or lost
      control, lost control, entered an active zone, in a close pass below 2.0 m / 1.4 m:
      exact McNemar per test, Holm over the
      tests. Flights of one trial share the air, so they are NOT fully independent: the p-values
      are optimistic; the episode-level tests below do not have that problem but have few pairs.
    Episode level (pairs: (seed, condition)): all arrived, any crash or lost control, any close
      pass < 2.0 m, any zone entry: exact McNemar per test, Holm.
    late_s (settled minus planned arrival) per flight, both settled: Wilcoxon signed-rank, Holm.

Lost control (as C1, eval_suite.lost_control): max tilt > 60 deg. Webots rows only (PyBullet's
rows carry no tilt: never lost). A lost-control flight is a FAILURE: it does not count as
settled / arrived (and has no late_s), and it counts in "crash or lost control"; the raw crash
rate stays in the descriptive tables.

Wilcoxon: zero differences dropped (Wilcoxon's rule), tied |differences| get average ranks, the
p-value is exact (the full null distribution of W+ given those ranks), two-sided.
Writes <root>/c5_compare.md and <root>/c5_compare.csv.
Usage (repo root): python scripts/c5_compare.py --root eval_results/c5_val

Other pairings (e.g. a sensitivity variant against the original Webots run, or against PyBullet):
  --a ROOT:PREFIX --b ROOT:PREFIX [--label-a NAME --label-b NAME] --out DIR
reads <ROOT>/routes_<PREFIX>_<map>... and <ROOT>/e1_<PREFIX>_<map>... for each side (default
<root>:wb vs <root>:pb, Webots vs PyBullet); differences are A - B; writes DIR/c5_compare.md/.csv.
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


def mark_lost(rows):
    """Per flight: lost = tilt > LOST_TILT_DEG (C1's rule); a lost flight is a failure: it is not
    settled / arrived and has no late_s; crash_or_lost = crashed or lost. Returns new dicts."""
    out = []
    for r in rows:
        r = dict(r)
        r['lost'] = r.get('max_tilt_deg', 0.0) > LOST_TILT_DEG
        r['crash_or_lost'] = bool(r['crashed'] or r['lost'])
        if r['lost']:
            for k in ('settled', 'arrived'):
                if k in r:
                    r[k] = False
            if 'late_s' in r:
                r['late_s'] = None
        out.append(r)
    return out


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
                  'any_crash_or_lost': any(r['episode_crash'] or r['lost'] for r in rs),
                  'any_stuck': any(r['stuck'] for r in rs),
                  'any_lost': any(r['lost'] for r in rs),
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
    return {k: {'all_arrived': all(r['arrived'] for r in rs), 'any_crash_or_lost': any(r['crash_or_lost'] for r in rs),
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


def compare_routes(root, a_src=None, b_src=None):
    """a_src, b_src: (root, prefix) of the two sides (default (root, 'wb') vs (root, 'pb'))."""
    a_src, b_src = a_src or (root, 'wb'), b_src or (root, 'pb')
    out, scen = {'episode': [], 'xt_p95': [], 'flight_rates': [], 'scenario_mismatch': {}}, []
    tests_ep, tests_fl = [], []
    for m in ROUTE_MAPS:
        a, b = route_rows(*a_src, m), route_rows(*b_src, m)
        if a is None or b is None:
            continue
        a, b = mark_lost(a), mark_lost(b)
        ea, eb = route_episodes(a), route_episodes(b)
        out['scenario_mismatch'][m] = sum(1 for k in set(ea) & set(eb)
                                          if ea[k]['scenario_id'] != eb[k]['scenario_id'])
        tests_ep.append((m, ea, eb))
        fa = {(r['seed'], r['episode'], r['drone']): r for r in a}
        fb = {(r['seed'], r['episode'], r['drone']): r for r in b}
        tests_fl.append((m, fa, fb))
        for sim, rows in (('wb', a), ('pb', b)):
            n = len(rows)
            rate = {k: sum(1 for r in rows if r[k]) for k in ('settled', 'crashed', 'stuck', 'crash_or_lost')}
            lost = (sum(1 for r in rows if r.get('max_tilt_deg', 0.0) > LOST_TILT_DEG)
                    if any('max_tilt_deg' in r for r in rows) else None)
            out['flight_rates'].append({'test': m, 'sim': sim, 'flights': n, **rate, 'lost_control': lost})
    for metric in ('all_settled', 'any_crash_or_lost', 'any_stuck', 'any_lost'):
        out['episode'] += mcnemar_family(tests_ep, metric, 'routes/episode')
    out['xt_p95'] = wilcoxon_family(tests_fl, 'xt_p95', 'routes/flight')
    return out


def compare_e1(root, a_src=None, b_src=None):
    """a_src, b_src: (root, prefix) of the two sides (default (root, 'wb') vs (root, 'pb'))."""
    a_src, b_src = a_src or (root, 'wb'), b_src or (root, 'pb')
    out = {'flight': [], 'episode': [], 'late_s': [], 'approvals': {}, 'flight_rates': []}
    tests_fl, tests_ep = [], []
    for m in E1_MAPS:
        a, b = e1_rows(*a_src, m), e1_rows(*b_src, m)
        if a is None or b is None:
            continue
        a, b = mark_lost(a), mark_lost(b)
        apa, apb = e1_approvals(*a_src, m), e1_approvals(*b_src, m)
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
                                               for k in ('arrived', 'crashed', 'crash_or_lost', 'zone_entered',
                                                         'close_2.0', 'close_1.4')},
                                            'lost_control': lost})
    for metric in ('arrived', 'crash_or_lost', 'lost', 'zone_entered', 'close_2.0', 'close_1.4'):
        out['flight'] += mcnemar_family(tests_fl, metric, 'e1/flight')
    for metric in ('all_arrived', 'any_crash_or_lost', 'any_close_2.0', 'any_zone_entered'):
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


def md_mcnemar(rows, title, labels=None):
    la, lb = (labels or SIMS)['wb'], (labels or SIMS)['pb']
    sa, sb = ('W', 'P') if labels is None else ('A', 'B')
    out = [f'### {title}', '', f'| metric | test | pairs | {la} | {lb} | only {sa} | only {sb} | p | |',
           '|---|---|---|---|---|---|---|---|---|']
    for r in rows:
        out.append(f"| {r['metric']} | {r['test']} | {r['n']} | {pct(r['wb'], r['n'])} | {pct(r['pb'], r['n'])} | "
                   f"{r['only_wb']} | {r['only_pb']} | {fmt_p(r['p'])} | {r['p_kind']} |")
    return out + ['']


def md_wilcoxon(rows, title, unit, labels=None):
    sa, sb = ('W', 'P') if labels is None else ('A', 'B')
    out = [f'### {title}', '', f'| test | pairs | non-zero | median {sa} ({unit}) | median {sb} ({unit}) | '
           f'median {sa}-{sb} ({unit}) | p (Holm) |', '|---|---|---|---|---|---|---|']
    f = lambda x: '-' if x is None else f'{x:.3f}'                          # noqa: E731
    for r in rows:
        out.append(f"| {r['test']} | {r['n']} | {r['n_nonzero']} | {f(r['median_wb'])} | {f(r['median_pb'])} | "
                   f"{f(r['median_diff'])} | {fmt_p(r['p'])} |")
    return out + ['']


def write_report(root, routes, e1, labels=None):
    """labels: {'wb': name of side A, 'pb': name of side B} (default Webots vs PyBullet)."""
    names = labels or SIMS
    head = ('# C5: Webots Level 2 vs PyBullet in the city worlds' if labels is None
            else f"# C5: {names['wb']} vs {names['pb']}")
    who = ('W = Webots (webots_l2_multi), P = PyBullet.' if labels is None
           else f"A = {names['wb']}, B = {names['pb']}; differences A - B.")
    md = [head, '',
          who + ' Rates with 95% Wilson intervals. McNemar exact, '
          'Holm over the tests of each metric; ALL rows raw. Lost control (tilt > 60 deg, Webots only) is a '
          'failure: not settled / arrived, and counted in crash_or_lost.', '']
    if routes['episode']:
        md += ['## City routes (route mode, 4 drones)', '',
               f"Scenario-id mismatches per map: {routes['scenario_mismatch']}", '']
        md += md_mcnemar(routes['episode'], 'Episode level', labels)
        md += ['### Per flight (descriptive)', '', '| test | sim | flights | settled | crashed | stuck | lost control '
               '| crash or lost |', '|---|---|---|---|---|---|---|---|']
        for r in routes['flight_rates']:
            md.append(f"| {r['test']} | {names[r['sim']]} | {r['flights']} | {pct(r['settled'], r['flights'])} | "
                      f"{pct(r['crashed'], r['flights'])} | {pct(r['stuck'], r['flights'])} | "
                      f"{'-' if r['lost_control'] is None else r['lost_control']} | "
                      f"{pct(r['crash_or_lost'], r['flights'])} |")
        md += ['']
        md += md_wilcoxon(routes['xt_p95'], 'Cross-track error, 95th percentile per flight (Wilcoxon)', 'm', labels)
    if e1['flight']:
        md += ['## E1-flown traffic', '', f"E1 approvals identical in both: "
               f"{ {m: v['identical'] for m, v in e1['approvals'].items()} }", '',
               'Main test: per flight. Flights of one trial share the air and are not fully independent, '
               'so these p-values are optimistic; the episode-level tests have no such dependence but few pairs.', '']
        md += md_mcnemar(e1['flight'], 'Per flight (main)', labels)
        md += md_mcnemar(e1['episode'], 'Episode level (trial x condition)', labels)
        md += md_wilcoxon(e1['late_s'], 'Lateness late_s, both settled (Wilcoxon)', 's', labels)
        md += ['### Per flight rates', '', '| test | sim | flights | arrived | crashed | zone entered | '
               'close < 2.0 m | close < 1.4 m | lost control | crash or lost |', '|---|---|---|---|---|---|---|---|---|---|']
        for r in e1['flight_rates']:
            n = r['flights']
            md.append(f"| {r['test']} | {names[r['sim']]} | {n} | {pct(r['arrived'], n)} | {pct(r['crashed'], n)} | "
                      f"{pct(r['zone_entered'], n)} | {pct(r['close_2.0'], n)} | {pct(r['close_1.4'], n)} | "
                      f"{'-' if r['lost_control'] is None else r['lost_control']} | {pct(r['crash_or_lost'], n)} |")
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
    ap.add_argument('--a', default=None, help='side A as ROOT:PREFIX (default <root>:wb)')
    ap.add_argument('--b', default=None, help='side B as ROOT:PREFIX (default <root>:pb)')
    ap.add_argument('--label-a', default=None)
    ap.add_argument('--label-b', default=None)
    ap.add_argument('--out', default=None, help='report folder (default --root)')
    a = ap.parse_args()

    def side(spec, default):
        if spec is None:
            return default
        root, sep, prefix = spec.rpartition(':')
        if not sep or not root or not prefix:
            ap.error(f'expected ROOT:PREFIX, got {spec!r}')
        return root, prefix
    sa, sb = side(a.a, (a.root, 'wb')), side(a.b, (a.root, 'pb'))
    custom = a.a is not None or a.b is not None
    labels = ({'wb': a.label_a or f'{sa[0]}:{sa[1]}', 'pb': a.label_b or f'{sb[0]}:{sb[1]}'}
              if custom or a.label_a or a.label_b else None)
    out = a.out or a.root
    routes, e1 = compare_routes(a.root, sa, sb), compare_e1(a.root, sa, sb)
    if not routes['episode'] and not e1['flight']:
        sys.exit(f'no paired results for {sa} vs {sb}')
    os.makedirs(out, exist_ok=True)
    print(write_report(out, routes, e1, labels))


if __name__ == '__main__':
    main()
