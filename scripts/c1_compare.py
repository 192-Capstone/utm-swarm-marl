"""
c1_compare.py
-------------
C1: Webots Level 2 (8 drones, one Crazyflie controller each) vs PyBullet, flight by flight, on the
9 multi-drone suite tests (standard: random headon crossing swap flythrough, 3 drones; crowd: the
same four without random, 8 drones, no obstacles), seeds 20-24, 100 flights per test. Reads
finished results only:
    <root>/c1_pb_paired, c1_pb_goalhold      PyBullet (scripts/run_suite.py)
    <root>/c1_wb_paired, c1_wb_goalhold      Webots   (--simulator webots_l2_multi)

Per test and per group (pooled):
    strict success      SwarmEnv's rule (inside 0.5 m and under 0.15 m/s for 31 steps, every drone)
    ended within 0.5 m  every drone's final 3D goal distance < 0.5 m (agents.csv)
    crash               any drone-drone or drone-obstacle collision (ends the episode)
    timeout             share of episodes that timed out, and, as in A1, every drone that did not
                        settle in a timed-out episode, by its final distance:
                        wobble (< 0.5 m), stop-short (0.5-1 m), stuck (>= 1 m)
    lost control        Webots only (eval_suite lost_control: true speed > 3 m/s or tilt > 60 deg)
Not reported: success at 0.20 m/s. It needs the per-step speed, which eval_suite does not save.

Paired comparisons on strict success (episodes matched on (test, seed, episode); scenario_id is a
cross-check only, mismatches are counted): exact McNemar per test, Holm-corrected over the 9 tests
of each comparison; the pooled group rows are exact McNemar without correction (as
scripts/results_report.py). Comparisons:
    W vs P        Webots vs PyBullet, goal-hold off   (c1_wb_paired   vs c1_pb_paired)
    W vs P (GH)   Webots vs PyBullet, goal-hold on    (c1_wb_goalhold vs c1_pb_goalhold)
    GH on vs off  goal-hold on vs off within Webots   (c1_wb_goalhold vs c1_wb_paired)

Writes <root>/c1_compare.md and <root>/c1_compare.csv.
Usage (repo root): python scripts/c1_compare.py [--root eval_results/c1_val]
"""
import argparse
import csv
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval_stats import wilson, mcnemar_exact, holm               # noqa: E402  (PR #9)
from results_report import test_name, paired, NICE                # noqa: E402  (PR #9)

RUNS = {'P': 'c1_pb_paired', 'P_GH': 'c1_pb_goalhold', 'W': 'c1_wb_paired', 'W_GH': 'c1_wb_goalhold'}
LABEL = {'P': 'PyBullet', 'P_GH': 'PyBullet + GH', 'W': 'Webots', 'W_GH': 'Webots + GH'}
TESTS = {'standard': ['random', 'headon', 'crossing', 'swap', 'flythrough'],
         'crowd': ['headon', 'crossing', 'swap', 'flythrough']}
COMPARE = [('W vs P', 'W', 'P'), ('W vs P (GH)', 'W_GH', 'P_GH'), ('GH on vs off (Webots)', 'W_GH', 'W')]
THR = 0.5
# standard random draws its scenarios from the global numpy generator, which PyBullet's lag model
# (envs/pybullet_adapter.py reset_world) also draws from and Webots does not: after episode 0 of
# each seed the scenarios differ, so this test is compared unpaired (rates only, no McNemar).
UNPAIRED = {('standard', 'random')}


def episode_lost(agents_rows):
    """Lost control in an episode (eval_suite.lost_control on each drone's saved stats), or None
    when the run has no per-drone stats (PyBullet)."""
    from eval_suite import lost_control
    if not agents_rows or 'max_roll_deg' not in agents_rows[0]:
        return None
    return any(lost_control({'max_speed': [float(a['max_true_speed'])],
                             'max_roll_deg': [float(a['max_roll_deg'])],
                             'max_pitch_deg': [float(a['max_pitch_deg'])]}, 0) for a in agents_rows)


def load(root, run):
    """{(group, test): [episode rows]}; each row gets 'within' (every drone ended < 0.5 m) and
    'agents' ([(settled, final goal distance)] per drone) from agents.csv."""
    out = {}
    for group in TESTS:
        for f in glob.glob(os.path.join(root, run, group, '*', 'episodes.csv')):
            folder = os.path.dirname(f)
            test = test_name(folder)
            if test not in TESTS[group]:
                continue
            eps = list(csv.DictReader(open(f, encoding='utf-8')))
            ags = {}
            for a in csv.DictReader(open(os.path.join(folder, 'agents.csv'), encoding='utf-8')):
                ags.setdefault((a['seed'], a['episode']), []).append(a)
            for r in eps:
                rows = ags[(r['seed'], r['episode'])]
                r['agents'] = [(a['settled'] == '1', float(a['final_goal_dist'])) for a in rows]
                r['within'] = all(d < THR for _, d in r['agents'])
                r['lost'] = episode_lost(rows)
            out[(group, test)] = eps
    return out


def metrics(eps):
    n = len(eps)
    to = [r for r in eps if r['outcome'] == 'timeout']
    unsettled = [d for r in to for s, d in r['agents'] if not s]       # drones, A1 style
    m = {'n': n,
         'success': sum(r['success'] == '1' for r in eps),
         'within': sum(r['within'] for r in eps),
         'crash': sum(r['any_drone_collision'] == '1' or r['any_obstacle_collision'] == '1' for r in eps),
         'timeout': len(to),
         'to_wobble': sum(d < THR for d in unsettled),
         'to_stop_short': sum(THR <= d < 1.0 for d in unsettled),
         'to_stuck': sum(d >= 1.0 for d in unsettled)}
    if eps and eps[0]['lost'] is not None:          # recomputed with the current rule (tilt only)
        m['lost'] = sum(r['lost'] for r in eps)
    if eps and 'reimported_drones' in eps[0]:       # Webots runs with the reset fix
        m['reimported'] = sum(int(r['reimported_drones']) for r in eps)
    return m


def pct(k, n):
    if not n:
        return '-'
    lo, hi = wilson(k, n)
    return f"{100 * k / n:.1f}% [{100 * lo:.0f}-{100 * hi:.0f}]"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', default='eval_results/c1_val')
    ap.add_argument('--wb-paired', default=RUNS['W'], help='Webots run without goal-hold')
    ap.add_argument('--wb-goalhold', default=RUNS['W_GH'], help='Webots run with goal-hold')
    args = ap.parse_args()
    root = args.root
    RUNS['W'], RUNS['W_GH'] = args.wb_paired, args.wb_goalhold

    data = {}
    for key, run in RUNS.items():
        if not os.path.isdir(os.path.join(root, run)):
            sys.exit(f"missing {os.path.join(root, run)} (run not finished?)")
        data[key] = load(root, run)
    cells = [(g, t) for g in TESTS for t in TESTS[g]]
    for key in RUNS:
        missing = [c for c in cells if c not in data[key]]
        if missing:
            sys.exit(f"{RUNS[key]} is missing tests: {missing}")

    rows, L = [], ['# C1: Webots Level 2 vs PyBullet (9 multi-drone tests, seeds 20-24)', '']
    # ---- rates per run, per test and pooled per group
    L += ['## Rates (95% Wilson CIs)', '',
          '| run | test | n | strict success | ended within 0.5 m | crash | timeout | '
          'unsettled drones in timeouts (wobble / stop-short / stuck) | lost control | re-imported drones |',
          '|---|---|---|---|---|---|---|---|---|---|']
    for key in RUNS:
        for g in TESTS:
            pool = None
            for t in TESTS[g] + ['ALL']:
                if t == 'ALL':
                    m, name = pool, f"**{g}, all**"
                else:
                    m, name = metrics(data[key][(g, t)]), f"{g}: {NICE.get(t, t)}"
                    pool = m if pool is None else {k: pool[k] + m[k] for k in m}
                lost = pct(m['lost'], m['n']) if 'lost' in m else '-'
                reimp = str(m['reimported']) if 'reimported' in m else '-'
                L.append(f"| {LABEL[key]} | {name} | {m['n']} | {pct(m['success'], m['n'])} | {pct(m['within'], m['n'])} | "
                         f"{pct(m['crash'], m['n'])} | {pct(m['timeout'], m['n'])} | "
                         f"{m['to_wobble']} / {m['to_stop_short']} / {m['to_stuck']} | {lost} | {reimp} |")
                rows.append({'kind': 'rate', 'run': RUNS[key], 'group': g, 'test': t, **m})
    L.append('')

    # ---- paired comparisons on strict success
    for title, a, b in COMPARE:
        L += [f"## {title}: {LABEL[a]} (A) vs {LABEL[b]} (B), strict success", '',
              '| test | n | A | B | A - B | only A | only B | p (Holm; pooled: raw) | scene mismatches |',
              '|---|---|---|---|---|---|---|---|---|']
        res = []
        for g in TESTS:
            pool = [0, 0, 0, 0, 0, 0]
            for t in TESTS[g]:
                ea, eb = data[a][(g, t)], data[b][(g, t)]
                if (g, t) in UNPAIRED:
                    res.append([g, t, (len(ea), len(eb)), sum(r['success'] == '1' for r in ea),
                                sum(r['success'] == '1' for r in eb), None, None, None, 'unpaired'])
                    continue                                     # not in the pooled row either
                bb, cc, n, mm = paired(ea, eb)
                keys_b = {(o['seed'], o['episode']) for o in eb}
                sa = sum(r['success'] == '1' for r in ea if (r['seed'], r['episode']) in keys_b)
                sb = sa - bb + cc
                res.append([g, t, n, sa, sb, bb, cc, mm, mcnemar_exact(bb, cc) if bb + cc else None])
                pool = [pool[0] + n, pool[1] + sa, pool[2] + sb, pool[3] + bb, pool[4] + cc, pool[5] + mm]
            res.append([g, 'ALL', *pool, 'pooled'])
        test_p = [r[8] for r in res if r[1] != 'ALL' and r[8] not in (None, 'unpaired')]
        adj = iter(holm(test_p))
        for r in res:
            g, t, n, sa, sb, bb, cc, mm, p = r
            if p == 'unpaired':
                na, nb = n
                L.append(f"| {g}: {NICE.get(t, t)} (unpaired) | {na} / {nb} | {100 * sa / na:.1f}% | "
                         f"{100 * sb / nb:.1f}% | {100 * (sa / na - sb / nb):+.1f} | – | – | unpaired | – |")
                rows.append({'kind': 'paired', 'comparison': title, 'run': f"{RUNS[a]} vs {RUNS[b]}", 'group': g,
                             'test': t, 'n': f"{na}/{nb}", 'success_a': sa, 'success_b': sb,
                             'p_kind': 'unpaired'})
                continue
            if t == 'ALL':
                pv = mcnemar_exact(bb, cc)
                name = (f"**{g}, all paired ({len(TESTS[g]) - sum((g, x) in UNPAIRED for x in TESTS[g])} tests, "
                        f"without {', '.join(x for x in TESTS[g] if (g, x) in UNPAIRED)})**"
                        if any((g, x) in UNPAIRED for x in TESTS[g]) else f"**{g}, all**")
            else:
                pv = next(adj) if p is not None else None
                name = f"{g}: {NICE.get(t, t)}"
            ptxt = '–' if pv is None or bb + cc == 0 else f"{pv:.3g}{' *' if pv < 0.05 else ''}"
            L.append(f"| {name} | {n} | {100 * sa / n:.1f}% | {100 * sb / n:.1f}% | {100 * (sa - sb) / n:+.1f} | "
                     f"{bb} | {cc} | {ptxt} | {mm} |")
            rows.append({'kind': 'paired', 'comparison': title, 'run': f"{RUNS[a]} vs {RUNS[b]}", 'group': g,
                         'test': t, 'n': n, 'success_a': sa, 'success_b': sb, 'only_a': bb, 'only_b': cc,
                         'p': pv, 'p_kind': 'raw' if t == 'ALL' else 'holm', 'scene_mismatch': mm})
        L.append('')

    L += ['Notes: strict success = SwarmEnv rule. Timeout drone counts: drones that did not settle in '
          'timed-out episodes, by final distance (A1 categories). '
          'Success at 0.20 m/s is not reported (needs per-step speed, not saved by eval_suite). '
          'Webots: one 4 ms extra command delay vs single-drone Level 2; yaw-rate command 0.', '',
          'Standard random is compared unpaired: PyBullet\'s lag model draws per-episode delays from the '
          'global numpy generator that also draws this test\'s scenarios; Webots ignores the lag model, so '
          'the scenarios differ after episode 0 of each seed. The pooled standard row uses the other 4 tests.', '',
          'Lost control = max |roll| or |pitch| > 60 deg (recomputed from the saved per-drone stats; the '
          'lost_control columns written during the runs used the older rule with speed > 3 m/s). The true '
          'speed after a flip is not physical (up to hundreds of m/s) and is not used.', '']
    open(os.path.join(root, 'c1_compare.md'), 'w').write('\n'.join(L))
    fields = sorted({k for r in rows for k in r})
    with open(os.path.join(root, 'c1_compare.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print('\n'.join(L))


if __name__ == '__main__':
    main()
