"""
eval_birds.py
-------------
Stage 2 of the birds work: the shipped policy (goal-hold on), no retraining, flown against birds
on the validation seeds 20-24. Every bird cell is paired with a no-bird cell: same seeds, same
spawns and goals, same lag and sensor noise (birds draw from their own generator), so the only
difference is the birds. Uses eval_suite.py for the flying and eval_stats.py for the statistics
(95% Wilson intervals, exact McNemar on the paired episodes, Holm correction per family).

    python scripts/eval_birds.py run      --checkpoint <ckpt> [--only single|speed|crowd]
    python scripts/eval_birds.py analyze                       # tables -> <out>/tables.md, results.json

Cells (episodes per cell = 5 seeds x EPS):
    single   1 drone, 7-10 m flight, with and without a static obstacle, 0 / 1 / 2 / 4 birds
    speed    1 drone, 2 birds at 1 / 2 / 3 m/s (no obstacle)
    crowd    8 drones head-on, no static obstacles, 0 / 2 / 4 birds

Encounter: a bird came within 1.5 m (centre to centre) of a drone. Only cells with a real encounter
rate say anything about avoidance. Detection: of the (drone, bird) pairs within the 6 m LiDAR range
in a step, the fraction where at least one ray (14-ray fan or 32-ray ring) hit the bird.
"""

import argparse
import csv
import glob
import json
import math
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, ROOT)
from eval_stats import wilson, mcnemar_exact, holm            # noqa: E402

SEEDS = ['20', '21', '22', '23', '24']                        # validation seeds only (never 10-19)
CONFIG = 'configs/config_stage2_round7_eval_goalhold.yaml'
ENCOUNTER_M = 1.5
NOBS = ['--scenario', 'random', '--goal-dist', '10', '--goal-dist-min', '7', '--n-agents', '1', '--no-obstacles']
OBS = ['--scenario', 'building']
CROWD = ['--scenario', 'headon', '--n-agents', '8', '--no-obstacles']


def cells(eps_single, eps_crowd):
    """name -> (family, baseline name or None, extra eval_suite args, episodes per seed, bird speed or None)."""
    out = {}
    for fam, base in (('single_noobs', NOBS), ('single_obs', OBS)):
        for n in (0, 1, 2, 4):
            out[f'{fam}_b{n}'] = (fam, f'{fam}_b0' if n else None, base + (['--birds', str(n)] if n else []), eps_single, None)
    for v in (1, 2, 3):
        out[f'speed_v{v}'] = ('speed', 'single_noobs_b0', NOBS + ['--birds', '2', '--bird-speed', str(v), str(v)], eps_single, v)
    for n in (0, 2, 4):
        out[f'crowd_b{n}'] = ('crowd', 'crowd_b0' if n else None, CROWD + (['--birds', str(n)] if n else []), eps_crowd, None)
    return out


def run(args):
    for name, (fam, _, extra, eps, _) in cells(args.eps_single, args.eps_crowd).items():
        if args.only and not name.startswith(args.only):
            continue
        out = os.path.join(args.out, name)
        if os.path.exists(os.path.join(out, 'summary.json')) and not args.overwrite:
            print(f'{name}: done already, skipping')
            continue
        cmd = [sys.executable, os.path.join(ROOT, 'eval_suite.py'), '--config', CONFIG, '--checkpoint', args.checkpoint,
               '--seeds', *SEEDS, '--episodes', str(eps), '--collision-physics', 'off',
               '--out-dir', out, '--overwrite', *extra]
        print(f'{name}: {" ".join(extra)}', flush=True)
        subprocess.run(cmd, check=True, cwd=ROOT, stdout=subprocess.DEVNULL)


def load(out, name):
    with open(os.path.join(out, name, 'episodes.csv'), newline='') as f:
        rows = list(csv.DictReader(f))
    with open(os.path.join(out, name, 'summary.json')) as f:
        return rows, json.load(f)


def num(r, k):
    return float(r[k]) if r.get(k, '') != '' else float('nan')


def analyze(args):
    spec = cells(args.eps_single, args.eps_crowd)
    data = {n: load(args.out, n) for n in spec if os.path.exists(os.path.join(args.out, n, 'episodes.csv'))}
    res, tests = {}, []
    for name, (fam, base, _, _, speed) in spec.items():
        if name not in data:
            continue
        rows, summ = data[name]
        n = len(rows)
        k = sum(int(r['success']) for r in rows)
        oc = [r['outcome'] for r in rows]
        d = {'family': fam, 'baseline': base, 'n': n, 'success': k, 'success_ci': wilson(k, n),
             'bird_crash': oc.count('bird'), 'other_crash': oc.count('collision'), 'timeout': oc.count('timeout')}
        gh = summ.get('goal_hold_stats')
        if gh:
            d['guard_frac'] = gh['guard_frac']
        if 'min_bird_dist' in rows[0]:
            md = np.array([num(r, 'min_bird_dist') for r in rows])
            enc = int((md < ENCOUNTER_M).sum())
            d.update(encounter=enc, encounter_ci=wilson(enc, n), min_bird_dist_q=[float(np.nanpercentile(md, q)) for q in (0, 10, 50, 90, 100)])
            e_idx = md < ENCOUNTER_M
            d['success_given_encounter'] = [int(sum(int(r['success']) for r, m in zip(rows, e_idx) if m)), enc]
            tin, tsn = (sum(int(r['bird_pair_steps']) for r in rows), sum(int(r['bird_seen_steps']) for r in rows))
            d['detect'] = [tsn, tin, tsn / tin if tin else float('nan')]
            d['detect_ci'] = wilson(tsn, tin)
            d['detect_bins'] = {}
            for b in ('0_2', '2_4', '4_6'):
                a, s = sum(int(r[f'bird_in_{b}']) for r in rows), sum(int(r[f'bird_seen_{b}']) for r in rows)
                d['detect_bins'][b] = [s, a, s / a if a else float('nan')]
            d['bird_speed'] = speed
        res[name] = d
    # paired tests against the family's no-bird cell
    for name, d in res.items():
        if not d['baseline'] or d['baseline'] not in data:
            continue
        a = {(r['seed'], r['episode']): r for r in data[d['baseline']][0]}
        b = {(r['seed'], r['episode']): r for r in data[name][0]}
        assert a.keys() == b.keys() and all(a[k]['scenario_id'] == b[k]['scenario_id'] for k in a), f'{name}: not paired'
        only_base = sum(1 for k in a if int(a[k]['success']) and not int(b[k]['success']))
        only_bird = sum(1 for k in a if int(b[k]['success']) and not int(a[k]['success']))
        d.update(paired_lost=only_base, paired_gained=only_bird, p_raw=mcnemar_exact(only_base, only_bird))
        tests.append(name)
    for fam in {res[t]['family'] for t in tests}:
        names = [t for t in tests if res[t]['family'] == fam]
        for t, p in zip(names, holm([res[t]['p_raw'] for t in names])):
            res[t]['p_holm'] = p
    with open(os.path.join(args.out, 'results.json'), 'w') as f:
        json.dump(res, f, indent=1, default=str)
    lines = ['| cell | n | success [95% CI] | bird crash | other crash | timeout | encounter <1.5 m [95% CI] | closest p10/med/p90 (m) | success given encounter | lost / gained vs no birds | p (Holm) | detection |',
             '|---|---|---|---|---|---|---|---|---|---|---|---|']
    for name, d in res.items():
        lo, hi = d['success_ci']
        enc = (f"{d['encounter']}/{d['n']} [{d['encounter_ci'][0]:.2f}-{d['encounter_ci'][1]:.2f}]" if 'encounter' in d else '-')
        q = d.get('min_bird_dist_q')
        sg = d.get('success_given_encounter')
        det = (f"{d['detect'][0]}/{d['detect'][1]} = {100 * d['detect'][2]:.1f}% (0-2 m {100 * d['detect_bins']['0_2'][2]:.0f}%, 2-4 m {100 * d['detect_bins']['2_4'][2]:.0f}%, 4-6 m {100 * d['detect_bins']['4_6'][2]:.0f}%)"
               if 'detect' in d else '-')
        lines.append(f"| {name} | {d['n']} | {d['success']}/{d['n']} = {100 * d['success'] / d['n']:.0f}% [{100 * lo:.0f}-{100 * hi:.0f}] | "
                     f"{d['bird_crash']} | {d['other_crash']} | {d['timeout']} | {enc} | "
                     f"{'-' if q is None else f'{q[1]:.2f} / {q[2]:.2f} / {q[3]:.2f}'} | "
                     f"{'-' if sg is None else f'{sg[0]}/{sg[1]}'} | "
                     f"{'-' if 'p_raw' not in d else str(d['paired_lost']) + ' / ' + str(d['paired_gained'])} | "
                     f"{'-' if 'p_holm' not in d else f'{d['p_holm']:.3f}'} | {det} |")
    with open(os.path.join(args.out, 'tables.md'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['run', 'analyze'])
    ap.add_argument('--checkpoint')
    ap.add_argument('--out', default=os.path.join('eval_results', 'birds'))
    ap.add_argument('--eps-single', type=int, default=10, help='episodes per seed, single-drone cells (5 seeds)')
    ap.add_argument('--eps-crowd', type=int, default=10, help='episodes per seed, crowd cells (5 seeds)')
    ap.add_argument('--only', default=None, help='cells whose name starts with this')
    ap.add_argument('--overwrite', action='store_true')
    args = ap.parse_args()
    if args.mode == 'run':
        if not args.checkpoint:
            ap.error('--checkpoint is required for run')
        run(args)
    else:
        analyze(args)


if __name__ == '__main__':
    main()
