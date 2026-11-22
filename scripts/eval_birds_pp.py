"""
eval_birds_pp.py
----------------
Stage 3a++ of the birds work: push the bird guard further (shared tracks, brake candidates, a Kalman tracker, tuned
parameters), then confirm it on episodes that were not used for tuning. Uses eval_suite.py for the flying and
eval_birds.py / eval_stats.py for the statistics (95 % Wilson intervals, paired exact McNemar, Holm).

    python scripts/eval_birds_pp.py run     --phase tune    --checkpoint <ckpt> [--only PREFIX]
    python scripts/eval_birds_pp.py analyze --phase tune
    (choose the final configuration, write it to <out>/pp/final_config.json, see FINAL_EXAMPLE)
    python scripts/eval_birds_pp.py run     --phase confirm --checkpoint <ckpt>
    python scripts/eval_birds_pp.py run     --phase final   --checkpoint <ckpt>
    python scripts/eval_birds_pp.py analyze --phase confirm | final

Episodes (no episode is used twice):
    tune     seeds 20-22, episodes 0-14 of each seed (45 per cell), worst-case and fair crowd, 4 birds
    confirm  seeds 23-24, episodes 0-19 (40 per cell), same cells
    final    seeds 20-22, episodes 15-44 (--episode-offset 15), and seeds 23-24, episodes 0-29: 150 per cell, all
             held out from the tuning; the seeds 23-24 episodes include the confirmation's
The configuration under test is a dict (DEFAULT_CFG below) rendered to eval_suite flags; birds are never in the
neighbour slots here (the guard uses the tracks), and the policy, goal-hold and sensor noise are unchanged.
"""

import argparse
import copy
import csv
import json
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_stats import wilson, mcnemar_exact, holm            # noqa: E402
import eval_birds as eb                                       # noqa: E402

CONFIG = eb.CONFIG
ENCOUNTER_M = eb.ENCOUNTER_M
# the previous best (stage 3a+ sensitivity): guard on the realistic detector with a 360 degree view and 12 m of range
PREV_CFG = {'range': 12, 'fov': 360, 'share': False, 'tracker': None, 'brake': None, 'guard_set': {}, 'detector_set': {}}
FINAL_EXAMPLE = {'range': 12, 'fov': 360, 'share': True, 'tracker': 'kalman', 'brake': [0.0, 0.5],
                 'guard_set': {'miss_m': 1.2, 'release_m': 1.6}, 'detector_set': {'kalman_q': 2.0}}
ORACLE_FLAGS = ['--bird-detector', 'oracle', '--no-bird-slots', '--bird-guard']


def render(cfg):
    """Detector/guard configuration (dict) -> eval_suite flags. Birds never fill the neighbour slots."""
    f = ['--bird-detector', 'realistic', '--no-bird-slots', '--bird-guard',
         '--detector-range', str(cfg['range']), '--detector-fov', str(cfg['fov'])]
    if cfg.get('share'):
        f += ['--detector-share']
    if cfg.get('tracker'):
        f += ['--detector-tracker', cfg['tracker']]
    if cfg.get('brake') is not None:
        f += ['--guard-brake', *[str(x) for x in cfg['brake']]]
    for k, v in (cfg.get('guard_set') or {}).items():
        f += ['--guard-set', f'{k}={v}']
    for k, v in (cfg.get('detector_set') or {}).items():
        f += ['--detector-set', f'{k}={v}']
    return f


def variant(**changes):
    cfg = copy.deepcopy(PREV_CFG)
    for k, v in changes.items():
        if k in ('guard_set', 'detector_set'):
            cfg[k] = {**cfg[k], **v}
        else:
            cfg[k] = v
    return cfg


# one or two settings at a time around the previous best (a threshold above release_m raises release_m with it)
TUNE_VARIANTS = {
    'miss0.8': variant(guard_set={'miss_m': 0.8}), 'miss1.2': variant(guard_set={'miss_m': 1.2}),
    'miss1.5': variant(guard_set={'miss_m': 1.5, 'release_m': 1.9}),
    'hor2': variant(guard_set={'horizon_s': 2.0}), 'hor4': variant(guard_set={'horizon_s': 4.0}),
    'evade1.0': variant(guard_set={'evade_speed': 1.0}), 'evade2.0': variant(guard_set={'evade_speed': 2.0}),
    'pred0.2': variant(guard_set={'predict_s': 0.2}), 'pred0.6': variant(guard_set={'predict_s': 0.6}),
    'rel1.2': variant(guard_set={'release_m': 1.2}), 'rel1.8': variant(guard_set={'release_m': 1.8}),
    'brake': variant(brake=[0.0, 0.5]),
    'kalman': variant(tracker='kalman'), 'kalman_q2': variant(tracker='kalman', detector_set={'kalman_q': 2.0}),
    'share': variant(share=True),
    'cheap': variant(range=8, fov=120), 'cheap_share': variant(range=8, fov=120, share=True),
    # added after the single-setting results: the best settings together (kalman_q2, miss1.2, pred0.2)
    'combo': variant(tracker='kalman', detector_set={'kalman_q': 2.0}, guard_set={'miss_m': 1.2, 'predict_s': 0.2}),
    'combo_share': variant(tracker='kalman', share=True, detector_set={'kalman_q': 2.0},
                           guard_set={'miss_m': 1.2, 'predict_s': 0.2}),
}
TUNE_CELLS = {'worst': ('crowd_b4', eb.CROWD + ['--birds', '4']), 'fair': ('fcrowd_b4', eb.CROWD + ['--birds', '4'] + eb.FAIR)}
SINGLES = ['single_noobs_b2', 'single_noobs_b4', 'single_obs_b2', 'single_obs_b4', 'speed_v3']
FINAL_CROWD = {'worst_b2': eb.CROWD + ['--birds', '2'], 'worst_b4': eb.CROWD + ['--birds', '4'],
               'fair_b2': eb.CROWD + ['--birds', '2'] + eb.FAIR, 'fair_b4': eb.CROWD + ['--birds', '4'] + eb.FAIR}


def strip_birds(args):
    """The same scenario without birds (the cost baseline)."""
    out, skip = [], 0
    for a in args:
        if skip:
            skip -= 1
        elif a in ('--birds', '--bird-spawn', '--bird-speed'):
            skip = 2 if a == '--bird-speed' else 1
        else:
            out.append(a)
    return out


def phase_runs(args):
    """list of (name, scenario args, extra flags, seeds, episodes, offset) for the phase."""
    runs = []
    if args.phase == 'tune':
        for grp, (cell, scen) in TUNE_CELLS.items():
            runs.append((f'tune_{cell}_none', scen, [], ['20', '21', '22'], 15, 0))
            runs.append((f'tune_{cell}_base', scen, render(PREV_CFG), ['20', '21', '22'], 15, 0))
            for v, cfg in TUNE_VARIANTS.items():
                runs.append((f'tune_{cell}_{v}', scen, render(cfg), ['20', '21', '22'], 15, 0))
        return runs
    final_cfg = json.load(open(os.path.join(args.out, 'final_config.json')))
    conds = {'none': [], 'prev': render(PREV_CFG), 'new': render(final_cfg), 'oracle': ORACLE_FLAGS}
    if args.phase == 'confirm':
        parts = {'B': (['23', '24'], 20, 0)}
        cells = {k: v for k, v in FINAL_CROWD.items() if k in ('worst_b4', 'fair_b4')}
        singles = []
    else:
        parts = {'A': (['20', '21', '22'], 30, 15), 'B': (['23', '24'], 30, 0)}
        cells, singles = FINAL_CROWD, SINGLES
    pre = args.phase
    for part, (seeds, eps, off) in parts.items():
        for c, scen in cells.items():
            for cond, flags in conds.items():
                runs.append((f'{pre}_{c}_{cond}_{part}', scen, flags, seeds, eps, off))
            runs.append((f'{pre}_{c}_nobird_{part}', strip_birds(scen), [], seeds, eps, off))
        base = eb.cells(10, 10)
        for c in singles:
            for cond in ('none', 'prev', 'new'):
                runs.append((f'{pre}_{c}_{cond}_{part}', base[c][2], conds[cond], seeds, eps, off))
        for nb in ({f'{pre}_single_noobs_nobird_{part}': eb.NOBS, f'{pre}_single_obs_nobird_{part}': eb.OBS} if singles else {}).items():
            runs.append((nb[0], nb[1], [], seeds, eps, off))
    return runs


def run(args):
    for name, scen, flags, seeds, eps, off in phase_runs(args):
        if args.only and not name.startswith(args.only):
            continue
        out = os.path.join(args.out, name)
        if os.path.exists(os.path.join(out, 'summary.json')) and not args.overwrite:
            print(f'{name}: done already, skipping', flush=True)
            continue
        cmd = [sys.executable, os.path.join(ROOT, 'eval_suite.py'), '--config', CONFIG, '--checkpoint', args.checkpoint,
               '--seeds', *seeds, '--episodes', str(eps), '--collision-physics', 'off', '--out-dir', out, '--overwrite',
               *(['--episode-offset', str(off)] if off else []), *scen, *flags]
        print(f'{name}: {" ".join(scen + flags)}', flush=True)
        if args.dry_run:
            continue
        subprocess.run(cmd, check=True, cwd=ROOT, stdout=subprocess.DEVNULL)


# ------------------------------------------------------------------ analysis
def rows_of(out, names):
    rows = []
    for n in names:
        with open(os.path.join(out, n, 'episodes.csv'), newline='') as f:
            rows += list(csv.DictReader(f))
    return rows


def agents_of(out, names):
    rows = []
    for n in names:
        with open(os.path.join(out, n, 'agents.csv'), newline='') as f:
            rows += list(csv.DictReader(f))
    return rows


def metrics(rows):
    d = eb.cell_metrics(rows)
    if 'bird_tracked_steps' in rows[0]:
        tot = sum(int(r['bird_pair_steps']) for r in rows)
        d['track_rate'] = sum(int(r['bird_tracked_steps']) for r in rows) / tot if tot else float('nan')
    return d


def paired(a_rows, b_rows):
    a = {(r['seed'], r['episode']): r for r in a_rows}
    b = {(r['seed'], r['episode']): r for r in b_rows}
    assert a.keys() == b.keys() and all(a[k]['scenario_id'] == b[k]['scenario_id'] for k in a), 'not paired'
    lost = sum(1 for k in a if int(a[k]['success']) and not int(b[k]['success']))
    gained = sum(1 for k in a if int(b[k]['success']) and not int(a[k]['success']))
    return lost, gained, mcnemar_exact(lost, gained)


def analyze_tune(args):
    lines = []
    for grp, (cell, _) in TUNE_CELLS.items():
        names = [f'tune_{cell}_{v}' for v in ('none', 'base', *TUNE_VARIANTS)]
        have = [n for n in names if os.path.exists(os.path.join(args.out, n, 'episodes.csv'))]
        data = {n: rows_of(args.out, [n]) for n in have}
        lines += ['', f"### Tuning, {'worst-case' if grp == 'worst' else 'fair'} crowd, 4 birds (seeds 20-22, episodes 0-14: {len(data[have[0]]) if have else 0} per run)", '',
                  '| variant | success | bird / d-d crashes | closest bird, median (m) | track rate | guard active | vs base (lost / gained), p | vs none (lost / gained) |', '|---|---|---|---|---|---|---|---|']
        for n in have:
            m = metrics(data[n])
            vb = paired(data[f'tune_{cell}_base'], data[n]) if n != f'tune_{cell}_base' and f'tune_{cell}_base' in data else None
            vn = paired(data[f'tune_{cell}_none'], data[n]) if n != f'tune_{cell}_none' and f'tune_{cell}_none' in data else None
            lines.append(f"| {n.split(cell + '_')[1]} | {m['success']}/{m['n']} | {m['bird']} / {m['drone_drone']} | {m.get('closest_median', float('nan')):.2f} | "
                         f"{('%.0f %%' % (100 * m['track_rate'])) if 'track_rate' in m else '-'} | {('%.1f %%' % (100 * m['guard_active_frac'])) if 'guard_active_frac' in m else '-'} | "
                         f"{('%d / %d, p %.2f' % (vb[0], vb[1], vb[2])) if vb else '-'} | {('%d / %d' % vn[:2]) if vn else '-'} |")
    with open(os.path.join(args.out, 'tables_tune.md'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


def cost(out, names, nobird_names):
    a = {(r['seed'], r['episode'], r['agent']): r for r in agents_of(out, nobird_names)}
    ratios, extra = [], []
    for r in agents_of(out, names):
        k = (r['seed'], r['episode'], r['agent'])
        if k in a and int(r['settled']) and int(a[k]['settled']) and float(a[k]['path_length']) > 0:
            ratios.append(float(r['path_length']) / float(a[k]['path_length']))
            extra.append(float(r['settle_time_s']) - float(a[k]['settle_time_s']))
    return (float(np.mean(ratios)), float(np.mean(extra))) if ratios else (float('nan'), float('nan'))


def analyze_final(args):
    pre, parts = args.phase, (['B'] if args.phase == 'confirm' else ['A', 'B'])
    groups = [('Worst-case crowd, %s birds', [(f'worst_b{n}', f'{n}') for n in ((4,) if pre == 'confirm' else (2, 4))], ('none', 'prev', 'new', 'oracle')),
              ('Fair crowd, %s birds', [(f'fair_b{n}', f'{n}') for n in ((4,) if pre == 'confirm' else (2, 4))], ('none', 'prev', 'new', 'oracle'))]
    if pre == 'final':
        groups.append(('Single drone: %s', [(c, c) for c in SINGLES], ('none', 'prev', 'new')))
    labels = {'none': 'no avoidance', 'prev': 'previous best (guard, 360 deg, 12 m)', 'new': 'new final configuration', 'oracle': 'oracle (upper bound)'}
    lines, res, tests = [], {}, []
    for title, cells_, conds in groups:
        for c, lab in cells_:
            data = {cond: rows_of(args.out, [f'{pre}_{c}_{cond}_{p}' for p in parts]) for cond in conds
                    if all(os.path.exists(os.path.join(args.out, f'{pre}_{c}_{cond}_{p}', 'episodes.csv')) for p in parts)}
            if 'none' not in data:
                continue
            nb = ('crowd' if c.startswith(('worst', 'fair')) else 'single_obs' if c.startswith('single_obs') else 'single_noobs')
            nbn = [f"{pre}_{'single_obs' if nb == 'single_obs' else 'single_noobs' if nb == 'single_noobs' else c}_nobird_{p}" for p in parts]
            lines += ['', f'### {title % lab} ({len(data["none"])} episodes per condition)', '',
                      '| condition | success [95 % CI] | bird / drone-drone / obstacle crashes, timeouts | closest bird, median (m) | guard active | extra path / settle time vs no birds | vs none: p (lost / gained) | vs previous best: p (lost / gained) |',
                      '|---|---|---|---|---|---|---|---|']
            for cond in conds:
                if cond not in data:
                    continue
                m = metrics(data[cond])
                lo, hi = m['success_ci']
                try:
                    co = cost(args.out, [f'{pre}_{c}_{cond}_{p}' for p in parts], nbn)
                except FileNotFoundError:
                    co = (float('nan'), float('nan'))
                vn = paired(data['none'], data[cond]) if cond != 'none' else None
                vp = paired(data['prev'], data[cond]) if cond not in ('none', 'prev') and 'prev' in data else None
                key = f'{c}:{cond}'
                res[key] = {**m, 'cost': co, 'vs_none': vn, 'vs_prev': vp}
                for tag, v in (('none', vn), ('prev', vp)):
                    if v:
                        tests.append((key, tag, v[2], title.split(' ')[0]))
                lines.append(f"| {labels[cond]} | {m['success']}/{m['n']} = {100 * m['success'] / m['n']:.0f} % [{100 * lo:.0f}-{100 * hi:.0f}] | "
                             f"{m['bird']} / {m['drone_drone']} / {m['obstacle']}, {m['timeout']} | {m.get('closest_median', float('nan')):.2f} | "
                             f"{('%.1f %%' % (100 * m['guard_active_frac'])) if 'guard_active_frac' in m else '-'} | "
                             f"{('x%.2f / %+.1f s' % co) if co[0] == co[0] else '-'} | "
                             f"{('%s (%d / %d)' % (f'{vn[2]:.3f}', vn[0], vn[1])) if vn else '-'} | {('%s (%d / %d)' % (f'{vp[2]:.3f}', vp[0], vp[1])) if vp else '-'} |")
    # Holm over all the comparisons of one family (crowd cells together, singles together)
    fam = {}
    for key, tag, p, grp in tests:
        fam.setdefault('single' if grp == 'Single' else 'crowd', []).append((key, tag, p))
    for f_, items in fam.items():
        adj = holm([p for _, _, p in items])
        for (key, tag, _), a in zip(items, adj):
            res[key].setdefault('holm', {})[tag] = a
    lines += ['', 'Holm-adjusted p-values (family: all crowd comparisons, or all single-drone comparisons):', '']
    for key, r in res.items():
        if 'holm' in r:
            lines.append(f"* {key}: " + ', '.join(f'vs {t} {v:.3f}' for t, v in r['holm'].items()))
    with open(os.path.join(args.out, f'tables_{pre}.md'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    with open(os.path.join(args.out, f'results_{pre}.json'), 'w') as f:
        json.dump(res, f, indent=1, default=str)
    print('\n'.join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['run', 'analyze'])
    ap.add_argument('--phase', choices=['tune', 'confirm', 'final'], required=True)
    ap.add_argument('--checkpoint')
    ap.add_argument('--out', default=os.path.join('eval_results', 'birds', 'pp'))
    ap.add_argument('--only', default=None, help='runs whose name starts with this')
    ap.add_argument('--overwrite', action='store_true')
    ap.add_argument('--dry-run', action='store_true', help='print the runs, fly nothing')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    if args.mode == 'run':
        if not args.checkpoint and not args.dry_run:
            ap.error('--checkpoint is required for run')
        run(args)
    elif args.phase == 'tune':
        analyze_tune(args)
    else:
        analyze_final(args)


if __name__ == '__main__':
    main()
