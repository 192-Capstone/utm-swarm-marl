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
    python scripts/eval_birds.py run --stage 3a --checkpoint <ckpt>   # stage 3a: onboard bird detector
    python scripts/eval_birds.py analyze --stage 3a            # -> <out>/tables_3a.md, results_3a.json
    python scripts/eval_birds.py run --stage 3a+ --checkpoint <ckpt>  # stage 3a+: bird guard, fair crowd
    python scripts/eval_birds.py analyze --stage 3a+           # -> <out>/tables_3ap.md, results_3ap.json
    python scripts/eval_birds.py run --stage sens --checkpoint <ckpt>  # guard sensitivity to the detector
    python scripts/eval_birds.py analyze --stage sens          # -> <out>/tables_sens.md, results_sens.json

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


DET_CELLS = ['single_noobs_b1', 'single_noobs_b2', 'single_noobs_b4', 'single_obs_b1', 'single_obs_b2', 'single_obs_b4',
             'speed_v1', 'speed_v2', 'speed_v3', 'crowd_b2', 'crowd_b4']
ORACLE_CELLS = ['crowd_b2', 'crowd_b4', 'single_noobs_b2', 'single_noobs_b4', 'single_obs_b4']
SENSITIVITY = [('crowd_b2', 4, 120), ('crowd_b2', 8, 360), ('single_noobs_b2', 4, 120), ('single_noobs_b2', 8, 360)]


def cells_3a(eps_single, eps_crowd):
    """Stage 3a: the stage-2 cells flown with an onboard detector. Each is paired with the same stage-2 cell
    (same seeds, spawns, noise and bird paths, no detector). name -> same tuple as cells(); baseline = the
    stage-2 cell."""
    base = cells(eps_single, eps_crowd)
    out = {}

    def add(name, src, flags):
        fam, _, extra, eps, speed = base[src]
        out[name] = ('3a_' + fam, src, extra + flags, eps, speed)
    for c in DET_CELLS:
        add(c + '_det', c, ['--bird-detector', 'realistic'])
    for c in ORACLE_CELLS:
        add(c + '_oracle', c, ['--bird-detector', 'oracle'])
    for c, r, f in SENSITIVITY:
        add(f'{c}_det_r{r}_f{f}', c, ['--bird-detector', 'realistic', '--detector-range', str(r), '--detector-fov', str(f)])
    return out


# ---------------------------------------------------------------- stage 3a+
# Conditions (every cell paired with the same episodes of the 'none' cell, same seeds, noise and bird paths):
#   none (stage 2: LiDAR only), det (stage 3a: detector + slots), g0 (guard, birds NOT in the slots),
#   g1 (guard, birds in the slots), go (guard on the ORACLE detector, birds not in the slots: an upper bound)
CONDITIONS = {'det': ['--bird-detector', 'realistic'],
              'g0': ['--bird-detector', 'realistic', '--no-bird-slots', '--bird-guard'],
              'g1': ['--bird-detector', 'realistic', '--bird-guard'],
              'go': ['--bird-detector', 'oracle', '--no-bird-slots', '--bird-guard']}
SINGLE_GUARD = ['single_noobs_b2', 'single_noobs_b4', 'single_obs_b2', 'single_obs_b4', 'speed_v3']
FAIR = ['--bird-spawn', 'random']


def cells_3ap(eps_single, eps_crowd):
    """name -> (group, none-cell name, extra args, episodes per seed, no-bird cell name). Stage 2 and 3a cells
    (none / det of the timed cells) are reused from their own runs."""
    base = cells(eps_single, eps_crowd)
    out = {}
    for c in ('crowd_b2', 'crowd_b4'):                                    # worst case: timed birds
        for cond in ('g0', 'g1', 'go'):
            out[f'{c}_{cond}'] = ('worst', c, base[c][2] + CONDITIONS[cond], eps_crowd, 'crowd_b0')
    for n in (2, 4):                                                      # fair: random straight crossings
        extra = CROWD + ['--birds', str(n)] + FAIR
        out[f'fcrowd_b{n}'] = ('fair', f'fcrowd_b{n}', extra, eps_crowd, 'crowd_b0')
        for cond in ('det', 'g0', 'g1', 'go'):
            out[f'fcrowd_b{n}_{cond}'] = ('fair', f'fcrowd_b{n}', extra + CONDITIONS[cond], eps_crowd, 'crowd_b0')
    for c in SINGLE_GUARD:
        for cond in ('g0', 'g1'):
            nob = 'single_obs_b0' if c.startswith('single_obs') else 'single_noobs_b0'
            out[f'{c}_{cond}'] = ('single', c, base[c][2] + CONDITIONS[cond], eps_single, nob)
    return out


# ---------------------------------------------------------------- guard sensitivity to the detector
# One detector setting changed at a time around the default (range 8 m, FOV 120 deg, detection probability 0.9 near
# falling to 0.5 at the range, latency 3 steps), plus one combined best case. Guard on the realistic detector, birds
# not in the slots, 4 birds; the default is the stage 3a+ 'g0' cell, the oracle is its 'go' cell.
SENS_VARIANTS = {
    'r4': ['--detector-range', '4'], 'r12': ['--detector-range', '12'], 'f360': ['--detector-fov', '360'],
    'plow': ['--detector-p', '0.7', '0.3'], 'phigh': ['--detector-p', '1.0', '1.0'],
    'L0': ['--detector-latency', '0'], 'L9': ['--detector-latency', '9'],
    'r12f360': ['--detector-range', '12', '--detector-fov', '360'],
}
SENS_CELLS = {'crowd_b4': 'worst', 'fcrowd_b4': 'fair'}


def cells_sens(eps_single, eps_crowd):
    """name -> (group, none-cell, extra args, episodes per seed, no-bird cell), like cells_3ap."""
    ap_cells = cells_3ap(eps_single, eps_crowd)
    out = {}
    for c, grp in SENS_CELLS.items():
        extra = ap_cells[f'{c}_g0'][2]                                   # the default guard cell's arguments
        for v, flags in SENS_VARIANTS.items():
            out[f'{c}_g0_{v}'] = (grp, c, extra + flags, eps_crowd, 'crowd_b0')
    return out


def all_cells(args):
    if args.stage == 'sens':
        return cells_sens(args.eps_single, args.eps_crowd)
    if args.stage == '3a':
        return cells_3a(args.eps_single, args.eps_crowd)
    if args.stage == '3a+':
        return cells_3ap(args.eps_single, args.eps_crowd)
    return cells(args.eps_single, args.eps_crowd)


def run(args):
    for name, (fam, _, extra, eps, _) in all_cells(args).items():
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


def load_agents(out, name):
    with open(os.path.join(out, name, 'agents.csv'), newline='') as f:
        return list(csv.DictReader(f))


def cell_metrics(rows):
    n = len(rows)
    oc = [r['outcome'] for r in rows]
    d = {'n': n, 'success': sum(int(r['success']) for r in rows), 'bird': oc.count('bird'),
         'drone_drone': sum(int(r['any_drone_collision']) for r in rows),
         'obstacle': sum(int(r['any_obstacle_collision']) for r in rows), 'timeout': oc.count('timeout')}
    d['success_ci'] = wilson(d['success'], n)
    if 'min_bird_dist' in rows[0]:
        md = np.array([num(r, 'min_bird_dist') for r in rows])
        d['closest_median'] = float(np.nanmedian(md))
        d['encounter'] = int((md < ENCOUNTER_M).sum())
    if 'guard_active_steps' in rows[0]:
        ds = sum(int(r['guard_drone_steps']) for r in rows)
        d['guard_active_frac'] = sum(int(r['guard_active_steps']) for r in rows) / max(ds, 1)
        d['guard_episodes'] = sum(int(r['guard_activated']) for r in rows)
    return d


def cost_vs(out, name, nobird):
    """Extra path length (ratio of means over drones that settled in both runs) and extra settle time (s)
    against the same drones in the same episodes without birds."""
    a = {(r['seed'], r['episode'], r['agent']): r for r in load_agents(out, nobird)}
    ratios, extra = [], []
    for r in load_agents(out, name):
        k = (r['seed'], r['episode'], r['agent'])
        if k in a and int(r['settled']) and int(a[k]['settled']) and float(a[k]['path_length']) > 0:
            ratios.append(float(r['path_length']) / float(a[k]['path_length']))
            extra.append(float(r['settle_time_s']) - float(a[k]['settle_time_s']))
    return (float(np.mean(ratios)), float(np.mean(extra)), len(ratios)) if ratios else (float('nan'), float('nan'), 0)


def analyze_sens(args):
    spec = cells_sens(args.eps_single, args.eps_crowd)
    need = set(spec) | {v[1] for v in spec.values()} | {f'{c}_g0' for c in SENS_CELLS} | {f'{c}_go' for c in SENS_CELLS}
    data = {n: load(args.out, n) for n in need if os.path.exists(os.path.join(args.out, n, 'episodes.csv'))}
    res = {n: cell_metrics(data[n][0]) for n in data}
    for n in data:
        rows = data[n][0]
        if 'bird_tracked_steps' in rows[0]:
            trk, tot = sum(int(r['bird_tracked_steps']) for r in rows), sum(int(r['bird_pair_steps']) for r in rows)
            res[n]['track_rate'] = trk / tot if tot else float('nan')
    for c in SENS_CELLS:                                              # paired against none; the variants against the default
        none = c
        default = f'{c}_g0'
        names = [n for n in data if n.startswith(f'{c}_g0_')] + [default, f'{c}_go']
        tests = []
        for n in names:
            if n not in data or none not in data:
                continue
            a = {(r['seed'], r['episode']): r for r in data[none][0]}
            b = {(r['seed'], r['episode']): r for r in data[n][0]}
            assert a.keys() == b.keys() and all(a[k]['scenario_id'] == b[k]['scenario_id'] for k in a), f'{n}: not paired'
            lost = sum(1 for k in a if int(a[k]['success']) and not int(b[k]['success']))
            gained = sum(1 for k in a if int(b[k]['success']) and not int(a[k]['success']))
            res[n].update(lost=lost, gained=gained, p_raw=mcnemar_exact(lost, gained))
            if default in data and n != default:
                d = {(r['seed'], r['episode']): r for r in data[default][0]}
                res[n].update(vs_default=(sum(1 for k in d if int(d[k]['success']) and not int(b[k]['success'])),
                                          sum(1 for k in d if int(b[k]['success']) and not int(d[k]['success']))))
            tests.append(n)
        for n, p in zip(tests, holm([res[t]['p_raw'] for t in tests])):
            res[n]['p_holm'] = p
    with open(os.path.join(args.out, 'results_sens.json'), 'w') as f:
        json.dump(res, f, indent=1, default=str)
    labels = {'r4': 'range 4 m', 'r12': 'range 12 m', 'f360': 'FOV 360 deg', 'plow': 'detection 0.7 -> 0.3',
              'phigh': 'detection 1.0 (no misses)', 'L0': 'latency 0', 'L9': 'latency 9 steps (0.29 s)', 'r12f360': 'range 12 m + FOV 360 deg'}
    lines = []
    for c, grp in SENS_CELLS.items():
        lines += ['', f"### {'Worst-case' if grp == 'worst' else 'Fair'} crowd, 4 birds ({c})", '',
                  '| detector | successes of 50 | bird / drone-drone crashes | track rate (pairs < 6 m) | closest bird, median (m) | guard active | raw p vs none (lost / gained) | Holm p | vs default guard (lost / gained) |',
                  '|---|---|---|---|---|---|---|---|---|']
        rows = [('none (no avoidance)', c), ('default: 8 m, 120 deg, 0.9 -> 0.5, 3 steps', f'{c}_g0')]
        rows += [(labels[v], f'{c}_g0_{v}') for v in SENS_VARIANTS] + [('oracle (upper bound)', f'{c}_go')]
        for label, n in rows:
            if n not in res:
                continue
            d = res[n]
            lines.append(f"| {label} | {d['success']} | {d['bird']} / {d['drone_drone']} | "
                         f"{('%.0f %%' % (100 * d['track_rate'])) if 'track_rate' in d else '-'} | {d.get('closest_median', float('nan')):.2f} | "
                         f"{('%.1f %%' % (100 * d['guard_active_frac'])) if 'guard_active_frac' in d else '-'} | "
                         f"{('%.3f (%d / %d)' % (d['p_raw'], d['lost'], d['gained'])) if 'p_raw' in d else '-'} | "
                         f"{('%.3f' % d['p_holm']) if 'p_holm' in d else '-'} | "
                         f"{('%d / %d' % d['vs_default']) if 'vs_default' in d else '-'} |")
    with open(os.path.join(args.out, 'tables_sens.md'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


def analyze_3ap(args):
    spec = cells_3ap(args.eps_single, args.eps_crowd)
    det_cells = cells_3a(args.eps_single, args.eps_crowd)
    names = set(spec) | {v[1] for v in spec.values()} | {v[4] for v in spec.values()} | {c for c in det_cells if c.endswith('_det') and c[:-4] in ('crowd_b2', 'crowd_b4') + tuple(SINGLE_GUARD)}
    data = {n: load(args.out, n) for n in names if os.path.exists(os.path.join(args.out, n, 'episodes.csv'))}
    res = {}
    for name in data:
        res[name] = cell_metrics(data[name][0])
    tests = []
    for name, (grp, none, _, _, nobird) in spec.items():
        if name not in data or none not in data:
            continue
        if name == none:                                              # the fair no-detector baseline itself
            res[name]['group'] = grp
            if nobird in data:
                res[name]['cost'] = cost_vs(args.out, name, nobird)
            continue
        a = {(r['seed'], r['episode']): r for r in data[none][0]}
        b = {(r['seed'], r['episode']): r for r in data[name][0]}
        assert a.keys() == b.keys() and all(a[k]['scenario_id'] == b[k]['scenario_id'] for k in a), f'{name}: not paired'
        lost = sum(1 for k in a if int(a[k]['success']) and not int(b[k]['success']))
        gained = sum(1 for k in a if int(b[k]['success']) and not int(a[k]['success']))
        res[name].update(lost=lost, gained=gained, p_raw=mcnemar_exact(lost, gained), none=none, group=grp)
        if nobird in data:
            res[name]['cost'] = cost_vs(args.out, name, nobird)
        tests.append(name)
    def nobird_of(nm):
        return 'crowd_b0' if nm.startswith(('crowd', 'fcrowd')) else 'single_obs_b0' if nm.startswith('single_obs') else 'single_noobs_b0'
    for nm in res:                                                    # the cost of birds without a guard too
        if 'cost' not in res[nm] and not nm.endswith('_b0') and nobird_of(nm) in data:
            res[nm]['cost'] = cost_vs(args.out, nm, nobird_of(nm))
    for det_name in [c for c in res if c.endswith('_det') and c in data and 'p_raw' not in res[c]]:
        none = det_name[:-4]
        if none in data:                                              # the stage-3a cell against its none cell
            a = {(r['seed'], r['episode']): r for r in data[none][0]}
            b = {(r['seed'], r['episode']): r for r in data[det_name][0]}
            lost = sum(1 for k in a if int(a[k]['success']) and not int(b[k]['success']))
            gained = sum(1 for k in a if int(b[k]['success']) and not int(a[k]['success']))
            res[det_name].update(lost=lost, gained=gained, p_raw=mcnemar_exact(lost, gained), none=none)
    for grp in ('worst', 'fair', 'single'):                           # Holm per group over the guard / oracle tests
        ts = [t for t in tests if res[t]['group'] == grp]
        for t, p in zip(ts, holm([res[t]['p_raw'] for t in ts])):
            res[t]['p_holm'] = p
    # single-drone cells pooled per condition
    pooled = {}
    for cond in ('none', 'det', 'g0', 'g1'):
        ks = []
        for c in SINGLE_GUARD:
            nm = c if cond == 'none' else f'{c}_{cond}'
            if nm in data:
                ks.append(nm)
        if len(ks) == len(SINGLE_GUARD):
            pooled[cond] = {k: sum(res[x][k] for x in ks) for k in ('n', 'success', 'bird', 'drone_drone', 'obstacle', 'timeout')}
            if cond in ('g0', 'g1'):
                pooled[cond]['guard_active_frac'] = float(np.mean([res[x]['guard_active_frac'] for x in ks]))
            if cond in ('none', 'g0', 'g1'):
                a = {}
                for x in ks:
                    a.update({(x, r['seed'], r['episode']): r for r in data[x][0]})
                pooled[cond]['_rows'] = a
    for cond in ('g0', 'g1'):
        if cond in pooled and 'none' in pooled:
            lost = gained = 0
            for c in SINGLE_GUARD:
                A = {(r['seed'], r['episode']): r for r in data[c][0]}
                Bc = {(r['seed'], r['episode']): r for r in data[f'{c}_{cond}'][0]}
                lost += sum(1 for k in A if int(A[k]['success']) and not int(Bc[k]['success']))
                gained += sum(1 for k in A if int(Bc[k]['success']) and not int(A[k]['success']))
            pooled[cond].update(lost=lost, gained=gained, p_raw=mcnemar_exact(lost, gained))
    for v in pooled.values():
        v.pop('_rows', None)
    with open(os.path.join(args.out, 'results_3ap.json'), 'w') as f:
        json.dump({'cells': res, 'single_pooled': pooled}, f, indent=1, default=str)

    def fmt(d):
        return f"{d['success']}/{d['n']} ({d['bird']} bird, {d['drone_drone']} d-d, {d['obstacle']} obst, {d['timeout']} t/o)"
    lines = []
    for grp, title, cells_list in (('worst', 'Worst-case crowd (birds timed to the drones)', ['crowd_b2', 'crowd_b4']),
                                   ('fair', 'Fair crowd (random straight crossings)', ['fcrowd_b2', 'fcrowd_b4'])):
        lines += ['', f'### {title}', '',
                  '| cell | none | detector + slots (3a) | guard, birds not in slots | guard, birds in slots | guard on the oracle (upper bound) |',
                  '|---|---|---|---|---|---|']
        for c in cells_list:
            def cell(nm, baseline=None):
                if nm not in res:
                    return '-'
                d = res[nm]
                extra = f" p {d['p_raw']:.2f} ({d['lost']}/{d['gained']})" if 'p_raw' in d and nm != c else ''
                return fmt(d) + extra
            lines.append(f"| {c} | {cell(c)} | {cell(c + '_det')} | {cell(c + '_g0')} | {cell(c + '_g1')} | {cell(c + '_go')} |")
        lines += ['', '| cell | condition | closest bird (median m) | encounters < 1.5 m | guard active (share of drone-steps / episodes) | extra path length / extra settle time vs no birds | Holm p vs none |', '|---|---|---|---|---|---|---|']
        for c in cells_list:
            for cond in ('', '_det', '_g0', '_g1', '_go'):
                nm = c + cond
                if nm not in res:
                    continue
                d = res[nm]
                ga = f"{100 * d['guard_active_frac']:.1f} % / {d['guard_episodes']}/{d['n']}" if 'guard_active_frac' in d else '-'
                co = d.get('cost')
                cs = f"x{co[0]:.2f} / {co[1]:+.1f} s" if co and co[2] else '-'
                ph = f"{d['p_holm']:.3f}" if 'p_holm' in d else '-'
                lines.append(f"| {c} | {cond[1:] or 'none'} | {d.get('closest_median', float('nan')):.2f} | {d.get('encounter', '-')}/{d['n']} | {ga} | {cs} | {ph} |")
    lines += ['', '### Single drone, pooled over the 5 cells (2 and 4 birds without and with a building, 2 birds at 3 m/s)', '',
              '| condition | successes | bird / drone-drone / obstacle crashes, timeouts | raw p vs none (lost / gained) | guard active |', '|---|---|---|---|---|']
    for cond, label in (('none', 'none (stage 2)'), ('det', 'detector + slots (3a)'), ('g0', 'guard, birds not in slots'), ('g1', 'guard, birds in slots')):
        if cond not in pooled:
            continue
        d = pooled[cond]
        lines.append(f"| {label} | {d['success']}/{d['n']} | {d['bird']} / {d['drone_drone']} / {d['obstacle']}, {d['timeout']} | "
                     f"{('%.2f (%d / %d)' % (d['p_raw'], d['lost'], d['gained'])) if 'p_raw' in d else '-'} | "
                     f"{('%.1f %%' % (100 * d['guard_active_frac'])) if 'guard_active_frac' in d else '-'} |")
    lines += ['', '### Single-drone cells', '', '| cell | none | detector + slots (3a) | guard, not in slots | guard, in slots | extra path / settle time (g0) |', '|---|---|---|---|---|---|']
    for c in SINGLE_GUARD:
        co = res.get(c + '_g0', {}).get('cost')
        lines.append(f"| {c} | {fmt(res[c]) if c in res else '-'} | {fmt(res[c + '_det']) if c + '_det' in res else '-'} | "
                     f"{fmt(res[c + '_g0']) if c + '_g0' in res else '-'} | {fmt(res[c + '_g1']) if c + '_g1' in res else '-'} | "
                     f"{('x%.2f / %+.1f s' % (co[0], co[1])) if co and co[2] else '-'} |")
    with open(os.path.join(args.out, 'tables_3ap.md'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


def analyze(args):
    if args.stage == '3a+':
        return analyze_3ap(args)
    if args.stage == 'sens':
        return analyze_sens(args)
    spec = all_cells(args)
    names = set(spec) | {v[1] for v in spec.values() if v[1]}
    data = {n: load(args.out, n) for n in names if os.path.exists(os.path.join(args.out, n, 'episodes.csv'))}
    if args.stage == '3a':                      # the stage-2 cells are part of the table: no detector
        spec = {**{b: cells(args.eps_single, args.eps_crowd)[b] for b in dict.fromkeys(v[1] for v in spec.values())}, **spec}
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
            if 'bird_tracked_steps' in rows[0]:
                trk = sum(int(r['bird_tracked_steps']) for r in rows)
                d['tracked'] = [trk, tin, trk / tin if tin else float('nan')]
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
    sfx = '_3a' if args.stage == '3a' else ''
    with open(os.path.join(args.out, f'results{sfx}.json'), 'w') as f:
        json.dump(res, f, indent=1, default=str)
    lines = ['| cell | n | success [95% CI] | bird crash | other crash | timeout | encounter <1.5 m [95% CI] | closest p10/med/p90 (m) | success given encounter | lost / gained vs no birds | p (Holm) | detection |',
             '|---|---|---|---|---|---|---|---|---|---|---|---|']
    for name, d in res.items():
        lo, hi = d['success_ci']
        enc = (f"{d['encounter']}/{d['n']} [{d['encounter_ci'][0]:.2f}-{d['encounter_ci'][1]:.2f}]" if 'encounter' in d else '-')
        q = d.get('min_bird_dist_q')
        sg = d.get('success_given_encounter')
        det = (f"tracked {100 * d['tracked'][2]:.0f}% / lidar {100 * d['detect'][2]:.1f}%" if 'tracked' in d else
               f"{d['detect'][0]}/{d['detect'][1]} = {100 * d['detect'][2]:.1f}% (0-2 m {100 * d['detect_bins']['0_2'][2]:.0f}%, 2-4 m {100 * d['detect_bins']['2_4'][2]:.0f}%, 4-6 m {100 * d['detect_bins']['4_6'][2]:.0f}%)"
               if 'detect' in d else '-')
        lines.append(f"| {name} | {d['n']} | {d['success']}/{d['n']} = {100 * d['success'] / d['n']:.0f}% [{100 * lo:.0f}-{100 * hi:.0f}] | "
                     f"{d['bird_crash']} | {d['other_crash']} | {d['timeout']} | {enc} | "
                     f"{'-' if q is None else f'{q[1]:.2f} / {q[2]:.2f} / {q[3]:.2f}'} | "
                     f"{'-' if sg is None else f'{sg[0]}/{sg[1]}'} | "
                     f"{'-' if 'p_raw' not in d else str(d['paired_lost']) + ' / ' + str(d['paired_gained'])} | "
                     f"{'-' if 'p_holm' not in d else f'{d['p_holm']:.3f}'} | {det} |")
    with open(os.path.join(args.out, f'tables{sfx}.md'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['run', 'analyze'])
    ap.add_argument('--checkpoint')
    ap.add_argument('--out', default=os.path.join('eval_results', 'birds'))
    ap.add_argument('--eps-single', type=int, default=10, help='episodes per seed, single-drone cells (5 seeds)')
    ap.add_argument('--eps-crowd', type=int, default=10, help='episodes per seed, crowd cells (5 seeds)')
    ap.add_argument('--stage', choices=['2', '3a', '3a+', 'sens'], default='2', help='2: LiDAR only (default); 3a: onboard bird detector; 3a+: bird guard and fair crowd')
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
