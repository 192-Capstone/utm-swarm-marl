"""Results tables, paired tests and figures from eval_results, in one command.

Reads every model's per-test summary.json (and episodes.csv for the paired tests) from either
layout:
    suite runner : <root>/<model>/<standard|crowd|heldout>/<test folder>/        (scripts/run_suite.py)
    flat         : <root>/<model>/<test folder>/                                 (scripts/eval_500.sh; = standard)

Examples (from the repo root):
    python scripts/results_report.py --root eval_results/suite17 \
        --models teacher_r7 teacher_lag_cw ww_s0_lagnoise --labels "teacher r7" "teacher cw" "ww_s0 lag+noise"
    python scripts/results_report.py --root eval_results/stage2_final_500 --models ww_s0 BCA \
        --compare ww_s0 BCA --out docs/results/ww_vs_bca.md --fig docs/figures/ww_vs_bca.png
    python scripts/results_report.py --root eval_results/stage2_final_500 --recipe ww=ww_s* w=w_s* --models ww_s0

Output (markdown, to --out or stdout):
  averages per group (success, obstacle crashes, drone-drone crashes, timeouts) with 95% CIs;
  per-test success and crash tables (95% Wilson CIs, best learned model in bold);
  time to settle at the goal; paired comparisons (exact McNemar per test, Holm-corrected, plus
  the pooled group); seed mean +- std and min-max per recipe. --csv writes every row as CSV,
  --fig a per-test success chart.
"""
import argparse
import csv
import fnmatch
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval_stats import wilson, mcnemar_exact, holm, spread  # noqa: E402

GROUPS = ['standard', 'crowd', 'heldout']
GROUP_NAMES = {'standard': 'standard 9', 'crowd': 'crowd 4 (8 drones)', 'heldout': 'held-out 4'}
ORDER = ['pillar', 'building', 'wide_building', 'long_wall', 'random', 'headon', 'crossing', 'swap',
         'flythrough', 'l_shape', 'gap', 'cluster', 'slalom']
NICE = {'wide_building': 'wide building', 'long_wall': 'long wall', 'headon': 'head-on',
        'flythrough': 'fly-through', 'l_shape': 'L-shape'}
TEACHERS = ('repulsion', 'lidar_teacher', 'pcontroller')


def test_name(folder):
    """'mappo_wide_building_dtrain_n1_...' -> 'wide_building'."""
    base = os.path.basename(folder.rstrip('/\\'))
    stem = base.split('_dtrain')[0]
    for t in sorted(ORDER, key=len, reverse=True):
        if stem.endswith('_' + t) or stem == t:
            return t
    return stem


def load_model(root, model):
    """{group: {test: {'summary': dict, 'episodes': [rows], 'policy': str}}}"""
    out = {}
    base = os.path.join(root, model)
    if any(os.path.isdir(os.path.join(base, g)) for g in GROUPS):        # suite-runner layout
        layouts = [(g, os.path.join(base, g, '*', 'summary.json')) for g in GROUPS]
    else:                                                                # flat layout = standard tests
        layouts = [('standard', os.path.join(base, '*', 'summary.json'))]
    for group, pattern in layouts:
        for f in glob.glob(pattern):
            folder = os.path.dirname(f)
            s = json.load(open(f, encoding='utf-8'))
            ep_file = os.path.join(folder, 'episodes.csv')
            eps = list(csv.DictReader(open(ep_file, encoding='utf-8'))) if os.path.exists(ep_file) else []
            policy = os.path.basename(folder).split('_')[0]
            out.setdefault(group, {})[test_name(folder)] = {'summary': s, 'episodes': eps, 'policy': policy}
    return out


def crash_rate(s):
    return s.get('any_crash_rate', s.get('collision_rate', 0.0))


def rates(s):
    """success, obstacle crash, drone crash, timeout as fractions, plus n."""
    n = s['episodes']
    obst = s.get('obstacle_collision_rate', 0.0)
    drone = s.get('collision_rate', 0.0)
    return {'n': n, 'success': s['success_rate'], 'obstacle': obst, 'drone': drone, 'timeout': s['timeout_rate'],
            'crash': crash_rate(s)}


def pooled(tests, key):
    n = sum(t['summary']['episodes'] for t in tests.values())
    k = sum(round(rates(t['summary'])[key] * t['summary']['episodes']) for t in tests.values())
    return k, n


def fmt_ci(k, n, bold=False):
    if n == 0:
        return '-'
    lo, hi = wilson(k, n)
    s = f"{100 * k / n:.1f}% [{100 * lo:.0f}–{100 * hi:.0f}]"
    return f"**{s}**" if bold else s


def paired(a_eps, b_eps):
    """Match episodes on (seed, episode); returns (b, c, n_matched, scenario mismatches)."""
    key = lambda r: (r['seed'], r['episode'])
    bmap = {key(r): r for r in b_eps}
    only_a = only_b = n = mismatch = 0
    for r in a_eps:
        o = bmap.get(key(r))
        if o is None:
            continue
        n += 1
        if r.get('scenario_id') and o.get('scenario_id') and r['scenario_id'] != o['scenario_id']:
            mismatch += 1
        sa, sb = r['success'] == '1', o['success'] == '1'
        only_a += sa and not sb
        only_b += sb and not sa
    return only_a, only_b, n, mismatch


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', nargs='+', required=True, help='one or more results roots (searched in order)')
    ap.add_argument('--models', nargs='+', default=[], help='model folder names, in table order')
    ap.add_argument('--labels', nargs='+', default=None, help='column labels for --models')
    ap.add_argument('--compare', nargs=2, action='append', default=[], metavar=('A', 'B'),
                    help='paired McNemar comparison A vs B (repeatable)')
    ap.add_argument('--recipe', nargs='+', default=[], help='name=glob, e.g. ww=ww_s* : seed spread per recipe')
    ap.add_argument('--title', default='Results')
    ap.add_argument('--out', default=None, help='markdown file (default: print)')
    ap.add_argument('--csv', default=None, help='CSV with one row per model x test')
    ap.add_argument('--fig', default=None, help='PNG: per-test success with 95%% CIs')
    args = ap.parse_args()

    def find(model):
        for r in args.root:
            if os.path.isdir(os.path.join(r, model)):
                return load_model(r, model)
        sys.exit(f"model '{model}' not found under {args.root}")

    names = list(args.models)
    for a, b in args.compare:
        names += [m for m in (a, b) if m not in names]
    recipes = {}
    for spec in args.recipe:
        rname, pattern = spec.split('=', 1)
        found = sorted({d for r in args.root for d in os.listdir(r) if fnmatch.fnmatch(d, pattern)
                        and os.path.isdir(os.path.join(r, d))})
        recipes[rname] = found
    data = {m: find(m) for m in names}
    for ms in recipes.values():
        for m in ms:
            data.setdefault(m, find(m))
    labels = dict(zip(args.models, args.labels or args.models))
    for m in names:
        labels.setdefault(m, m)
    L = [f"# {args.title}", '',
         f"Generated by `scripts/results_report.py` from {', '.join('`' + r + '`' for r in args.root)}. "
         "Success = every drone settled at its goal; CIs are 95% Wilson intervals.", '']

    groups = [g for g in GROUPS if any(g in data[m] for m in args.models)]
    if args.models:
        L += ['## Averages', '', '| group | measure | ' + ' | '.join(labels[m] for m in args.models) + ' |',
              '|---|---|' + '---|' * len(args.models)]
        for g in groups:
            for key, title in (('success', 'success'), ('obstacle', 'obstacle crashes'),
                               ('drone', 'drone-drone crashes'), ('timeout', 'timeouts')):
                if g == 'crowd' and key == 'obstacle':
                    continue
                cells = []
                for m in args.models:
                    tests = data[m].get(g, {})
                    k, n = pooled(tests, key) if tests else (0, 0)
                    cells.append(f"{100 * k / n:.1f}% ({len(tests)} tests)" if n else '-')
                L.append(f"| {GROUP_NAMES[g]} | {title} | " + ' | '.join(cells) + ' |')
        L.append('')

        for key, title in (('success', 'Success rate'), ('crash', 'Crashes (any)'), ('timeout', 'Timeouts')):
            L += [f"## {title} (95% confidence interval)", '']
            if key == 'success':
                L += ['Best learned model per test in bold (rule teachers are references, not competitors).', '']
            L += ['| test | ' + ' | '.join(labels[m] for m in args.models) + ' |', '|---|' + '---|' * len(args.models)]
            for g in groups:
                tests = [t for t in ORDER if any(t in data[m].get(g, {}) for m in args.models)]
                for t in tests:
                    vals = {}
                    for m in args.models:
                        e = data[m].get(g, {}).get(t)
                        if e:
                            vals[m] = (round(rates(e['summary'])[key] * e['summary']['episodes']), e['summary']['episodes'],
                                       e['policy'])
                    learned = [m for m, v in vals.items() if v[2] not in TEACHERS]
                    best = max(learned, key=lambda m: vals[m][0] / vals[m][1]) if (key == 'success' and len(learned) > 1) else None
                    row = [fmt_ci(vals[m][0], vals[m][1], bold=(m == best)) if m in vals else '-' for m in args.models]
                    L.append(f"| {GROUP_NAMES[g].split(' ')[0]}: {NICE.get(t, t)} | " + ' | '.join(row) + ' |')
            L.append('')

        L += ['## Time to settle at the goal (mean seconds, successful flights)', '',
              '| test | ' + ' | '.join(labels[m] for m in args.models) + ' |', '|---|' + '---|' * len(args.models)]
        for g in groups:
            for t in [t for t in ORDER if any(t in data[m].get(g, {}) for m in args.models)]:
                row = []
                for m in args.models:
                    e = data[m].get(g, {}).get(t)
                    st = (e['summary'].get('settle_time_s') or {}) if e else {}
                    row.append(f"{st['mean']:.1f}" if st.get('mean') is not None else '-')
                L.append(f"| {GROUP_NAMES[g].split(' ')[0]}: {NICE.get(t, t)} | " + ' | '.join(row) + ' |')
        L.append('')

    if args.compare:
        L += ['## Paired comparisons (exact McNemar test)', '',
              'Episodes are matched on (seed, episode), so both policies flew the same scenes. b = only the first '
              'policy succeeded, c = only the second. p is Holm-corrected across the tests of each comparison; '
              'the pooled row tests all matched episodes of a group together; – = the two never differed.', '']
        for a, b in args.compare:
            L += [f"### {labels[a]} vs {labels[b]}", '',
                  '| test | n | success A | success B | difference | b | c | p (Holm) |', '|---|---|---|---|---|---|---|---|']
            rows, ps = [], []
            for g in GROUPS:
                pool = [0, 0, 0, 0, 0]
                for t in ORDER:
                    ea, eb = data[a].get(g, {}).get(t), data[b].get(g, {}).get(t)
                    if not (ea and eb and ea['episodes'] and eb['episodes']):
                        continue
                    bb, cc, n, mm = paired(ea['episodes'], eb['episodes'])
                    if n == 0:
                        continue
                    sa = sum(r['success'] == '1' for r in ea['episodes'] if (r['seed'], r['episode']) in
                             {(o['seed'], o['episode']) for o in eb['episodes']})
                    sb = sa - bb + cc
                    rows.append((g, t, n, sa, sb, bb, cc, mm))
                    ps.append(mcnemar_exact(bb, cc) if bb + cc else None)   # no discordant pairs: no information
                    pool = [pool[0] + n, pool[1] + sa, pool[2] + sb, pool[3] + bb, pool[4] + cc]
                if pool[0]:
                    rows.append((g, 'POOLED', *pool, 0))
                    ps.append(None)
            adj = iter(holm([p for p in ps if p is not None]))
            for row, p in zip(rows, ps):
                g, t, n, sa, sb, bb, cc, mm = row
                pv = next(adj) if p is not None else mcnemar_exact(bb, cc)
                star = ' *' if pv < 0.05 else ''
                pv_txt = f"{pv:.3g}{star}" if (bb + cc) else '–'
                name = f"**{GROUP_NAMES[g]}, all**" if t == 'POOLED' else f"{GROUP_NAMES[g].split(' ')[0]}: {NICE.get(t, t)}"
                warn = f" (scene mismatch in {mm})" if mm else ''
                L.append(f"| {name}{warn} | {n} | {100 * sa / n:.1f}% | {100 * sb / n:.1f}% | {100 * (sa - sb) / n:+.1f} | "
                         f"{bb} | {cc} | {pv_txt} |")
            L.append('')

    if recipes:
        L += ['## Seed spread per recipe', '', 'Success over each group, one value per training seed.', '',
              '| recipe | seeds | group | mean | std | min | max |', '|---|---|---|---|---|---|---|']
        for rname, ms in recipes.items():
            for g in GROUPS:
                vals = []
                for m in ms:
                    tests = data[m].get(g, {})
                    if tests:
                        k, n = pooled(tests, 'success')
                        vals.append(100 * k / n)
                if vals:
                    mean, sd, lo, hi = spread(vals)
                    L.append(f"| {rname} | {len(vals)} ({', '.join(ms)}) | {GROUP_NAMES[g]} | {mean:.1f}% | {sd:.1f} | {lo:.1f}% | {hi:.1f}% |")
        L.append('')

    text = '\n'.join(L)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        open(args.out, 'w', encoding='utf-8', newline='\n').write(text)
        print('wrote', args.out)
    else:
        print(text)

    if args.csv:
        with open(args.csv, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['model', 'group', 'test', 'n', 'success', 'success_lo', 'success_hi', 'obstacle_crash',
                        'drone_crash', 'timeout', 'settle_s'])
            for m in data:
                for g, tests in data[m].items():
                    for t, e in tests.items():
                        r = rates(e['summary'])
                        lo, hi = wilson(round(r['success'] * r['n']), r['n'])
                        w.writerow([m, g, t, r['n'], r['success'], lo, hi, r['obstacle'], r['drone'], r['timeout'],
                                    (e['summary'].get('settle_time_s') or {}).get('mean')])
        print('wrote', args.csv)

    if args.fig and args.models:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        cols = [(g, t) for g in groups for t in ORDER if any(t in data[m].get(g, {}) for m in args.models)]
        fig, ax = plt.subplots(figsize=(min(14, 3 + 0.6 * len(cols)), 4.6))
        width = 0.8 / len(args.models)
        for i, m in enumerate(args.models):
            xs, ys, err = [], [], [[], []]
            for j, (g, t) in enumerate(cols):
                e = data[m].get(g, {}).get(t)
                if not e:
                    continue
                r = rates(e['summary'])
                lo, hi = wilson(round(r['success'] * r['n']), r['n'])
                xs.append(j + (i - (len(args.models) - 1) / 2) * width)
                ys.append(100 * r['success'])
                err[0].append(max(0.0, 100 * (r['success'] - lo))); err[1].append(max(0.0, 100 * (hi - r['success'])))
            ax.bar(xs, ys, width, yerr=err, capsize=2, label=labels[m])
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels([NICE.get(t, t) + ('*' if g == 'crowd' else '') for g, t in cols], rotation=35, ha='right')
        ax.set_ylabel('success (%)')
        ax.set_ylim(0, 105)
        ax.grid(axis='y', alpha=0.3)
        ax.legend(fontsize=8, ncol=min(4, len(args.models)), loc='lower center', bbox_to_anchor=(0.5, 1.0), frameon=False)
        ax.set_title(args.title + ('   (* = 8-drone crowd test)' if any(g == 'crowd' for g, _ in cols) else ''), fontsize=10, pad=26)
        fig.tight_layout()
        os.makedirs(os.path.dirname(os.path.abspath(args.fig)), exist_ok=True)
        fig.savefig(args.fig, dpi=160)
        print('wrote', args.fig)


if __name__ == '__main__':
    main()
