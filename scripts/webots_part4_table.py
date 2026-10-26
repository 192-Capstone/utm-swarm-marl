"""Part 4 tables from eval_results/webots_l2_handover/<variant>_<test>.json (repo root)."""
import glob
import json
import math
import sys

import numpy as np

ROOT = sys.argv[1] if len(sys.argv) > 1 else 'eval_results/webots_l2_handover'
V = ['V1', 'V2', 'V3', 'V4']
NAMES = {'V1': 'R1 policy-only', 'V2': 'R1 + hand-over (outer-loop hold)', 'V3': 'r7v3ft policy-only',
         'V4': 'rule teacher (round-7)'}
T = ['pillar', 'building', 'wide_building', 'long_wall']
D = {}
for f in glob.glob(f'{ROOT}/*.json'):
    r = json.load(open(f))
    D[(r['variant'], r['test'])] = r['episodes']


def pct(x):
    return f'{100 * np.mean(x):.1f}' if len(x) else '-'


def exact_mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, p)


print('### Outcomes (validation seeds 20-24 x 100 episodes = 500 per test)')
print('| variant | test | strict success | success @0.20 m/s | ended within 0.5 m | crashes | timeouts: inside goal, moving | 0.5-1 m short | stuck > 1 m |')
print('|---|---|---|---|---|---|---|---|---|')
for v in V:
    rows_all = []
    for t in T + ['ALL']:
        e = [x for tt in T for x in D.get((v, tt), [])] if t == 'ALL' else D.get((v, t), [])
        if not e:
            continue
        n = len(e)
        to = [x['timeout_cause'] for x in e if x['outcome'] == 'timeout']
        label = f'**{t} ({n})**' if t == 'ALL' else f'{t} ({n})'
        print(f"| {v} {NAMES[v]} | {label} | {pct([x['success_strict'] for x in e])} | {pct([x['success_020'] for x in e])} | "
              f"{pct([x['ended_within_0.5'] for x in e])} | {pct([x['outcome'] == 'collision' for x in e])} | "
              f"{100 * to.count('inside_moving') / n:.1f} | {100 * to.count('short_0.5_1') / n:.1f} | {100 * to.count('stuck_gt1') / n:.1f} |")

print('\n### At the goal')
print('| variant | test | median speed inside goal sphere (m/s) | longest still stretch, median (steps < 0.15 m/s) | time to settle after first entry, median (s) | flights that entered the goal sphere |')
print('|---|---|---|---|---|---|')
for v in V:
    for t in T + ['ALL']:
        e = [x for tt in T for x in D.get((v, tt), [])] if t == 'ALL' else D.get((v, t), [])
        if not e:
            continue
        ent = [x for x in e if x['first_entry_step'] is not None]
        ms = [x['median_speed_in_goal'] for x in ent if x['median_speed_in_goal'] is not None]
        ls = [x['longest_still_in_goal'] for x in ent]
        ts = [x['time_to_settle_s'] for x in ent if x['time_to_settle_s'] is not None]
        label = f'**{t}**' if t == 'ALL' else t
        print(f"| {v} | {label} | {np.median(ms):.3f} | {np.median(ls):.0f} | {np.median(ts) if ts else float('nan'):.2f} | {len(ent)}/{len(e)} |")

print('\n### V1 vs V2 on matched episodes (exact McNemar, strict rule)')
print('| test | matched pairs | both succeed | only V1 | only V2 | neither | V1 | V2 | exact p |')
print('|---|---|---|---|---|---|---|---|---|')
tot = [0, 0, 0, 0, 0]
for t in T + ['ALL']:
    pairs = []
    for tt in (T if t == 'ALL' else [t]):
        a = {(x['seed'], x['episode']): x for x in D.get(('V1', tt), [])}
        b = {(x['seed'], x['episode']): x for x in D.get(('V2', tt), [])}
        for k in a:
            if k in b and a[k]['scenario_id'] == b[k]['scenario_id']:
                pairs.append((a[k]['success_strict'], b[k]['success_strict']))
    if not pairs:
        continue
    both = sum(1 for x, y in pairs if x and y); only1 = sum(1 for x, y in pairs if x and not y)
    only2 = sum(1 for x, y in pairs if y and not x); nei = sum(1 for x, y in pairs if not x and not y)
    n = len(pairs)
    label = f'**{t}**' if t == 'ALL' else t
    print(f"| {label} | {n} | {both} | {only1} | {only2} | {nei} | {100 * (both + only1) / n:.1f}% | "
          f"{100 * (both + only2) / n:.1f}% | {exact_mcnemar(only1, only2):.2g} |")
mism = 0
for tt in T:
    a = {(x['seed'], x['episode']): x['scenario_id'] for x in D.get(('V1', tt), [])}
    for v in ('V2', 'V3', 'V4'):
        b = {(x['seed'], x['episode']): x['scenario_id'] for x in D.get((v, tt), [])}
        mism += sum(1 for k in a if k in b and a[k] != b[k])
print(f'\nScenario check: episodes whose scenario differs between V1 and another variant: {mism}')
mx = {v: max((x['max_speed'] for tt in T for x in D.get((v, tt), [])), default=float('nan')) for v in V}
print('Max true speed reached per variant (m/s): ' + ', '.join(f'{v} {mx[v]:.2f}' for v in V))
