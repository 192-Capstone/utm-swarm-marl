"""PyBullet vs Webots (Level 1) table from eval_results/<root>/<model>/ (repo root).
Usage: python scripts/webots_l1_table.py [eval_results/webots_l1/ww_s0]"""
import glob
import json
import re
import sys

root = sys.argv[1] if len(sys.argv) > 1 else 'eval_results/webots_l1/ww_s0'
TESTS = ['pillar', 'building', 'wide_building', 'long_wall', 'random', 'headon', 'crossing', 'swap', 'flythrough']
rows = {}
for f in glob.glob(f'{root}/*/summary.json'):
    name = f.split('/')[-2]
    t = re.match(r'[a-z_]+?_(' + '|'.join(TESTS) + r')_d', name).group(1)
    rows[(t, 'webots' if name.endswith('_webots') else 'pybullet')] = json.load(open(f))


def cell(r, k):
    return f"{100 * r[k]:.1f}"


print('| test | PyBullet success | Webots success | diff | PyBullet crashes | Webots crashes | PyBullet timeouts | Webots timeouts | mean steps PB / WB |')
print('|---|---|---|---|---|---|---|---|---|')
tot = {'pybullet': [0, 0], 'webots': [0, 0]}
for t in TESTS:
    pb, wb = rows.get((t, 'pybullet')), rows.get((t, 'webots'))
    if not pb or not wb:
        print(f'| {t} | - | - | - | - | - | - | - | - |'); continue
    for sim, r in (('pybullet', pb), ('webots', wb)):
        tot[sim][0] += r['success_rate'] * r['episodes']; tot[sim][1] += r['episodes']
    print(f"| {t} | {cell(pb, 'success_rate')} | {cell(wb, 'success_rate')} | {100 * (wb['success_rate'] - pb['success_rate']):+.1f} | "
          f"{cell(pb, 'any_crash_rate')} | {cell(wb, 'any_crash_rate')} | {cell(pb, 'timeout_rate')} | {cell(wb, 'timeout_rate')} | "
          f"{pb['mean_episode_length_steps']:.0f} / {wb['mean_episode_length_steps']:.0f} |")
if tot['pybullet'][1] and tot['webots'][1]:
    a, b = 100 * tot['pybullet'][0] / tot['pybullet'][1], 100 * tot['webots'][0] / tot['webots'][1]
    print(f'| **overall** | **{a:.1f}** | **{b:.1f}** | {b - a:+.1f} | | | | | |')
