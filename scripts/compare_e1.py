"""
compare_e1.py
-------------
E1 on paper (no flying): how much each of its tools helps, whether the order requests are filed
in matters, and how approval and delay grow with traffic (capacity). Evaluation only.

    strategies   none        approve only plans that are already conflict-free
                 delay       wait (up to max_delay), one layer
                 layer       change layer, no waiting
                 both        E1 as designed: wait and/or change layer
    orders       filed       the order the requests were made (random wrt departure time)
                 departure   earliest departure first
                 shortest    shortest flight first
                 longest     longest flight first

Traffic: random routable requests (scripts/preview_strategic.random_requests: pads >= 1.5 m
from buildings, >= 3 m apart) wanting to leave within `window` seconds.

    python scripts/compare_e1.py --maps city osm-crop --flights 10 20 40 60 80 --seeds 20 21 22 23 24

Seeds 10-19 are refused (reserved for test runs).
"""

import argparse
import dataclasses
import json
import os
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from preview_route import load_map                                 # noqa: E402
from preview_strategic import random_requests                      # noqa: E402
from utm.strategic_planner import Settings, StrategicPlanner       # noqa: E402

STRATEGIES = {'none': dict(n_layers=1, max_delay=0.0), 'delay': dict(n_layers=1),
              'layer': dict(max_delay=0.0), 'both': {}}
ORDERS = ('filed', 'departure', 'shortest', 'longest')


def ordered(reqs, order, sky):
    if order == 'filed':
        return list(reqs)
    if order == 'departure':
        return sorted(reqs, key=lambda r: r.t_earliest)
    length = {r.id: sky.profile(sky.route(r.start, r.goal, 0), 0)[1] for r in reqs}
    return sorted(reqs, key=lambda r: length[r.id], reverse=(order == 'longest'))


def run(layout, bounds, reqs, strategy, order):
    sky = StrategicPlanner(layout, bounds, dataclasses.replace(Settings(), **STRATEGIES[strategy]))
    for fp in ordered(reqs, order, sky):
        sky.submit(fp)
    return sky.summary()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--maps', nargs='+', default=['city', 'osm-crop'])
    ap.add_argument('--level', default='medium')
    ap.add_argument('--flights', type=int, nargs='+', default=[10, 20, 40, 60, 80])
    ap.add_argument('--window', type=float, default=60.0)
    ap.add_argument('--seeds', type=int, nargs='+', default=[20, 21, 22, 23, 24])
    ap.add_argument('--out', default=os.path.join('eval_results', 'e1_compare', 'compare.json'))
    args = ap.parse_args()
    if any(10 <= s <= 19 for s in args.seeds):
        ap.error('seeds 10-19 are reserved for test runs')

    rows = []
    for m in args.maps:
        for seed in args.seeds:
            layout, bounds = load_map(argparse.Namespace(map=m, level=args.level, seed=seed))
            base = StrategicPlanner(layout, bounds)
            for n in args.flights:
                reqs = random_requests(np.random.RandomState(seed), base, n, window=args.window)
                for strategy in STRATEGIES:
                    for order in (ORDERS if strategy == 'both' else ('filed',)):
                        s = run(layout, bounds, reqs, strategy, order)
                        rows.append({'map': m, 'seed': seed, 'flights': n, 'requested': len(reqs),
                                     'strategy': strategy, 'order': order, **s})
                print(f'{m} seed {seed} n {n}: ' + ', '.join(
                    f"{r['strategy']}/{r['order']} {r['approved_pct']:.0f}% {r['delay_mean_s']:.1f}s"
                    for r in rows if r['map'] == m and r['seed'] == seed and r['flights'] == n), flush=True)

    print('\nmean over seeds: approved % / mean delay s / p95 delay s')
    for m in args.maps:
        for n in args.flights:
            for strategy in STRATEGIES:
                for order in (ORDERS if strategy == 'both' else ('filed',)):
                    sel = [r for r in rows if (r['map'], r['flights'], r['strategy'], r['order']) == (m, n, strategy, order)]
                    req = np.mean([r['requested'] for r in sel])
                    print(f"{m:8s} n {n:3d} (got {req:5.1f})  {strategy:5s} {order:9s} "
                          f"{np.mean([r['approved_pct'] for r in sel]):5.1f}%  "
                          f"{np.mean([r['delay_mean_s'] for r in sel]):5.1f} s  "
                          f"{np.mean([r['delay_p95_s'] for r in sel]):5.1f} s")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump({'args': vars(args), 'rows': rows}, f, indent=1, default=int)
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
