"""Run the full 17-test suite for one policy and print one table.

    standard 9 : pillar building wide_building long_wall (1 drone), random headon crossing swap flythrough (3)
    crowd 4    : headon crossing swap flythrough, 8 drones, no obstacles
    held-out 4 : l_shape gap cluster slalom (1 drone)

Examples (from the repo root):
    python scripts/run_suite.py --name teacher_r7 --policy repulsion --config configs/config_stage2_round7_teacher.yaml
    python scripts/run_suite.py --name ww_s0_lag --checkpoint checkpoints/stage2_imitation_ww_s0/bc_policy.pth \
        --config configs/config_stage2_round7_teacher.yaml --seeds 10 11 12 13 14 --episodes 100

Every eval_suite.py job runs at below-normal priority (Windows priority class / nice 10), at most
--jobs at a time. Tests that already have a summary.json are skipped, so a stopped run resumes.
Results: <root>/<name>/<group>/<test folder>/summary.json, logs next to them.
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import time

STANDARD = ['pillar', 'building', 'wide_building', 'long_wall', 'random', 'headon', 'crossing', 'swap', 'flythrough']
CROWD = ['headon', 'crossing', 'swap', 'flythrough']
HELDOUT = ['l_shape', 'gap', 'cluster', 'slalom']


def jobs_for(groups):
    out = []
    if 'standard' in groups:
        out += [('standard', t, []) for t in STANDARD]
    if 'crowd' in groups:
        out += [('crowd', t, ['--n-agents', '8', '--no-obstacles']) for t in CROWD]
    if 'heldout' in groups:
        out += [('heldout', t, []) for t in HELDOUT]
    return out


def low_priority_kwargs():
    if os.name == 'nt':
        return {'creationflags': subprocess.BELOW_NORMAL_PRIORITY_CLASS}
    return {'preexec_fn': lambda: os.nice(10)}


def crash(s):
    """Any crash; obstacle-free tests only report drone-drone collisions."""
    return s.get('any_crash_rate', s.get('collision_rate', 0.0))


def summarise(root, name):
    rows = []
    for group in ('standard', 'crowd', 'heldout'):
        for f in sorted(glob.glob(os.path.join(root, name, group, '*', 'summary.json'))):
            s = json.load(open(f))
            test = os.path.basename(os.path.dirname(f))
            st = s.get('settle_time_s') or {}
            rows.append((group, test, s))
    if not rows:
        return
    print(f"\n{name}")
    print(f"{'group':9s} {'test':44s} {'n':>4s} {'success':>8s} {'crash':>6s} {'obst':>5s} {'timeout':>8s} {'settle s':>9s} {'end v':>6s}")
    for group in ('standard', 'crowd', 'heldout'):
        g = [(t, s) for gg, t, s in rows if gg == group]
        if not g:
            continue
        for t, s in g:
            st = s.get('settle_time_s') or {}
            print(f"{group:9s} {t[:44]:44s} {s['episodes']:4d} {100 * s['success_rate']:7.1f}% "
                  f"{100 * crash(s):5.1f}% {100 * s.get('obstacle_collision_rate', 0.0):4.1f}% "
                  f"{100 * s['timeout_rate']:7.1f}% {st.get('mean', float('nan')):9.2f} {s.get('final_speed_mean', float('nan')):6.3f}")
        n = sum(s['episodes'] for _, s in g)
        sr = sum(s['success_rate'] * s['episodes'] for _, s in g) / n
        cr = sum(crash(s) * s['episodes'] for _, s in g) / n
        to = sum(s['timeout_rate'] * s['episodes'] for _, s in g) / n
        print(f"{group:9s} {'== ' + group + f' ({len(g)} tests)':44s} {n:4d} {100 * sr:7.1f}% {100 * cr:5.1f}%       {100 * to:7.1f}%")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--name', required=True)
    ap.add_argument('--config', required=True)
    ap.add_argument('--policy', default='mappo',
                    choices=['mappo', 'repulsion', 'lidar_teacher', 'pcontroller', 'orca', 'orca_map'])
    ap.add_argument('--checkpoint', default=None)
    ap.add_argument('--groups', nargs='+', default=['standard', 'crowd', 'heldout'])
    ap.add_argument('--tests', nargs='+', default=None, help='only these tests (e.g. pillar building)')
    ap.add_argument('--seeds', type=int, nargs='+', default=[10, 11, 12, 13, 14])
    ap.add_argument('--episodes', type=int, default=20, help='episodes PER seed (default 20 -> 100 flights per test)')
    ap.add_argument('--jobs', type=int, default=2, help='evaluations running at the same time')
    ap.add_argument('--root', default='eval_results/suite17')
    ap.add_argument('--summary-only', action='store_true')
    args = ap.parse_args()

    if not args.summary_only:
        todo = []
        for group, test, extra in jobs_for(args.groups):
            if args.tests and test not in args.tests:
                continue
            out = os.path.join(args.root, args.name, group)
            if glob.glob(os.path.join(out, f'*_{test}_*', 'summary.json')):
                continue
            cmd = [sys.executable, 'eval_suite.py', '--config', args.config, '--policy', args.policy,
                   '--scenario', test, '--seeds', *map(str, args.seeds), '--episodes', str(args.episodes),
                   '--collision-physics', 'off', '--results-root', out, *extra]
            if args.checkpoint:
                cmd += ['--checkpoint', args.checkpoint]
            todo.append((group, test, cmd, out))
        print(f"{time.strftime('%H:%M:%S')}  {args.name}: {len(todo)} tests to run, {args.jobs} at a time, low priority")
        running = []
        env = dict(os.environ, PYTHONIOENCODING='utf-8')
        while todo or running:
            while todo and len(running) < args.jobs:
                group, test, cmd, out = todo.pop(0)
                os.makedirs(out, exist_ok=True)
                log = open(os.path.join(out, f'{test}.log'), 'w', encoding='utf-8')
                running.append((group, test, subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env,
                                                              **low_priority_kwargs()), log, time.time()))
            time.sleep(1)
            for item in list(running):
                group, test, p, log, t0 = item
                if p.poll() is not None:
                    log.close()
                    running.remove(item)
                    status = 'done  ' if p.returncode == 0 else f'FAILED ({p.returncode})'
                    print(f"{time.strftime('%H:%M:%S')}  {status} {group:8s} {test:14s} {time.time() - t0:6.0f} s", flush=True)
    summarise(args.root, args.name)


if __name__ == '__main__':
    main()
