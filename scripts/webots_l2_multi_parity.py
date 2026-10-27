"""
webots_l2_multi_parity.py
-------------------------
C1: is the new multi-drone adapter with ONE drone (envs/webots_l2_multi.py, the drone flown by its
own cf_velocity controller) the same as Aman's single-drone Level 2 (envs/webots_l2.py,
WebotsL2Adapter, official Crazyflie)?

The shipped policy flies the same 40 pillar episodes in both: seeds 20 and 21, 20 episodes each,
drawn exactly as eval_suite.py draws an obstacle test (per seed: np.random.seed, torch.manual_seed,
a RandomState for setup_obstacle_test, episodes in order), deterministic policy, config
config_stage2_round7_eval_paired.yaml (per-episode noise seed, so both see the same noise).
Each adapter runs in its OWN subprocess (one Webots connection per Python process), one after
the other.

Per episode: outcome (SwarmEnv's reason: success / collision / timeout), steps, the true
track, and max tilt / true speed (lost-control rule of eval_suite.lost_control).
Episodes are paired by (seed, episode); the scenario hash is a cross-check only.

Pass rule (agreed): the same outcome in at least 38 of 40 episodes.
Reported for information: per-episode max |track difference| over the common steps (median and
max over episodes), lost-control counts, scenario-hash mismatches, wall time of each run.
A command reaches the motors one basic step (4 ms) later in the new adapter, so the tracks are
not expected to be identical.

Usage (repo root, DISPLAY explicit, low priority, nothing else heavy running):
  DISPLAY=:2 OMP_NUM_THREADS=4 nice -n 19 python scripts/webots_l2_multi_parity.py \
      --checkpoint checkpoints/stage2_r1_crowd_rl_r7v3ft_s0_lab/checkpoint_final.pth
Writes logs/webots/c1_parity/{old,new}.json and report.json; Webots logs next to them.
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

CONFIG = 'configs/config_stage2_round7_eval_paired.yaml'
OUT = os.path.join(ROOT, 'logs', 'webots', 'c1_parity')
SEEDS = (20, 21)
EPISODES = 20
TEST = 'pillar'
NEED = 38


def worker(which, checkpoint, out):
    import torch
    import yaml
    import eval_suite as es
    from envs.swarm_env import SwarmEnv

    cfg = yaml.safe_load(open(os.path.join(ROOT, CONFIG)))
    cfg.pop('scenario_mix', None)                        # as eval_suite
    cfg['environment']['collision_physics'] = False
    cfg['environment']['n_agents'] = 1
    log = os.path.join(OUT, f'{which}.webots.log')
    if which == 'old':
        cfg['environment']['simulator'] = 'webots_l2'
        cfg['webots'] = {'real': True, 'log': log}
    else:
        cfg['environment']['simulator'] = 'webots_l2_multi'
        cfg['webots'] = {'log': log}
    env = SwarmEnv(cfg, device=torch.device('cpu'))
    pol = es.MappoPolicy(cfg, checkpoint, torch.device('cpu'), True)
    pol.restore_normalizer(env)
    max_steps = cfg['environment']['max_steps']
    rows, t0 = [], time.time()
    for seed in SEEDS:
        np.random.seed(seed); torch.manual_seed(seed)
        rng = np.random.RandomState(seed)
        for ep in range(EPISODES):
            obs = es.setup_obstacle_test(env, rng, TEST)
            sid = es.scenario_hash(env)
            track, tilt, speed, info = [], 0.0, 0.0, {}
            for _ in range(max_steps):
                obs, _, done, info = env.step(pol.act(obs, env, 1))
                ad = env.adapter
                track.append(ad._pos[0].tolist())
                tilt = max(tilt, float(np.degrees(np.abs(ad._orn[0, :2]).max())))
                speed = max(speed, float(np.linalg.norm(ad._vel[0])))
                if done:
                    break
            stats = {'max_speed': [speed], 'max_roll_deg': [tilt], 'max_pitch_deg': [tilt]}
            rows.append({'seed': seed, 'episode': ep, 'scenario_id': sid,
                         'outcome': 'success' if info.get('success') else (info.get('reason') or 'timeout'),
                         'steps': len(track), 'track': track, 'max_tilt_deg': tilt, 'max_speed': speed,
                         'lost_control': bool(es.lost_control(stats, 0))})
            print(f"{which} seed {seed} ep {ep}: {rows[-1]['outcome']} ({len(track)} steps)", flush=True)
    env.close()
    json.dump({'which': which, 'wall_s': time.time() - t0, 'episodes': rows}, open(out, 'w'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--worker', choices=['old', 'new'], default=None)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if a.worker:
        worker(a.worker, a.checkpoint, a.out)
        return

    res = {}
    for which in ('old', 'new'):                         # one after the other, own process each
        out = os.path.join(OUT, f'{which}.json')
        p = subprocess.run([sys.executable, os.path.abspath(__file__), '--worker', which,
                            '--checkpoint', a.checkpoint, '--out', out], cwd=ROOT)
        if p.returncode != 0:
            sys.exit(f"{which} run failed (exit {p.returncode}); see {OUT}/{which}.webots.log")
        res[which] = json.load(open(out))

    old = {(r['seed'], r['episode']): r for r in res['old']['episodes']}
    new = {(r['seed'], r['episode']): r for r in res['new']['episodes']}
    keys = sorted(old)
    same = [old[k]['outcome'] == new[k]['outcome'] for k in keys]
    diffs = []
    for k in keys:
        a_, b_ = np.array(old[k]['track']), np.array(new[k]['track'])
        m = min(len(a_), len(b_))
        diffs.append(float(np.linalg.norm(a_[:m] - b_[:m], axis=1).max()) if m else float('nan'))
    report = {
        'episodes': len(keys),
        'same_outcome': int(sum(same)),
        'pass': bool(sum(same) >= NEED),
        'rule': f'same outcome in >= {NEED}/{len(keys)}',
        'outcomes_old': {o: sum(old[k]['outcome'] == o for k in keys) for o in ('success', 'collision', 'timeout')},
        'outcomes_new': {o: sum(new[k]['outcome'] == o for k in keys) for o in ('success', 'collision', 'timeout')},
        'disagreements': [{'seed': k[0], 'episode': k[1], 'old': old[k]['outcome'], 'new': new[k]['outcome']}
                          for k, s in zip(keys, same) if not s],
        'track_diff_m': {'median': float(np.nanmedian(diffs)), 'max': float(np.nanmax(diffs))},
        'lost_control': {'old': sum(old[k]['lost_control'] for k in keys),
                         'new': sum(new[k]['lost_control'] for k in keys)},
        'scenario_hash_mismatches': sum(old[k]['scenario_id'] != new[k]['scenario_id'] for k in keys),
        'wall_s': {'old': res['old']['wall_s'], 'new': res['new']['wall_s']},
    }
    json.dump(report, open(os.path.join(OUT, 'report.json'), 'w'), indent=2)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
