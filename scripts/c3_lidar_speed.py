"""
c3_lidar_speed.py
-----------------
C3: the speed cost of the Webots Lidar (webots.lidar), as C5's speed case (scripts/c5_webots_city.py
case_speed: hard city, MAP_SEED 20, routes, 10 simulated seconds) for 4 / 8 / 12 / 16 / 20 drones,
each in its own Python process (one Webots each, one at a time):

    off       Lidar off (as C5)
    compare   Lidar on, policy on the analytic scan, both logged
    policy    Lidar on, policy on the Lidar scan

Per run: wall seconds per simulated second for the adapter alone (zero commands: Webots + the drone
controllers, incl. rendering and sending the scans) and for env + policy (steps actually run), and,
over the env + policy phase, per simulated second:
    recv_s      supervisor: receiving and unpacking the scans
    pick_s      supervisor: picking the 46 rays per drone + logging (envs/webots_lidar.py)
    copy_s      drone controllers: copying the range image out (summed over drones; they run in
                parallel processes, so this is CPU, not necessarily wall time)
    MB          scan bytes received
Usage: DISPLAY=:1 OMP_NUM_THREADS=4 nice -n 19 python scripts/c3_lidar_speed.py --checkpoint <ckpt>
Writes eval_results/c3/speed/report.json and prints a table.
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, 'eval_results', 'c3', 'speed')
CONFIG = 'configs/config_stage2_round7_eval_goalhold.yaml'
MAP_SEED, SECS = 20, 10.0
NS, MODES = (4, 8, 12, 16, 20), ('off', 'compare', 'policy')


def worker(n, mode, checkpoint):
    import torch
    import eval_suite as es
    from envs.city_generator import generate_city, sample_flights
    from envs.route_planner import plan, simplify
    from envs.swarm_env import SwarmEnv
    from modules.goal_hold import GoalHoldPolicy, goal_hold_enabled
    city = generate_city(np.random.RandomState(MAP_SEED), 'hard')
    sp, gl = sample_flights(np.random.RandomState(MAP_SEED), city, n)
    routes = {}
    for k in sp:
        p = plan(city['grid'], sp[k][:2], gl[k][:2])
        routes[k] = simplify(city['grid'], p) if p is not None else [tuple(sp[k][:2]), tuple(gl[k][:2])]
    cfg = yaml.safe_load(open(os.path.join(ROOT, CONFIG)))
    cfg.pop('scenario_mix', None)
    cfg['environment'].update(n_agents=n, collision_physics=False, simulator='webots_l2_multi',
                              route_follow={'enabled': True})
    cfg.setdefault('obstacles', {})['enabled'] = False
    cfg['webots'] = {'log': os.path.join(OUT, f'n{n}_{mode}.webots.log')}
    if mode != 'off':
        cfg['webots']['lidar'] = {'enabled': True, 'use_for_policy': mode == 'policy'}
    t0 = time.time()
    env = SwarmEnv(cfg, device=torch.device('cpu'))
    startup = time.time() - t0
    ad = env.adapter
    steps = int(round(SECS / ad.control_dt))
    t0 = time.time(); env.start_episode(sp, gl, obstacles=city['layout'], routes=routes); reset1 = time.time() - t0
    t0 = time.time()
    for _ in range(steps):                                      # adapter alone
        ad.apply_actions({f'drone_{i}': [0.0] * 4 for i in range(n)})
        ad.step_simulation()
    wall_adapter = time.time() - t0
    policy = es.MappoPolicy(cfg, checkpoint, torch.device('cpu'), True)
    policy.restore_normalizer(env)
    if goal_hold_enabled(cfg):
        policy = GoalHoldPolicy(policy, cfg)
    obs = env.start_episode(sp, gl, obstacles=city['layout'], routes=routes)
    s0 = ad.lidar_stats()
    n_log0 = len(ad._scan_log.rows['cpu_send']) if s0 else 0
    t0 = time.time()
    k = 0
    for k in range(1, steps + 1):
        obs, _, done, _ = env.step(policy.act(obs, env, n))
        if done:
            break
    wall = time.time() - t0
    sim = k * ad.control_dt
    out = {'n': n, 'mode': mode, 'startup_s': startup, 'reset_s': reset1, 'policy_steps': k,
           'wall_per_sim_s_adapter': wall_adapter / SECS, 'wall_per_sim_s_env_policy': wall / sim}
    if s0:
        s1 = ad.lidar_stats()
        cpu = np.asarray(ad._scan_log.rows['cpu_send'][n_log0:], dtype=float)
        out.update(recv_s_per_sim_s=(s1['recv_s'] - s0['recv_s']) / sim,
                   pick_s_per_sim_s=(s1['pick_s'] - s0['pick_s']) / sim,
                   copy_s_per_sim_s=float(np.nansum(cpu)) / sim,
                   mb_per_sim_s=(s1['bytes'] - s0['bytes']) / 1e6 / sim,
                   missing=s1['missing'], stale=s1['stale'], wrong_time=s1['wrong_time'])
    env.close()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--ns', type=int, nargs='+', default=list(NS))
    ap.add_argument('--modes', nargs='+', default=list(MODES))
    ap.add_argument('--worker', nargs=2, default=None, help=argparse.SUPPRESS)
    ap.add_argument('--out', default=None, help=argparse.SUPPRESS)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if a.worker:
        json.dump(worker(int(a.worker[0]), a.worker[1], a.checkpoint), open(a.out, 'w'))
        return
    rep = []
    for n in a.ns:
        for mode in a.modes:
            out = os.path.join(OUT, f'n{n}_{mode}.json')
            t0 = time.time()
            p = subprocess.run([sys.executable, os.path.abspath(__file__), '--checkpoint', a.checkpoint,
                                '--worker', str(n), mode, '--out', out], cwd=ROOT, capture_output=True, text=True)
            if p.returncode:
                r = {'n': n, 'mode': mode, 'error': (p.stderr or p.stdout)[-2000:]}
            else:
                r = json.load(open(out))
            r['wall_s'] = time.time() - t0
            rep.append(r)
            print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()
                              if k != 'error'}) + ('  ERROR' if 'error' in r else ''), flush=True)
    json.dump(rep, open(os.path.join(OUT, 'report.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
