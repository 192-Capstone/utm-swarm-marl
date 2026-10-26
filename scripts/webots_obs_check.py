"""
webots_obs_check.py
-------------------
Does the Level-2 adapter give the policy the same observation as PyBullet for the same state?

R1 (or any policy) flies one episode on the Level-2 Crazyflie. After every step the PyBullet
drone is put in exactly the same state (position, velocity, orientation; same goal, same
obstacle layout, same LiDAR heading), and both raw states go through two identical
ObservationProcessors in lockstep (so the previous-step LiDAR inputs match too). Sensor noise
is switched off for the comparison (it is added by SwarmEnv the same way for any simulator).
Prints the max |difference| of the 74-input actor observation per input group, and how large
the orientation inputs get on the real quadrotor (PyBullet's velocity-controlled body never
rolls or pitches in its own flights).

Usage (repo root): python scripts/webots_obs_check.py --checkpoint <ckpt> [--test building] [--steps 300]
"""
import argparse
import os
import sys

import numpy as np
import pybullet as p
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import eval_suite as es                                          # noqa: E402
from envs.swarm_env import SwarmEnv                              # noqa: E402
from envs.pybullet_adapter import PyBulletAdapter                # noqa: E402
from modules.observation_processor import ObservationProcessor   # noqa: E402

GROUPS = {'position [0:3]': slice(0, 3), 'velocity [3:6]': slice(3, 6), 'orientation [6:9]': slice(6, 9),
          'goal vector+dist [9:13]': slice(9, 13), 'LiDAR fan [13:27]': slice(13, 27),
          'previous fan [27:41]': slice(27, 41), 'log goal dist [41]': slice(41, 42), 'ring [42:74]': slice(42, 74)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/config_stage2_round7_teacher.yaml')
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--test', default='building')
    ap.add_argument('--seed', type=int, default=20)
    ap.add_argument('--episode', type=int, default=0)
    ap.add_argument('--steps', type=int, default=400)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    cfg.pop('scenario_mix', None)
    cfg['environment']['collision_physics'] = False
    cfg['environment']['n_agents'] = 1
    cfg['environment'].pop('sensor_noise', None)                  # compare noise-free
    cfg['environment'].setdefault('dynamics', {})['lag'] = False   # states are set directly anyway
    wcfg = dict(cfg); wcfg['environment'] = dict(cfg['environment'], simulator='webots_l2')
    wcfg['webots'] = {'real': True}
    env = SwarmEnv(wcfg, device=torch.device('cpu'))
    wb = env.adapter
    pb = PyBulletAdapter(cfg)
    pol = es.MappoPolicy(wcfg, a.checkpoint, torch.device('cpu'), True)
    pol.restore_normalizer(env)
    rng = np.random.RandomState(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    for _ in range(a.episode):
        es.setup_obstacle_test(env, rng, a.test)
    obs = es.setup_obstacle_test(env, rng, a.test)
    spawn = {'drone_0': wb.get_positions()[0].tolist()}
    goal = {'drone_0': list(map(float, env.goals[0]))}
    pb.reset_world(spawn, goal, env.layout)
    proc_wb, proc_pb = ObservationProcessor(wcfg, torch.device('cpu')), ObservationProcessor(cfg, torch.device('cpu'))
    for pr in (proc_wb, proc_pb):
        pr.reset_history()
    extra = None
    if proc_wb.critic_version >= 2:
        extra = {'settle_progress': np.zeros(1), 'reached': np.zeros(1, dtype=np.float32)}
    diffs = {g: 0.0 for g in GROUPS}
    worst_total, orient_max, n, done, info = 0.0, np.zeros(3), 0, False, {}
    big, total = 0, 0
    for step in range(a.steps):
        if step > 0:
            obs, _, done, info = env.step(pol.act(obs, env, 1))
        raw_wb = wb.get_raw_state()
        d = raw_wb['drone_0']
        p.resetBasePositionAndOrientation(pb.drone_ids[0], d['position'],
                                          p.getQuaternionFromEuler(d['orientation']), physicsClientId=pb.client)
        p.resetBaseVelocity(pb.drone_ids[0], d['velocity'], [0, 0, 0], physicsClientId=pb.client)
        pb._heading = dict(wb._heading)
        # the Webots adapter scanned with the heading BEFORE this call; give PyBullet the same start
        raw_pb = pb.get_raw_state()
        lf_wb = proc_wb.process(raw_wb, env.curriculum_stage, critic_extra=extra)[0]
        lf_pb = proc_pb.process(raw_pb, env.curriculum_stage, critic_extra=extra)[0]
        x_wb = np.asarray(lf_wb.cpu() if hasattr(lf_wb, 'cpu') else lf_wb)[0]
        x_pb = np.asarray(lf_pb.cpu() if hasattr(lf_pb, 'cpu') else lf_pb)[0]
        dd = np.abs(x_wb - x_pb)
        for g, sl in GROUPS.items():
            diffs[g] = max(diffs[g], float(dd[sl].max()))
        worst_total = max(worst_total, float(dd.max()))
        big += int((dd > 0.01).sum()); total += dd.size
        orient_max = np.maximum(orient_max, np.abs(np.asarray(d['orientation'])))
        n += 1
        if done:
            break
    env.close(); pb.close()
    print(f"{a.test}, seed {a.seed} episode {a.episode}: {n} steps compared, observation size {x_wb.shape[0]}; "
          f"episode {info.get('reason') or ('running' if not done else '?')}")
    print(f"max |obs difference| over all 74 inputs and steps: {worst_total:.6f}; "
          f"inputs differing by > 0.01: {big} of {total} ({100 * big / max(total, 1):.3f}%)")
    for g, v in diffs.items():
        print(f"  {g:24s} {v:.6f}")
    print(f"max |roll|, |pitch|, |yaw| on the Level-2 Crazyflie: {np.degrees(orient_max).round(2)} deg "
          f"(inputs [6:9] = angle / pi; PyBullet's own flights keep roll = pitch = 0)")


if __name__ == '__main__':
    main()
