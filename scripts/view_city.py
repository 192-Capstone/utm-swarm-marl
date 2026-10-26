"""Fly drones through a generated city district in PyBullet's 3D window.

    python scripts/view_city.py --level hard --seed 0 --drones 4
    python scripts/view_city.py --level medium --checkpoint checkpoints/stage2_imitation_ww_s0/bc_policy.pth

Buildings are shaded by height (darker = taller), goals are green spheres, and each episode is a
new district (seed, seed + 1, ...). Default pilot: the rule teacher; --checkpoint flies a trained
policy instead. Mouse: left-drag rotates, right-drag pans, wheel zooms. Close the window or
Ctrl-C to stop.
"""
import argparse
import os
import sys
import time

import numpy as np
import pybullet as p
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from envs.swarm_env import SwarmEnv  # noqa: E402
from baselines.repulsion_controller import RepulsionController  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--level', default='medium', choices=['easy', 'medium', 'hard'])
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--drones', type=int, default=4)
    ap.add_argument('--episodes', type=int, default=5)
    ap.add_argument('--config', default='configs/config_stage2_ring_wider_teacher.yaml')
    ap.add_argument('--checkpoint', default=None, help='fly a trained policy instead of the teacher')
    ap.add_argument('--speed', type=float, default=1.0, help='playback speed (1 = real time)')
    args = ap.parse_args()

    config = yaml.safe_load(open(args.config, encoding='utf-8'))
    config['environment']['n_agents'] = args.drones
    config['scenario_mix'] = {f'city_{args.level}': 1.0}
    env = SwarmEnv(config, gui=True)
    cid = env.adapter.client
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=cid)
    if args.checkpoint:
        from eval_suite import MappoPolicy
        pilot = MappoPolicy(config, args.checkpoint, 'cpu', True)
    else:
        pilot = RepulsionController(config)
    dt = float(config['environment'].get('control_timestep', 0.032))
    goal_vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.3, rgbaColor=[0.2, 0.85, 0.4, 0.8], physicsClientId=cid)
    try:
        for ep in range(args.episodes):
            np.random.seed(args.seed + ep)
            obs = env.reset()
            for b, ob in zip(env.adapter.obstacle_ids, env.layout):
                t = (ob['height'] - 6.0) / 14.0                     # 0 = 6 m, 1 = 20 m
                p.changeVisualShape(b, -1, rgbaColor=[0.55 - 0.35 * t, 0.62 - 0.35 * t, 0.75 - 0.3 * t, 1],
                                    physicsClientId=cid)
            markers = [p.createMultiBody(baseMass=0, baseVisualShapeIndex=goal_vis, basePosition=list(g),
                                         physicsClientId=cid) for g in env.goals]
            p.resetDebugVisualizerCamera(cameraDistance=45, cameraYaw=30 + 40 * ep, cameraPitch=-55,
                                         cameraTargetPosition=[0, 0, 0], physicsClientId=cid)
            print(f"episode {ep + 1}: city_{args.level}, {len(env.layout)} buildings, {args.drones} drones", flush=True)
            done, steps, info = False, 0, {}
            while not done and steps < config['environment']['max_steps']:
                t0 = time.time()
                obs, _, done, info = env.step(pilot.act(obs, env, args.drones))
                steps += 1
                time.sleep(max(0.0, dt / args.speed - (time.time() - t0)))
            print(f"   -> {info.get('reason')} after {steps * dt:.1f} s, "
                  f"{sum(info.get('drone_reached_goal', []))}/{args.drones} drones at their goal", flush=True)
            time.sleep(1.5)
            for m in markers:
                p.removeBody(m, physicsClientId=cid)
    except p.error:
        pass                                                         # window closed
    finally:
        try:
            env.close()
        except p.error:
            pass


if __name__ == '__main__':
    main()
