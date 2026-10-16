"""
webots_record_video.py
----------------------
Record one evaluation episode in Webots (Level 1 or Level 2) as an MP4.

The episode is reproduced exactly as eval_suite.py draws it (same seed, the
scenario RNG advanced through the earlier episodes), and its scenario hash is
checked against eval_suite's episodes.csv when given. Adds goal markers (green),
a flight trail and a caption. Needs a display: Webots renders the scene
(a window opens while recording).

Usage (repo root):
  python scripts/webots_record_video.py --simulator webots --scenario headon --n-agents 3 \
      --policy mappo --checkpoint <ckpt> --seed 10 --episode 1 --out logs/webots/video_l1_headon.mp4
"""
import argparse
import copy
import csv
import math
import os
import sys
import time

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import eval_suite as es                                        # noqa: E402
from envs.swarm_env import SwarmEnv                            # noqa: E402
from envs import scenarios                                     # noqa: E402
from record_rollout import (setup_crossing_scenario, setup_swap_scenario,   # noqa: E402
                            setup_headon_scenario, _apply_scenario)
from baselines.repulsion_controller import RepulsionController  # noqa: E402

COLORS = ['0.1 0.4 0.9', '0.9 0.3 0.1', '0.6 0.2 0.8', '0.8 0.6 0.1', '0.3 0.3 0.6',
          '0.1 0.7 0.8', '0.9 0.2 0.6', '0.4 0.4 0.4']   # no green: goals are green


def setup(env, rng, scenario, n):
    if scenario in es.HELDOUT_TESTS:
        return es.setup_heldout_test(env, rng, scenario)
    if scenario in es.OBSTACLE_TESTS:
        return es.setup_obstacle_test(env, rng, scenario)
    if scenario == 'crossing':
        return setup_crossing_scenario(env, rng, n)
    if scenario == 'swap':
        return setup_swap_scenario(env, rng, n)
    if scenario == 'headon':
        return setup_headon_scenario(env, rng, n)
    if scenario == 'flythrough':
        spawns, goals = scenarios.flythrough(rng, n)
        return _apply_scenario(env, spawns, goals, n, rng=rng)
    raise ValueError(scenario)


def axis_angle(R):
    """3x3 rotation -> Webots axis-angle [x, y, z, a]."""
    a = math.acos(max(-1.0, min(1.0, (np.trace(R) - 1) / 2)))
    if a < 1e-6:
        return [0, 0, 1, 0]
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * math.sin(a))
    return [float(v[0]), float(v[1]), float(v[2]), a]


def set_view(sup, pts):
    """Camera behind/above the scene (looking +y and down) framing all points."""
    pts = np.asarray(pts)
    c = (pts.min(0) + pts.max(0)) / 2
    span = max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1]), 4.0)
    pitch = math.radians(65)                                     # steep, so tall obstacles hide less
    dist = 1.25 * span + 3.0
    pos = c + np.array([0.0, -dist * math.cos(pitch), dist * math.sin(pitch)])
    pos[2] = max(pos[2], 4.0)
    yaw = math.pi / 2                                           # look along +y
    Rz = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
    Ry = np.array([[math.cos(pitch), 0, math.sin(pitch)], [0, 1, 0], [-math.sin(pitch), 0, math.cos(pitch)]])
    vp = sup.getFromDef('VIEW') or next(n for n in [sup.getRoot().getField('children').getMFNode(i)
                                                    for i in range(sup.getRoot().getField('children').getCount())]
                                        if n.getTypeName() == 'Viewpoint')
    vp.getField('position').setSFVec3f(pos.tolist())
    vp.getField('orientation').setSFRotation(axis_angle(Rz @ Ry))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/config_stage2_ring.yaml')
    ap.add_argument('--simulator', choices=['webots', 'webots_l2'], default='webots')
    ap.add_argument('--scenario', required=True)
    ap.add_argument('--n-agents', type=int, default=3)
    ap.add_argument('--policy', choices=['mappo', 'repulsion'], default='mappo')
    ap.add_argument('--checkpoint', default=None)
    ap.add_argument('--seed', type=int, default=10)
    ap.add_argument('--episode', type=int, default=0)
    ap.add_argument('--episodes-csv', default=None, help="eval_suite episodes.csv to check the scenario hash")
    ap.add_argument('--out', required=True)
    ap.add_argument('--title', default='')
    ap.add_argument("--accel", type=int, default=2, help="video speed-up (whole number)")
    ap.add_argument('--width', type=int, default=1280)
    ap.add_argument('--height', type=int, default=720)
    ap.add_argument('--tail-s', type=float, default=1.5, help="seconds recorded after the episode ends")
    a = ap.parse_args()

    cfg = yaml.safe_load(open(a.config))
    cfg.pop('scenario_mix', None)
    cfg['environment']['collision_physics'] = False
    obstacle = a.scenario in es.OBSTACLE_TESTS or a.scenario in es.HELDOUT_TESTS
    n = 1 if obstacle else a.n_agents
    cfg['environment']['n_agents'] = n
    cfg['environment']['simulator'] = a.simulator
    cfg['webots'] = dict(cfg.get('webots') or {}, render=True)
    env = SwarmEnv(cfg, device=torch.device('cpu'))
    sup = env.adapter.sup
    if a.policy == 'mappo':
        pol = es.MappoPolicy(cfg, a.checkpoint, torch.device('cpu'), True)
        pol.restore_normalizer(env)
    else:
        pol = RepulsionController(cfg)

    # reproduce eval_suite's draw: global seeds, scenario RNG through earlier episodes
    np.random.seed(a.seed); torch.manual_seed(a.seed)
    rng = np.random.RandomState(a.seed)
    for _ in range(a.episode):
        setup(env, rng, a.scenario, n)
    obs = setup(env, rng, a.scenario, n)
    sid = es.scenario_hash(env)
    if a.episodes_csv:
        want = {int(r['episode']): r for r in csv.DictReader(open(a.episodes_csv)) if int(r['seed']) == a.seed}[a.episode]
        print(f"scenario {sid} (evaluation had {want['scenario_id']}, outcome {want['outcome']}, "
              f"{want['length_steps']} steps): {'MATCH' if sid == want['scenario_id'] else 'DIFFERENT'}")

    root = sup.getRoot().getField('children')
    root.importMFNodeFromString(-1, 'DEF MARKS Group { }')
    marks = sup.getFromDef('MARKS').getField('children')
    goals = [env.adapter.goals[f'drone_{i}'] for i in range(n)]
    for i, g in enumerate(goals):
        marks.importMFNodeFromString(-1, f'Solid {{ translation {g[0]} {g[1]} {g[2]} name "goal_{i}" children [ Shape {{ '
                                         f'appearance PBRAppearance {{ baseColor 0.1 0.9 0.2 transparency 0.6 metalness 0 }} '
                                         f'geometry Sphere {{ radius 0.5 subdivision 2 }} }} ] }}')
    if a.simulator == 'webots':
        for i, nd in enumerate(env.adapter.nodes):     # colour the drones apart
            nd.getField('children').getMFNode(0).getField('appearance').getSFNode().getField('baseColor').setSFColor(
                [float(x) for x in COLORS[i % len(COLORS)].split()])
    pts = [env.adapter.get_positions()[i] for i in range(n)] + goals
    for ob in env.layout:
        pts.append([ob['center'][0], ob['center'][1], 0.0])
    set_view(sup, pts)
    sup.step(env.adapter.basic_ms)

    title = a.title or f"{a.simulator} | {a.scenario} | {a.policy}"
    sup.setLabel(0, title, 0.01, 0.01, 0.07, 0x000000, 0.0, 'Arial')
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    sup.movieStartRecording(os.path.abspath(a.out), a.width, a.height, 0, 90, a.accel, False)
    max_steps = cfg['environment']['max_steps']
    done, info, step = False, {}, 0
    for step in range(1, max_steps + 1):
        obs, _, done, info = env.step(pol.act(obs, env, n))
        if step % 6 == 0:                       # trail dots every ~0.2 s
            for i, p in enumerate(env.adapter.get_positions()):
                marks.importMFNodeFromString(-1, f'Solid {{ translation {p[0]} {p[1]} {p[2]} name "t{step}_{i}" children [ Shape {{ '
                                                 f'appearance PBRAppearance {{ baseColor {COLORS[i % len(COLORS)]} metalness 0 }} '
                                                 f'geometry Sphere {{ radius 0.04 }} }} ] }}')
        sup.setLabel(1, f"t = {step * env.adapter.control_dt:5.1f} s   step {step}", 0.01, 0.07, 0.06, 0x333333, 0.0, 'Arial')
        if done:
            break
    outcome = info.get('reason') or ('timeout' if step >= max_steps else 'running')
    sup.setLabel(2, f"outcome: {outcome} after {step} steps", 0.01, 0.12, 0.07,
                 0x118811 if outcome == 'success' else 0xbb2211, 0.0, 'Arial')
    for _ in range(int(a.tail_s / env.adapter.control_dt)):
        sup.step(env.adapter.step_ms)
    sup.movieStopRecording()
    t0 = time.time()
    while not sup.movieIsReady() and time.time() - t0 < 300:
        sup.step(env.adapter.basic_ms)
    print(f"episode outcome: {outcome} after {step} steps; movie {'FAILED' if sup.movieFailed() else 'written'}: {a.out}")
    env.close()


if __name__ == '__main__':
    main()
