"""
render_videos.py
----------------
Multi-angle MP4 of one deterministic episode, rendered off-screen (no PyBullet
window). The episode is flown first exactly as in eval_suite.py (same policy
loading, deterministic actions, drone-drone contact physics off); the recorded
positions are then replayed in a separate render-only world and filmed from
four cameras at once:

    top-down  |  3/4 view
    ----------+-----------
    side view |  chase cam (behind drone 0)

Each drone gets its own colour and a trail of where it has flown; goals are
translucent spheres that turn solid once that drone has settled. A run can
retry new seeds until an episode succeeds (--require-success), so the video
shows a typical success; every attempt's outcome is printed, so how many tries
it took is on record.

Usage (from the repo root):
    python scripts/render_videos.py --checkpoint <ckpt> --scenario wide_building --out videos/x.mp4
    python scripts/render_videos.py --checkpoint <ckpt> --scenario headon --n-agents 8 --no-obstacles ...

Scenarios: pillar, building, wide_building, long_wall (1 drone, 1 obstacle on
its path, as in eval_suite.py), the held-out l_shape, gap, cluster, slalom
(1 drone), and headon, crossing, swap, flythrough (multi-drone; obstacles
generated as in training unless --no-obstacles).
Pass the evaluation config with --config (e.g. config_stage2_round7_eval_goalhold.yaml);
when it enables goal_hold, the policy is wrapped with goal-hold as in eval_suite.py.
Needs ffmpeg on PATH.
"""
import argparse
import os
import subprocess
import sys

import numpy as np
import pybullet as p
import pybullet_data
import torch
import yaml
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import eval_suite as es                                    # noqa: E402
from envs import scenarios                                 # noqa: E402
from envs.swarm_env import SwarmEnv                        # noqa: E402
from modules.goal_hold import GoalHoldPolicy, goal_hold_enabled   # noqa: E402

COLOURS = [(0.12, 0.47, 0.71), (0.84, 0.15, 0.16), (0.17, 0.63, 0.17), (1.0, 0.5, 0.05),
           (0.58, 0.40, 0.74), (0.55, 0.34, 0.29), (0.89, 0.47, 0.76), (0.09, 0.75, 0.81)]


# ------------------------------------------------------------- fly episode --
def fly(config, ckpt, scenario, seed, no_obstacles):
    """One deterministic episode. Returns the layout, goals, per-step drone
    positions, per-step settled flags and the final info dict."""
    n = config['environment']['n_agents']
    env = SwarmEnv(config, device=torch.device('cpu'))
    es.disable_all_collision_physics(env, n)
    pol = es.MappoPolicy(config, ckpt, torch.device('cpu'), True)
    if goal_hold_enabled(config):                 # as in eval_suite.main()
        pol = GoalHoldPolicy(pol, config)
    rng = np.random.RandomState(seed)
    np.random.seed(seed)
    if scenario in es.OBSTACLE_TESTS:
        obs = es.setup_obstacle_test(env, rng, scenario)
    elif scenario in es.HELDOUT_TESTS:
        obs = es.setup_heldout_test(env, rng, scenario)
    else:
        spawns, goals = scenarios.GENERATORS[scenario](rng, n)
        obs = env.start_episode(spawns, goals, rng=rng, obstacles=[] if no_obstacles else 'auto')
    layout, goals = list(env.layout), env.goals.copy()
    pos, settled, info = [env.adapter.get_positions().copy()], [np.zeros(n, bool)], {}
    for _ in range(config['environment']['max_steps']):
        obs, _, done, info = env.step(pol.act(obs, env, n))
        pos.append(env.adapter.get_positions().copy())
        settled.append(np.array(env.drone_reached_goal, bool))
        if done:
            break
    env.close()
    return layout, goals, np.array(pos), np.array(settled), info


# ----------------------------------------------------------- render world --
class RenderWorld:
    """Separate DIRECT client holding only visuals: ground, obstacles, goals,
    drones. Drones are moved to recorded positions before each frame."""

    def __init__(self, layout, goals, n, drone_half):
        self.c = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.c)
        p.loadURDF('plane.urdf', physicsClientId=self.c)
        for ob in layout:
            h = ob.get('height', 8.0)
            cx, cy = ob['center']
            if ob['type'] == 'pillar':
                vis = p.createVisualShape(p.GEOM_CYLINDER, radius=ob['radius'], length=h,
                                          rgbaColor=[0.62, 0.52, 0.47, 1], physicsClientId=self.c)
            else:
                hx, hy = ob['half']
                vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[hx, hy, h / 2],
                                          rgbaColor=[0.68, 0.68, 0.72, 1], physicsClientId=self.c)
            p.createMultiBody(0, baseVisualShapeIndex=vis, basePosition=[cx, cy, h / 2],
                              baseOrientation=p.getQuaternionFromEuler([0, 0, ob.get('yaw', 0.0)]),
                              physicsClientId=self.c)
        self.drone_ids = []
        for i in range(n):
            r, g, b = COLOURS[i % len(COLOURS)]
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[drone_half] * 3, rgbaColor=[r, g, b, 1],
                                      physicsClientId=self.c)
            self.drone_ids.append(p.createMultiBody(0, baseVisualShapeIndex=vis, basePosition=[0, 0, -5],
                                                    physicsClientId=self.c))

    def update(self, positions):
        for i, b in enumerate(self.drone_ids):
            p.resetBasePositionAndOrientation(b, positions[i].tolist(), [0, 0, 0, 1], physicsClientId=self.c)

    def shoot(self, cam, w, h):
        view = p.computeViewMatrixFromYawPitchRoll(cam['target'], cam['dist'], cam['yaw'], cam['pitch'], 0, 2,
                                                   physicsClientId=self.c)
        proj = p.computeProjectionMatrixFOV(cam.get('fov', 50), w / h, 0.05, 200, physicsClientId=self.c)
        img = p.getCameraImage(w, h, view, proj, renderer=p.ER_TINY_RENDERER, shadow=1,
                               lightDirection=[0.4, 0.3, 1.0], physicsClientId=self.c)[2]
        rgb = np.reshape(np.asarray(img, dtype=np.uint8), (h, w, 4))[:, :, :3]
        return Image.fromarray(rgb), np.array(view).reshape(4, 4).T, np.array(proj).reshape(4, 4).T


def to_pixels(points, view, proj, w, h):
    """World points (k, 3) -> pixel coords (k, 2) and a visible mask."""
    hom = np.c_[points, np.ones(len(points))]
    clip = (proj @ view @ hom.T).T
    ok = clip[:, 3] > 1e-6
    ndc = clip[:, :3] / np.where(ok, clip[:, 3], 1.0)[:, None]
    px = np.c_[(ndc[:, 0] + 1) * 0.5 * w, (1 - ndc[:, 1]) * 0.5 * h]
    return px, ok


def overlay(img, view, proj, w, h, pos, goals, settled, k, trails, fnt):
    """Goal rings (filled once that drone has settled), trails, and a numbered
    marker on each drone, drawn on top of the rendered image."""
    n = pos.shape[1]
    right = view[0, :3]                                     # camera's right axis in world frame
    ov = Image.new('RGBA', img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    for i in range(n):
        col = tuple(int(255 * c) for c in COLOURS[i % len(COLOURS)])
        (c,), ok = to_pixels(goals[i:i + 1], view, proj, w, h)
        (e,), _ = to_pixels(goals[i:i + 1] + 0.5 * right, view, proj, w, h)
        if ok[0]:
            r = max(float(np.linalg.norm(e - c)), 4.0)
            d.ellipse([c[0] - r, c[1] - r, c[0] + r, c[1] + r], fill=col + ((130,) if settled[i] else (35,)),
                      outline=col + (255,), width=2)
        if trails:
            px, okk = to_pixels(pos[:k + 1, i], view, proj, w, h)
            seg = [tuple(q) for q, o in zip(px, okk) if o]
            if len(seg) > 1:
                d.line(seg, fill=col + (230,), width=3)
    img = Image.alpha_composite(img.convert('RGBA'), ov)
    d = ImageDraw.Draw(img)
    px, okk = to_pixels(pos[k], view, proj, w, h)
    for i in range(n):
        if okk[i] and -20 < px[i][0] < w + 20 and -20 < px[i][1] < h + 20:
            col = tuple(int(255 * c) for c in COLOURS[i % len(COLOURS)])
            x, y = px[i]
            d.ellipse([x - 7, y - 7, x + 7, y + 7], fill=col, outline=(255, 255, 255), width=2)
            if n > 1:
                d.text((x + 9, y - 22), str(i), fill=col, font=fnt, stroke_width=2, stroke_fill=(255, 255, 255))
    return img.convert('RGB')


def font(size):
    for name in ('arialbd.ttf', 'DejaVuSans-Bold.ttf', 'arial.ttf'):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


# ----------------------------------------------------------------- cameras --
def cameras(pos, goals, layout, k, w, h):
    """The four cameras for frame k."""
    pts = np.vstack([pos.reshape(-1, 3), goals])
    lo, hi = pts.min(0), pts.max(0)
    centre = (lo + hi) / 2
    centre[2] = 1.5
    span = max(hi[0] - lo[0], hi[1] - lo[1], 4.0)
    travel = goals.mean(0) - pos[0].mean(0)
    main_yaw = np.degrees(np.arctan2(travel[1], travel[0])) if np.linalg.norm(travel[:2]) > 0.5 else 0.0
    d0 = pos[k, 0]
    v = pos[min(k + 8, len(pos) - 1), 0] - pos[max(k - 8, 0), 0]
    chase_yaw = (np.degrees(np.arctan2(v[1], v[0])) - 90.0) if np.linalg.norm(v[:2]) > 0.05 else None
    return [
        ('TOP-DOWN', dict(target=centre.tolist(), dist=span * 2.7 + 6, yaw=0, pitch=-89.9, fov=24), True),
        ('3/4 VIEW', dict(target=centre.tolist(), dist=span * 1.9 + 6, yaw=main_yaw + 35, pitch=-38, fov=32), True),
        ('SIDE VIEW', dict(target=centre.tolist(), dist=span * 1.9 + 6, yaw=main_yaw, pitch=-8, fov=32), True),
        ('CHASE CAM (drone 0)', dict(target=d0.tolist(), dist=3.2,
                                     yaw=chase_yaw if chase_yaw is not None else main_yaw - 90, pitch=-22,
                                     fov=60), True),
    ]


def render(layout, goals, pos, settled, out, title, subtitle, fps, frame_every, tile=(640, 360), drone_half=0.15):
    n = pos.shape[1]
    world = RenderWorld(layout, goals, n, drone_half)
    w, h = tile
    W, H = 2 * w, 2 * h + 96
    ff = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                           '-s', f'{W}x{H}', '-r', str(fps), '-i', '-', '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                           '-crf', '20', out], stdin=subprocess.PIPE)
    f_big, f_small, f_tile = font(26), font(17), font(18)
    frames = list(range(0, len(pos), frame_every))
    if frames[-1] != len(pos) - 1:
        frames.append(len(pos) - 1)
    frames += [len(pos) - 1] * int(1.5 * fps)                 # hold the last frame
    chase_yaw_prev = None
    for k in frames:
        world.update(pos[k])
        canvas = Image.new('RGB', (W, H), (250, 250, 250))
        for t, (label, cam, trails) in enumerate(cameras(pos, goals, layout, k, w, h)):
            if label.startswith('CHASE'):                     # smooth the chase camera's turning
                if chase_yaw_prev is not None:
                    d = (cam['yaw'] - chase_yaw_prev + 180) % 360 - 180
                    cam['yaw'] = chase_yaw_prev + 0.15 * d
                chase_yaw_prev = cam['yaw']
            img, view, proj = world.shoot(cam, w, h)
            img = overlay(img, view, proj, w, h, pos, goals, settled[k], k, trails, f_tile)
            draw = ImageDraw.Draw(img)
            draw.rectangle([0, 0, len(label) * 11 + 16, 28], fill=(255, 255, 255))
            draw.text((8, 4), label, fill=(30, 30, 30), font=f_tile)
            canvas.paste(img, ((t % 2) * w, 96 + (t // 2) * h))
        d = ImageDraw.Draw(canvas)
        d.text((14, 8), title, fill=(20, 20, 20), font=f_big)
        n_set = int(settled[k].sum())
        status = f"t = {k * 0.032:5.1f} s    drones settled at goal: {n_set}/{n}"
        d.text((14, 42), subtitle, fill=(90, 90, 90), font=f_small)
        d.text((14, 68), status, fill=(20, 110, 20) if n_set == n else (90, 90, 90), font=f_small)
        d.line([(w, 96), (w, H)], fill=(255, 255, 255), width=3)
        d.line([(0, 96 + h), (W, 96 + h)], fill=(255, 255, 255), width=3)
        ff.stdin.write(canvas.tobytes())
    ff.stdin.close()
    ff.wait()
    p.disconnect(world.c)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/config_stage2_ring.yaml')
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--scenario', required=True)
    ap.add_argument('--n-agents', type=int, default=None)
    ap.add_argument('--no-obstacles', action='store_true')
    ap.add_argument('--seed', type=int, default=100)
    ap.add_argument('--require-success', action='store_true')
    ap.add_argument('--max-attempts', type=int, default=30)
    ap.add_argument('--title', default='')
    ap.add_argument('--fps', type=int, default=16)
    ap.add_argument('--frame-every', type=int, default=2, help='steps per frame (2 steps x 0.032 s at 16 fps = real time)')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()

    config = yaml.safe_load(open(a.config))
    config['environment']['collision_physics'] = False
    if a.n_agents:
        config['environment']['n_agents'] = a.n_agents
    if a.scenario in es.OBSTACLE_TESTS + es.HELDOUT_TESTS:    # one drone, as in eval_suite.py
        config['environment']['n_agents'] = 1
    if a.no_obstacles:
        config.setdefault('obstacles', {})['enabled'] = False
    for attempt in range(a.max_attempts):
        seed = a.seed + attempt
        layout, goals, pos, settled, info = fly(config, a.checkpoint, a.scenario, seed, a.no_obstacles)
        reason = info.get('reason') or 'timeout'
        print(f"  seed {seed}: {reason}, {len(pos) - 1} steps, {int(settled[-1].sum())}/{pos.shape[1]} settled",
              flush=True)
        if reason == 'success' or not a.require_success:
            break
    n = pos.shape[1]
    sep = min(np.linalg.norm(pos[:, i] - pos[:, j], axis=1).min() for i in range(n) for j in range(i + 1, n)) \
        if n > 1 else None
    sub = (f"{a.scenario}, {n} drone{'s' if n > 1 else ''}, "
           f"{'no obstacles' if a.no_obstacles or not layout else f'{len(layout)} obstacle(s)'}, "
           f"outcome: {reason}" + (", goal-hold on" if goal_hold_enabled(config) else "")
           + (f", closest drone-drone distance {sep:.2f} m" if sep is not None else "")
           + f"   (seed {seed}, try {attempt + 1})")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    render(layout, goals, pos, settled, a.out, a.title or a.scenario, sub, a.fps, a.frame_every)
    print(f"wrote {a.out}: {sub}")


if __name__ == '__main__':
    main()
