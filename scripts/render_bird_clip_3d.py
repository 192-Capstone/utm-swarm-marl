"""
render_bird_clip_3d.py
----------------------
The 3D version of the bird demo clips: the same episode as scripts/render_bird_clip.py (same seed, bird paths, lag and
sensor noise), flown (a) without avoidance and (b) with the final bird-avoidance configuration, rendered with PyBullet's
own camera (getCameraImage, TinyRenderer: offscreen, no GUI window, no GPU). Drones are blue (orange while the bird
guard is active, red once crashed), birds are red spheres with a short trail, buildings (when the episode has any) are
the environment's own boxes and pillars. Each half has a soft chase camera (same azimuth, same rules) that keeps its encounter in view. Playback is real time, slowed to 0.5x around the encounter.

    python scripts/render_bird_clip_3d.py --checkpoint <ckpt>                    # both default episodes, one at a time
    python scripts/render_bird_clip_3d.py --checkpoint <ckpt> --cell single      # only the single-drone episode
    python scripts/render_bird_clip_3d.py --checkpoint <ckpt> --cell crowd --seed 24 --episode 21

Output (eval_results/birds/clips/, git-ignored), per episode:
    <name>_without_3d.mp4   <name>_with_3d.mp4   <name>_side_by_side_3d.mp4   (+ <name>_3d.json)
Seeds 20-24 only. Needs the ffmpeg on the PATH (the one the 2D clips used); nothing is committed or deleted.
"""

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch                                                                    # noqa: E402
import pybullet as p                                                            # noqa: E402
import pybullet_data                                                            # noqa: E402
from PIL import Image, ImageDraw, ImageFont                                     # noqa: E402
import eval_birds as eb                                                         # noqa: E402
import render_bird_clip as rb                                                   # noqa: E402  (flying, schedule, helpers)

# the episodes of the 2D clips
DEFAULTS = {'single': (22, 36), 'crowd': (24, 21)}
BLUE, ORANGE, RED = (0.12, 0.45, 0.95, 1.0), (1.0, 0.55, 0.05, 1.0), (0.85, 0.1, 0.1, 1.0)
BIRD_RGBA, GOAL_RGBA = (0.85, 0.1, 0.1, 1.0), (0.17, 0.7, 0.25, 1.0)
PX = {'blue': (31, 119, 230), 'orange': (255, 140, 13), 'red': (214, 39, 40), 'bird': (200, 30, 30)}


# ------------------------------------------------------------- the caption
def dodge_style(rec, min_m=0.4):
    """'climbing', 'sidestepping' or 'braking' (majority over the drones whose guard fired): from where the drone was
    when its guard first fired, until 1.3 s after it last fired, the biggest rise, or sideways offset from the straight
    line to its goal; neither above min_m means it mostly slowed down."""
    votes = []
    P, G, dt, goals = rec['pos'], rec['guard'], rec['dt'], rec['goals']
    for k in range(P.shape[1]):
        act = np.flatnonzero(G[:, k])
        if not len(act):
            continue
        i0, i1 = int(act[0]), min(int(act[-1]) + 40, len(P) - 1)
        seg = P[i0:i1 + 1, k] - P[i0, k]
        to_goal = goals[k, :2] - P[i0, k, :2]
        u = to_goal / max(np.linalg.norm(to_goal), 1e-6)
        dz = float(np.abs(seg[:, 2]).max())
        lat = float(np.abs(seg[:, 0] * -u[1] + seg[:, 1] * u[0]).max())
        votes.append('braking' if max(dz, lat) < min_m else 'climbing' if dz >= lat else 'sidestepping')
    return max(set(votes), key=votes.count) if votes else None


def captions(a, b):
    n = a['pos'].shape[1]
    na = int(a['crashed'][-1].sum())
    if na:
        cap_a = 'Without bird guard: crash' if n == 1 else f'Without bird guard: {na} of {n} drones crash'
    else:
        cap_a = f'Without bird guard: near miss ({a["min_dist"]:.2f} m)'
    nb = int(b['crashed'][-1].sum())
    style = dodge_style(b)
    if nb:
        cap_b = 'With bird guard: crash' if n == 1 else f'With bird guard: {nb} of {n} drones crash'
    elif style:
        cap_b = f'With bird guard: dodged by {style}'
    else:
        cap_b = 'With bird guard: no crash'
    return cap_a, cap_b, style


# --------------------------------------------------------------- the camera
def smooth(x, sigma):
    """Zero-phase Gaussian smoothing along axis 0 (edges held), sigma in samples."""
    r = int(3 * sigma) + 1
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    xp = np.concatenate([np.repeat(x[:1], r, axis=0), x, np.repeat(x[-1:], r, axis=0)])
    return np.stack([np.convolve(xp[:, j], k, mode='valid') for j in range(x.shape[1])], axis=1)


def camera_track(a, b, near=7.0, sigma_s=0.9, pitch=-33.0, azim_off=28.0):
    """Per sim step: (target xyz, distance) and a fixed azimuth, shared by both halves. The focus is the drones that
    have a bird within `near` m (all drones when none has) and those birds; the camera frames their bounding box,
    side-on to the flight direction so that climbing and sidestepping both read."""
    n = max(len(a['pos']), len(b['pos']))
    idx = lambda r: np.minimum(np.arange(n), len(r['pos']) - 1)                       # noqa: E731
    Pa, Pb, B = a['pos'][idx(a)], b['pos'][idx(b)], a['bird'][np.minimum(np.arange(n), len(a['bird']) - 1)]
    tgt, ext = np.zeros((n, 3)), np.zeros(n)
    for t in range(n):
        D = np.concatenate([Pa[t], Pb[t]])
        d = np.linalg.norm(D[:, None] - B[t][None], axis=2)                          # (drones, birds)
        close = d.min(axis=1) < near
        pts = D[close] if close.any() else D
        if close.any():
            pts = np.concatenate([pts, B[t][(d[close] < near).any(axis=0)]])
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        tgt[t], ext[t] = (lo + hi) / 2, max(hi[0] - lo[0], hi[1] - lo[1], 1.4 * (hi[2] - lo[2]))
    dist = np.clip(3.5 + 0.9 * ext, 4.5, 24.0)
    sg = sigma_s / a['dt']
    sm = smooth(np.c_[tgt, dist], sg)
    sm[:, 2] = np.maximum(sm[:, 2], 0.9)                                               # keep the ground in view
    g0, p0 = a['goals'][0], a['pos'][0, 0]
    heading = math.degrees(math.atan2(g0[1] - p0[1], g0[0] - p0[0]))
    return sm[:, :3], sm[:, 3], heading + 90.0 + azim_off, pitch


def view_proj(target, dist, azim, pitch, W, H, fov=58.0):
    az, pt = math.radians(azim), math.radians(pitch)
    eye = np.asarray(target) + dist * np.array([math.cos(pt) * math.cos(az), math.cos(pt) * math.sin(az), -math.sin(pt)])
    V = p.computeViewMatrix(eye.tolist(), list(map(float, target)), [0, 0, 1])
    P = p.computeProjectionMatrixFOV(fov, W / H, 0.1, 250.0)
    return V, P


def project(V, P, pts, W, H):
    M = np.array(P).reshape(4, 4).T @ np.array(V).reshape(4, 4).T
    h = np.c_[np.asarray(pts, float).reshape(-1, 3), np.ones(len(np.reshape(pts, (-1, 3))))] @ M.T
    w = np.where(np.abs(h[:, 3]) < 1e-6, 1e-6, h[:, 3])
    return np.c_[(h[:, 0] / w + 1) * 0.5 * W, (1 - h[:, 1] / w) * 0.5 * H], h[:, 3] > 0


# --------------------------------------------------------------- the scene
class Scene:
    """One PyBullet DIRECT world (its own client) holding the ground, the buildings, the goals, the drones and the birds."""

    def __init__(self, rec):
        self.cid = p.connect(p.DIRECT)
        c = self.cid
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=c)
        p.loadURDF('plane.urdf', physicsClientId=c)
        self.n, self.nb = rec['pos'].shape[1], rec['bird'].shape[1]
        for ob in rec['layout']:                                                      # the same boxes / pillars as the adapter
            h = ob['height']
            if ob['type'] == 'pillar':
                vis = p.createVisualShape(p.GEOM_CYLINDER, radius=ob['radius'], length=h, rgbaColor=[0.55, 0.43, 0.39, 1], physicsClientId=c)
            else:
                hx, hy = ob['half']
                vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[hx, hy, h / 2], rgbaColor=[0.62, 0.62, 0.66, 1], physicsClientId=c)
            p.createMultiBody(0, -1, vis, [ob['center'][0], ob['center'][1], h / 2],
                              p.getQuaternionFromEuler([0, 0, ob.get('yaw', 0.0)]), physicsClientId=c)
        for g in rec['goals']:                                                        # a pole with a ball at the goal height
            h = max(float(g[2]), 0.1)
            pole = p.createVisualShape(p.GEOM_CYLINDER, radius=0.025, length=h, rgbaColor=[0.17, 0.7, 0.25, 1], physicsClientId=c)
            ball = p.createVisualShape(p.GEOM_SPHERE, radius=0.11, rgbaColor=list(GOAL_RGBA), physicsClientId=c)
            disc = p.createVisualShape(p.GEOM_CYLINDER, radius=0.3, length=0.01, rgbaColor=[0.17, 0.7, 0.25, 1], physicsClientId=c)
            p.createMultiBody(0, -1, pole, [g[0], g[1], h / 2], physicsClientId=c)
            p.createMultiBody(0, -1, ball, [g[0], g[1], h], physicsClientId=c)
            p.createMultiBody(0, -1, disc, [g[0], g[1], 0.006], physicsClientId=c)
        r = rec['radius']
        sph = p.createVisualShape(p.GEOM_SPHERE, radius=r, rgbaColor=list(BIRD_RGBA), physicsClientId=c)
        self.birds = [p.createMultiBody(0, -1, sph, [0, 0, 50], physicsClientId=c) for _ in range(self.nb)]
        # a drone: a centre box, two crossed arms and four rotor discs (roughly the 0.3 m body of the env)
        shapes = [(p.GEOM_BOX, [0.09, 0.09, 0.035], [0, 0, 0]), (p.GEOM_BOX, [0.21, 0.014, 0.01], [0, 0, 0.0]),
                  (p.GEOM_BOX, [0.014, 0.21, 0.01], [0, 0, 0.0])]
        for sx in (-0.21, 0.21):
            for sy in (-0.21, 0.21):
                shapes.append((p.GEOM_CYLINDER, None, [sx, sy, 0.02]))
        kinds = [s[0] for s in shapes]
        vis = p.createVisualShapeArray(
            shapeTypes=kinds, radii=[0.0 if s[0] == p.GEOM_BOX else 0.085 for s in shapes],
            halfExtents=[s[1] if s[1] else [0, 0, 0] for s in shapes],
            lengths=[0.0 if s[0] == p.GEOM_BOX else 0.012 for s in shapes],
            rgbaColors=[list(BLUE)] * len(shapes), visualFramePositions=[s[2] for s in shapes],
            physicsClientId=c)
        self.drones = [p.createMultiBody(0, -1, vis, [0, 0, 50], physicsClientId=c) for _ in range(self.n)]
        self.color = [None] * self.n
        self.crash_marks = []
        mark = p.createVisualShape(p.GEOM_SPHERE, radius=0.28, rgbaColor=[0.9, 0.1, 0.1, 1], physicsClientId=c)
        self.crash_marks = [p.createMultiBody(0, -1, mark, [0, 0, 50], physicsClientId=c) for _ in range(self.n)]
        self.yaw = np.zeros(self.n)
        # cheap blob shadows on the ground (TinyRenderer's real shadows double the render time): the height cue
        sd = p.createVisualShape(p.GEOM_CYLINDER, radius=0.22, length=0.004, rgbaColor=[0.12, 0.12, 0.14, 1], physicsClientId=c)
        sb = p.createVisualShape(p.GEOM_CYLINDER, radius=max(r, 0.05) * 1.1, length=0.004, rgbaColor=[0.12, 0.12, 0.14, 1], physicsClientId=c)
        self.sh_d = [p.createMultiBody(0, -1, sd, [0, 0, 50], physicsClientId=c) for _ in range(self.n)]
        self.sh_b = [p.createMultiBody(0, -1, sb, [0, 0, 50], physicsClientId=c) for _ in range(self.nb)]

    def set_color(self, k, rgba):
        if self.color[k] != rgba:
            n = len(p.getVisualShapeData(self.drones[k], physicsClientId=self.cid))
            for s in range(n):
                p.changeVisualShape(self.drones[k], -1, shapeIndex=s, rgbaColor=list(rgba), physicsClientId=self.cid)
            self.color[k] = rgba

    def place(self, pos, bird, vel, crashed, active):
        c = self.cid
        for k in range(self.n):
            if np.hypot(vel[k][0], vel[k][1]) > 0.3 and not crashed[k]:
                self.yaw[k] = math.atan2(vel[k][1], vel[k][0])
            p.resetBasePositionAndOrientation(self.drones[k], pos[k].tolist(), p.getQuaternionFromEuler([0, 0, self.yaw[k] + math.pi / 4]), physicsClientId=c)
            self.set_color(k, RED if crashed[k] else ORANGE if active[k] else BLUE)
            p.resetBasePositionAndOrientation(self.crash_marks[k], pos[k].tolist() if crashed[k] else [0, 0, 50],
                                              [0, 0, 0, 1], physicsClientId=c)
        for k in range(self.nb):
            p.resetBasePositionAndOrientation(self.birds[k], bird[k].tolist(), [0, 0, 0, 1], physicsClientId=c)
            p.resetBasePositionAndOrientation(self.sh_b[k], [bird[k][0], bird[k][1], 0.012], [0, 0, 0, 1], physicsClientId=c)
        for k in range(self.n):
            p.resetBasePositionAndOrientation(self.sh_d[k], [pos[k][0], pos[k][1], 0.012], [0, 0, 0, 1], physicsClientId=c)

    def shot(self, V, P, W, H):
        _, _, rgba, _, _ = p.getCameraImage(W, H, V, P, renderer=p.ER_TINY_RENDERER, flags=p.ER_NO_SEGMENTATION_MASK,
                                            shadow=0, lightDirection=[-0.4, -0.5, 1.0], physicsClientId=self.cid)
        return np.asarray(rgba, dtype=np.uint8).reshape(H, W, 4)[..., :3]

    def close(self):
        p.disconnect(self.cid)


# ------------------------------------------------------------- the overlay
_FONTS = {}


def font(size, bold=True):
    key = (size, bold)
    if key not in _FONTS:
        for f in (r'C:\Windows\Fonts\arialbd.ttf' if bold else r'C:\Windows\Fonts\arial.ttf', 'DejaVuSans-Bold.ttf', 'DejaVuSans.ttf'):
            try:
                _FONTS[key] = ImageFont.truetype(f, size)
                break
            except OSError:
                continue
        else:
            _FONTS[key] = ImageFont.load_default()
    return _FONTS[key]


def lerp(arr, s):
    return rb.lerp_rows(arr, s)


def draw_frame(img, rec, s, V, P, caption, clock_txt, W, H):
    L = len(rec['pos'])
    sc = min(s, L - 1.0)
    i = min(int(sc), L - 1)
    pos, bird = lerp(rec['pos'], sc), lerp(rec['bird'], min(s, len(rec['bird']) - 1.0))
    crashed, active = rec['crashed'][i], rec['guard'][i] & ~rec['crashed'][i]
    im = Image.fromarray(img)
    dr = ImageDraw.Draw(im, 'RGBA')
    n, nb = pos.shape[0], bird.shape[0]
    # drop lines to the ground shadows (the height cue)
    for arr in (pos, bird):
        top, _ = project(V, P, arr, W, H)
        bot, _ = project(V, P, np.c_[arr[:, :2], np.zeros(len(arr))], W, H)
        for u, v in zip(top, bot):
            dr.line([tuple(u), tuple(v)], fill=(60, 60, 60, 110), width=1)
    # trails: drone path so far (orange where the guard was active), bird path over the last ~1.3 s (fading)
    for k in range(n):
        path = np.vstack([rec['pos'][:i + 1, k], pos[k][None]])
        xy, ok = project(V, P, path, W, H)
        g = np.append(rec['guard'][:i + 1, k], active[k])
        for j in range(1, len(path)):
            if ok[j] and ok[j - 1]:
                dr.line([tuple(xy[j - 1]), tuple(xy[j])], fill=PX['orange'] + (235,) if g[j] else PX['blue'] + (200,), width=4 if g[j] else 3)
    j0 = max(0, int(min(s, len(rec['bird']) - 1.0)) - 40)
    for k in range(nb):
        path = np.vstack([rec['bird'][j0:int(min(s, len(rec['bird']) - 1.0)) + 1, k], bird[k][None]])
        xy, ok = project(V, P, path, W, H)
        for j in range(1, len(path)):
            if ok[j] and ok[j - 1]:
                dr.line([tuple(xy[j - 1]), tuple(xy[j])], fill=PX['bird'] + (int(40 + 190 * j / len(path)),), width=3)
    # tracked-by-the-detector rings are left out on purpose: the video shows what flew, not what the drone believed
    d = np.linalg.norm(pos[:, None] - bird[None], axis=2)
    di, bi = np.unravel_index(int(np.argmin(d)), d.shape)
    dmin, dsf = float(d[di, bi]), float(rec['dist'][:i + 1].min())
    if dmin < 3.0:
        xy, _ = project(V, P, np.array([pos[di], bird[bi]]), W, H)
        dr.line([tuple(xy[0]), tuple(xy[1])], fill=(255, 255, 255, 230) if dmin >= 0.5 else PX['red'] + (255,), width=2)
    # frame, caption, status
    nc = int(crashed.sum())
    edge = PX['red'] if nc else PX['orange'] if active.any() else (190, 190, 190)
    dr.rectangle([0, 0, W - 1, H - 1], outline=edge, width=9 if (nc or active.any()) else 2)
    f_cap = font(40)
    tw = dr.textlength(caption, font=f_cap)
    dr.rounded_rectangle([W / 2 - tw / 2 - 24, 16, W / 2 + tw / 2 + 24, 78], radius=14, fill=(255, 255, 255, 225), outline=edge + (255,), width=3)
    dr.text((W / 2 - tw / 2, 22), caption, font=f_cap, fill=(30, 30, 30, 255))
    badges = []
    if nc:
        badges.append(('CRASH' if nc == 1 else f'CRASH x{nc}', PX['red']))
    if active.any():
        badges.append(('GUARD ACTIVE' if n == 1 else f'GUARD ACTIVE ({int(active.sum())} of {n})', PX['orange']))
    y = 96
    for txt, col in badges:
        w = dr.textlength(txt, font=font(34))
        dr.rounded_rectangle([24, y, 24 + w + 28, y + 52], radius=10, fill=(255, 255, 255, 230), outline=col + (255,), width=4)
        dr.text((38, y + 5), txt, font=font(34), fill=col + (255,))
        y += 64
    box = f'drone-bird distance {dmin:4.1f} m\nclosest so far {dsf:4.2f} m\n{clock_txt}'
    dr.rounded_rectangle([20, H - 118, 470, H - 20], radius=10, fill=(255, 255, 255, 220), outline=(PX['red'] if dmin < 0.5 else (190, 190, 190)) + (255,), width=3)
    dr.multiline_text((34, H - 110), box, font=font(24), fill=(30, 30, 30, 255), spacing=4)
    return np.asarray(im)


# ---------------------------------------------------------------- encoding
def open_ffmpeg(path, W, H, fps):
    ffm = shutil.which('ffmpeg')
    if not ffm:
        raise SystemExit('ffmpeg not found on the PATH')
    return subprocess.Popen([ffm, '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', str(fps),
                             '-i', '-', '-an', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18', '-preset', 'medium', path],
                            stdin=subprocess.PIPE)


def render_half(rec, caption, sched, speed, cam, W, H, fps, out_path, preview=None):
    """Render one run to out_path. preview: {frame index: png path} to save stills too."""
    tgt, dist, azim, pitch = cam
    sc = Scene(rec)
    proc = open_ffmpeg(out_path, W, H, fps) if out_path else None
    t0 = time.time()
    try:
        for f, s in enumerate(sched):
            if preview and f not in preview:                       # stills only: skip every other frame
                continue
            ci = min(s, len(tgt) - 1.0)
            T, D = lerp(tgt, ci), float(np.interp(ci, np.arange(len(dist)), dist))
            V, P = view_proj(T, D, azim, pitch, W, H)
            L = len(rec['pos'])
            i = min(int(min(s, L - 1.0)), L - 1)
            pos = lerp(rec['pos'], min(s, L - 1.0))
            bird = lerp(rec['bird'], min(s, len(rec['bird']) - 1.0))
            vel = (rec['pos'][min(i + 1, L - 1)] - rec['pos'][max(i - 1, 0)]) / (2 * rec['dt'])
            crashed = rec['crashed'][i]
            sc.place(pos, bird, vel, crashed, rec['guard'][i] & ~crashed)
            img = sc.shot(V, P, W, H)
            sp = float(np.interp(s, np.arange(len(speed)), speed))
            clock = f'sim time {s * rec["dt"]:5.2f} s   playback {1.0 / sp:.2f}x' + ('  (slow motion)' if sp > 1.2 else '')
            img = draw_frame(img, rec, s, V, P, caption, clock, W, H)
            if proc is not None:
                proc.stdin.write(img.tobytes())
            if preview and f in preview:
                Image.fromarray(img).save(preview[f])
            if f % 100 == 0:
                print(f'    frame {f}/{len(sched)}  ({time.time() - t0:.0f} s)', flush=True)
    finally:
        if proc is not None:
            proc.stdin.close()
            proc.wait()
        sc.close()
    return time.time() - t0


def side_by_side(left, right, out_path):
    subprocess.run([shutil.which('ffmpeg'), '-y', '-loglevel', 'error', '-i', left, '-i', right, '-filter_complex', 'hstack=inputs=2',
                    '-an', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18', '-preset', 'medium', out_path], check=True)


# --------------------------------------------------------------------- main
def run_cell(cell, seed, ep, args, final_cfg):
    t0 = time.time()
    print(f'=== {cell}: seed {seed}, episode {ep}', flush=True)
    fliers = [rb.Flier(cell, False, final_cfg, args), rb.Flier(cell, True, final_cfg, args)]
    a = fliers[0].fly(seed, [ep])[ep]
    b = fliers[1].fly(seed, [ep])[ep]
    for fl in fliers:
        fl.env.close()
    t_fly = time.time() - t0
    cap_a, cap_b, style = captions(a, b)
    nb_ = min(len(a['bird']), len(b['bird']))
    diff = float(np.abs(a['bird'][:nb_] - b['bird'][:nb_]).max())
    print(f'  without: {a["outcome"]}, {int(a["crashed"][-1].sum())} crashed, closest {a["min_dist"]:.2f} m, {len(a["pos"])} steps | '
          f'with: {b["outcome"]}, {int(b["crashed"][-1].sum())} crashed, closest {b["min_dist"]:.2f} m, {len(b["pos"])} steps, '
          f'guard {int(b["guard"].sum())} drone-steps, style {style} | bird paths differ by {diff:.1e} m', flush=True)
    sched, _, speed, total = rb.time_map(a, b, args.fps, args.slow, args.hold_end)
    W, H = args.width, args.height
    name = args.name or f'bird_clip_{cell}_s{seed}_e{ep}'
    od = args.out_dir
    os.makedirs(od, exist_ok=True)
    preview = None
    if args.preview_dir:
        os.makedirs(args.preview_dir, exist_ok=True)
        d = np.minimum(a['dist'].min(axis=(1, 2))[:, None], 99).ravel()
        s_close = int(np.argmin(d))
        picks = {int(np.argmin(np.abs(sched - q))): q for q in (0, s_close - 25, s_close, min(s_close + 25, len(a['pos']) - 1), n_last(a, b))}
    paths = {'without': os.path.join(od, f'{name}_without_3d.mp4'), 'with': os.path.join(od, f'{name}_with_3d.mp4'),
             'both': os.path.join(od, f'{name}_side_by_side_3d.mp4')}
    times = {}
    for key, rec, cap in (('without', a, cap_a), ('with', b, cap_b)):
        if args.half not in ('all', key):
            continue
        cam = camera_track(rec, rec)                                          # each half keeps its own run in view
        print(f'  rendering "{cap}" ({len(sched)} frames, {total:.1f} s)', flush=True)
        if args.preview_dir:
            preview = {f: os.path.join(args.preview_dir, f'{name}_{key}_{k}.png') for k, f in enumerate(sorted(picks))}
        times[key] = render_half(rec, cap, sched, speed, cam, W, H, args.fps, None if args.preview_dir else paths[key], preview)
        print(f'    {times[key]:.0f} s', flush=True)
    if not args.preview_dir and args.half in ('all', 'stitch'):
        t1 = time.time()
        side_by_side(paths['without'], paths['with'], paths['both'])
        times['stitch'] = time.time() - t1
        meta = {'cell': cell, 'seed': seed, 'episode': ep, 'captions': [cap_a, cap_b], 'dodge_style': style, 'clip_seconds': round(total, 1),
                'frames': len(sched), 'size_each': [W, H], 'files': paths, 'bird_path_max_diff_m': diff,
                'without_guard': {'outcome': a['outcome'], 'crashed_drones': int(a['crashed'][-1].sum()), 'closest_m': round(a['min_dist'], 3)},
                'with_guard': {'outcome': b['outcome'], 'crashed_drones': int(b['crashed'][-1].sum()), 'closest_m': round(b['min_dist'], 3)},
                'seconds': {'flying': round(t_fly, 1), **{k: round(v, 1) for k, v in times.items()}, 'total': round(time.time() - t0, 1)}}
        with open(os.path.join(od, f'{name}_3d.json'), 'w') as f:
            json.dump(meta, f, indent=1)
        for v in paths.values():
            print(f'  wrote {v}', flush=True)
    print(f'  done in {time.time() - t0:.0f} s (flying {t_fly:.0f} s)', flush=True)


def n_last(a, b):
    return max(len(a['pos']), len(b['pos'])) - 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cell', choices=['single', 'crowd', 'both'], default='both')
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--config', default=os.path.join(ROOT, eb.CONFIG))
    ap.add_argument('--pp', default=os.path.join(ROOT, 'eval_results', 'birds', 'pp'))
    ap.add_argument('--out-dir', default=os.path.join(ROOT, 'eval_results', 'birds', 'clips'))
    ap.add_argument('--seed', type=int, default=None, help='with --episode and one --cell: another episode than the 2D clip')
    ap.add_argument('--episode', type=int, default=None)
    ap.add_argument('--width', type=int, default=1280, help='of each half; the side-by-side file is twice as wide')
    ap.add_argument('--height', type=int, default=720)
    ap.add_argument('--fps', type=int, default=30)
    ap.add_argument('--slow', type=float, default=2.0, help='slow-motion factor near the encounter (2 = 0.5x)')
    ap.add_argument('--hold-end', type=float, default=2.5)
    ap.add_argument('--name', default=None)
    ap.add_argument('--half', choices=['all', 'without', 'with', 'stitch'], default='all',
                    help='render only that half (one process per half keeps memory low); stitch = hstack the two existing halves')
    ap.add_argument('--preview-dir', default=None, help='write a few PNG stills there instead of videos (to check the look)')
    args = ap.parse_args()
    if (args.seed is None) != (args.episode is None):
        ap.error('--seed and --episode go together')
    if args.seed is not None and (args.cell == 'both' or args.seed not in rb.SEEDS_OK):
        ap.error('--seed/--episode need one --cell, seeds 20-24 only')
    torch.set_num_threads(2)
    with open(os.path.join(args.pp, 'final_config.json')) as f:
        final_cfg = json.load(f)
    for cell in (['single', 'crowd'] if args.cell == 'both' else [args.cell]):        # one render at a time
        seed, ep = (args.seed, args.episode) if args.seed is not None else DEFAULTS[cell]
        run_cell(cell, seed, ep, args, final_cfg)
    return 0


if __name__ == '__main__':
    sys.exit(main())
