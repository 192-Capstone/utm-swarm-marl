"""
render_bird_clip.py
-------------------
A short side-by-side demo clip: the same episode flown twice with identical seed, bird paths, lag and sensor noise,
(a) with birds and no avoidance, (b) with the final bird-avoidance configuration (eval_results/birds/pp/final_config.json:
bird guard, realistic detector, Kalman tracker). Records per step the drone positions, the bird positions, whether the
guard is active, which birds each drone tracks and the crashes, picks a good example automatically, and renders
"Without bird guard" | "With bird guard" (top-down view + height strip) to an MP4 (or a GIF if ffmpeg is missing).

    python scripts/render_bird_clip.py --cell single --checkpoint <ckpt>       # 1 drone, 2 birds
    python scripts/render_bird_clip.py --cell crowd  --checkpoint <ckpt>       # 8 drones, 4 birds, fair crowd
    python scripts/render_bird_clip.py --cell single --seed 20 --episode 40 --checkpoint <ckpt>   # a given episode

Candidates for the automatic pick come from the stage 3a++ final runs (episodes.csv of <pp>/final_<cell>_none_* and
_new_*): episodes where (a) ended in a bird crash or a pass under 0.3 m and (b) was a success. Each candidate is
flown again here and the best reproduction is rendered. Validation seeds 20-24 only. Nothing is committed or deleted;
the clips go to eval_results/birds/clips/ (git-ignored).
"""

import argparse
import csv
import json
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
import yaml                                                                     # noqa: E402
import eval_birds as eb                                                         # noqa: E402
import eval_suite as es                                                         # noqa: E402
from envs.swarm_env import SwarmEnv                                             # noqa: E402
from modules.goal_hold import GoalHoldPolicy, goal_hold_enabled                 # noqa: E402
from modules.bird_guard import BirdGuardPolicy                                  # noqa: E402
from record_rollout import setup_headon_scenario, disable_all_collision_physics  # noqa: E402

SEEDS_OK = {20, 21, 22, 23, 24}
PARKED_Z = 1.0e3
# the two example families: (scenario args, number of birds, final-run cell name, drones)
CELLS = {'single': (eb.NOBS, 2, 'single_noobs_b2', 1),
         'crowd': (eb.CROWD + eb.FAIR, 4, 'fair_b4', 8)}
COL = {'drone': '#1f77b4', 'guard': '#ff7f0e', 'bird': '#b22222', 'goal': '#2ca02c', 'crash': '#d62728', 'grey': '#888888'}


# ------------------------------------------------------------------ flying
def make_config(cell, avoid, final_cfg, config_path):
    scen, n_birds, _, n_agents = CELLS[cell]
    with open(config_path) as f:
        config = yaml.safe_load(f)
    config.pop('scenario_mix', None)
    config['environment']['collision_physics'] = False
    config['environment']['n_agents'] = n_agents
    config.setdefault('obstacles', {})['enabled'] = False                 # both cells are --no-obstacles
    birds = {'enabled': True, 'count': n_birds}
    if '--bird-spawn' in scen:
        birds['spawn'] = scen[scen.index('--bird-spawn') + 1]
    if avoid:                                                             # same mapping as eval_suite.py's flags
        birds['guard'] = {'enabled': True, **final_cfg.get('guard_set', {})}
        if final_cfg.get('brake') is not None:
            birds['guard']['brake'] = list(final_cfg['brake'])
        det = {'enabled': True, 'mode': 'realistic', 'birds_in_slots': False,
               'range_m': final_cfg['range'], 'fov_deg': final_cfg['fov']}
        if final_cfg.get('tracker'):
            det['tracker'] = final_cfg['tracker']
        if final_cfg.get('share'):
            det['share'] = True
        det.update(final_cfg.get('detector_set', {}))
        birds['detector'] = det
    config['environment']['birds'] = birds
    return config


class Flier:
    """One environment + policy for one condition; flies chosen episodes of a seed with the suite's seeding."""

    def __init__(self, cell, avoid, final_cfg, args):
        self.cell, self.avoid = cell, avoid
        self.config = make_config(cell, avoid, final_cfg, args.config)
        self.n = CELLS[cell][3]
        self.env = SwarmEnv(self.config, device=torch.device('cpu'))
        disable_all_collision_physics(self.env, self.n)
        policy = es.MappoPolicy(self.config, args.checkpoint, torch.device('cpu'), True)
        policy.restore_normalizer(self.env)
        if goal_hold_enabled(self.config):
            policy = GoalHoldPolicy(policy, self.config)
        self.guard = None
        if avoid:
            policy = self.guard = BirdGuardPolicy(policy, self.config)
        self.policy = policy
        self.dt = self.config['environment'].get('control_timestep', 0.032)
        self.max_steps = self.config['environment']['max_steps']

    def fly(self, seed, episodes):
        """{episode: record} for the wanted episodes of this seed. The earlier ones are only generated (as in
        eval_suite --episode-offset), so the scenario, bird paths and noise match a plain run."""
        env, out = self.env, {}
        np.random.seed(seed)
        torch.manual_seed(seed)
        rng = np.random.RandomState(seed)
        for ep in range(max(episodes) + 1):
            if self.cell == 'crowd':
                obs = setup_headon_scenario(env, rng, self.n)
            else:
                obs = es.sample_random_band(env, rng, self.n, 7.0, 10.0)
            if ep in episodes:
                out[ep] = self.record(obs)
        return out

    def record(self, obs):
        env, n = self.env, self.n
        rec = {'dt': self.dt, 'goals': env.goals.copy(), 'radius': float(env.birds.radii[0]), 'layout': list(env.layout or [])}
        P, B, G, T, C = [], [], [], [], []
        last = env.adapter.get_positions().copy()

        def snap(active, tracked, crashed):
            nonlocal last
            pos = env.adapter.get_positions().copy()
            for i in range(n):                                   # a crashed drone is parked far away: keep where it crashed
                if pos[i][2] > PARKED_Z / 2:
                    pos[i] = last[i]
            last = pos
            P.append(pos)
            B.append(np.asarray(env.birds.positions(), dtype=float).copy())
            G.append(np.array(active, dtype=bool))
            T.append(np.array(tracked, dtype=bool))
            C.append(np.array(crashed, dtype=bool))

        nb = len(env.birds)
        snap(np.zeros(n, bool), np.zeros((n, nb), bool), np.zeros(n, bool))
        info = {}
        for _ in range(self.max_steps):
            obs, _, done, info = env.step(self.policy.act(obs, env, n))
            tracked = env._tracked_matrix() if env.detector is not None and env._bird_tracks is not None else np.zeros((n, nb), bool)
            snap(self.guard.active if self.guard is not None else np.zeros(n, bool), tracked, env.crashed)
            if done:
                break
        P, B = np.array(P), np.array(B)
        d = np.linalg.norm(P[:, :, None, :] - B[:, None, :, :], axis=3)            # (steps, drone, bird)
        bc = np.array(info.get('bird_crashed', [False] * n), dtype=bool)
        C[-1] = C[-1] | bc                                          # the episode ends on the crash step: env.crashed is already reset
        crashed = np.array(C)
        rec.update({'pos': P, 'bird': B, 'guard': np.array(G), 'tracked': np.array(T), 'crashed': crashed, 'dist': d,
                    'bird_crashed': np.array(info.get('bird_crashed', [False] * n), dtype=bool),
                    'outcome': 'success' if info.get('success') else ('bird' if info.get('reason') == 'bird' else
                                                                      'timeout' if info.get('reason') is None else info.get('reason')),
                    'reached': np.array(info.get('drone_reached_goal', [False] * n), dtype=bool),
                    'min_dist': float(d.min())})
        return rec


# --------------------------------------------------------------- selection
def csv_candidates(pp, cell, limit):
    """(seed, episode) where the final run's no-avoidance episode ended in a bird crash or a pass under 0.3 m and the
    final configuration succeeded without a bird crash; shortest first (a short clip, an easy story)."""
    name = CELLS[cell][2]

    def load(cond):
        rows = {}
        for part in 'AB':
            f = os.path.join(pp, f'final_{name}_{cond}_{part}', 'episodes.csv')
            if os.path.exists(f):
                with open(f, newline='') as fh:
                    for r in csv.DictReader(fh):
                        rows[(int(r['seed']), int(r['episode']))] = r
        return rows
    a, b = load('none'), load('new')
    out = []
    for k in a.keys() & b.keys():
        A, B = a[k], b[k]
        bad = int(A['bird_crash']) == 1 or float(A['min_bird_dist'] or 9) < 0.3
        if bad and int(B['success']) and not int(B['bird_crash']) and int(B.get('guard_activated') or 0):
            out.append((int(B['length_steps']) + int(A['length_steps']), k))
    return [k for _, k in sorted(out)[:limit]]


def score(cell, a, b):
    """None if (a, b) is not a usable example, else a score (higher is better)."""
    n = a['pos'].shape[1]
    if not (b['outcome'] == 'success' and not b['crashed'][-1].any() and b['guard'].any()):
        return None
    a_bad = a['crashed'][-1].any() or a['min_dist'] < 0.3
    if not a_bad:
        return None
    n_crash = int(a['crashed'][-1].sum())
    if cell == 'crowd':                                   # a clear difference: more drones saved, then a short, calm episode
        return n_crash * 10 + min(b['min_dist'], 1.0) - 0.002 * len(b['pos'])
    return (5 if a['crashed'][-1].any() else 0) + min(b['min_dist'], 1.0) - 0.002 * len(b['pos'])


def pick(cell, fliers, args, pp):
    if args.seed is not None and args.episode is not None:
        cands = [(args.seed, args.episode)]
    else:
        cands = csv_candidates(pp, cell, args.candidates)
        print(f'candidates from the final runs ({len(cands)}): {cands}', flush=True)
    best = None
    for seed, ep in cands:
        a = fliers[0].fly(seed, [ep])[ep]
        b = fliers[1].fly(seed, [ep])[ep]
        s = score(cell, a, b)
        n_a, n_b = int(a['crashed'][-1].sum()), int(b['crashed'][-1].sum())
        print(f'  seed {seed} episode {ep}: (a) {a["outcome"]}, {n_a} crashed, closest {a["min_dist"]:.2f} m | '
              f'(b) {b["outcome"]}, {n_b} crashed, closest {b["min_dist"]:.2f} m, {len(b["pos"])} steps | score {s}', flush=True)
        if args.seed is not None and args.episode is not None:
            return seed, ep, a, b
        if s is not None and (best is None or s > best[0]):
            best = (s, seed, ep, a, b)
    if best is None:
        return None
    return best[1], best[2], best[3], best[4]


# --------------------------------------------------------------- rendering
def time_map(a, b, fps, slow, hold_end):
    """Playback schedule: the sim step shown at each output frame. Real time (1x) away from the encounter, `slow` x
    (smoothly) when any drone-bird pair is within 4 m in either run. Both panels use the same sim time."""
    dt = a['dt']
    na, nb = len(a['pos']), len(b['pos'])
    n = max(na, nb)

    def near(r):
        d = r['dist'].min(axis=(1, 2))
        return np.concatenate([d, np.full(n - len(d), 99.0)])
    d = np.minimum(near(a), near(b))
    w = np.clip((5.0 - d) / 1.5, 0.0, 1.0)                    # 0 beyond 5 m, 1 inside 3.5 m
    w = w * w * (3 - 2 * w)
    w = np.convolve(w, np.ones(15) / 15, mode='same')         # soften the speed changes
    speed = 1.0 + (slow - 1.0) * w                            # output seconds per sim second
    t_out = np.concatenate([[0.0], np.cumsum(dt * speed[:-1])])      # output time of each sim step
    total = t_out[-1] + hold_end
    frames_t = np.arange(0.0, total, 1.0 / fps)
    return np.interp(frames_t, t_out, np.arange(n)), frames_t, speed, total


def lerp_rows(arr, s):
    s = min(max(s, 0.0), len(arr) - 1.0)
    i = int(np.floor(s))
    j = min(i + 1, len(arr) - 1)
    return arr[i] + (arr[j] - arr[i]) * (s - i)


def draw_layout(ax, layout):
    from matplotlib.patches import Circle, Rectangle
    for ob in layout:
        if ob['type'] == 'pillar':
            ax.add_patch(Circle(ob['center'][:2], ob['radius'], fc='#bbbbbb', ec='#666666', zorder=1))
        else:
            hx, hy = ob['half']
            ax.add_patch(Rectangle((ob['center'][0] - hx, ob['center'][1] - hy), 2 * hx, 2 * hy,
                                   angle=0, fc='#bbbbbb', ec='#666666', zorder=1))


class Panel:
    def __init__(self, fig, rect_top, rect_strip, rec, title, avoid, lims, zlim, tmax):
        import matplotlib.pyplot as plt                         # noqa: F401
        self.rec, self.avoid = rec, avoid
        self.n = rec['pos'].shape[1]
        self.ax = fig.add_axes(rect_top)
        self.st = fig.add_axes(rect_strip)
        ax, st = self.ax, self.st
        ax.set_xlim(*lims[0]); ax.set_ylim(*lims[1]); ax.set_aspect('equal', adjustable='box')
        ax.grid(True, color='#e3e3e3', lw=0.6); ax.set_axisbelow(True)
        ax.tick_params(labelsize=8)
        ax.set_xlabel('x (m)', fontsize=8, labelpad=1); ax.set_ylabel('y (m)', fontsize=8, labelpad=1)
        ax.set_title(title, fontsize=15, fontweight='bold', color='#222222', pad=6)
        draw_layout(ax, rec['layout'])
        g = rec['goals']
        ax.scatter(g[:, 0], g[:, 1], marker='*', s=140, c=COL['goal'], zorder=3, label='goal')
        p0 = rec['pos'][0]
        ax.scatter(p0[:, 0], p0[:, 1], marker='x', s=30, c=COL['grey'], zorder=2)
        self.trail = [ax.plot([], [], '-', lw=2, c=COL['drone'], zorder=4, solid_capstyle='round')[0] for _ in range(self.n)]
        self.trail_g = [ax.plot([], [], '-', lw=3, c=COL['guard'], zorder=4, solid_capstyle='round')[0] for _ in range(self.n)]
        self.drone = ax.scatter(np.zeros(self.n), np.zeros(self.n), s=95, c=COL['drone'], ec='white', lw=1.2, zorder=6)
        self.nb = rec['bird'].shape[1]
        self.btrail = [ax.plot([], [], '-', lw=1.6, c=COL['bird'], alpha=0.45, zorder=3)[0] for _ in range(self.nb)]
        self.bird = ax.scatter(np.zeros(self.nb), np.zeros(self.nb), s=60, c=COL['bird'], ec='white', lw=0.8, zorder=5)
        self.ring = ax.scatter([], [], s=230, facecolors='none', edgecolors='#6a3d9a', lw=1.4, zorder=4) if avoid else None
        self.link, = ax.plot([], [], '--', lw=1.2, c='#444444', zorder=3)
        self.xmark, = ax.plot([], [], 'x', ms=22, mew=5, c=COL['crash'], zorder=8)
        self.crash_txt = ax.text(0.5, 0.87, 'CRASH', transform=ax.transAxes, ha='center', va='center', fontsize=28,
                                 fontweight='bold', color=COL['crash'], zorder=9, visible=False,
                                 bbox=dict(boxstyle='round,pad=0.25', fc='white', ec=COL['crash'], lw=3))
        self.guard_txt = ax.text(0.03, 0.80, '', transform=ax.transAxes, ha='left', va='top', fontsize=13, fontweight='bold',
                                 color=COL['guard'], zorder=9)
        self.dist_txt = ax.text(0.03, 0.03, '', transform=ax.transAxes, ha='left', va='bottom', fontsize=12,
                                fontweight='bold', color='#222222', zorder=9,
                                bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='#cccccc', alpha=0.9))
        # height strip
        st.set_xlim(0, tmax); st.set_ylim(*zlim)
        st.grid(True, color='#e3e3e3', lw=0.6); st.set_axisbelow(True); st.tick_params(labelsize=8)
        st.set_xlabel('time (s)', fontsize=8, labelpad=1); st.set_ylabel('height (m)', fontsize=8, labelpad=1)
        t = np.arange(len(rec['pos'])) * rec['dt']
        self.t = t
        for k in range(self.nb):
            st.plot(t, rec['bird'][:, k, 2], '-', lw=1.0, c=COL['bird'], alpha=0.25, zorder=1)
        self.zline = [st.plot([], [], '-', lw=2 if self.n == 1 else 1.4, c=COL['drone'], zorder=3)[0] for _ in range(self.n)]
        self.zbird = st.scatter(np.zeros(self.nb), np.zeros(self.nb), s=22, c=COL['bird'], zorder=4)
        self.zdot = st.scatter(np.zeros(self.n), np.zeros(self.n), s=36, c=COL['drone'], zorder=5)
        self.cursor = st.axvline(0, c='#999999', lw=1)
        self.frame = fig.add_artist(plt_rect(rect_top, rect_strip))

    def update(self, s):
        r, n = self.rec, self.n
        L = len(r['pos'])
        sc = min(s, L - 1.0)                                            # frozen after the run ends (a crash or the end)
        i = min(int(sc), L - 1)
        pos = lerp_rows(r['pos'], sc)
        bird = lerp_rows(r['bird'], s)
        bird_full = r['bird']
        crashed = r['crashed'][i]
        active = r['guard'][i] & ~crashed
        self.drone.set_offsets(pos[:, :2])
        self.drone.set_facecolors([COL['crash'] if crashed[k] else COL['guard'] if active[k] else COL['drone'] for k in range(n)])
        self.bird.set_offsets(bird[:, :2])
        for k in range(n):
            xs = np.append(r['pos'][:i + 1, k, 0], pos[k, 0]); ys = np.append(r['pos'][:i + 1, k, 1], pos[k, 1])
            self.trail[k].set_data(xs, ys)
            ga = np.append(r['guard'][:i + 1, k], active[k])
            self.trail_g[k].set_data(np.where(ga, xs, np.nan), np.where(ga, ys, np.nan))
        j0 = max(0, int(s) - 40)                                          # about 1.3 s of bird trail
        for k in range(self.nb):
            self.btrail[k].set_data(np.append(bird_full[j0:int(s) + 1, k, 0], bird[k, 0]),
                                    np.append(bird_full[j0:int(s) + 1, k, 1], bird[k, 1]))
        if self.ring is not None:
            tr = np.flatnonzero(r['tracked'][i].any(axis=0))
            self.ring.set_offsets(bird[tr, :2] if len(tr) else np.zeros((0, 2)))
        # nearest drone-bird pair (live drones) and the closest approach so far
        d = np.linalg.norm(pos[:, None, :] - bird[None, :, :], axis=2)
        di, bi = np.unravel_index(int(np.argmin(d)), d.shape)
        dmin, dso_far = float(d[di, bi]), float(r['dist'][:i + 1].min())
        if dmin < 3.0:
            self.link.set_data([pos[di, 0], bird[bi, 0]], [pos[di, 1], bird[bi, 1]])
            self.link.set_color(COL['crash'] if dmin < 0.5 else '#444444')
        else:
            self.link.set_data([], [])
        self.dist_txt.set_text(f'drone-bird distance {dmin:4.1f} m\nclosest so far {dso_far:4.2f} m')
        self.dist_txt.get_bbox_patch().set_edgecolor(COL['crash'] if dmin < 0.5 else '#cccccc')
        nc = int(r['crashed'][i].sum())
        if nc:
            self.xmark.set_data(pos[r['crashed'][i], 0], pos[r['crashed'][i], 1])
            self.crash_txt.set_visible(True)
            self.crash_txt.set_text('CRASH' if nc == 1 else f'CRASH x{nc}')
            self.frame.set_edgecolor(COL['crash'])
        else:
            self.xmark.set_data([], [])
            self.crash_txt.set_visible(False)
            self.frame.set_edgecolor(COL['guard'] if active.any() else '#bbbbbb')
        if self.avoid:
            k = int(active.sum())
            self.guard_txt.set_text('' if not k else ('GUARD ACTIVE' if n == 1 else f'GUARD ACTIVE ({k} of {n} drones)'))
        self.frame.set_linewidth(5 if (nc or active.any()) else 1.5)
        tt = sc * r['dt']
        for k in range(n):
            self.zline[k].set_data(np.append(self.t[:i + 1], tt), np.append(r['pos'][:i + 1, k, 2], pos[k, 2]))
            self.zline[k].set_color(COL['crash'] if crashed[k] else COL['drone'])
        self.zbird.set_offsets(np.c_[np.full(self.nb, s * r['dt']), bird[:, 2]])
        self.zdot.set_offsets(np.c_[np.full(n, tt), pos[:, 2]])
        self.zdot.set_facecolors([COL['crash'] if crashed[k] else COL['guard'] if active[k] else COL['drone'] for k in range(n)])
        self.cursor.set_xdata([s * r['dt']] * 2)


def plt_rect(top, strip):
    from matplotlib.patches import Rectangle
    x0, y0 = strip[0] - 0.045, strip[1] - 0.06
    x1, y1 = top[0] + top[2] + 0.008, top[1] + top[3] + 0.065
    return Rectangle((x0, y0), x1 - x0, y1 - y0, transform=None, fill=False, ec='#bbbbbb', lw=1.5, zorder=0)


def render(a, b, out_path, fps, slow, hold_end, title_extra):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    W, H, dpi = 1280, 720, 100
    fig = plt.figure(figsize=(W / dpi, H / dpi), dpi=dpi, facecolor='white')
    P = np.concatenate([a['pos'].reshape(-1, 3), b['pos'].reshape(-1, 3)])
    pts = np.concatenate([P[:, :2], a['goals'][:, :2]])
    lo, hi = pts.min(axis=0) - 2.2, pts.max(axis=0) + 2.2
    box_w, box_h = 6.0, 4.4                                           # panel box aspect: keep the same data scale in both
    cx, cy = (lo + hi) / 2, (lo + hi) / 2
    span = hi - lo
    scale = max(span[0] / box_w, span[1] / box_h)
    lims = ((cx[0] - scale * box_w / 2, cx[0] + scale * box_w / 2), (cy[1] - scale * box_h / 2, cy[1] + scale * box_h / 2))
    n = max(len(a['pos']), len(b['pos']))
    tmax = n * a['dt']
    zs = np.concatenate([P[:, 2], a['bird'][..., 2].ravel(), b['bird'][..., 2].ravel()])
    zlim = (max(0.0, zs.min() - 0.3), zs.max() + 0.3)
    # figure layout (fractions): two columns, each a top-down box above a height strip
    top_h, strip_h = 0.50, 0.16
    cols = [0.065, 0.565]
    panels = []
    for col, (rec, title, avoid) in zip(cols, ((a, 'Without bird guard', False), (b, 'With bird guard', True))):
        panels.append(Panel(fig, [col, 0.285, 0.41, top_h * 0.95], [col, 0.075, 0.41, strip_h], rec, title, avoid, lims, zlim, tmax))
    fig.text(0.5, 0.975, title_extra, ha='center', va='top', fontsize=13, color='#444444')
    clock = fig.text(0.5, 0.012, '', ha='center', va='bottom', fontsize=12, color='#444444')
    sched, ft, speed, total = time_map(a, b, fps, slow, hold_end)
    while total < 11.0 and slow < 8.0:                            # a clip of 10-20 s: slow the encounter more if the run is short
        slow += 0.5
        sched, ft, speed, total = time_map(a, b, fps, slow, hold_end)
    ffm = shutil.which('ffmpeg')
    if ffm and out_path.endswith('.mp4'):
        proc = subprocess.Popen([ffm, '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgba', '-s', f'{W}x{H}', '-r', str(fps),
                                 '-i', '-', '-an', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '20', '-preset', 'medium', out_path],
                                stdin=subprocess.PIPE)
        sink = proc.stdin
        gif = None
    else:
        from PIL import Image
        out_path = os.path.splitext(out_path)[0] + '.gif'
        proc, sink, gif = None, None, []
    for f, s in enumerate(sched):
        for p in panels:
            p.update(s)
        sp = float(np.interp(s, np.arange(len(speed)), speed))
        clock.set_text(f'sim time {s * a["dt"]:5.2f} s   playback {1.0 / sp:.2f}x' + ('   (slow motion)' if sp > 1.2 else ''))
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())
        if sink is not None:
            sink.write(buf.tobytes())
        else:
            gif.append(Image.fromarray(buf[..., :3]).convert('P', palette=Image.ADAPTIVE))
    if proc is not None:
        sink.close()
        proc.wait()
    else:
        gif[0].save(out_path, save_all=True, append_images=gif[1:], duration=int(1000 / fps), loop=0)
    plt.close(fig)
    return out_path, total, len(sched)


# --------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cell', choices=list(CELLS), required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--config', default=os.path.join(ROOT, eb.CONFIG))
    ap.add_argument('--pp', default=os.path.join(ROOT, 'eval_results', 'birds', 'pp'), help='stage 3a++ run folder (final_config.json, final_* runs)')
    ap.add_argument('--out-dir', default=os.path.join(ROOT, 'eval_results', 'birds', 'clips'))
    ap.add_argument('--seed', type=int, default=None, help='use this episode instead of picking (with --episode)')
    ap.add_argument('--episode', type=int, default=None)
    ap.add_argument('--candidates', type=int, default=6, help='how many candidates from the final runs to fly and compare')
    ap.add_argument('--fps', type=int, default=30)
    ap.add_argument('--slow', type=float, default=2.0, help='slow-motion factor near the encounter (2 = 0.5x)')
    ap.add_argument('--hold-end', type=float, default=2.5, help='seconds the last frame stays on screen')
    ap.add_argument('--name', default=None)
    args = ap.parse_args()
    if (args.seed is None) != (args.episode is None):
        ap.error('--seed and --episode go together')
    if args.seed is not None and args.seed not in SEEDS_OK:
        ap.error('seeds 20-24 only')
    torch.set_num_threads(2)
    t0 = time.time()
    with open(os.path.join(args.pp, 'final_config.json')) as f:
        final_cfg = json.load(f)
    fliers = [Flier(args.cell, False, final_cfg, args), Flier(args.cell, True, final_cfg, args)]
    chosen = pick(args.cell, fliers, args, args.pp)
    t_fly = time.time() - t0
    if chosen is None:
        print('no usable example among the candidates')
        return 1
    seed, ep, a, b = chosen
    same_birds = float(np.abs(a['bird'][:min(len(a['bird']), len(b['bird']))] - b['bird'][:min(len(a['bird']), len(b['bird']))]).max())
    same_goals = float(np.abs(a['goals'] - b['goals']).max())
    print(f'chosen: cell {args.cell}, seed {seed}, episode {ep}; bird paths max difference between the runs {same_birds:.2e} m, '
          f'goals {same_goals:.2e} m', flush=True)
    os.makedirs(args.out_dir, exist_ok=True)
    name = args.name or f'bird_clip_{args.cell}_s{seed}_e{ep}'
    n_agents, n_birds = CELLS[args.cell][3], CELLS[args.cell][1]
    sub = (f'{"1 drone" if n_agents == 1 else f"{n_agents} drones (fair crowd)"}, {n_birds} birds | seed {seed}, episode {ep} | '
           f'same seed, bird paths and noise in both runs')
    path, total, nframes = render(a, b, os.path.join(args.out_dir, name + '.mp4'), args.fps, args.slow, args.hold_end, sub)
    t_all = time.time() - t0
    meta = {'cell': args.cell, 'seed': seed, 'episode': ep, 'clip': path, 'clip_seconds': round(total, 1), 'frames': nframes,
            'without_guard': {'outcome': a['outcome'], 'crashed_drones': int(a['crashed'][-1].sum()), 'closest_m': round(a['min_dist'], 3), 'steps': len(a['pos'])},
            'with_guard': {'outcome': b['outcome'], 'crashed_drones': int(b['crashed'][-1].sum()), 'closest_m': round(b['min_dist'], 3), 'steps': len(b['pos']),
                           'guard_active_steps': int(b['guard'].sum())},
            'bird_path_max_diff_m': same_birds, 'final_config': final_cfg,
            'seconds_flying_and_picking': round(t_fly, 1), 'seconds_total': round(t_all, 1)}
    with open(os.path.splitext(path)[0] + '.json', 'w') as f:
        json.dump(meta, f, indent=1)
    print(json.dumps(meta, indent=1))
    print(f'wrote {path}  ({total:.1f} s, {nframes} frames); flying+picking {t_fly:.0f} s, total {t_all:.0f} s')
    return 0


if __name__ == '__main__':
    sys.exit(main())
