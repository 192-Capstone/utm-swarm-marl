"""
render_scenario.py
------------------
A video of a recorded scenario run (scripts/run_scenario.py -> utm/flight_log.py), from the
recording only: nothing is flown again.

    python scripts/render_scenario.py eval_results/scenarios/ecity_demo.json.gz --view top
    python scripts/render_scenario.py eval_results/scenarios/ecity_demo.json.gz --view 3d

    top   top-down map (matplotlib): buildings, drones coloured by operator with short trails,
          intruders (black) and the emergency flight (red), the no-fly zone while it is active,
          high-risk alerts as a ring, events listed as they happen
    3d    the city rebuilt in an off-screen PyBullet world, the drones placed where the recording
          has them, filmed by four cameras at once (overhead, 3/4 view, low view across the zone,
          chase camera on the emergency flight)

Frames are piped straight to ffmpeg (on PATH), so memory stays small. --speed: video seconds
per simulated second is 1 / speed (default 4x).
"""

import argparse
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, ROOT)

from utm import flight_log  # noqa: E402

OPERATOR_COLOURS = ['#378ADD', '#1D9E75', '#BA7517', '#7F77DD', '#D4537E', '#5F5E5A']
INTRUDER, EMERGENCY, ZONE = '#2C2C2A', '#E24B4A', '#E24B4A'


def ffmpeg(path, w, h, fps):
    return subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                             '-s', f'{w}x{h}', '-r', str(fps), '-i', '-', '-pix_fmt', 'yuv420p',
                             '-vcodec', 'libx264', '-crf', '23', path], stdin=subprocess.PIPE)


def drone_colours(log):
    ops = sorted({d['operator'] for d in log['drones'] if d['kind'] == 'flight'})
    out = []
    for d in log['drones']:
        if d['kind'] == 'intruder':
            out.append(INTRUDER)
        elif d['kind'] == 'emergency':
            out.append(EMERGENCY)
        else:
            out.append(OPERATOR_COLOURS[ops.index(d['operator']) % len(OPERATOR_COLOURS)])
    return out, ops


def zone_active(z, t):
    if z['t_on'] is None or t < z['t_on']:
        return False
    return not z['active'] or any(a <= t < b for a, b in z['active'])


STATE_WORDS = {'late': 'running late', 'early': 'running early', 'off_route': 'off its route',
               'intrusion': "in another flight's air", 'overstay': 'still over its pad', 'unknown': 'unplanned drone in the air'}


def reason(why):
    """An E4 re-plan reason in plain words (utm/contingency.py)."""
    w = why.replace(' -> cancelled', '')
    if w.startswith('zone'):
        return 'the no-fly zone'
    if w.startswith('intruder'):
        return 'the intruder'
    if w.startswith('displaced by'):
        return 'the emergency flight'
    return STATE_WORDS.get(w.split()[0], w)


def short(e):
    """A few words over the drone for an E4 re-plan (the 3D labels)."""
    if e['why'] == 'emergency':
        return 'cleared for take-off' if e['status'] == 'approved' else 'emergency refused'
    if e['why'].endswith('-> cancelled'):
        return 'cancelled'
    if e['status'] != 'approved':
        return 'keeps its route'
    if e.get('reason') == 'return to base':
        return 'back to its pad'
    return {'the no-fly zone': 'avoids the zone', 'the intruder': 'avoids the intruder',
            'the emergency flight': 'makes way'}.get(reason(e['why']),
                                                     'makes room' if e['why'].startswith('intrusion by') else 're-planned')


def describe(e):
    """One line of the event list, in plain words."""
    if e['type'] == 'zone_on':
        return 'No-fly zone closes the area'
    if e['type'] == 'replan':
        f, r = e['flight'], reason(e['why'])
        if e['why'] == 'emergency':
            return 'Emergency flight cleared for take-off' if e['status'] == 'approved' else 'Emergency flight refused'
        if e['why'].endswith('-> cancelled'):
            return f"{f} can't fly safely ({r}): cancelled"
        if e['status'] != 'approved':
            if r.startswith('the '):
                return f"{f}: no free route around {r} yet, keeps its route"
            return f"{f} {r}: no free alternative, keeps its route"
        if r == 'the no-fly zone':
            return f"{f} re-routed around the no-fly zone"
        if r == 'the intruder':
            return f"{f} re-booked to stay clear of the intruder"
        if r == 'the emergency flight':
            return f"{f} moves aside for the emergency flight"
        if e['why'].startswith('intrusion by '):
            return f"{f} re-planned: {e['why'].split()[-1]} drifted into its air"
        return f"{f} re-planned ({r})"
    if e['type'] == 'alert' and e['risk'] == 'high':
        return f"Alert: {e['flight']} {STATE_WORDS.get(e['state'], e['state'])}"
    return None


def caption(log):
    """'UTM demo: 2 delivery operators, 1 intruder, a no-fly zone, an emergency flight'."""
    kinds = [d['kind'] for d in log['drones']]
    ops = len({d['operator'] for d in log['drones'] if d['kind'] == 'flight'})
    parts = [f"{ops} delivery operator{'s' * (ops != 1)}"]
    if kinds.count('intruder'):
        parts.append(f"{kinds.count('intruder')} intruder{'s' * (kinds.count('intruder') != 1)}")
    if log['zones']:
        parts.append('a no-fly zone' if len(log['zones']) == 1 else f"{len(log['zones'])} no-fly zones")
    if kinds.count('emergency'):
        parts.append('an emergency flight' if kinds.count('emergency') == 1 else f"{kinds.count('emergency')} emergency flights")
    return 'UTM demo: ' + ', '.join(parts)


def render_top(log, out, fps, w=1280, h=720):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon, Rectangle

    cols, ops = drone_colours(log)
    T = np.array(log['frames']['t'])
    P = np.array(log['frames']['pos'])
    S = np.array(log['frames']['status'])
    ticks = {round(k['t']): k for k in log['ticks']}
    events = [(e['t'], describe(e)) for e in log['events'] if describe(e)]
    (x0, x1), (y0, y1) = log['map']['bounds']

    fig = plt.figure(figsize=(w / 100, h / 100), dpi=100)
    ax = fig.add_axes([0.02, 0.04, 0.56, 0.92])
    side = fig.add_axes([0.61, 0.04, 0.37, 0.92])
    side.axis('off')
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor('#F7F6F2')
    for ob in log['map']['obstacles']:
        (cx, cy), (hx, hy) = ob['center'], ob['half']
        ax.add_patch(Rectangle((cx - hx, cy - hy), 2 * hx, 2 * hy, angle=np.degrees(ob.get('yaw', 0.0)),
                               rotation_point='center', color='#B4B2A9', lw=0))
    zpatch = [ax.add_patch(Polygon(z['polygon'], closed=True, fc=ZONE, ec=ZONE, alpha=0.0, lw=1.5))
              for z in log['zones']]
    pads = np.array([d['goal'] for d in log['drones']])
    ax.scatter(pads[:, 0], pads[:, 1], s=30, marker='s', facecolors='none', edgecolors='#888780', lw=0.6)
    trails = [ax.plot([], [], '-', color=c, lw=1.0, alpha=0.5)[0] for c in cols]
    dots = ax.scatter(P[0, :, 0], P[0, :, 1], s=40, c=cols, zorder=5)
    rings = ax.scatter([], [], s=260, facecolors='none', edgecolors=EMERGENCY, lw=1.6, zorder=6)
    title = ax.set_title('', loc='left', fontsize=12)
    side.text(0, 1.0, caption(log), fontsize=12, va='top', weight='bold', wrap=True)
    for k, o in enumerate(ops):
        side.add_patch(Rectangle((0, 0.9 - 0.04 * k), 0.03, 0.025, color=OPERATOR_COLOURS[k], transform=side.transAxes))
        side.text(0.05, 0.9 - 0.04 * k, f'operator {o}', fontsize=10, va='bottom')
    k0 = 0.9 - 0.04 * len(ops)
    for c, label in ((INTRUDER, 'intruder (no flight plan)'), (EMERGENCY, 'emergency flight / high-risk alert')):
        side.add_patch(Rectangle((0, k0), 0.03, 0.025, color=c, transform=side.transAxes))
        side.text(0.05, k0, label, fontsize=10, va='bottom')
        k0 -= 0.04
    stats = side.text(0, k0 - 0.03, '', fontsize=11, va='top', family='monospace')
    log_txt = side.text(0, k0 - 0.20, '', fontsize=9.5, va='top')

    ff = ffmpeg(out, w, h, fps)
    for f in range(len(T)):
        t = T[f]
        pos, st = P[f], S[f]
        air = st == 1
        dots.set_offsets(pos[:, :2])
        dots.set_sizes(np.where(air, 40, 12))
        lo = max(0, f - 16)
        for i, tr in enumerate(trails):              # only while flying (parked drones sit off-map)
            fl = S[lo:f + 1, i] == 1
            s0 = lo + (len(fl) - np.argmin(fl[::-1]) if not fl.all() else 0)
            tr.set_data(*((P[s0:f + 1, i, 0], P[s0:f + 1, i, 1]) if air[i] else ([], [])))
        for patch, z in zip(zpatch, log['zones']):
            patch.set_alpha(0.25 if zone_active(z, t) else 0.0)
        tk = ticks.get(int(t))
        hot = [i for i, d in enumerate(log['drones']) if tk and tk.get('risk', {}).get(d['id'], ['', ''])[1] == 'high' and air[i]]
        rings.set_offsets(pos[hot, :2] if hot else np.empty((0, 2)))
        title.set_text(f't = {t:5.1f} s')
        stats.set_text(f"flying    {int(air.sum()):3d}\nwaiting   {int((st == 0).sum()):3d}\n"
                       f"landed    {int((st == 2).sum()):3d}\ncrashed   {int((st == 3).sum()):3d}")
        log_txt.set_text('\n'.join(f'{te:5.1f} s  {msg}' for te, msg in [x for x in events if x[0] <= t][-12:]))
        fig.canvas.draw()
        ff.stdin.write(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].tobytes())
    ff.stdin.close()
    ff.wait()
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('recording')
    ap.add_argument('--view', choices=['top', '3d'], default='top')
    ap.add_argument('--speed', type=float, default=4.0, help='x real time')
    ap.add_argument('--out', default=None)
    ap.add_argument('--tiny', action='store_true', help='3d: software renderer, no window (plainer)')
    a = ap.parse_args()
    log = flight_log.read(a.recording)
    fps = max(1, int(round(a.speed / log['sim']['frame_dt'])))
    out = a.out or a.recording.replace('.json.gz', f'_{a.view}.mp4')
    if a.view == 'top':
        render_top(log, out, fps)
    else:
        from render_scenario_3d import render_3d
        render_3d(log, out, fps, gui=not a.tiny)
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
