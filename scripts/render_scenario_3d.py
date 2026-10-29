"""
render_scenario_3d.py
---------------------
The 3D view for scripts/render_scenario.py --view 3d: the recorded run replayed in a PyBullet
world, filmed by four cameras at once (2 x 2):

    overhead                    |  3/4 view looking down into the streets
    ----------------------------+-------------------------------------------
    close-up on the no-fly zone |  chase camera on the emergency flight

Rendered with PyBullet's OpenGL renderer (a PyBullet window opens while it runs; leave it open):
lit and shaded quadcopters (body, arms, rotors) coloured by operator, the emergency flight
larger and red, intruders dark; helipad-style pads (depots larger); the no-fly zone as a
translucent red column while it is active. Trails and short labels (EMERGENCY, intruder, what E4
just did to a drone) are drawn on top, projected with the same camera. --tiny uses the software
renderer instead (no window, plainer). Nothing is simulated: positions come from the recording.
"""

import numpy as np
import pybullet as p
from PIL import Image, ImageDraw, ImageFont

from render_scenario import EMERGENCY, INTRUDER, ZONE, caption, drone_colours, ffmpeg, short, zone_active

W, H = 640, 360                                   # one camera; the video is 2 x 2 of these
BAR = 40                                          # the caption bar on top
TRAIL = 16                                        # frames of trail (4 s at 0.25 s per frame)
LABEL_S = 4.0                                     # s an E4 label stays over a drone


def _rgba(hex_colour, a=1.0):
    h = hex_colour.lstrip('#')
    return [int(h[k:k + 2], 16) / 255 for k in (0, 2, 4)] + [a]


def _rgb255(hex_colour):
    h = hex_colour.lstrip('#')
    return tuple(int(h[k:k + 2], 16) for k in (0, 2, 4))


def quad_shape(c, colour, scale=1.0):
    """A small quadcopter: body, two crossed arms, four rotor discs."""
    s = 0.6 * scale                               # drawn a few times real size, to be visible at 50 m
    shapes, half, rad, length, frames, colours = [], [], [], [], [], []
    shapes += [p.GEOM_BOX]; half += [[0.14 * s, 0.14 * s, 0.05 * s]]; rad += [0]; length += [0]
    frames += [[0, 0, 0]]; colours += [_rgba(colour)]
    for yaw in (np.pi / 4, -np.pi / 4):           # the two arms (boxes along x, rotated)
        shapes += [p.GEOM_BOX]; half += [[0.42 * s, 0.03 * s, 0.02 * s]]; rad += [0]; length += [0]
        frames += [[0, 0, 0]]; colours += [[0.2, 0.2, 0.2, 1]]
    for dx, dy in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
        shapes += [p.GEOM_CYLINDER]; half += [[0, 0, 0]]; rad += [0.16 * s]; length += [0.015 * s]
        frames += [[0.3 * s * dx, 0.3 * s * dy, 0.04 * s]]; colours += [[0.85, 0.85, 0.85, 0.9]]
    orns = [[0, 0, 0, 1], p.getQuaternionFromEuler([0, 0, np.pi / 4]), p.getQuaternionFromEuler([0, 0, -np.pi / 4])]
    orns += [[0, 0, 0, 1]] * 4
    # createVisualShapeArray takes one colour for all parts; make one shape per part instead
    parts = []
    for k in range(len(shapes)):
        kw = dict(shapeType=shapes[k], rgbaColor=colours[k], visualFramePosition=frames[k],
                  visualFrameOrientation=orns[k], physicsClientId=c)
        if shapes[k] == p.GEOM_BOX:
            kw['halfExtents'] = half[k]
        else:
            kw['radius'], kw['length'] = rad[k], length[k]
        parts.append(p.createVisualShape(**kw))
    return parts


class Scene:
    def __init__(self, log, gui=True):
        self.gui = gui
        self.c = p.connect(p.GUI if gui else p.DIRECT, options='--width=640 --height=360' if gui else '')
        if gui:
            for flag in (p.COV_ENABLE_GUI, p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, p.COV_ENABLE_DEPTH_BUFFER_PREVIEW,
                         p.COV_ENABLE_RGB_BUFFER_PREVIEW):
                p.configureDebugVisualizer(flag, 0, physicsClientId=self.c)
            p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1, physicsClientId=self.c)
        (x0, x1), (y0, y1) = log['map']['bounds']
        self._body([p.createVisualShape(p.GEOM_BOX, halfExtents=[(x1 - x0) / 2 + 5, (y1 - y0) / 2 + 5, 0.01],
                                        rgbaColor=[0.80, 0.84, 0.78, 1], physicsClientId=self.c)],
                   [(x0 + x1) / 2, (y0 + y1) / 2, -0.01])
        for ob in log['map']['obstacles']:
            (cx, cy), (hx, hy), h = ob['center'], ob['half'], ob['height']
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[hx, hy, h / 2], rgbaColor=[0.86, 0.84, 0.80, 1],
                                      physicsClientId=self.c)
            self._body([vis], [cx, cy, h / 2], p.getQuaternionFromEuler([0, 0, ob.get('yaw', 0.0)]))
        starts = [tuple(d['start']) for d in log['drones'] if d['kind'] == 'flight']
        for q in {tuple(d[k]) for d in log['drones'] for k in ('start', 'goal')}:
            depot = starts.count(q) >= 2
            hospital = any(d['kind'] == 'emergency' and tuple(d['start']) == q for d in log['drones'])
            r = 1.1 if depot else 0.7
            base = [0.95, 0.88, 0.88, 1] if hospital else [0.97, 0.97, 0.95, 1]
            parts = [p.createVisualShape(p.GEOM_CYLINDER, radius=r, length=0.03, rgbaColor=base, physicsClientId=self.c),
                     p.createVisualShape(p.GEOM_CYLINDER, radius=r * 0.82, length=0.035,
                                         rgbaColor=[0.75, 0.2, 0.2, 1] if hospital else [0.35, 0.35, 0.35, 1],
                                         physicsClientId=self.c),
                     p.createVisualShape(p.GEOM_CYLINDER, radius=r * 0.72, length=0.04, rgbaColor=base, physicsClientId=self.c)]
            for part in parts:
                self._body([part], [q[0], q[1], 0.02])
            hw = r * 0.38                          # the "H" (a cross for the hospital)
            bars = ([[0.06, hw, 0.01], [0.06, hw, 0.01], [hw, 0.05, 0.01]], [[-hw * 0.7, 0], [hw * 0.7, 0], [0, 0]])
            if hospital:
                bars = ([[0.1, hw, 0.01], [hw, 0.1, 0.01]], [[0, 0], [0, 0]])
            for he, (dx, dy) in zip(*bars):
                v = p.createVisualShape(p.GEOM_BOX, halfExtents=he, rgbaColor=[0.75, 0.2, 0.2, 1] if hospital
                                        else [0.35, 0.35, 0.35, 1], physicsClientId=self.c)
                self._body([v], [q[0] + dx, q[1] + dy, 0.05])
        cols, _ = drone_colours(log)
        self.drones = []
        for d, col in zip(log['drones'], cols):
            scale = (1.6 if d['kind'] == 'emergency' else 1.0) * (1.0 if gui else 1.6)
            self.drones.append([self._body([v], [0, 0, -20]) for v in quad_shape(self.c, col, scale)])
        top = max(log['map']['layers']) + 1.5
        self.zones = []
        for z in (log['zones'] if gui else []):           # software renderer: the overlay draws it
            xy = np.array(z['polygon'])
            (zx0, zy0), (zx1, zy1) = xy.min(0), xy.max(0)
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[(zx1 - zx0) / 2, (zy1 - zy0) / 2, top / 2],
                                      rgbaColor=_rgba(ZONE, 0.28), physicsClientId=self.c)
            self.zones.append((z, self._body([vis], [0, 0, -50]), [(zx0 + zx1) / 2, (zy0 + zy1) / 2, top / 2]))

    def _body(self, vis, pos, orn=(0, 0, 0, 1)):
        return p.createMultiBody(0, baseVisualShapeIndex=vis[0], basePosition=pos, baseOrientation=orn,
                                 physicsClientId=self.c)

    def update(self, pos, status, t):
        for parts, q, s in zip(self.drones, pos, status):
            at = list(q) if s == 1 else [q[0], q[1], -20]
            for b in parts:
                p.resetBasePositionAndOrientation(b, at, [0, 0, 0, 1], physicsClientId=self.c)
        for z, b, centre in self.zones:
            p.resetBasePositionAndOrientation(b, centre if zone_active(z, t) else [0, 0, -50], [0, 0, 0, 1],
                                              physicsClientId=self.c)

    def shoot(self, target, dist, yaw, pitch):
        view = p.computeViewMatrixFromYawPitchRoll(target, dist, yaw, pitch, 0, 2, physicsClientId=self.c)
        proj = p.computeProjectionMatrixFOV(50, W / H, 0.1, 300, physicsClientId=self.c)
        renderer = p.ER_BULLET_HARDWARE_OPENGL if self.gui else p.ER_TINY_RENDERER
        img = p.getCameraImage(W, H, view, proj, renderer=renderer, shadow=1, lightDirection=[0.4, 0.3, 1.0],
                               physicsClientId=self.c)[2]
        rgb = np.reshape(np.asarray(img, dtype=np.uint8), (H, W, 4))[:, :, :3]
        return Image.fromarray(rgb), np.array(view).reshape(4, 4).T, np.array(proj).reshape(4, 4).T


def to_pixels(points, view, proj):
    """World points (k, 3) -> pixel coords (k, 2) and an in-front-of-the-camera mask."""
    clip = (proj @ view @ np.c_[points, np.ones(len(points))].T).T
    ok = clip[:, 3] > 1e-6
    ndc = clip[:, :3] / np.where(ok, clip[:, 3], 1.0)[:, None]
    return np.c_[(ndc[:, 0] + 1) * 0.5 * W, (1 - ndc[:, 1]) * 0.5 * H], ok


def labels_at(log, t):
    """Drone id -> a short label for what is happening to it now."""
    out = {}
    for d in log['drones']:
        if d['kind'] == 'emergency':
            out[d['id']] = 'EMERGENCY'
        elif d['kind'] == 'intruder':
            out[d['id']] = 'intruder (no plan)'
    for e in log['events']:
        if e['type'] == 'replan' and e['t'] <= t <= e['t'] + LABEL_S and e['flight'] in {d['id'] for d in log['drones']}:
            if e['why'] == 'emergency' or e['flight'] in out:
                continue
            out[e['flight']] = short(e)
    return out


def draw_zone(d, z, view, proj, top):
    """A translucent red column: filled footprint, outline, vertical edges and the top rim."""
    xy = np.array(z['polygon'], dtype=float)
    low, ok1 = to_pixels(np.c_[xy, np.zeros(len(xy))], view, proj)
    high, ok2 = to_pixels(np.c_[xy, np.full(len(xy), top)], view, proj)
    if not (ok1.all() and ok2.all()):
        return
    r, g, b = _rgb255(ZONE)
    n = len(xy)
    faces = [[tuple(low[k]), tuple(low[(k + 1) % n]), tuple(high[(k + 1) % n]), tuple(high[k])] for k in range(n)]
    for face in faces:
        d.polygon(face, fill=(r, g, b, 28))
    d.polygon([tuple(q) for q in low], fill=(r, g, b, 90))
    for k in range(n):
        d.line([tuple(low[k]), tuple(high[k])], fill=(r, g, b, 140), width=1)
    d.line([tuple(q) for q in high] + [tuple(high[0])], fill=(r, g, b, 140), width=1)
    d.line([tuple(q) for q in low] + [tuple(low[0])], fill=(r, g, b, 255), width=3)


def overlay(img, view, proj, f, P, S, log, cols, hot, labels, font, zones=(), t=0.0, top=5.5):
    layer = Image.new('RGBA', img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    for z in zones:
        if zone_active(z, t):
            draw_zone(d, z, view, proj, top)
    lo = max(0, f - TRAIL)
    for i, (dr, col) in enumerate(zip(log['drones'], cols)):
        if S[f][i] != 1:
            continue
        flying = S[lo:f + 1, i] == 1                   # trail: only while flying (parked drones sit off-map)
        start = lo + (len(flying) - np.argmin(flying[::-1]) if not flying.all() else 0)
        px, ok = to_pixels(P[start:f + 1, i], view, proj)
        rgb = _rgb255(col)
        if ok.all() and len(px) > 1:
            d.line([tuple(q) for q in px], fill=rgb + (170,), width=3 if dr['kind'] == 'emergency' else 2)
        (x, y), seen = px[-1], ok[-1]
        if not (seen and -40 < x < W + 40 and -40 < y < H + 40):
            continue
        if i in hot:
            d.ellipse([x - 16, y - 16, x + 16, y + 16], outline=_rgb255(EMERGENCY) + (255,), width=3)
        text = labels.get(dr['id'])
        if text:
            tw = d.textlength(text, font=font)
            bg = _rgb255(EMERGENCY) if text == 'EMERGENCY' else _rgb255(INTRUDER) if 'intruder' in text else (255, 255, 255)
            fg = (255, 255, 255) if text == 'EMERGENCY' or 'intruder' in text else (40, 40, 40)
            d.rounded_rectangle([x + 10, y - 26, x + 18 + tw, y - 8], radius=4, fill=bg + (230,))
            d.text((x + 14, y - 25), text, fill=fg, font=font)
    return Image.alpha_composite(img.convert('RGBA'), layer).convert('RGB')


def render_3d(log, out, fps, gui=True):
    scene = Scene(log, gui)
    cols, _ = drone_colours(log)
    T = np.array(log['frames']['t'])
    P = np.array(log['frames']['pos'])
    S = np.array(log['frames']['status'])
    ticks = {round(k['t']): k for k in log['ticks']}
    (x0, x1), (y0, y1) = log['map']['bounds']
    mid = [(x0 + x1) / 2, (y0 + y1) / 2, 0.0]
    zc = list(np.array(log['zones'][0]['polygon']).mean(0)) + [1.5] if log['zones'] else mid
    em = [i for i, d in enumerate(log['drones']) if d['kind'] == 'emergency']
    try:
        font, title_font = ImageFont.truetype('arial.ttf', 13), ImageFont.truetype('arialbd.ttf', 15)
    except OSError:
        font = title_font = ImageFont.load_default()
    ff = ffmpeg(out, 2 * W, 2 * H + BAR, fps)
    try:
        bar_font = ImageFont.truetype('arialbd.ttf', 20)
    except OSError:
        bar_font = title_font
    text = caption(log)
    for f in range(len(T)):
        t = T[f]
        scene.update(P[f], S[f], t)
        tk = ticks.get(int(t)) or {}
        hot = {i for i, d in enumerate(log['drones']) if tk.get('risk', {}).get(d['id'], ['', ''])[1] == 'high'}
        labels = labels_at(log, t)
        pitch = -45
        if em and S[f][em[0]] == 1:
            chase, name = list(P[f][em[0]]), 'chase: emergency flight'
        elif em and S[f][em[0]] == 2:
            chase, name, pitch = [*log['drones'][em[0]]['goal'], 1.0], 'emergency flight: landed', -75
        elif em:
            chase, name, pitch = [*log['drones'][em[0]]['start'], 1.0], 'hospital pad (emergency not filed yet)', -75
        else:
            chase, name = mid, 'close-up'
        cams = [(mid, 58, 0, -89.9, 'overhead'),
                (mid, 46, 35, -55, '3/4 view'),
                (zc, 18, -60, -45, 'close-up: no-fly zone'),
                (chase, 9, 30, pitch, name)]
        frame = Image.new('RGB', (2 * W, 2 * H + BAR), (34, 34, 32))
        ImageDraw.Draw(frame).text((16, 9), text, fill=(250, 250, 248), font=bar_font)
        for k, (target, dist, yaw, ptc, cam_name) in enumerate(cams):
            img, view, proj = scene.shoot(target, dist, yaw, ptc)
            img = overlay(img, view, proj, f, P, S, log, cols, hot, labels, font,
                          () if gui else log['zones'], t, max(log['map']['layers']) + 1.5)
            dr = ImageDraw.Draw(img)
            dr.rectangle([0, 0, W, 24], fill=(250, 250, 248))
            dr.text((10, 4), cam_name, fill=(30, 30, 30), font=title_font)
            dr.text((W - 90, 5), f't = {t:5.1f} s', fill=(80, 80, 80), font=font)
            frame.paste(img, ((k % 2) * W, BAR + (k // 2) * H))
        ff.stdin.write(frame.tobytes())
    ff.stdin.close()
    ff.wait()
    p.disconnect(scene.c)
