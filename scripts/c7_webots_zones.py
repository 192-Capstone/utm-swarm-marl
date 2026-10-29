"""
c7_webots_zones.py
------------------
C7: D2's no-fly zones drawn in the Webots L2 multi world (envs/webots_zones.py, webots.draw_zones).
Every case runs in its OWN subprocess (one Webots connection per Python process), one after the
other. Map: D1 hard city, seed 20 (as C5's speed case), sample_flights spawns, route following,
the C5 scripted carrot chaser.

Cases
  track     Two 10 s episodes (the second reuses the layout and the zones). Records every drone's
            true position, velocity and orientation after every control step (float64) and their
            sha256. --save also writes the scene (Supervisor.worldSave) and the exportString of
            OBSTACLES and DRONES after the first reset and at the end. Options:
              --zones     draw the hard-city zone set (webots.draw_zones + adapter.set_zones)
              --render    webots.render (3D view drawn live)
              --radio     webots.radio.enabled (E2): collects adapter.reports() after every step;
                          counts per episode and drone, seq, rejected, sha256 of every report
            Used for: check 0 (baseline, before any C7 code), A (flag off == baseline, byte for
            byte) and C (tracks with zones == without).
  drawn     Check B. One drone, the zone file's own map, webots.draw_zones from the config
            (--file, --map: zones_file / zones_map), rendering on. Reads every zone node back from
            Webots (Cylinder translation / radius / height, IndexedFaceSet points and caps, outline
            height) and compares with the GeoJSON coordinates converted straight from the file;
            no Solid / boundingObject / Physics under ZONES; LiDAR layout unchanged. Screenshots
            (top-down and oblique) next to the json.
  visibility Check B, colour independent. Empty map, top-down camera over (0, 3) from 30 m, rendering
            on; screenshots at episode times t = 1, 3, 5 s. Two runs, same camera and times:
              --tag B_vis_ref           no zones drawn (webots.draw_zones off): the reference
              --tag B_vis --zones       switching zone: red square (-3..3, -3..3), active [2, 4) s;
                                        permanent zone: amber circle, centre (0, 8), radius 2.5
  viscompare --tag B_vis: compares the two runs over each zone's pixel region: the zone's footprint
            shrunk inward (square |x|, |y| <= 2.4 m; circle r <= 2.0 m), projected straight down
            (pinhole, fieldOfView 0.785 across the image width), so every pixel in it looks through
            the zone's volume and none through its outline.
            Pixel difference d = max over R, G, B of |a - b| (0..255). Over a region:
              SAME(A, B)  at most 1 % of the pixels have d > 8
              DIFF(A, B)  at least 90 % of the pixels have d > 30
            PASS when all of:
              both runs took their shots at the same control steps (t = 1, 3, 5 s)
              switching:  SAME(zones t1, ref t1), DIFF(zones t3, ref t3), SAME(zones t5, ref t5)
              permanent:  SAME(zones t1, zones t3), SAME(zones t3, zones t5), SAME(zones t1, zones t5),
                          DIFF(zones t, ref t) for t = 1, 3, 5
            Reports the fractions, max d and whether regions are exactly equal.
  speed     Check D. C5's speed case (hard city, seed 20, sample_flights spawns, route following):
            wall s per sim s for (a) the adapter alone, zero command, 10 s and (b) SwarmEnv + the
            chaser, 10 s. --render, --zones as above.

Usage (repo root, one at a time, low priority):
  DISPLAY=:1 OMP_NUM_THREADS=4 nice -n 19 python scripts/c7_webots_zones.py --worker track --tag baseline --save
Writes logs/webots/c7/<tag>.json / .npz (+ saved worlds); Webots logs next to them.
"""
import argparse
import hashlib
import json
import math
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

from c5_webots_city import MAP_SEED, chase, load_cfg     # noqa: E402

OUT = os.path.join(ROOT, 'logs', 'webots', 'c7')
EP_S = 10.0


def hard_city():
    from envs.city_generator import generate_city, sample_flights
    from envs.route_planner import plan, simplify
    return generate_city(np.random.RandomState(MAP_SEED), 'hard'), sample_flights, plan, simplify


def episode(n):
    city, sample_flights, plan, simplify = hard_city()
    sp, gl = sample_flights(np.random.RandomState(MAP_SEED), city, n)
    routes = {}
    for k in sp:
        p = plan(city['grid'], sp[k][:2], gl[k][:2])
        routes[k] = simplify(city['grid'], p) if p is not None else [tuple(sp[k][:2]), tuple(gl[k][:2])]
    return city, sp, gl, routes


def city_zones(city):
    """The hard-city zone set: blocks covering ~15 % of the map (red, some permanent, some with
    windows inside the 10 s episodes, so they switch mid-flight), one amber block with a height
    band, one red circle and one amber circle with a ceiling."""
    import dataclasses
    from utm.airspace_zones import Zone, city_blocks, random_block_zones
    sx, sy = city['size']
    zones = random_block_zones(city_blocks(city), sx * sy, 0.15, np.random.RandomState(7))
    wins = [[], [[2.0, 6.0]], [[5.0, 8.0], [9.0, 30.0]], [], [[0.0, 3.0]]]
    zones = [dataclasses.replace(z, active=wins[k % len(wins)]) for k, z in enumerate(zones)]
    x0, x1, y0, y1 = city_blocks(city)[12]
    zones.append(Zone('amber_block', 'amber block', 'authorisation', lower_m=1.5, upper_m=4.0,
                      polygon=[(x0, y0), (x1, y0), (x1, y1), (x0, y1)]))
    zones.append(Zone('red_circle', 'red circle', circle=((0.0, 0.0), 3.0), active=[[3.0, 7.0]]))
    zones.append(Zone('amber_circle', 'amber circle', 'authorisation', upper_m=3.0, circle=((15.0, -15.0), 2.5)))
    return zones


def sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


# ------------------------------------------------------------------ track (0, A, C)
def case_track(a):
    import torch
    from envs.swarm_env import SwarmEnv
    n = a.n
    city, sp, gl, routes = episode(n)
    cfg = load_cfg(n, sim='webots_l2_multi')
    cfg['webots'] = {'log': os.path.join(OUT, f'{a.tag}.webots.log')}
    if a.render:
        cfg['webots']['render'] = True
    if a.zones:
        cfg['webots']['draw_zones'] = True
    if a.radio:
        cfg['webots']['radio'] = {'enabled': True}
    cfg['environment']['route_follow'] = {'enabled': True}
    t0 = time.time()
    env = SwarmEnv(cfg, device=torch.device('cpu'))
    startup = time.time() - t0
    ad = env.adapter
    if a.zones:
        ad.set_zones(city_zones(city))
    steps = int(round(EP_S / ad.control_dt))
    pos, vel, orn, saved, resets = [], [], [], {}, []
    radio = []                                                 # (episode, report fields) in arrival order

    def save(label):
        if not a.save:
            return
        path = os.path.join(OUT, f'{a.tag}.{label}.wbt')
        ad.sup.worldSave(path)
        saved[label] = {'world': path,
                        'obstacles': ad.sup.getFromDef('OBSTACLES').exportString(),
                        'drones': ad.sup.getFromDef('DRONES').exportString()}

    for ep in range(2):
        t0 = time.time()
        env.start_episode(sp, gl, obstacles=city['layout'], routes=routes)
        resets.append(time.time() - t0)
        if ep == 0:
            save('reset1')
        for _ in range(steps):
            env.step(chase(env))
            pos.append(ad._pos.copy()); vel.append(ad._vel.copy()); orn.append(ad._orn.copy())
            if a.radio:
                radio += [(ep + 1, r.id, r.seq, float(r.t_sent), float(r.t_received), np.asarray(r.pos).tolist(),
                           np.asarray(r.vel).tolist()) for r in ad.reports()]
    save('end')
    stats = {'layout_builds': ad.layout_builds}
    if a.radio:
        per = {}
        for ep, rid, seq, *_ in radio:
            per.setdefault(f'ep{ep}', {}).setdefault(rid, []).append(seq)
        stats['radio'] = {'reports': len(radio), 'rejected': ad.radio_rejected,
                          'counts': {e: {d: len(q) for d, q in sorted(v.items())} for e, v in sorted(per.items())},
                          'seq_consecutive_from_0': all(q == list(range(len(q))) for v in per.values() for q in v.values()),
                          'seqs': per,
                          'sha256': hashlib.sha256(json.dumps(radio).encode()).hexdigest()}
        with open(os.path.join(OUT, f'{a.tag}.radio.json'), 'w') as f:
            json.dump(radio, f)
    zd = getattr(ad, '_zones', None)
    if zd is not None:
        stats.update(zone_builds=zd.builds, zone_build_s=zd.build_s, zone_nodes=len(zd.names))
    env.close()
    pos, vel, orn = np.array(pos), np.array(vel), np.array(orn)
    np.savez(os.path.join(OUT, f'{a.tag}.npz'), pos=pos, vel=vel, orn=orn)
    for label, s in saved.items():
        for k in ('obstacles', 'drones'):
            with open(os.path.join(OUT, f'{a.tag}.{label}.{k}.txt'), 'w') as f:
                f.write(s[k])
            s[k] = hashlib.sha256(s[k].encode()).hexdigest()
        s['world_sha256'] = hashlib.sha256(open(s['world'], 'rb').read()).hexdigest()
    return {'tag': a.tag, 'n': n, 'zones': a.zones, 'radio': a.radio, 'render': a.render, 'steps': len(pos),
            'sha256': {'pos': sha(pos), 'vel': sha(vel), 'orn': sha(orn)}, 'saved': saved,
            'startup_s': startup, 'reset_s': resets, **stats}


# ------------------------------------------------------------------ B: drawn vs GeoJSON
def local_from_file(doc, sky, c):
    """A GeoJSON position in local metres, straight from the file (not through Zone / load_zones)."""
    if doc.get('frame', 'wgs84') == 'local_m':
        return float(c[0]), float(c[1])
    x, y = sky.to_local(float(c[1]), float(c[0]))
    return float(x), float(y)


def shoelace(pts):
    pts = np.asarray(pts, dtype=float)
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def mf_vec3(field):
    return np.array([field.getMFVec3f(i) for i in range(field.getCount())])


def faces_of(field):
    faces, cur = [], []
    for i in range(field.getCount()):
        v = field.getMFInt32(i)
        if v == -1:
            faces.append(cur); cur = []
        else:
            cur.append(v)
    return faces


def view(ad, pos, orn):
    vp = ad.sup.getFromDef('VIEW')
    vp.getField('position').setSFVec3f([float(v) for v in pos])
    vp.getField('orientation').setSFRotation([float(v) for v in orn])


def look(ad, target, dist, elev_deg, aspect=0.5):
    """Point VIEW at target from the south (image up = north for a top-down view), `dist` metres
    away along a line `elev_deg` above the horizon."""
    e = math.radians(elev_deg)
    pos = [target[0], target[1] - dist * math.cos(e), target[2] + dist * math.sin(e)]
    rz = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])          # face +y
    ry = np.array([[math.cos(e), 0.0, math.sin(e)], [0.0, 1.0, 0.0], [-math.sin(e), 0.0, math.cos(e)]])
    R = rz @ ry
    ang = math.acos(max(-1.0, min(1.0, (np.trace(R) - 1) / 2)))
    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * math.sin(ang))
    view(ad, pos, [*axis, ang])


def fit_dist(ext, aspect=0.5, margin=1.3):
    """Distance at which `ext` metres fill the window's height (fieldOfView 0.785 is horizontal)."""
    vfov = 2 * math.atan(math.tan(0.3927) * aspect)
    return margin * ext / (2 * math.tan(vfov / 2))


def shoot(ad, path, steps=5):
    for _ in range(steps):
        ad.sup.step(ad.basic_ms)
    ad.sup.exportImage(path, 95)
    for _ in range(200):                                       # the file appears asynchronously
        if os.path.exists(path) and os.path.getsize(path) > 0:
            break
        ad.sup.step(ad.basic_ms)
    return path


def case_drawn(a):
    from envs.webots_l2_multi import WebotsL2MultiAdapter
    from utm.airspace_zones import map_airspace
    path = os.path.join(ROOT, a.file)
    doc = json.load(open(path))
    sky = map_airspace(a.map)
    if a.map == 'd1':
        from envs.city_generator import generate_city
        layout = generate_city(np.random.RandomState(3), 'medium')['layout']    # the file's city
    else:
        from envs.osm_scenario import load_osm_layout
        layout = load_osm_layout(os.path.join(ROOT, doc['layout']))
    cfg = load_cfg(1, sim='webots_l2_multi')
    cfg['webots'] = {'log': os.path.join(OUT, f'{a.tag}.webots.log'), 'render': True, 'draw_zones': True,
                     'zones_file': path, 'zones_map': a.map}
    ad = WebotsL2MultiAdapter(cfg)
    zd = ad._zones
    sp = {'drone_0': [-40.0, -40.0, 1.0]}
    ad.reset_world(sp, sp, layout)
    ad.sup.step(ad.basic_ms)
    ceiling = zd.ceiling_m
    feats = {f['id']: f for f in doc['features'] if f['properties']['type'] == 'zone'}
    rows, ok = [], sorted(zd.names) == sorted(feats)
    for k, zid in enumerate(zd.names):
        f, p = feats[zid], feats[zid]['properties']
        lo = 0.0 if p['lower_m'] is None else float(p['lower_m'])
        hi = ceiling if p['upper_m'] is None else float(p['upper_m'])
        kids = ad.sup.getFromDef(f'ZONE_{k}').getField('children')
        vol, outl = kids.getMFNode(0), kids.getMFNode(1)
        ocoord = mf_vec3(outl.getField('geometry').getSFNode().getField('coord').getSFNode().getField('point'))
        r = {'id': zid, 'restriction': p['restriction'], 'lo': lo, 'hi': hi, 'active': p['active'],
             'outline_z': sorted(set(np.round(ocoord[:, 2], 6).tolist())),
             'outline_colour': outl.getField('appearance').getSFNode().getField('emissiveColor').getSFColor(),
             'colour': (vol.getField('children').getMFNode(0) if vol.getTypeName() == 'Pose' else vol)
                       .getField('appearance').getSFNode().getField('baseColor').getSFColor()}
        if f['geometry']['type'] == 'Point':
            cx, cy = local_from_file(doc, sky, f['geometry']['coordinates'])
            cyl = vol.getField('children').getMFNode(0).getField('geometry').getSFNode()
            tr = vol.getField('translation').getSFVec3f()
            got = [tr[0], tr[1], tr[2], cyl.getField('radius').getSFFloat(), cyl.getField('height').getSFFloat()]
            want = [cx, cy, (lo + hi) / 2, float(p['radius_m']), hi - lo]
            r.update(kind='circle', type=cyl.getTypeName(), got=got, want=want,
                     err_xy=max(abs(got[0] - want[0]), abs(got[1] - want[1])),
                     err_z=max(abs(got[2] - want[2]), abs(got[4] - want[4])), err_r=abs(got[3] - want[3]))
            r['pass'] = (r['type'] == 'Cylinder' and r['err_xy'] < 1e-4 and r['err_z'] < 1e-6
                         and r['err_r'] < 1e-6)
        else:
            ring = [local_from_file(doc, sky, c) for c in f['geometry']['coordinates'][0][:-1]]
            ifs = vol.getField('geometry').getSFNode()
            pts = mf_vec3(ifs.getField('coord').getSFNode().getField('point'))
            faces = faces_of(ifs.getField('coordIndex'))
            n = len(ring)
            bottom, top = pts[:n], pts[n:]
            # match each file corner to the nearest drawn bottom corner
            err_xy = max(float(np.min(np.hypot(*(bottom[:, :2] - q).T))) for q in ring)
            caps = [sum(abs(shoelace(pts[fc][:, :2])) for fc in faces
                        if len(fc) == 3 and np.allclose(pts[fc][:, 2], zc)) for zc in (lo, hi)]
            area = abs(shoelace(ring))
            r.update(kind='polygon', corners=n, drawn_points=len(pts), err_xy=err_xy,
                     z_bottom=sorted(set(np.round(bottom[:, 2], 9).tolist())),
                     z_top=sorted(set(np.round(top[:, 2], 9).tolist())),
                     area_file=area, area_caps=caps, walls=sum(1 for fc in faces if len(fc) == 4))
            r['pass'] = (len(pts) == 2 * n and err_xy < 1e-4 and r['z_bottom'] == [round(lo, 9)]
                         and r['z_top'] == [round(hi, 9)] and all(abs(c - area) < 1e-3 * max(area, 1) for c in caps)
                         and r['walls'] == n)
        want_col = [1.0, 0.0, 0.0] if p['restriction'] == 'prohibited' else [1.0, 0.5, 0.0]
        r['pass'] = bool(r['pass'] and np.allclose(r['colour'], want_col) and np.allclose(r['outline_colour'], want_col)
                         and r['outline_z'] == [0.02])
        ok = ok and r['pass']
        rows.append(r)
    exported = ad.sup.getFromDef('ZONES').exportString()
    visual_only = not any(w in exported for w in ('Solid', 'boundingObject', 'Physics'))
    layout_same = ad.layout == list(layout)
    # screenshots: top-down and oblique over the zones
    allp = np.vstack([np.asarray(ad._zones._drawn[i][0].outline()) for i in range(len(ad._zones._drawn))])
    c = (allp.max(axis=0) + allp.min(axis=0)) / 2
    ext = float(np.max(allp.max(axis=0) - allp.min(axis=0)))
    shots = {}
    look(ad, [c[0], c[1], 0.0], fit_dist(ext) + 10, 89.9)
    shots['top'] = shoot(ad, os.path.join(OUT, f'{a.tag}_top.png'))
    look(ad, [c[0], c[1], 0.0], 1.1 * fit_dist(ext), 40.0)
    shots['oblique'] = shoot(ad, os.path.join(OUT, f'{a.tag}_oblique.png'))
    ad.close()
    return {'tag': a.tag, 'file': a.file, 'map': a.map, 'ceiling_m': ceiling, 'zones': rows,
            'visual_only': visual_only, 'layout_unchanged': layout_same, 'screenshots': shots,
            'pass': bool(ok and visual_only and layout_same)}


VIS_TIMES = (1.0, 3.0, 5.0)
VIS_CAM = ([0.0, 3.0, 0.0], 30.0, 89.9)                       # look(): target, distance, elevation
VIS_SWITCH = [(-3.0, -3.0), (3.0, -3.0), (3.0, 3.0), (-3.0, 3.0)]
VIS_PERM = ((0.0, 8.0), 2.5)
SAME_D, SAME_FRAC = 8, 0.01                                    # pass rule: see the docstring
DIFF_D, DIFF_FRAC = 30, 0.90


def case_visibility(a):
    from envs.webots_l2_multi import WebotsL2MultiAdapter
    from utm.airspace_zones import Zone
    cfg = load_cfg(1, sim='webots_l2_multi')
    cfg['webots'] = {'log': os.path.join(OUT, f'{a.tag}.webots.log'), 'render': True}
    if a.zones:
        cfg['webots']['draw_zones'] = True
    ad = WebotsL2MultiAdapter(cfg)
    if a.zones:
        ad.set_zones([Zone('red_temp', 'red, active 2-4 s', active=[[2.0, 4.0]], polygon=VIS_SWITCH),
                      Zone('amber_perm', 'amber, permanent', 'authorisation', circle=VIS_PERM)])
    sp = {'drone_0': [-30.0, -30.0, 1.0]}
    ad.reset_world(sp, sp, [])
    look(ad, *VIS_CAM)
    shots, k = {}, 0
    for t in VIS_TIMES:
        while k * ad.control_dt < t - 1e-9:
            ad.apply_actions({'drone_0': [0.0] * 4})
            ad.step_simulation()
            k += 1
        path = os.path.join(OUT, f'{a.tag}_t{t:g}.png')
        if os.path.exists(path):
            os.replace(path, f'{path}.{int(time.time())}.old')  # keep the previous shot, never wait on it
        ad.sup.exportImage(path, 95)
        for _ in range(200):                                   # rendering only; the episode clock stops
            if os.path.exists(path) and os.path.getsize(path) > 0:
                break
            ad.sup.step(ad.basic_ms)
        shots[f'{t:g}'] = {'step': k, 'png': path}
    zd = ad._zones
    ad.close()
    return {'tag': a.tag, 'zones': a.zones, 'shots': shots, 'toggles': zd.toggles if zd else None}


def region_mask(shape, poly=None, circle=None):
    """Pixels whose ground point (z = 0, straight below the VIS_CAM camera) is inside poly / circle."""
    from matplotlib.path import Path
    h, w = shape[:2]
    target, dist, elev = VIS_CAM
    e = math.radians(elev)
    cx, cy, cz = target[0], target[1] - dist * math.cos(e), dist * math.sin(e)
    mpp = 2 * cz * math.tan(0.3927) / w                       # metres per pixel on the ground
    u, v = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    x, y = cx + (u - w / 2) * mpp, cy - (v - h / 2) * mpp
    if circle is not None:
        (ox, oy), r = circle
        return np.hypot(x - ox, y - oy) <= r
    return Path(poly).contains_points(np.column_stack([x.ravel(), y.ravel()])).reshape(h, w)


def region_diff(a, b, mask):
    d = np.abs(a.astype(int) - b.astype(int)).max(axis=2)[mask]
    return {'pixels': int(mask.sum()), 'frac_gt_same': float((d > SAME_D).mean()),
            'frac_gt_diff': float((d > DIFF_D).mean()), 'max_d': int(d.max()), 'exact': bool((d == 0).all())}


def case_viscompare(a):
    from PIL import Image
    z = json.load(open(os.path.join(OUT, f'{a.tag}.json')))
    r = json.load(open(os.path.join(OUT, f'{a.tag}_ref.json')))
    img = {(k, t): np.asarray(Image.open(run['shots'][t]['png']).convert('RGB'))
           for k, run in (('z', z), ('r', r)) for t in run['shots']}
    shape = img[('z', '1')].shape
    masks = {'switching': region_mask(shape, poly=[(-2.4, -2.4), (2.4, -2.4), (2.4, 2.4), (-2.4, 2.4)]),
             'permanent': region_mask(shape, circle=(VIS_PERM[0], 2.0))}
    same = lambda c: c['frac_gt_same'] <= SAME_FRAC            # noqa: E731
    diff = lambda c: c['frac_gt_diff'] >= DIFF_FRAC            # noqa: E731
    cmp, rules = {}, {}
    steps_ok = all(z['shots'][t]['step'] == r['shots'][t]['step'] for t in ('1', '3', '5'))
    m = masks['switching']
    for t in ('1', '3', '5'):
        cmp[f'switching zones_t{t} vs ref_t{t}'] = c = region_diff(img[('z', t)], img[('r', t)], m)
        rules[f'switching t{t} ' + ('DIFF' if t == '3' else 'SAME')] = diff(c) if t == '3' else same(c)
    m = masks['permanent']
    for t1, t2 in (('1', '3'), ('3', '5'), ('1', '5')):
        cmp[f'permanent zones_t{t1} vs zones_t{t2}'] = c = region_diff(img[('z', t1)], img[('z', t2)], m)
        rules[f'permanent t{t1}-t{t2} SAME'] = same(c)
    for t in ('1', '3', '5'):
        cmp[f'permanent zones_t{t} vs ref_t{t}'] = c = region_diff(img[('z', t)], img[('r', t)], m)
        rules[f'permanent t{t} DIFF ref'] = diff(c)
    rules['same control steps'] = steps_ok
    return {'tag': a.tag, 'image_shape': list(shape), 'region_pixels': {k: int(v.sum()) for k, v in masks.items()},
            'rule': {'SAME': f'<= {SAME_FRAC:.0%} of pixels with d > {SAME_D}',
                     'DIFF': f'>= {DIFF_FRAC:.0%} of pixels with d > {DIFF_D}'},
            'comparisons': cmp, 'rules': rules, 'pass': bool(all(rules.values()))}


# ------------------------------------------------------------------ D: speed
def case_speed(a):
    import torch
    from envs.swarm_env import SwarmEnv
    n = a.n
    city, sp, gl, routes = episode(n)
    cfg = load_cfg(n, sim='webots_l2_multi')
    cfg['webots'] = {'log': os.path.join(OUT, f'{a.tag}.webots.log')}
    if a.render:
        cfg['webots']['render'] = True
    if a.zones:
        cfg['webots']['draw_zones'] = True
    cfg['environment']['route_follow'] = {'enabled': True}
    t0 = time.time()
    env = SwarmEnv(cfg, device=torch.device('cpu'))
    startup = time.time() - t0
    ad = env.adapter
    if a.zones:
        ad.set_zones(city_zones(city))
    steps = int(round(EP_S / ad.control_dt))
    t0 = time.time(); env.start_episode(sp, gl, obstacles=city['layout'], routes=routes); reset1 = time.time() - t0
    t0 = time.time()
    for _ in range(steps):                                     # (a) adapter alone
        ad.apply_actions({f'drone_{i}': [0.0] * 4 for i in range(n)})
        ad.step_simulation()
    wall_adapter = time.time() - t0
    t0 = time.time(); env.start_episode(sp, gl, obstacles=city['layout'], routes=routes); reset2 = time.time() - t0
    t0 = time.time()
    for _ in range(steps):                                     # (b) env + route following + chaser
        _, _, done, _ = env.step(chase(env))
        if done:
            break
    wall_env = time.time() - t0
    zd = ad._zones
    out = {'tag': a.tag, 'n': n, 'render': a.render, 'zones': a.zones, 'startup_s': startup,
           'reset_s': [reset1, reset2], 'wall_per_sim_s_adapter': wall_adapter / EP_S,
           'wall_per_sim_s_env_chaser': wall_env / EP_S}
    if zd is not None:
        out.update(zone_nodes=len(zd.names), zone_builds=zd.builds, zone_build_s=zd.build_s, toggles=zd.toggles)
    env.close()
    return out


CASES = {'track': case_track, 'drawn': case_drawn, 'visibility': case_visibility,
         'viscompare': case_viscompare, 'speed': case_speed}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--worker', required=True, choices=sorted(CASES))
    ap.add_argument('--tag', required=True)
    ap.add_argument('--n', type=int, default=4)
    ap.add_argument('--zones', action='store_true')
    ap.add_argument('--render', action='store_true')
    ap.add_argument('--save', action='store_true')
    ap.add_argument('--radio', action='store_true')
    ap.add_argument('--file', default=None)
    ap.add_argument('--map', default=None)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    r = CASES[a.worker](a)
    r['wall_s'] = time.time() - t0
    path = os.path.join(OUT, f'{a.tag}.json')
    json.dump(r, open(path, 'w'), indent=2)
    print(json.dumps(r, indent=2))
    print(f'written: {path}')


if __name__ == '__main__':
    main()
