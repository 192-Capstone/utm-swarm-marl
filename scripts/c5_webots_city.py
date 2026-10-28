"""
c5_webots_city.py
-----------------
C5: the city worlds in Webots Level 2 (envs/webots_l2_multi.py: official Crazyflies, one
cf_velocity controller each). Short checks for parts 1, 2, 3 and 5. Every case runs in its OWN
subprocess (one Webots connection per Python process), one after the other.
Config: configs/config_stage2_round7_eval_goalhold.yaml. Maps: D1 cities easy / medium / hard
(envs/city_generator.py) and the E-City crop, drawn as scripts/eval_city_routes.py does (seed 20).

Cases
  maps       Part 1. One Webots, 4 drones; for each map: the buildings Webots holds (OBSTACLES
             children) against the layout: count, name, translation, footprint corners (with the
             box's yaw), winding (clockwise, as SimpleBuilding expects), area, height; every
             building corner, spawn and goal inside the floor. Reset time: 3 resets rebuilding the
             buildings every time (reuse_layout off), then 3 reusing them, and an empty map;
             the building time itself (adapter.layout_build_s).
  route      Part 2. SwarmEnv with environment.route_follow, one drone, the corner-wall route of
             tests/test_route_follow.py; a scripted carrot chaser (v = carrot - position, capped
             at 1 m/s) flies it. Settled?, steps, cross-track error (median / p95 / max), closest
             approach to the wall's surface. The same in PyBullet (route_pb) for comparison.
  place      Part 3. 3 drones on pads at x = -10, 0, 10, z = 1 m. drone_0 and drone_2 fly scripts
             from the start; drone_1 is parked right after the reset (as SwarmEnv.start_episode),
             placed on its pad at t = 5 s (SwarmEnv.step's call order: apply_actions,
             step_simulation, place, park), then: zero command 1.5 s, vy = 0.5 m/s for 1 s, zero
             2.5 s.
  place_ref  The same, but drone_1 is never parked: it flies its script from the reset.
             Compared:
               start      drone_1 after place vs after the reset release (place_ref): speed in
                          control steps 0-5, climb / drop and horizontal drift over the zero-
                          command 1.5 s, max tilt
               response   drone_1's track after place vs from release (relative to the pad)
               others     drone_0 and drone_2's tracks with vs without the park / place
             Pass (C1's start check and cross-talk limits): after place, speed in steps 0-5
             < 0.05 m/s, |climb| < 0.05 m, drift < 0.05 m; response and others < 0.01 m.
  land       Part 3. 3 drones; drone_0 flies vx 1 m/s for 1 s and is parked at t = 2 s (landed);
             drone_2 climbs at +1 m/s and is parked mid-climb at t = 1.5 s (as a crashed drone);
             parked until t = 5 s; how far a parked drone moves between pins (one control step);
             then reset_world and zero command for 1.5 s. Pass: no re-import, every drone's
             speed in steps 0-5 < 0.05 m/s and |climb| < 0.05 m.
  schedule   Part 3. SwarmEnv with schedule, route_follow and crash_mode remove (as
             eval_e1_flown.py), 3 drones, E1 layer profile routes (pad z 1 m, cruise 3 m),
             departures 0 / 2 / 4 s; a scripted carrot chaser flies. Per drone: take-off delay
             (first step it is more than 5 cm from its pad, minus the departure), landing time;
             episode end reason. schedule_pb: the same in PyBullet.
  speed      Part 5. Hard city (seed 20), n = 4, 8, 12, 16, 20 drones on sample_flights spawns.
             Wall seconds per simulated second: (a) the adapter alone, zero command, 10 s;
             (b) SwarmEnv with route following and the scripted chaser (and the policy when
             --checkpoint is given), 10 s. Plus start-up and reset wall time.

Usage (repo root, DISPLAY explicit, low priority, nothing else heavy running):
  DISPLAY=:2 OMP_NUM_THREADS=4 nice -n 19 python scripts/c5_webots_city.py --cases maps route place land schedule
  ... --cases speed [--checkpoint <ckpt>]
Writes logs/webots/c5/<case>_<k>.json and report_<cases>.json; Webots logs next to them.
"""
import argparse
import json
import math
import os
import subprocess
import sys
import time

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

CONFIG = 'configs/config_stage2_round7_eval_goalhold.yaml'
OUT = os.path.join(ROOT, 'logs', 'webots', 'c5')
MAPS = [('easy', 'city'), ('medium', 'city'), ('hard', 'city'), ('crop', 'osm-crop')]
MAP_SEED = 20
SPEED_N = (4, 8, 12, 16, 20)
PAD_Z = 1.0
WALL = {'type': 'box', 'center': (0.0, -2.0), 'half': (0.4, 8.0), 'height': 10.0}


# ------------------------------------------------------------------ helpers (worker side)
def load_cfg(n, sim=None, tag=None):
    cfg = yaml.safe_load(open(os.path.join(ROOT, CONFIG)))
    cfg.pop('scenario_mix', None)
    cfg['environment']['n_agents'] = n
    cfg['environment']['collision_physics'] = False
    cfg.setdefault('obstacles', {})['enabled'] = False
    if sim:
        cfg['environment']['simulator'] = sim
    if tag:
        cfg['webots'] = {'log': os.path.join(OUT, f'{tag}.webots.log')}
    return cfg


def make_multi(n, tag):
    from envs.webots_l2_multi import WebotsL2MultiAdapter
    return WebotsL2MultiAdapter(load_cfg(n, tag=tag))


def map_episode(name, kind, drones):
    """layout, spawns, goals, routes as scripts/eval_city_routes.py draws them (seed MAP_SEED)."""
    import eval_city_routes as ecr
    a = argparse.Namespace(map=kind, level=name if kind == 'city' else 'medium', drones=drones)
    return ecr.make_episode(np.random.RandomState(MAP_SEED), a)


def chase(env):
    """Scripted carrot chaser: v = adapter goal (carrot) - position, capped at 1 m/s."""
    pos = env.adapter.get_positions().astype(np.float64)
    out = {}
    for i in range(env.n_agents):
        v = np.asarray(env.adapter.goals[f'drone_{i}'], dtype=np.float64) - pos[i]
        v *= min(1.0, 1.0 / max(float(np.linalg.norm(v)), 1e-9))
        out[f'drone_{i}'] = [*v, 0.0]
    return out


def tilt_deg(ad):
    return np.degrees(np.abs(ad._orn[:, :2])).max(axis=1)


# ------------------------------------------------------------------ part 1: maps
def expected_corners(ob):
    """Footprint corners relative to the centre, rotated by the box's yaw (independent of
    envs/webots_adapter.building_node)."""
    hx, hy = ob['half']
    yaw = ob.get('yaw', 0.0)
    c, s = math.cos(yaw), math.sin(yaw)
    return [(c * px - s * py, s * px + c * py) for px, py in ((hx, hy), (hx, -hy), (-hx, -hy), (-hx, hy))]


def signed_area(pts):
    return 0.5 * sum(x0 * y1 - x1 * y0 for (x0, y0), (x1, y1) in zip(pts, pts[1:] + pts[:1]))


def check_buildings(ad, layout):
    root = ad._obstacle_root
    bad = []
    if root.getCount() != len(layout):
        bad.append(f'count {root.getCount()} != {len(layout)}')
    worst = {'translation_m': 0.0, 'corner_m': 0.0, 'height_m': 0.0, 'area_rel': 0.0}
    corners_world = []
    for k, ob in enumerate(layout[:root.getCount()]):
        nd = root.getMFNode(k)
        if nd.getTypeName() != 'SimpleBuilding' or nd.getField('name').getSFString() != f'obstacle_{k}':
            bad.append(f'node {k}: {nd.getTypeName()} {nd.getField("name").getSFString()!r}')
            continue
        tr = nd.getField('translation').getSFVec3f()
        worst['translation_m'] = max(worst['translation_m'], math.dist(tr, [*ob['center'], 0.0]))
        f = nd.getField('corners')
        got = [tuple(f.getMFVec2f(j)) for j in range(f.getCount())]
        want = expected_corners(ob)
        if len(got) != 4:
            bad.append(f'node {k}: {len(got)} corners')
            continue
        worst['corner_m'] = max(worst['corner_m'], max(math.dist(g, w) for g, w in zip(got, want)))
        if signed_area(got) >= 0:
            bad.append(f'node {k}: corners not clockwise')
        area = 4 * ob['half'][0] * ob['half'][1]
        worst['area_rel'] = max(worst['area_rel'], abs(-signed_area(got) - area) / area)
        h = nd.getField('floorHeight').getSFFloat() * nd.getField('floorNumber').getSFInt32()
        worst['height_m'] = max(worst['height_m'], abs(h - ob['height']))
        corners_world += [(tr[0] + x, tr[1] + y) for x, y in got]
    ok = (not bad and worst['translation_m'] < 1e-6 and worst['corner_m'] < 1e-3
          and worst['height_m'] < 1e-3 and worst['area_rel'] < 1e-3)
    return {'n_layout': len(layout), 'n_nodes': root.getCount(),
            'n_rotated': sum(1 for ob in layout if ob.get('yaw', 0.0)), 'worst': worst,
            'problems': bad[:10], 'pass': bool(ok)}, corners_world


def floor_size(ad):
    ch = ad.sup.getRoot().getField('children')
    for k in range(ch.getCount()):
        nd = ch.getMFNode(k)
        if nd.getTypeName() == 'Floor':
            return nd.getField('size').getSFVec2f()
    return None


def case_maps(a):
    n = 4
    ad = make_multi(n, 'maps')
    fs = floor_size(ad)
    out = {'floor_size_m': fs, 'maps': {}}
    sp0 = {f'drone_{i}': [2.0 * i, 0.0, PAD_Z] for i in range(n)}
    t0 = time.time(); ad.reset_world(sp0, sp0, []); t_empty = [time.time() - t0]
    for _ in range(2):
        t0 = time.time(); ad.reset_world(sp0, sp0, []); t_empty.append(time.time() - t0)
    out['reset_empty_s'] = t_empty
    for name, kind in MAPS:
        layout, sp, gl, _ = map_episode(name, kind, n)
        ad.reuse_layout = False
        rebuild, b0 = [], ad.layout_build_s
        for _ in range(3):
            t0 = time.time(); ad.reset_world(sp, gl, layout); rebuild.append(time.time() - t0)
        build_s = (ad.layout_build_s - b0) / 3
        chk, corners = check_buildings(ad, layout)
        ad.reuse_layout = True
        reuse, nb = [], ad.layout_builds
        for _ in range(3):
            t0 = time.time(); ad.reset_world(sp, gl, layout); reuse.append(time.time() - t0)
        chk_reuse, _ = check_buildings(ad, layout)
        pts = np.array(corners + [p[:2] for p in sp.values()] + [p[:2] for p in gl.values()], dtype=float)
        inside = bool(fs is not None and np.all(np.abs(pts) <= np.array(fs) / 2))
        out['maps'][name] = {**chk, 'still_correct_after_reuse': chk_reuse['pass'],
                             'rebuilds_while_reusing': ad.layout_builds - nb,
                             'extent_m': [float(np.abs(pts[:, 0]).max()), float(np.abs(pts[:, 1]).max())],
                             'inside_floor': inside, 'build_s': build_s,
                             'reset_rebuild_s': rebuild, 'reset_reuse_s': reuse}
    ad.close()
    out['pass'] = all(m['pass'] and m['still_correct_after_reuse'] and m['inside_floor']
                      and m['rebuilds_while_reusing'] == 0 for m in out['maps'].values())
    return out


# ------------------------------------------------------------------ part 2: route following
def case_route(a):
    import torch
    from envs.route_planner import build_grid, plan, simplify
    from envs.scenario_manager import distance_to_obstacles
    from envs.swarm_env import SwarmEnv
    from eval_city_routes import cross_track
    sim = 'pybullet' if a.k else 'webots_l2_multi'
    cfg = load_cfg(1, sim=sim, tag=f'route_{sim}')
    cfg['environment']['route_follow'] = {'enabled': True}
    cfg['environment']['max_steps'] = 1500
    g = build_grid([WALL], ((-15.0, 15.0), (-15.0, 15.0)), res=0.25, inflate=0.5)
    route = simplify(g, plan(g, (-6.0, 0.0), (6.0, 0.0)))
    env = SwarmEnv(cfg, device=torch.device('cpu'))
    spawns, goals = {'drone_0': [-6.0, 0.0, 2.0]}, {'drone_0': [6.0, 0.0, 2.0]}
    env.start_episode(spawns, goals, obstacles=[WALL], routes={'drone_0': route})
    r = np.asarray(route, dtype=float)
    xt, gap, info, steps, carrot_ok = [], [], {}, 0, True
    for steps in range(1, 1501):
        _, _, done, info = env.step(chase(env))
        p = env.adapter.get_positions()[0]
        xt.append(cross_track(p[:2], r))
        gap.append(float(distance_to_obstacles([WALL], p)))
        carrot_ok &= bool(np.allclose(env.goals[0], goals['drone_0']))
        if done:
            break
    env.close()
    return {'simulator': sim, 'route': [list(map(float, q)) for q in route],
            'settled': bool(info['drone_reached_goal'][0]), 'reason': info.get('reason'), 'steps': steps,
            'xt_median_m': float(np.median(xt)), 'xt_p95_m': float(np.percentile(xt, 95)),
            'xt_max_m': float(np.max(xt)), 'min_wall_gap_m': float(min(gap)),
            'scoring_goal_unchanged': carrot_ok}


# ------------------------------------------------------------------ part 3: park / place
PLACE_T = 5.0
RESP_T = (1.5, 2.5)          # after release/place: zero until 1.5 s, vy 0.5 until 2.5 s, then zero


def ped_script(i, t):
    if i == 0:
        return [1.0 if t < 2.0 else 0.0, 0.0, 0.0]
    if i == 2:
        return [0.8 * math.cos(t), 0.8 * math.sin(t), 0.0]
    return [0.0, 0.5 if RESP_T[0] <= t < RESP_T[1] else 0.0, 0.0]


def place_spawns():
    sp = {f'drone_{i}': [10.0 * (i - 1), 0.0, PAD_Z] for i in range(3)}
    return sp, {k: [v[0], v[1] + 5.0, v[2]] for k, v in sp.items()}


def case_place(a, parked_until=PLACE_T):
    ad = make_multi(3, 'place' if parked_until else 'place_ref')
    sp, gl = place_spawns()
    ad.reset_world(sp, gl, [])
    parked = []
    if parked_until:
        ad.park([1]); parked = [1]
    dt = ad.control_dt
    n_steps = int(round(10.0 / dt))
    k_place = int(round(parked_until / dt)) if parked_until else None
    t1 = None                                    # drone_1's clock (since release / place)
    pos, vel, tilt, t1s = [], [], [], []
    if not parked_until:
        t1 = 0.0
    for s in range(n_steps):
        t = s * dt
        acts = {f'drone_{i}': ped_script(i, t) + [0.0] for i in (0, 2)}
        acts['drone_1'] = ped_script(1, t1) + [0.0] if t1 is not None else [0.0] * 4
        ad.apply_actions(acts)
        ad.step_simulation()
        if t1 is not None:
            t1 += dt
        if k_place is not None and s + 1 == k_place:             # SwarmEnv: place after the step
            ad.place(1, sp['drone_1']); parked = []; t1 = 0.0
        if parked:
            ad.park(parked)
        pos.append(ad._pos.copy()); vel.append(np.linalg.norm(ad._vel, axis=1)); tilt.append(tilt_deg(ad))
        t1s.append(t1)
    ad.close()
    return {'pos': np.array(pos).tolist(), 'speed': np.array(vel).tolist(), 'tilt': np.array(tilt).tolist(),
            't1': t1s, 'k_place': k_place, 'pad': sp['drone_1']}


def case_place_ref(a):
    return case_place(a, parked_until=None)


def start_metrics(pos, speed, tilt, k0, pad, dt):
    """drone_1 from control step k0 (its first step after release / place)."""
    k1 = k0 + int(round(RESP_T[0] / dt))
    p = pos[k0:k1, 1]
    return {'speed_steps0_5': speed[k0:k0 + 6, 1].tolist(),
            'climb_m': float(p[:, 2].max() - pad[2]), 'drop_m': float(pad[2] - p[:, 2].min()),
            'drift_m': float(np.linalg.norm(p[:, :2] - np.asarray(pad[:2]), axis=1).max()),
            'max_tilt_deg': float(tilt[k0:k1, 1].max())}


def compare_place(A, B):
    dt = 0.032
    pa, pb = np.array(A['pos']), np.array(B['pos'])
    sa, sb = np.array(A['speed']), np.array(B['speed'])
    ta, tb = np.array(A['tilt']), np.array(B['tilt'])
    k0 = A['k_place']                                          # first step after place: index k0
    after = start_metrics(pa, sa, ta, k0, A['pad'], dt)
    ref = start_metrics(pb, sb, tb, 0, B['pad'], dt)
    m = min(len(pa) - k0, len(pb))
    resp = float(np.linalg.norm(pa[k0:k0 + m, 1] - pb[:m, 1], axis=1).max())
    others = {f'drone_{i}': float(np.linalg.norm(pa[:, i] - pb[:, i], axis=1).max()) for i in (0, 2)}
    ok = (max(after['speed_steps0_5']) < 0.05 and max(after['climb_m'], after['drop_m']) < 0.05
          and after['drift_m'] < 0.05 and resp < 0.01 and max(others.values()) < 0.01)
    return {'after_place': after, 'after_reset_release': ref, 'response_track_diff_m': resp,
            'others_track_diff_m': others, 'pass': bool(ok)}


def case_land(a):
    ad = make_multi(3, 'land')
    sp, gl = place_spawns()
    ad.reset_world(sp, gl, [])
    dt = ad.control_dt
    parked, pin_move = [], []
    from envs.webots_l2_multi import park_spot
    for s in range(int(round(5.0 / dt))):
        t = s * dt
        acts = {'drone_0': [1.0 if t < 1.0 else 0.0, 0.0, 0.0, 0.0], 'drone_1': [0.0] * 4,
                'drone_2': [0.0, 0.0, 1.0, 0.0]}
        ad.apply_actions(acts)
        ad.step_simulation()
        for i in parked:                                       # moved since last pinned
            pin_move.append(math.dist(ad._pos[i], park_spot(i)))
        if t + dt >= 2.0 and 0 not in parked:
            parked.append(0)
        if t + dt >= 1.5 and 2 not in parked:
            parked.append(2)
        if parked:
            ad.park(parked)
    err, speed, z = None, [], []
    try:
        ad.reset_world(sp, gl, [])
        for s in range(int(round(1.5 / dt))):
            ad.apply_actions({f'drone_{i}': [0.0] * 4 for i in range(3)})
            ad.step_simulation()
            speed.append(np.linalg.norm(ad._vel, axis=1).tolist())
            z.append(ad._pos[:, 2].tolist())
    except RuntimeError as e:
        err = str(e)
    log = [list(x) for x in ad.reimport_log]
    ad.close()
    speed, z = np.array(speed), np.array(z)
    climb = (z - PAD_Z).max(axis=0).tolist() if len(z) else None
    drop = (PAD_Z - z).max(axis=0).tolist() if len(z) else None
    r = {'pinned_move_per_step_m': {'median': float(np.median(pin_move)), 'max': float(max(pin_move))},
         'reimport_log': log, 'error': err,
         'speed_steps0_5_max': speed[:6].max(axis=0).tolist() if len(speed) else None,
         'climb_m': climb, 'drop_m': drop}
    r['pass'] = bool(err is None and all(not d for _, d in log) and len(speed) >= 6
                     and speed[:6].max() < 0.05 and max(climb) < 0.05 and max(drop) < 0.05)
    return r


def case_schedule(a):
    import torch
    from envs.swarm_env import SwarmEnv
    sim = 'pybullet' if a.k else 'webots_l2_multi'
    cfg = load_cfg(3, sim=sim, tag=f'schedule_{sim}')
    cfg['environment'].update(crash_mode='remove', schedule={'enabled': True}, route_follow={'enabled': True},
                              max_steps=1500)
    env = SwarmEnv(cfg, device=torch.device('cpu'))
    pads = [(-8.0, -4.0), (0.0, -4.0), (8.0, -4.0)]
    goals_xy = [(-8.0, 4.0), (0.0, 4.0), (8.0, 4.0)]
    dep = [0.0, 2.0, 4.0]
    spawns = {f'drone_{i}': [*pads[i], PAD_Z] for i in range(3)}
    goals = {f'drone_{i}': [*goals_xy[i], PAD_Z] for i in range(3)}
    routes = {f'drone_{i}': [(*pads[i], PAD_Z), (*pads[i], 3.0), (*goals_xy[i], 3.0), (*goals_xy[i], PAD_Z)]
              for i in range(3)}
    env.start_episode(spawns, goals, obstacles=[], routes=routes,
                      departures={f'drone_{i}': dep[i] for i in range(3)})
    dt = env.control_dt
    off_pad, landed, info = [None] * 3, [None] * 3, {}
    max_tilt = np.zeros(3)
    for s in range(1, 1501):
        _, _, done, info = env.step(chase(env))
        p = env.adapter.get_positions()
        for i in range(3):
            if off_pad[i] is None and not env.waiting[i] and math.dist(p[i], spawns[f'drone_{i}']) > 0.05:
                off_pad[i] = s * dt
            if landed[i] is None and env.landed[i]:
                landed[i] = s * dt
        if sim != 'pybullet':
            fly = ~(env.waiting | env.landed | env.crashed)
            max_tilt = np.where(fly, np.maximum(max_tilt, tilt_deg(env.adapter)), max_tilt)
        if done:
            break
    env.close()
    return {'simulator': sim, 'reason': info.get('reason'), 'end_s': s * dt,
            'takeoff_delay_s': [None if o is None else o - d for o, d in zip(off_pad, dep)],
            'landed_s': landed, 'crashed': env.crashed.tolist(),
            'max_tilt_airborne_deg': max_tilt.tolist() if sim != 'pybullet' else None}


# ------------------------------------------------------------------ part 5: speed
def case_speed(a):
    import torch
    from envs.city_generator import generate_city, sample_flights
    from envs.route_planner import plan, simplify
    from envs.swarm_env import SwarmEnv
    n = a.k
    city = generate_city(np.random.RandomState(MAP_SEED), 'hard')
    sp, gl = sample_flights(np.random.RandomState(MAP_SEED), city, n)
    routes = {}
    for k in sp:
        p = plan(city['grid'], sp[k][:2], gl[k][:2])
        routes[k] = simplify(city['grid'], p) if p is not None else [tuple(sp[k][:2]), tuple(gl[k][:2])]
    cfg = load_cfg(n, sim='webots_l2_multi', tag=f'speed_n{n}')
    cfg['environment']['route_follow'] = {'enabled': True}
    t0 = time.time()
    env = SwarmEnv(cfg, device=torch.device('cpu'))
    startup = time.time() - t0
    ad = env.adapter
    secs = 10.0
    steps = int(round(secs / ad.control_dt))
    t0 = time.time(); env.start_episode(sp, gl, obstacles=city['layout'], routes=routes); reset1 = time.time() - t0
    t0 = time.time()
    for _ in range(steps):                                     # (a) adapter alone
        ad.apply_actions({f'drone_{i}': [0.0] * 4 for i in range(n)})
        ad.step_simulation()
    wall_adapter = time.time() - t0
    t0 = time.time(); obs = env.start_episode(sp, gl, obstacles=city['layout'], routes=routes); reset2 = time.time() - t0
    t0 = time.time()
    for _ in range(steps):                                     # (b) env + route following + chaser
        obs, _, done, _ = env.step(chase(env))
        if done:
            break
    wall_env = time.time() - t0
    out = {'n': n, 'startup_s': startup, 'reset_s': [reset1, reset2],
           'wall_per_sim_s_adapter': wall_adapter / secs, 'wall_per_sim_s_env_chaser': wall_env / secs}
    if a.checkpoint:
        import eval_suite as es
        from modules.goal_hold import GoalHoldPolicy, goal_hold_enabled
        policy = es.MappoPolicy(cfg, a.checkpoint, torch.device('cpu'), True)
        policy.restore_normalizer(env)
        if goal_hold_enabled(cfg):
            policy = GoalHoldPolicy(policy, cfg)
        obs = env.start_episode(sp, gl, obstacles=city['layout'], routes=routes)
        t0 = time.time()
        k = 0
        for k in range(1, steps + 1):
            obs, _, done, _ = env.step(policy.act(obs, env, n))
            if done:
                break
        out['wall_per_sim_s_env_policy'] = (time.time() - t0) / (k * ad.control_dt)
    env.close()
    return out


CASES = {'maps': case_maps, 'route': case_route, 'place': case_place, 'place_ref': case_place_ref,
         'land': case_land, 'schedule': case_schedule, 'speed': case_speed}


# ------------------------------------------------------------------ main side
def run_worker(case, k=0, checkpoint=None):
    out = os.path.join(OUT, f'{case}_{k}.json')
    cmd = [sys.executable, os.path.abspath(__file__), '--worker', case, '--k', str(k), '--out', out]
    if checkpoint:
        cmd += ['--checkpoint', checkpoint]
    t0 = time.time()
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if p.returncode != 0:
        return {'error': (p.stderr or p.stdout)[-3000:], 'wall_s': time.time() - t0}
    r = json.load(open(out))
    r['wall_s'] = time.time() - t0
    return r


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--cases', nargs='+', default=['maps', 'route', 'place', 'land', 'schedule'])
    ap.add_argument('--checkpoint', default=None)
    ap.add_argument('--worker', default=None)
    ap.add_argument('--k', type=int, default=0)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if a.worker:
        json.dump(CASES[a.worker](a), open(a.out, 'w'))
        return
    report = {}
    if 'maps' in a.cases:
        report['maps'] = run_worker('maps')
    if 'route' in a.cases:
        report['route'] = {'webots': run_worker('route', 0), 'pybullet': run_worker('route', 1)}
    if 'place' in a.cases:
        A, B = run_worker('place'), run_worker('place_ref')
        report['place'] = ({'error': [A.get('error'), B.get('error')]} if 'error' in A or 'error' in B
                           else compare_place(A, B))
    if 'land' in a.cases:
        report['land'] = run_worker('land')
    if 'schedule' in a.cases:
        report['schedule'] = {'webots': run_worker('schedule', 0), 'pybullet': run_worker('schedule', 1)}
    if 'speed' in a.cases:
        report['speed'] = [run_worker('speed', n, a.checkpoint) for n in SPEED_N]
    path = os.path.join(OUT, f"report_{'_'.join(a.cases)}.json")
    json.dump(report, open(path, 'w'), indent=2)
    print(json.dumps(report, indent=2))
    print(f'written: {path}')


if __name__ == '__main__':
    main()
