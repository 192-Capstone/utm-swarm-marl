"""
webots_l2_multi_speed.py
------------------------
C1 step 1: does the multi-drone Level-2 adapter (envs/webots_l2_multi.py) fly correctly, and how
fast is it? Every case runs in its OWN subprocess (one Webots connection per Python process),
one after the other.

Cases
  hover      3 drones, each: vy = +0.5 m/s for 1 s, then vz = +0.3 m/s for 1 s, then zero for 13 s
             (15 s; Bitcraze's altitude loop needs several seconds to settle). Per drone: the
             measured move (final y and z minus the spawn), drift and max true speed over the
             last 5 s, max tilt, max true speed, and the z track (every control step, in the
             JSON only). (A zero command from a perfect hover proves nothing: a stopped
             controller would hold just as still.)
  crosstalk  3 drones 10 m apart, each with a different scripted command (10 s); then each
             script alone (n = 1, same spawn) in its own subprocess. Per drone: max |track
             difference| multi vs solo.
  determ     the same episode (3 drones, scripted) twice in one process (reset in between) and
             once more in a second process. Max |track difference| within and across processes.
  speed      wall seconds per simulated second for n = 1, 3, 8 (zero command, 20 s), and the
             wall time of reset_world (2 s spin-up hold).
  delay      step command vx 0 -> 1 m/s, one drone, sampled every basic step (4 ms): new adapter
             (n = 1) vs WebotsL2Adapter (official Crazyflie). Time at which vx first crosses
             0.05 / 0.2 / 0.5 m/s in each, and the difference.
  badcmd     drone_1 of 3 gets a malformed customData; the drone must fall and the adapter must
             raise naming drone_1 (at the next reset if Webots has already flushed its log,
             otherwise at close). Reports where it was caught.
  crash      3 drones; at 1 s drone_0 is turned upside down and keeps getting a +2 m/s climb
             command for 25 s (as the policy did after a real crash: the altitude setpoint runs
             away and the motors saturate). Then reset_world and zero command for 1.5 s. Without
             the re-import drone_0 shot up at 31 -> 153 m/s and climbed 118 m (Propeller motor
             speed survives resetPhysics). Reports drone_0's speed in the first 6 control steps,
             its climb, and the adapter's re-import log.
  crash_l2   the same crash with the single-drone adapter (envs/webots_l2.py, official Crazyflie),
             which cannot re-import itself: its reset holds the drone until the estimated rotor
             speeds are back at hover. Reports the rotor estimate before the reset and the hold time.

Drones are placed at x = 10 * (i - (n - 1) / 2), y = 0, z = 1.5 m in every case, so 8 drones stay
within +-35 m (the floor is 120 m wide).

Pass rules (agreed: hover "a few cm", no lost control; cross-talk < 0.01 m; determinism and speed
reported only; delay expected ~4 ms extra; bad command must be reported):
  hover      every drone: it moved (y change 0.3-0.7 m, z change 0.15-0.45 m); it stopped and
             holds (last 5 s: horizontal and vertical drift < 0.05 m, true speed < 0.05 m/s);
             tilt <= 60 deg, true speed <= 3 m/s, no drop-check error
  crosstalk  every drone: max track difference multi vs solo < 0.01 m
  determ     REPORT ONLY (max difference, not assumed zero)
  speed      REPORT ONLY
  delay      REPORT; expected new - old ~ +4 ms (one basic step)
  badcmd     a RuntimeError naming drone_1 is raised (at reset or at close)
  crash      drone_0 was re-imported before episode 2; its speed in control steps 0-5 < 0.05 m/s
             and its climb in 1.5 s < 0.05 m; no check raised
  crash_l2   the drone's speed in control steps 0-5 < 0.05 m/s and its climb in 1.5 s < 0.05 m;
             no check raised

Usage (repo root, DISPLAY explicit, low priority, nothing else heavy running):
  DISPLAY=:2 OMP_NUM_THREADS=4 nice -n 19 python scripts/webots_l2_multi_speed.py --cases hover crosstalk determ speed delay badcmd
Writes logs/webots/c1_speed/<case>.json and a summary to stdout. Webots logs next to them.
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

CONFIG = 'configs/config_stage2_round7_eval_paired.yaml'
OUT = os.path.join(ROOT, 'logs', 'webots', 'c1_speed')
SPACING = 10.0                     # m between neighbouring drones, centred on x = 0
ALT = 1.5


# ------------------------------------------------------------------ helpers (worker side)
def load_cfg(n):
    cfg = yaml.safe_load(open(os.path.join(ROOT, CONFIG)))
    cfg['environment']['n_agents'] = n
    cfg['environment']['collision_physics'] = False
    cfg.setdefault('observation', {})
    return cfg


def make_multi(n, tag):
    from envs.webots_l2_multi import WebotsL2MultiAdapter
    cfg = load_cfg(n)
    cfg['webots'] = {'log': os.path.join(OUT, f'{tag}.webots.log')}
    return WebotsL2MultiAdapter(cfg)


def spawns_for(n):
    """x = 10 * (i - (n - 1) / 2): centred, so 8 drones stay within +-35 m of the 120 m floor."""
    sp = {f'drone_{i}': [SPACING * (i - (n - 1) / 2), 0.0, ALT] for i in range(n)}
    gl = {k: [v[0], v[1] + 5.0, v[2]] for k, v in sp.items()}     # goals only feed the LiDAR heading
    return sp, gl


def script(k, t):
    """Scripted world-frame velocity command for drone script k at time t (s)."""
    if k == 'hover':                                            # move, climb, then hold
        return [0.0, 0.5 if t < 1.0 else 0.0, 0.3 if 1.0 <= t < 2.0 else 0.0]
    if k == 0:
        return [1.0 if t < 2.0 else 0.0, 0.0, 0.0]
    if k == 1:
        return [0.0, -1.0 if t < 3.0 else 0.0, 0.5 if t < 1.0 else 0.0]
    return [0.8 * math.cos(t), 0.8 * math.sin(t), 0.0]          # k == 2: circle


def fly(ad, scripts, seconds):
    """Run the adapter with the scripted commands; returns positions (steps, n, 3) and
    per-step tilt (deg) and true speed."""
    n_steps = int(round(seconds / ad.control_dt))
    pos, tilt, speed = [], [], []
    for s in range(n_steps):
        t = s * ad.control_dt
        ad.apply_actions({f'drone_{i}': script(k, t) + [0.0] if k is not None else [0.0, 0.0, 0.0, 0.0]
                          for i, k in enumerate(scripts)})
        ad.step_simulation()
        pos.append(ad._pos.copy())
        tilt.append(np.degrees(np.abs(ad._orn[:, :2])).max(axis=1))
        speed.append(np.linalg.norm(ad._vel, axis=1))
    return np.array(pos), np.array(tilt), np.array(speed)


# ------------------------------------------------------------------ cases (worker side)
def case_hover(a):
    ad = make_multi(3, 'hover')
    sp, gl = spawns_for(3)
    ad.reset_world(sp, gl, [])
    pos, tilt, speed = fly(ad, ['hover'] * 3, 15.0)
    ad.close()
    start = np.array([sp[f'drone_{i}'] for i in range(3)])
    move = pos[-1] - start                                     # measured move, per drone
    k5 = len(pos) - int(round(5.0 / ad.control_dt))            # the last 5 s
    hold, hold_speed = pos[k5:], speed[k5:]
    r = {'move_y_m': move[:, 1].tolist(), 'move_z_m': move[:, 2].tolist(),
         'hold_horizontal_drift_m': np.linalg.norm(hold[:, :, :2] - hold[0, :, :2], axis=2).max(axis=0).tolist(),
         'hold_vertical_drift_m': np.abs(hold[:, :, 2] - hold[0, :, 2]).max(axis=0).tolist(),
         'hold_max_speed': hold_speed.max(axis=0).tolist(),
         'max_tilt_deg': tilt.max(axis=0).tolist(), 'max_speed': speed.max(axis=0).tolist()}
    moved = all(0.3 <= y <= 0.7 for y in r['move_y_m']) and all(0.15 <= z <= 0.45 for z in r['move_z_m'])
    holds = (max(r['hold_horizontal_drift_m']) < 0.05 and max(r['hold_vertical_drift_m']) < 0.05
             and max(r['hold_max_speed']) < 0.05)
    safe = max(r['max_tilt_deg']) <= 60 and max(r['max_speed']) <= 3
    r.update({'moved': bool(moved), 'holds': bool(holds), 'safe': bool(safe),
              'pass': bool(moved and holds and safe)})
    r['z_track_m'] = pos[:, :, 2].T.tolist()                   # per drone, every control step
    return r


def case_crosstalk_multi(a):
    ad = make_multi(3, 'crosstalk_multi')
    sp, gl = spawns_for(3)
    ad.reset_world(sp, gl, [])
    pos, _, _ = fly(ad, [0, 1, 2], 10.0)
    ad.close()
    return {'pos': pos.tolist()}


def case_crosstalk_solo(a):
    k = a.k
    ad = make_multi(1, f'crosstalk_solo{k}')
    sp, gl = spawns_for(3)
    ad.reset_world({'drone_0': sp[f'drone_{k}']}, {'drone_0': gl[f'drone_{k}']}, [])
    pos, _, _ = fly(ad, [k], 10.0)
    ad.close()
    return {'pos': pos[:, 0].tolist()}


def case_determ(a):
    ad = make_multi(3, f'determ_p{a.k}')
    sp, gl = spawns_for(3)
    runs = []
    for _ in range(2):
        ad.reset_world(sp, gl, [])
        runs.append(fly(ad, [0, 1, 2], 10.0)[0])
    ad.close()
    return {'run1': runs[0].tolist(), 'run2': runs[1].tolist()}


def case_speed(a):
    n = a.k
    t0 = time.time()
    ad = make_multi(n, f'speed_n{n}')
    startup = time.time() - t0
    sp, gl = spawns_for(n)
    resets = []
    for _ in range(2):
        t0 = time.time()
        ad.reset_world(sp, gl, [])
        resets.append(time.time() - t0)
    t0 = time.time()
    fly(ad, [None] * n, 20.0)
    wall = time.time() - t0
    ad.close()
    return {'n': n, 'startup_s': startup, 'reset_s': resets, 'wall_per_sim_s': wall / 20.0,
            'episode_1000_steps_est_s': wall / 20.0 * 32.0 + np.mean(resets)}


def _step_trace(ad, sample, seconds=1.5, t_step=0.3):
    """Hold, then command vx = 1 at t_step; sample(ad) every basic step. Returns (t, vx)."""
    out, n = [], int(round(seconds / ad.dt))
    for k in range(n):
        t = k * ad.dt
        out.append((t, sample(ad)))
        cmd = [1.0 if t >= t_step else 0.0, 0.0, 0.0, 0.0]
        if k % ad.n_sub == 0:                         # a new command once per control step
            ad.apply_actions({'drone_0': cmd})
        yield_step(ad)
    return out


def yield_step(ad):
    """Advance one basic step (4 ms) for either adapter."""
    if hasattr(ad, 'pid'):                            # WebotsL2Adapter: run its own loop for one sub-step
        n_sub = ad.n_sub
        ad.n_sub = 1
        ad.step_simulation()
        ad.n_sub = n_sub
    else:                                             # multi: the drone controllers run the PID
        ad.sup.step(ad.basic_ms)


def case_delay(a):
    sp, gl = spawns_for(1)                            # x = 0
    if a.k == 0:
        ad = make_multi(1, 'delay_new')
    else:
        from envs.webots_l2 import WebotsL2Adapter
        cfg = load_cfg(1)
        cfg['webots'] = {'real': True, 'log': os.path.join(OUT, 'delay_old.webots.log')}
        ad = WebotsL2Adapter(cfg)
    ad.reset_world(sp, gl, [])
    tr = _step_trace(ad, lambda d: d.nodes[0].getVelocity()[0])
    ad.close()
    t = np.array([x[0] for x in tr]); vx = np.array([x[1] for x in tr])
    cross = {}
    for thr in (0.05, 0.2, 0.5):
        k = np.flatnonzero(vx >= thr)
        cross[str(thr)] = float(t[k[0]]) if k.size else None
    return {'which': 'new' if a.k == 0 else 'old', 'cross_s': cross}


def case_badcmd(a):
    ad = make_multi(3, 'badcmd')
    sp, gl = spawns_for(3)
    ad.reset_world(sp, gl, [])
    fly(ad, [None] * 3, 1.0)                          # past the drop-check window (16 steps)
    ad._cd[1].setSFString('garbage')                  # malformed: controller must exit
    z0 = float(ad._pos[1, 2])
    ad.sup.step(ad.basic_ms)
    pos, _, _ = fly(ad, [None] * 3, 1.0)              # apply_actions overwrites drone_1's command again,
    fell = z0 - float(pos[-1, 1, 2])                  # but its controller has already exited
    caught = None
    try:
        ad.reset_world(sp, gl, [])
    except RuntimeError as e:
        caught = ('reset', str(e))
    if caught is None:
        try:
            ad.close()
        except RuntimeError as e:
            caught = ('close', str(e))
    else:
        try:
            ad.close()
        except RuntimeError:
            pass
    return {'drone_1_fell_m': fell, 'caught_at': caught[0] if caught else None,
            'message': caught[1] if caught else None,
            'pass': bool(caught and 'drone_1' in caught[1])}


def case_crash(a):
    ad = make_multi(3, 'crash')
    sp, gl = spawns_for(3)
    zero = {f'drone_{i}': [0.0, 0.0, 0.0, 0.0] for i in range(3)}
    climb = dict(zero, drone_0=[0.0, 0.0, 2.0, 0.0])
    ad.reset_world(sp, gl, [])
    k_flip = int(round(1.0 / ad.control_dt))
    for s in range(int(round(26.0 / ad.control_dt))):
        if s == k_flip:
            ad._rot[0].setSFRotation([1.0, 0.0, 0.0, math.pi])        # drone_0 upside down
        ad.apply_actions(climb if s >= k_flip else zero)
        ad.step_simulation()
    ep1 = {'drone0_final_z': float(ad._pos[0, 2]),
           'drone0_max_tilt_deg': float(np.degrees(max(ad._max_roll[0], ad._max_pitch[0])))}
    err, speed, z = None, [], []
    try:
        ad.reset_world(sp, gl, [])
        for s in range(int(round(1.5 / ad.control_dt))):
            ad.apply_actions(zero)
            ad.step_simulation()
            speed.append(float(np.linalg.norm(ad._vel[0])))
            z.append(float(ad._pos[0, 2]))
    except RuntimeError as e:
        err = str(e)
    log = [list(x) for x in ad.reimport_log]
    ad.close()
    climb_m = (max(z) - ALT) if z else None
    r = {'episode1': ep1, 'reimport_log': log, 'error': err,
         'drone0_speed_steps0_5': speed[:6], 'drone0_climb_m': climb_m}
    r['pass'] = bool(err is None and log and 0 in log[-1][1] and len(speed) >= 6
                     and max(speed[:6]) < 0.05 and climb_m < 0.05)
    return r


def case_crash_l2(a):
    """Single-drone Level 2 (WebotsL2Adapter, official Crazyflie): the same crash as 'crash'."""
    from envs.webots_l2 import WebotsL2Adapter
    cfg = load_cfg(1)
    cfg['webots'] = {'real': True, 'log': os.path.join(OUT, 'crash_l2.webots.log')}
    ad = WebotsL2Adapter(cfg)
    sp, gl = spawns_for(1)
    ad.reset_world(sp, gl, [])
    k_flip = int(round(1.0 / ad.control_dt))
    for s in range(int(round(26.0 / ad.control_dt))):
        if s == k_flip:
            ad._rot[0].setSFRotation([1.0, 0.0, 0.0, math.pi])
        ad.apply_actions({'drone_0': [0.0, 0.0, 2.0 if s >= k_flip else 0.0, 0.0]})
        ad.step_simulation()
    rotor_before = ad._rotor_est.round(1).tolist()
    err, speed, z = None, [], []
    try:
        ad.reset_world(sp, gl, [])
        for s in range(int(round(1.5 / ad.control_dt))):
            ad.apply_actions({'drone_0': [0.0, 0.0, 0.0, 0.0]})
            ad.step_simulation()
            speed.append(float(np.linalg.norm(ad._vel[0])))
            z.append(float(ad._pos[0, 2]))
    except RuntimeError as e:
        err = str(e)
    hold = ad.last_hold_s
    ad.close()
    climb_m = (max(z) - ALT) if z else None
    r = {'rotor_est_before_reset': rotor_before, 'hold_s': hold, 'error': err,
         'speed_steps0_5': speed[:6], 'climb_m': climb_m}
    r['pass'] = bool(err is None and len(speed) >= 6 and max(speed[:6]) < 0.05 and climb_m < 0.05)
    return r


CASES = {'hover': case_hover, 'crosstalk_multi': case_crosstalk_multi,
         'crosstalk_solo': case_crosstalk_solo, 'determ': case_determ, 'speed': case_speed,
         'delay': case_delay, 'badcmd': case_badcmd, 'crash': case_crash, 'crash_l2': case_crash_l2}


# ------------------------------------------------------------------ main side
def run_worker(case, k=0):
    """One case in its own process; returns its JSON result (or the error)."""
    out = os.path.join(OUT, f'{case}_{k}.json')
    cmd = [sys.executable, os.path.abspath(__file__), '--worker', case, '--k', str(k), '--out', out]
    t0 = time.time()
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if p.returncode != 0:
        return {'error': (p.stderr or p.stdout)[-2000:], 'wall_s': time.time() - t0}
    r = json.load(open(out))
    r['wall_s'] = time.time() - t0
    return r


def maxdiff(a, b):
    a, b = np.asarray(a), np.asarray(b)
    m = min(len(a), len(b))
    return float(np.linalg.norm(a[:m] - b[:m], axis=-1).max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cases', nargs='+', default=['hover', 'crosstalk', 'determ', 'speed', 'delay', 'badcmd', 'crash', 'crash_l2'])
    ap.add_argument('--worker', default=None)
    ap.add_argument('--k', type=int, default=0)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if a.worker:
        json.dump(CASES[a.worker](a), open(a.out, 'w'))
        return

    report = {}
    if 'hover' in a.cases:
        report['hover'] = run_worker('hover')
    if 'crosstalk' in a.cases:
        m = run_worker('crosstalk_multi')
        solo = [run_worker('crosstalk_solo', k) for k in range(3)]
        if 'error' in m or any('error' in s for s in solo):
            report['crosstalk'] = {'error': [m.get('error')] + [s.get('error') for s in solo]}
        else:
            pm = np.array(m['pos'])
            d = [maxdiff(pm[:, k], solo[k]['pos']) for k in range(3)]
            report['crosstalk'] = {'max_track_diff_m': d, 'pass': bool(max(d) < 0.01)}
    if 'determ' in a.cases:
        p0, p1 = run_worker('determ', 0), run_worker('determ', 1)
        if 'error' in p0 or 'error' in p1:
            report['determ'] = {'error': [p0.get('error'), p1.get('error')]}
        else:
            report['determ'] = {'same_process_max_m': maxdiff(p0['run1'], p0['run2']),
                                'across_process_max_m': maxdiff(p0['run1'], p1['run1'])}
    if 'speed' in a.cases:
        report['speed'] = [run_worker('speed', n) for n in (1, 3, 8)]
    if 'delay' in a.cases:
        new, old = run_worker('delay', 0), run_worker('delay', 1)
        r = {'new': new, 'old': old}
        if 'error' not in new and 'error' not in old:
            r['new_minus_old_ms'] = {thr: (None if new['cross_s'][thr] is None or old['cross_s'][thr] is None
                                           else 1000 * (new['cross_s'][thr] - old['cross_s'][thr]))
                                     for thr in new['cross_s']}
        report['delay'] = r
    if 'badcmd' in a.cases:
        report['badcmd'] = run_worker('badcmd')
    if 'crash' in a.cases:
        report['crash'] = run_worker('crash')
    if 'crash_l2' in a.cases:
        report['crash_l2'] = run_worker('crash_l2')
    json.dump(report, open(os.path.join(OUT, 'report.json'), 'w'), indent=2)
    shown = {k: ({x: y for x, y in v.items() if not x.endswith('_track_m')} if isinstance(v, dict) else v)
             for k, v in report.items()}                       # tracks stay in the JSON files
    print(json.dumps({k: (v if k != 'speed' else [{x: s.get(x) for x in ('n', 'wall_per_sim_s', 'reset_s',
                     'episode_1000_steps_est_s', 'error')} for s in v]) for k, v in shown.items()}, indent=2))


if __name__ == '__main__':
    main()
