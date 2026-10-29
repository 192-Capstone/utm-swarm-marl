"""
check_dynamic_events.py
-----------------------
Short PyBullet check for D5 drone failures (environment.failures, envs/disturbances.py), before
they are hooked into scripts/eval_layered.py fly() (after PR #37). Evaluation only.

The scenario is E1's (as eval_layered.py, mode e1): a city map, random requests, E1-approved
flights flown along their booked routes from their booked departure times (schedule,
route_follow, crash_mode remove), by the learned policy (+ goal hold) or the simple pilot.

  --part identity   hash every step (obs, rewards, done, info) with no failures block and with
                    failures: {enabled: false}; run once per code tree (--root: e.g. a git
                    archive of dev) and compare the printed hashes: they must all be equal.
                    Also with failures enabled but no schedule (this tree only): obs / rewards /
                    done and info without the three new keys must be equal too.
  --part eval       the same flights twice: without failures, and with one failure of each type
                    (motor, land, comms) on three flights picked at random, at a random time in
                    the middle of their booked flight. A perfect-link TelemetryModel stands in for
                    what the UTM hears (reports once per second; the comms-lost drone is muted; a
                    forced-landing drone reports status 'landing'; a motor failure sends one extra
                    report with status 'emergency' at once (if env.emergency_report), since its
                    fall is over before the next period). Prints what happened.

    python scripts/check_dynamic_events.py --part identity --root /path/to/dev_archive
    python scripts/check_dynamic_events.py --part eval

Seeds 10-19 are refused (reserved for test runs).
"""

import argparse
import hashlib
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CKPT = os.path.expanduser(
    '~/projects/utm-swarm-marl/checkpoints/stage2_r1_crowd_rl_r7v3ft_s0_lab/checkpoint_final.pth')
NEW_INFO = ('drone_failed', 'drone_forced_down', 'drone_comms_lost')


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--part', choices=['identity', 'eval'], required=True)
    ap.add_argument('--root', default=os.path.dirname(HERE), help='code tree to import from')
    ap.add_argument('--config', default='configs/config_stage2_round7_eval_goalhold.yaml', help='relative to --root')
    ap.add_argument('--checkpoint', default=DEFAULT_CKPT)
    ap.add_argument('--pilot', choices=['learned', 'simple'], default='learned')
    ap.add_argument('--map', default='city')
    ap.add_argument('--level', default='medium')
    ap.add_argument('--seed', type=int, default=20)
    ap.add_argument('--flights', type=int, default=8)
    ap.add_argument('--window', type=float, default=30.0)
    ap.add_argument('--steps', type=int, default=None, help='stop after this many steps (default: episode end)')
    ap.add_argument('--variants', nargs='+', default=None, help='identity: none off on_empty')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    if 10 <= a.seed <= 19:
        ap.error('seeds 10-19 are reserved for test runs')
    return a


args = parse()
ROOT = os.path.abspath(args.root)
sys.path[:] = [p for p in sys.path if os.path.abspath(p or '.') != HERE]
sys.path[:0] = [ROOT, os.path.join(ROOT, 'scripts')]

import numpy as np                                                 # noqa: E402
import torch                                                       # noqa: E402
import yaml                                                        # noqa: E402

import eval_suite as es                                            # noqa: E402
from eval_e1_flown import plans                                    # noqa: E402
from envs.swarm_env import SwarmEnv                                # noqa: E402
from modules.goal_hold import GoalHoldPolicy, goal_hold_enabled    # noqa: E402
from preview_route import load_map                                 # noqa: E402
from preview_strategic import random_requests                      # noqa: E402
from record_rollout import disable_all_collision_physics           # noqa: E402
from utm.strategic_planner import StrategicPlanner                 # noqa: E402


class SimplePilot:
    """As eval_layered.py's: straight at the carrot (env.adapter.goals)."""

    def __init__(self, kp=1.5, v_max=1.5):
        self.kp, self.v_max = kp, v_max

    def act(self, obs, env, n):
        pos = env.adapter.get_positions()
        out = {}
        for i in range(n):
            v = self.kp * (np.asarray(env.adapter.goals[f'drone_{i}'], dtype=float) - pos[i])
            sp = float(np.linalg.norm(v))
            if sp > self.v_max:
                v *= self.v_max / sp
            out[f'drone_{i}'] = [*v.tolist(), 0.0]
        return out


def scenario():
    with open(os.path.join(ROOT, args.config), encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    cfg.pop('scenario_mix', None)
    cfg['environment']['collision_physics'] = False
    cfg.setdefault('obstacles', {})['enabled'] = False
    layout, bounds = load_map(args)
    sky = StrategicPlanner(layout, bounds)
    reqs = random_requests(np.random.RandomState(args.seed), sky, args.flights, window=args.window)
    _, flights = plans(sky, reqs)
    return cfg, layout, sky, flights


def make_env(cfg, layout, sky, flights, failures_block):
    n = len(flights)
    cfg = json.loads(json.dumps(cfg))
    cfg['environment'].update(n_agents=n, crash_mode='remove', schedule={'enabled': True},
                              route_follow={'enabled': True})
    if failures_block is not None:
        cfg['environment']['failures'] = failures_block
    dt = cfg['environment'].get('control_timestep', 0.032)
    deps = [f['e1'][0] for f in flights]
    routes = [f['e1'][1] for f in flights]
    longest = max(sum(math.dist(a[:2], b[:2]) for a, b in zip(r, r[1:])) for r in routes)
    cfg['environment']['max_steps'] = int((max(deps) + 3.0 * longest + 30.0) / dt)
    env = SwarmEnv(cfg, device=torch.device('cpu'))
    disable_all_collision_physics(env, n)
    if args.pilot == 'simple':
        policy = SimplePilot(v_max=sky.s.v_cruise)
    else:
        policy = es.MappoPolicy(cfg, args.checkpoint, torch.device('cpu'), True)
        policy.restore_normalizer(env)
        if goal_hold_enabled(cfg):
            policy = GoalHoldPolicy(policy, cfg)
    return cfg, env, policy, deps, routes, dt


def start(env, flights, sky, deps, routes, failures=None):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    z0 = sky.s.start_z
    key = [f'drone_{i}' for i in range(len(flights))]
    kw = {} if failures is None else {'failures': failures}
    return env.start_episode({k: [*f['start'], z0] for k, f in zip(key, flights)},
                             {k: [*f['goal'], z0] for k, f in zip(key, flights)},
                             obstacles=sky_layout, routes=dict(zip(key, routes)),
                             departures=dict(zip(key, deps)), **kw)


def digest(h, obs, rew, done, info, drop_new):
    for k in sorted(obs):
        h.update(k.encode())
        h.update(obs[k].detach().cpu().numpy().tobytes())
    h.update(json.dumps(rew, sort_keys=True).encode())
    h.update(str(bool(done)).encode())
    if info is not None:
        if drop_new:
            info = {k: v for k, v in info.items() if k not in NEW_INFO}
        h.update(json.dumps(info, sort_keys=True, default=repr).encode())


# ---------------------------------------------------------------- identity
def identity():
    cfg, layout, sky, flights = scenario()
    global sky_layout
    sky_layout = layout
    variants = args.variants or ['none', 'off']
    blocks = {'none': None, 'off': {'enabled': False, 'land_vz': 0.3, 'p_emergency_report': 0.0},
              'on_empty': {'enabled': True}}
    out = {}
    for v in variants:
        t0 = time.time()
        cfg_v, env, policy, deps, routes, dt = make_env(cfg, layout, sky, flights, blocks[v])
        n = len(flights)
        obs = start(env, flights, sky, deps, routes)
        full, core = hashlib.sha256(), hashlib.sha256()
        digest(full, obs, {}, False, None, False)
        digest(core, obs, {}, False, None, True)
        limit = args.steps or cfg_v['environment']['max_steps']
        for step in range(limit):
            obs, rew, done, info = env.step(policy.act(obs, env, n))
            digest(full, obs, rew, done, info, False)
            digest(core, obs, rew, done, info, True)
            if done:
                break
        reached = int(sum(info.get('drone_reached_goal', [])))
        env.close()
        out[v] = {'steps': step + 1, 'full': full.hexdigest(), 'core': core.hexdigest(), 'reached': reached,
                  'crashed': int(env.crashed.sum()), 'seconds': round(time.time() - t0, 1)}
        print(f"{ROOT}  {v:8s} steps {step + 1:5d}  reached {reached}/{n}  crashed {out[v]['crashed']}  "
              f"full {out[v]['full'][:16]}  core {out[v]['core'][:16]}  ({out[v]['seconds']} s)", flush=True)
    return {'root': ROOT, 'flights': len(flights), 'variants': out}


# ---------------------------------------------------------------- eval
def pick_failures(flights, dt):
    from envs.disturbances import draw_failures
    rng = np.random.RandomState(3000 + args.seed)
    who = rng.choice(len(flights), 3, replace=False)
    out = []
    for i, kind in zip(who, ('motor', 'land', 'comms')):
        f = flights[i]
        out += draw_failures(rng, [(f'drone_{i}', f['e1'][0], f['arrival']['e1'])], 1.0, types=(kind,),
                             frac=(0.3, 0.7))
    return out


def run_eval(cfg, layout, sky, flights, failures):
    from utm.telemetry import TelemetryModel
    block = {'enabled': True} if failures is not None else None
    cfg_v, env, policy, deps, routes, dt = make_env(cfg, layout, sky, flights, block)
    n = len(flights)
    ids = [f['id'] for f in flights]
    obs = start(env, flights, sky, deps, routes, failures)
    tm = TelemetryModel()
    every = max(1, int(round(1.0 / dt)))
    prev, sent_emergency = {}, set()
    track = {}                                   # failed drone -> per-step (t, pos)
    fail_idx = {int(f['drone'].split('_')[1]): f for f in (failures or [])}
    min_sep, close, info = np.inf, {}, {}
    crash_t = {}
    t0 = time.time()
    for step in range(cfg_v['environment']['max_steps']):
        acts = policy.act(obs, env, n)
        obs, _, done, info = env.step(acts)
        t = (step + 1) * dt
        air = np.flatnonzero(~(env.waiting | env.landed | env.crashed))
        pos = env.adapter.get_positions()
        for i in fail_idx:
            if i in air:
                track.setdefault(i, []).append((t, pos[i].copy()))
        for i in np.flatnonzero(env.crashed):
            crash_t.setdefault(int(i), t)
        for i in fail_idx:                       # a motor failure's emergency report goes out at once:
            if (env.failed[i] == 'motor' and env.emergency_report[i] and i not in sent_emergency
                    and i in air):               # the fall (< 1 s) is over before the next period
                sent_emergency.add(i)
                v_now = np.asarray(env.adapter.get_raw_state()[f'drone_{i}']['velocity'])
                tm.send(t, {ids[i]: (pos[i], v_now, [], 'emergency')})
                tm.receive(t)
        if len(air) > 1:
            p = pos[air]
            d = np.linalg.norm(p[:, None, :] - p[None, :, :], axis=2)
            iu = np.triu_indices(len(air), 1)
            min_sep = min(min_sep, float(d[iu].min()))
            for k in np.flatnonzero(d[iu] < 1.4):
                close.setdefault((int(air[iu[0][k]]), int(air[iu[1][k]])), round(t, 2))
        if (step + 1) % every == 0:
            states = {}
            for i in air:
                vel = (pos[i] - prev[i]) / (every * dt) if i in prev else np.zeros(3)
                if env.failed[i] == 'comms':
                    tm.mute(ids[i])
                status = 'ok'
                if env.failed[i] == 'motor':
                    status = 'emergency'
                elif env.failed[i] == 'land':
                    status = 'landing'
                states[ids[i]] = (pos[i], vel, [], status)
            tm.send(t, states)
            tm.receive(t)
            prev = {i: pos[i].copy() for i in air}
        if done:
            break
    env.close()
    reached = np.asarray(info.get('drone_reached_goal', [False] * n), dtype=bool)
    arrived = [bool(reached[i] and not env.forced_down[i]) for i in range(n)]
    res = {'steps': step + 1, 'sim_s': round(t, 1), 'reason': info.get('reason'),
           'arrived': int(sum(arrived)), 'crashed': int(env.crashed.sum()),
           'forced_down': int(env.forced_down.sum()), 'min_separation_m': round(min_sep, 3),
           'pairs_below_1.4': len(close), 'close_pairs': [[ids[a], ids[b], tc] for (a, b), tc in close.items()],
           'skipped': env.failures_skipped, 'wall_s': round(time.time() - t0, 1),
           'telemetry': {'sent': tm.sent, 'lost': tm.lost, 'muted': tm.muted}}
    if failures is not None:
        res['failures'] = []
        for i, f in fail_idx.items():
            tr = track.get(i, [])
            at = [p for tt, p in tr if tt >= f['t'] - 1e-9]
            row = {'flight': ids[i], 'drone': f['drone'], 'type': f['type'], 't': round(f['t'], 2),
                   'fired': env.failed[i] == f['type'], 'arrived': arrived[i], 'crashed': bool(env.crashed[i]),
                   'forced_down': bool(env.forced_down[i]), 'z_at_failure': round(float(at[0][2]), 2) if at else None}
            if f['type'] == 'motor':
                row['down_after_s'] = round(crash_t[i] - f['t'], 2) if i in crash_t else None
                row['emergency_report'] = bool(env.emergency_report[i])
                row['others_crashed_with_it'] = [ids[j] for j, tc in crash_t.items()
                                                 if j != i and abs(tc - crash_t.get(i, -9)) < 0.1]
            if f['type'] == 'land' and at:
                tgt = env._forced_target[i]
                last = tr[-1][1]
                zs = np.array([p[2] for p in at])
                row.update(target=[round(float(v), 2) for v in tgt],
                           landed_xy_off_m=round(float(np.hypot(*(last[:2] - at[0][:2]))), 2),
                           landed_z=round(float(last[2]), 2),
                           descent_s=round(tr[-1][0] - f['t'], 1),
                           max_down_speed=round(float(-np.diff(zs).min() / dt), 2) if len(zs) > 1 else None)
            r = tm.latest.get(ids[i])
            row['utm_last_report'] = None if r is None else {'t_sent': round(r.t_sent, 1), 'status': r.status}
            res['failures'].append(row)
    return res


def evaluate():
    cfg, layout, sky, flights = scenario()
    global sky_layout
    sky_layout = layout
    dt = cfg['environment'].get('control_timestep', 0.032)
    fails = pick_failures(flights, dt)
    print(f"{len(flights)} approved flights; failures: "
          + ', '.join(f"{f['drone']} ({flights[int(f['drone'][6:])]['id']}) {f['type']} at {f['t']:.1f} s" for f in fails))
    out = {'flights': len(flights), 'failures': fails}
    for name, fl in (('baseline', None), ('failures', fails)):
        r = run_eval(cfg, layout, sky, flights, fl)
        out[name] = r
        print(f"\n{name}: arrived {r['arrived']}/{len(flights)}, crashed {r['crashed']}, forced down "
              f"{r['forced_down']}, pairs < 1.4 m {r['pairs_below_1.4']}, closest {r['min_separation_m']} m, "
              f"reason {r['reason']}, {r['sim_s']} s sim in {r['wall_s']} s; telemetry {r['telemetry']}", flush=True)
        for row in r.get('failures', []):
            print('   ', json.dumps(row))
        if r['skipped']:
            print('    skipped:', r['skipped'])
    return out


if __name__ == '__main__':
    sky_layout = None
    res = identity() if args.part == 'identity' else evaluate()
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, 'w') as f:
            json.dump({'args': vars(args), **res}, f, indent=1, default=str)
        print(f"wrote {args.out}")
