"""
eval_suite.py — Evaluation suite v1 for the frozen Stage-1 baseline.

Evaluation only. Nothing here changes training code, configs, checkpoints or
the normalizer. One invocation = one experiment (one W&B run if --wandb).

What it adds on top of eval_checkpoint.py:
  - several seeds in one run (--seeds 0 1 2 3 4)
  - MAPPO or the P-controller through the exact same loop (--policy)
  - goal-distance sweep (--goal-dist 5 / 10 / 20)
  - zero-shot agent-count override (--n-agents 1..10)
  - collision physics on or off, plus PyBullet contact events when on
  - per-episode and per-agent CSVs, a summary.json, optional W&B logging
  - separation-threshold counts (0.2/0.3/0.4/0.5 m), path efficiency,
    time to settle, Wilson 95% intervals, observation clip rate

Pairing: for a given (scenario, seeds, n_agents, goal-dist), MAPPO and the
P-controller see identical spawns and goals. Each episode row carries a
scenario_id hash so the pairing can be checked after the fact.

Examples:
    # 3 agents, random 2 m tasks, 5 seeds x 100 episodes, policy-only safety
    python eval_suite.py --config configs/config_3agent_review.yaml \
        --checkpoint checkpoints/stage1_v1/best_eval_policy.pth \
        --seeds 0 1 2 3 4 --episodes 100 --scenario random --collision-physics off

    # Same scenarios, P-controller
    python eval_suite.py --config configs/config_3agent_review.yaml \
        --policy pcontroller --seeds 0 1 2 3 4 --episodes 100 --scenario random \
        --collision-physics off

    # Distance sweep point: goals 5-10 m away
    python eval_suite.py ... --goal-dist 10

    # Zero-shot scaling point: 8 agents
    python eval_suite.py ... --n-agents 8
"""

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time

import numpy as np
import pybullet as p
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from training.networks.actor_network import ActorNetwork
from envs.swarm_env import SwarmEnv
from baselines.repulsion_controller import RepulsionController
from envs import scenarios
from record_rollout import (setup_crossing_scenario, setup_swap_scenario,
                            setup_headon_scenario, disable_all_collision_physics,
                            _apply_scenario)

SUITE_VERSION = "v1"
SEP_THRESHOLDS = (0.2, 0.3, 0.4, 0.5)


# ---------------------------------------------------------------- helpers ---
def wilson(k, n, z=1.96):
    """95% Wilson score interval for k successes out of n. Behaves sensibly
    at k=0 and k=n, unlike the normal approximation."""
    if n == 0:
        return (float('nan'), float('nan'))
    phat = k / n
    denom = 1 + z * z / n
    centre = (phat + z * z / (2 * n)) / denom
    half = z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def pct(values, q):
    v = [x for x in values if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return float(np.percentile(v, q)) if v else float('nan')


def mean(values):
    v = [x for x in values if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return float(np.mean(v)) if v else float('nan')


def scenario_hash(env):
    """Short hash of this episode's spawns + goals (rounded to mm)."""
    pos = np.round(env.adapter.get_positions(), 3)
    goals = np.round(env.goals, 3)
    return hashlib.sha1(np.concatenate([pos.ravel(), goals.ravel()]).tobytes()).hexdigest()[:10]


def sample_random_band(env, rng, n_agents, d_min, d_max):
    """Random scenario like SwarmEnv._sample_positions, but with each goal
    d_min..d_max metres from its own spawn (training used 1..2 m). Spawns use
    the same box and separation rule as training. Used only for the distance
    sweep, so the in-distribution path stays exactly env.reset()."""
    reach = (env.world_size / 2.0) * 0.4

    def rand_xyz():
        return np.array([rng.uniform(-reach, reach), rng.uniform(-reach, reach),
                         rng.uniform(1.0, 5.0)], dtype=np.float32)

    placed = []
    for _ in range(n_agents):
        cand = rand_xyz()
        for _ in range(100):
            if all(np.linalg.norm(cand - q) >= env.min_spawn_sep for q in placed):
                break
            cand = rand_xyz()
        placed.append(cand)

    goals = []
    for s in placed:
        for _ in range(1000):
            g = rand_xyz()
            if d_min <= np.linalg.norm(g - s) <= d_max:
                break
        else:
            # Fallback: exactly d_max away in a random horizontal direction
            ang = rng.uniform(0, 2 * np.pi)
            g = s + np.array([np.cos(ang), np.sin(ang), 0.0], dtype=np.float32) * d_max
        goals.append(g)

    spawns = {f'drone_{i}': placed[i].tolist() for i in range(n_agents)}
    goal_d = {f'drone_{i}': goals[i].tolist() for i in range(n_agents)}
    return _apply_scenario(env, spawns, goal_d, n_agents)


def drone_contact(env, n_agents):
    """True if PyBullet reports any physical contact between two drones."""
    ids, cid = env.adapter.drone_ids, env.adapter.client
    for i in range(n_agents):
        for j in range(i + 1, n_agents):
            if p.getContactPoints(bodyA=ids[i], bodyB=ids[j], physicsClientId=cid):
                return True
    return False


# ------------------------------------------------------------- policies ----
class MappoPolicy:
    def __init__(self, config, ckpt_path, device, deterministic, blind_neighbors=False):
        self.device = device
        self.deterministic = deterministic
        # Ablation: hide every neighbour from the actor (all slots masked, as
        # if flying alone). Collision differences vs the normal run show how
        # much of the avoidance actually comes from seeing other drones.
        self.blind_neighbors = blind_neighbors
        self.actor = ActorNetwork(config).to(device)
        self.ckpt = torch.load(ckpt_path, map_location=device)
        state = self.ckpt.get('actor_state_dict', self.ckpt)
        self.actor.load_state_dict(state)
        self.actor.eval()
        # Same restore rules as eval_checkpoint.py: the ceiling/floor are
        # plain attributes, not in the state_dict.
        if 'log_std_ceiling' in self.ckpt:
            self.actor.set_log_std_ceiling(self.ckpt['log_std_ceiling'])
        if 'log_std_floor' in self.ckpt:
            self.actor.set_log_std_floor(self.ckpt['log_std_floor'])
        self.n_active = 3 if self.ckpt.get('curriculum_stage', 1) < 3 else 4

    def restore_normalizer(self, env):
        if 'normalizer_mean' not in self.ckpt:
            raise RuntimeError("Checkpoint has no normalizer state. Suite v1 requires "
                               "a checkpoint saved with normalizer_mean/var/count.")
        norm = env.obs_processor.normalizer
        norm.mean = self.ckpt['normalizer_mean']
        norm.var = self.ckpt['normalizer_var']
        norm.count = self.ckpt['normalizer_count']
        norm.freeze()

    def act(self, obs, env, n_agents):
        lf = obs['local_fixed'].to(self.device)
        ns = obs['neighbor_states'].to(self.device)
        nm = obs['neighbor_mask'].to(self.device)
        if self.blind_neighbors:
            ns = torch.zeros_like(ns)
            nm = torch.ones_like(nm)
        with torch.no_grad():
            actions, _ = self.actor.get_action(lf, ns, nm, deterministic=self.deterministic,
                                               active_dims=self.n_active)
        actions = actions.clone()
        actions[:, self.n_active:] = 0.0
        arr = actions.cpu().numpy()
        return {f'drone_{i}': arr[i].tolist() for i in range(n_agents)}


class PController:
    """Same law as baseline_pcontroller.py: v = clip(kp * error, ±v_max)."""
    def __init__(self, config, kp):
        self.kp = kp
        self.v_max = config['environment'].get('v_max', 2.0)

    def act(self, obs, env, n_agents):
        pos = env.adapter.get_positions()
        return {f'drone_{i}': np.clip(self.kp * (env.goals[i] - pos[i]),
                                      -self.v_max, self.v_max).tolist() + [0.0]
                for i in range(n_agents)}


# ------------------------------------------------- interaction scenarios ---
# Two drones. drone_0 is the "ego" run by the policy under test. drone_1 is
# scripted and ignores everyone, so a collision happens unless the ego
# itself reacts. Each one is built so that flying style alone can't avoid it.
#   intruder  : ego hovers on its goal, drone_1 flies straight through that
#               exact point at 0.5-2 m/s and carries on past it
#   blocker   : drone_1 hovers exactly on the path the ego would fly if it
#               ignored other drones (40-60% of the way along, 3-6 m trip).
#               The path is recorded first with the ego's neighbour view off,
#               so curved or angled paths can't miss it by accident.
#   neighbour : ego hovers on its goal, drone_1 sits frozen 0.3-0.5 m away,
#               the controlled version of dragging a drone next to another
INTERACTION_SCENARIOS = ("intruder", "blocker", "neighbour")


def record_unaware_path(env, policy, start, goal, max_steps):
    """Fly the ego from start to goal with its neighbour view switched off
    (drone_1 parked far away) and return the positions it passes through."""
    park = (start + np.array([40.0, 40.0, 0.0])).tolist()
    obs = _apply_scenario(env, {'drone_0': start.tolist(), 'drone_1': park},
                          {'drone_0': goal.tolist(), 'drone_1': park}, 2)
    was_blind = getattr(policy, 'blind_neighbors', False)
    policy.blind_neighbors = True
    path = [start.copy()]
    for _ in range(max_steps):
        acts = policy.act(obs, env, 2)
        acts['drone_1'] = [0.0, 0.0, 0.0, 0.0]
        obs, _, _, info = env.step(acts)
        path.append(env.adapter.get_positions()[0].copy())
        if info['drone_reached_goal'][0]:
            break
    policy.blind_neighbors = was_blind
    return np.array(path)


def point_along(path, frac):
    """Point at `frac` of the total arc length of a polyline."""
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    target = frac * cum[-1]
    i = min(int(np.searchsorted(cum, target)), len(path) - 1)
    return path[i].copy()


def setup_interaction(env, rng, scenario, dt, max_steps, policy):
    z = rng.uniform(2.0, 3.0)
    ang = rng.uniform(0, 2 * np.pi)
    u = np.array([np.cos(ang), np.sin(ang), 0.0])
    centre = np.array([rng.uniform(-3, 3), rng.uniform(-3, 3), z])

    if scenario == "intruder":
        speed = rng.uniform(0.5, 2.0)
        goal, start, end = centre, centre - 5.0 * u, centre + 5.0 * u
        spawns, goals = {'drone_0': goal, 'drone_1': start}, {'drone_0': goal, 'drone_1': end}
        script = {'kind': 'fly', 'end': end, 'speed': speed}
        horizon = int(np.ceil((10.0 / speed + 3.0) / dt))     # pass + 3 s to recover
        param = round(speed, 3)
    elif scenario == "blocker":
        length, frac = rng.uniform(3.0, 6.0), rng.uniform(0.4, 0.6)
        start, goal = centre - length / 2 * u, centre + length / 2 * u
        path = record_unaware_path(env, policy, start, goal, max_steps)
        block = point_along(path, frac)
        spawns, goals = {'drone_0': start, 'drone_1': block}, {'drone_0': goal, 'drone_1': block}
        script = {'kind': 'hold', 'pos': block, 'path': path}
        horizon = max_steps
        param = round(length, 3)
    else:  # neighbour
        r = rng.uniform(0.3, 0.5)
        goal, other = centre, centre + r * u
        spawns, goals = {'drone_0': goal, 'drone_1': other}, {'drone_0': goal, 'drone_1': other}
        script = {'kind': 'hold', 'pos': other}
        horizon = int(np.ceil(4.0 / dt))                        # watch for 4 s
        param = round(r, 3)

    spawns = {k: v.tolist() for k, v in spawns.items()}
    goals = {k: v.tolist() for k, v in goals.items()}
    obs = _apply_scenario(env, spawns, goals, 2)
    return obs, script, min(horizon, max_steps), param


def scripted_velocity(script, pos):
    """Velocity for the scripted drone_1. 'fly' heads for its end point at a
    constant speed (slowing only in the last few cm); 'hold' stays put."""
    target = script['end'] if script['kind'] == 'fly' else script['pos']
    err = target - pos
    dist = np.linalg.norm(err)
    if dist < 1e-6:
        return [0.0, 0.0, 0.0]
    cap = script['speed'] if script['kind'] == 'fly' else 2.0
    return (err / dist * min(cap, 3.0 * dist)).tolist()


def run_interaction_episode(env, policy, obs, scenario, script, horizon):
    goal = env.goals[0].copy()
    pos = env.adapter.get_positions()
    start0, start1 = pos[0].copy(), pos[1].copy()
    min_sep, max_disp, max_lateral, disp_at_closest = np.inf, 0.0, 0.0, 0.0
    collided, steps, info = False, 0, {}

    for _ in range(horizon):
        acts = policy.act(obs, env, 2)
        acts['drone_1'] = scripted_velocity(script, env.adapter.get_positions()[1]) + [0.0]
        obs, _, done, info = env.step(acts)
        steps += 1
        pos = env.adapter.get_positions()
        sep = float(np.linalg.norm(pos[0] - pos[1]))
        disp = float(np.linalg.norm(pos[0] - goal))
        if sep < min_sep:
            min_sep, disp_at_closest = sep, disp
        max_disp = max(max_disp, disp)
        if scenario == "blocker":
            # how far the ego strays from the path it flies when ignoring others
            max_lateral = max(max_lateral,
                              float(np.min(np.linalg.norm(script['path'] - pos[0], axis=1))))
        if any(info.get('drone_collision', [])):
            collided = True
            break
        if scenario == "blocker" and done:
            break

    final_dist = float(np.linalg.norm(pos[0] - goal))
    final_speed = float(info['per_agent_speed'][0])
    row = {
        'collided': int(collided),
        'min_sep': round(min_sep, 4),
        'steps': steps,
        'final_goal_dist': round(final_dist, 4),
        'final_speed': round(final_speed, 4),
    }
    if scenario == "intruder":
        row.update({'max_disp': round(max_disp, 4), 'disp_at_closest': round(disp_at_closest, 4),
                    'back_on_goal': int(not collided and final_dist < 0.5 and final_speed < 0.15)})
    elif scenario == "blocker":
        row.update({'max_lateral': round(max_lateral, 4),
                    'reached_goal': int(not collided and bool(info['drone_reached_goal'][0]))})
    else:
        away = (start0 - start1) / np.linalg.norm(start0 - start1)
        row.update({'max_disp': round(max_disp, 4),
                    'moved_away': round(float(np.dot(pos[0] - goal, away)), 4),
                    'sep_change': round(float(np.linalg.norm(pos[0] - pos[1]) -
                                              np.linalg.norm(start0 - start1)), 4)})
    return row


def summarize_interaction(episodes, scenario):
    n = len(episodes)
    k = sum(e['collided'] for e in episodes)
    seps = [e['min_sep'] for e in episodes]
    s = {'kind': 'interaction', 'episodes': n,
         'collision_rate': k / n, 'collision_ci95': wilson(k, n),
         'min_sep': {'min': pct(seps, 0), 'p5': pct(seps, 5), 'median': pct(seps, 50)},
         'sep_below_0.3_rate': mean([int(x < 0.3) for x in seps]),
         'sep_below_0.5_rate': mean([int(x < 0.5) for x in seps])}
    if scenario == "intruder":
        s['moved_over_0.1m_rate'] = mean([int(e['max_disp'] > 0.1) for e in episodes])
        s['max_disp_median'] = pct([e['max_disp'] for e in episodes], 50)
        s['disp_at_closest_median'] = pct([e['disp_at_closest'] for e in episodes], 50)
        s['back_on_goal_rate'] = mean([e['back_on_goal'] for e in episodes])
        s['by_speed'] = {}
        for lo, hi in ((0.5, 1.0), (1.0, 1.5), (1.5, 2.01)):
            sub = [e for e in episodes if lo <= e['param'] < hi]
            if sub:
                kk = sum(e['collided'] for e in sub)
                s['by_speed'][f"{lo:g}-{min(hi, 2):g} m/s"] = {
                    'episodes': len(sub), 'collision_rate': kk / len(sub),
                    'min_sep_median': pct([e['min_sep'] for e in sub], 50)}
    elif scenario == "blocker":
        k2 = sum(e['reached_goal'] for e in episodes)
        s['reached_goal_rate'], s['reached_goal_ci95'] = k2 / n, wilson(k2, n)
        s['max_lateral_median'] = pct([e['max_lateral'] for e in episodes], 50)
    else:
        s['moved_over_0.1m_rate'] = mean([int(e['max_disp'] > 0.1) for e in episodes])
        s['moved_away_mean'] = mean([e['moved_away'] for e in episodes])
        s['sep_change_mean'] = mean([e['sep_change'] for e in episodes])
        s['sep_change_median'] = pct([e['sep_change'] for e in episodes], 50)
    return s


def print_interaction_summary(s, tag, scenario):
    lo, hi = s['collision_ci95']
    ms = s['min_sep']
    print(f"\n{'=' * 64}\n{tag}\n{'=' * 64}")
    print(f"episodes                 {s['episodes']}")
    print(f"ego collided             {s['collision_rate']:.3f}  [95% CI {lo:.3f}-{hi:.3f}]")
    print(f"min separation  min/p5/median  {ms['min']:.3f} / {ms['p5']:.3f} / {ms['median']:.3f} m")
    print(f"episodes with sep < 0.3 / 0.5 m  {s['sep_below_0.3_rate']:.3f} / {s['sep_below_0.5_rate']:.3f}")
    if scenario == "intruder":
        print(f"ego moved > 0.1 m        {s['moved_over_0.1m_rate']:.3f}")
        print(f"ego max displacement     median {s['max_disp_median']:.3f} m "
              f"(at closest approach {s['disp_at_closest_median']:.3f} m)")
        print(f"back on goal at the end  {s['back_on_goal_rate']:.3f}")
        for b, v in s['by_speed'].items():
            print(f"  intruder {b:12s} collisions {v['collision_rate']:.3f} "
                  f"(n={v['episodes']}), median min sep {v['min_sep_median']:.3f} m")
    elif scenario == "blocker":
        lo, hi = s['reached_goal_ci95']
        print(f"ego reached goal         {s['reached_goal_rate']:.3f}  [95% CI {lo:.3f}-{hi:.3f}]")
        print(f"max sideways detour      median {s['max_lateral_median']:.3f} m")
    else:
        print(f"ego moved > 0.1 m        {s['moved_over_0.1m_rate']:.3f}")
        print(f"moved away from it       mean {s['moved_away_mean']:+.3f} m")
        print(f"separation change        mean {s['sep_change_mean']:+.3f} m, "
              f"median {s['sep_change_median']:+.3f} m")


# ----------------------------------------------------------------- main ----
def run_episode(env, policy, obs, n_agents, max_steps, dt, physics_on, clip_val):
    pos_prev = env.adapter.get_positions().copy()
    path_len = np.zeros(n_agents)
    settle_step = [None] * n_agents
    entry_step = [None] * n_agents
    min_sep = None
    contact = False
    any_drone_collision = False
    clip_hits, clip_total = 0, 0
    info, steps = {}, 0

    for step in range(max_steps):
        lf = obs['local_fixed']
        clip_hits += int((lf.abs() >= clip_val - 1e-4).sum().item())
        clip_total += lf.numel()

        obs, _, done, info = env.step(policy.act(obs, env, n_agents))
        steps += 1

        pos = env.adapter.get_positions()
        path_len += np.linalg.norm(pos - pos_prev, axis=1)
        pos_prev = pos.copy()

        reached = info['drone_reached_goal']
        dist_now = np.linalg.norm(pos - env.goals, axis=1)
        for i in range(n_agents):
            if settle_step[i] is None and reached[i]:
                settle_step[i] = steps
            if entry_step[i] is None and dist_now[i] < env.goal_threshold:
                entry_step[i] = steps

        sep = info.get('min_inter_drone_distance')
        if sep is not None:
            min_sep = sep if min_sep is None else min(min_sep, sep)
        if any(info.get('drone_collision', [])):
            any_drone_collision = True
        if physics_on and n_agents > 1 and not contact:
            contact = drone_contact(env, n_agents)
        if done:
            break

    pos = env.adapter.get_positions()
    final_dist = np.linalg.norm(pos - env.goals, axis=1)
    reason = info.get('reason')
    outcome = 'success' if info.get('success') else (reason if reason in ('collision',) else 'timeout')

    agents = []
    for i in range(n_agents):
        init_d = float(env.initial_dist[i])
        settled = bool(info['drone_reached_goal'][i])
        # Path efficiency only means something for a drone that got there;
        # truncated or stuck episodes would give ratios above 1.
        eff = round(init_d / path_len[i], 4) if settled and path_len[i] > 1e-6 else ''
        agents.append({
            'agent': i,
            'settled': int(settled),
            'settle_time_s': settle_step[i] * dt if settle_step[i] else '',
            'first_entry_time_s': entry_step[i] * dt if entry_step[i] else '',
            'initial_goal_dist': round(init_d, 4),
            'final_goal_dist': round(float(final_dist[i]), 4),
            'min_goal_dist': round(float(env.min_dist_ever[i]), 4),
            'final_speed': round(float(info['per_agent_speed'][i]), 4),
            'path_length': round(float(path_len[i]), 4),
            'path_efficiency': eff,
        })

    episode = {
        'success': int(bool(info.get('success'))),
        'outcome': outcome,
        'length_steps': steps,
        'agents_settled': int(sum(info['drone_reached_goal'])),
        'any_drone_collision': int(any_drone_collision),
        'min_sep': round(min_sep, 4) if min_sep is not None else '',
        'physical_contact': int(contact) if physics_on and n_agents > 1 else '',
        'mean_initial_goal_dist': round(float(np.mean(env.initial_dist)), 4),
        'obs_clip_frac': round(clip_hits / clip_total, 5) if clip_total else 0.0,
    }
    for t in SEP_THRESHOLDS:
        episode[f'sep_below_{t}'] = int(min_sep is not None and min_sep < t) if n_agents > 1 else ''
    return episode, agents


def summarize(episodes, agents, n_agents, physics_on):
    n = len(episodes)
    k_succ = sum(e['success'] for e in episodes)
    k_coll = sum(e['any_drone_collision'] for e in episodes)
    settle_times = [a['settle_time_s'] for a in agents if a['settle_time_s'] != '']
    s = {
        'episodes': n,
        'success_rate': k_succ / n,
        'success_ci95': wilson(k_succ, n),
        'collision_rate': k_coll / n,
        'collision_ci95': wilson(k_coll, n),
        'timeout_rate': sum(e['outcome'] == 'timeout' for e in episodes) / n,
        'per_agent_success_pooled': mean([a['settled'] for a in agents]),
        'per_agent_success_by_index': [mean([a['settled'] for a in agents if a['agent'] == i])
                                       for i in range(n_agents)],
        'mean_episode_length_steps': mean([e['length_steps'] for e in episodes]),
        'mean_initial_goal_dist': mean([a['initial_goal_dist'] for a in agents]),
        'final_goal_err': {q: pct([a['final_goal_dist'] for a in agents], v)
                           for q, v in (('median', 50), ('p95', 95), ('max', 100))},
        'final_goal_err_mean': mean([a['final_goal_dist'] for a in agents]),
        'final_speed': {q: pct([a['final_speed'] for a in agents], v)
                        for q, v in (('median', 50), ('p95', 95), ('max', 100))},
        'final_speed_mean': mean([a['final_speed'] for a in agents]),
        'settle_time_s': {'mean': mean(settle_times), 'median': pct(settle_times, 50),
                          'p95': pct(settle_times, 95)},
        'path_efficiency_mean': mean([a['path_efficiency'] or None for a in agents]),
        'path_efficiency_p5': pct([a['path_efficiency'] or None for a in agents], 5),
        'obs_clip_frac_mean': mean([e['obs_clip_frac'] for e in episodes]),
    }
    if n_agents > 1:
        seps = [e['min_sep'] for e in episodes if e['min_sep'] != '']
        s['min_sep'] = {'min': pct(seps, 0), 'p1': pct(seps, 1), 'p5': pct(seps, 5),
                        'median': pct(seps, 50)}
        for t in SEP_THRESHOLDS:
            k = sum(e[f'sep_below_{t}'] for e in episodes)
            s[f'sep_below_{t}_rate'] = k / n
            s[f'sep_below_{t}_ci95'] = wilson(k, n)
        if physics_on:
            k = sum(e['physical_contact'] for e in episodes)
            s['physical_contact_rate'] = k / n
            s['physical_contact_ci95'] = wilson(k, n)
    return s


def print_summary(s, tag):
    lo, hi = s['success_ci95']
    print(f"\n{'=' * 64}\n{tag}\n{'=' * 64}")
    print(f"episodes                 {s['episodes']}")
    print(f"mean start-goal dist     {s['mean_initial_goal_dist']:.2f} m")
    print(f"all-agent success        {s['success_rate']:.3f}  [95% CI {lo:.3f}-{hi:.3f}]")
    print(f"per-agent success        {s['per_agent_success_pooled']:.3f}  "
          f"(by index: {', '.join(f'{x:.2f}' for x in s['per_agent_success_by_index'])})")
    lo, hi = s['collision_ci95']
    print(f"drone collision (<r)     {s['collision_rate']:.3f}  [95% CI {lo:.3f}-{hi:.3f}]")
    print(f"timeout rate             {s['timeout_rate']:.3f}")
    print(f"final goal err  mean/med/p95/max  {s['final_goal_err_mean']:.3f} / "
          f"{s['final_goal_err']['median']:.3f} / {s['final_goal_err']['p95']:.3f} / "
          f"{s['final_goal_err']['max']:.3f} m")
    print(f"final speed     mean/med/p95/max  {s['final_speed_mean']:.3f} / "
          f"{s['final_speed']['median']:.3f} / {s['final_speed']['p95']:.3f} / "
          f"{s['final_speed']['max']:.3f} m/s")
    st = s['settle_time_s']
    print(f"settle time     mean/med/p95      {st['mean']:.2f} / {st['median']:.2f} / {st['p95']:.2f} s")
    print(f"path efficiency mean / p5         {s['path_efficiency_mean']:.3f} / {s['path_efficiency_p5']:.3f}")
    print(f"obs clip fraction        {s['obs_clip_frac_mean']:.4f}")
    if 'min_sep' in s:
        ms = s['min_sep']
        print(f"min separation  min/p1/p5/median  {ms['min']:.3f} / {ms['p1']:.3f} / "
              f"{ms['p5']:.3f} / {ms['median']:.3f} m")
        for t in SEP_THRESHOLDS:
            lo, hi = s[f'sep_below_{t}_ci95']
            print(f"  episodes with sep < {t:.1f} m   {s[f'sep_below_{t}_rate']:.3f}  "
                  f"[95% CI {lo:.3f}-{hi:.3f}]")
    if 'physical_contact_rate' in s:
        lo, hi = s['physical_contact_ci95']
        print(f"PyBullet contact rate    {s['physical_contact_rate']:.3f}  [95% CI {lo:.3f}-{hi:.3f}]")


def main():
    ap = argparse.ArgumentParser(description="Stage-1 evaluation suite v1 (evaluation only)")
    ap.add_argument("--config", required=True)
    ap.add_argument("--policy", choices=["mappo", "pcontroller", "repulsion"], default="mappo",
                    help="repulsion = rule-based goal-seeking + push away from nearby drones "
                         "(baselines/repulsion_controller.py)")
    ap.add_argument("--checkpoint", help="required for --policy mappo")
    ap.add_argument("--mode", choices=["deterministic", "stochastic"], default="deterministic")
    ap.add_argument("--kp", type=float, default=3.0, help="P-controller gain (baseline default)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--episodes", type=int, default=100, help="episodes PER seed")
    ap.add_argument("--scenario", default="random",
                    choices=["random", "crossing", "swap", "headon", "flythrough", *INTERACTION_SCENARIOS],
                    help="intruder / blocker / neighbour are 2-drone tests where drone_1 is "
                         "scripted and only drone_0 (the policy under test) can avoid a collision")
    ap.add_argument("--goal-dist", type=float, default=None,
                    help="random scenario only: max start-goal distance in metres. "
                         "Omit for the training distribution (env.reset, 1-2 m).")
    ap.add_argument("--goal-dist-min", type=float, default=None,
                    help="min start-goal distance for --goal-dist (default: half of it)")
    ap.add_argument("--n-agents", type=int, default=None, help="zero-shot agent-count override")
    ap.add_argument("--max-steps", type=int, default=None,
                    help="episode time limit override (e.g. 1000 to tell slow traffic from gridlock)")
    ap.add_argument("--blind-neighbors", action="store_true",
                    help="ablation: MAPPO actor sees no other drones (all neighbour slots masked)")
    ap.add_argument("--collision-physics", choices=["on", "off"], required=True,
                    help="off = policy-only safety (PyBullet contact response disabled); "
                         "on = system-level, also records PyBullet contact events")
    ap.add_argument("--out-dir", default=None, help="default: <results-root>/<run tag>")
    ap.add_argument("--results-root", default=os.path.join("eval_results", SUITE_VERSION),
                    help="folder that holds one sub-folder per run (keeps different models apart)")
    ap.add_argument("--overwrite", action="store_true",
                    help="allow replacing an existing results folder")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb-project", default="utm-swarm-marl")
    ap.add_argument("--wandb-entity", default=None)
    ap.add_argument("--source-run", default="cje4yxrd", help="W&B run id the checkpoint came from")
    args = ap.parse_args()

    if args.policy == "mappo" and not args.checkpoint:
        ap.error("--checkpoint is required for --policy mappo")
    if args.goal_dist is not None and args.scenario != "random":
        ap.error("--goal-dist only applies to --scenario random")
    if args.blind_neighbors and args.policy != "mappo":
        ap.error("--blind-neighbors only applies to --policy mappo")
    policy_label = "mappo-blind" if args.blind_neighbors else args.policy
    interaction = args.scenario in INTERACTION_SCENARIOS
    if interaction and args.n_agents not in (None, 2):
        ap.error(f"--scenario {args.scenario} always uses 2 drones")

    with open(args.config) as f:
        config = yaml.safe_load(f)
    # The suite picks the scenario and the collision physics itself, so
    # training-only settings in the config must not leak in: a scenario_mix
    # would turn the "random" test into a mix, and collision_physics: false
    # would make --collision-physics on silently stay off.
    config.pop('scenario_mix', None)
    config['environment']['collision_physics'] = True     # disabled below if "off"
    if interaction:
        config['environment']['n_agents'] = 2
    elif args.n_agents is not None:
        config['environment']['n_agents'] = args.n_agents
    if args.max_steps is not None:
        config['environment']['max_steps'] = args.max_steps
    n_agents = config['environment']['n_agents']
    max_steps = config['environment']['max_steps']
    dt = config['environment'].get('control_timestep', 0.032)
    physics_on = args.collision_physics == "on"
    device = torch.device(args.device)

    d_max = args.goal_dist
    d_min = None
    if d_max is not None:
        d_min = args.goal_dist_min if args.goal_dist_min is not None else max(1.0, d_max / 2)

    dist_tag = f"d{d_max:g}" if d_max is not None else "dtrain"
    tag = (f"{policy_label}_{args.scenario}_{dist_tag}_n{n_agents}_"
           f"phys{args.collision_physics}_s{'-'.join(map(str, args.seeds))}_e{args.episodes}"
           + (f"_T{max_steps}" if args.max_steps is not None else ""))
    out_dir = args.out_dir or os.path.join(args.results_root, tag)
    if os.path.exists(os.path.join(out_dir, "summary.json")) and not args.overwrite:
        ap.error(f"{out_dir} already has results. Pass --overwrite or choose --out-dir.")
    os.makedirs(out_dir, exist_ok=True)

    env = SwarmEnv(config, device=device)
    if not physics_on:
        disable_all_collision_physics(env, n_agents)

    if args.policy == "mappo":
        policy = MappoPolicy(config, args.checkpoint, device, args.mode == "deterministic",
                             blind_neighbors=args.blind_neighbors)
        policy.restore_normalizer(env)
        ckpt_episode = policy.ckpt.get('episode', policy.ckpt.get('episode_count'))
    else:
        policy = PController(config, args.kp) if args.policy == 'pcontroller' else RepulsionController(config)
        ckpt_episode = None
    clip_val = env.obs_processor.normalizer.clip

    run_cfg = {
        'evaluation_suite_version': SUITE_VERSION,
        'observation_schema_version': 'v1',
        'policy': policy_label,
        'neighbor_view': 'blind' if args.blind_neighbors else 'normal',
        'policy_mode': args.mode if args.policy == 'mappo' else 'n/a',
        'checkpoint': args.checkpoint,
        'checkpoint_episode': ckpt_episode,
        'source_run': args.source_run if args.policy == 'mappo' else None,
        'normalizer_frozen': args.policy == 'mappo',
        'config_file': args.config,
        'scenario': args.scenario,
        'goal_dist_min': d_min, 'goal_dist_max': d_max,
        'n_agents': n_agents,
        'k_neighbors_max': config['observation']['k_neighbors_max'],
        'collision_physics': args.collision_physics,
        'collision_radius': env.collision_radius,
        'seeds': args.seeds, 'episodes_per_seed': args.episodes,
        'max_steps': max_steps, 'control_timestep': dt,
        'kp': args.kp if args.policy == 'pcontroller' else None,
    }

    wb = None
    if args.wandb:
        import wandb
        wb = wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                        job_type="evaluation", name=f"eval-{tag}", config=run_cfg,
                        tags=[f"suite-{SUITE_VERSION}", policy_label, args.scenario])

    print(f"[eval_suite {SUITE_VERSION}] {tag}")
    if n_agents - 1 > config['observation']['k_neighbors_max']:
        print(f"  note: {n_agents} agents, actor observes only the "
              f"{config['observation']['k_neighbors_max']} nearest neighbours")

    episodes, agents = [], []
    t0 = time.time()
    for seed in args.seeds:
        # Global seed drives env.reset(); a separate RNG drives the
        # hand-built scenarios. Both are identical for MAPPO and P-controller.
        np.random.seed(seed)
        torch.manual_seed(seed)
        rng = np.random.RandomState(seed)
        for ep in range(args.episodes):
            if interaction:
                obs, script, horizon, param = setup_interaction(env, rng, args.scenario, dt,
                                                                max_steps, policy)
                sid = scenario_hash(env)
                row = run_interaction_episode(env, policy, obs, args.scenario, script, horizon)
                episodes.append({'seed': seed, 'episode': ep, 'scenario_id': sid,
                                 'param': param, **row})
                continue
            if args.scenario == "crossing":
                obs = setup_crossing_scenario(env, rng, n_agents)
            elif args.scenario == "swap":
                obs = setup_swap_scenario(env, rng, n_agents)
            elif args.scenario == "headon":
                obs = setup_headon_scenario(env, rng, n_agents)
            elif args.scenario == "flythrough":
                # every drone runs the policy: drone_0 hovers on its goal while
                # the others fly straight through it (envs/scenarios.py)
                spawns, goals = scenarios.flythrough(rng, n_agents)
                obs = _apply_scenario(env, spawns, goals, n_agents)
            elif d_max is not None:
                obs = sample_random_band(env, rng, n_agents, d_min, d_max)
            else:
                obs = env.reset()
            sid = scenario_hash(env)

            e, ags = run_episode(env, policy, obs, n_agents, max_steps, dt, physics_on, clip_val)
            base = {'seed': seed, 'episode': ep, 'scenario_id': sid}
            episodes.append({**base, **e})
            agents.extend({**base, **a} for a in ags)
        done_eps = [x for x in episodes if x['seed'] == seed]
        if interaction:
            print(f"  seed {seed}: ego collided {np.mean([x['collided'] for x in done_eps]):.3f} "
                  f"over {len(done_eps)} episodes  ({time.time() - t0:.0f}s elapsed)")
        else:
            print(f"  seed {seed}: success {np.mean([x['success'] for x in done_eps]):.3f} "
                  f"over {len(done_eps)} episodes  ({time.time() - t0:.0f}s elapsed)")
    env.close()

    if interaction:
        summary = summarize_interaction(episodes, args.scenario)
    else:
        summary = summarize(episodes, agents, n_agents, physics_on)
    summary['config'] = run_cfg
    summary['wall_time_s'] = round(time.time() - t0, 1)

    for name, rows in (("episodes.csv", episodes), ("agents.csv", agents)):
        if not rows:
            continue
        with open(os.path.join(out_dir, name), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    if interaction:
        print_interaction_summary(summary, tag, args.scenario)
    else:
        print_summary(summary, tag)
    print(f"\nwrote results to {out_dir}")

    if wb is not None:
        import wandb
        flat = {}
        for k, v in summary.items():
            if k == 'config':
                continue
            if isinstance(v, dict):
                flat.update({f"{k}/{kk}": vv for kk, vv in v.items()})
            elif isinstance(v, (list, tuple)):
                flat.update({f"{k}/{i}": vv for i, vv in enumerate(v)})
            else:
                flat[k] = v
        wb.summary.update(flat)
        # CSV uses '' for "not applicable"; W&B tables need None there
        def rows(rs):
            return [[None if v == '' else v for v in r.values()] for r in rs]
        tables = {"episodes": wandb.Table(columns=list(episodes[0].keys()), data=rows(episodes))}
        if agents:
            tables["agents"] = wandb.Table(columns=list(agents[0].keys()), data=rows(agents))
        wb.log(tables)
        wb.finish()


if __name__ == "__main__":
    main()
