"""
imitation_warmstart.py — teach the actor to avoid other drones by copying a
rule-based teacher, then hand it back to RL.

Why: RL alone never discovered avoidance (the actor ignored its neighbour
input). Copying a controller that already avoids gives the neighbour pathway
a direct, dense training signal ("in this situation, fly this velocity").
RL fine-tuning afterwards can then improve on it.

Steps (all evaluation-free; the result is checked separately with eval_suite):
  1. Start from the baseline actor (keeps its navigation and settling) and its
     frozen observation normalizer, with the given config's neighbour inputs.
  2. Collect: the teacher (baselines/repulsion_controller.py) flies the
     config's scenario mix. For every drone and step we store what the actor
     would observe and the velocity the teacher chose.
  3. Clone: train the actor's mean action to match the teacher (MSE).
  4. DAgger rounds: let the CLONED actor fly, label the states it reaches with
     the teacher's action, add them to the dataset and train again. This fixes
     the usual imitation problem where small mistakes lead into situations the
     teacher never showed it.
  5. Save two files in --out-dir:
       bc_policy.pth        actor + normalizer, for eval_suite / sensitivity
       bc_resume.pth        the baseline training checkpoint with the cloned
                            actor swapped in, for `train.py --resume`

    python scripts/imitation_warmstart.py --config configs/config_3agent_deconfliction_v3.yaml \
        --out-dir checkpoints/imitation

Refining an existing imitation actor on harder situations (continues from its
weights, with its own scenario mix and DAgger schedule):

    python scripts/imitation_warmstart.py --config configs/config_3agent_deconfliction_v3.yaml \
        --init-actor checkpoints/imitation/bc_policy.pth \
        --mix headon=0.35,crossing=0.35,flythrough=0.15,swap=0.05,random=0.1 \
        --betas 0.3,0.1,0.0 --lr 1e-4 --out-dir checkpoints/imitation_v2

Stage 2 (obstacles): the teacher gets each episode's obstacle layout, the
actor only its LiDAR. --n-agents 3-8 picks the drone count per episode, so
the actor sees crowded and sparse traffic. With observation schema v2 the
base and init checkpoints must be converted first
(scripts/convert_checkpoint_inputs.py); there is no normalizer to restore.

Balanced loss (--loss-parts G,P,A --arrival-dist D): instead of per-sample
weights, the error is averaged separately over general flight, passing an
obstacle, and arrival (within D metres of the goal, 3D) and the three means
are combined with weights G,P,A. With 4x/8x obstacle weights the student got
better at passing but stopped short of the goal after detours (outputs 0 at
0.6-0.7 m while the teacher commands ~1.2 m/s) or kept creeping inside it;
the balanced loss keeps every part's share fixed regardless of sample counts.

Near-obstacle weighting (--near-weight W --near-dist D): the moments close to
an obstacle are a small share of the data (~8% within 0.7 m) but where every
obstacle crash happens, and an unweighted average lets them be copied ~10x
worse than open-air flight. Samples where the drone is within D metres of an
obstacle surface count W times in the cloning loss.

Passing-phase weighting (--along-weight A): on its own states the student
stopped pushing outward once level with an obstacle (sideways command 0.01 vs
the teacher's 0.75 m/s) and cut back early. Samples where the closest point
of an obstacle surface is within 1 m and mostly to the side (same definition
as eval_suite's passing measures) count A times instead.
"""

import argparse
import copy
import os
import sys
import time

import numpy as np
import torch
import yaml

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)

from baselines.repulsion_controller import RepulsionController
from envs.swarm_env import SwarmEnv
from envs.scenario_manager import distance_to_obstacles, DRONE_RADIUS
from baselines.repulsion_controller import closest_point
from training.networks.actor_network import ActorNetwork


def rollout(pick_env, teacher, actor, episodes, beta, device, data):
    """Fly `episodes` episodes. Each step the executed action comes from the
    teacher with probability beta, else from the actor; the TEACHER's action is
    always stored as the label. pick_env() gives the environment for each
    episode (one per drone count)."""
    for _ in range(episodes):
        env = pick_env()
        n = env.n_agents
        obs = env.reset()
        for _ in range(env.config['environment']['max_steps']):
            raw = env.adapter.get_raw_state()
            pos = np.array([raw[f'drone_{i}']['position'] for i in range(n)])
            vel = np.array([raw[f'drone_{i}']['velocity'] for i in range(n)])
            lidar = np.array([raw[f'drone_{i}']['lidar'] for i in range(n)])
            target = teacher.velocities(pos, vel, env.goals, env.layout, lidar)
            data['gap'].append(np.array([distance_to_obstacles(env.layout, pos[i]) if env.layout
                                         else np.inf for i in range(n)], dtype=np.float32))
            data['along'].append(np.array([alongside(env, pos[i], env.goals[i]) for i in range(n)]))
            data['goal_d'].append(np.linalg.norm(env.goals[:n] - pos, axis=1).astype(np.float32))
            data['settled'].append(np.array(env.drone_reached_goal[:n], dtype=bool))
            d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=2) + np.eye(n) * 1e9
            data['near'].append(d.min(axis=1).astype(np.float32) if n > 1 else np.full(n, np.inf, np.float32))
            data['lf'].append(obs['local_fixed'].numpy().copy())
            data['ns'].append(obs['neighbor_states'].numpy().copy())
            data['nm'].append(obs['neighbor_mask'].numpy().copy())
            data['y'].append(target.astype(np.float32))
            if actor is None or np.random.rand() < beta:
                act = target
            else:
                with torch.no_grad():
                    mean, _ = actor.forward(obs['local_fixed'].to(device), obs['neighbor_states'].to(device),
                                            obs['neighbor_mask'].to(device))
                act = mean[:, :3].cpu().numpy()
            obs, _, done, _ = env.step({f'drone_{i}': list(act[i]) + [0.0] for i in range(n)})
            if done:
                break


def alongside(env, pos, goal):
    """Passing phase: the closest obstacle surface point is within 1 m and
    less than 60 deg from perpendicular to the goal direction."""
    g = np.asarray(goal[:2]) - pos[:2]
    obs_below = [ob for ob in (env.layout or []) if pos[2] <= ob['height']]
    if np.linalg.norm(g) < 0.3 or not obs_below:
        return False
    u = g / np.linalg.norm(g)
    rel = min((closest_point(ob, pos[:2]) - pos[:2] for ob in obs_below), key=np.linalg.norm)
    dist = float(np.linalg.norm(rel))
    return bool(dist > 1e-6 and dist - DRONE_RADIUS < 1.0 and abs(rel @ u) / dist < 0.5)


def train(actor, data, epochs, device, lr, batch=1024, near_weight=1.0, near_dist=0.0,
          along_weight=1.0, loss_parts=None, arrival_dist=1.0, crowd_dist=1.5):
    """Weighted MSE on the action mean. Two modes:
    per-sample weights (near_weight / along_weight), or, with loss_parts =
    (general, passing, arrival), a balanced loss: every batch's error is
    averaged within each part separately and the part means are combined
    with these weights, so a part's influence does not depend on how many
    of its samples the data happens to contain.
        arrival : within arrival_dist (3D) of the goal and not yet settled
                  (the final approach and calming down; drones that have
                  settled hover at the goal for the rest of the episode,
                  ~40% of all samples, and those easy steps go to general)
        passing : obstacle beside the drone (see alongside())
        crowd   : (only with a 4th weight) another drone within crowd_dist m; in 8-drone
                  crowds students dodged at ~30% of the teacher's strength and passed
                  each other 21-25 cm apart
        general : everything else"""
    lf = torch.tensor(np.concatenate(data['lf']), device=device)
    ns = torch.tensor(np.concatenate(data['ns']), device=device)
    nm = torch.tensor(np.concatenate(data['nm']), device=device)
    y = torch.tensor(np.concatenate(data['y']), device=device)
    gap = np.concatenate(data['gap'])
    along = np.concatenate(data['along'])
    if loss_parts:
        arriving = (np.concatenate(data['goal_d']) < arrival_dist) & ~np.concatenate(data['settled'])
        part = np.where(arriving, 2, np.where(along, 1, 0))
        if len(loss_parts) > 3:
            crowd = np.concatenate(data['near']) < crowd_dist
            part = np.where((part == 0) & crowd, 3, part)
        part = torch.tensor(part, device=device)
        part_w = torch.tensor(loss_parts, dtype=torch.float32, device=device)
        shares = [f"{name} {np.mean(part.cpu().numpy() == k) * 100:.1f}%"
                  for k, name in enumerate(('general', 'passing', 'arrival', 'crowd')[:len(loss_parts)])]
        print(f"      balanced loss {tuple(loss_parts)}; samples: {', '.join(shares)}")
    else:
        w = np.where(gap < near_dist, near_weight, 1.0)
        w = np.where(along, along_weight, w)
        w = torch.tensor(w.astype(np.float32), device=device)
    opt = torch.optim.Adam([p for name, p in actor.named_parameters() if 'log_std' not in name], lr=lr)
    n = len(y)
    for ep in range(epochs):
        perm = torch.randperm(n, device=device)
        tot = 0.0
        for k in range(0, n, batch):
            idx = perm[k:k + batch]
            mean, _ = actor.forward(lf[idx], ns[idx], nm[idx])
            err = ((mean[:, :3] - y[idx]) ** 2).mean(dim=1)
            if loss_parts:
                pb = part[idx]
                K = len(loss_parts)
                means = [err[pb == k].mean() for k in range(K) if (pb == k).any()]
                ws = [part_w[k] for k in range(K) if (pb == k).any()]
                loss = sum(wk * m for wk, m in zip(ws, means)) / sum(ws)
            else:
                loss = (w[idx] * err).sum() / w[idx].sum()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(idx)
        print(f"      epoch {ep + 1:2d}/{epochs}  mse {tot / n:.4f}  ({n} samples)")
    return actor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--base", default="checkpoints/stage1_v1/checkpoint_final.pth",
                    help="training checkpoint to start from (actor, critic, optimizers, normalizer)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--episodes", type=int, default=400, help="teacher episodes for the first dataset")
    ap.add_argument("--dagger-rounds", type=int, default=2)
    ap.add_argument("--dagger-episodes", type=int, default=200)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init-actor", default=None,
                    help="start from this actor's weights (e.g. an earlier bc_policy.pth) instead of the base actor")
    ap.add_argument("--mix", default=None,
                    help="override the config's scenario mix, e.g. headon=0.35,crossing=0.35,flythrough=0.15,swap=0.05,random=0.1")
    ap.add_argument("--betas", default=None,
                    help="teacher share per DAgger round, e.g. 0.3,0.1,0.0 (overrides --dagger-rounds)")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--near-weight", type=float, default=1.0,
                    help="loss weight of samples near an obstacle (1 = unweighted)")
    ap.add_argument("--loss-parts", default=None,
                    help="balanced loss weights general,passing,arrival[,crowd] (e.g. 1,1,1 or "
                         "1,1,1,1); replaces --near-weight/--along-weight")
    ap.add_argument("--crowd-dist", type=float, default=1.5,
                    help="'crowd' = another drone within this many metres (4th loss part)")
    ap.add_argument("--arrival-dist", type=float, default=1.0,
                    help="'arrival' = within this many metres (3D) of the goal")
    ap.add_argument("--along-weight", type=float, default=1.0,
                    help="loss weight of passing-phase samples (obstacle beside the drone)")
    ap.add_argument("--near-dist", type=float, default=1.5,
                    help="'near' = within this many metres of an obstacle surface")
    ap.add_argument("--n-agents", default=None,
                    help="drone count per episode, e.g. 3-8 (uniform); default: the config's n_agents")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(args.device)
    cfg = yaml.safe_load(open(os.path.join(ROOT, args.config)))
    cfg['environment']['collision_physics'] = False
    if args.mix:
        cfg['scenario_mix'] = {k: float(v) for k, v in (x.split("=") for x in args.mix.split(","))}
    parts = [float(x) for x in args.loss_parts.split(",")] if args.loss_parts else None
    betas = ([float(b) for b in args.betas.split(",")] if args.betas
             else [0.5 / (r + 1) for r in range(args.dagger_rounds)])
    base = torch.load(os.path.join(ROOT, args.base), map_location="cpu", weights_only=False)

    schema = cfg['observation'].get('schema', 'v1')
    if base.get('observation_schema', 'v1') != schema:
        sys.exit(f"--base uses observation schema {base.get('observation_schema', 'v1')}, config uses "
                 f"{schema}; convert it with scripts/convert_checkpoint_inputs.py")

    if args.n_agents:
        lo, hi = (int(x) for x in args.n_agents.split("-"))
        counts = list(range(lo, hi + 1))
    else:
        counts = [cfg['environment']['n_agents']]
    envs = {}
    for n in counts:
        c = copy.deepcopy(cfg)
        c['environment']['n_agents'] = n
        envs[n] = SwarmEnv(c)
        if schema == 'v1':
            norm = envs[n].obs_processor.normalizer
            norm.mean, norm.var, norm.count = (base['normalizer_mean'], base['normalizer_var'],
                                               base['normalizer_count'])
            norm.freeze()
    pick_env = lambda: envs[counts[np.random.randint(len(counts))]]

    actor = ActorNetwork(cfg).to(device)
    if args.init_actor:
        init = torch.load(os.path.join(ROOT, args.init_actor), map_location="cpu", weights_only=False)
        if init.get('observation_schema', 'v1') != schema:
            sys.exit(f"--init-actor uses observation schema {init.get('observation_schema', 'v1')}, "
                     f"config uses {schema}; convert it with scripts/convert_checkpoint_inputs.py")
        actor.load_state_dict(init['actor_state_dict'])
        print(f"starting from actor in {args.init_actor}")
    else:
        actor.load_state_dict(base['actor_state_dict'])
    teacher = RepulsionController(cfg)
    data = {'lf': [], 'ns': [], 'nm': [], 'y': [], 'gap': [], 'along': [], 'goal_d': [], 'settled': [],
            'near': []}

    t0 = time.time()
    print(f"[1] teacher flies {args.episodes} episodes of the scenario mix {cfg.get('scenario_mix')}, "
          f"{counts[0]}-{counts[-1]} drones")
    rollout(pick_env, teacher, None, args.episodes, 1.0, device, data)
    print(f"    collected {sum(len(x) for x in data['y'])} drone-steps ({time.time() - t0:.0f}s)")
    print("[2] cloning")
    actor.train(); train(actor, data, args.epochs, device, args.lr,
                         near_weight=args.near_weight, near_dist=args.near_dist,
                         along_weight=args.along_weight,
                         loss_parts=parts, arrival_dist=args.arrival_dist,
                             crowd_dist=args.crowd_dist)

    for r, beta in enumerate(betas):
        print(f"[3.{r + 1}] DAgger: cloned actor flies {args.dagger_episodes} episodes (teacher steps in {beta:.0%} of the time)")
        actor.eval()
        rollout(pick_env, teacher, actor, args.dagger_episodes, beta, device, data)
        print(f"    dataset now {sum(len(x) for x in data['y'])} drone-steps ({time.time() - t0:.0f}s)")
        actor.train(); train(actor, data, args.epochs, device, args.lr,
                             near_weight=args.near_weight, near_dist=args.near_dist,
                             along_weight=args.along_weight,
                             loss_parts=parts, arrival_dist=args.arrival_dist,
                             crowd_dist=args.crowd_dist)
    for env in envs.values():
        env.close()
    actor.eval()

    os.makedirs(os.path.join(ROOT, args.out_dir), exist_ok=True)
    state = {k: v.cpu() for k, v in actor.state_dict().items()}
    policy = {'actor_state_dict': state, 'curriculum_stage': base.get('curriculum_stage', 1),
              'log_std_ceiling': base.get('log_std_ceiling'), 'log_std_floor': base.get('log_std_floor'),
              'episode': base.get('episode_count'),
              'source': f"imitation of RepulsionController, from {args.init_actor or args.base}, "
                        f"mix {cfg.get('scenario_mix')}, {counts[0]}-{counts[-1]} drones"}
    if schema == 'v1':
        policy.update(normalizer_mean=base['normalizer_mean'], normalizer_var=base['normalizer_var'],
                      normalizer_count=base['normalizer_count'])
    else:
        policy['observation_schema'] = schema
    torch.save(policy, os.path.join(ROOT, args.out_dir, "bc_policy.pth"))
    resume = copy.copy(base)
    resume['actor_state_dict'] = state
    if 'best_success_rate' in resume:                  # plain float: loadable with weights_only=True
        resume['best_success_rate'] = float(resume['best_success_rate'])
    torch.save(resume, os.path.join(ROOT, args.out_dir, "bc_resume.pth"))
    print(f"saved {args.out_dir}/bc_policy.pth and bc_resume.pth  (total {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
