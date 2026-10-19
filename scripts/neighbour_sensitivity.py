"""
neighbour_sensitivity.py — does the actor actually react to nearby drones?

Evaluation only, no training. For each checkpoint we play head-on and
flythrough episodes with the deterministic policy. Every time a drone has
another drone within 1.5 m (the proximity-penalty zone), we take that exact
observation and ask the actor for its velocity twice:
    normal : with its real neighbour input
    blind  : with every neighbour slot masked (as if flying alone)
and compare the two answers:
    |difference|     how much the neighbour input changes the command (m/s)
    away component   the part of that difference pointing away from the
                     nearest drone (positive = it steers away)
    sensitivity      how much the command changes per metre the nearest
                     neighbour moves (from the gradient), in m/s per m
    input size       the neighbour position value the network actually sees

If |difference| and sensitivity are near zero, the network is ignoring its
neighbours, however much the reward pushes it to avoid them.

    python scripts/neighbour_sensitivity.py --config configs/config_3agent_deconfliction_v2.yaml \
        --ckpt baseline=checkpoints/stage1_v1/best_eval_policy.pth \
        --ckpt seed0=checkpoints/deconfliction_v2/seed0/checkpoint_ep800.pth
"""

import argparse
import os
import sys

import numpy as np
import torch
import yaml

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)

from envs import scenarios
from envs.swarm_env import SwarmEnv
from record_rollout import _apply_scenario
from training.networks.actor_network import ActorNetwork

DANGER = 1.5


def load(cfg, path, env):
    actor = ActorNetwork(cfg)
    ck = torch.load(path, map_location="cpu", weights_only=False)
    actor.load_state_dict(ck.get("actor_state_dict", ck))
    actor.eval()
    if env.obs_processor.schema == "v1":       # v2 has fixed input scaling, no normalizer
        norm = env.obs_processor.normalizer
        norm.mean, norm.var, norm.count = ck["normalizer_mean"], ck["normalizer_var"], ck["normalizer_count"]
        norm.freeze()
    return actor


def collect(cfg, actor, env, episodes, seed):
    """Close-encounter snapshots: (local, neighbours, mask, nearest rel. position in m)."""
    rng = np.random.RandomState(seed)
    n = cfg["environment"]["n_agents"]
    snaps = []
    for sc in ("headon", "flythrough"):
        for _ in range(episodes):
            spawns, goals = scenarios.GENERATORS[sc](rng, n)
            obs = _apply_scenario(env, spawns, goals, n)
            for _ in range(cfg["environment"]["max_steps"]):
                lf, ns, nm = obs["local_fixed"], obs["neighbor_states"], obs["neighbor_mask"]
                pos = env.adapter.get_positions()
                for i in range(n):
                    d = np.linalg.norm(np.delete(pos, i, axis=0) - pos[i], axis=1)
                    if d.min() < DANGER and not nm[i, 0]:
                        j = [k for k in range(n) if k != i][int(np.argmin(d))]
                        snaps.append((lf[i].clone(), ns[i].clone(), nm[i].clone(), pos[j] - pos[i]))
                with torch.no_grad():
                    a, _ = actor.get_action(lf, ns, nm, deterministic=True, active_dims=3)
                a = a.clone(); a[:, 3:] = 0.0
                obs, _, done, _ = env.step({f"drone_{k}": a[k].tolist() for k in range(n)})
                if done:
                    break
    return snaps


def analyse(actor, snaps, pos_scale):
    lf = torch.stack([s[0] for s in snaps])
    ns = torch.stack([s[1] for s in snaps]).requires_grad_(True)
    nm = torch.stack([s[2] for s in snaps])
    rel = np.stack([s[3] for s in snaps])
    dist = np.linalg.norm(rel, axis=1)

    mean_n, _ = actor.forward(lf, ns, nm)
    a_normal = mean_n[:, :3]
    # gradient of the command w.r.t. the nearest neighbour's position input
    grads = []
    for k in range(3):
        g, = torch.autograd.grad(a_normal[:, k].sum(), ns, retain_graph=True)
        grads.append(g[:, 0, :3])
    jac = torch.stack(grads, dim=1).detach().numpy()          # (B, 3 out, 3 in)
    sens = np.linalg.norm(jac, axis=(1, 2)) / pos_scale        # per metre, not per input unit

    with torch.no_grad():
        mean_b, _ = actor.forward(lf, torch.zeros_like(ns), torch.ones_like(nm))
    diff = (a_normal - mean_b[:, :3]).detach().numpy()
    away = -(rel / dist[:, None])
    away_comp = np.sum(diff * away, axis=1)
    inp = ns[:, 0, :3].detach().norm(dim=1).numpy()
    return dist, np.linalg.norm(diff, axis=1), away_comp, sens, inp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", action="append", required=True, help="label=path, repeatable")
    ap.add_argument("--episodes", type=int, default=15, help="per scenario")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(os.path.join(ROOT, args.config)))
    cfg.pop("scenario_mix", None)
    cfg["environment"]["collision_physics"] = False
    pos_scale = float(cfg["observation"].get("neighbor_pos_scale", 1.0))
    print(f"neighbour position scale: {pos_scale} (a drone 1 m away is fed to the network as {1/pos_scale:.3f})\n")
    print(f"{'model':12s}{'moments':>9s}{'  |diff| median':>16s}{'  |diff| p90':>13s}"
          f"{'  steers away':>14s}{'  sensitivity':>14s}{'  input size':>13s}")
    print(f"{'':12s}{'':>9s}{'(m/s)':>16s}{'(m/s)':>13s}{'(% of moments)':>14s}{'(m/s per m)':>14s}{'(median)':>13s}")
    for spec in args.ckpt:
        label, path = spec.split("=", 1)
        env = SwarmEnv(cfg)
        actor = load(cfg, os.path.join(ROOT, path), env)
        snaps = collect(cfg, actor, env, args.episodes, args.seed)
        env.close()
        if not snaps:
            print(f"{label:12s} no close encounters"); continue
        dist, d, away, sens, inp = analyse(actor, snaps, pos_scale)
        print(f"{label:12s}{len(snaps):9d}{np.median(d):16.3f}{np.percentile(d, 90):13.3f}"
              f"{(away > 0.05).mean()*100:13.0f}%{np.median(sens):14.3f}{np.median(inp):13.3f}")
        for lo, hi in ((0.0, 0.5), (0.5, 1.0), (1.0, 1.5)):
            m = (dist >= lo) & (dist < hi)
            if m.any():
                print(f"{'':14s}within {lo:.1f}-{hi:.1f} m: {m.sum():4d} moments, |diff| median {np.median(d[m]):.3f} m/s, "
                      f"steers away in {(away[m] > 0.05).mean()*100:.0f}%")


if __name__ == "__main__":
    main()
