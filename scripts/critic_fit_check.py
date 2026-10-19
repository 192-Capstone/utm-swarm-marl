"""
critic_fit_check.py
-------------------
Can a critic predict what actually happens? Frozen-actor test of the critic
design, with no RL: the actor never changes, so every critic is judged on the
exact same flights.

1. collect: fly a fixed policy (default BCA) with the training exploration noise
   on the training scenario mix, store each step's 32-value-per-drone global
   state (critic.version 2 layout; the old critic uses the first 17) and each
   drone's discounted return from that step (gamma from the config).
2. fit: train four critics on the first episodes and measure explained variance
   on held-out episodes (split by whole episode):
       team value,      17 inputs   (the critic used so far)
       team value,      32 inputs   (new inputs only)
       per-drone value, 17 inputs   (per-drone output only)
       per-drone value, 32 inputs   (critic.version 2)
   All use the CriticNetwork architecture (2-layer 128 encoder, mean pool,
   2 x 256 head) and the training critic learning rate.

Returns here are Monte-Carlo (no bootstrapping); a timeout counts as the end of
the episode for every critic alike, so the comparison stays fair.

Usage (from the repo root):
    python scripts/critic_fit_check.py collect --episodes 130 --out <file.npz>
    python scripts/critic_fit_check.py fit --data <file.npz> --train-episodes 100
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CONFIG = 'configs/config_stage2_rlfix_AB.yaml'
BCA = 'checkpoints/stage2_imitation_BCA/bc_policy.pth'


def collect(args):
    import eval_suite as es
    from envs.swarm_env import SwarmEnv
    cfg = yaml.safe_load(open(args.config))
    cfg['critic'] = {'version': 2}
    cfg['environment']['collision_physics'] = False
    gamma = cfg['training']['gamma']
    n = cfg['environment']['n_agents']
    env = SwarmEnv(cfg, device=torch.device('cpu'))
    pol = es.MappoPolicy(cfg, args.checkpoint, torch.device('cpu'), True)
    rng = np.random.RandomState(args.seed)
    np.random.seed(args.seed)
    states, returns, episode, outcomes = [], [], [], {}
    for ep in range(args.episodes):
        obs = env.reset()
        gs, rews = [], []
        for _ in range(cfg['environment']['max_steps']):
            gs.append(obs['global_state'].numpy())
            a = pol.act(obs, env, n)
            a = {k: [float(x) + (rng.normal(0, args.noise) if j < 3 else 0.0)
                     for j, x in enumerate(v)] for k, v in a.items()}
            obs, r, done, info = env.step(a)
            rews.append([r[f'drone_{i}'] for i in range(n)])
            if done:
                break
        rews = np.array(rews, dtype=np.float32)
        ret = np.zeros_like(rews)
        g = np.zeros(n, dtype=np.float32)
        for t in range(len(rews) - 1, -1, -1):
            g = rews[t] + gamma * g
            ret[t] = g
        states.append(np.array(gs, dtype=np.float32))
        returns.append(ret)
        episode.append(np.full(len(rews), ep))
        k = info.get('reason') or 'timeout'
        outcomes[k] = outcomes.get(k, 0) + 1
        if (ep + 1) % 10 == 0:
            print(f"  {ep + 1}/{args.episodes} episodes  {outcomes}", flush=True)
    np.savez_compressed(args.out, states=np.concatenate(states), returns=np.concatenate(returns),
                        episode=np.concatenate(episode))
    env.close()
    print(f"saved {args.out}: {sum(len(s) for s in states)} steps, outcomes {outcomes}")


class Critic(nn.Module):
    """Same layout as training/networks/critic_network.py, with the input size
    and team / per-drone output as options."""

    def __init__(self, in_dim, per_agent, embed=128, hidden=256):
        super().__init__()
        self.per_agent = per_agent
        self.enc = nn.Sequential(nn.Linear(in_dim, embed), nn.Tanh(), nn.Linear(embed, embed), nn.Tanh())
        head_in = 2 * embed if per_agent else embed
        self.mlp = nn.Sequential(nn.Linear(head_in, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh())
        self.out = nn.Linear(hidden, 1)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.zeros_(m.bias)

    def forward(self, gs):                                   # (B, N, D) -> (B, N)
        e = self.enc(gs)
        if self.per_agent:
            team = e.mean(1, keepdim=True).expand_as(e)
            return self.out(self.mlp(torch.cat([e, team], -1))).squeeze(-1)
        return self.out(self.mlp(e.mean(1))).expand(-1, gs.shape[1])


def explained_variance(pred, target):
    return float(1 - (target - pred).var() / target.var())


def fit(args):
    d = np.load(args.data)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    S = torch.tensor(d['states'], device=dev)
    R = torch.tensor(d['returns'], device=dev)
    train = torch.tensor(d['episode'] < args.train_episodes, device=dev)
    test = ~train
    near = S[..., 20] < 1.0                                  # drone within 1 m of its goal
    print(f"steps: {int(train.sum())} train / {int(test.sum())} held-out, "
          f"return std {float(R.std()):.1f}, device {dev}")
    print(f"{'critic':28s} {'train EV':>9s} {'held-out EV':>12s} {'held-out, <1 m from goal':>25s}   (final epoch)")
    for name, dim, per_agent in [('team value, 17 inputs', 17, False),
                                 ('team value, 32 inputs', 32, False),
                                 ('per-drone value, 17 inputs', 17, True),
                                 ('per-drone value, 32 inputs', 32, True)]:
        torch.manual_seed(args.seed)
        c = Critic(dim, per_agent).to(dev)
        opt = torch.optim.Adam(c.parameters(), lr=args.lr)
        Xtr, Ytr = S[train][..., :dim], R[train]
        Xte, Yte = S[test][..., :dim], R[test]
        best = (-1e9, 0)                     # best held-out EV and its epoch
        for epoch in range(1, args.epochs + 1):
            perm = torch.randperm(len(Xtr), device=dev)
            for i in range(0, len(Xtr), args.batch):
                idx = perm[i:i + args.batch]
                loss = ((c(Xtr[idx]) - Ytr[idx]) ** 2).mean()
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(c.parameters(), 0.5)
                opt.step()
            with torch.no_grad():
                ev = explained_variance(c(Xte), Yte)
            if ev > best[0]:
                best = (ev, epoch)
        with torch.no_grad():
            ptr = c(Xtr)
            pte = c(S[test][..., :dim])
        nt = near[test]
        print(f"{name:28s} {explained_variance(ptr, Ytr):9.2f} "
              f"{explained_variance(pte, R[test]):12.2f} "
              f"{explained_variance(pte[nt], R[test][nt]):25.2f}"
              f"   best held-out {best[0]:.2f} at epoch {best[1]}", flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='cmd', required=True)
    c = sub.add_parser('collect')
    c.add_argument('--config', default=CONFIG)
    c.add_argument('--checkpoint', default=BCA)
    c.add_argument('--episodes', type=int, default=130)
    c.add_argument('--noise', type=float, default=0.05)
    c.add_argument('--seed', type=int, default=0)
    c.add_argument('--out', required=True)
    f = sub.add_parser('fit')
    f.add_argument('--data', required=True)
    f.add_argument('--train-episodes', type=int, default=100)
    f.add_argument('--epochs', type=int, default=30)
    f.add_argument('--batch', type=int, default=256)
    f.add_argument('--lr', type=float, default=3e-4)
    f.add_argument('--seed', type=int, default=0)
    a = p.parse_args()
    collect(a) if a.cmd == 'collect' else fit(a)
