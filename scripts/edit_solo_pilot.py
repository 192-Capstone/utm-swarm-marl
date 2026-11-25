"""
edit_solo_pilot.py
------------------
Edit the learned pilot's weights so that it flies on time when it is alone. No retraining.

The attention block turns the neighbours into 32 numbers (the context). With no neighbour in view the
context is exactly 0, which training never showed the pilot (every training sample had a drone within
14.1 m), and alone it flies 0.8-1.8 m/s depending on its direction instead of the 1.5 m/s E1 plans.
This computes the context c the pilot should have read when alone, and writes it into two biases:

    mlp.0.bias                += W_context c        the first layer reads an empty sky as c
    attention.value_proj.bias -= c                  and any neighbour's context drops by c, which cancels it
                                                    (the attention weights sum to 1)

So the commands with any neighbour do not change (checked below), and 288 of 98,120 weights do.

c is fitted by damped Gauss-Newton least squares to what the rule teacher (the one the pilot learned
from) would command at solo-flight states. Two rounds: the states are collected with the shipped pilot,
then also with the pilot as edited by round 1. Fly the result with sim.cruise_z_bias 0: it holds its height.

    python scripts/edit_solo_pilot.py --out eval_results/solo_fix/edit2.pth
"""

import argparse
import os
import sys

import numpy as np
import torch
import yaml
from torch.func import jacrev, vmap

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import eval_suite as es                                            # noqa: E402
import run_scenario as rs                                          # noqa: E402
from baselines.repulsion_controller import RepulsionController     # noqa: E402
from envs.swarm_env import SwarmEnv                                 # noqa: E402
from modules.goal_hold import GoalHoldPolicy                        # noqa: E402
from record_rollout import disable_all_collision_physics           # noqa: E402
from training.networks.actor_network import ActorNetwork           # noqa: E402
from utm import flight_log, scenario as scn                         # noqa: E402

CKPT = '../utm-swarm-marl/checkpoints/stage2_r1_crowd_rl_r7v3ft_s0_lab/checkpoint_final.pth'
CONFIG = 'configs/config_stage2_round7_eval_goalhold.yaml'
TEACHER = 'configs/config_stage2_round7_teacher.yaml'
FLIGHT_OPTIONS = ('land_latch_m', 'land_vz_max', 'land_wait_clear_m', 'cruise_z_bias', 'land_step_m',
                  'land_vz_high', 'land_slow_m', 'los_below_cruise')
LOCAL, CTX = 74, 32                                # the actor's first layer: 74 own numbers, then the context
ROUTES = (('ecity_rooftops', ('f0', 'f2', 'f7', 'f8', 'f9', 'f11')),      # solo routes flown on the city map
          ('ecity_rush_hour', ('f6', 'f14', 'f16', 'f13')))
KEEP = ('log_std_ceiling', 'log_std_floor', 'curriculum_stage', 'observation_schema')


def load(path):
    with open(os.path.join(ROOT, path), encoding='utf-8') as f:
        return yaml.safe_load(f)


def actor(cfg, sd):
    net = ActorNetwork(cfg)
    net.load_state_dict(sd)
    return net.eval()


def collect(ckpt, logs):
    """Solo states and what the teacher would command there. The teacher and the pilot take turns at
    the controls (drawn once a second), so the states are varied. Returns X (n, 74), Y (n, 3)."""
    sc = scn.load(os.path.join(ROOT, 'scenarios', 'ecity_rooftops.yaml'))
    sc['sim']['cruise_z_bias'] = 0.0
    layout, bounds, _ = rs.planner_for(sc)
    cfg, args = rs.sim_setup(sc, bounds)
    sch = {'enabled': True, **{k: getattr(args, k) for k in FLIGHT_OPTIONS if getattr(args, k, None) is not None}}
    cfg['environment'].update(n_agents=1, crash_mode='remove', schedule=sch, route_follow={'enabled': True},
                              max_steps=10 ** 9)
    dt = cfg['environment'].get('control_timestep', 0.032)
    env = SwarmEnv(cfg, device=torch.device('cpu'))
    disable_all_collision_physics(env, 1)
    pilot = GoalHoldPolicy(es.MappoPolicy(cfg, ckpt, torch.device('cpu'), True), cfg)
    teacher = RepulsionController(load(TEACHER))
    rng, X, Y = np.random.default_rng(20), [], []

    def fly(route, buildings, beta, seed, tmax=220):
        np.random.seed(seed)
        torch.manual_seed(seed)
        obs = env.start_episode({'drone_0': route[0]}, {'drone_0': route[-1]}, obstacles=buildings,
                                routes={'drone_0': route}, departures={'drone_0': 0.1})
        taught = False
        for k in range(int(tmax / dt)):
            if k % 30 == 0:
                taught = rng.random() < beta
            raw = env.adapter.get_raw_state()['drone_0']
            pos, vel = np.array(raw['position'])[None], np.array(raw['velocity'])[None]
            carrot = np.array(env.adapter.goals['drone_0'])[None]
            near = [o for o in buildings if np.hypot(o['center'][0] - pos[0, 0], o['center'][1] - pos[0, 1])
                    < 12 + np.hypot(*o['half'])]
            y = teacher.velocities(pos, vel, carrot, near, np.array(raw['lidar'])[None])[0]
            a = pilot.act(obs, env, 1)
            if not env.waiting[0] and k % 3 == 0:
                X.append(obs['local_fixed'].numpy()[0].copy())
                Y.append(y.copy())
            obs, _, _, _ = env.step({'drone_0': list(y) + [0.0]} if taught else a)
            if env.landed[0] or env.crashed[0]:
                break

    for j, deg in enumerate(np.arange(-180, 180, 20)):              # open ground, every 20 degrees
        x, y = 45 * np.cos(np.radians(deg)), 45 * np.sin(np.radians(deg))
        z = (6.0, 8.0, 14.0, 16.0)[j % 4]
        fly([(0.0, 0.0, 0.5), (0.0, 0.0, z), (x, y, z), (x, y, 0.5)], [], 0.5 if j % 2 else 0.15, 20 + j % 5)
    for name, ids in ROUTES:                                        # and over the city, on planned routes
        for d in flight_log.read(os.path.join(logs, name + '.json.gz'))['drones']:
            if d['id'] in ids:
                fly([tuple(map(float, w)) for w in d['route']], layout, 0.5, 21, 140)
    return np.array(X, dtype=float), np.array(Y, dtype=float)


def fit(net, X, Y, damping=1e-3, iters=25, n=6000):
    """The context c (32 numbers) that brings the pilot closest to the teacher at these states."""
    net = net.double()
    X, Y = torch.tensor(X), torch.tensor(Y)
    test = torch.arange(len(X)) % 5 == 0
    held, train = torch.nonzero(test)[:, 0], torch.nonzero(~test)[:, 0]
    train = train[torch.randperm(len(train), generator=torch.Generator().manual_seed(20))[:n]]

    def one(c, x):
        return net.action_mean_head(net.mlp(torch.cat([x, c])))[:3]

    def error(c, i):
        with torch.no_grad():
            return Y[i] - net.action_mean_head(net.mlp(torch.cat([X[i], c.expand(len(i), -1)], 1)))[:, :3]

    def loss(c, i):
        return float(error(c, i).pow(2).sum(1).mean())

    c = torch.zeros(CTX, dtype=torch.float64)
    print(f'  rms error on held-out states, context 0: {loss(c, held) ** 0.5:.3f} m/s')
    for _ in range(iters):
        J = vmap(jacrev(one), in_dims=(None, 0))(c, X[train]).detach()
        H = torch.einsum('nij,nik->jk', J, J) / len(train) + damping * torch.eye(CTX, dtype=torch.float64)
        step = torch.linalg.solve(H, torch.einsum('nij,ni->j', J, error(c, train)) / len(train))
        before, scale = loss(c, train), 1.0
        while scale > 1e-3 and loss(c + scale * step, train) >= before:
            scale /= 2
        if scale <= 1e-3:
            break
        c = c + scale * step
        if before - loss(c, train) < 1e-5:
            break
    print(f'  rms error on held-out states, fitted c:  {loss(c, held) ** 0.5:.3f} m/s')
    return c


def edit(sd, c):
    """A copy of the actor's weights with c written in."""
    new = {k: v.clone() for k, v in sd.items()}
    new['mlp.0.bias'] = (sd['mlp.0.bias'].double() + sd['mlp.0.weight'].double()[:, LOCAL:LOCAL + CTX] @ c).float()
    new['attention.value_proj.bias'] = (sd['attention.value_proj.bias'].double() - c).float()
    return new


def check(cfg, old, new, n=200000):
    """Prints how many weights differ and the biggest change in any command with 1-5 neighbours
    (random inputs); returns the latter (m/s)."""
    changed = sum(int((new[k] != old[k]).sum()) for k in old)
    total = sum(v.numel() for v in old.values())
    g = torch.Generator().manual_seed(20)
    own, nbr = torch.randn(n, LOCAL, generator=g), torch.randn(n, 5, 6, generator=g) * 2
    mask = torch.arange(5)[None] >= torch.randint(1, 6, (n, 1), generator=g)
    nbr[mask] = 0
    with torch.no_grad():
        diff = float((actor(cfg, old)(own, nbr, mask)[0] - actor(cfg, new)(own, nbr, mask)[0]).abs().max())
    print(f'edited {changed} of {total} weights ({100 * changed / total:.2f}%); '
          f'with a neighbour the commands change by at most {diff:.0e} m/s', flush=True)
    return diff


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--checkpoint', default=os.path.join(ROOT, CKPT))
    ap.add_argument('--logs', default=os.path.join(ROOT, 'eval_results', 'scenarios'),
                    help='the run_scenario.py logs of the two scenes the city routes come from')
    ap.add_argument('--rounds', type=int, default=2)
    ap.add_argument('--out', default=os.path.join(ROOT, 'eval_results', 'solo_fix', 'edit2.pth'))
    a = ap.parse_args()
    cfg = load(CONFIG)
    ck = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    old, flown, X, Y = ck['actor_state_dict'], a.checkpoint, [], []
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    for r in range(a.rounds):
        x, y = collect(flown, a.logs)
        X.append(x)
        Y.append(y)
        print(f'round {r + 1}: {len(x)} solo states', flush=True)
        new = edit(old, fit(actor(cfg, old), np.concatenate(X), np.concatenate(Y)))
        check(cfg, old, new)
        flown = a.out
        torch.save({'actor_state_dict': new, **{k: v for k, v in ck.items() if k in KEEP}}, a.out)
    print('saved', a.out)


if __name__ == '__main__':
    main()
