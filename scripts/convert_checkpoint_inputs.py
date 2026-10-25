"""
convert_checkpoint_inputs.py — turn an actor trained on observation schema v1
(27 inputs, running normalizer) into one for schema v2 (42 inputs, fixed
scaling), without retraining.

Why this works: a frozen v1 normalizer is a fixed linear map,
    z = (x - mean) / std
and v2 is another fixed linear map of the same raw values (below the caps),
    f = x / scale
so x = f * scale and z = (f * scale - mean) / std. The only layers that read
the local inputs directly are local_proj (attention query) and the first MLP
layer; folding that map into their weights and biases gives the same outputs
from the v2 inputs. Per input group:

    position, velocity, goal vector, goal distance : weight * scale / std,
                                                     bias - weight * mean / std
    orientation      : always exactly 0 in PyBullet; its constant contribution
                       goes into the bias and the weight becomes 0 (v1 divided
                       it by a ~0.001 spread, so any real tilt would explode)
    LiDAR proximity  : unchanged (v1 already used a fixed scale)
    new inputs       : previous LiDAR reading (14) and log goal distance (1)
                       get zero weights, so they change nothing until training
                       learns to use them

Above the v2 caps (goal vector 2.5 m, distance 4 m) the outputs differ on
purpose: v1 clipped each goal axis separately and bent the direction of far
goals; v2 keeps the direction.

The actor optimizer state is dropped (the first-layer shapes changed); the
critic, its optimizer and everything else are copied.

Usage:
    python scripts/convert_checkpoint_inputs.py \
        --checkpoint checkpoints/rl_frozen_norm/seed1/checkpoint_final.pth \
        --config configs/config_stage2_obstacles.yaml \
        --out checkpoints/rl_frozen_norm/seed1/checkpoint_final_v2.pth

Later schemas only append inputs, so those steps pad instead of fold: the new
columns of local_proj and the first MLP layer start at zero and the outputs are
unchanged until training uses them.
    v2 -> v3 : the horizontal LiDAR ring          (pass a v3 config)
    v3 -> v4 : the drone's last K velocity commands (pass a v4 config)
For v3 -> v4, convert both files an imitation run writes: bc_resume.pth (for
imitation_warmstart.py --base) and bc_policy.pth (for --init-actor).
"""

import argparse
import os
import sys

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.networks.actor_network import ActorNetwork
from modules.observation_processor import ObservationProcessor, RunningNormalizer

V1_DIM, V2_DIM = 27, 42


def fold_layer(weight, bias, mean, std, scales, z_orient, n_tail):
    """Fold the v1 normalization into one layer that reads the local inputs.

    weight : (out, 27 + n_tail)  columns 0..26 are the v1 local inputs,
                                 the n_tail after them (context) are moved along
    """
    out_dim = weight.shape[0]
    w_new = torch.zeros(out_dim, V2_DIM + n_tail, dtype=weight.dtype)
    b_new = bias.clone()
    w_loc = weight[:, :V1_DIM]
    for j in range(13):
        if 6 <= j <= 8:                                   # orientation
            b_new += w_loc[:, j] * z_orient[j - 6]
            continue
        w_new[:, j] = w_loc[:, j] * scales[j] / std[j]
        b_new -= w_loc[:, j] * mean[j] / std[j]
    w_new[:, 13:27] = w_loc[:, 13:27]                     # LiDAR, unchanged
    w_new[:, V2_DIM:] = weight[:, V1_DIM:]                # attention context
    return w_new, b_new


def pad_v2_to_v3(ckpt, config, args):
    """v2 -> v3: the horizontal LiDAR ring is appended after the 42 v2 inputs.
    Its columns in local_proj and the first MLP layer start at zero, so the
    actor's outputs are unchanged until training uses the ring."""
    new_dim = config['observation']['local_obs_dim']
    ctx = config['observation']['neighbor_context_dim']
    cfg_v2 = yaml.safe_load(open(args.config))
    cfg_v2['observation'].update(schema='v2', local_obs_dim=V2_DIM, total_obs_dim=V2_DIM + ctx)
    old = ActorNetwork(cfg_v2)
    old.load_state_dict(ckpt['actor_state_dict'])
    new = ActorNetwork(config)
    state = {k: v.clone() for k, v in old.state_dict().items()}
    w = state['local_proj.weight']
    state['local_proj.weight'] = torch.cat([w, torch.zeros(w.shape[0], new_dim - V2_DIM)], dim=1)
    w = state['mlp.0.weight']
    state['mlp.0.weight'] = torch.cat([w[:, :V2_DIM], torch.zeros(w.shape[0], new_dim - V2_DIM),
                                       w[:, V2_DIM:]], dim=1)
    new.load_state_dict(state)

    rng = np.random.default_rng(0)
    n = 4096
    f2 = torch.tensor(rng.uniform(-1, 1, (n, V2_DIM)), dtype=torch.float32)
    f3 = torch.cat([f2, torch.tensor(rng.uniform(0, 1, (n, new_dim - V2_DIM)), dtype=torch.float32)], 1)
    ns = torch.tensor(rng.normal(size=(n, 5, 6)), dtype=torch.float32)
    nm = torch.tensor(rng.uniform(size=(n, 5)) < 0.4)
    with torch.no_grad():
        err = (old(f2, ns, nm)[0] - new(f3, ns, nm)[0]).abs().max().item()
    print(f"max action difference v2 vs v3 (random ring readings): {err:.2e}")
    if err > 1e-5:
        sys.exit("conversion check failed, nothing written")
    out = {k: v for k, v in ckpt.items() if k != 'actor_optim'}
    out.update(actor_state_dict=new.state_dict(), observation_schema='v3',
               converted_from=args.checkpoint, config=config)
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    torch.save(out, args.out)
    print(f"written {args.out}")


def pad_v3_to_v4(ckpt, config, args):
    """v3 -> v4: the action history (the drone's last K velocity commands,
    observation.action_history) is appended after the v3 inputs. Its columns
    in local_proj and the first MLP layer start at zero, so the actor's
    outputs are unchanged until training uses the history. Works on both
    bc_policy.pth (actor only) and bc_resume.pth (full training checkpoint;
    the critic doesn't read these inputs and is copied unchanged)."""
    new_dim = config['observation']['local_obs_dim']
    ctx = config['observation']['neighbor_context_dim']
    v3_dim = ckpt['actor_state_dict']['local_proj.weight'].shape[1]   # 42 + ring rays
    cfg_v3 = yaml.safe_load(open(args.config))
    cfg_v3['observation'].update(schema='v3', local_obs_dim=v3_dim, total_obs_dim=v3_dim + ctx)
    cfg_v3['observation'].pop('action_history', None)
    old = ActorNetwork(cfg_v3)
    old.load_state_dict(ckpt['actor_state_dict'])
    new = ActorNetwork(config)
    extra = new_dim - v3_dim
    state = {k: v.clone() for k, v in old.state_dict().items()}
    w = state['local_proj.weight']
    state['local_proj.weight'] = torch.cat([w, torch.zeros(w.shape[0], extra)], dim=1)
    w = state['mlp.0.weight']
    state['mlp.0.weight'] = torch.cat([w[:, :v3_dim], torch.zeros(w.shape[0], extra),
                                       w[:, v3_dim:]], dim=1)
    new.load_state_dict(state)

    rng = np.random.default_rng(0)
    n = 4096
    f3 = torch.tensor(rng.uniform(-1, 1, (n, v3_dim)), dtype=torch.float32)
    f4 = torch.cat([f3, torch.tensor(rng.uniform(-1, 1, (n, extra)), dtype=torch.float32)], 1)
    ns = torch.tensor(rng.normal(size=(n, 5, 6)), dtype=torch.float32)
    nm = torch.tensor(rng.uniform(size=(n, 5)) < 0.4)
    with torch.no_grad():
        err = (old(f3, ns, nm)[0] - new(f4, ns, nm)[0]).abs().max().item()
    print(f"max action difference v3 vs v4 (random command histories): {err:.2e}")
    if err > 1e-5:
        sys.exit("conversion check failed, nothing written")
    out = {k: v for k, v in ckpt.items() if k != 'actor_optim'}
    out.update(actor_state_dict=new.state_dict(), observation_schema='v4',
               converted_from=args.checkpoint, config=config)
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    torch.save(out, args.out)
    print(f"written {args.out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument("--checkpoint", required=True,
                    help="schema v1 checkpoint (or v2 / v3 to pad to v3 / v4)")
    ap.add_argument("--config", required=True,
                    help="a schema v2 config (scales and network size), or the v3 / v4 target config")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    config = yaml.safe_load(open(args.config))
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if ckpt.get('observation_schema', 'v1') == 'v2' and config['observation'].get('schema') == 'v3':
        return pad_v2_to_v3(ckpt, config, args)
    if ckpt.get('observation_schema', 'v1') == 'v3' and config['observation'].get('schema') == 'v4':
        return pad_v3_to_v4(ckpt, config, args)
    if config['observation'].get('schema', 'v1') != 'v2':
        sys.exit("--config must use observation.schema: v2 (or v3 for a v2 checkpoint, "
                 "v4 for a v3 checkpoint)")
    if ckpt.get('observation_schema', 'v1') != 'v1':
        sys.exit(f"{args.checkpoint} is already schema {ckpt['observation_schema']}")
    if 'normalizer_mean' not in ckpt:
        sys.exit(f"{args.checkpoint} has no normalizer state to fold")

    proc = ObservationProcessor(config, torch.device('cpu'))
    norm = RunningNormalizer(V1_DIM)
    norm.mean = np.asarray(ckpt['normalizer_mean'], dtype=np.float32)
    norm.var = np.asarray(ckpt['normalizer_var'], dtype=np.float32)
    norm.count = int(ckpt['normalizer_count'])
    norm.freeze()
    mean = torch.tensor(norm.mean[:13], dtype=torch.float64)
    std = torch.tensor(np.sqrt(norm.var[:13]) + norm.epsilon, dtype=torch.float64)
    scales = torch.tensor(np.concatenate([proc.position_scale, [proc.velocity_scale] * 3,
                                          [np.pi] * 3, [proc.goal_vector_cap] * 3,
                                          [proc.goal_distance_cap]]), dtype=torch.float64)
    z_orient = torch.tensor(norm.normalize(np.zeros((1, V1_DIM), np.float32))[0, 6:9],
                            dtype=torch.float64)

    # v1 actor built from the same config with the v1 sizes
    cfg_v1 = yaml.safe_load(open(args.config))
    cfg_v1['observation'].update(schema='v1', local_obs_dim=V1_DIM,
                                 total_obs_dim=V1_DIM + config['observation']['neighbor_context_dim'])
    old = ActorNetwork(cfg_v1)
    old.load_state_dict(ckpt['actor_state_dict'])
    new = ActorNetwork(config)

    state = {k: v.clone() for k, v in old.state_dict().items()}
    for name, n_tail in (('local_proj', 0), ('mlp.0', config['observation']['neighbor_context_dim'])):
        w, b = fold_layer(state[f'{name}.weight'].double(), state[f'{name}.bias'].double(),
                          mean, std, scales, z_orient, n_tail)
        state[f'{name}.weight'], state[f'{name}.bias'] = w.float(), b.float()
    new.load_state_dict(state)
    for attr in ('log_std_ceiling', 'log_std_floor'):
        if attr in ckpt:
            setattr(new, attr, ckpt[attr])
            setattr(old, attr, ckpt[attr])

    # Check: same raw situations through both pipelines, below the caps
    rng = np.random.default_rng(0)
    n = 4096
    raw = np.zeros((n, V1_DIM), np.float32)
    raw[:, 0:2] = rng.uniform(-10, 10, (n, 2))
    raw[:, 2] = rng.uniform(1, 5, n)
    raw[:, 3:6] = rng.uniform(-2, 2, (n, 3))
    g = rng.normal(size=(n, 3))
    g *= (rng.uniform(0, 2.3, n) / np.linalg.norm(g, axis=1))[:, None]
    raw[:, 9:12] = g
    raw[:, 12] = np.linalg.norm(g, axis=1)
    prox = rng.uniform(0, 1, (n, 14)).astype(np.float32) * (rng.uniform(size=(n, 14)) < 0.3)
    raw[:, 13:27] = prox
    z = norm.normalize(raw)
    z[:, 13:27] = prox
    proc.reset_history()
    f = proc._fixed_scale(raw, prox)
    ns = torch.tensor(rng.normal(size=(n, 5, 6)), dtype=torch.float32)
    nm = torch.tensor(rng.uniform(size=(n, 5)) < 0.4)
    with torch.no_grad():
        a_old, _ = old(torch.tensor(z), ns, nm)
        a_new, _ = new(torch.tensor(f), ns, nm)
    err = (a_old - a_new).abs().max().item()
    print(f"max action difference v1 vs v2 on {n} random states below the caps: {err:.2e}")
    if err > 1e-4:
        sys.exit("conversion check failed, nothing written")

    out = {k: v for k, v in ckpt.items()
           if not k.startswith('normalizer_') and k != 'actor_optim'}
    out.update(actor_state_dict=new.state_dict(), observation_schema='v2',
               converted_from=args.checkpoint, config=config)
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    torch.save(out, args.out)
    print(f"written {args.out}")


if __name__ == "__main__":
    main()
