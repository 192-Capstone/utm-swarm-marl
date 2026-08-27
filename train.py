"""
train.py — Stage-1 training launcher.

Usage:
    python train.py                     # default: Stage 1 on CPU
    python train.py --gui               # with PyBullet GUI window
    python train.py --episodes 2000     # custom episode count
    python train.py --resume checkpoints/checkpoint_ep500.pth

This wires SwarmEnv (PyBullet backend) to the MAPPO engine and starts
curriculum-based training. Stage 1 = empty world, no obstacles, yaw masked.
"""

import argparse
import os
import random
import sys
import yaml
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))

from training.networks.actor_network import ActorNetwork
from training.networks.critic_network import CriticNetwork
from training.mappo_trainer import MAPPOTrainer
from envs.swarm_env import SwarmEnv


def set_seed(seed: int):
    """Seed all RNGs so a run (network init + env spawns) is reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(description="MAPPO Stage-1 Training")
    parser.add_argument("--config", default="configs/default_config.yaml")
    parser.add_argument("--episodes", type=int, default=None,
                        help="Override total_episodes from config")
    parser.add_argument("--gui", action="store_true",
                        help="Open PyBullet GUI window")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--device", type=str, default=None,
                        help="Force device (cpu/cuda)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducible runs (network init + env spawns)")
    args = parser.parse_args()

    config_path = os.path.join(os.path.dirname(__file__), args.config)
    with open(config_path) as f:
        config = yaml.safe_load(f)

    if args.seed is not None:
        set_seed(args.seed)

    if args.episodes is not None:
        config["training"]["total_episodes"] = args.episodes

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Seed   : {args.seed}")
    print(f"Device : {device}")
    print(f"Agents : {config['environment']['n_agents']}")
    print(f"Episodes: {config['training']['total_episodes']}")
    print(f"Buffer : {config['training']['buffer_size']}")
    print(f"GUI    : {args.gui}")
    print()

    env = SwarmEnv(config, device=device, gui=args.gui)

    actor = ActorNetwork(config)
    critic = CriticNetwork(config)

    trainer = MAPPOTrainer(
        config=config,
        actor=actor,
        critic=critic,
        env=env,
        device=device,
    )

    if args.resume:
        trainer.load_checkpoint(args.resume)

    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving checkpoint...")
        trainer._save_checkpoint(tag="interrupted")
    finally:
        env.close()


if __name__ == "__main__":
    main()
