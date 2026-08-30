"""
verify_eval_fix.py — one-off check that the _run_eval() normalizer-sync fix
actually repairs the built-in evaluator against an existing checkpoint,
without retraining.

Loads checkpoint_final.pth into a real MAPPOTrainer (actor + critic + env),
restores the training env's normalizer/log_std bounds from the checkpoint,
then calls the now-patched trainer._run_eval() directly and prints the
result. Expected (per eval_checkpoint.py's standalone result on the same
checkpoint): success_rate ≈ 1.0, mean_min_dist ≈ 0.058, settle_fraction ≈ 1.0.
"""

import argparse
import os
import sys

os.environ.setdefault("WANDB_MODE", "disabled")  # one-off verification, no need for a real run

import torch
import yaml

sys.path.insert(0, os.path.dirname(__file__))

from training.networks.actor_network import ActorNetwork
from training.networks.critic_network import CriticNetwork
from training.mappo_trainer import MAPPOTrainer
from envs.swarm_env import SwarmEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    actor = ActorNetwork(config)
    critic = CriticNetwork(config)
    env = SwarmEnv(config, device=device)

    trainer = MAPPOTrainer(config, actor, critic, env=env, device=device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    trainer.actor.load_state_dict(ckpt['actor_state_dict'])
    if 'critic_state_dict' in ckpt:
        trainer.critic.load_state_dict(ckpt['critic_state_dict'])

    # Restore the bounds and normalizer INTO THE TRAINING ENV — _run_eval()
    # syncs eval_env FROM self.env, so self.env must carry the checkpoint's
    # actual training-time state for the sync to reproduce the right numbers.
    if 'log_std_ceiling' in ckpt:
        trainer.actor.set_log_std_ceiling(ckpt['log_std_ceiling'])
    if 'log_std_floor' in ckpt:
        trainer.actor.set_log_std_floor(ckpt['log_std_floor'])
    if 'normalizer_mean' in ckpt:
        norm = trainer.env.obs_processor.normalizer
        norm.mean  = ckpt['normalizer_mean']
        norm.var   = ckpt['normalizer_var']
        norm.count = ckpt['normalizer_count']
        print(f"Restored training-env normalizer from checkpoint (count={norm.count}).")
    else:
        print("WARNING: checkpoint has no saved normalizer state — "
              "training env starts fresh, sync will carry near-empty stats.")

    trainer.episode_count = ckpt.get('episode_count', ckpt.get('episode', 0))
    print(f"Loaded checkpoint at episode {trainer.episode_count}. "
          f"actor log_std_ceiling={trainer.actor.log_std_ceiling}, "
          f"log_std_floor={trainer.actor.log_std_floor}")

    print("\nRunning trainer._run_eval() with the normalizer-sync fix applied...\n")
    metrics = trainer._run_eval()

    print("\n" + "=" * 60)
    print("Built-in _run_eval() result (post-fix)")
    print("=" * 60)
    for k, v in metrics.items():
        print(f"{k}: {v}")

    env.close()
    if trainer.eval_env is not None:
        trainer.eval_env.close()


if __name__ == "__main__":
    main()
