"""
optuna_tune.py — Automated hyperparameter tuning for MAPPO Stage 1.

Uses Optuna with TPE sampler and MedianPruner to search over PPO
hyperparameters. Each trial runs training for a fixed episode budget,
reporting mean reward every 50 episodes so bad trials get killed early.

Usage:
    python optuna_tune.py                          # 20 trials, 300 eps each
    python optuna_tune.py --n-trials 50            # more trials
    python optuna_tune.py --episodes-per-trial 500 # longer per trial
    python optuna_tune.py --resume                 # continue previous study

Results are saved to an SQLite database (optuna_mappo.db) so you can
resume studies and inspect results with Optuna's dashboard:
    optuna-dashboard sqlite:///optuna_mappo.db
"""

import argparse
import copy
import os
import sys
import yaml
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from training.networks.actor_network import ActorNetwork
from training.networks.critic_network import CriticNetwork
from training.mappo_trainer import MAPPOTrainer
from envs.swarm_env import SwarmEnv


def create_objective(base_config: dict, device: torch.device,
                     episodes_per_trial: int, gui: bool):
    """
    Factory that returns an Optuna objective function.

    The objective function:
    1. Samples hyperparameters from the search space
    2. Overrides the base config
    3. Runs training for episodes_per_trial episodes
    4. Reports intermediate reward for pruning
    5. Returns mean reward over the last 100 episodes as the final score
    """

    def objective(trial: optuna.Trial) -> float:
        config = copy.deepcopy(base_config)

        # ── Search Space ─────────────────────────────────────────────
        # These are the parameters we've been manually tuning.
        # Ranges are informed by what we've tried so far.

        config['training']['entropy_coef'] = trial.suggest_float(
            "entropy_coef", 0.001, 0.02, log=True
        )
        config['training']['actor_lr'] = trial.suggest_float(
            "actor_lr", 3e-5, 5e-4, log=True
        )
        config['training']['critic_lr'] = trial.suggest_float(
            "critic_lr", 1e-4, 1e-3, log=True
        )
        config['training']['target_kl'] = trial.suggest_float(
            "target_kl", 0.02, 0.10
        )
        config['training']['batch_size'] = trial.suggest_categorical(
            "batch_size", [512, 1024, 2048]
        )

        config['training']['total_episodes'] = episodes_per_trial

        # Separate WandB run per trial so charts don't collide
        trial_name = f"optuna-trial-{trial.number}"
        config['logging']['wandb_project'] = "utm-swarm-marl-optuna"

        print(f"\n{'='*60}")
        print(f"Trial {trial.number}")
        print(f"  entropy_coef = {config['training']['entropy_coef']:.4f}")
        print(f"  actor_lr     = {config['training']['actor_lr']:.6f}")
        print(f"  critic_lr    = {config['training']['critic_lr']:.6f}")
        print(f"  target_kl    = {config['training']['target_kl']:.3f}")
        print(f"  batch_size   = {config['training']['batch_size']}")
        print(f"{'='*60}\n")

        # Fresh networks for each trial — no weight sharing between trials
        actor = ActorNetwork(config)
        critic = CriticNetwork(config)

        env = SwarmEnv(config, device=device, gui=gui)

        trainer = MAPPOTrainer(
            config=config,
            actor=actor,
            critic=critic,
            env=env,
            device=device,
        )

        try:
            trainer.train(optuna_trial=trial, report_interval=50)
        except optuna.TrialPruned:
            env.close()
            print(f"\nTrial {trial.number} PRUNED at episode {trainer.episode_count}")
            raise
        except Exception as e:
            env.close()
            print(f"\nTrial {trial.number} FAILED: {e}")
            return float('-inf')
        finally:
            # Finish the WandB run so the next trial gets a fresh one
            try:
                import wandb
                if wandb.run is not None:
                    wandb.finish(quiet=True)
            except Exception:
                pass

        env.close()

        # Score: mean reward over last 100 episodes
        window = min(100, len(trainer.episode_rewards))
        mean_reward = float(np.mean(trainer.episode_rewards[-window:]))

        # Bonus for any successes — heavily weight actual goal completion
        success_rate = float(np.mean(trainer.success_history[-window:]))
        score = mean_reward + 500.0 * success_rate

        print(f"\nTrial {trial.number} COMPLETE")
        print(f"  Mean reward (last {window}): {mean_reward:.1f}")
        print(f"  Success rate (last {window}): {success_rate:.3f}")
        print(f"  Final score: {score:.1f}")

        return score

    return objective


def main():
    parser = argparse.ArgumentParser(description="Optuna HPO for MAPPO")
    parser.add_argument("--config", default="configs/default_config.yaml")
    parser.add_argument("--n-trials", type=int, default=20,
                        help="Number of Optuna trials to run")
    parser.add_argument("--episodes-per-trial", type=int, default=300,
                        help="Training episodes per trial")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--resume", action="store_true",
                        help="Resume a previous study from the database")
    parser.add_argument("--db", default="sqlite:///optuna_mappo.db",
                        help="Optuna storage URL")
    parser.add_argument("--study-name", default="mappo-stage1",
                        help="Name for this study")
    args = parser.parse_args()

    config_path = os.path.join(os.path.dirname(__file__), args.config)
    with open(config_path) as f:
        base_config = yaml.safe_load(f)

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Optuna HPO — {args.n_trials} trials, {args.episodes_per_trial} eps each")
    print(f"Device: {device}")
    print(f"Study: {args.study_name}")
    print(f"DB: {args.db}")
    print()

    # TPE sampler: Bayesian optimization via Tree-structured Parzen Estimators
    sampler = TPESampler(seed=42)

    # MedianPruner: kill trials that fall below the median of completed trials
    # at the same reporting step. Waits until 5 trials finish before pruning.
    pruner = MedianPruner(
        n_startup_trials=5,
        n_warmup_steps=100,
    )

    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.db,
        sampler=sampler,
        pruner=pruner,
        direction="maximize",
        load_if_exists=args.resume,
    )

    objective = create_objective(
        base_config, device, args.episodes_per_trial, args.gui
    )

    study.optimize(objective, n_trials=args.n_trials)

    # ── Print results ────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("OPTUNA STUDY COMPLETE")
    print("=" * 60)

    print(f"\nBest trial: #{study.best_trial.number}")
    print(f"Best score: {study.best_value:.1f}")
    print("\nBest hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    print("\nAll trials:")
    for trial in study.trials:
        status = trial.state.name
        value = f"{trial.value:.1f}" if trial.value is not None else "N/A"
        params = ", ".join(f"{k}={v:.4g}" for k, v in trial.params.items())
        print(f"  #{trial.number:3d} [{status:9s}] score={value:>8s}  {params}")

    print(f"\nTo explore results interactively:")
    print(f"  pip install optuna-dashboard")
    print(f"  optuna-dashboard {args.db}")


if __name__ == "__main__":
    main()
