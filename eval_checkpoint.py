"""
eval_checkpoint.py — Standalone checkpoint evaluator.

Loads a saved actor checkpoint and runs N episodes against SwarmEnv,
reporting success rate, min distance, and final speed. Supports both
deterministic (mean-action, matches training's periodic eval) and
stochastic (sampled-action, matches what training rollouts actually see)
modes.

Purpose: after effortless-sun-44 showed the deterministic mean policy
settling reliably (100% eval) well before stochastic training itself ever
produced a full 31-step settle, a single training run's handful of late
stochastic successes (5 out of the last 6 episodes) isn't a large enough
sample to trust as "robust." This script runs a much larger stochastic
sample from a frozen checkpoint to check whether that success rate holds up.

Usage:
    python eval_checkpoint.py --config configs/config_1agent_settle.yaml \
        --checkpoint checkpoints_settle/checkpoint_final.pth \
        --episodes 100 --mode stochastic

    python eval_checkpoint.py --config configs/config_1agent_settle.yaml \
        --checkpoint checkpoints_settle/checkpoint_final.pth \
        --episodes 20 --mode deterministic
"""

import argparse
import os
import sys

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(__file__))

from training.networks.actor_network import ActorNetwork
from envs.swarm_env import SwarmEnv
from record_rollout import (setup_crossing_scenario, setup_swap_scenario,
                             setup_headon_scenario, disable_all_collision_physics)


def unpack_obs(obs_dict, device):
    return (
        obs_dict['local_fixed'].to(device),
        obs_dict['neighbor_states'].to(device),
        obs_dict['neighbor_mask'].to(device),
        obs_dict['global_state'].to(device),
    )


def active_action_dims(curriculum_stage=1):
    """Matches MAPPOTrainer._active_action_dims: yaw (dim 3) stays masked
    and causally inert before curriculum Stage 3."""
    return 3 if curriculum_stage < 3 else 4


def apply_yaw_mask(actions, n_active_dims):
    if n_active_dims >= actions.shape[-1]:
        return actions
    masked = actions.clone()
    masked[:, n_active_dims:] = 0.0
    return masked


def actions_to_dict(actions, n_agents):
    arr = actions.detach().cpu().numpy()
    return {f'drone_{i}': arr[i].tolist() for i in range(n_agents)}


def main():
    parser = argparse.ArgumentParser(description="Evaluate a saved checkpoint over N episodes")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--mode", choices=["deterministic", "stochastic"], default="stochastic")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--log-std-ceiling", type=float, default=None,
                         help="Override the exploration log_std ceiling for stochastic "
                              "eval. Defaults to config['exploration']['log_std_final'] "
                              "if present, else the network's built-in default (0.5). "
                              "REQUIRED for a meaningful stochastic eval of an annealed "
                              "checkpoint — the annealed ceiling is a training-time "
                              "schedule value, not something saved in the checkpoint "
                              "itself (the underlying learned parameter typically never "
                              "moves once clamped, since torch.clamp's gradient is zero "
                              "outside the clamp range), so loading the checkpoint alone "
                              "silently resets it to 0.5 (std≈1.0) unless set explicitly.")
    parser.add_argument("--log-std-floor", type=float, default=None,
                         help="Override the log_std floor. Defaults to the checkpoint's "
                              "saved value if present, else the network's built-in "
                              "default (-3.0). Matters for checkpoints taken during a "
                              "floor-squeeze phase (exploration.log_std_floor_final).")
    parser.add_argument("--scenario", choices=["random", "crossing", "swap", "headon"],
                         default="random",
                         help="Same scenario generators as record_rollout.py / "
                              "baseline_pcontroller.py — use the same scenario+seed on "
                              "both this script and baseline_pcontroller.py for a direct, "
                              "apples-to-apples comparison against the P-controller baseline.")
    parser.add_argument("--disable-collision-physics", action="store_true",
                         help="Disable PyBullet's native rigid-body collision response "
                              "between every drone pair. Without this, drones are real "
                              "dynamic bodies whose own physics passively prevents true "
                              "interpenetration below ~0.3m regardless of policy — so any "
                              "collision-rate/separation claim WITHOUT this flag is partly "
                              "measuring the physics engine, not the policy. Use this for "
                              "the real collision-avoidance comparison.")
    args = parser.parse_args()

    config_path = os.path.join(os.path.dirname(__file__), args.config)
    with open(config_path) as f:
        config = yaml.safe_load(f)

    if args.seed is not None:
        import random
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    device = torch.device(args.device) if args.device else torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'
    )

    actor = ActorNetwork(config).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    state_dict = ckpt['actor_state_dict'] if 'actor_state_dict' in ckpt else ckpt
    actor.load_state_dict(state_dict)
    actor.eval()

    if 'episode' in ckpt:
        print(f"Loaded checkpoint from episode {ckpt['episode']} "
              f"(recorded success_rate={ckpt.get('success_rate', '?')}, "
              f"mean_min_dist={ckpt.get('mean_min_dist', '?')}, "
              f"final_speed={ckpt.get('final_speed', '?')})")
    else:
        print(f"Loaded checkpoint: {args.checkpoint} "
              f"(episode_count={ckpt.get('episode_count', '?')})")

    # CRITICAL: log_std_ceiling is a plain Python attribute, not part of the
    # actor's state_dict, so load_state_dict() above leaves it at the network's
    # default (0.5). If training annealed the ceiling down (see config's
    # 'exploration' block), the raw action_log_std parameter typically never
    # moved on its own — torch.clamp's gradient is zero outside the clamp
    # range, so once the ceiling pinned it there, no gradient reached it.
    # Without restoring the ceiling here, stochastic eval would silently run
    # at std≈1.0 instead of the intended ~0.05, invalidating the result.
    # Priority: explicit CLI override > the ACTUAL ceiling saved in this
    # checkpoint (most accurate — captures mid-anneal values too) > config's
    # eventual annealed target (only a guess if the checkpoint predates
    # ceiling tracking) > network default (almost certainly wrong for an
    # annealed run).
    ceiling = args.log_std_ceiling
    ceiling_source = "CLI override"
    if ceiling is None and 'log_std_ceiling' in ckpt:
        ceiling = ckpt['log_std_ceiling']
        ceiling_source = "checkpoint's saved value"
    if ceiling is None:
        ceiling = config.get('exploration', {}).get('log_std_final', None)
        ceiling_source = "config exploration.log_std_final (guess — checkpoint predates ceiling tracking)"
    if ceiling is not None:
        actor.set_log_std_ceiling(ceiling)
        print(f"log_std ceiling source: {ceiling_source}")
    elif args.mode == "stochastic":
        print("WARNING: no log_std ceiling override given, checkpoint has no "
              "saved ceiling, and config has no 'exploration.log_std_final' "
              f"— using network default ceiling ({actor.log_std_ceiling}). "
              "If this checkpoint came from an annealed run, this eval is "
              "almost certainly NOT using the trained exploration noise level.")

    # Same priority scheme for the floor. Matters for checkpoints taken
    # during a floor-squeeze phase (see config's exploration.log_std_floor_final)
    # where the effective bound in play was tighter than the network's
    # hardcoded -3.0 default.
    floor = args.log_std_floor
    floor_source = "CLI override"
    if floor is None and 'log_std_floor' in ckpt:
        floor = ckpt['log_std_floor']
        floor_source = "checkpoint's saved value"
    if floor is not None:
        actor.set_log_std_floor(floor)
        print(f"log_std floor source:   {floor_source}")

    effective_log_std = torch.clamp(
        actor.action_log_std, actor.log_std_floor, actor.log_std_ceiling
    )
    n_active = active_action_dims(ckpt.get('curriculum_stage', 1))

    n_agents = config['environment']['n_agents']
    max_steps = config['environment']['max_steps']
    deterministic = (args.mode == "deterministic")

    env = SwarmEnv(config, device=device)
    if args.disable_collision_physics:
        disable_all_collision_physics(env, n_agents)
        print(f"Collision physics DISABLED between all {n_agents} drone pairs — "
              f"any avoidance shown is purely from the policy, not PyBullet.")

    # Restore the training-time observation normalizer stats if the
    # checkpoint has them. Without this, eval's ObservationProcessor starts
    # a FRESH RunningNormalizer (mean=0, var=1, count=0) that only converges
    # to this eval run's own statistics after ~1000 samples — a mismatch
    # against whatever normalization the actor was actually trained on.
    normalizer_restored = 'normalizer_mean' in ckpt
    if normalizer_restored:
        norm = env.obs_processor.normalizer
        norm.mean  = ckpt['normalizer_mean']
        norm.var   = ckpt['normalizer_var']
        norm.count = ckpt['normalizer_count']
        # Freeze so eval observations don't drift the restored stats away
        # from what the actor was actually trained on — without this, the
        # normalizer keeps updating every step and slowly changes the
        # actor's effective inputs mid-evaluation.
        norm.freeze()
    else:
        print("WARNING: checkpoint has no saved observation-normalizer state "
              "(older checkpoint, saved before this was tracked). Eval will "
              "accumulate its own normalizer stats from scratch as it runs — "
              "NOT frozen — which is not guaranteed to match what the actor "
              "was trained on (early actions are taken under wrongly-scaled "
              "observations, which changes the states used to build the new "
              "stats). Treat results from this path as an approximate smoke "
              "test only, not a trustworthy robustness measurement — retrain "
              "to get a checkpoint with normalizer state, then re-evaluate.")

    print(f"\n{'-'*60}")
    print("Evaluation diagnostics")
    print(f"{'-'*60}")
    print(f"checkpoint episode:       {ckpt.get('episode', ckpt.get('episode_count', '?'))}")
    print(f"normalizer restored:      {'yes' if normalizer_restored else 'no'}")
    print(f"normalizer updates enabled: {'no (frozen)' if normalizer_restored else 'yes (unfrozen — see warning above)'}")
    print(f"log_std ceiling:          {actor.log_std_ceiling}")
    print(f"log_std floor:            {actor.log_std_floor}")
    print(f"active action std:        {effective_log_std[:n_active].exp().mean().item():.4f}")
    print(f"active action dims:       {n_active}")
    # NOTE: PyBulletAdapter always reads config['environment']['training'] for
    # physics_timestep/action_repeat — config['environment']['evaluation'] is
    # DEAD CONFIG, never read anywhere in the codebase. This means both this
    # script AND MAPPOTrainer._run_eval's built-in eval_env use the SAME
    # (training) settings — there is no train/eval dynamics mismatch on this
    # axis, but the intended higher-fidelity eval physics (action_repeat=4,
    # physics_timestep=0.008) never actually runs anywhere.
    train_cfg = config['environment'].get('training', {})
    print(f"action_repeat in use:    {train_cfg.get('action_repeat', 2)} "
          f"(from environment.training — environment.evaluation is unused dead config)")
    print(f"physics_timestep in use: {train_cfg.get('physics_timestep', 1/240)}")
    print(f"{'-'*60}\n")

    successes, min_dists, final_speeds, max_settle_fracs, lengths = [], [], [], [], []

    # Safety / multi-agent metrics — mirrors MAPPOTrainer._run_eval() exactly
    # so both evaluators report identically. N/A-safe for n_agents==1.
    per_agent_success = [[] for _ in range(n_agents)]
    any_collision_episodes = []
    drone_collision_episodes = []
    obstacle_collision_episodes = []
    min_inter_drone_seps = []
    agents_settled_counts = []

    scenario_rng = np.random.RandomState(args.seed)
    for ep in range(args.episodes):
        if args.scenario == "crossing":
            obs = setup_crossing_scenario(env, scenario_rng, n_agents)
        elif args.scenario == "swap":
            obs = setup_swap_scenario(env, scenario_rng, n_agents)
        elif args.scenario == "headon":
            obs = setup_headon_scenario(env, scenario_rng, n_agents)
        else:
            obs = env.reset()
        lf, ns, nm, gs = unpack_obs(obs, device)
        ep_max_settle_frac = 0.0
        steps = 0
        info = {}
        ep_any_drone_collision = False
        ep_any_obstacle_collision = False
        ep_min_inter_drone = None

        for step in range(max_steps):
            with torch.no_grad():
                actions, _ = actor.get_action(
                    lf, ns, nm, deterministic=deterministic, active_dims=n_active
                )
            masked = apply_yaw_mask(actions, n_active)
            next_obs, _, done, info = env.step(actions_to_dict(masked, n_agents))
            ep_max_settle_frac = max(ep_max_settle_frac, info.get('settle_fraction', 0.0))
            steps += 1

            if any(info.get('drone_collision', [])):
                ep_any_drone_collision = True
            if any(info.get('obstacle_collision', [])):
                ep_any_obstacle_collision = True
            step_sep = info.get('min_inter_drone_distance', None)
            if step_sep is not None:
                ep_min_inter_drone = step_sep if ep_min_inter_drone is None \
                    else min(ep_min_inter_drone, step_sep)

            if done:
                break
            lf, ns, nm, gs = unpack_obs(next_obs, device)

        successes.append(float(info.get('success', False)))
        min_dists.append(info.get('min_dist_to_goal', float('nan')))
        final_speeds.append(info.get('mean_speed', float('nan')))
        max_settle_fracs.append(ep_max_settle_frac)
        lengths.append(steps)

        reached = info.get('drone_reached_goal', [False] * n_agents)
        for i in range(n_agents):
            per_agent_success[i].append(float(reached[i]))
        agents_settled_counts.append(info.get('drones_at_goal', 0))
        any_collision_episodes.append(float(ep_any_drone_collision or ep_any_obstacle_collision))
        drone_collision_episodes.append(float(ep_any_drone_collision))
        obstacle_collision_episodes.append(float(ep_any_obstacle_collision))
        min_inter_drone_seps.append(ep_min_inter_drone)

        if (ep + 1) % 10 == 0 or ep == args.episodes - 1:
            running_sr = np.mean(successes)
            print(f"  [{ep+1:4d}/{args.episodes}] running success_rate={running_sr:.3f}")

    env.close()

    valid_seps = [s for s in min_inter_drone_seps if s is not None]
    min_sep_str = 'N/A' if not valid_seps else f"{min(valid_seps):.3f} m"

    print(f"\n{'='*60}")
    print(f"Mode: {args.mode}  |  Episodes: {args.episodes}  |  Checkpoint: {args.checkpoint}")
    print(f"{'='*60}")
    print(f"Success rate:        {np.mean(successes):.3f}  ({int(sum(successes))}/{args.episodes})")
    print(f"Mean min distance:   {np.mean(min_dists):.3f} m")
    print(f"Mean final speed:    {np.mean(final_speeds):.3f} m/s")
    print(f"Mean settle frac:    {np.mean(max_settle_fracs):.3f}")
    print(f"Mean episode length: {np.mean(lengths):.1f} steps")
    print(f"Mean agents settled: {np.mean(agents_settled_counts):.2f} / {n_agents}")
    for i in range(n_agents):
        print(f"Agent {i} success rate: {np.mean(per_agent_success[i]):.3f}")
    print(f"Any-collision episode rate:     {np.mean(any_collision_episodes):.3f}")
    print(f"Drone-drone collision rate:     {np.mean(drone_collision_episodes):.3f}")
    print(f"Obstacle collision rate:        {np.mean(obstacle_collision_episodes):.3f}")
    print(f"Min inter-drone separation:     {min_sep_str} (worst case across all episodes)")
    mean_sep_str = 'N/A' if not valid_seps else f"{np.mean(valid_seps):.3f} m"
    print(f"Mean inter-drone separation:     {mean_sep_str} (avg of each episode's closest approach)")

    # Wilson-ish quick confidence gut check: for a Bernoulli success rate p
    # over n episodes, report the naive standard error so a "95%" success
    # rate off 100 episodes can be told apart from noise off 6.
    p = np.mean(successes)
    n = args.episodes
    se = (p * (1 - p) / n) ** 0.5 if n > 0 else float('nan')
    print(f"Success rate std error: ±{se:.3f} (n={n})")


if __name__ == "__main__":
    main()
