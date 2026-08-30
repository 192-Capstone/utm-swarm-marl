"""
baseline_pcontroller.py — Run a simple proportional (P) controller through
the same scenarios and metrics as eval_checkpoint.py, for a direct,
apples-to-apples comparison against the trained MAPPO policy.

The P-controller has NO learning, NO inter-drone awareness, and NO concept
of settling — it just does:

    velocity = clip(kp * (goal_position - drone_position), -v_max, v_max)

per drone, independently. It answers the question "did the RL policy learn
something non-trivial, or could simple geometry do just as well?"

Usage:
    python baseline_pcontroller.py --config configs/config_3agent_review.yaml \
        --episodes 100 --scenario random

    python baseline_pcontroller.py --config configs/config_3agent_review.yaml \
        --episodes 50 --scenario headon --seed 3
"""

import argparse
import os
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(__file__))

from envs.swarm_env import SwarmEnv
from record_rollout import (setup_crossing_scenario, setup_swap_scenario,
                             setup_headon_scenario, disable_all_collision_physics)


def main():
    parser = argparse.ArgumentParser(description="P-controller baseline evaluation")
    parser.add_argument("--config", required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--scenario", choices=["random", "crossing", "swap", "headon"],
                         default="random")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--kp", type=float, default=3.0,
                         help="Proportional gain — velocity = clip(kp * error, -v_max, v_max). "
                              "3.0 matches the gain used throughout this session's earlier "
                              "P-controller validation tests.")
    parser.add_argument("--disable-collision-physics", action="store_true",
                         help="Disable PyBullet's native rigid-body collision response "
                              "between every drone pair. Without this, drones are real "
                              "dynamic bodies whose own physics passively prevents true "
                              "interpenetration below ~0.3m regardless of policy — so any "
                              "collision-rate/separation claim WITHOUT this flag is partly "
                              "measuring the physics engine, not the P-controller itself.")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    n_agents = config['environment']['n_agents']
    max_steps = config['environment']['max_steps']
    v_max = config['environment'].get('v_max', 2.0)

    env = SwarmEnv(config)
    if args.disable_collision_physics:
        disable_all_collision_physics(env, n_agents)
        print(f"Collision physics DISABLED between all {n_agents} drone pairs — "
              f"any collisions/avoidance shown are purely from the controller, not PyBullet.")
    rng = np.random.RandomState(args.seed)

    successes, min_dists, final_speeds, max_settle_fracs, lengths = [], [], [], [], []
    per_agent_success = [[] for _ in range(n_agents)]
    any_collision_episodes, drone_collision_episodes, obstacle_collision_episodes = [], [], []
    min_inter_drone_seps, agents_settled_counts = [], []

    for ep in range(args.episodes):
        if args.scenario == "crossing":
            obs = setup_crossing_scenario(env, rng, n_agents)
        elif args.scenario == "swap":
            obs = setup_swap_scenario(env, rng, n_agents)
        elif args.scenario == "headon":
            obs = setup_headon_scenario(env, rng, n_agents)
        else:
            obs = env.reset()

        ep_max_settle_frac = 0.0
        ep_any_drone_collision = False
        ep_any_obstacle_collision = False
        ep_min_inter_drone = None
        steps = 0
        info = {}

        for step in range(max_steps):
            positions = env.adapter.get_positions()
            actions = {}
            for i in range(n_agents):
                err = env.goals[i] - positions[i]
                vel = np.clip(args.kp * err, -v_max, v_max)
                actions[f'drone_{i}'] = vel.tolist() + [0.0]

            _, _, done, info = env.step(actions)
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
            print(f"  [{ep+1:4d}/{args.episodes}] running success_rate={np.mean(successes):.3f}")

    env.close()

    valid_seps = [s for s in min_inter_drone_seps if s is not None]
    min_sep_str = 'N/A' if not valid_seps else f"{min(valid_seps):.3f} m"

    print(f"\n{'='*60}")
    print(f"P-CONTROLLER BASELINE  |  scenario={args.scenario}  |  "
          f"episodes={args.episodes}  |  kp={args.kp}")
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

    p = np.mean(successes)
    n = args.episodes
    se = (p * (1 - p) / n) ** 0.5 if n > 0 else float('nan')
    print(f"Success rate std error: ±{se:.3f} (n={n})")


if __name__ == "__main__":
    main()
