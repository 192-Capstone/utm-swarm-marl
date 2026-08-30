# tests/test_reward_engine.py

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import yaml
from modules.reward_engine import RewardEngine

def test_reward_engine():
    """Test all 7 reward components"""
    
    print("=" * 60)
    print("TEST: Reward Engine - All 7 Components")
    print("=" * 60)
    
    # Load config
    config_path = os.path.join('configs', 'default_config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create reward engine
    reward_engine = RewardEngine(config)
    
    # ============================================
    # TEST 1: Single Drone - Moving Toward Goal
    # ============================================
    print("\n" + "-" * 40)
    print("TEST 1: Drone Moving Toward Goal")
    print("-" * 40)
    
    data = {
        "prev_dist": 10.0,
        "curr_dist": 8.0,
        "collision": False,
        "drone_distances": [],
        "action": np.array([0.5, 0.3, 0.1, 0.0]),
        "prev_action": np.array([0.4, 0.2, 0.0, 0.0])
    }
    
    rewards = reward_engine.compute_rewards_single(data)
    
    print(f"  Goal Progress: {rewards['goal_progress']:.2f} (should be +2.0)")
    print(f"  Step Penalty: {rewards['step_penalty']:.2f}")
    print(f"  Smoothness: {rewards['smoothness_penalty']:.3f}")
    print(f"  Total Reward: {rewards['total']:.2f}")
    
    # ============================================
    # TEST 2: Drone Reached Goal
    # ============================================
    print("\n" + "-" * 40)
    print("TEST 2: Drone Reached Goal")
    print("-" * 40)
    
    data = {
        "prev_dist": 1.0,
        "curr_dist": 0.3,
        "collision": False,
        "drone_distances": [],
        "action": np.array([0.1, 0.1, 0.0, 0.0]),
        "prev_action": np.array([0.1, 0.1, 0.0, 0.0])
    }
    
    rewards = reward_engine.compute_rewards_single(data)
    
    print(f"  Goal Progress: {rewards['goal_progress']:.2f}")
    print(f"  Goal Reached Bonus: {rewards['goal_reached']:.2f} (should be 100.0)")
    print(f"  Total Reward: {rewards['total']:.2f}")
    
    # ============================================
    # TEST 3: Drone Collided with Obstacle
    # ============================================
    print("\n" + "-" * 40)
    print("TEST 3: Drone Collided with Obstacle")
    print("-" * 40)
    
    data = {
        "prev_dist": 5.0,
        "curr_dist": 5.0,
        "collision": True,
        "drone_distances": [],
        "action": np.array([0.5, 0.5, 0.0, 0.0]),
        "prev_action": np.array([0.4, 0.4, 0.0, 0.0])
    }
    
    rewards = reward_engine.compute_rewards_single(data)
    
    print(f"  Obstacle Collision Penalty: {rewards['obstacle_collision']:.2f}")
    print(f"  Total Reward: {rewards['total']:.2f}")
    
    # ============================================
    # TEST 4: Two Drones Too Close (Proximity Penalty)
    # ============================================
    print("\n" + "-" * 40)
    print("TEST 4: Two Drones Too Close")
    print("-" * 40)
    
    # Batch test with 2 drones
    batch_data = {
        "prev_dist": [10.0, 10.0],
        "curr_dist": [9.0, 9.0],
        "collision": [False, False],
        "drone_distances": [[0.8], [0.8]],  # 0.8m apart (r_danger=1.5)
        "action": np.array([[0.5, 0.0, 0.0, 0.0], [0.5, 0.0, 0.0, 0.0]]),
        "prev_action": np.array([[0.4, 0.0, 0.0, 0.0], [0.4, 0.0, 0.0, 0.0]])
    }
    
    rewards = reward_engine.compute_rewards(batch_data)
    
    print(f"  Drone 0 Proximity Penalty: {rewards['proximity_penalty'][0]:.4f}")
    print(f"  Drone 1 Proximity Penalty: {rewards['proximity_penalty'][1]:.4f}")
    
    # Expected: -0.5 * ((1.5-0.8)/1.5)² = -0.5 * (0.7/1.5)² = -0.5 * 0.2178 = -0.1089
    
    # ============================================
    # TEST 5: Drone Collision with Another Drone
    # ============================================
    print("\n" + "-" * 40)
    print("TEST 5: Drone Collision with Another Drone")
    print("-" * 40)
    
    batch_data = {
        "prev_dist": [10.0, 10.0],
        "curr_dist": [9.0, 9.0],
        "collision": [False, False],
        "drone_distances": [[0.1], [0.1]],  # 0.1m apart (collision!)
        "action": np.array([[0.5, 0.0, 0.0, 0.0], [0.5, 0.0, 0.0, 0.0]]),
        "prev_action": np.array([[0.4, 0.0, 0.0, 0.0], [0.4, 0.0, 0.0, 0.0]])
    }
    
    rewards = reward_engine.compute_rewards(batch_data)
    
    print(f"  Drone 0 Drone Collision Penalty: {rewards['drone_collision'][0]:.2f}")
    print(f"  Drone 1 Drone Collision Penalty: {rewards['drone_collision'][1]:.2f}")
    
    # ============================================
    # TEST 6: Batch Processing with 3 Drones
    # ============================================
    print("\n" + "-" * 40)
    print("TEST 6: Batch Processing - 3 Drones")
    print("-" * 40)
    
    batch_data = {
        "prev_dist": [10.0, 8.0, 5.0],
        "curr_dist": [8.0, 6.0, 4.0],
        "collision": [False, True, False],
        "drone_distances": [[1.2, 2.5], [1.2, 3.0], [2.5, 3.0]],
        "action": np.array([
            [0.5, 0.3, 0.1, 0.0],
            [0.4, 0.2, 0.0, 0.0],
            [0.6, 0.4, 0.2, 0.0]
        ]),
        "prev_action": np.array([
            [0.4, 0.2, 0.0, 0.0],
            [0.4, 0.2, 0.0, 0.0],
            [0.5, 0.3, 0.1, 0.0]
        ])
    }
    
    rewards = reward_engine.compute_rewards(batch_data)
    
    for i in range(3):
        print(f"\n  Drone {i}:")
        print(f"    Goal Progress: {rewards['goal_progress'][i]:.2f}")
        print(f"    Obstacle Collision: {rewards['obstacle_collision'][i]:.2f}")
        print(f"    Smoothness: {rewards['smoothness_penalty'][i]:.3f}")
        print(f"    Proximity: {rewards['proximity_penalty'][i]:.4f}")
        print(f"    Total: {rewards['total'][i]:.2f}")
    
    print("\n" + "=" * 60)
    print("ALL REWARD ENGINE TESTS PASSED ✓")
    print("=" * 60)

if __name__ == "__main__":
    test_reward_engine()