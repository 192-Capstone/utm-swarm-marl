"""
episode_manager.py
------------------
Episode Manager for drone swarm training.

Tracks episode lifecycle, termination conditions, and curriculum progression.
Ensures sustained performance before advancing difficulty.
"""

import numpy as np


class EpisodeManager:
    """
    Episode Manager for drone swarm training.
    
    Responsibilities:
        - Track episode steps
        - Decide termination (collision / success / timeout)
        - Track success history for curriculum learning
        - Advance curriculum only when performance is sustained
    """
    
    def __init__(self, config):
        """
        Initialize Episode Manager with configuration.
        
        Args:
            config: Dictionary containing environment and curriculum settings
        """
        # Load from config
        self.max_steps = config["environment"]["max_steps"]
        self.goal_threshold = config["environment"]["goal_threshold"]
        self.window = config["curriculum"]["progression_window"]
        self.threshold = config["curriculum"]["progression_threshold"]
        
        # Internal state
        self.history = []          # Stores success flags (1=success, 0=failure)
        self.current_step = 0
        self.current_episode = 0
        
        # Statistics for logging
        self.stats = {
            "total_episodes": 0,
            "successes": 0,
            "collisions": 0,
            "timeouts": 0
        }
    
    def reset(self):
        """Reset episode counter for new episode."""
        self.current_step = 0
    
    def step(self):
        """Increment step counter and return current step."""
        self.current_step += 1
        return self.current_step
    
    def check_done(self, drone_positions, goal_positions, collisions):
        """
        Check if episode should end.
        
        Args:
            drone_positions: list of (x, y, z) tuples for each drone
            goal_positions: list of (x, y, z) tuples for each drone
            collisions: list of bool for each drone (True if collided)
        
        Returns:
            done: bool - True if episode should end
            reason: str - "collision", "success", or "timeout"
        """
        # Safety check for empty inputs
        if not drone_positions or not goal_positions:
            return False, None
        
        # 1. Any collision?
        if any(collisions):
            return True, "collision"
        
        # 2. All drones reached their goals?
        all_at_goal = True
        for pos, goal in zip(drone_positions, goal_positions):
            distance = np.linalg.norm(np.array(pos) - np.array(goal))
            if distance >= self.goal_threshold:
                all_at_goal = False
                break
        
        if all_at_goal:
            return True, "success"
        
        # 3. Timeout?
        if self.current_step >= self.max_steps:
            return True, "timeout"
        
        return False, None
    
    def record_episode_result(self, reason):
        """
        Record episode result for curriculum tracking.
        
        Args:
            reason: str - "success", "collision", or "timeout"
        """
        # Update statistics
        self.current_episode += 1
        self.stats["total_episodes"] += 1
        
        if reason == "success":
            self.stats["successes"] += 1
            success_flag = 1
        elif reason == "collision":
            self.stats["collisions"] += 1
            success_flag = 0
        else:  # timeout
            self.stats["timeouts"] += 1
            success_flag = 0
        
        # Add to sliding window
        self.history.append(success_flag)
        
        # Keep only last N episodes
        if len(self.history) > self.window:
            self.history.pop(0)
    
    def get_success_rate(self):
        """
        Get moving average success rate over last N episodes.
        
        Returns:
            float: success rate (0.0 to 1.0)
        """
        if len(self.history) == 0:
            return 0.0
        return np.mean(self.history)
    
    def should_advance_curriculum(self):
        """
        Check if curriculum should advance to next stage.
        
        Requires BOTH:
            1. Overall success rate over full window >= threshold
            2. Recent success rate (last 20% of window) >= threshold
        
        This ensures sustained performance, not just historical luck.
        
        Returns:
            bool: True if curriculum should advance
        """
        # Not enough episodes yet
        if len(self.history) < self.window:
            return False
        
        # Criteria 1: Overall performance over full window
        overall_rate = self.get_success_rate()
        if overall_rate < self.threshold:
            return False
        
        # Criteria 2: Recent performance (last 20% of window, min 10 episodes)
        recent_window = max(10, self.window // 5)
        recent_rate = np.mean(self.history[-recent_window:])
        
        return recent_rate >= self.threshold
    
    def get_stats(self):
        """
        Get episode statistics for logging.
        
        Returns:
            dict: Statistics for current training progress
        """
        return {
            "total_episodes": self.stats["total_episodes"],
            "successes": self.stats["successes"],
            "collisions": self.stats["collisions"],
            "timeouts": self.stats["timeouts"],
            "success_rate": self.get_success_rate(),
            "current_step": self.current_step
        }