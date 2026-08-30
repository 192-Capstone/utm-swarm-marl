import numpy as np

class RewardEngine:
    """
    Reward Engine for drone swarm.
    
    Computes 7 reward components for each drone:
    1. goal_progress: positive when moving toward goal
    2. goal_reached: large bonus when reaching goal
    3. obstacle_collision: penalty for hitting obstacles
    4. drone_collision: penalty for colliding with other drones
    5. step_penalty: small negative per step (encourages efficiency)
    6. smoothness_penalty: penalizes erratic actions
    7. proximity_penalty: penalizes being too close to other drones
    """
    
    def __init__(self, config):
        r = config["reward"]
        env = config["environment"]
        
        # Goal-related rewards
        self.goal_progress_scale = r["goal_progress_scale"]
        self.goal_reached_bonus = r["goal_reached_bonus"]
        self.goal_threshold = env["goal_threshold"]
        
        # Collision penalties
        self.obstacle_penalty = r["obstacle_collision_penalty"]
        self.drone_penalty = r["drone_collision_penalty"]
        self.collision_radius = env.get("collision_radius", 0.2)  # Add to config
        
        # Step penalty
        self.step_penalty_val = r["step_penalty"]
        
        # Smoothness penalty
        self.lambda_smooth = r["lambda_smooth"]
        
        # Proximity penalty
        self.lambda_prox = r["lambda_prox"]
        self.r_danger = r["r_danger"]

        # Braking penalty — teaches deceleration near the goal instead of
        # flying through at full speed. Defaults to 0 (no-op) so callers that
        # don't supply these keys (e.g. reward_validation.py's minimal config)
        # keep working unchanged.
        self.lambda_brake = r.get("lambda_brake", 0.0)

        self.settle_speed = env.get("settle_speed", 0.15)
        self.settle_shape_scale = r.get("settle_shape_scale", 10.0)
        self.slow_shaping_speed = r.get("slow_shaping_speed", 0.5)
        self.goal_entry_bonus = r.get("goal_entry_bonus", 10.0)
        self.settle_counter_scale = r.get("settle_counter_scale", 2.0)
    
    def compute_rewards(self, data):
        """
        Compute all reward components for all drones.
        
        Args:
            data: Dictionary with arrays for each drone:
                - prev_dist: (batch_size,) - previous distance to goal
                - curr_dist: (batch_size,) - current distance to goal
                - collision: (batch_size,) - boolean obstacle collision flags
                - drone_distances: (batch_size, n_pairs) - distances to other drones
                - action: (batch_size, 4) - current action
                - prev_action: (batch_size, 4) - previous action
                - speed: (batch_size,) - current drone speed (optional, defaults to 0)

        Returns:
            rewards: Dictionary with arrays of shape (batch_size,) for each component
        """
        batch_size = len(data["curr_dist"])
        speed = data.get("speed", np.zeros(batch_size))

        # Initialize reward arrays
        rewards = {
            "goal_progress": np.zeros(batch_size),
            "goal_reached": np.zeros(batch_size),
            "goal_entry": np.zeros(batch_size),
            "goal_hold": np.zeros(batch_size),
            "obstacle_collision": np.zeros(batch_size),
            "drone_collision": np.zeros(batch_size),
            "step_penalty": np.zeros(batch_size),
            "smoothness_penalty": np.zeros(batch_size),
            "proximity_penalty": np.zeros(batch_size),
            "settle_counter_progress": np.zeros(batch_size),
            "total": np.zeros(batch_size)
        }
        
        for i in range(batch_size):
            # 1. Goal Progress Reward
            progress = self.goal_progress_scale * (
                data["prev_dist"][i] - data["curr_dist"][i]
            )
            rewards["goal_progress"][i] = progress
            
            # 2. Goal Reached Bonus (one-time per drone per episode)
            if data.get("goal_just_reached", [False] * batch_size)[i]:
                rewards["goal_reached"][i] = self.goal_reached_bonus

            # 2b. Goal Entry Bonus — one-time reward for first entry into goal zone
            goal_entry_just = data.get("goal_entry_just", [False] * batch_size)
            if goal_entry_just[i]:
                rewards["goal_entry"][i] = self.goal_entry_bonus

            # 2c. Goal Hold — small per-step reward for staying near goal
            # after arrival, so the policy learns to stop rather than drift
            reached = data.get("drone_reached_goal", [False] * batch_size)
            if reached[i] and data["curr_dist"][i] < self.goal_threshold * 2:
                rewards["goal_hold"][i] = 0.1
            
            # 3. Obstacle Collision Penalty
            if data["collision"][i]:
                rewards["obstacle_collision"][i] = self.obstacle_penalty
            
            # 4. Drone Collision Penalty
            if len(data["drone_distances"][i]) > 0:
                if min(data["drone_distances"][i]) < self.collision_radius:
                    rewards["drone_collision"][i] = self.drone_penalty
            
            # 5. Step Penalty (constant for all)
            rewards["step_penalty"][i] = self.step_penalty_val
            
            # 6. Smoothness Penalty
            diff = data["action"][i] - data["prev_action"][i]
            smoothness = -self.lambda_smooth * np.sum(diff ** 2)
            rewards["smoothness_penalty"][i] = smoothness
            
            # 7. Proximity Penalty (per drone, sum over all other drones)
            prox_penalty = 0.0
            for d in data["drone_distances"][i]:
                if d < self.r_danger:
                    prox_penalty += ((self.r_danger - d) / self.r_danger) ** 2
            
            # wrong logic, needs to be reviewed 
            # if len(data["drone_distances"][i]) > 0:
            #     prox_penalty = prox_penalty / len(data["drone_distances"][i])
            
            rewards["proximity_penalty"][i] = -self.lambda_prox * prox_penalty

            # 8. Settle Counter Progress — rewards highwater-mark improvement
            # in consecutive settle steps. settle_counter_delta is computed by
            # SwarmEnv as max(0, new_best - old_best), so it is non-negative
            # and monotonically bounded at settle_steps per drone per episode
            # (i.e. settle_counter_scale × 31 = 62). This is what prevents an
            # oscillating drone from farming reward by repeatedly climbing
            # partway up the counter and resetting.
            counter_delta = data.get("settle_counter_delta", np.zeros(batch_size))[i]
            rewards["settle_counter_progress"][i] = self.settle_counter_scale * counter_delta

            # 9. Total Reward (weighted sum)
            rewards["total"][i] = sum([
                rewards["goal_progress"][i],
                rewards["goal_reached"][i],
                rewards["goal_entry"][i],
                rewards["goal_hold"][i],
                rewards["obstacle_collision"][i],
                rewards["drone_collision"][i],
                rewards["step_penalty"][i],
                rewards["smoothness_penalty"][i],
                rewards["proximity_penalty"][i],
                rewards["settle_counter_progress"][i],
            ])
        
        return rewards
    
    def compute_rewards_single(self, data):
        """
        Convenience method for single drone (for testing).
        
        Args:
            data: Dictionary with scalar values for a single drone
        """
        # Wrap scalars in arrays for compute_rewards
        batch_data = {
            "prev_dist": [data["prev_dist"]],
            "curr_dist": [data["curr_dist"]],
            "collision": [data["collision"]],
            "drone_distances": [data["drone_distances"]],
            "action": [data["action"]],
            "prev_action": [data["prev_action"]],
            "speed": [data.get("speed", 0.0)]
        }
        
        rewards_batch = self.compute_rewards(batch_data)
        
        # Return single dict with scalars
        return {k: v[0] for k, v in rewards_batch.items()}