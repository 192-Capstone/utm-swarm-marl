"""
repulsion_controller.py — a rule-based controller that flies to its goal AND
steers clear of other drones. Used as the teacher for imitation learning
(the actor is trained to copy it before RL fine-tuning), and as a stronger
non-learning baseline than the plain P-controller.

Each drone, independently, every step:
    1. attraction : head for the goal, speed = min(cruise, k_goal * distance)
                    (slows smoothly near the goal so it can settle)
    2. avoidance  : for every other drone, predict the closest point of
                    approach (CPA) if both keep their current velocities:
                        t*  = when they will be closest (0..horizon seconds)
                        c   = where i will be relative to j at that moment
                    If they would pass closer than `safe_dist` within the
                    horizon, push i in the direction that widens the miss
                    (along c), harder the closer and the sooner it is.
                    If they are heading exactly at each other (c ~ 0), push
                    to the RIGHT of the relative motion, so everyone in a
                    head-on picks the same side and they pass like a
                    roundabout instead of deadlocking.
    3. separation : if another drone is already within `close_dist`, also push
                    straight away from it.
The total is capped at v_max per axis, like every other controller.

It uses only what a drone could know from neighbour broadcasts: relative
positions and velocities of nearby drones, plus its own goal.
"""

import numpy as np


class RepulsionController:
    def __init__(self, config, cruise=1.5, k_goal=2.0, safe_dist=1.0, horizon=2.0,
                 k_avoid=2.0, close_dist=0.6, k_close=2.0):
        self.v_max = config['environment'].get('v_max', 2.0)
        self.cruise, self.k_goal = cruise, k_goal
        self.safe_dist, self.horizon, self.k_avoid = safe_dist, horizon, k_avoid
        self.close_dist, self.k_close = close_dist, k_close

    @staticmethod
    def _right_of(v):
        """Unit vector 90 degrees to the right of v in the horizontal plane."""
        h = np.array([v[1], -v[0], 0.0])
        n = np.linalg.norm(h)
        return h / n if n > 1e-6 else np.array([1.0, 0.0, 0.0])

    def velocities(self, positions, velocities, goals):
        n = len(positions)
        out = np.zeros((n, 3))
        for i in range(n):
            err = goals[i] - positions[i]
            dist = np.linalg.norm(err)
            v = err / dist * min(self.cruise, self.k_goal * dist) if dist > 1e-6 else np.zeros(3)
            for j in range(n):
                if j == i:
                    continue
                p = positions[i] - positions[j]          # j -> i
                rv = velocities[i] - velocities[j]       # i's motion relative to j
                d = np.linalg.norm(p)
                # closest point of approach
                speed2 = float(rv @ rv)
                t_star = float(np.clip(-(p @ rv) / speed2, 0.0, self.horizon)) if speed2 > 1e-6 else 0.0
                c = p + rv * t_star
                dc = np.linalg.norm(c)
                if dc < self.safe_dist and (t_star < self.horizon or d < self.safe_dist):
                    direction = c / dc if dc > 0.05 else self._right_of(rv)
                    urgency = (self.safe_dist - dc) / self.safe_dist * (1.0 - t_star / self.horizon)
                    v = v + self.k_avoid * urgency * direction
                if 1e-6 < d < self.close_dist:
                    v = v + self.k_close * (self.close_dist - d) / self.close_dist * (p / d)
            out[i] = np.clip(v, -self.v_max, self.v_max)
        return out

    def act(self, obs, env, n_agents):
        """Same interface as the policies in eval_suite.py."""
        raw = env.adapter.get_raw_state()
        pos = np.array([raw[f'drone_{i}']['position'] for i in range(n_agents)])
        vel = np.array([raw[f'drone_{i}']['velocity'] for i in range(n_agents)])
        v = self.velocities(pos, vel, env.goals[:n_agents])
        return {f'drone_{i}': v[i].tolist() + [0.0] for i in range(n_agents)}
