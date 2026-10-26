"""
goal_hold.py
------------
A1 part 3: once a drone is at its goal, a simple position controller holds it
there instead of the learned policy making the final stop.

Why: under realistic lag (0.10-0.29 s dead time, ~0.35-0.43 s response) the
imitation and RL policies reach the goal zone but keep drifting, and time out
without settling (validation suite: R1 51 drones, r7v3ft 75). A real
Crazyflie holds a position setpoint onboard; this is the PyBullet stand-in
for that handover.

GoalHoldPolicy wraps any policy that has .act(obs, env, n_agents) and changes
only the velocity command of drones that are holding:

    holding  starts when the drone's goal distance is under engage_dist and
             continues until it is over release_dist (hysteresis). A drone
             that has already settled is therefore held too.
    command  v = gain * (goal - position), capped at max_speed; yaw rate 0.
    guard    the wrapped policy keeps control while a neighbour is closer
             than neighbour_release or the LiDAR sees an obstacle closer than
             obstacle_release. Holding resumes only once neighbours are
             beyond neighbour_resume and obstacles beyond obstacle_resume.

Honest inputs: everything comes from the drone's own observation, with the
same sensor noise the policy sees (goal vector local_fixed[9:12], neighbour
slots, LiDAR). Nothing reads the simulator's true state or the settle flag.
(SwarmEnv.episode_index tells it when a new episode starts.)

Config (off when the block is missing or enabled is false):
    goal_hold:
      enabled: true
      engage_dist: 0.5         # m, the goal zone
      release_dist: 0.7        # m
      gain: 1.0                # 1/s; about 50 deg phase margin at worst-case lag
      max_speed: 0.3           # m/s
      neighbour_release: 0.9   # m, the teacher's close_dist
      neighbour_resume: 1.4    # m, the teacher's safe_dist
      obstacle_release: 0.5    # m
      obstacle_resume: 0.7     # m
"""

import numpy as np

DEFAULTS = {
    'engage_dist': 0.5,
    'release_dist': 0.7,
    'gain': 1.0,
    'max_speed': 0.3,
    'neighbour_release': 0.9,
    'neighbour_resume': 1.4,
    'obstacle_release': 0.5,
    'obstacle_resume': 0.7,
}

# The 14-ray LiDAR's straight-up and straight-down rays (schema v2 only),
# left out so the ground is never mistaken for an obstacle.
VERTICAL_RAYS = (4, 5)


def goal_hold_enabled(config):
    """True when the config switches goal-hold on."""
    return bool((config.get('goal_hold') or {}).get('enabled', False))


def _as_numpy(x):
    """Observation entries may be torch tensors or numpy arrays."""
    if hasattr(x, 'detach'):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


class GoalHoldPolicy:
    def __init__(self, inner, config):
        block = dict(config.get('goal_hold') or {})
        block.pop('enabled', None)
        unknown = set(block) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"goal_hold has unknown settings: {sorted(unknown)}")
        cfg = {**DEFAULTS, **block}
        for k in DEFAULTS:
            setattr(self, k, float(cfg[k]))
        if not self.engage_dist < self.release_dist:
            raise ValueError("goal_hold: engage_dist must be smaller than release_dist")
        if not self.neighbour_release < self.neighbour_resume:
            raise ValueError("goal_hold: neighbour_release must be smaller than neighbour_resume")
        if not self.obstacle_release < self.obstacle_resume:
            raise ValueError("goal_hold: obstacle_release must be smaller than obstacle_resume")
        if self.gain <= 0 or self.max_speed <= 0:
            raise ValueError("goal_hold: gain and max_speed must be positive")
        schema = config['observation'].get('schema', 'v1')
        if schema not in ('v2', 'v3', 'v4'):
            raise ValueError("goal_hold needs observation schema v2 or later "
                             "(the fixed-scale goal vector)")

        self.inner = inner
        self.holding = None          # per drone: inside the hold zone (with hysteresis)
        self.guarded = None          # per drone: guard has handed control back to the policy
        self._episode = None         # the episode the hold state belongs to
        self.stats = {'drone_steps': 0, 'holding_steps': 0, 'guard_steps': 0}

    def __getattr__(self, name):
        # Anything else (ckpt, restore_normalizer, ...) belongs to the wrapped policy.
        if name == 'inner':
            raise AttributeError(name)
        return getattr(self.inner, name)

    def act(self, obs, env, n_agents):
        actions = self.inner.act(obs, env, n_agents)
        self._start_episode_if_new(env, n_agents)

        goal_vec, neighbour_dist, obstacle_dist = self._sense(obs, env, n_agents)
        dist = np.linalg.norm(goal_vec, axis=1)

        self.holding = (dist < self.engage_dist) | (self.holding & (dist < self.release_dist))
        too_close = (neighbour_dist < self.neighbour_release) | (obstacle_dist < self.obstacle_release)
        clear = (neighbour_dist >= self.neighbour_resume) & (obstacle_dist >= self.obstacle_resume)
        self.guarded = too_close | (self.guarded & ~clear)
        active = self.holding & ~self.guarded

        for i in np.flatnonzero(active):
            v = self.gain * goal_vec[i]
            speed = float(np.linalg.norm(v))
            if speed > self.max_speed:
                v = v * (self.max_speed / speed)
            a = [float(x) for x in actions[f'drone_{i}']]
            a[:3] = v.tolist()
            a[3:] = [0.0] * (len(a) - 3)          # no yaw while holding
            actions[f'drone_{i}'] = a

        self.stats['drone_steps'] += n_agents
        self.stats['holding_steps'] += int(active.sum())
        self.stats['guard_steps'] += int((self.holding & self.guarded).sum())
        return actions

    # ------------------------------------------------------------------ helpers
    def _start_episode_if_new(self, env, n_agents):
        # SwarmEnv counts its episodes; an env without the counter falls back
        # to "the goals changed".
        episode = getattr(env, 'episode_index', None)
        if episode is None:
            episode = np.asarray(env.goals, dtype=np.float32).tobytes()
        if self.holding is None or len(self.holding) != n_agents or episode != self._episode:
            self.holding = np.zeros(n_agents, dtype=bool)
            self.guarded = np.zeros(n_agents, dtype=bool)
            self._episode = episode

    def _sense(self, obs, env, n_agents):
        """Goal vector (m), nearest neighbour (m) and nearest obstacle (m),
        all from the drone's own noisy observation."""
        proc = env.obs_processor
        lf = _as_numpy(obs['local_fixed']).astype(np.float64)

        # [9:12] = (goal - position) / goal_vector_cap, shrunk only beyond the
        # cap (2.5 m), so it is exact inside the hold zone.
        goal_vec = lf[:, 9:12] * proc.goal_vector_cap

        ns = _as_numpy(obs['neighbor_states']).astype(np.float64)
        empty = _as_numpy(obs['neighbor_mask']).astype(bool)        # True = empty slot
        neighbour_dist = np.full(n_agents, np.inf)
        if ns.ndim == 3 and ns.shape[1] > 0:
            d = np.linalg.norm(ns[:, :, :3], axis=2) * proc.neighbor_pos_scale
            d[empty] = np.inf
            neighbour_dist = d.min(axis=1)

        # LiDAR proximity = 1 - distance / range (0 = nothing seen). Schemas
        # v3/v4: the horizontal ring at [42:42+ring]; v2: the 14 rays minus
        # straight up/down.
        if proc.schema in ('v3', 'v4') and proc.ring_rays:
            prox = lf[:, 42:42 + proc.ring_rays]
        else:
            keep = [r for r in range(14) if r not in VERTICAL_RAYS]
            prox = lf[:, 13:27][:, keep]
        nearest = prox.max(axis=1)
        obstacle_dist = np.where(nearest > 0, (1.0 - nearest) * env.lidar_range, np.inf)
        return goal_vec, neighbour_dist, obstacle_dist
