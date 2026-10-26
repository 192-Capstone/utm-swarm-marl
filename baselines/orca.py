"""
orca.py
-------
B1 baseline: ORCA (Optimal Reciprocal Collision Avoidance, van den Berg,
Guy, Lin and Manocha, "Reciprocal n-body collision avoidance", 2011), the
standard classical method for many agents sharing space. No learning.

Each step, every drone picks the velocity closest to "straight at my goal"
that keeps it clear of every neighbour for time_horizon seconds (sharing the
avoidance 50/50 with the neighbour) and of every obstacle for
time_horizon_obst seconds (taking full responsibility, obstacles don't move).
The velocity is found with the same small linear programs as the reference
library RVO2 (linear_program_1/2/3 below are a direct port), in the
horizontal plane; height is a simple P-controller towards the goal's height.

Two versions (eval_suite.py --policy):
    orca      ORCA-LiDAR: obstacles = the LiDAR ring hits, each a small static
              circle. Same information as the learned policy: the main, fair
              comparison.
    orca_map  ORCA-map: obstacles = the true obstacle shapes (privileged, like
              the rule teacher): a "best case for classical methods" reference.

What the drone knows comes from its own observation, with the same sensor
noise as the policy: goal vector local_fixed[9:12], own velocity [3:6], own
position [0:3] (ORCA-map only), the policy's neighbour slots (relative
position and velocity; the drone's own velocity is added back to get the
neighbour's), and the LiDAR ring [42:42+ring]. ORCA-map also reads env.layout.

Obstacles in ORCA-map: each pillar / box contributes its closest point to the
drone as a static circle (recomputed every step). For convex shapes this is a
simpler stand-in for RVO2's polygon obstacles.

Config (all optional; block 'orca'):
    orca:
      radius: 0.25              # m, per drone; two drones keep 2 x radius apart
      time_horizon: 2.0         # s, look-ahead for other drones
      time_horizon_obst: 1.0    # s, look-ahead for obstacles
      obstacle_radius: 0.3      # m, drone centre to obstacle point
      obstacle_dist: 3.0        # m, obstacle points further away are ignored
      vertical_clearance: 0.5   # m, neighbours further above/below are ignored
      pref_speed: 2.0           # m/s, cruise speed towards the goal (<= v_max)
      k_goal: 1.2               # 1/s, slow down near the goal: min(pref_speed, k_goal * d)
      k_z: 1.0                  # 1/s, height P-controller
      vz_max: 1.0               # m/s
"""

import math

import numpy as np

from envs.scenario_manager import DRONE_RADIUS

EPS = 1e-5

DEFAULTS = {
    'radius': 0.25,
    'time_horizon': 2.0,
    'time_horizon_obst': 1.0,
    'obstacle_radius': 0.3,
    'obstacle_dist': 3.0,
    'vertical_clearance': 0.5,
    'pref_speed': 2.0,
    'k_goal': 1.2,
    'k_z': 1.0,
    'vz_max': 1.0,
}


# ------------------------------------------------------------- ORCA core (2D)
def det(a, b):
    return a[0] * b[1] - a[1] * b[0]


def orca_line(rel_pos, rel_vel, own_vel, combined_radius, time_horizon, dt, responsibility):
    """One ORCA half-plane (RVO2 Agent::computeNewVelocity).

    rel_pos : other position - own position
    rel_vel : own velocity - other velocity
    Returns (point, direction): allowed velocities lie to the LEFT of the
    directed line, i.e. det(direction, point - v) <= 0.
    """
    rel_pos = np.asarray(rel_pos, dtype=float)
    rel_vel = np.asarray(rel_vel, dtype=float)
    dist_sq = rel_pos @ rel_pos
    r_sq = combined_radius * combined_radius
    inv_th = 1.0 / time_horizon
    if dist_sq > r_sq:
        w = rel_vel - inv_th * rel_pos                 # from cut-off circle centre to rel_vel
        w_len_sq = w @ w
        dot1 = w @ rel_pos
        if dot1 < 0.0 and dot1 * dot1 > r_sq * w_len_sq:
            # project on the cut-off circle
            w_len = math.sqrt(w_len_sq)
            unit_w = w / w_len
            direction = np.array([unit_w[1], -unit_w[0]])
            u = (combined_radius * inv_th - w_len) * unit_w
        else:
            # project on a leg of the cone
            leg = math.sqrt(dist_sq - r_sq)
            if det(rel_pos, w) > 0.0:                  # left leg
                direction = np.array([rel_pos[0] * leg - rel_pos[1] * combined_radius,
                                      rel_pos[0] * combined_radius + rel_pos[1] * leg]) / dist_sq
            else:                                      # right leg
                direction = -np.array([rel_pos[0] * leg + rel_pos[1] * combined_radius,
                                       -rel_pos[0] * combined_radius + rel_pos[1] * leg]) / dist_sq
            u = (rel_vel @ direction) * direction - rel_vel
    else:
        # already overlapping: get apart within one time step
        inv_dt = 1.0 / dt
        w = rel_vel - inv_dt * rel_pos
        w_len = math.sqrt(w @ w)
        unit_w = w / w_len if w_len > EPS else np.array([1.0, 0.0])
        direction = np.array([unit_w[1], -unit_w[0]])
        u = (combined_radius * inv_dt - w_len) * unit_w
    return (np.asarray(own_vel, dtype=float) + responsibility * u, direction)


def linear_program_1(lines, line_no, radius, opt_velocity, direction_opt):
    """Best velocity on line line_no within the speed circle and all earlier
    lines; None when infeasible (RVO2 linearProgram1)."""
    point, direction = lines[line_no]
    dot = point @ direction
    disc = dot * dot + radius * radius - point @ point
    if disc < 0.0:
        return None                                    # line misses the speed circle
    sqrt_disc = math.sqrt(disc)
    t_left, t_right = -dot - sqrt_disc, -dot + sqrt_disc
    for i in range(line_no):
        p_i, d_i = lines[i]
        denominator = det(direction, d_i)
        numerator = det(d_i, point - p_i)
        if abs(denominator) <= EPS:
            if numerator < 0.0:
                return None                            # parallel and on the wrong side
            continue
        t = numerator / denominator
        if denominator >= 0.0:
            t_right = min(t_right, t)
        else:
            t_left = max(t_left, t)
        if t_left > t_right:
            return None
    if direction_opt:
        t = t_right if opt_velocity @ direction > 0.0 else t_left
    else:
        t = min(max(direction @ (opt_velocity - point), t_left), t_right)
    return point + t * direction


def linear_program_2(lines, radius, opt_velocity, direction_opt):
    """Velocity closest to opt_velocity satisfying all lines within the speed
    circle. Returns (velocity, index of the first failing line or len(lines))
    (RVO2 linearProgram2)."""
    opt_velocity = np.asarray(opt_velocity, dtype=float)
    if direction_opt:
        result = opt_velocity * radius
    elif opt_velocity @ opt_velocity > radius * radius:
        result = opt_velocity / np.linalg.norm(opt_velocity) * radius
    else:
        result = opt_velocity.copy()
    for i, (p_i, d_i) in enumerate(lines):
        if det(d_i, p_i - result) > 0.0:               # result violates line i
            new = linear_program_1(lines, i, radius, opt_velocity, direction_opt)
            if new is None:
                return result, i
            result = new
    return result, len(lines)


def linear_program_3(lines, n_obst_lines, begin_line, radius, result):
    """Infeasible case: keep the obstacle lines hard and minimise the largest
    violation of the drone lines (RVO2 linearProgram3)."""
    distance = 0.0
    for i in range(begin_line, len(lines)):
        p_i, d_i = lines[i]
        if det(d_i, p_i - result) > distance:
            proj = list(lines[:n_obst_lines])
            for j in range(n_obst_lines, i):
                p_j, d_j = lines[j]
                determinant = det(d_i, d_j)
                if abs(determinant) <= EPS:
                    if d_i @ d_j > 0.0:
                        continue                       # same direction
                    point = 0.5 * (p_i + p_j)          # opposite direction
                else:
                    point = p_i + (det(d_j, p_i - p_j) / determinant) * d_i
                d = d_j - d_i
                n = np.linalg.norm(d)
                if n <= EPS:
                    continue
                proj.append((point, d / n))
            new, fail = linear_program_2(proj, radius, np.array([-d_i[1], d_i[0]]), True)
            if fail >= len(proj):
                result = new
            distance = det(d_i, p_i - result)
    return result


def orca_velocity(pref_velocity, obstacle_lines, agent_lines, max_speed):
    """The ORCA velocity: obstacle lines first (hard), then the drone lines."""
    lines = list(obstacle_lines) + list(agent_lines)
    v, fail = linear_program_2(lines, max_speed, pref_velocity, False)
    if fail < len(lines):
        v = linear_program_3(lines, len(obstacle_lines), fail, max_speed, v)
    return v


# ---------------------------------------------------------------- the policy
def _as_numpy(x):
    if hasattr(x, 'detach'):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


class OrcaPolicy:
    """ORCA as a policy for eval_suite.py: act(obs, env, n_agents) -> actions."""

    def __init__(self, config, use_map=False):
        block = dict(config.get('orca') or {})
        unknown = set(block) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"orca has unknown settings: {sorted(unknown)}")
        cfg = {**DEFAULTS, **block}
        for k in DEFAULTS:
            setattr(self, k, float(cfg[k]))
        self.v_max = float(config['environment'].get('v_max', 2.0))
        if self.pref_speed > self.v_max:
            raise ValueError("orca.pref_speed must not exceed environment.v_max")
        if self.radius <= 0 or self.obstacle_radius <= 0:
            raise ValueError("orca.radius and orca.obstacle_radius must be positive")
        if self.time_horizon <= 0 or self.time_horizon_obst <= 0:
            raise ValueError("orca time horizons must be positive")
        schema = config['observation'].get('schema', 'v1')
        if schema not in ('v3', 'v4'):
            raise ValueError("orca needs observation schema v3 or v4 (the LiDAR ring)")
        self.dt = float(config['environment'].get('control_timestep', 0.032))
        self.use_map = bool(use_map)
        self._yaw = {}                                   # last ring heading per drone

    # observation -> what the drone knows
    def _sense(self, obs, env):
        proc = env.obs_processor
        lf = _as_numpy(obs['local_fixed']).astype(np.float64)
        ns = _as_numpy(obs['neighbor_states']).astype(np.float64)
        empty = _as_numpy(obs['neighbor_mask']).astype(bool)
        goal_vec = lf[:, 9:12] * proc.goal_vector_cap          # length min(d, cap), direction exact
        goal_dist = lf[:, 12] * proc.goal_distance_cap         # capped at goal_distance_cap
        vel = lf[:, 3:6] * proc.velocity_scale
        pos = lf[:, 0:3] * proc.position_scale
        ring = lf[:, 42:42 + proc.ring_rays]
        nb_pos = ns[..., :3] * proc.neighbor_pos_scale         # neighbour - own
        nb_vel = ns[..., 3:6] * proc.neighbor_vel_scale        # neighbour - own
        return goal_vec, goal_dist, vel, pos, ring, nb_pos, nb_vel, empty

    def _ring_points(self, i, ring_i, goal_vec_i, lidar_range):
        """Horizontal positions (relative to the drone) of the ring hits.
        Ray 0 points along the heading to the goal, counter-clockwise."""
        if goal_vec_i[0] ** 2 + goal_vec_i[1] ** 2 >= 1e-4:
            self._yaw[i] = math.atan2(goal_vec_i[1], goal_vec_i[0])
        yaw = self._yaw.get(i, 0.0)
        n = len(ring_i)
        hit = ring_i > 0.0
        d = (1.0 - ring_i[hit]) * lidar_range
        a = yaw + 2.0 * np.pi * np.flatnonzero(hit) / n
        return np.stack([d * np.cos(a), d * np.sin(a)], axis=1)

    def _map_points(self, pos_i, layout):
        """Closest point of every obstacle tall enough to matter, relative to the drone."""
        from baselines.repulsion_controller import closest_point
        pts = []
        for ob in layout:
            if pos_i[2] > ob['height'] + DRONE_RADIUS + self.obstacle_radius:
                continue                                   # flying above it
            pts.append(closest_point(ob, pos_i[:2]) - pos_i[:2])
        return np.array(pts).reshape(-1, 2)

    def act(self, obs, env, n_agents):
        goal_vec, goal_dist, vel, pos, ring, nb_pos, nb_vel, empty = self._sense(obs, env)
        layout = getattr(env, 'layout', None) or []
        out = {}
        for i in range(n_agents):
            # preferred velocity: towards the goal, slowing down near it
            gh = goal_vec[i, :2]
            gh_len = float(np.linalg.norm(gh))
            g_len = float(np.linalg.norm(goal_vec[i]))
            if gh_len > 1e-6 and g_len > 1e-6:
                h_dist = goal_dist[i] * gh_len / g_len       # horizontal distance to the goal
                pref = gh / gh_len * min(self.pref_speed, self.k_goal * h_dist)
            else:
                pref = np.zeros(2)
            own_v = vel[i, :2]

            # obstacles: static, full responsibility
            pts = (self._map_points(pos[i], layout) if self.use_map
                   else self._ring_points(i, ring[i], goal_vec[i], env.lidar_range))
            obstacle_lines = [orca_line(p, own_v, own_v, self.obstacle_radius,
                                        self.time_horizon_obst, self.dt, 1.0)
                              for p in pts if p @ p <= self.obstacle_dist ** 2]

            # other drones: reciprocal, half the responsibility each
            agent_lines = []
            for k in range(nb_pos.shape[1] if nb_pos.ndim == 3 else 0):
                if empty[i, k] or abs(nb_pos[i, k, 2]) > self.vertical_clearance:
                    continue
                other_v = nb_vel[i, k, :2] + own_v
                agent_lines.append(orca_line(nb_pos[i, k, :2], own_v - other_v, own_v,
                                             2.0 * self.radius, self.time_horizon, self.dt, 0.5))

            v = orca_velocity(pref, obstacle_lines, agent_lines, self.pref_speed)
            vz = float(np.clip(self.k_z * goal_vec[i, 2], -self.vz_max, self.vz_max))
            out[f'drone_{i}'] = [float(v[0]), float(v[1]), vz, 0.0]
        return out
