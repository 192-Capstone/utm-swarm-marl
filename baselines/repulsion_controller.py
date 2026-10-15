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

Static obstacles (Stage 2), when a layout is given:
    5. obstacle avoidance : follow the drone's INTENDED path (straight towards
                    its goal) for the next obs_horizon seconds and find where
                    it comes closest to each obstacle's surface. If that is
                    closer than obs_safe, push SIDEWAYS (perpendicular to the
                    intended direction), towards the side the drone is already
                    offset to, so it goes around the obstacle rather than
                    backing away from it; dead-centre approaches veer right.
                    The chosen side is remembered until the obstacle is
                    cleared: without that memory a drone close to the centre
                    line flips sides every step and hovers in front of the
                    obstacle forever.
                    The intended path is used instead of the current velocity
                    so a drone that has slowed down in front of an obstacle
                    still "sees" the obstacle it is trying to fly through.
    6. obstacle separation: within obs_close of a surface, also push straight
                    away from it.
The obstacle geometry comes from the episode's layout (the teacher is
privileged); the learned actor has to make do with its LiDAR.

Side choice from the LiDAR (config teacher.side_from_lidar, Stage 2): with the
geometric rule above, a near dead-centre obstacle is passed left or right
depending on a few centimetres of offset the student's LiDAR cannot resolve,
so imitation sees the same picture labelled both ways and learns the average:
a weak dodge that clips the obstacle. With this switch the side is picked from
the drone's own LiDAR instead: pass on the right unless the front-left rays
are clearly more open than the front-right ones. Every label is then
something the student can see. How hard and when to dodge still come from
the exact geometry.
"""

import numpy as np

from envs.scenario_manager import surface_distance


def closest_point(ob, q):
    """Closest point of an obstacle's footprint (2D) to the horizontal point q."""
    c = np.asarray(ob['center'], dtype=float)
    if ob['type'] == 'pillar':
        d = q - c
        n = np.linalg.norm(d)
        return c + (d / n if n > 1e-9 else np.array([1.0, 0.0])) * ob['radius']
    hx, hy = ob['half']
    rel = q - c
    if abs(rel[0]) <= hx and abs(rel[1]) <= hy:            # inside: nearest edge
        if hx - abs(rel[0]) < hy - abs(rel[1]):
            return c + np.array([np.sign(rel[0]) * hx, rel[1]])
        return c + np.array([rel[0], np.sign(rel[1]) * hy])
    return c + np.array([np.clip(rel[0], -hx, hx), np.clip(rel[1], -hy, hy)])


class RepulsionController:
    def __init__(self, config, cruise=1.5, k_goal=2.0, safe_dist=1.0, horizon=2.0,
                 k_avoid=2.0, close_dist=0.6, k_close=2.0,
                 obs_safe=0.7, obs_horizon=2.0, k_obs=2.5, obs_close=0.4, k_obs_close=3.0):
        self.v_max = config['environment'].get('v_max', 2.0)
        self.cruise, self.k_goal = cruise, k_goal
        self.safe_dist, self.horizon, self.k_avoid = safe_dist, horizon, k_avoid
        self.close_dist, self.k_close = close_dist, k_close
        self.obs_safe, self.obs_horizon, self.k_obs = obs_safe, obs_horizon, k_obs
        self.obs_close, self.k_obs_close = obs_close, k_obs_close
        self._side = {}      # (drone index, obstacle centre) -> +1 right / -1 left, until cleared
        teacher_cfg = config.get('teacher') or {}
        self.side_from_lidar = bool(teacher_cfg.get('side_from_lidar', False))
        self.side_margin = float(teacher_cfg.get('side_margin', 0.5))   # metres
        # Only obstacles centred within side_band of the intended path count as
        # "ambiguous"; clearly off-centre ones are passed on the open side, as
        # before (from the LiDAR alone, wide boxes got passed the long way).
        self.side_band = float(teacher_cfg.get('side_band', 0.4))       # metres

    def _lidar_side(self, lidar):
        """+1 (right) unless the front-left rays (8, 12) see clearly further
        than the front-right rays (6, 10); ray order in envs/lidar.py."""
        right_clear = min(lidar[6], lidar[10])
        left_clear = min(lidar[8], lidar[12])
        return -1.0 if left_clear > right_clear + self.side_margin else 1.0

    def _obstacle_push(self, pos, v_goal, layout, i=0, lidar=None):
        """Velocity correction for drone i from the static obstacles."""
        push = np.zeros(3)
        p = pos[:2]
        u = v_goal[:2]
        speed = np.linalg.norm(u)
        if speed < 1e-6:
            u_dir = None
        else:
            u_dir = u / speed
        ts = np.linspace(0.0, self.obs_horizon, 11)
        for ob in layout:
            if pos[2] > ob['height']:
                continue
            # 5. where along the intended path does it come closest?
            if u_dir is not None:
                qs = p[None, :] + ts[:, None] * u[None, :]
                ds = np.array([surface_distance(ob, q) for q in qs])
                k = int(np.argmin(ds))
                d_min, t_star, q_star = ds[k], ts[k], qs[k]
                key = (i, tuple(ob['center']))
                if d_min < self.obs_safe:
                    right = np.array([u_dir[1], -u_dir[0]])
                    if key not in self._side:                    # first sighting: pick a side
                        # obstacle centre's offset to the right of the intended path
                        c_off = float((np.asarray(ob['center'], dtype=float)[:2] - p) @ right)
                        if self.side_from_lidar and lidar is not None and abs(c_off) < self.side_band:
                            self._side[key] = self._lidar_side(lidar)
                        elif self.side_from_lidar:
                            self._side[key] = -1.0 if c_off > 0 else 1.0   # pass on the open side
                        else:
                            away = q_star - closest_point(ob, q_star)
                            lateral = away - (away @ u_dir) * u_dir
                            off = float(lateral @ right)
                            self._side[key] = 1.0 if (abs(off) < 0.05 or off > 0) else -1.0
                    side = self._side[key] * right
                    urgency = (self.obs_safe - d_min) / self.obs_safe * (1.0 - t_star / self.obs_horizon)
                    push[:2] += self.k_obs * min(urgency, 2.0) * side
                else:
                    self._side.pop(key, None)                    # cleared: forget the side
            # 6. already close: push straight away from the surface
            d_now = surface_distance(ob, p)
            if d_now < self.obs_close:
                away = p - closest_point(ob, p)
                n = np.linalg.norm(away)
                if n > 1e-6:
                    push[:2] += self.k_obs_close * (self.obs_close - d_now) / self.obs_close * away / n
        return push

    @staticmethod
    def _right_of(v):
        """Unit vector 90 degrees to the right of v in the horizontal plane."""
        h = np.array([v[1], -v[0], 0.0])
        n = np.linalg.norm(h)
        return h / n if n > 1e-6 else np.array([1.0, 0.0, 0.0])

    def velocities(self, positions, velocities, goals, layout=None, lidar=None):
        """lidar: (n, 14) raw ray distances, used only with teacher.side_from_lidar."""
        n = len(positions)
        out = np.zeros((n, 3))
        if layout is not getattr(self, '_layout_ref', None):   # new episode: new layout object
            self._side.clear()
            self._layout_ref = layout
        for i in range(n):
            err = goals[i] - positions[i]
            dist = np.linalg.norm(err)
            v = err / dist * min(self.cruise, self.k_goal * dist) if dist > 1e-6 else np.zeros(3)
            if layout:
                v = v + self._obstacle_push(np.asarray(positions[i], dtype=float), v, layout, i,
                                            None if lidar is None else lidar[i])
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
        lidar = np.array([raw[f'drone_{i}']['lidar'] for i in range(n_agents)])
        v = self.velocities(pos, vel, env.goals[:n_agents], getattr(env, 'layout', None), lidar)
        return {f'drone_{i}': v[i].tolist() + [0.0] for i in range(n_agents)}
