"""
lidar_teacher.py — the teacher's rules, but obstacles seen ONLY through the
drone's own horizontal LiDAR ring (no map).

Why: the map-based teacher (repulsion_controller.py) reaches 99.8% while every
learned policy stalls around 74-77%. If rules that only see what the student
sees also stall there, the limit is the sensing; if they do much better, the
information is there and the student just hasn't learned to use it.

Each step, per drone:
    - every ring ray that hits something gives a point (drone position +
      distance x ray direction, at the drone's altitude); together they are the
      only obstacle "map" this controller has, rebuilt from scratch every step
      (no memory of anything out of view)
    - 5. path check: along the intended straight path to the goal for the next
      obs_horizon seconds, the closest approach to any hit point; if under
      obs_safe, push sideways, harder the closer and the sooner (same formula
      as the teacher)
    - side choice: the more open side of the ring's front half (right unless
      the left is clearly clearer), then kept until no hit point has threatened
      the path for `release_steps` steps; without that memory the side would
      flip between steps because the hit points move with the drone
    - 6. separation: within obs_close of the nearest hit point, push straight
      away from it
Drone-drone avoidance is the teacher's (neighbour broadcasts, unchanged).
Ring hits on other drones (lidar.sees_drones) are dropped: drones are handled
by the broadcast rules, as for the teacher.
"""

import numpy as np

from baselines.repulsion_controller import RepulsionController
from envs import lidar as lidar_mod


class LidarTeacher(RepulsionController):
    def __init__(self, config, release_steps=10, **kw):
        super().__init__(config, **kw)
        self.lidar_range = lidar_mod.lidar_config(config)['range']
        self.ring_rays = lidar_mod.lidar_config(config)['ring_rays']
        if self.ring_rays < 8:
            raise ValueError("LidarTeacher needs the horizontal LiDAR ring (lidar.ring_rays)")
        self.release_steps = release_steps
        self._lside, self._quiet = {}, {}

    def _hit_points(self, pos, ring, yaw, other_drones):
        dirs = lidar_mod.ring_directions(yaw, len(ring))[:, :2]
        d = np.asarray(ring, dtype=float)
        hit = d < self.lidar_range - 1e-3
        pts = pos[None, :2] + d[hit, None] * dirs[hit]
        if len(pts) and len(other_drones):                     # drop hits on other drones
            near = np.min(np.linalg.norm(pts[:, None, :] - other_drones[None, :, :2], axis=2), axis=1)
            pts = pts[near > 0.35]
        return pts, d, dirs

    def _open_side(self, ring_d, dirs, u_dir):
        """+1 right / -1 left: the side whose front quarter of the ring is more open."""
        right = np.array([u_dir[1], -u_dir[0]])
        fwd = dirs @ u_dir
        lat = dirs @ right
        front = fwd > 0.2
        r = ring_d[front & (lat > 0.2)]
        l = ring_d[front & (lat < -0.2)]
        r_clear = r.min() if len(r) else self.lidar_range
        l_clear = l.min() if len(l) else self.lidar_range
        return -1.0 if l_clear > r_clear + self.side_margin else 1.0

    def _lidar_push(self, i, pos, v_goal, pts, ring_d, dirs):
        push = np.zeros(3)
        if not len(pts):
            self._quiet[i] = self._quiet.get(i, 0) + 1
            if self._quiet[i] >= self.release_steps:
                self._lside.pop(i, None)
            return push
        p = pos[:2]
        u = v_goal[:2]
        speed = np.linalg.norm(u)
        threatened = False
        if speed > 1e-6:
            u_dir = u / speed
            ts = np.linspace(0.0, self.obs_horizon, 11)
            qs = p[None, :] + ts[:, None] * u[None, :]
            ds = np.linalg.norm(qs[:, None, :] - pts[None, :, :], axis=2).min(axis=1)
            k = int(np.argmin(ds))
            d_min, t_star = ds[k], ts[k]
            if d_min < self.obs_safe:
                threatened = True
                if i not in self._lside:
                    self._lside[i] = self._open_side(ring_d, dirs, u_dir)
                right = np.array([u_dir[1], -u_dir[0]])
                urgency = (self.obs_safe - d_min) / self.obs_safe * (1.0 - t_star / self.obs_horizon)
                push[:2] += self.k_obs * min(urgency, 2.0) * self._lside[i] * right
        self._quiet[i] = 0 if threatened else self._quiet.get(i, 0) + 1
        if self._quiet[i] >= self.release_steps:
            self._lside.pop(i, None)
        dn = np.linalg.norm(pts - p[None, :], axis=1)
        j = int(np.argmin(dn))
        if dn[j] < self.obs_close and dn[j] > 1e-6:
            push[:2] += self.k_obs_close * (self.obs_close - dn[j]) / self.obs_close * (p - pts[j]) / dn[j]
        return push

    def act(self, obs, env, n_agents):
        raw = env.adapter.get_raw_state()
        pos = np.array([raw[f'drone_{i}']['position'] for i in range(n_agents)], dtype=float)
        vel = np.array([raw[f'drone_{i}']['velocity'] for i in range(n_agents)], dtype=float)
        goals = env.goals[:n_agents]
        if getattr(env, 'layout', None) is not getattr(self, '_layout_ref2', None):   # new episode
            self._lside.clear(); self._quiet.clear()
            self._layout_ref2 = getattr(env, 'layout', None)
        # goal attraction + drone-drone rules from the teacher, with NO obstacle layout
        v = self.velocities(pos, vel, goals, layout=None)
        for i in range(n_agents):
            err = goals[i] - pos[i]
            dist = np.linalg.norm(err)
            v_goal = err / dist * min(self.cruise, self.k_goal * dist) if dist > 1e-6 else np.zeros(3)
            ring = raw[f'drone_{i}'].get('lidar_ring')
            yaw = env.adapter._heading.get(f'drone_{i}', 0.0)
            others = np.delete(pos, i, axis=0)
            pts, ring_d, dirs = self._hit_points(pos[i], ring, yaw, others)
            v[i] = np.clip(v[i] + self._lidar_push(i, pos[i], v_goal, pts, ring_d, dirs),
                           -self.v_max, self.v_max)
        return {f'drone_{i}': v[i].tolist() + [0.0] for i in range(n_agents)}
