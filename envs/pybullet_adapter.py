"""
pybullet_adapter.py
-------------------
Stage-1 PyBullet backend for the UTM swarm.

This is the "hands" that touch the simulator. It is the ONLY file that knows we
are using PyBullet. It exposes the same small interface a WebotsAdapter would,
so SwarmEnv never has to care which simulator is underneath:

    reset_world(spawn_positions, goal_positions, obstacles=None)
    apply_actions(actions)          # {'drone_i': [vx, vy, vz, yaw_rate]}
    step_simulation()
    get_raw_state()  -> dict in the exact schema ObservationProcessor expects
    get_positions()  -> (N, 3) array
    close()

The raw-state schema (the "contract" a Webots adapter must also satisfy):
    {
      'drone_0': {'position':[x,y,z], 'velocity':[vx,vy,vz],
                  'orientation':[roll,pitch,yaw], 'lidar':[14 floats]},
      'drone_1': {...}, ...
      'goals':   {'drone_0':[gx,gy,gz], ...}
    }
'lidar' holds 14 distances in metres in the ray order of envs/lidar.py, with
the fan rotated so ray 0 points at the drone's goal; a ray that hits nothing
reports the LiDAR range. The ground is not reported.

Simplifications (deliberate, documented):
  - Velocity control: the policy's velocity command is applied directly to the
    drone body (resetBaseVelocity). Stable and ideal for learning basic
    navigation; realistic motor dynamics come later / in Webots.
  - Gravity disabled -> the drone is a free-flying velocity-controlled body.
  - Obstacles (Stage 2) are static bodies built from a layout produced by
    envs/scenario_manager.py; collisions with them are detected
    geometrically by SwarmEnv, so contact physics with them can be off.
  - LiDAR is off unless lidar.enabled is set; then every ray reports
    "nothing seen", exactly what Stage-1 checkpoints were trained on.
"""

import numpy as np
import pybullet as p
import pybullet_data

from envs import lidar as lidar_mod

RAY_START = 0.3     # rays start this far from the drone centre, outside its own body


class PyBulletAdapter:
    def __init__(self, config, gui: bool = False):
        self.config   = config
        self.n_agents = config['environment']['n_agents']
        self.v_max    = config['environment'].get('v_max', 2.0)
        self.omega_max = config['environment'].get('omega_max', 1.0)
        self.n_lidar  = config['observation'].get('n_lidar_rays', 14)

        train_cfg = config['environment'].get('training', {})
        self.physics_dt    = train_cfg.get('physics_timestep', 1.0 / 240.0)
        self.action_repeat = train_cfg.get('action_repeat', 2)

        self.drone_ids = []
        self.goals     = {}          # {'drone_i': np.array([x, y, z])}
        self.obstacle_ids = []       # bodies of the current episode's obstacles
        self.layout       = []       # the layout they were built from
        self.lidar_cfg    = lidar_mod.lidar_config(config)
        self._heading     = {}       # last LiDAR heading per drone (used when on the goal)

        # Velocity lag (environment.dynamics.lag): a real quadrotor follows a velocity
        # command with dead time and a first-order lag. Measured in Webots on a
        # Crazyflie flown by Bitcraze's velocity PID: horizontal dead time 0.10-0.25 s,
        # time constant 0.37-0.39 s; vertical (an altitude hold) ~0.1 s / ~0.14 s but
        # underdamped. Per control step:
        #     v[k+1] = v[k] + alpha * (v_cmd[k - d] - v[k]),  alpha = 1 - exp(-dt / tau)
        # with tau and d drawn per drone and episode from the configured ranges (defaults:
        # d 3-5 steps = 0.10-0.16 s, tau 0.33-0.43 s; with these the rule teachers score as in
        # Webots: original teacher 60/34/14% on building / wide / long wall vs 60/35/6%,
        # wide-gap teacher 96-100% vs 100%). Without
        # it (the default) the command is the velocity, as before.
        dyn = (config['environment'].get('dynamics') or {})
        self.lag = bool(dyn.get('lag', False))
        self.lag_tau_xy = tuple(dyn.get('tau_xy', (0.33, 0.43)))
        self.lag_tau_z = tuple(dyn.get('tau_z', (0.10, 0.20)))
        self.lag_delay = tuple(dyn.get('delay_steps', (3, 5)))
        self.control_dt = float(config['environment'].get('control_timestep', 0.032))

        # D5 drone failures (envs/disturbances.py), used only when SwarmEnv calls set_falling
        # (environment.failures on); otherwise nothing changes.
        fcfg = config['environment'].get('failures') or {}
        self.fall_g = float(fcfg.get('gravity', 9.81))
        self.fall_v_max = float(fcfg.get('fall_v_max', 10.0))
        self.fall_drag_tau = float(fcfg.get('fall_drag_tau', 1.0))
        self._falling = {}           # drone index -> its velocity while it falls (motor failure)

        # Every PyBullet call below passes physicsClientId=self.client. Without
        # it, calls go to world 0 by default, so a second adapter in the same
        # process (the trainer's eval env) put its drones into the TRAINING
        # world: eval ran with collision physics on, and its drones sat in the
        # training world as invisible obstacles after every evaluation.
        self.client = p.connect(p.GUI if gui else p.DIRECT)
        cid = self.client
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=cid)
        p.setGravity(0, 0, 0, physicsClientId=cid)    # Stage 1: free velocity control
        p.setTimeStep(self.physics_dt, physicsClientId=cid)
        p.setRealTimeSimulation(0, physicsClientId=cid)
        self.plane_id = p.loadURDF("plane.urdf", physicsClientId=cid)

        self._spawn_drones()

    def _spawn_drones(self):
        """Spawn N simple velocity-controlled bodies (no external URDF needed)."""
        half = 0.15
        for i in range(self.n_agents):
            col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[half] * 3,
                                         physicsClientId=self.client)
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[half] * 3,
                                      rgbaColor=[0.1, 0.4, 0.9, 1],
                                      physicsClientId=self.client)
            body = p.createMultiBody(
                baseMass=1.0,
                baseCollisionShapeIndex=col,
                baseVisualShapeIndex=vis,
                basePosition=[i * 2.0, 0.0, 1.0],
                physicsClientId=self.client,
            )
            self.drone_ids.append(body)

        # environment.collision_physics: false switches off PyBullet's
        # contact response between drones. With it on (the default), touching
        # boxes get pushed apart by the physics engine, so centres can't get
        # closer than ~0.3 m and the 0.2 m collision check (and its penalty)
        # never fires. Off means drones can overlap and a crash is a crash.
        if not self.config['environment'].get('collision_physics', True):
            for i in range(self.n_agents):
                for j in range(i + 1, self.n_agents):
                    p.setCollisionFilterPair(self.drone_ids[i], self.drone_ids[j], -1, -1,
                                             enableCollision=0, physicsClientId=self.client)

    def reset_world(self, spawn_positions: dict, goal_positions: dict, obstacles=None):
        """spawn_positions / goal_positions: {'drone_i': [x, y, z]}.
        obstacles: layout from envs/scenario_manager.py (None or [] = empty world)."""
        for i, body in enumerate(self.drone_ids):
            key = f'drone_{i}'
            p.resetBasePositionAndOrientation(body, spawn_positions[key], [0, 0, 0, 1],
                                              physicsClientId=self.client)
            p.resetBaseVelocity(body, [0, 0, 0], [0, 0, 0], physicsClientId=self.client)
            self.goals[key] = np.array(goal_positions[key], dtype=np.float32)
        self._heading = {}
        self._falling = {}
        if self.lag:
            n = len(self.drone_ids)
            tau = np.stack([np.random.uniform(*self.lag_tau_xy, n), np.random.uniform(*self.lag_tau_xy, n),
                            np.random.uniform(*self.lag_tau_z, n)], axis=1)
            self._alpha = 1.0 - np.exp(-self.control_dt / tau)                  # (n, 3)
            self._delay = np.random.randint(self.lag_delay[0], self.lag_delay[1] + 1, n)
            self._vel = np.zeros((n, 3), dtype=np.float32)
            self._queue = [[np.zeros(3, dtype=np.float32)] * int(d) for d in self._delay]
        self._build_obstacles(obstacles or [])

    def _build_obstacles(self, layout):
        """Replace the previous episode's obstacles with static bodies for this
        layout. Shapes change size every episode, so bodies are rebuilt rather
        than pooled (cheap in PyBullet; pooling matters in Webots)."""
        for b in self.obstacle_ids:
            p.removeBody(b, physicsClientId=self.client)
        self.obstacle_ids, self.layout = [], list(layout)
        contact = self.config['environment'].get('collision_physics', True)
        for ob in layout:
            h = ob['height']
            if ob['type'] == 'pillar':
                col = p.createCollisionShape(p.GEOM_CYLINDER, radius=ob['radius'], height=h,
                                             physicsClientId=self.client)
                vis = p.createVisualShape(p.GEOM_CYLINDER, radius=ob['radius'], length=h,
                                          rgbaColor=[0.55, 0.43, 0.39, 1], physicsClientId=self.client)
            else:
                hx, hy = ob['half']
                col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[hx, hy, h / 2],
                                             physicsClientId=self.client)
                vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[hx, hy, h / 2],
                                          rgbaColor=[0.6, 0.6, 0.62, 1], physicsClientId=self.client)
            body = p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=col,
                                     baseVisualShapeIndex=vis,
                                     basePosition=[ob['center'][0], ob['center'][1], h / 2],
                                     baseOrientation=p.getQuaternionFromEuler([0, 0, ob.get('yaw', 0.0)]),
                                     physicsClientId=self.client)
            if not contact:
                for d in self.drone_ids:
                    p.setCollisionFilterPair(d, body, -1, -1, enableCollision=0,
                                             physicsClientId=self.client)
            self.obstacle_ids.append(body)

    def apply_actions(self, actions: dict):
        """actions: {'drone_i': [vx, vy, vz, yaw_rate]} (yaw_rate optional)."""
        for i, body in enumerate(self.drone_ids):
            if i in self._falling:                                   # motor failure: commands ignored
                self._fall_step(i, body)
                continue
            a = actions[f'drone_{i}']
            v = np.clip(np.asarray(a[:3], dtype=np.float32), -self.v_max, self.v_max)
            if self.lag:
                self._queue[i].append(v)
                cmd = self._queue[i].pop(0)                          # issued d steps ago
                self._vel[i] = self._vel[i] + self._alpha[i] * (cmd - self._vel[i])
                v = self._vel[i]
            yaw_rate = float(np.clip(a[3], -self.omega_max, self.omega_max)) if len(a) > 3 else 0.0
            p.resetBaseVelocity(body, linearVelocity=v.tolist(),
                                angularVelocity=[0.0, 0.0, yaw_rate],
                                physicsClientId=self.client)

    def set_falling(self, i):
        """Motor failure (SwarmEnv environment.failures): from now on drone i ignores its commands
        and falls (gravity is off in this world, so the fall is integrated here): vz gains -g per
        second up to -fall_v_max, and the horizontal speed decays with fall_drag_tau. SwarmEnv
        ends it as a crash on the ground or a building."""
        lin, _ = p.getBaseVelocity(self.drone_ids[i], physicsClientId=self.client)
        self._falling[int(i)] = np.asarray(lin, dtype=np.float64)

    def _fall_step(self, i, body):
        dt = self.control_dt
        v = self._falling[i]
        v[:2] *= np.exp(-dt / self.fall_drag_tau)
        v[2] = max(v[2] - self.fall_g * dt, -self.fall_v_max)
        p.resetBaseVelocity(body, linearVelocity=v.tolist(), angularVelocity=[0.0, 0.0, 0.0],
                            physicsClientId=self.client)

    def park(self, indices):
        """Move these drones far out of the world and stop them (SwarmEnv crash_mode: remove).
        Each gets its own spot 100 m apart, so parked drones are out of every sensor's range
        and never touch each other."""
        for i in indices:
            self._falling.pop(int(i), None)
            p.resetBasePositionAndOrientation(self.drone_ids[i], [1e5 + 100.0 * i, 1e5, 50.0],
                                              [0, 0, 0, 1], physicsClientId=self.client)
            p.resetBaseVelocity(self.drone_ids[i], [0, 0, 0], [0, 0, 0], physicsClientId=self.client)
            if self.lag:
                self._vel[i] = 0.0
                self._queue[i] = [np.zeros(3, dtype=np.float32)] * len(self._queue[i])

    def place(self, i, pos):
        """Put drone i at pos, still (SwarmEnv schedule: a parked drone taking off from its pad)."""
        p.resetBasePositionAndOrientation(self.drone_ids[i], [float(v) for v in pos], [0, 0, 0, 1],
                                          physicsClientId=self.client)
        p.resetBaseVelocity(self.drone_ids[i], [0, 0, 0], [0, 0, 0], physicsClientId=self.client)
        if self.lag:
            self._vel[i] = 0.0
            self._queue[i] = [np.zeros(3, dtype=np.float32)] * len(self._queue[i])

    def step_simulation(self):
        for _ in range(self.action_repeat):
            p.stepSimulation(physicsClientId=self.client)

    def get_raw_state(self) -> dict:
        state = {'goals': {}}
        positions = []
        for i, body in enumerate(self.drone_ids):
            key = f'drone_{i}'
            pos, orn = p.getBasePositionAndOrientation(body, physicsClientId=self.client)
            lin, _   = p.getBaseVelocity(body, physicsClientId=self.client)
            euler    = p.getEulerFromQuaternion(orn)     # (roll, pitch, yaw)
            positions.append(np.array(pos))
            state[key] = {
                'position':    list(pos),
                'velocity':    list(lin),
                'orientation': list(euler),
            }
            state['goals'][key] = self.goals[key].tolist()
        scans, rings = self._scan_lidar(positions)
        for i in range(self.n_agents):
            state[f'drone_{i}']['lidar'] = scans[i]
            if self.lidar_cfg['ring_rays']:
                state[f'drone_{i}']['lidar_ring'] = rings[i]
        return state

    def _scan_lidar(self, positions):
        """14 fan distances per drone, plus lidar.ring_rays horizontal ring
        distances (metres, range = nothing hit), one batched ray test for the
        whole swarm. A ray that hits the ground first reports nothing; hits on
        other drones count only when lidar.sees_drones is set."""
        rng = self.lidar_cfg['range']
        n_ring = self.lidar_cfg['ring_rays']
        if not self.lidar_cfg['enabled']:
            return ([[rng] * self.n_lidar for _ in range(self.n_agents)],
                    [[rng] * n_ring for _ in range(self.n_agents)])
        yaws = []
        for i, pos in enumerate(positions):
            key = f'drone_{i}'
            yaw = lidar_mod.heading_to_goal(pos, self.goals[key], self._heading.get(key, 0.0))
            self._heading[key] = yaw
            yaws.append(yaw)
        # all rays of all drones as arrays (no Python loop per ray; the same values bit for bit)
        dirs = lidar_mod.ray_directions_batch(yaws)                       # (N, 14, 3)
        if n_ring:
            dirs = np.concatenate([dirs, lidar_mod.ring_directions_batch(yaws, n_ring)], axis=1)
        pos = np.asarray(positions, dtype=float)[:, None, :]
        starts = (pos + RAY_START * dirs).reshape(-1, 3).tolist()
        ends = (pos + rng * dirs).reshape(-1, 3).tolist()
        hits = p.rayTestBatch(starts, ends, physicsClientId=self.client)
        obj = np.fromiter((h[0] for h in hits), dtype=np.int64, count=len(hits))
        frac = np.fromiter((h[2] for h in hits), dtype=float, count=len(hits))
        seen = (obj >= 0) & (obj != self.plane_id)
        if not self.lidar_cfg['sees_drones']:
            seen &= ~np.isin(obj, self.drone_ids)
        dist = np.where(seen, RAY_START + frac * (rng - RAY_START), rng).reshape(self.n_agents, -1)
        return dist[:, :self.n_lidar].tolist(), dist[:, self.n_lidar:].tolist()

    def get_positions(self) -> np.ndarray:
        return np.array(
            [p.getBasePositionAndOrientation(b, physicsClientId=self.client)[0]
             for b in self.drone_ids],
            dtype=np.float32,
        )

    def close(self):
        try:
            p.disconnect(physicsClientId=self.client)
        except Exception:
            pass
