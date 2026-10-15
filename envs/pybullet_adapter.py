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
                                     physicsClientId=self.client)
            if not contact:
                for d in self.drone_ids:
                    p.setCollisionFilterPair(d, body, -1, -1, enableCollision=0,
                                             physicsClientId=self.client)
            self.obstacle_ids.append(body)

    def apply_actions(self, actions: dict):
        """actions: {'drone_i': [vx, vy, vz, yaw_rate]} (yaw_rate optional)."""
        for i, body in enumerate(self.drone_ids):
            a = actions[f'drone_{i}']
            v = np.clip(np.asarray(a[:3], dtype=np.float32), -self.v_max, self.v_max)
            yaw_rate = float(np.clip(a[3], -self.omega_max, self.omega_max)) if len(a) > 3 else 0.0
            p.resetBaseVelocity(body, linearVelocity=v.tolist(),
                                angularVelocity=[0.0, 0.0, yaw_rate],
                                physicsClientId=self.client)

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
        per_drone = self.n_lidar + n_ring
        starts, ends = [], []
        for i, pos in enumerate(positions):
            key = f'drone_{i}'
            yaw = lidar_mod.heading_to_goal(pos, self.goals[key], self._heading.get(key, 0.0))
            self._heading[key] = yaw
            dirs = lidar_mod.ray_directions(yaw)
            if n_ring:
                dirs = np.vstack([dirs, lidar_mod.ring_directions(yaw, n_ring)])
            for d in dirs:
                starts.append((pos + RAY_START * d).tolist())
                ends.append((pos + rng * d).tolist())
        hits = p.rayTestBatch(starts, ends, physicsClientId=self.client)
        drones = set(self.drone_ids)
        out, rings = [], []
        for i in range(self.n_agents):
            row = []
            for k in range(per_drone):
                obj, _, frac = hits[i * per_drone + k][:3]
                seen = obj >= 0 and obj != self.plane_id and \
                    (obj not in drones or self.lidar_cfg['sees_drones'])
                row.append(RAY_START + frac * (rng - RAY_START) if seen else rng)
            out.append(row[:self.n_lidar])
            rings.append(row[self.n_lidar:])
        return out, rings

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
