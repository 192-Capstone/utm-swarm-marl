"""
pybullet_adapter.py
-------------------
Stage-1 PyBullet backend for the UTM swarm.

This is the "hands" that touch the simulator. It is the ONLY file that knows we
are using PyBullet. It exposes the same small interface a WebotsAdapter would,
so SwarmEnv never has to care which simulator is underneath:

    reset_world(spawn_positions, goal_positions)
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

Stage-1 simplifications (deliberate, documented):
  - Empty world, no obstacles.
  - Velocity control: the policy's velocity command is applied directly to the
    drone body (resetBaseVelocity). Stable and ideal for learning basic
    navigation; realistic motor dynamics come later / in Webots.
  - Gravity disabled -> the drone is a free-flying velocity-controlled body.
  - LiDAR returns zeros (Stage 1 has nothing to sense; the config zero-pads it).
"""

import numpy as np
import pybullet as p
import pybullet_data


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

        self.client = p.connect(p.GUI if gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, 0)                # Stage 1: free velocity control
        p.setTimeStep(self.physics_dt)
        p.setRealTimeSimulation(0)
        p.loadURDF("plane.urdf")

        self._spawn_drones()

    def _spawn_drones(self):
        """Spawn N simple velocity-controlled bodies (no external URDF needed)."""
        half = 0.15
        for i in range(self.n_agents):
            col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[half] * 3)
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[half] * 3,
                                      rgbaColor=[0.1, 0.4, 0.9, 1])
            body = p.createMultiBody(
                baseMass=1.0,
                baseCollisionShapeIndex=col,
                baseVisualShapeIndex=vis,
                basePosition=[i * 2.0, 0.0, 1.0],
            )
            self.drone_ids.append(body)

    def reset_world(self, spawn_positions: dict, goal_positions: dict):
        """spawn_positions / goal_positions: {'drone_i': [x, y, z]}."""
        for i, body in enumerate(self.drone_ids):
            key = f'drone_{i}'
            p.resetBasePositionAndOrientation(body, spawn_positions[key], [0, 0, 0, 1])
            p.resetBaseVelocity(body, [0, 0, 0], [0, 0, 0])
            self.goals[key] = np.array(goal_positions[key], dtype=np.float32)

    def apply_actions(self, actions: dict):
        """actions: {'drone_i': [vx, vy, vz, yaw_rate]} (yaw_rate optional)."""
        for i, body in enumerate(self.drone_ids):
            a = actions[f'drone_{i}']
            v = np.clip(np.asarray(a[:3], dtype=np.float32), -self.v_max, self.v_max)
            yaw_rate = float(np.clip(a[3], -self.omega_max, self.omega_max)) if len(a) > 3 else 0.0
            p.resetBaseVelocity(body, linearVelocity=v.tolist(),
                                angularVelocity=[0.0, 0.0, yaw_rate])

    def step_simulation(self):
        for _ in range(self.action_repeat):
            p.stepSimulation()

    def get_raw_state(self) -> dict:
        state = {'goals': {}}
        for i, body in enumerate(self.drone_ids):
            key = f'drone_{i}'
            pos, orn = p.getBasePositionAndOrientation(body)
            lin, _   = p.getBaseVelocity(body)
            euler    = p.getEulerFromQuaternion(orn)     # (roll, pitch, yaw)
            state[key] = {
                'position':    list(pos),
                'velocity':    list(lin),
                'orientation': list(euler),
                'lidar':       [0.0] * self.n_lidar,     # Stage 1: nothing to sense
            }
            state['goals'][key] = self.goals[key].tolist()
        return state

    def get_positions(self) -> np.ndarray:
        return np.array(
            [p.getBasePositionAndOrientation(b)[0] for b in self.drone_ids],
            dtype=np.float32,
        )

    def close(self):
        try:
            p.disconnect(self.client)
        except Exception:
            pass
