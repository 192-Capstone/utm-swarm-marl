"""
webots_l2.py
------------
Webots Level 2: the policy flies a quadrotor with real dynamics instead of a
velocity-controlled body. Same interface as PyBulletAdapter / WebotsAdapter
(environment.simulator: webots_l2), single drone for now (the obstacle tests).

The drone (webots/worlds/swarm_l2.wbt) is a Crazyflie-equivalent: 50 g,
gravity, four Propellers with the Crazyflie's thrust and torque constants.
It is flown by Bitcraze's velocity controller pid_velocity_fixed_height
(ported from Webots' crazyflie_py_wallfollowing/pid_controller.py, gains
unchanged), run every basic step (4 ms):
    policy action [vx, vy, vz] (world frame)
      -> PID setpoint: vx, vy rotated into the body frame; vz moves the
         altitude setpoint (z_des += vz * dt), the PID holds that altitude
      -> motor speeds m1..m4 -> propeller thrust and drag torque
The policy sees the TRUE position and velocity (Supervisor), like a good state
estimator; the PID uses the IMU, gyro and GPS as Bitcraze's controller does.

LiDAR: still the analytic scan of envs/webots_adapter.py (identical rules), so
the only difference from Level 1 is the flight dynamics.
"""

import math

import numpy as np

from envs import lidar as lidar_mod
from envs.webots_adapter import (WebotsAdapter, scan_lidar, PILLAR_NODE, BOX_NODE,
                                 ROOT, _euler_from_matrix)
import os

L2_WORLD = os.path.join(ROOT, 'webots', 'worlds', 'swarm_l2.wbt')
L2_REAL_WORLD = os.path.join(ROOT, 'webots', 'worlds', 'swarm_l2_real.wbt')   # official Crazyflie model
HOVER_RAD_S = math.sqrt(0.05 * 9.81 / (4 * 4e-05))      # ~55.4 rad/s per motor at hover


class VelocityPID:
    """Bitcraze's pid_velocity_fixed_height_controller (Webots Crazyflie
    example, MIT licence), same gains and structure, without numpy."""
    G = {"kp_att_y": 1, "kd_att_y": 0.5, "kp_att_rp": 0.5, "kd_att_rp": 0.1,
         "kp_vel_xy": 2, "kd_vel_xy": 0.5, "kp_z": 10, "ki_z": 5, "kd_z": 5}

    def __init__(self):
        self.reset()

    def reset(self, integrator=0.0):
        self.past_vx_error = self.past_vy_error = self.past_alt_error = 0.0
        self.past_pitch_error = self.past_roll_error = 0.0
        self.altitude_integrator = integrator

    @staticmethod
    def _clip(x, lo, hi):
        return lo if x < lo else hi if x > hi else x

    def __call__(self, dt, vx_d, vy_d, yaw_rate_d, alt_d, roll, pitch, yaw_rate, alt, vx, vy):
        g, c = self.G, self._clip
        vx_e, vy_e = vx_d - vx, vy_d - vy
        vx_der, vy_der = (vx_e - self.past_vx_error) / dt, (vy_e - self.past_vy_error) / dt
        pitch_d = g["kp_vel_xy"] * c(vx_e, -1, 1) + g["kd_vel_xy"] * vx_der
        roll_d = -g["kp_vel_xy"] * c(vy_e, -1, 1) - g["kd_vel_xy"] * vy_der
        self.past_vx_error, self.past_vy_error = vx_e, vy_e
        alt_e = alt_d - alt
        alt_der = (alt_e - self.past_alt_error) / dt
        self.altitude_integrator += alt_e * dt
        alt_cmd = g["kp_z"] * alt_e + g["kd_z"] * alt_der + g["ki_z"] * c(self.altitude_integrator, -2, 2) + 48
        self.past_alt_error = alt_e
        pitch_e, roll_e = pitch_d - pitch, roll_d - roll
        pitch_der, roll_der = (pitch_e - self.past_pitch_error) / dt, (roll_e - self.past_roll_error) / dt
        roll_cmd = g["kp_att_rp"] * c(roll_e, -1, 1) + g["kd_att_rp"] * roll_der
        pitch_cmd = -g["kp_att_rp"] * c(pitch_e, -1, 1) - g["kd_att_rp"] * pitch_der
        yaw_cmd = g["kp_att_y"] * c(yaw_rate_d - yaw_rate, -1, 1)
        self.past_pitch_error, self.past_roll_error = pitch_e, roll_e
        m1 = alt_cmd - roll_cmd + pitch_cmd + yaw_cmd
        m2 = alt_cmd - roll_cmd - pitch_cmd - yaw_cmd
        m3 = alt_cmd + roll_cmd - pitch_cmd + yaw_cmd
        m4 = alt_cmd + roll_cmd + pitch_cmd - yaw_cmd
        return [c(m, 0, 600) for m in (m1, m2, m3, m4)]


class WebotsL2Adapter(WebotsAdapter):
    def __init__(self, config, gui: bool = False):
        if config['environment']['n_agents'] != 1:
            raise ValueError("Webots Level 2 flies one drone for now (obstacle tests)")
        config = dict(config)
        w = dict(config.get('webots') or {})
        self.real = bool(w.get('real', False))
        w['world'] = w.get('l2_world', L2_REAL_WORLD if self.real else L2_WORLD)
        # the official model's motors spin up at 30 rad/s^2 (1.8 s to hover speed); ours at 1000
        self.spinup_hold_s = float(w.get('spinup_hold_s', 2.0 if self.real else 0.12))
        config['webots'] = w
        self.config = config
        self.n_agents = 1
        self.v_max = config['environment'].get('v_max', 2.0)
        self.omega_max = config['environment'].get('omega_max', 1.0)
        self.n_lidar = config['observation'].get('n_lidar_rays', 14)
        self.control_dt = config['environment'].get('control_timestep', 0.032)
        self.world = config['webots']['world']
        self.match_pybullet = False
        self._vel_report = None
        self.goals, self.layout, self._heading = {}, [], {}
        self.lidar_cfg = lidar_mod.lidar_config(config)
        self._start_webots(gui, config['webots'])
        self.basic_ms = int(round(self.sup.getBasicTimeStep()))
        self.dt = self.basic_ms / 1000.0
        self.step_ms = int(round(self.control_dt * 1000))
        self.n_sub = self.step_ms // self.basic_ms
        self._obstacle_root = self.sup.getFromDef('OBSTACLES').getField('children')
        self.node = self.sup.getSelf()
        self.nodes = [self.node]
        self._tr = [self.node.getField('translation')]
        self._rot = [self.node.getField('rotation')]
        self.drone_ids = [self.node.getId()]
        self.motors = []
        for k in range(1, 5):
            m = self.sup.getDevice(f'm{k}_motor')
            m.setPosition(float('inf'))
            m.setVelocity(0.0)
            self.motors.append(m)
        self.imu = self.sup.getDevice('inertial_unit'); self.imu.enable(self.basic_ms)
        self.gps = self.sup.getDevice('gps'); self.gps.enable(self.basic_ms)
        self.gyro = self.sup.getDevice('gyro'); self.gyro.enable(self.basic_ms)
        self.pid = VelocityPID()
        self.cmd = np.zeros(3)
        self.z_des = 1.0
        self.sup.step(self.basic_ms)
        self._pos = np.array([self.node.getPosition()], dtype=np.float64)
        self._vel = np.zeros((1, 3))
        self._orn = np.zeros((1, 3))
        self._prev_gps = np.array(self.gps.getValues())
        self.trace = None            # set to a list to record (t, pos, vel, rpy, cmd, z_des) per basic step

    # ---------------------------------------------------------------- world
    def reset_world(self, spawn_positions: dict, goal_positions: dict, obstacles=None):
        start = [float(v) for v in spawn_positions['drone_0']]
        self._tr[0].setSFVec3f(start)
        self._rot[0].setSFRotation([0.0, 0.0, 1.0, 0.0])
        self.node.resetPhysics()
        self.goals['drone_0'] = np.array(goal_positions['drone_0'], dtype=np.float32)
        self._heading = {}
        self._build_obstacles(obstacles or [])
        # start at hover: motors at hover speed, altitude integrator preloaded,
        # setpoint = current altitude, zero velocity command
        self.pid.reset(integrator=(HOVER_RAD_S - 48.0) / VelocityPID.G["ki_z"])
        self.cmd[:] = 0.0
        self.z_des = start[2]
        self._set_motors([HOVER_RAD_S] * 4)
        # hold the drone at the spawn while the rotors spin up to hover speed
        # (55 ms at our motors' 1000 rad/s^2, ~1.85 s for the official model), so the
        # episode starts in hover
        for _ in range(max(1, int(round(self.spinup_hold_s / self.dt)))):
            self.sup.step(self.basic_ms)
            self._tr[0].setSFVec3f(start)
            self._rot[0].setSFRotation([0.0, 0.0, 1.0, 0.0])
            self.node.setVelocity([0.0] * 6)
        self.sup.step(self.basic_ms)
        self._prev_gps = np.array(self.gps.getValues())
        self._read_state()

    def _set_motors(self, m):
        # spin directions as in Bitcraze's controller: m1, m3 negative
        self.motors[0].setVelocity(-m[0]); self.motors[1].setVelocity(m[1])
        self.motors[2].setVelocity(-m[2]); self.motors[3].setVelocity(m[3])

    def _read_state(self):
        self._pos[0] = self.node.getPosition()
        self._vel[0] = self.node.getVelocity()[:3]
        self._orn[0] = _euler_from_matrix(self.node.getOrientation())

    # -------------------------------------------------------------- control
    def apply_actions(self, actions: dict):
        a = actions['drone_0']
        self.cmd = np.clip(np.asarray(a[:3], dtype=np.float64), -self.v_max, self.v_max)

    def step_simulation(self):
        for _ in range(self.n_sub):
            roll, pitch, yaw = self.imu.getRollPitchYaw()
            yaw_rate = self.gyro.getValues()[2]
            g = np.array(self.gps.getValues())
            v_world = (g - self._prev_gps) / self.dt          # as Bitcraze: velocity from GPS differences
            self._prev_gps = g
            cy, sy = math.cos(yaw), math.sin(yaw)
            vx_b = v_world[0] * cy + v_world[1] * sy
            vy_b = -v_world[0] * sy + v_world[1] * cy
            cx_b = self.cmd[0] * cy + self.cmd[1] * sy
            cy_b = -self.cmd[0] * sy + self.cmd[1] * cy
            self.z_des = max(0.2, self.z_des + self.cmd[2] * self.dt)
            m = self.pid(self.dt, cx_b, cy_b, 0.0, self.z_des, roll, pitch, yaw_rate, g[2], vx_b, vy_b)
            self._set_motors(m)
            if self.sup.step(self.basic_ms) == -1:
                raise RuntimeError("Webots terminated")
            if self.trace is not None:
                self._read_state()
                self.trace.append((self.sup.getTime(), self._pos[0].copy(), self._vel[0].copy(),
                                   self._orn[0].copy(), self.cmd.copy(), self.z_des))
        self._read_state()
