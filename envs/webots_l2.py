"""
webots_l2.py
------------
Webots Level 2: the policy flies a quadrotor with real dynamics instead of a
velocity-controlled body. Same interface as PyBulletAdapter / WebotsAdapter
(environment.simulator: webots_l2), single drone for now (the obstacle tests).

The drone (webots/worlds/swarm_l2.wbt) is a Crazyflie-equivalent: 50 g,
gravity, four Propellers with the Crazyflie's thrust and torque constants.
It is flown by Bitcraze's velocity controller pid_velocity_fixed_height
(envs/cf_velocity_pid.py, ported from Webots' crazyflie_py_wallfollowing/pid_controller.py, gains
unchanged), run every basic step (4 ms):
    policy action [vx, vy, vz] (world frame)
      -> PID setpoint: vx, vy rotated into the body frame; vz moves the
         altitude setpoint (z_des += vz * dt), the PID holds that altitude
      -> motor speeds m1..m4 -> propeller thrust and drag torque
The policy sees the TRUE position and velocity (Supervisor), like a good state
estimator; the PID uses the IMU, gyro and GPS as Bitcraze's controller does.

LiDAR: still the analytic scan of envs/webots_adapter.py (identical rules), so
the only difference from Level 1 is the flight dynamics.

Safe reset: Webots' Propeller motors keep their speed through resetPhysics and change speed at
most spin_rate rad/s^2 (the world's maxTorque with unit rotor inertia: 30 for the official model,
1000 for the equivalent; measured exact to 0.001 rad/s). A crashed drone that keeps getting climb
commands ends with its motors near 600 rad/s and would shoot up at the next release. So the
adapter tracks each rotor's speed from the commands it sends (every basic step), and reset_world
holds the drone at least spinup_hold_s and until every rotor is within 2 rad/s of hover speed
(at most 30 s, else it raises). Start check: a drone faster than 1 m/s within 5 control steps of
release raises (no command can do that).
"""

import math

import numpy as np

from envs import lidar as lidar_mod
from envs.cf_velocity_pid import VelocityPID, HOVER_RAD_S
from envs.webots_adapter import (WebotsAdapter, scan_lidar, PILLAR_NODE, BOX_NODE,
                                 ROOT, _euler_from_matrix, _note_lag_ignored)
import os

L2_WORLD = os.path.join(ROOT, 'webots', 'worlds', 'swarm_l2.wbt')
L2_REAL_WORLD = os.path.join(ROOT, 'webots', 'worlds', 'swarm_l2_real.wbt')   # official Crazyflie model
SPIN_RATE = {True: 30.0, False: 1000.0}   # rad/s^2: official model (real) / equivalent (worlds' maxTorque)
HOVER_TOL = 2.0                           # rad/s: every rotor this close to HOVER_RAD_S before release
MAX_HOLD_S = 30.0                         # longest spin-up hold before reset_world raises
START_CHECK_STEPS = 5                     # control steps after release (0.16 s)
START_SPEED_LIMIT = 1.0                   # m/s


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
        _note_lag_ignored(config)
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
        self.spin_rate = float(w.get('spin_rate', SPIN_RATE[self.real]))
        self._rotor_target = np.zeros(4)   # |commanded| rotor speeds (rad/s)
        self._rotor_est = np.zeros(4)      # estimated true rotor speeds: kept across resets, as Webots does
        self.last_hold_s = None
        self.episode = 0                   # reset_world calls so far (1 = first episode)
        self._steps_since_release = START_CHECK_STEPS
        self._step_basic()
        self._pos = np.array([self.node.getPosition()], dtype=np.float64)
        self._vel = np.zeros((1, 3))
        self._orn = np.zeros((1, 3))
        self._prev_gps = np.array(self.gps.getValues())
        self.trace = None            # set to a list to record (t, pos, vel, rpy, cmd, z_des) per basic step

    # ---------------------------------------------------------------- world
    def reset_world(self, spawn_positions: dict, goal_positions: dict, obstacles=None):
        self.episode += 1
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
        self.hold = None
        self.z_des = start[2]
        self._set_motors([HOVER_RAD_S] * 4)
        # hold the drone at the spawn while the rotors spin up to hover speed
        # (55 ms at our motors' 1000 rad/s^2, ~1.85 s for the official model), so the
        # episode starts in hover
        k_min, k_max = max(1, int(round(self.spinup_hold_s / self.dt))), int(round(MAX_HOLD_S / self.dt))
        k = 0
        while k < k_min or np.abs(self._rotor_est - HOVER_RAD_S).max() > HOVER_TOL:
            if k >= k_max:
                raise RuntimeError(f"webots_l2: rotors not at hover speed after {MAX_HOLD_S} s hold "
                                   f"(estimate {self._rotor_est.round(1)} rad/s)")
            self._step_basic()
            k += 1
            self._tr[0].setSFVec3f(start)
            self._rot[0].setSFRotation([0.0, 0.0, 1.0, 0.0])
            self.node.setVelocity([0.0] * 6)
        self.last_hold_s = k * self.dt
        self._step_basic()
        self._prev_gps = np.array(self.gps.getValues())
        self._steps_since_release = 0
        self._read_state()

    def _step_basic(self):
        """One basic step; the rotor estimate moves towards the command by at most spin_rate * dt."""
        d = np.clip(self._rotor_target - self._rotor_est, -self.spin_rate * self.dt, self.spin_rate * self.dt)
        self._rotor_est += d
        return self.sup.step(self.basic_ms)

    def _start_check(self):
        speed = float(np.linalg.norm(self._vel[0]))
        if speed > START_SPEED_LIMIT:
            raise RuntimeError(f"webots_l2: episode {self.episode}: drone faster than {START_SPEED_LIMIT} m/s "
                               f"({speed:.1f} m/s) within "
                               f"{START_CHECK_STEPS} control steps of release; it started with leftover motor speed")

    def _set_motors(self, m):
        self._rotor_target = np.abs(np.asarray(m, dtype=np.float64))
        # spin directions as in Bitcraze's controller: m1, m3 negative
        self.motors[0].setVelocity(-m[0]); self.motors[1].setVelocity(m[1])
        self.motors[2].setVelocity(-m[2]); self.motors[3].setVelocity(m[3])

    def _read_state(self):
        self._pos[0] = self.node.getPosition()
        self._vel[0] = self.node.getVelocity()[:3]
        self._orn[0] = _euler_from_matrix(self.node.getOrientation())

    # -------------------------------------------------------------- control
    def set_position_hold(self, goal=None, gain=1.0, v_cap=0.3):
        """Hand control to a position hold at `goal` (x, y, z), or back to the policy
        (goal=None). Bitcraze's velocity controller has no x/y position mode, so this is a
        simple outer loop run every basic step on the GPS position (as onboard):
        horizontal velocity setpoint = gain * (goal - position), capped at v_cap m/s;
        altitude setpoint = goal z. While holding, apply_actions is ignored."""
        self.hold = None if goal is None else (np.asarray(goal, dtype=np.float64), float(gain), float(v_cap))
        if self.hold is not None:
            self.z_des = float(self.hold[0][2])

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
            if getattr(self, 'hold', None) is not None:      # position hold (outer loop)
                goal, gain, cap = self.hold
                vxy = gain * (goal[:2] - g[:2])
                n = float(np.linalg.norm(vxy))
                if n > cap:
                    vxy *= cap / n
                cmd = np.array([vxy[0], vxy[1], 0.0])
                self.z_des = float(goal[2])
            else:
                cmd = self.cmd
                self.z_des = max(0.2, self.z_des + cmd[2] * self.dt)
            cx_b = cmd[0] * cy + cmd[1] * sy
            cy_b = -cmd[0] * sy + cmd[1] * cy
            m = self.pid(self.dt, cx_b, cy_b, 0.0, self.z_des, roll, pitch, yaw_rate, g[2], vx_b, vy_b)
            self._set_motors(m)
            if self._step_basic() == -1:
                raise RuntimeError("Webots terminated")
            if self.trace is not None:
                self._read_state()
                self.trace.append((self.sup.getTime(), self._pos[0].copy(), self._vel[0].copy(),
                                   self._orn[0].copy(), self.cmd.copy(), self.z_des))
        self._read_state()
        if self._steps_since_release < START_CHECK_STEPS:
            self._start_check()
            self._steps_since_release += 1
