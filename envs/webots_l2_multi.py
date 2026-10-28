"""
webots_l2_multi.py
------------------
Webots Level 2 with several drones (environment.simulator: webots_l2_multi): n official
Crazyflies, each flown by its OWN Webots controller (webots/controllers/cf_velocity), which
runs Bitcraze's velocity PID onboard every basic step (4 ms), as WebotsL2Adapter does for one
drone. Same interface as PyBulletAdapter / WebotsAdapter / WebotsL2Adapter.

    world       webots/worlds/swarm_l2_multi.wbt: bodiless Supervisor (this process, <extern>),
                the drones are imported at start-up with controller "cf_velocity"
    commands    once per control step the supervisor writes "<ep> <mode> vx vy vz" into each
                drone's customData (envs/cf_velocity_pid.py). H = held during spin-up, F = flying.
                A command reaches the motors one basic step (4 ms) later than in WebotsL2Adapter.
    state       the TRUE position, velocity and orientation of every drone (Supervisor),
                so neighbour observations come from SwarmEnv / ObservationProcessor unchanged
    LiDAR       the analytic scan of envs/webots_adapter.py, as Level 1 / Level 2
    yaw rate    always 0 (as WebotsL2Adapter)

Safety checks (all raise RuntimeError naming the drone):
    drop check  in the first DROP_CHECK_STEPS control steps after release, a drone more than
                DROP_LIMIT below the LOWER of its spawn height and the altitude setpoint its
                controller should hold: the controller is not running (motors off, the drone
                falls). The lower of the two, so a healthy drone lagging behind a climb command
                is not a false alarm.
    log check   at every reset (the episode before) and in close() (the last episode): a
                "cf_velocity [" error line in the Webots log
    start check in the first START_CHECK_STEPS control steps after release, a drone faster than
                START_SPEED_LIMIT: no command can do that (the drone needs ~0.2 s to respond), so
                it started with leftover motor speed
Re-import: at every reset, a drone that lost control (tilt > BROKEN_TILT_DEG) or ended below
BROKEN_Z in the episode before is removed and imported again (its controller restarts). Such a
drone's motors can still be near full speed: a crashed drone keeps getting climb commands, its
altitude setpoint runs away and the motors saturate (600 rad/s); Webots' Propeller motors keep
that speed through resetPhysics and the official model spins them down at only 30 rad/s^2, so
it would shoot up at the next release (seen in crowd crossing, seed 22, episodes 6-8).
The Webots log (Webots + all drone controllers) is always written: webots.log in the config,
else logs/webots/l2_multi_<time>_<pid>.log.

Scheduled departures (SwarmEnv environment.schedule, E1/E4 Stage 3): park(indices) holds drones
far out of the world (still, motors at hover speed, no PID: the spin-up hold), so they are out of
every sensor's range; place(i, pos) puts one on its pad and releases it there. Each launch gives
that drone's controller a new command token, which it treats as a new episode: PID reset, and its
first F step sets the altitude setpoint and the previous GPS reading from the pad (with the old
token it would keep the parking spot's). The motors are already at hover, so no new 2 s spin-up;
the drop and start checks run for that drone after its own release.

Per episode and drone, episode_stats() gives max |roll|, max |pitch| (deg) and max true speed
(m/s) over the control steps, for the lost-control rule of scripts/webots_part4_table.py.
"""

import os
import time

import numpy as np

from envs import lidar as lidar_mod
from envs.cf_velocity_pid import format_command
from envs.webots_adapter import WebotsAdapter, ROOT, _euler_from_matrix, _note_lag_ignored

L2_MULTI_WORLD = os.path.join(ROOT, 'webots', 'worlds', 'swarm_l2_multi.wbt')
DRONE_NODE = """DEF DRONE_{i} Crazyflie {{
  translation {x} 0 0.015
  name "drone_{i}"
  controller "cf_velocity"
}}"""
DROP_CHECK_STEPS = 16          # control steps after release (0.51 s)
DROP_LIMIT = 0.3               # m; free fall covers this in 0.25 s
CF_ERROR = 'cf_velocity ['     # prefix of the controllers' error lines
START_CHECK_STEPS = 5          # control steps after release (0.16 s)
START_SPEED_LIMIT = 1.0        # m/s; the Crazyflie needs ~0.2 s to respond to any command
BROKEN_TILT_DEG = 60.0         # a drone that tilted past this in the last episode ...
BROKEN_Z = 0.1                 # ... or ended below this height (m) is re-imported at the next reset
PARK = (500.0, 500.0, 20.0)    # m: parked drone i is held at PARK + (5 i, 0, 0), far out of the world
TOKENS_PER_EP = 1000           # command tokens: episode e uses e * 1000, each launch the next one


class WebotsL2MultiAdapter(WebotsAdapter):
    def __init__(self, config, gui: bool = False):
        config = dict(config)
        w = dict(config.get('webots') or {})
        self.real = True                                    # official Crazyflies, buildings
        w['world'] = w.get('l2_multi_world', L2_MULTI_WORLD)
        self.spinup_hold_s = float(w.get('spinup_hold_s', 2.0))
        if not w.get('log'):
            w['log'] = os.path.join(ROOT, 'logs', 'webots',
                                    f"l2_multi_{time.strftime('%Y%m%d-%H%M%S')}_{os.getpid()}.log")
        os.makedirs(os.path.dirname(os.path.abspath(w['log'])), exist_ok=True)
        self.log_path = w['log']
        self._log_pos = 0
        config['webots'] = w
        self.config = config
        self.n_agents = config['environment']['n_agents']
        self.v_max = config['environment'].get('v_max', 2.0)
        self.omega_max = config['environment'].get('omega_max', 1.0)
        self.n_lidar = config['observation'].get('n_lidar_rays', 14)
        self.control_dt = config['environment'].get('control_timestep', 0.032)
        self.world = w['world']
        self.match_pybullet = False
        self._vel_report = None
        self.goals, self.layout, self._heading = {}, [], {}
        self.lidar_cfg = lidar_mod.lidar_config(config)
        _note_lag_ignored(config)

        self._start_webots(gui, w)                          # inherited: own port, writes w['log']
        self.basic_ms = int(round(self.sup.getBasicTimeStep()))
        self.dt = self.basic_ms / 1000.0
        self.step_ms = int(round(self.control_dt * 1000))
        if self.step_ms % self.basic_ms:
            raise ValueError(f"control_timestep {self.control_dt}s is not a multiple of "
                             f"basicTimeStep {self.basic_ms} ms")
        self.n_sub = self.step_ms // self.basic_ms
        self._obstacle_root = self.sup.getFromDef('OBSTACLES').getField('children')
        self._drones_root = self.sup.getFromDef('DRONES').getField('children')
        n = self.n_agents
        self.nodes, self._tr, self._rot, self._cd, self.drone_ids = [None] * n, [None] * n, [None] * n, [None] * n, [None] * n
        for i in range(n):                                  # on the ground, 2 m apart, motors off
            self._drones_root.importMFNodeFromString(-1, DRONE_NODE.format(i=i, x=2.0 * i))
            self._attach(i)
        self.reimport_log = []                              # (episode about to start, [drones re-imported])
        self._reimported_now = []                           # drones re-imported before this episode
        self.ep = 0
        self.sup.step(self.basic_ms)                        # let the imports take effect
        self._pos = np.array([nd.getPosition() for nd in self.nodes], dtype=np.float64)
        self._vel, self._orn = np.zeros((n, 3)), np.zeros((n, 3))
        self._cmd = np.zeros((n, 3))
        self._z_ref = np.zeros(n)                           # the controllers' altitude setpoints (drop check)
        self._z_spawn = np.zeros(n)                         # spawn heights (drop check)
        self._since = np.full(n, DROP_CHECK_STEPS)        # control steps since each drone's release
        self._parked = np.zeros(n, dtype=bool)
        self._token = np.zeros(n, dtype=np.int64)           # each drone's command token (its controller's episode)
        self._reset_stats()

    # ---------------------------------------------------------------- world
    def _attach(self, i):
        nd = self.sup.getFromDef(f'DRONE_{i}')
        self.nodes[i], self._tr[i], self._rot[i] = nd, nd.getField('translation'), nd.getField('rotation')
        self._cd[i], self.drone_ids[i] = nd.getField('customData'), nd.getId()

    def _broken_drones(self):
        """Drones that lost control or ended on the ground in the episode that just ended."""
        tilt = np.degrees(np.maximum(self._max_roll, self._max_pitch))
        return [i for i in range(self.n_agents) if tilt[i] > BROKEN_TILT_DEG or self._pos[i, 2] < BROKEN_Z]

    def _reimport(self, i):
        """Remove drone i (its controller is terminated) and import a fresh one."""
        self.nodes[i].remove()
        self.sup.step(self.basic_ms)
        self._drones_root.importMFNodeFromString(-1, DRONE_NODE.format(i=i, x=2.0 * i))
        self._attach(i)

    def reset_world(self, spawn_positions: dict, goal_positions: dict, obstacles=None):
        if self.ep > 0:
            self._check_controller_log()                    # the episode that just ended
            broken = self._broken_drones()
            for i in broken:
                self._reimport(i)
            if broken:
                self.sup.step(self.basic_ms)                # let the imports take effect
            self.reimport_log.append((self.ep + 1, broken))
            self._reimported_now = broken
        self.ep += 1
        starts = [[float(v) for v in spawn_positions[f'drone_{i}']] for i in range(self.n_agents)]
        self._token[:] = self.ep * TOKENS_PER_EP
        self._parked[:] = False
        for i, nd in enumerate(self.nodes):                 # L2 94-97
            self._tr[i].setSFVec3f(starts[i])
            self._rot[i].setSFRotation([0.0, 0.0, 1.0, 0.0])
            nd.resetPhysics()
            self._cd[i].setSFString(self._command(i, 'H'))  # L2 102-106 (in the controller): PID reset, hover motors
            self.goals[f'drone_{i}'] = np.array(goal_positions[f'drone_{i}'], dtype=np.float32)
        self._heading = {}                                  # L2 98-99
        self._build_obstacles(obstacles or [])
        self._cmd[:] = 0.0                                  # L2 103
        # L2 110-114: hold every drone at its spawn while the rotors spin up (2 s, official model)
        for _ in range(max(1, int(round(self.spinup_hold_s / self.dt)))):
            self.sup.step(self.basic_ms)
            for i, nd in enumerate(self.nodes):
                self._tr[i].setSFVec3f(starts[i])
                self._rot[i].setSFRotation([0.0, 0.0, 1.0, 0.0])
                nd.setVelocity([0.0] * 6)
        for i in range(self.n_agents):                      # first F: controller sets z_des, prev_gps
            self._cd[i].setSFString(self._command(i, 'F'))
        self.sup.step(self.basic_ms)                        # L2 115
        self._z_spawn[:] = [s[2] for s in starts]
        self._z_ref[:] = self._z_spawn                      # L2 105
        self._since[:] = 0
        self._reset_stats()
        self._read_state()                                  # L2 117

    def _command(self, i, mode, v=(0.0, 0.0, 0.0)):
        return format_command(int(self._token[i]), mode, v)

    # ------------------------------------------------------------ schedule
    def _hold(self, i, pos):
        self._tr[i].setSFVec3f([float(v) for v in pos])
        self._rot[i].setSFRotation([0.0, 0.0, 1.0, 0.0])
        self.nodes[i].setVelocity([0.0] * 6)

    def park(self, indices):
        """Hold these drones far out of the world, still, motors at hover speed (SwarmEnv schedule:
        not departed yet, or landed; crash_mode remove). Called every control step."""
        for i in indices:
            self._parked[i] = True
            self._cmd[i] = 0.0
            self._cd[i].setSFString(self._command(i, 'H'))
            self._hold(i, (PARK[0] + 5.0 * i, PARK[1], PARK[2]))

    def place(self, i, pos):
        """Put parked drone i on its pad at pos and release it there (SwarmEnv schedule: take-off).
        A new token makes its controller start as in a new episode; the next apply_actions gives it
        its first F step."""
        self._parked[i] = False
        self._token[i] += 1
        self._cmd[i] = 0.0
        self._cd[i].setSFString(self._command(i, 'H'))      # PID reset, hover motors
        self._hold(i, pos)
        self._pos[i] = [float(v) for v in pos]
        self._z_spawn[i] = self._z_ref[i] = float(pos[2])
        self._since[i] = 0

    # -------------------------------------------------------------- control
    def apply_actions(self, actions: dict):
        """actions: {'drone_i': [vx, vy, vz, yaw_rate]}; yaw rate ignored (always 0, as Level 2).
        Parked drones stay on hold."""
        for i in range(self.n_agents):                      # L2 140-142, per axis as PyBullet
            if self._parked[i]:
                continue
            v = np.clip(np.asarray(actions[f'drone_{i}'][:3], dtype=np.float64), -self.v_max, self.v_max)
            self._cmd[i] = v
            self._cd[i].setSFString(self._command(i, 'F', v))

    def step_simulation(self):
        if self.sup.step(self.step_ms) == -1:               # the controllers run n_sub basic steps
            raise RuntimeError("Webots terminated")
        self._read_state()
        new = (self._since < DROP_CHECK_STEPS) & ~self._parked     # drones just released
        if new.any():
            start = new & (self._since < START_CHECK_STEPS)
            if start.any():
                self._start_check(start)
            self._drop_check(new)
            self._since[new] += 1
        self._max_roll = np.maximum(self._max_roll, np.abs(self._orn[:, 0]))
        self._max_pitch = np.maximum(self._max_pitch, np.abs(self._orn[:, 1]))
        self._max_speed = np.maximum(self._max_speed, np.linalg.norm(self._vel, axis=1))

    def _read_state(self):                                  # L2 124-127, every drone
        for i, nd in enumerate(self.nodes):
            self._pos[i] = nd.getPosition()
            self._vel[i] = nd.getVelocity()[:3]
            self._orn[i] = _euler_from_matrix(nd.getOrientation())

    def _drop_check(self, which=None):
        """which: boolean mask of the drones to check (default all): those just released."""
        which = np.ones(self.n_agents, dtype=bool) if which is None else which
        # the setpoint each controller should now hold: z_des += vz * dt per basic step, floor 0.2
        for _ in range(self.n_sub):
            self._z_ref = np.where(which, np.maximum(0.2, self._z_ref + self._cmd[:, 2] * self.dt), self._z_ref)
        drop = np.minimum(self._z_ref, self._z_spawn) - self._pos[:, 2]
        bad = np.flatnonzero((drop > DROP_LIMIT) & which)
        if bad.size:
            names = ', '.join(f'drone_{i} ({drop[i]:.2f} m)' for i in bad)
            raise RuntimeError(f"webots_l2_multi: {names} fell below its spawn height and altitude "
                               f"setpoint right after spin-up; its cf_velocity controller is probably "
                               f"not running. See the Webots log: {self.log_path}")

    def _start_check(self, which=None):
        which = np.ones(self.n_agents, dtype=bool) if which is None else which
        speed = np.linalg.norm(self._vel, axis=1)
        bad = np.flatnonzero((speed > START_SPEED_LIMIT) & which)
        if bad.size:
            names = ', '.join(f'drone_{i} ({speed[i]:.1f} m/s)' for i in bad)
            raise RuntimeError(f"webots_l2_multi: {names} faster than {START_SPEED_LIMIT} m/s within "
                               f"{START_CHECK_STEPS} control steps of release; no command can do that, "
                               f"so it started with leftover motor speed. See the Webots log: {self.log_path}")

    def _check_controller_log(self):
        """Raise if the drone controllers wrote an error line since the last check."""
        try:
            with open(self.log_path, 'rb') as f:
                f.seek(self._log_pos)
                new = f.read().decode('utf-8', errors='replace')
                self._log_pos = f.tell()
        except OSError:
            return
        bad = [ln.strip() for ln in new.splitlines() if CF_ERROR in ln]
        if bad:
            raise RuntimeError(f"webots_l2_multi: drone controller error in episode {self.ep}: "
                               f"{bad[0]} ({len(bad)} error line(s); see {self.log_path})")

    # ------------------------------------------------------------- stats
    def _reset_stats(self):
        n = self.n_agents
        self._max_roll, self._max_pitch, self._max_speed = np.zeros(n), np.zeros(n), np.zeros(n)

    def episode_stats(self) -> dict:
        """Per drone, over this episode's control steps: max |roll|, max |pitch| (deg),
        max true speed (m/s), and whether it was re-imported before this episode."""
        return {'max_roll_deg': np.degrees(self._max_roll).tolist(),
                'max_pitch_deg': np.degrees(self._max_pitch).tolist(),
                'max_speed': self._max_speed.tolist(),
                'reimported': [i in self._reimported_now for i in range(self.n_agents)]}

    def close(self):
        if getattr(self, 'proc', None) is None:
            return
        super().close()                                     # quit Webots first: the log is then complete
        self._check_controller_log()                        # the last episode
