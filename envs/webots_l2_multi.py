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

Per episode and drone, episode_stats() gives max |roll|, max |pitch| (deg) and max true speed
(m/s) over the control steps, for the lost-control rule of scripts/webots_part4_table.py.

Park / place (SwarmEnv crash_mode remove and schedule, as PyBulletAdapter.park / place):
    park(i)     drone i goes to its own spot PARK_X0 + PARK_DX * i, PARK_Y, PARK_Z (far beyond
                every sensor's range, 100 m apart), held in H (hover motor speed, no PID) and
                pinned there (position, level rotation, zero velocity) at every call; SwarmEnv
                calls park for every parked drone every step. apply_actions keeps it in H.
    place(i, p) drone i is teleported onto p (resetPhysics resets velocities, not rotor speeds:
                a parked drone's rotors are at hover speed). It stays in H for the first basic
                step of the next control step, so the teleport has landed and the GPS reads the
                pad, then gets F: its controller arms afresh (ControllerState, envs/
                cf_velocity_pid.py) and it departs 4 ms into that control step. The other
                drones are not affected. Start and drop checks run for it as after a reset.
Both update the cached state at once (SwarmEnv reads positions right after).

Buildings are only rebuilt when the layout changes (webots.reuse_layout, default true): the
same layout (E-City crop, both conditions of an E1 trial) keeps its nodes. They are visual
only, so this changes nothing in the flight.

E2 radio (webots.radio, off by default; envs/radio_msg.py):
    webots.radio: {enabled: false, period_s: 1.0, noise_m: 0.0, seed: 20, range_m: 200.0}
With it on, every drone is imported with an Emitter (range range_m) and a Receiver in its
extensionSlot and "--radio" controllerArgs: it reports its own GPS position, velocity and
intent once per period_s while flying (never while held or parked). The supervisor's Receiver
"utm_rx" (in the world file, inert when off) is emptied every control step; reports() returns
what arrived since the last call as utm/telemetry.py Reports (id drone_i, times in EPISODE time:
seconds since the release, as SwarmEnv counts them; reports from another episode are dropped).
send_intent(i, intent) sends drone i its intent on the shared downlink. range_m is measured from
the supervisor (the origin): parked drones (>= 1.4 km away) are out of range as well as silent.
The state SwarmEnv sees is still the TRUE state; the radio is what the UTM hears
(utm/telemetry.py RadioTelemetry). With it off, the drone nodes, their commands and the
supervisor's devices are exactly as before.

No-fly zones (C7, off by default): with webots.draw_zones the D2 zones (utm/airspace_zones.py) are
drawn too, visual only (envs/webots_zones.py): from webots.zones_file + webots.zones_map (GeoJSON,
wgs84 or local_m) and/or set_zones(zones) (local metres, used from the next reset). Top for zones
without a ceiling: webots.zones_ceiling_m (default E1's airspace top). Zones with active windows
are shown only while active, on the episode clock (control steps since the reset * control_dt,
SwarmEnv's clock for E1 departures). Rebuilt only when the zones change (as the buildings, with
reuse_layout). With the flag off nothing is imported and the world is exactly as without C7.
"""

import ctypes
import json
import os
import time

import numpy as np

from envs import lidar as lidar_mod
from envs.cf_velocity_pid import format_command
from envs.radio_msg import (radio_config, controller_args, decode_report, encode_intent,
                            UPLINK, DOWNLINK)
from envs.webots_adapter import WebotsAdapter, ROOT, _euler_from_matrix, _note_lag_ignored
from envs.lidar_msg import format_tag, decode_scan
from envs.webots_lidar import (lidar_webots_config, device_nodes, controller_args as lidar_controller_args,
                               pick_rays, analytic_dirs, ScanLog)
from utm.telemetry import Report

L2_MULTI_WORLD = os.path.join(ROOT, 'webots', 'worlds', 'swarm_l2_multi.wbt')
DRONE_NODE = """DEF DRONE_{i} Crazyflie {{
  translation {x} 0 0.015
  name "drone_{i}"
  controller "cf_velocity"
}}"""
DRONE_RADIO_NODE = """DEF DRONE_{i} Crazyflie {{
  translation {x} 0 0.015
  name "drone_{i}"
  controller "cf_velocity"
  controllerArgs [ {args} ]
  extensionSlot [
    Emitter {{
      name "radio_tx"
      channel {up}
      range {range}
    }}
    Receiver {{
      name "radio_rx"
      channel {down}
    }}
  ]
}}"""
# C3 Lidar on (webots.lidar, envs/webots_lidar.py): the radio's devices (if on) and the Lidar's
DRONE_EXT_NODE = """DEF DRONE_{i} Crazyflie {{
  translation {x} 0 0.015
  name "drone_{i}"
  controller "cf_velocity"
  controllerArgs [ {args} ]
  extensionSlot [
    {slot}
  ]
}}"""
DROP_CHECK_STEPS = 16          # control steps after release (0.51 s)
DROP_LIMIT = 0.3               # m; free fall covers this in 0.25 s
CF_ERROR = 'cf_velocity ['     # prefix of the controllers' error lines
START_CHECK_STEPS = 5          # control steps after release (0.16 s)
START_SPEED_LIMIT = 1.0        # m/s; the Crazyflie needs ~0.2 s to respond to any command
BROKEN_TILT_DEG = 60.0         # a drone that tilted past this in the last episode ...
BROKEN_Z = 0.1                 # ... or ended below this height (m) is re-imported at the next reset
PARK_X0, PARK_DX, PARK_Y, PARK_Z = 1000.0, 100.0, 1000.0, 5.0     # parking spots (m)
LEVEL = [0.0, 0.0, 1.0, 0.0]


def park_spot(i):
    return [PARK_X0 + PARK_DX * i, PARK_Y, PARK_Z]


def drone_node(i, rc=None, lc=None, max_range=6.0):
    """The node string of drone i (on the ground at x = 2 i); with the radio on (rc: an enabled
    radio_config), with its radio devices and controllerArgs; with the C3 Lidar on (lc: an enabled
    lidar_webots_config), with the Lidar and its Emitter (range max_range) as well."""
    radio = rc is not None and rc['enabled']
    if lc is None or not lc['enabled']:
        if not radio:
            return DRONE_NODE.format(i=i, x=2.0 * i)
        args = ' '.join(f'"{a}"' for a in controller_args(rc))
        return DRONE_RADIO_NODE.format(i=i, x=2.0 * i, args=args, up=UPLINK, down=DOWNLINK, range=rc['range_m'])
    args = (controller_args(rc) if radio else []) + lidar_controller_args()
    slot = ([f'Emitter {{\n      name "radio_tx"\n      channel {UPLINK}\n      range {rc["range_m"]}\n    }}',
             f'Receiver {{\n      name "radio_rx"\n      channel {DOWNLINK}\n    }}'] if radio else [])
    slot += device_nodes(lc, max_range)
    return DRONE_EXT_NODE.format(i=i, x=2.0 * i, args=' '.join(f'"{a}"' for a in args), slot='\n    '.join(slot))


def layout_key(layout):
    """Exact, hashable form of a layout (what the building nodes are made from)."""
    return tuple((ob['type'], tuple(float(v) for v in ob['center']),
                  tuple(float(v) for v in ob.get('half', ())), float(ob.get('radius', 0.0)),
                  float(ob['height']), float(ob.get('yaw', 0.0))) for ob in layout)


class WebotsL2MultiAdapter(WebotsAdapter):
    _rx = _tx = None                                        # radio devices: None = radio off
    _zones = None                                           # C7 ZoneDrawer, only with webots.draw_zones
    _lrx = None                                             # C3 Lidar receiver: None = Lidar off
    _scan_log = None                                        # C3 ScanLog, only with webots.lidar

    def __init__(self, config, gui: bool = False):
        config = dict(config)
        w = dict(config.get('webots') or {})
        self.real = True                                    # official Crazyflies, buildings
        w['world'] = w.get('l2_multi_world', L2_MULTI_WORLD)
        self.spinup_hold_s = float(w.get('spinup_hold_s', 2.0))
        self.reuse_layout = bool(w.get('reuse_layout', True))
        self._built_key = None                              # layout_key of the buildings in the world
        self.layout_builds = 0                              # how often buildings were (re)built
        self.layout_build_s = 0.0                           # wall seconds spent building them
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
        self.radio = radio_config(w)                        # E2 radio (off by default)
        self.lidar = lidar_webots_config(w)                 # C3 Lidar (off by default)
        if self.lidar['enabled'] and not self.lidar_cfg['enabled']:
            raise ValueError('webots.lidar needs lidar.enabled (the analytic scan it is compared with)')

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
            self._drones_root.importMFNodeFromString(-1, drone_node(i, self.radio, self.lidar, self.lidar_cfg['range']))
            self._attach(i)
        self._rx = self._tx = None
        self._t0 = 0.0                                      # Webots time of the release (episode t = 0)
        self._inbox = []                                    # radio reports not yet returned by reports()
        self.radio_rejected = 0                             # reports from another episode, dropped
        if self.radio['enabled']:
            self._rx = self.sup.getDevice('utm_rx')
            self._rx.enable(self.basic_ms)
            self._tx = self.sup.getDevice('utm_tx')
            self._tx.setRange(self.radio['range_m'])
        self._scan_k = 0                                    # C3: scan tag of the current control step
        self._scans = [None] * n                            # C3: the scans received for it
        self._scan_log, self._logged = None, None
        if self.lidar['enabled']:
            self._lrx = self.sup.getDevice('lidar_rx')
            self._lrx.enable(self.basic_ms)
            self._scan_log = ScanLog(self.lidar['log'])
        self.reimport_log = []                              # (episode about to start, [drones re-imported])
        self._reimported_now = []                           # drones re-imported before this episode
        self.ep = 0
        self.sup.step(self.basic_ms)                        # let the imports take effect
        self._pos = np.array([nd.getPosition() for nd in self.nodes], dtype=np.float64)
        self._vel, self._orn = np.zeros((n, 3)), np.zeros((n, 3))
        self._cmd = np.zeros((n, 3))
        self._z_ref = np.zeros(n)                           # the controllers' altitude setpoints (drop check)
        self._z_spawn = np.zeros(n)                         # spawn heights (drop check)
        self._since_release = np.full(n, DROP_CHECK_STEPS)  # control steps since release / place, per drone
        self._parked = np.zeros(n, dtype=bool)
        self._pending = set()                               # placed, F not yet sent
        self._reset_stats()
        self._zones, self._zone_list = None, []             # C7 no-fly zones (webots.draw_zones)
        if w.get('draw_zones'):
            from envs.webots_zones import ZoneDrawer, zones_from_config
            self._zone_list = zones_from_config(w)
            self._ep_step = 0
            self._zones = ZoneDrawer(self.sup, self.sup.getFromDef('VIEW'), w.get('zones_ceiling_m'),
                                     self.reuse_layout)

    # ---------------------------------------------------------------- world
    def set_zones(self, zones):
        """D2 zones (utm/airspace_zones.Zone, local metres) to draw from the next reset on;
        replaces the config's zones. Does nothing visible unless webots.draw_zones is on."""
        self._zone_list = list(zones)

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
        self._drones_root.importMFNodeFromString(-1, drone_node(i, self.radio, self.lidar, self.lidar_cfg['range']))
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
        hold = format_command(self.ep, 'H', (0.0, 0.0, 0.0))
        for i, nd in enumerate(self.nodes):                 # L2 94-97
            self._tr[i].setSFVec3f(starts[i])
            self._rot[i].setSFRotation([0.0, 0.0, 1.0, 0.0])
            nd.resetPhysics()
            self._cd[i].setSFString(hold)                   # L2 102-106 (in the controller): PID reset, hover motors
            self.goals[f'drone_{i}'] = np.array(goal_positions[f'drone_{i}'], dtype=np.float32)
        self._heading = {}                                  # L2 98-99
        self._build_obstacles(obstacles or [])
        if self._zones is not None:                         # C7: visual only
            self._ep_step = 0
            self._zones.build(self._zone_list)
        if self._scan_log is not None:
            self._scan_log.layouts[self.ep] = json.loads(json.dumps(list(obstacles or []), default=float))
        self._cmd[:] = 0.0                                  # L2 103
        # L2 110-114: hold every drone at its spawn while the rotors spin up (2 s, official model)
        n_hold = max(1, int(round(self.spinup_hold_s / self.dt)))
        tag, tag_at = '', -1
        if self._lrx is not None:                           # C3: scan in the last basic step before the read
            self._scan_k += 1
            n_tag = min(self.n_sub, n_hold + 1)
            tag, tag_at = ' ' + format_tag(self._scan_k, n_tag), n_hold + 1 - n_tag
        for h in range(n_hold):
            if h == tag_at:
                for i in range(self.n_agents):
                    self._cd[i].setSFString(hold + tag)
            self.sup.step(self.basic_ms)
            for i, nd in enumerate(self.nodes):
                self._tr[i].setSFVec3f(starts[i])
                self._rot[i].setSFRotation([0.0, 0.0, 1.0, 0.0])
                nd.setVelocity([0.0] * 6)
        release = format_command(self.ep, 'F', (0.0, 0.0, 0.0)) + tag
        for i in range(self.n_agents):                      # first F: controller sets z_des, prev_gps
            self._cd[i].setSFString(release)
        self.sup.step(self.basic_ms)                        # L2 115
        self._z_spawn[:] = [s[2] for s in starts]
        self._z_ref[:] = self._z_spawn                      # L2 105
        self._since_release[:] = 0
        self._parked[:] = False
        self._pending.clear()
        self._reset_stats()
        self._read_state()                                  # L2 117
        if self._lrx is not None:
            self._collect_scans()
        if self._rx is not None:                            # radio: a fresh inbox, episode time from now
            self._inbox = []                                # (the last episode's: not collected, dropped)
            self._t0 = self.sup.getTime()
            self._drain()                                   # the old episode's are rejected, the release reports kept

    def _build_obstacles(self, layout):
        """Rebuild the buildings only when the layout changed (reuse_layout)."""
        key = layout_key(layout)
        if self.reuse_layout and key == self._built_key:
            self.layout = list(layout)
            return
        t0 = time.time()
        super()._build_obstacles(layout)
        self.layout_build_s += time.time() - t0
        self.layout_builds += 1
        self._built_key = key

    # ---------------------------------------------------------- park / place
    def park(self, indices):
        """Hold these drones in H at their own far-away spots (SwarmEnv crash_mode remove,
        schedule). Called for every parked drone every step: re-pins them each time."""
        hold = format_command(self.ep, 'H', (0.0, 0.0, 0.0))
        for i in indices:
            i = int(i)
            if not self._parked[i]:
                self._parked[i] = True
                self._pending.discard(i)
                self._cd[i].setSFString(hold)
                self.nodes[i].resetPhysics()
            spot = park_spot(i)
            self._tr[i].setSFVec3f(spot)
            self._rot[i].setSFRotation(LEVEL)
            self.nodes[i].setVelocity([0.0] * 6)
            self._cmd[i] = 0.0
            self._pos[i], self._vel[i], self._orn[i] = spot, 0.0, 0.0

    def place(self, i, pos):
        """Put parked drone i on its pad, still; it takes off in the next control step."""
        i = int(i)
        if self._lrx is not None and self.lidar['use_for_policy']:
            raise NotImplementedError('webots_l2_multi: place() with webots.lidar.use_for_policy is not '
                                      'supported (C3 covers city routes; a placed drone has no scan yet)')
        pos = [float(v) for v in pos]
        self._parked[i] = False
        self._tr[i].setSFVec3f(pos)
        self._rot[i].setSFRotation(LEVEL)
        self.nodes[i].resetPhysics()                        # velocities only: rotors keep hover speed
        self._pos[i], self._vel[i], self._orn[i] = pos, 0.0, 0.0
        self._z_spawn[i] = self._z_ref[i] = pos[2]
        self._since_release[i] = 0
        self._pending.add(i)

    # -------------------------------------------------------------- control
    def apply_actions(self, actions: dict):
        """actions: {'drone_i': [vx, vy, vz, yaw_rate]}; yaw rate ignored (always 0, as Level 2)."""
        tag = ''
        if self._lrx is not None:                           # C3: every flying drone scans this step
            self._scan_k += 1
            tag = ' ' + format_tag(self._scan_k, self.n_sub)
        for i in range(self.n_agents):                      # L2 140-142, per axis as PyBullet
            if self._parked[i]:                             # parked: stays in H, no command
                self._cmd[i] = 0.0
                continue
            v = np.clip(np.asarray(actions[f'drone_{i}'][:3], dtype=np.float64), -self.v_max, self.v_max)
            self._cmd[i] = v
            if i not in self._pending:                      # just placed: F after the first basic step
                self._cd[i].setSFString(format_command(self.ep, 'F', v) + tag)

    def step_simulation(self):
        if self._pending:
            # just-placed drones: one basic step still in H (the teleport lands, the GPS reads the
            # pad), then F for the rest of the control step; the other drones fly on unchanged
            if self.sup.step(self.basic_ms) == -1:
                raise RuntimeError("Webots terminated")
            for i in sorted(self._pending):
                self._cd[i].setSFString(format_command(self.ep, 'F', self._cmd[i]))
            self._pending.clear()
            if self.sup.step(self.step_ms - self.basic_ms) == -1:
                raise RuntimeError("Webots terminated")
        elif self.sup.step(self.step_ms) == -1:             # the controllers run n_sub basic steps
            raise RuntimeError("Webots terminated")
        self._read_state()
        if self._zones is not None:                         # C7: show / hide on the episode clock
            self._ep_step += 1
            self._zones.update(self._ep_step * self.control_dt)
        if self._lrx is not None:
            self._collect_scans()
        win = (self._since_release < DROP_CHECK_STEPS) & ~self._parked
        if win.any():
            start = win & (self._since_release < START_CHECK_STEPS)
            if start.any():
                self._start_check(np.flatnonzero(start))
            self._drop_check(np.flatnonzero(win))
            self._since_release[win] += 1
        self._max_roll = np.maximum(self._max_roll, np.abs(self._orn[:, 0]))
        self._max_pitch = np.maximum(self._max_pitch, np.abs(self._orn[:, 1]))
        self._max_speed = np.maximum(self._max_speed, np.linalg.norm(self._vel, axis=1))
        if self._rx is not None:
            self._drain()

    # ---------------------------------------------------------------- Lidar (C3)
    def _collect_scans(self):
        """The scans the drones took for this control step (tag _scan_k), image 4 ms before now."""
        t0 = time.perf_counter()
        st, rx = self._scan_log.stats, self._lrx
        shape = (self.lidar['layers'], self.lidar['horizontal_res'])
        self._scans = [None] * self.n_agents
        now = self.sup.getTime()
        while rx.getQueueLength() > 0:
            b = ctypes.string_at(rx.data, rx.getDataSize())
            rx.nextPacket()
            d = decode_scan(b)
            if d['ep'] != self.ep or d['k'] != self._scan_k:
                st['stale'] += 1
                continue
            if abs(d['t'] - (now - self.dt)) > 1e-6:
                st['wrong_time'] += 1
            d['img'] = np.frombuffer(d['ranges'], dtype=np.float32).reshape(shape)
            self._scans[d['idx']] = d
            st['scans'] += 1
            st['bytes'] += len(b)
        st['recv_s'] += time.perf_counter() - t0

    def _scan_lidar(self):
        scans, rings = super()._scan_lidar()                # analytic; also updates the headings
        if self._lrx is None:
            return scans, rings
        t0 = time.perf_counter()
        log, rng, n_ring = self._scan_log, self.lidar_cfg['range'], self.lidar_cfg['ring_rays']
        policy = self.lidar['use_for_policy']
        key = (self.ep, self._scan_k)
        log_now, self._logged = self._logged != key, key     # get_raw_state may run twice in a step
        out_s, out_r = [], []
        for i in range(self.n_agents):
            sc = self._scans[i]
            ana = np.asarray(scans[i] + rings[i], dtype=np.float64)
            heading = self._heading[f'drone_{i}']
            if self._parked[i] or sc is None:
                if not self._parked[i]:
                    if log_now:
                        log.stats['missing'] += 1
                    if policy:
                        raise RuntimeError(f"webots_l2_multi: no Lidar scan from drone_{i} for control "
                                           f"step {self._scan_k} (episode {self.ep})")
                lid = np.full(len(ana), rng)
            else:
                lid = pick_rays(sc['img'], analytic_dirs(heading, n_ring), np.asarray(sc['pos']), sc['rpy'], rng)
            if log_now:
                nan3 = [np.nan] * 3
                log.add(ep=self.ep, k=self._scan_k, drone=i, t_state=self.sup.getTime(),
                        t_scan=np.nan if sc is None else sc['t'], pos=self._pos[i].copy(), orn=self._orn[i].copy(),
                        heading=heading, scan_pos=nan3 if sc is None else list(sc['pos']),
                        scan_rpy=nan3 if sc is None else list(sc['rpy']),
                        lidar=lid.astype(np.float32), analytic=ana.astype(np.float32),
                        all_pos=self._pos.copy(), all_yaw=self._orn[:, 2].copy(),
                        cpu_send=np.nan if sc is None else sc['cpu'], parked=bool(self._parked[i]))
            if policy:
                out_s.append(lid[:self.n_lidar].tolist())
                out_r.append(lid[self.n_lidar:].tolist())
        log.stats['pick_s'] += time.perf_counter() - t0
        return (out_s, out_r) if policy else (scans, rings)

    def lidar_stats(self):
        """C3: counters and timing of the Lidar scans so far (None with the Lidar off)."""
        return None if self._scan_log is None else dict(self._scan_log.stats)

    # ---------------------------------------------------------------- radio (E2)
    def _drain(self):
        """Move the reports that arrived into the inbox, stamped with the time now."""
        rx, now = self._rx, self.sup.getTime() - self._t0
        while rx.getQueueLength() > 0:
            d = decode_report(rx.getString())
            rx.nextPacket()
            if d['ep'] != self.ep:
                self.radio_rejected += 1
                continue
            self._inbox.append(Report(d['id'], d['t'] - self._t0, np.array(d['p']), np.array(d['v']),
                                      d['i'], d['seq'], now))

    def reports(self):
        """The radio reports received since the last call (oldest first), ids drone_i,
        t_sent / t_received in episode time."""
        if self._rx is None:
            raise RuntimeError('webots_l2_multi: reports() needs webots.radio.enabled')
        got, self._inbox = self._inbox, []
        return got

    def send_intent(self, i, intent):
        """Send drone i its intent [[t, x, y, z], ...] (episode time) over the radio."""
        if self._tx is None:
            raise RuntimeError('webots_l2_multi: send_intent() needs webots.radio.enabled')
        self._tx.send(encode_intent(f'drone_{int(i)}', self.ep, intent))

    def _read_state(self):                                  # L2 124-127, every drone
        for i, nd in enumerate(self.nodes):
            self._pos[i] = nd.getPosition()
            self._vel[i] = nd.getVelocity()[:3]
            self._orn[i] = _euler_from_matrix(nd.getOrientation())

    def _drop_check(self, idx=None):
        """idx: the drones to check (default all)."""
        # the setpoint each controller should now hold: z_des += vz * dt per basic step, floor 0.2
        for _ in range(self.n_sub):
            self._z_ref = np.maximum(0.2, self._z_ref + self._cmd[:, 2] * self.dt)
        drop = np.minimum(self._z_ref, self._z_spawn) - self._pos[:, 2]
        idx = np.arange(self.n_agents) if idx is None else np.asarray(idx, dtype=int)
        bad = idx[drop[idx] > DROP_LIMIT]
        if bad.size:
            names = ', '.join(f'drone_{i} ({drop[i]:.2f} m)' for i in bad)
            raise RuntimeError(f"webots_l2_multi: {names} fell below its spawn height and altitude "
                               f"setpoint right after spin-up; its cf_velocity controller is probably "
                               f"not running. See the Webots log: {self.log_path}")

    def _start_check(self, idx=None):
        """idx: the drones to check (default all)."""
        speed = np.linalg.norm(self._vel, axis=1)
        idx = np.arange(self.n_agents) if idx is None else np.asarray(idx, dtype=int)
        bad = idx[speed[idx] > START_SPEED_LIMIT]
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
        if self._scan_log is not None:
            self._scan_log.save()
        self._check_controller_log()                        # the last episode
