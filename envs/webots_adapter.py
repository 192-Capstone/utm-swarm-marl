"""
webots_adapter.py
-----------------
Webots backend for the UTM swarm, Level 1: the same interface and raw-state
schema as envs/pybullet_adapter.py (PyBulletAdapter), so SwarmEnv, the
policies and eval_suite run unchanged on either simulator
(environment.simulator: pybullet | webots).

    reset_world(spawn_positions, goal_positions, obstacles=None)
    apply_actions(actions)          # {'drone_i': [vx, vy, vz, yaw_rate]}
    step_simulation()
    get_raw_state()  -> {'drone_i': {'position', 'velocity', 'orientation',
                                     'lidar' (14), 'lidar_ring' (n)}, 'goals': {...}}
    get_positions()  -> (N, 3) array
    close()

How it works
  - The adapter starts its own Webots process (webots/worlds/swarm_l1.wbt:
    ENU, z-up, gravity 0, basicTimeStep 8 ms) and connects to the world's
    bodiless Supervisor robot as an <extern> controller, on a free TCP port,
    so several evaluations can run in parallel, one Webots each.
  - Drones (DEF DRONE_i) and obstacles are Solid nodes imported by the
    Supervisor. Drones are 0.3 m boxes with Physics but no boundingObject:
    they fly through everything, like PyBullet with collision_physics off;
    crashes are detected geometrically by SwarmEnv, as before.
  - Velocity control, as in PyBullet: each control step the policy's
    velocity command is written to the drone body (Supervisor setVelocity)
    and Webots integrates it for control_timestep (32 ms = 4 x 8 ms).
  - PyBullet parity (webots.match_pybullet, default true): PyBullet's body
    loses a little speed every physics sub-step (Bullet's default damping
    0.04 plus a speed-proportional part): per sub-step of dt seconds the
    velocity is multiplied by f = 1 - 0.04 * dt * (1 + |v|), fitted to within
    0.3 um per step (training physics: 2 x 16 ms per control step). Level 1
    reproduces this, so the same commands give the same trajectory and the
    same velocity readings; without it a drone drifts ~1 cm per 100 steps.
  - LiDAR (Level 1): computed ANALYTICALLY from the obstacle layout and the
    drone positions, with the ray directions of envs/lidar.py and the rules
    of PyBulletAdapter._scan_lidar: rays start RAY_START from the centre,
    range 6 m, the first thing hit counts, a ray that hits the ground first
    sees nothing, other drones count only with lidar.sees_drones, and a body
    the ray starts inside is ignored (PyBullet's ray test does the same).
    Sensing is therefore identical to PyBullet by construction. A Webots
    Lidar device comes with Level 2.

Limitations
  - One WebotsAdapter per Python process (the Webots controller library
    holds one robot per process), so the trainer's separate eval env can't
    use Webots; eval_suite and the scripts that make one env are fine.
  - Drone boxes for drone-vs-drone LiDAR hits are rotated by yaw only
    (velocity control never rolls or pitches the body).
"""

import atexit
import os
import socket
import subprocess
import sys
import time

import numpy as np

from envs import lidar as lidar_mod

RAY_START = 0.3          # same as PyBulletAdapter: rays start outside the drone body
DRONE_HALF = 0.15        # drone box half-size (PyBullet: 0.15)
WEBOTS_HOME = os.environ.get('WEBOTS_HOME', '/usr/local/webots')
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_WORLD = os.path.join(ROOT, 'webots', 'worlds', 'swarm_l1.wbt')
ROBOT_NAME = 'swarm_supervisor'

DRONE_NODE = """DEF DRONE_{i} Solid {{
  translation {x} {y} {z}
  name "drone_{i}"
  children [
    Shape {{
      appearance PBRAppearance {{ baseColor 0.1 0.4 0.9 roughness 0.5 metalness 0 }}
      geometry Box {{ size 0.3 0.3 0.3 }}
    }}
  ]
  physics Physics {{
    density -1
    mass 1
    centerOfMass [ 0 0 0 ]
    inertiaMatrix [ 0.015 0.015 0.015, 0 0 0 ]
    damping Damping {{ linear 0 angular 0 }}
  }}
}}"""

PILLAR_NODE = """Solid {{
  translation {x} {y} {zc}
  name "obstacle_{k}"
  children [
    Shape {{
      appearance PBRAppearance {{ baseColor 0.55 0.43 0.39 roughness 1 metalness 0 }}
      geometry Cylinder {{ radius {r} height {h} }}
    }}
  ]
}}"""

BOX_NODE = """Solid {{
  translation {x} {y} {zc}
  name "obstacle_{k}"
  children [
    Shape {{
      appearance PBRAppearance {{ baseColor 0.6 0.6 0.62 roughness 1 metalness 0 }}
      geometry Box {{ size {sx} {sy} {h} }}
    }}
  ]
}}"""


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('localhost', 0))
        return s.getsockname()[1]


def _euler_from_matrix(m):
    """Webots 3x3 orientation (row-major, 9 floats) -> (roll, pitch, yaw) in
    the same convention as PyBullet's getEulerFromQuaternion (R = Rz Ry Rx)."""
    r = np.asarray(m, dtype=np.float64).reshape(3, 3)
    pitch = float(np.arcsin(np.clip(-r[2, 0], -1.0, 1.0)))
    roll = float(np.arctan2(r[2, 1], r[2, 2]))
    yaw = float(np.arctan2(r[1, 0], r[0, 0]))
    return [roll, pitch, yaw]


# ------------------------------------------------------- analytic LiDAR ----
def _slab_entry(o, d, lo, hi):
    """Ray-vs-axis-aligned-box entry/exit parameters for many rays.
    o, d: (R, k); lo, hi: (k,). Returns (t_in, t_out), each (R,)."""
    with np.errstate(divide='ignore', invalid='ignore'):
        inv = 1.0 / d
        t1 = (lo - o) * inv
        t2 = (hi - o) * inv
    near, far = np.minimum(t1, t2), np.maximum(t1, t2)
    # a ray parallel to a slab: inside the slab -> (-inf, inf), outside -> empty
    par = d == 0.0
    inside = (o >= lo) & (o <= hi)
    near = np.where(par, np.where(inside, -np.inf, np.inf), near)
    far = np.where(par, np.where(inside, np.inf, -np.inf), far)
    return near.max(axis=1), far.min(axis=1)


def _cylinder_entry(o, d, cx, cy, r, h):
    """Ray-vs-vertical-cylinder (z from 0 to h) entry/exit parameters."""
    ox, oy = o[:, 0] - cx, o[:, 1] - cy
    dx, dy = d[:, 0], d[:, 1]
    a = dx * dx + dy * dy
    b = 2.0 * (ox * dx + oy * dy)
    c = ox * ox + oy * oy - r * r
    disc = b * b - 4.0 * a * c
    with np.errstate(divide='ignore', invalid='ignore'):
        sq = np.sqrt(np.maximum(disc, 0.0))
        s1 = (-b - sq) / (2.0 * a)
        s2 = (-b + sq) / (2.0 * a)
    vertical = a < 1e-12
    side_in = np.where(vertical, np.where(c <= 0.0, -np.inf, np.inf), np.where(disc >= 0.0, s1, np.inf))
    side_out = np.where(vertical, np.where(c <= 0.0, np.inf, -np.inf), np.where(disc >= 0.0, s2, -np.inf))
    zt, zb = _slab_entry(o[:, 2:3], d[:, 2:3], np.array([0.0]), np.array([h]))
    return np.maximum(side_in, zt), np.minimum(side_out, zb)


def scan_lidar(positions, yaws, headings, layout, cfg, n_lidar):
    """Analytic version of PyBulletAdapter._scan_lidar.
    positions: (N, 3); yaws: (N,) body yaw (for drone boxes); headings: (N,)
    fan heading. Returns (fan distances [N][n_lidar], ring distances [N][ring])."""
    rng = cfg['range']
    n_ring = cfg['ring_rays']
    n = len(positions)
    seg = rng - RAY_START
    per = n_lidar + n_ring
    dirs, origins, owner = [], [], []
    for i in range(n):
        d = lidar_mod.ray_directions(headings[i])
        if n_ring:
            d = np.vstack([d, lidar_mod.ring_directions(headings[i], n_ring)])
        dirs.append(d)
        origins.append(positions[i] + RAY_START * d)
        owner.append(np.full(len(d), i))
    d = np.vstack(dirs)
    o = np.vstack(origins)
    owner = np.concatenate(owner)
    R = len(d)
    best = np.full(R, np.inf)          # nearest entry t along the segment (0..seg)
    seen = np.zeros(R, dtype=bool)     # does the nearest thing count as "seen"

    def consider(t_in, t_out, counts):
        nonlocal best, seen
        # PyBullet: the ray must enter the body (start outside it, t_in >= 0)
        # within the segment; a body the ray starts inside is not reported.
        hit = (t_in >= 0.0) & (t_in <= t_out) & (t_in <= seg)
        closer = hit & (t_in < best)
        best = np.where(closer, t_in, best)
        seen = np.where(closer, counts, seen)

    # ground plane z = 0: hit when going down from above
    with np.errstate(divide='ignore', invalid='ignore'):
        tg = np.where(d[:, 2] < 0.0, -o[:, 2] / d[:, 2], np.inf)
    consider(tg, np.full(R, np.inf), False)
    for ob in layout:
        cx, cy = ob['center']
        h = ob['height']
        if ob['type'] == 'pillar':
            t_in, t_out = _cylinder_entry(o, d, cx, cy, ob['radius'], h)
        else:
            hx, hy = ob['half']
            t_in, t_out = _slab_entry(o, d, np.array([cx - hx, cy - hy, 0.0]), np.array([cx + hx, cy + hy, h]))
        consider(t_in, t_out, True)
    for j in range(n):
        # drone j's box, rotated by its yaw: express the rays in its frame
        c, s = np.cos(yaws[j]), np.sin(yaws[j])
        rel = o - positions[j]
        ol = np.stack([c * rel[:, 0] + s * rel[:, 1], -s * rel[:, 0] + c * rel[:, 1], rel[:, 2]], axis=1)
        dl = np.stack([c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1], d[:, 2]], axis=1)
        t_in, t_out = _slab_entry(ol, dl, np.full(3, -DRONE_HALF), np.full(3, DRONE_HALF))
        t_in = np.where(owner == j, np.inf, t_in)          # never its own body
        consider(t_in, t_out, cfg['sees_drones'])
    dist = np.where(seen & np.isfinite(best), RAY_START + best, rng).reshape(n, per)
    return [row[:n_lidar].tolist() for row in dist], [row[n_lidar:].tolist() for row in dist]


# -------------------------------------------------------------- adapter ----
class WebotsAdapter:
    def __init__(self, config, gui: bool = False):
        self.config = config
        self.n_agents = config['environment']['n_agents']
        self.v_max = config['environment'].get('v_max', 2.0)
        self.omega_max = config['environment'].get('omega_max', 1.0)
        self.n_lidar = config['observation'].get('n_lidar_rays', 14)
        self.control_dt = config['environment'].get('control_timestep', 0.032)
        wcfg = config.get('webots') or {}
        self.world = os.path.expanduser(wcfg.get('world', DEFAULT_WORLD))
        # PyBullet parity: the same physics sub-steps PyBulletAdapter uses
        self.match_pybullet = bool(wcfg.get('match_pybullet', True))
        train_cfg = config['environment'].get('training', {})
        self.pb_dt = train_cfg.get('physics_timestep', 1.0 / 240.0)
        self.pb_repeat = train_cfg.get('action_repeat', 2)
        self._vel_report = None

        self.goals = {}
        self.layout = []
        self.lidar_cfg = lidar_mod.lidar_config(config)
        self._heading = {}

        self._start_webots(gui, wcfg)
        self.basic_ms = int(round(self.sup.getBasicTimeStep()))
        self.step_ms = int(round(self.control_dt * 1000))
        if self.step_ms % self.basic_ms:
            raise ValueError(f"control_timestep {self.control_dt}s is not a multiple of "
                             f"basicTimeStep {self.basic_ms} ms")
        self._obstacle_root = self.sup.getFromDef('OBSTACLES').getField('children')
        drones_root = self.sup.getFromDef('DRONES').getField('children')
        for i in range(self.n_agents):
            drones_root.importMFNodeFromString(-1, DRONE_NODE.format(i=i, x=i * 2.0, y=0.0, z=1.0))
        self.nodes = [self.sup.getFromDef(f'DRONE_{i}') for i in range(self.n_agents)]
        self._tr = [nd.getField('translation') for nd in self.nodes]
        self._rot = [nd.getField('rotation') for nd in self.nodes]
        self.drone_ids = [nd.getId() for nd in self.nodes]
        self.sup.step(self.basic_ms)               # let the imports take effect
        self._pos = np.array([nd.getPosition() for nd in self.nodes], dtype=np.float64)
        self._vel = np.zeros((self.n_agents, 3))
        self._orn = np.zeros((self.n_agents, 3))

    # ---------------------------------------------------------------- setup
    def _start_webots(self, gui, wcfg):
        sys.path.insert(0, os.path.join(WEBOTS_HOME, 'lib', 'controller', 'python'))
        os.environ['WEBOTS_HOME'] = WEBOTS_HOME
        port = int(wcfg.get('port') or _free_port())
        cmd = [os.path.join(WEBOTS_HOME, 'webots'), '--batch', '--stdout', '--stderr', f'--port={port}']
        cmd += ['--mode=realtime'] if gui else ['--mode=fast', '--no-rendering', '--minimize']
        cmd.append(self.world)
        log = wcfg.get('log') or os.devnull
        self._log = open(log, 'w')
        self.proc = subprocess.Popen(cmd, stdout=self._log, stderr=subprocess.STDOUT)
        atexit.register(self.close)
        os.environ['WEBOTS_CONTROLLER_URL'] = f'tcp://localhost:{port}/{ROBOT_NAME}'
        from controller import Supervisor       # noqa: E402  (needs WEBOTS_HOME / URL first)
        t0 = time.time()
        self.sup = Supervisor()                  # blocks until Webots accepts the connection
        self.connect_s = time.time() - t0

    # ---------------------------------------------------------------- world
    def reset_world(self, spawn_positions: dict, goal_positions: dict, obstacles=None):
        for i, nd in enumerate(self.nodes):
            key = f'drone_{i}'
            self._tr[i].setSFVec3f([float(v) for v in spawn_positions[key]])
            self._rot[i].setSFRotation([0.0, 0.0, 1.0, 0.0])
            nd.resetPhysics()
            nd.setVelocity([0.0] * 6)
            self.goals[key] = np.array(goal_positions[key], dtype=np.float32)
            self._pos[i] = np.array(spawn_positions[key], dtype=np.float64)
        self._vel[:] = 0.0
        self._orn[:] = 0.0
        self._heading = {}
        self._build_obstacles(obstacles or [])
        # Supervisor requests are applied at the next step, and resetPhysics
        # would land AFTER the first action's setVelocity and cancel it (the
        # drone would sit still for the first control step). One basic step
        # at zero velocity applies the reset first; nothing moves.
        self.sup.step(self.basic_ms)
        for i, nd in enumerate(self.nodes):
            self._pos[i] = nd.getPosition()

    def _build_obstacles(self, layout):
        """Remove the previous episode's obstacle nodes and import new ones.
        They are only visual here (the LiDAR is analytic and crashes are
        checked geometrically); the Level-2 Lidar device will see them."""
        for _ in range(self._obstacle_root.getCount()):
            self._obstacle_root.removeMF(-1)
        self.layout = list(layout)
        for k, ob in enumerate(layout):
            h = ob['height']
            x, y = ob['center']
            if ob['type'] == 'pillar':
                s = PILLAR_NODE.format(k=k, x=x, y=y, zc=h / 2, r=ob['radius'], h=h)
            else:
                hx, hy = ob['half']
                s = BOX_NODE.format(k=k, x=x, y=y, zc=h / 2, sx=2 * hx, sy=2 * hy, h=h)
            self._obstacle_root.importMFNodeFromString(-1, s)

    # -------------------------------------------------------------- control
    def apply_actions(self, actions: dict):
        """actions: {'drone_i': [vx, vy, vz, yaw_rate]} (yaw_rate optional)."""
        report = np.zeros((self.n_agents, 3))
        for i, nd in enumerate(self.nodes):
            a = actions[f'drone_{i}']
            v = np.clip(np.asarray(a[:3], dtype=np.float64), -self.v_max, self.v_max)
            yaw_rate = float(np.clip(a[3], -self.omega_max, self.omega_max)) if len(a) > 3 else 0.0
            if self.match_pybullet:
                # PyBullet's per-sub-step speed loss: fly the same displacement
                # over the control step, and report the velocity PyBullet would
                f = 1.0 - 0.04 * self.pb_dt * (1.0 + float(np.linalg.norm(v)))
                disp = v * self.pb_dt * sum(f ** (k + 1) for k in range(self.pb_repeat))
                report[i] = v * f ** self.pb_repeat
                v = disp / self.control_dt
            nd.setVelocity([float(v[0]), float(v[1]), float(v[2]), 0.0, 0.0, yaw_rate])
        self._vel_report = report if self.match_pybullet else None

    def step_simulation(self):
        if self.sup.step(self.step_ms) == -1:
            raise RuntimeError("Webots terminated")
        for i, nd in enumerate(self.nodes):
            self._pos[i] = nd.getPosition()
            self._vel[i] = nd.getVelocity()[:3] if self._vel_report is None else self._vel_report[i]
            self._orn[i] = _euler_from_matrix(nd.getOrientation())

    # --------------------------------------------------------------- state
    def get_raw_state(self) -> dict:
        state = {'goals': {}}
        for i in range(self.n_agents):
            key = f'drone_{i}'
            state[key] = {
                'position': self._pos[i].tolist(),
                'velocity': self._vel[i].tolist(),
                'orientation': self._orn[i].tolist(),
            }
            state['goals'][key] = self.goals[key].tolist()
        scans, rings = self._scan_lidar()
        for i in range(self.n_agents):
            state[f'drone_{i}']['lidar'] = scans[i]
            if self.lidar_cfg['ring_rays']:
                state[f'drone_{i}']['lidar_ring'] = rings[i]
        return state

    def _scan_lidar(self):
        rng = self.lidar_cfg['range']
        n_ring = self.lidar_cfg['ring_rays']
        if not self.lidar_cfg['enabled']:
            return ([[rng] * self.n_lidar for _ in range(self.n_agents)],
                    [[rng] * n_ring for _ in range(self.n_agents)])
        headings = []
        for i in range(self.n_agents):
            key = f'drone_{i}'
            yaw = lidar_mod.heading_to_goal(self._pos[i], self.goals[key], self._heading.get(key, 0.0))
            self._heading[key] = yaw
            headings.append(yaw)
        return scan_lidar(self._pos, self._orn[:, 2], headings, self.layout, self.lidar_cfg, self.n_lidar)

    def get_positions(self) -> np.ndarray:
        return self._pos.astype(np.float32)

    def close(self):
        proc = getattr(self, 'proc', None)
        if proc is None:
            return
        self.proc = None
        try:
            self.sup.simulationQuit(0)
        except Exception:
            pass
        try:
            proc.wait(timeout=15)
        except Exception:
            proc.kill()
        try:
            self._log.close()
        except Exception:
            pass
