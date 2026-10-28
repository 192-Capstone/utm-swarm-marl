"""
test_webots_l2_multi_city.py
----------------------------
C5 pieces of the multi-drone Webots Level 2 (envs/webots_l2_multi.py) that can be tested
without Webots (a fake supervisor records what the adapter asks for):

  A. The controller's command handling (envs/cf_velocity_pid.py ControllerState): every
     H -> F is a fresh start (arm); a normal episode (H during spin-up, then F) arms exactly
     where the old logic did, so C1 flights are unchanged.
  B. park / place: parked drones are held in H at their own far spots and re-pinned every call;
     apply_actions leaves them in H; a placed drone gets F only after one basic step; the
     cached state changes at once; start / drop checks run for a placed drone, not for parked ones.
  C. Buildings are rebuilt only when the layout changes (reuse_layout).
  D. Route following through the Webots adapter's own goal / LiDAR code: the carrot is the
     adapter's goal and the LiDAR fan points at it.

Run: python -m pytest tests/test_webots_l2_multi_city.py -q
"""

import math
import os
import sys

import numpy as np
import pytest
import torch
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from envs import cf_velocity_pid as cf                                         # noqa: E402
from envs.webots_l2_multi import (WebotsL2MultiAdapter, park_spot, layout_key,  # noqa: E402
                                  START_CHECK_STEPS, DROP_CHECK_STEPS)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ======================================================== A. controller command handling
def _cmd(ep, mode):
    return (ep, mode, (0.0, 0.0, 0.0))


def _old_arm_steps(cmds):
    """The controller's logic before C5: PID reset at a new episode number, altitude setpoint and
    previous GPS taken at the first F of each episode. Returns the steps where the old controller
    took them ('arm'), for comparison with ControllerState."""
    ep, flying_ep, out = None, None, []
    for c in cmds:
        if c is None:
            out.append('idle')
            continue
        if c[0] != ep:
            ep = c[0]
        if c[1] == 'H':
            out.append('hold')
            continue
        if flying_ep != ep:
            flying_ep = ep
            out.append('arm')
        else:
            out.append('fly')
    return out


def _new(cmds):
    s = cf.ControllerState()
    return [s.update(c) for c in cmds]


def test_normal_episodes_arm_where_the_old_controller_did():
    """C1 sequences (idle before the first reset; per episode: H during spin-up, then F): same
    actions step by step. The PID is never run in H, so resetting it at the arm instead of at the
    new episode number leaves the same PID state."""
    cmds = [None] * 3
    for ep in (1, 2, 3):
        cmds += [_cmd(ep, 'H')] * 500 + [_cmd(ep, 'F')] * 40
    assert _new(cmds) == _old_arm_steps(cmds)


def test_parked_then_placed_drone_arms_again():
    cmds = [_cmd(1, 'H')] * 3 + [_cmd(1, 'F')] * 3 + [_cmd(1, 'H')] * 4 + [_cmd(1, 'F')] * 2
    assert _new(cmds) == ['hold'] * 3 + ['arm', 'fly', 'fly'] + ['hold'] * 4 + ['arm', 'fly']
    assert _old_arm_steps(cmds)[-2] == 'fly'          # the old controller did not: stale GPS / setpoint


def test_new_episode_without_hold_and_idle_arm():
    assert _new([_cmd(1, 'F'), _cmd(1, 'F'), _cmd(2, 'F')]) == ['arm', 'fly', 'arm']
    assert _new([_cmd(1, 'F'), None, _cmd(1, 'F')]) == ['arm', 'idle', 'arm']


def test_controller_uses_controller_state():
    src = open(os.path.join(ROOT, 'webots', 'controllers', 'cf_velocity', 'cf_velocity.py')).read()
    assert 'ControllerState()' in src and 'flying_ep' not in src


# ======================================================== B. park / place (fake Webots)
class _Field:
    def __init__(self, node, name):
        self.node, self.name = node, name

    def setSFVec3f(self, v):
        self.node.log.append(('tr', self.node.i, [float(x) for x in v]))
        self.node.pos = [float(x) for x in v]

    def setSFRotation(self, r):
        self.node.log.append(('rot', self.node.i, list(r)))

    def setSFString(self, s):
        self.node.log.append(('cd', self.node.i, s))
        self.node.cd = s


class _Node:
    def __init__(self, i, log):
        self.i, self.log, self.cd = i, log, ''
        self.pos, self.vel = [2.0 * i, 0.0, 1.0], [0.0] * 3

    def getPosition(self):
        return list(self.pos)

    def getVelocity(self):
        return list(self.vel) + [0.0] * 3

    def getOrientation(self):
        return [1, 0, 0, 0, 1, 0, 0, 0, 1]

    def resetPhysics(self):
        self.log.append(('reset', self.i))

    def setVelocity(self, v):
        self.log.append(('vel', self.i))
        self.vel = list(v[:3])


class _Sup:
    def __init__(self, log):
        self.log = log

    def step(self, ms):
        self.log.append(('step', ms))
        return 0


def _fake(n=3):
    ad = WebotsL2MultiAdapter.__new__(WebotsL2MultiAdapter)
    log = []
    ad.log = log
    ad.n_agents, ad.n_sub, ad.dt, ad.basic_ms, ad.step_ms, ad.ep, ad.v_max = n, 8, 0.004, 4, 32, 1, 2.0
    ad.sup = _Sup(log)
    ad.nodes = [_Node(i, log) for i in range(n)]
    ad._tr = [_Field(nd, 'translation') for nd in ad.nodes]
    ad._rot = [_Field(nd, 'rotation') for nd in ad.nodes]
    ad._cd = [_Field(nd, 'customData') for nd in ad.nodes]
    ad._pos = np.array([nd.pos for nd in ad.nodes], dtype=np.float64)
    ad._vel, ad._orn, ad._cmd = np.zeros((n, 3)), np.zeros((n, 3)), np.zeros((n, 3))
    ad._z_spawn, ad._z_ref = np.ones(n), np.ones(n)
    ad._since_release = np.full(n, DROP_CHECK_STEPS)
    ad._parked = np.zeros(n, dtype=bool)
    ad._pending = set()
    ad.log_path, ad._log_pos = '/nonexistent', 0
    ad._reset_stats()
    return ad


def _act(n, v=(0.5, 0.0, 0.0)):
    return {f'drone_{i}': [*v, 0.0] for i in range(n)}


def test_park_holds_and_pins_each_drone_at_its_own_far_spot():
    ad = _fake()
    ad.park([0, 2])
    for i in (0, 2):
        assert cf.parse_command(ad.nodes[i].cd)[1] == 'H'
        assert ad.nodes[i].pos == park_spot(i)
        assert np.allclose(ad.get_positions()[i], park_spot(i))          # cached at once
    assert ad.nodes[1].cd == ''                                           # untouched
    assert math.dist(park_spot(0), park_spot(2)) >= 100.0
    assert min(park_spot(i)[0] for i in range(20)) > 900.0                # beyond any map + 30 m sensing
    ad.nodes[0].pos = [0.0, 0.0, 0.0]                                      # drifted
    ad.park([0, 2])                                                        # SwarmEnv: every step
    assert ad.nodes[0].pos == park_spot(0)                                # re-pinned
    assert sum(1 for e in ad.log if e == ('reset', 0)) == 1               # resetPhysics only when parking


def test_apply_actions_leaves_parked_drones_in_hold():
    ad = _fake()
    ad.park([1])
    ad.apply_actions(_act(3, (0.5, 0.0, 2.0)))
    assert cf.parse_command(ad.nodes[1].cd)[1] == 'H'
    assert np.allclose(ad._cmd[1], 0.0)                                   # drop check setpoint stays put
    assert cf.parse_command(ad.nodes[0].cd)[1:] == ('F', (0.5, 0.0, 2.0))


def test_placed_drone_gets_F_after_one_basic_step():
    ad = _fake()
    ad.park([1])
    pad = [3.0, 4.0, 1.0]
    ad.place(1, pad)
    assert ad.nodes[1].pos == pad and np.allclose(ad.get_positions()[1], pad)     # cached at once
    assert ('reset', 1) in ad.log[-3:]
    del ad.log[:]
    ad.apply_actions(_act(3, (0.3, 0.0, 0.0)))
    assert [e[1] for e in ad.log if e[0] == 'cd'] == [0, 2]               # not drone_1 yet
    assert cf.parse_command(ad.nodes[1].cd)[1] == 'H'
    del ad.log[:]
    ad.step_simulation()
    assert ad.log[0] == ('step', 4)
    assert ad.log[1][:2] == ('cd', 1) and cf.parse_command(ad.log[1][2])[1:] == ('F', (0.3, 0.0, 0.0))
    assert ad.log[2] == ('step', 28)
    del ad.log[:]
    ad.apply_actions(_act(3)); ad.step_simulation()                       # then normal steps
    assert [e for e in ad.log if e[0] == 'step'] == [('step', 32)]
    assert cf.parse_command(ad.nodes[1].cd)[1] == 'F'


def test_start_check_covers_a_placed_drone_not_parked_ones():
    ad = _fake()
    ad.park([0, 1])
    ad.nodes[0].vel = [50.0, 0.0, 0.0]                                    # parked: never checked
    ad.apply_actions(_act(3)); ad.step_simulation()
    ad.place(1, [0.0, 0.0, 1.0])
    ad.nodes[1].vel = [0.0, 0.0, 5.0]                                     # leftover motor speed
    ad.apply_actions(_act(3))
    with pytest.raises(RuntimeError, match='drone_1'):
        ad.step_simulation()


def test_placed_drone_check_window_ends():
    ad = _fake()
    ad.park([1]); ad.place(1, [0.0, 0.0, 1.0])
    for _ in range(DROP_CHECK_STEPS):
        ad.apply_actions(_act(3, (0.0, 0.0, 0.0))); ad.step_simulation()
    assert ad._since_release[1] == DROP_CHECK_STEPS
    ad.nodes[1].vel = [0.0, 0.0, 5.0]; ad.nodes[1].pos = [0.0, 0.0, 0.0]  # later: not a start fault
    ad.apply_actions(_act(3, (0.0, 0.0, 0.0))); ad.step_simulation()


def test_drop_check_on_a_placed_drone_names_it():
    ad = _fake()
    ad.park([2]); ad.place(2, [0.0, 0.0, 2.0])
    ad.nodes[2].pos = [0.0, 0.0, 1.5]                                     # fell 0.5 m
    ad.apply_actions(_act(3, (0.0, 0.0, 0.0)))
    with pytest.raises(RuntimeError, match='drone_2'):
        ad.step_simulation()
    assert START_CHECK_STEPS < DROP_CHECK_STEPS


# ======================================================== C. layout reuse
class _Root:
    def __init__(self):
        self.nodes, self.imports = [], 0

    def getCount(self):
        return len(self.nodes)

    def removeMF(self, k):
        self.nodes.pop(k)

    def importMFNodeFromString(self, k, s):
        self.nodes.append(s)
        self.imports += 1


BOX = {'type': 'box', 'center': (1.0, 2.0), 'half': (0.5, 1.5), 'height': 8.0}


def _builder(reuse=True):
    ad = WebotsL2MultiAdapter.__new__(WebotsL2MultiAdapter)
    ad.real, ad.reuse_layout, ad._built_key = True, reuse, None
    ad.layout_builds, ad.layout_build_s = 0, 0.0
    ad._obstacle_root = _Root()
    return ad


def test_buildings_rebuilt_only_when_the_layout_changes():
    ad = _builder()
    ad._build_obstacles([BOX])
    ad._build_obstacles([dict(BOX)])                                      # equal layout: kept
    assert (ad.layout_builds, ad._obstacle_root.imports) == (1, 1)
    ad._build_obstacles([dict(BOX, yaw=0.3)])                             # changed: rebuilt
    assert (ad.layout_builds, ad._obstacle_root.getCount()) == (2, 1)
    assert 'obstacle_0' in ad._obstacle_root.nodes[0]
    ad._build_obstacles([])
    assert ad._obstacle_root.getCount() == 0 and ad.layout == []


def test_reuse_off_always_rebuilds():
    ad = _builder(reuse=False)
    ad._build_obstacles([BOX]); ad._build_obstacles([BOX])
    assert ad.layout_builds == 2 and ad._obstacle_root.imports == 2


def test_layout_key_sees_every_building_field():
    base = layout_key([BOX])
    for change in ({'center': (1.0, 2.001)}, {'half': (0.5, 1.6)}, {'height': 8.5}, {'yaw': 0.1}):
        assert layout_key([dict(BOX, **change)]) != base
    assert layout_key([dict(BOX, center=np.array([1.0, 2.0]))]) == base


# ======================================================== D. route following via the Webots adapter
class _KinematicWebots(WebotsL2MultiAdapter):
    """The Webots adapter's goals, LiDAR and raw state (inherited, unchanged), with the
    simulator replaced by perfect velocity tracking."""

    def __init__(self, config):
        from envs import lidar as lidar_mod
        self.config, self.n_agents = config, config['environment']['n_agents']
        self.n_lidar = config['observation'].get('n_lidar_rays', 14)
        self.control_dt = config['environment'].get('control_timestep', 0.032)
        self.lidar_cfg = lidar_mod.lidar_config(config)
        self.goals, self.layout, self._heading = {}, [], {}
        n = self.n_agents
        self._pos, self._vel, self._orn = np.zeros((n, 3)), np.zeros((n, 3)), np.zeros((n, 3))

    def reset_world(self, spawn_positions, goal_positions, obstacles=None):
        for i in range(self.n_agents):
            self._pos[i] = spawn_positions[f'drone_{i}']
            self.goals[f'drone_{i}'] = np.array(goal_positions[f'drone_{i}'], dtype=np.float32)
        self._vel[:] = 0.0
        self._heading, self.layout = {}, list(obstacles or [])

    def apply_actions(self, actions):
        self._vel = np.array([actions[f'drone_{i}'][:3] for i in range(self.n_agents)], dtype=np.float64)

    def step_simulation(self):
        self._pos += self._vel * self.control_dt

    def close(self):
        pass


WALL = {'type': 'box', 'center': (0.0, -2.0), 'half': (0.4, 8.0), 'height': 10.0}


def test_route_carrot_drives_the_webots_goal_and_lidar_heading():
    from envs.route_planner import build_grid, plan, simplify
    from envs.swarm_env import SwarmEnv
    cfg = yaml.safe_load(open(os.path.join(ROOT, 'configs', 'config_stage2_round7_eval_goalhold.yaml')))
    cfg.pop('scenario_mix', None)
    cfg['environment'].update(n_agents=1, collision_physics=False, route_follow={'enabled': True})
    cfg['environment'].pop('sensor_noise', None)
    cfg.setdefault('obstacles', {})['enabled'] = False
    g = build_grid([WALL], ((-15.0, 15.0), (-15.0, 15.0)), res=0.25, inflate=0.5)
    route = simplify(g, plan(g, (-6.0, 0.0), (6.0, 0.0)))
    ad = _KinematicWebots(cfg)
    env = SwarmEnv(cfg, device=torch.device('cpu'), adapter=ad)
    spawns, goals = {'drone_0': [-6.0, 0.0, 2.0]}, {'drone_0': [6.0, 0.0, 2.0]}
    obs = env.start_episode(spawns, goals, obstacles=[WALL], routes={'drone_0': route})
    xt_max = 0.0
    for _ in range(1500):                                 # chase the carrot the adapter holds
        pos = ad.get_positions()[0].astype(np.float64)
        carrot = np.asarray(ad.goals['drone_0'], dtype=np.float64)
        assert np.allclose(env.goals[0], goals['drone_0'])                         # scoring: real goal
        heading = ad._heading.get('drone_0')
        if heading is not None and np.hypot(*(carrot[:2] - pos[:2])) > 0.3:       # LiDAR fan -> carrot
            want = math.atan2(carrot[1] - pos[1], carrot[0] - pos[0])
            assert abs((heading - want + math.pi) % (2 * math.pi) - math.pi) < 1e-4
        v = carrot - pos
        v *= min(1.0, 1.0 / max(np.linalg.norm(v), 1e-9))
        obs, _, done, info = env.step({'drone_0': [*v, 0.0]})
        xy = ad.get_positions()[0, :2]
        assert xy[1] > -2.0 + 8.0 or abs(xy[0]) > 0.4 + 0.15                      # never through the wall
        if done:
            break
    assert info['drone_reached_goal'][0]
    env.close()
