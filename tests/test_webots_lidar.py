"""C3: the optional Webots Lidar (envs/webots_lidar.py, envs/lidar_msg.py). No Webots needed."""
import ast
import math
import os
import struct

import numpy as np
import pytest

from envs import lidar as lidar_mod
from envs import lidar_msg as lm
from envs import radio_msg as rm
from envs import webots_lidar as wl
from envs.cf_velocity_pid import format_command, parse_command
from envs.webots_adapter import WebotsAdapter, _euler_from_matrix
from envs.webots_l2_multi import (WebotsL2MultiAdapter, drone_node, DRONE_NODE, DRONE_RADIO_NODE)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORLD = os.path.join(ROOT, 'webots', 'worlds', 'swarm_l2_multi.wbt')
ON = wl.lidar_webots_config({'lidar': {'enabled': True}})
RADIO = rm.radio_config({'radio': {'enabled': True, 'noise_m': 0.5, 'seed': 21, 'range_m': 150}})


# ---------------------------------------------------------------- config
def test_config_defaults_off():
    assert wl.lidar_webots_config({}) == wl.DEFAULTS and wl.DEFAULTS['enabled'] is False
    assert wl.lidar_webots_config(None)['enabled'] is False


@pytest.mark.parametrize('bad', [{'lidar': {'colour': 1}}, {'lidar': {'use_for_policy': True}},
                                 {'lidar': {'enabled': True, 'layers': 1}},
                                 {'lidar': {'enabled': True, 'horizontal_res': 4}}])
def test_config_rejects_bad_values(bad):
    with pytest.raises(ValueError):
        wl.lidar_webots_config(bad)


def test_lidar_refuses_draw_zones():
    with pytest.raises(ValueError, match='draw_zones'):
        wl.lidar_webots_config({'lidar': {'enabled': True}, 'draw_zones': True})
    assert wl.lidar_webots_config({'draw_zones': True})['enabled'] is False   # zones alone: fine


# ---------------------------------------------------------------- drone node, flag off unchanged
def test_flag_off_drone_node_unchanged():
    off = wl.lidar_webots_config({})
    for i in range(12):
        assert drone_node(i, None, off) == drone_node(i) == DRONE_NODE.format(i=i, x=2.0 * i)
        args = ' '.join(f'"{a}"' for a in rm.controller_args(RADIO))
        radio_only = DRONE_RADIO_NODE.format(i=i, x=2.0 * i, args=args, up=rm.UPLINK, down=rm.DOWNLINK,
                                             range=RADIO['range_m'])
        assert drone_node(i, RADIO, off) == drone_node(i, RADIO) == radio_only


def test_flag_on_drone_node_has_lidar():
    s = drone_node(3, None, ON, 6.0)
    assert s.startswith(DRONE_NODE.format(i=3, x=6.0)[:-2])
    for part in ('controllerArgs [ "--lidar" ]', 'name "lidar"', 'type "fixed"', 'projection "cylindrical"',
                 'fieldOfView 6.283185', 'verticalFieldOfView 3.14159', 'numberOfLayers 5',
                 'horizontalResolution 1440', 'near 0.3', 'minRange 0.3', 'maxRange 6.0',
                 'name "lidar_tx"', f'channel {lm.LIDAR_CHANNEL}', 'range -1'):
        assert part in s, part
    assert 'boundingObject' not in s and 'physics' not in s


def test_radio_and_lidar_node_keeps_the_radio():
    s = drone_node(2, RADIO, ON, 6.0)
    assert 'controllerArgs [ "--radio" "period_s=1.0" "noise_m=0.5" "seed=21" "--lidar" ]' in s
    for part in ('name "radio_tx"', f'channel {rm.UPLINK}', 'range 150.0', 'name "radio_rx"',
                 f'channel {rm.DOWNLINK}', 'name "lidar"', 'name "lidar_tx"'):
        assert part in s, part


# ---------------------------------------------------------------- controller args
def test_radio_args_parse_identical_with_and_without_lidar():
    args = rm.controller_args(RADIO)
    want = {'period_s': 1.0, 'noise_m': 0.5, 'seed': 21}
    assert rm.parse_controller_args(args) == want
    assert rm.parse_controller_args(args + ['--lidar']) == want
    assert rm.parse_controller_args(['--lidar'] + args) == want
    assert rm.parse_controller_args(args + ['--lidar', 'res=1440']) == want     # a later flag's k=v stay out
    assert rm.parse_controller_args(['--lidar']) is None and rm.parse_controller_args([]) is None
    with pytest.raises(ValueError):
        rm.parse_controller_args(['--radio', 'period_s=1.0', '--lidar', 'noise_m=0.5', 'seed=21'])


# ---------------------------------------------------------------- commands, tags, timer
def test_tag_round_trip_and_untagged():
    cmd = format_command(3, 'F', (0.1, -2.0, 1 / 3))
    assert lm.split_tag(cmd) == (cmd, None)
    raw, tag = lm.split_tag(cmd + ' ' + lm.format_tag(57, 8))
    assert raw == cmd and tag == (57, 8) and parse_command(raw) == parse_command(cmd)
    for bad in ('L5', 'Lx/8', 'L5/2'):
        with pytest.raises(ValueError):
            lm.split_tag(cmd + ' ' + bad)
    with pytest.raises(ValueError):                         # a tagged command without --lidar
        parse_command(cmd + ' L5/8')


def test_timer_enables_at_n_minus_2_reads_at_n_minus_1():
    t = lm.ScanTimer()
    got = [t.step((1, 8)) for _ in range(8)]                # j = 1 .. 8
    assert got == [None] * 5 + ['enable', 'read', None]
    got = [t.step((2, 8)) for _ in range(8)]                # a new tag restarts the count
    assert got == [None] * 5 + ['enable', 'read', None]
    assert t.step(None) is None and t.step((2, 8)) is None   # untagged resets; same tag counts again from 1
    t = lm.ScanTimer()
    assert [t.step((9, 3)) for _ in range(3)] == ['enable', 'read', None]


def test_scan_packet_round_trip_is_exact():
    img = np.random.default_rng(0).uniform(0.3, 6, (5, 1440)).astype(np.float32)
    img[0, :10] = np.inf
    b = lm.encode_scan(3, 7, 123, 12.344, (1.0, 2.0, 3.5), (0.01, -0.02, 1.5), 0.0012, img.tobytes())
    assert len(b) == lm.HEADER.size + img.nbytes and lm.HEADER.size == 76
    d = lm.decode_scan(b)
    assert (d['idx'], d['ep'], d['k'], d['t']) == (3, 7, 123, 12.344)
    assert d['pos'] == (1.0, 2.0, 3.5) and d['rpy'] == (0.01, -0.02, 1.5) and d['cpu'] == 0.0012
    assert np.array_equal(np.frombuffer(d['ranges'], np.float32).reshape(5, 1440), img)
    with pytest.raises(ValueError):
        lm.decode_scan(b[:50])


def test_lidar_msg_is_stdlib_only():
    tree = ast.parse(open(os.path.join(ROOT, 'envs', 'lidar_msg.py')).read())
    names = {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    names |= {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    assert names == {'struct'}


# ---------------------------------------------------------------- picking rays
def test_layer_and_column_angles_match_the_step0_probe():
    el = np.degrees(wl.layer_elevations(5))
    assert np.allclose(el, [89.9999, 44.99997, 0, -44.99997, -89.9999], atol=1e-3)
    az = np.degrees(wl.column_azimuths(1440))
    assert np.isclose(az[0], 179.875) and np.isclose(az[720], -0.125) and np.isclose(az[1080], -90.125)


@pytest.mark.parametrize('seed', range(5))
def test_every_analytic_ray_lands_on_its_nearest_lidar_ray(seed):
    rng = np.random.default_rng(seed)
    az_c, el_l = wl.column_azimuths(1440), wl.layer_elevations(5)
    for _ in range(50):
        heading, yaw = rng.uniform(-math.pi, math.pi, 2)
        dirs = wl.analytic_dirs(heading, 32)
        lay, col = wl.ray_index(dirs, yaw, 5, 1440)
        want_el = np.arcsin(dirs[:, 2])
        assert np.all(np.abs(el_l[lay] - want_el) < np.radians(0.01))
        horiz = np.abs(dirs[:, 2]) < 0.999                  # straight up / down: any column
        want_az = np.arctan2(dirs[:, 1], dirs[:, 0]) - yaw
        err = np.abs((az_c[col] - want_az + math.pi) % (2 * math.pi) - math.pi)
        assert np.all(err[horiz] <= np.radians(0.125) + 1e-9)


def test_rot_rpy_matches_the_adapter_euler_convention():
    rng = np.random.default_rng(1)
    for _ in range(100):
        rpy = rng.uniform([-0.5, -0.5, -math.pi], [0.5, 0.5, math.pi])
        assert np.allclose(_euler_from_matrix(wl.rot_rpy(rpy).ravel()), rpy, atol=1e-12)


def test_ground_inf_and_range_rules():
    img = np.full((5, 1440), 3.0, np.float32)
    dirs = wl.analytic_dirs(0.0, 0)                       # 14 fan rays
    level = (0.0, 0.0, 0.0)
    d = wl.pick_rays(img, dirs, np.array([0, 0, 2.0]), level, 6.0)
    down = [k for k, (_, e) in enumerate(lidar_mod.RAYS) if e < 0]
    other = [k for k, (_, e) in enumerate(lidar_mod.RAYS) if e >= 0]
    assert np.all(d[down] == 6.0) and np.all(d[other] == 3.0)       # 3 m down / at -45 from 2 m: below z = 0.05
    d = wl.pick_rays(img, dirs, np.array([0, 0, 5.0]), level, 6.0)
    assert np.all(d == 3.0)                                          # hits at z = 2 / 2.88: not the ground
    img[:] = np.inf
    assert np.all(wl.pick_rays(img, dirs, np.array([0, 0, 3.0]), level, 6.0) == 6.0)
    img[:] = 6.5
    assert np.all(wl.pick_rays(img, dirs, np.array([0, 0, 9.0]), level, 6.0) == 6.0)


def test_ground_rule_uses_the_tilted_pose():
    img = np.full((5, 1440), 4.0, np.float32)
    dirs = wl.analytic_dirs(0.0, 0)[:1]                    # the front ray, horizontal in the world
    # nose 50 deg down (pitch > 0 is nose down in ENU / FLU): the body-horizontal ray goes to the ground
    d = wl.pick_rays(img, dirs, np.array([0, 0, 3.0]), (0.0, math.radians(50), 0.0), 6.0)
    assert d[0] == 6.0
    d = wl.pick_rays(img, dirs, np.array([0, 0, 3.0]), (0.0, math.radians(-50), 0.0), 6.0)
    assert d[0] == 4.0


# ---------------------------------------------------------------- adapter, without Webots
class Field:
    def __init__(self):
        self.v = []

    def setSFString(self, s):
        self.v.append(s)


def fake_adapter(lidar=None):
    ad = WebotsL2MultiAdapter.__new__(WebotsL2MultiAdapter)
    ad.n_agents, ad.v_max, ad.ep, ad.n_sub = 2, 2.0, 4, 8
    ad._parked = np.zeros(2, bool)
    ad._pending = set()
    ad._cmd = np.zeros((2, 3))
    ad._cd = [Field(), Field()]
    if lidar:
        ad._lrx, ad.lidar, ad._scan_k = object(), lidar, 10
    return ad


def test_flag_off_commands_unchanged():
    ad = fake_adapter()
    ad.apply_actions({'drone_0': [0.1, -2.0, 1 / 3, 0.0], 'drone_1': [0, 0, 0, 0]})
    assert ad._cd[0].v == ['4 F 0.1 -2.0 0.3333333333333333'] and ad._cd[1].v == ['4 F 0.0 0.0 0.0']


def test_flag_on_commands_carry_the_tag():
    ad = fake_adapter(ON)
    ad.apply_actions({'drone_0': [0.1, -2.0, 1 / 3, 0.0], 'drone_1': [0, 0, 0, 0]})
    assert ad._cd[0].v == ['4 F 0.1 -2.0 0.3333333333333333 L11/8'] and ad._scan_k == 11


class Sup:
    def getTime(self):
        return 5.0


def scan_adapter(policy):
    cfg = {'lidar': {'enabled': True, 'range': 6.0, 'sees_drones': True, 'ring_rays': 32},
           'webots': {'lidar': {'enabled': True, 'use_for_policy': policy}}}
    ad = fake_adapter(wl.lidar_webots_config(cfg['webots']))
    ad.lidar_cfg = lidar_mod.lidar_config(cfg)
    ad.n_lidar, ad.sup = 14, Sup()
    ad._pos = np.array([[0.0, 0.0, 3.0], [3.0, 0.0, 3.0]])
    ad._orn = np.zeros((2, 3))
    ad.goals = {'drone_0': np.array([10.0, 0, 3]), 'drone_1': np.array([3.0, 10, 3])}
    ad._heading, ad.layout, ad._logged = {}, [], None
    ad._scan_log = wl.ScanLog(None)
    img = np.full((5, 1440), 2.5, np.float32)
    ad._scans = [{'img': img, 'pos': (0.0, 0.0, 2.99), 'rpy': (0.0, 0.0, 0.0), 't': 4.996, 'cpu': 1e-4}, None]
    return ad


def test_compare_mode_returns_the_analytic_scan_unchanged_and_logs_both():
    ad = scan_adapter(policy=False)
    want = WebotsAdapter._scan_lidar(scan_adapter(policy=False))
    got = ad._scan_lidar()
    assert got == want                                      # the same floats, bit for bit
    rows = ad._scan_log.rows
    assert len(rows['ep']) == 2 and ad._scan_log.stats['missing'] == 1
    assert np.all(rows['lidar'][0][[0, 1, 2, 3, 4]] == 2.5)
    ad._scan_lidar()                                        # same control step: not logged twice
    assert len(rows['ep']) == 2


def test_policy_mode_uses_the_lidar_and_needs_every_scan():
    ad = scan_adapter(policy=True)
    with pytest.raises(RuntimeError, match='no Lidar scan from drone_1'):
        ad._scan_lidar()
    ad = scan_adapter(policy=True)
    ad._parked[1] = True                                    # parked: range everywhere, no error
    scans, rings = ad._scan_lidar()
    assert scans[0][0] == 2.5 and scans[1] == [6.0] * 14 and len(rings[0]) == 32


# ---------------------------------------------------------------- world
def test_world_lidar_receiver_is_inert_by_default():
    w = open(WORLD).read()
    sup = w.split('DEF SUPERVISOR')[1].split('DEF DRONES')[0]
    assert 'name "lidar_rx"' in sup and f'channel {lm.LIDAR_CHANNEL}' in sup
    code = '\n'.join(ln.split('#')[0] for ln in sup.splitlines())
    assert 'physics' not in code and 'boundingObject' not in code
    tree = ast.parse(open(os.path.join(ROOT, 'envs', 'webots_l2_multi.py')).read())
    init = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == '__init__')
    guard = next(n for n in init.body if isinstance(n, ast.If) and "self.lidar['enabled']" == ast.unparse(n.test))
    outside = [ast.unparse(n) for n in init.body if n is not guard]
    assert "getDevice('lidar_rx')" in ast.unparse(guard) and not any("lidar_rx" in u for u in outside)
