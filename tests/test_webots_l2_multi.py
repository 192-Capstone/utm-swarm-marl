"""Webots Level 2 multi-drone pieces that can be tested without Webots:
the customData command format, the shared PID module, the controller's imports, and the
adapter's drop check, log check and episode stats (envs/webots_l2_multi.py)."""
import ast
import math
import os

import numpy as np
import pytest

from envs import cf_velocity_pid as cf
from envs.webots_l2_multi import WebotsL2MultiAdapter, DROP_LIMIT

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONTROLLER = os.path.join(ROOT, 'webots', 'controllers', 'cf_velocity', 'cf_velocity.py')


# ---------------------------------------------------------------- command format
@pytest.mark.parametrize('v', [(0.0, 0.0, 0.0), (0.1, -2.0, 1e-17), (2.0, 2.0, -2.0),
                               (np.float32(0.3), np.float64(-1.25), 1 / 3)])
@pytest.mark.parametrize('mode', ['H', 'F'])
def test_command_round_trip_is_exact(mode, v):
    ep, m, back = cf.parse_command(cf.format_command(7, mode, v))
    assert (ep, m) == (7, mode)
    assert back == tuple(float(x) for x in v)          # exact, not approximate


@pytest.mark.parametrize('s', ['', '   ', '\n'])
def test_empty_command_is_idle(s):
    assert cf.parse_command(s) is None


@pytest.mark.parametrize('s', ['3 F 0 0', '3 F 0 0 0 0', '3 X 0 0 0', 'x F 0 0 0', '3 F a 0 0'])
def test_malformed_command_raises(s):
    with pytest.raises(ValueError):
        cf.parse_command(s)


def test_format_rejects_unknown_mode():
    with pytest.raises(ValueError):
        cf.format_command(1, 'P', (0.0, 0.0, 0.0))


# ---------------------------------------------------------------- shared PID
def test_single_drone_level2_uses_the_shared_pid():
    import envs.webots_l2 as l2
    assert l2.VelocityPID is cf.VelocityPID
    assert l2.HOVER_RAD_S == cf.HOVER_RAD_S


def test_hover_preload_gives_hover_motor_speed():
    """With the pre-load and no errors, all four motors get HOVER_RAD_S (the episode starts in hover)."""
    assert cf.HOVER_INTEGRATOR == pytest.approx((cf.HOVER_RAD_S - 48.0) / 5)
    pid = cf.VelocityPID()
    pid.reset(integrator=cf.HOVER_INTEGRATOR)
    m = pid(0.004, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0)
    assert m == pytest.approx([cf.HOVER_RAD_S] * 4)


def _imports(path):
    tree = ast.parse(open(path).read())
    names = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            names |= {a.name for a in n.names}
        elif isinstance(n, ast.ImportFrom):
            names.add(n.module)
    return names


def test_controller_imports_only_controller_and_stdlib():
    assert _imports(CONTROLLER) == {'ctypes', 'math', 'os', 'sys', 'time', 'controller', 'envs.cf_velocity_pid',
                                    'envs.radio_msg', 'envs.lidar_msg'}


def test_pid_module_is_stdlib_only():
    assert _imports(os.path.join(ROOT, 'envs', 'cf_velocity_pid.py')) == {'math'}


# ---------------------------------------------------------------- adapter checks
def _adapter(n=3, z=1.0, log_path=None):
    ad = WebotsL2MultiAdapter.__new__(WebotsL2MultiAdapter)   # no Webots
    ad.n_agents, ad.n_sub, ad.dt, ad.ep = n, 8, 0.004, 1
    ad._z_spawn = np.full(n, z)
    ad._z_ref = np.full(n, z)
    ad._cmd = np.zeros((n, 3))
    ad._pos = np.zeros((n, 3)); ad._pos[:, 2] = z
    ad._vel = np.zeros((n, 3))
    ad._reimported_now = []
    ad.log_path, ad._log_pos = log_path or '/nonexistent', 0
    return ad


def test_drop_check_passes_hovering_drones():
    _adapter()._drop_check()


def test_drop_check_ignores_a_healthy_drone_lagging_behind_a_climb():
    ad = _adapter()
    ad._cmd[1, 2] = 2.0                         # climb at 2 m/s: the setpoint runs ahead
    for _ in range(10):                         # 0.32 s: setpoint 0.64 m above the spawn
        ad._drop_check()                        # drone still at the spawn height: no alarm


def test_drop_check_names_the_falling_drone():
    ad = _adapter()
    ad._pos[2, 2] -= DROP_LIMIT + 0.05          # drone_2 fell below spawn and setpoint
    with pytest.raises(RuntimeError, match='drone_2'):
        ad._drop_check()


def test_drop_check_follows_a_descent_command():
    ad = _adapter()
    ad._cmd[0, 2] = -1.0                        # commanded descent: setpoint goes down
    ad._pos[0, 2] = 1.0 - 0.032 - 0.2           # following it, within the limit
    ad._drop_check()


def test_start_check_passes_a_slow_start():
    ad = _adapter(); ad._vel = np.full((3, 3), 0.3)        # |v| = 0.52 m/s
    ad._start_check()


def test_start_check_names_the_fast_drone():
    ad = _adapter(); ad._vel = np.zeros((3, 3)); ad._vel[1, 2] = 31.0
    with pytest.raises(RuntimeError, match='drone_1'):
        ad._start_check()


@pytest.mark.parametrize('tilt_deg, z, broken', [(180.0, 0.01, True), (70.0, 2.0, True),
                                                 (20.0, 0.05, True), (20.0, 1.5, False)])
def test_broken_drones(tilt_deg, z, broken):
    ad = _adapter(n=2); ad._reset_stats()
    ad._max_roll[0] = math.radians(tilt_deg); ad._pos[0, 2] = z
    assert ad._broken_drones() == ([0] if broken else [])


def test_log_check(tmp_path):
    log = tmp_path / 'webots.log'
    log.write_text('INFO: world loaded\n')
    ad = _adapter(log_path=str(log))
    ad._check_controller_log()                  # no error line
    with open(log, 'a') as f:
        f.write("cf_velocity [drone_4]: bad command '1 Q 0 0 0'\n")
    with pytest.raises(RuntimeError, match=r'drone_4'):
        ad._check_controller_log()
    ad._check_controller_log()                  # the same line is not reported twice


def test_log_check_without_a_log_file():
    _adapter()._check_controller_log()          # missing file: nothing to check


def test_episode_stats_in_degrees():
    ad = _adapter(n=2)
    ad._reset_stats()
    ad._max_roll[:] = [math.radians(30), 0.0]
    ad._max_pitch[:] = [0.0, math.radians(61)]
    ad._max_speed[:] = [1.5, 3.2]
    st = ad.episode_stats()
    assert st['max_roll_deg'] == pytest.approx([30.0, 0.0])
    assert st['max_pitch_deg'] == pytest.approx([0.0, 61.0])
    assert st['max_speed'] == [1.5, 3.2]


# ---------------------------------------------------------------- lost-control rule (eval_suite)
def _stats(speed, roll=0.0, pitch=0.0):
    return {'max_speed': [speed], 'max_roll_deg': [roll], 'max_pitch_deg': [pitch]}


@pytest.mark.parametrize('speed, roll, pitch, lost', [
    (3.2, 0.0, 0.0, False),       # fast but level: speed is no longer part of the rule
    (2.9, 30.0, 30.0, False),     # below both limits
    (1.0, 0.0, 61.0, True),       # pitch over 60 deg
    (1.0, 60.0, 0.0, False),      # exactly 60 deg: the rule is "> 60"
    (426.0, 180.0, 0.0, True),    # a flip (the speed after a flip is not physical)
])
def test_lost_control_rule(speed, roll, pitch, lost):
    from eval_suite import lost_control
    assert lost_control(_stats(speed, roll, pitch), 0) is lost


# ---------------------------------------------------------------- single-drone rotor estimate
class _FakeSupervisor:
    def __init__(self):
        self.steps = 0

    def step(self, ms):
        self.steps += 1
        return 0


@pytest.mark.parametrize('rate, start, target', [(30.0, cf.HOVER_RAD_S, 100.0), (30.0, 600.0, cf.HOVER_RAD_S),
                                                 (1000.0, cf.HOVER_RAD_S, 100.0), (1000.0, cf.HOVER_RAD_S, 0.0)])
def test_rotor_estimate_ramps_at_spin_rate(rate, start, target):
    """WebotsL2Adapter._step_basic: each basic step the estimate moves towards the command by at
    most spin_rate * dt (measured in Webots: exact for both models), then stays there."""
    from envs.webots_l2 import WebotsL2Adapter
    ad = WebotsL2Adapter.__new__(WebotsL2Adapter)             # no Webots
    ad.sup, ad.basic_ms, ad.dt, ad.spin_rate = _FakeSupervisor(), 4, 0.004, rate
    ad._rotor_est = np.full(4, start)
    ad._rotor_target = np.full(4, target)
    n = int(abs(target - start) / (rate * ad.dt))              # whole steps strictly before arrival
    for k in range(1, n + 1):
        ad._step_basic()
        expect = start + np.sign(target - start) * rate * ad.dt * k
        assert ad._rotor_est == pytest.approx(np.full(4, expect))
    for _ in range(3):                                         # arrives and does not overshoot
        ad._step_basic()
    assert ad._rotor_est == pytest.approx(np.full(4, target))
    assert ad.sup.steps == n + 3


def test_episode_stats_reports_reimported_drones():
    ad = _adapter(n=3); ad._reset_stats(); ad._reimported_now = [2]
    assert ad.episode_stats()['reimported'] == [False, False, True]
