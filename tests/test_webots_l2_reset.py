"""Webots Level 2 single-drone safe reset (envs/webots_l2.py), tested without Webots:
the shared PID module, the rotor-speed estimate, the spin-up hold in reset_world and the start check."""
import ast
import os

import numpy as np
import pytest

from envs import cf_velocity_pid as cf
from envs.webots_l2 import WebotsL2Adapter, HOVER_TOL, MAX_HOLD_S, START_SPEED_LIMIT, START_CHECK_STEPS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------- shared PID module
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


def test_pid_module_is_stdlib_only():
    tree = ast.parse(open(os.path.join(ROOT, 'envs', 'cf_velocity_pid.py')).read())
    names = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            names |= {a.name for a in n.names}
        elif isinstance(n, ast.ImportFrom):
            names.add(n.module)
    assert names == {'math'}


# ---------------------------------------------------------------- fakes (no Webots)
class _Supervisor:
    def __init__(self):
        self.steps = 0

    def step(self, ms):
        self.steps += 1
        return 0


class _Field:
    def setSFVec3f(self, v): pass
    def setSFRotation(self, v): pass


class _Node:
    def resetPhysics(self): pass
    def setVelocity(self, v): pass


class _Motor:
    def setVelocity(self, v): pass


class _GPS:
    def getValues(self): return [0.0, 0.0, 1.0]


def _adapter(rate=30.0, start=cf.HOVER_RAD_S, target=None, hold_s=2.0):
    ad = WebotsL2Adapter.__new__(WebotsL2Adapter)
    ad.sup, ad.basic_ms, ad.dt, ad.spin_rate = _Supervisor(), 4, 0.004, rate
    ad._rotor_est = np.full(4, float(start))
    ad._rotor_target = np.full(4, float(start if target is None else target))
    ad.spinup_hold_s, ad.episode, ad.goals = hold_s, 0, {}
    ad._tr, ad._rot, ad.node = [_Field()], [_Field()], _Node()
    ad.motors, ad.gps = [_Motor() for _ in range(4)], _GPS()
    ad.pid, ad.cmd = cf.VelocityPID(), np.zeros(3)
    ad._build_obstacles = lambda obstacles: None
    ad._read_state = lambda: None
    ad._vel = np.zeros((1, 3))
    return ad


def _reset(ad):
    ad.reset_world({'drone_0': [0.0, 0.0, 1.0]}, {'drone_0': [5.0, 0.0, 1.0]})


# ---------------------------------------------------------------- rotor estimate
@pytest.mark.parametrize('rate, start, target', [(30.0, cf.HOVER_RAD_S, 100.0), (30.0, 600.0, cf.HOVER_RAD_S),
                                                 (1000.0, cf.HOVER_RAD_S, 100.0), (1000.0, 600.0, 0.0)])
def test_rotor_estimate_ramps_at_spin_rate(rate, start, target):
    """Each basic step the estimate moves towards the command by at most spin_rate * dt
    (measured in Webots: exact for both models), then stays there."""
    ad = _adapter(rate, start, target)
    n = int(abs(target - start) / (rate * ad.dt))              # whole steps strictly before arrival
    for k in range(1, n + 1):
        ad._step_basic()
        assert ad._rotor_est == pytest.approx(np.full(4, start + np.sign(target - start) * rate * ad.dt * k))
    for _ in range(3):                                         # arrives and does not overshoot
        ad._step_basic()
    assert ad._rotor_est == pytest.approx(np.full(4, target))
    assert ad.sup.steps == n + 3


def test_set_motors_records_absolute_targets():
    ad = _adapter()
    ad._set_motors([-60.0, 61.0, -62.0, 63.0])
    assert list(ad._rotor_target) == [60.0, 61.0, 62.0, 63.0]


# ---------------------------------------------------------------- spin-up hold
def test_hold_from_hover_is_exactly_spinup_hold_s():
    ad = _adapter(start=cf.HOVER_RAD_S)
    _reset(ad)
    assert ad.last_hold_s == pytest.approx(2.0)
    assert ad.sup.steps == 500 + 1                             # hold + the release step


def test_hold_after_a_crash_waits_for_the_rotors():
    """crash_l2 in Webots: rotors near 582 rad/s after a crash; the official model slows at 30 rad/s^2."""
    ad = _adapter(rate=30.0, start=582.0)
    _reset(ad)
    expect = (582.0 - cf.HOVER_RAD_S - HOVER_TOL) / 30.0       # ~17.5 s
    assert ad.last_hold_s == pytest.approx(expect, abs=ad.dt)
    assert ad.last_hold_s == pytest.approx(17.5, abs=0.1)
    assert np.abs(ad._rotor_est - cf.HOVER_RAD_S).max() <= HOVER_TOL


def test_hold_longer_than_max_raises():
    ad = _adapter(rate=30.0, start=1000.0)                      # ~31.4 s to hover > 30 s
    with pytest.raises(RuntimeError, match='not at hover speed'):
        _reset(ad)
    assert ad.sup.steps == int(round(MAX_HOLD_S / ad.dt))


def test_reset_counts_episodes():
    ad = _adapter()
    _reset(ad); _reset(ad)
    assert ad.episode == 2


# ---------------------------------------------------------------- start check
def test_start_check_passes_a_slow_start():
    ad = _adapter(); ad._vel[0] = [0.3, 0.3, 0.3]
    ad._start_check()


def test_start_check_message_names_episode_and_speed():
    ad = _adapter(); ad.episode = 7
    ad._vel[0] = [0.0, 0.0, 3.4]
    with pytest.raises(RuntimeError) as e:
        ad._start_check()
    msg = str(e.value)
    assert 'episode 7' in msg and '3.4 m/s' in msg
    assert f'{START_SPEED_LIMIT} m/s' in msg and f'{START_CHECK_STEPS} control steps' in msg
