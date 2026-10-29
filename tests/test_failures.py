"""
test_failures.py
----------------
D5 drone failures (environment.failures in envs/swarm_env.py, envs/disturbances.py,
envs/pybullet_adapter.py set_falling) and the telemetry side (utm/telemetry.py status, mute).

Two drones fly 6 m at 4 m height under a P-controller towards their adapter goal (the carrot,
as the policy and SimplePilot do); drone_0 gets the failure.
  motor  it falls (vz down to -g t), crashes on the ground or the roof under it; drone_1 still
         arrives, the episode ends as a crash episode; emergency_report follows p_emergency_report
  land   its carrot goes straight down to land_z above the ground or the roof, |vz| <= land_vz;
         settled there it is landed + forced_down (not arrived), parked, the episode ends
  comms  it flies on and arrives; set_route is refused
  plus: skipped when not flying, schedule checks, failures on without a schedule = off (bar
  info keys), draw_failures, roof_under, TelemetryModel status / mute, RadioTelemetry status.

Run: python -m pytest tests/test_failures.py -q
"""

import os
import sys

import numpy as np
import pytest
import torch
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from envs import disturbances as dist                              # noqa: E402
from envs.swarm_env import SwarmEnv                                # noqa: E402
from utm.telemetry import RadioTelemetry, Report, TelemetryModel   # noqa: E402

CONFIG = os.path.join(os.path.dirname(__file__), '..', 'configs', 'config_stage2_round7_teacher.yaml')
SPAWNS = {'drone_0': [0.0, 0.0, 4.0], 'drone_1': [6.0, 0.0, 4.0]}
GOALS = {'drone_0': [0.0, 6.0, 4.0], 'drone_1': [6.0, -6.0, 4.0]}
BOX = {'type': 'box', 'center': [0.0, 1.5, 0.0], 'half': (1.0, 1.0), 'height': 2.5, 'yaw': 0.0}
# a 12 m wall whose face is 0.3 m east of drone_0's track (x = 0, y 0 -> 6)
WALL = {'type': 'box', 'center': [0.8, 3.0, 0.0], 'half': (0.5, 3.0), 'height': 12.0, 'yaw': 0.0}
# a long thin building turned 45 degrees, under drone_0's track at y ~ 1
ROT = {'type': 'box', 'center': [0.0, 1.0, 0.0], 'half': (1.5, 0.3), 'height': 2.0, 'yaw': np.pi / 4}


def make_env(failures=True, crash_mode='remove', schedule=False, **fcfg):
    cfg = yaml.safe_load(open(CONFIG, encoding='utf-8'))
    cfg.pop('scenario_mix', None)
    cfg['environment'].update(n_agents=2, collision_physics=False, crash_mode=crash_mode)
    cfg['environment'].pop('sensor_noise', None)
    cfg['environment']['dynamics'] = {'lag': False}
    cfg.setdefault('obstacles', {})['enabled'] = False
    if failures:
        cfg['environment']['failures'] = {'enabled': True, **fcfg}
    if schedule:
        cfg['environment']['schedule'] = {'enabled': True}
    return SwarmEnv(cfg, device=torch.device('cpu'))


def carrot_pilot(env, v_max=1.0):
    pos = env.adapter.get_positions()
    out = {}
    for i in range(env.n_agents):
        v = np.clip(1.5 * (np.asarray(env.adapter.goals[f'drone_{i}']) - pos[i]), -v_max, v_max)
        out[f'drone_{i}'] = [*v, 0.0]
    return out


def fly(env, steps=600, record=None):
    info, done = {}, False
    for k in range(steps):
        acts = carrot_pilot(env)
        if record is not None:
            record.append((env.adapter.get_positions().copy(), acts))
        _, _, done, info = env.step(acts)
        if done:
            break
    return k + 1, done, info


# ---------------------------------------------------------------- motor
def test_motor_failure_falls_and_crashes():
    env = make_env()
    env.start_episode(SPAWNS, GOALS, obstacles=[], failures=[{'drone': 'drone_0', 't': 1.0, 'type': 'motor'}])
    dt = env.control_dt
    z = []
    for k in range(int(1.0 / dt) + 60):
        _, _, done, info = env.step(carrot_pilot(env))
        if not env.crashed[0]:
            z.append(float(env.adapter.get_positions()[0][2]))
        if env.crashed[0]:
            break
    assert env.failed[0] == 'motor' and env.crashed[0]
    assert env.failure_t[0] == pytest.approx(1.0)
    assert env.emergency_report[0]                       # p_emergency_report defaults to 1
    fall_s = (k + 1) * dt - 1.0
    assert 0.75 < fall_s < 1.05                          # ~sqrt(2 * 3.85 / 9.81) = 0.89 s
    assert z[-1] < 1.0
    assert info['drone_failed'] == ['motor', ''] and info['drone_crashed'] == [True, False]
    assert env.adapter.get_positions()[0][0] > 1e4      # parked
    steps, done, info = fly(env)
    assert done and info['reason'] == 'collision' and info['drone_reached_goal'] == [False, True]


def test_motor_failure_ends_on_a_roof():
    env = make_env()
    env.start_episode(SPAWNS, GOALS, obstacles=[BOX], failures=[{'drone': 'drone_0', 't': 0.9, 'type': 'motor'}])
    zs = []
    for _ in range(80):
        env.step(carrot_pilot(env))
        if env.crashed[0]:
            break
        zs.append(env.adapter.get_positions()[0])
    assert env.crashed[0]
    assert 2.0 < zs[-1][2] < 3.6 and abs(zs[-1][1] - 1.5) < 1.0   # stopped by the 2.5 m roof


@pytest.mark.parametrize('p', [0.0, 1.0])
def test_emergency_report_probability(p):
    env = make_env(p_emergency_report=p)
    env.start_episode(SPAWNS, GOALS, obstacles=[], failures=[{'drone': 'drone_0', 't': 0.1, 'type': 'motor'}])
    for _ in range(5):
        env.step(carrot_pilot(env))
    assert env.emergency_report[0] == bool(p)


# ---------------------------------------------------------------- forced landing
@pytest.mark.parametrize('layout, z_land', [([], 1.0),          # open ground
                                            ([BOX], 3.5),       # over a roof: on the roof
                                            ([WALL], 1.0),      # 0.3 m beside a 12 m wall: the ground
                                            ([ROT], 3.0)])      # over a rotated building: its roof
def test_forced_landing_where_it_is(layout, z_land):
    env = make_env(land_vz=0.5)
    env.start_episode(SPAWNS, GOALS, obstacles=layout, failures=[{'drone': 'drone_0', 't': 1.0, 'type': 'land'}])
    rec = []
    steps, done, info = fly(env, record=rec)
    assert done and info['reason'] == 'success'           # everyone settled, landed or done; none crashed
    assert env.failed[0] == 'land' and env.forced_down[0] and env.landed[0] and not env.crashed[0]
    assert info['drone_reached_goal'] == [False, True] and info['drone_forced_down'] == [True, False]
    target = env._forced_target[0]
    assert target[2] == pytest.approx(z_land)
    k_fail = int(round(1.0 / env.control_dt))
    assert target[:2] == pytest.approx(rec[k_fail][0][0][:2], abs=0.05)   # where it was at t = 1 s
    vz = [a['drone_0'][2] for _, a in rec[k_fail + 1:]]
    # (the pilot asked for up to 1 m/s down; the env clipped it before the adapter: the drone's
    # true vertical speed is what shows it, read from consecutive positions)
    z = np.array([p[0][2] for p, _ in rec[k_fail + 1:] if p[0][0] < 1e4])
    assert max(vz) <= 1.0 and min(np.diff(z)) / env.control_dt >= -0.5 - 1e-3
    env.step(carrot_pilot(env))                           # (parked from the step after it landed)
    assert env.adapter.get_positions()[0][0] > 1e4


def test_failed_drones_refuse_new_routes():
    env = make_env()
    env.start_episode(SPAWNS, GOALS, obstacles=[], failures=[{'drone': 'drone_0', 't': 0.5, 'type': 'comms'}])
    env.followers, env._cruise, env._climbed = {}, {}, {}
    for _ in range(int(0.5 / env.control_dt) + 1):
        env.step(carrot_pilot(env))
    assert env.comms_lost[0] and env.failed[0] == 'comms'
    assert env.set_route(0, [(0, 0, 4), (0, 3, 4), (0, 6, 4)]) is False
    assert env.set_route(1, [(6, 0, 4), (6, -3, 4), (6, -6, 4)]) is True


# ---------------------------------------------------------------- comms loss
def test_comms_loss_flies_on():
    env = make_env()
    env.start_episode(SPAWNS, GOALS, obstacles=[], failures=[{'drone': 'drone_0', 't': 1.0, 'type': 'comms'}])
    steps, done, info = fly(env)
    assert done and info['success'] and info['drone_comms_lost'] == [True, False]
    assert info['drone_reached_goal'] == [True, True]


# ---------------------------------------------------------------- schedule rules
def test_failure_of_a_drone_not_flying_is_skipped():
    env = make_env(schedule=True)
    f = {'drone': 'drone_1', 't': 0.5, 'type': 'motor'}
    env.start_episode(SPAWNS, GOALS, obstacles=[], departures={'drone_1': 2.0}, failures=[f])
    for _ in range(int(1.0 / env.control_dt)):
        env.step(carrot_pilot(env))
    assert env.failed == ['', ''] and env.failures_skipped == [f]


@pytest.mark.parametrize('bad', [[{'drone': 'drone_5', 't': 1.0, 'type': 'motor'}],
                                 [{'drone': 'drone_0', 't': 1.0, 'type': 'fire'}],
                                 [{'drone': 'drone_0', 't': -1.0, 'type': 'land'}],
                                 [{'drone': 'drone_0', 't': 1.0, 'type': 'land'},
                                  {'drone': 'drone_0', 't': 2.0, 'type': 'motor'}]])
def test_bad_schedules_are_refused(bad):
    env = make_env()
    with pytest.raises(ValueError):
        env.start_episode(SPAWNS, GOALS, obstacles=[], failures=bad)


def test_needs_enabled_and_remove():
    env = make_env(failures=False)
    with pytest.raises(ValueError):
        env.start_episode(SPAWNS, GOALS, obstacles=[], failures=[])
    with pytest.raises(ValueError):
        make_env(crash_mode='end')
    with pytest.raises(ValueError):
        make_env(land_vz=0.0)
    with pytest.raises(ValueError):
        make_env(p_emergency_report=1.5)


def test_enabled_without_schedule_changes_only_info():
    def run(env):
        np.random.seed(3)
        obs = [env.start_episode(SPAWNS, GOALS, obstacles=[])]
        rew, infos = [], []
        for _ in range(60):
            o, r, d, i = env.step(carrot_pilot(env))
            obs.append(o)
            rew.append(r)
            infos.append(i)
        return obs, rew, infos
    a = run(make_env(failures=False))
    b = run(make_env(failures=True))
    for oa, ob in zip(a[0], b[0]):
        assert all(torch.equal(oa[k], ob[k]) for k in oa)
    assert a[1] == b[1]
    for ia, ib in zip(a[2], b[2]):
        extra = {k: ib.pop(k) for k in ('drone_failed', 'drone_forced_down', 'drone_comms_lost')}
        assert ia == ib and extra['drone_failed'] == ['', '']


def _same(a, b):
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    if isinstance(a, torch.Tensor):
        return torch.equal(a, b)
    if isinstance(a, np.ndarray):
        return np.array_equal(a, b)
    return a == b or (a != a and b != b)


@pytest.mark.parametrize('lag', [False, True])
def test_disabled_block_is_identical(lag):
    """failures: {enabled: false, ...} = no block: every step's obs, rewards, done, info and the
    np.random state after the run (with and without the velocity lag, which draws from it)."""
    def run(block):
        cfg = yaml.safe_load(open(CONFIG, encoding='utf-8'))
        cfg.pop('scenario_mix', None)
        cfg['environment'].update(n_agents=2, collision_physics=False, crash_mode='remove')
        cfg['environment'].pop('sensor_noise', None)
        cfg['environment']['dynamics'] = {'lag': lag}
        cfg.setdefault('obstacles', {})['enabled'] = False
        if block is not None:
            cfg['environment']['failures'] = block
        np.random.seed(7)
        torch.manual_seed(7)
        env = SwarmEnv(cfg, device=torch.device('cpu'))
        out = [env.start_episode(SPAWNS, GOALS, obstacles=[])]
        for _ in range(60):
            out.append(env.step(carrot_pilot(env)))
        env.close()
        return out, np.random.get_state()[1].copy()
    a, sa = run(None)
    b, sb = run({'enabled': False, 'land_vz': 0.3, 'p_emergency_report': 0.0})
    assert _same(a, b) and np.array_equal(sa, sb)


# ---------------------------------------------------------------- helpers
def test_draw_failures_is_seeded_and_mid_flight():
    flights = [(f'drone_{i}', 10.0 * i, 10.0 * i + 50.0) for i in range(40)]
    a = dist.draw_failures(np.random.RandomState(1), flights, 0.3)
    b = dist.draw_failures(np.random.RandomState(1), flights, 0.3)
    assert a == b and 4 < len(a) < 25
    for f in a:
        i = int(f['drone'].split('_')[1])
        assert 10.0 * i + 10.0 <= f['t'] <= 10.0 * i + 40.0
        assert f['type'] in dist.FAILURE_TYPES
    assert dist.draw_failures(np.random.RandomState(1), flights, 0.0) == []
    every = dist.draw_failures(np.random.RandomState(1), flights, 1.0, types=('land',))
    assert len(every) == 40 and {f['type'] for f in every} == {'land'}
    # one flight's draw doesn't depend on the others' (3 numbers per flight)
    short = dist.draw_failures(np.random.RandomState(1), flights[:10], 0.3)
    assert short == [f for f in a if int(f['drone'].split('_')[1]) < 10]
    with pytest.raises(ValueError):
        dist.draw_failures(np.random.RandomState(1), flights, 1.5)


def test_roof_under():
    assert dist.roof_under([BOX], 0.0, 1.5) == 2.5
    assert dist.roof_under([BOX], 0.99, 1.5) == 2.5             # just inside the edge
    assert dist.roof_under([BOX], 1.3, 1.5) == 0.0              # 0.3 m beside it: the ground
    assert dist.roof_under([], 0.0, 0.0) == 0.0
    assert dist.roof_under([WALL], 0.0, 3.0) == 0.0             # 0.3 m beside a 12 m wall (review, PR #39)
    assert dist.roof_under([WALL], 0.25, 3.0) == 0.0            # 0.05 m beside it
    assert dist.roof_under([WALL], 0.35, 3.0) == 12.0           # 0.05 m inside it
    assert dist.roof_under([WALL, BOX], 0.5, 1.5) == 12.0       # on both footprints: the higher roof
    pillar = {'type': 'pillar', 'center': [5.0, 5.0, 0.0], 'radius': 1.0, 'height': 6.0}
    assert dist.roof_under([pillar], 5.7, 5.7) == 6.0           # 0.99 m from the axis
    assert dist.roof_under([pillar], 5.8, 5.8) == 0.0           # 1.13 m


def test_roof_under_rotated_building():
    """ROT turned 45 degrees: a point on its long diagonal is inside although the unturned box
    would not contain it, and a point the unturned box would contain is outside."""
    assert dist.roof_under([ROT], 1.0, 2.0) == 2.0              # (1, 1) from the centre, along its length
    assert dist.roof_under([ROT], 1.4, 1.0) == 0.0              # (1.4, 0): inside the unturned box only
    assert dist.roof_under([ROT], -0.9, 0.1) == 2.0
    assert dist.roof_under([ROT], 0.9, 0.1) == 0.0


# ---------------------------------------------------------------- telemetry
def test_telemetry_status_and_mute():
    tm = TelemetryModel(noise_m=0.5, loss=0.2, seed=4)
    ref = TelemetryModel(noise_m=0.5, loss=0.2, seed=4)
    states = {'f1': ((0, 0, 4), (1, 0, 0), []), 'f2': ((5, 0, 4), (0, 1, 0), [], 'emergency')}
    tm.mute('f1')
    for t in range(20):
        tm.send(float(t), states)
        ref.send(float(t), {'f2': states['f2']})
    got = tm.receive(100.0)
    exp = ref.receive(100.0)
    assert all(r.id == 'f2' and r.status == 'emergency' for r in got)
    assert [(r.seq, tuple(r.pos)) for r in got] == [(r.seq, tuple(r.pos)) for r in exp]   # same draws
    assert tm.muted == 20 and tm.sent == ref.sent == 20 and tm.lost == ref.lost
    tm.unmute('f1')
    tm.send(20.0, states)
    r1 = [r for r in tm.receive(100.0) if r.id == 'f1']
    if r1:                                                 # (unless lost) its seq shows the gap
        assert r1[0].seq == 20 and r1[0].status == 'ok'
    assert Report('x', 0.0, np.zeros(3), np.zeros(3)).status == 'ok'


def test_radio_telemetry_passes_status():
    class FakeAdapter:
        def reports(self):
            return [Report('drone_0', 1.0, np.zeros(3), np.zeros(3), [], 0, 1.1, 'landing'),
                    Report('drone_1', 1.0, np.zeros(3), np.zeros(3), [], 0, 1.1)]
    rt = RadioTelemetry(FakeAdapter(), ['f1', 'f2'])
    got = {r.id: r.status for r in rt.receive(2.0)}
    assert got == {'f1': 'landing', 'f2': 'ok'}
