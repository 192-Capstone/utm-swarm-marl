"""E2 option 3, the Webots radio, without Webots: the message format (envs/radio_msg.py), when a
drone reports (RadioSender), RadioTelemetry's interface and id mapping (utm/telemetry.py), and
that with the flag off nothing changes (envs/webots_l2_multi.py)."""
import ast
import os
import statistics

import numpy as np
import pytest

from envs import radio_msg as rm
from envs.cf_velocity_pid import format_command
from envs.webots_l2_multi import DRONE_NODE, WebotsL2MultiAdapter, drone_node
from utm.telemetry import RadioTelemetry, Report, TelemetryModel

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORLD = os.path.join(ROOT, 'webots', 'worlds', 'swarm_l2_multi.wbt')
DT = 0.004
SETTINGS = {'period_s': 1.0, 'noise_m': 0.0, 'seed': 20}
INTENT = [[12.0, 1.5, -2.25, 3.0], [14.0, 2.5, -2.0, 3.0], [16.0, 1 / 3, 1e-17, 3.0]]


# ---------------------------------------------------------------- messages
def test_report_round_trip_is_exact():
    s = rm.encode_report('drone_3', 2, 7, 12.344, (1 / 3, -2.0, 3.000000001), (0.1, 0.2, -1e-17), INTENT)
    d = rm.decode_report(s)
    assert d == {'id': 'drone_3', 'ep': 2, 'seq': 7, 't': 12.344, 'p': [1 / 3, -2.0, 3.000000001],
                 'v': [0.1, 0.2, -1e-17], 'i': INTENT}


def test_report_accepts_numpy_values():
    s = rm.encode_report('drone_0', np.int64(1), np.int32(0), np.float32(0.5), np.zeros(3), np.ones(3), [])
    assert rm.decode_report(s)['p'] == [0.0, 0.0, 0.0]


def test_intent_round_trip_is_exact():
    assert rm.decode_intent(rm.encode_intent('drone_12', 3, INTENT)) == ('drone_12', 3, INTENT)
    assert rm.decode_intent(rm.encode_intent('drone_0', 1, [])) == ('drone_0', 1, [])


@pytest.mark.parametrize('s', ['', 'nope', '{}', '{"id":"drone_0","ep":1,"seq":0,"t":0,"p":[0,0],"v":[0,0,0],"i":[]}',
                               '{"id":"drone_0","ep":1,"seq":0,"t":0,"p":[0,0,0],"v":[0,0,0],"i":[[1,2,3]]}'])
def test_malformed_report_raises(s):
    with pytest.raises(ValueError):
        rm.decode_report(s)


@pytest.mark.parametrize('s', ['', '[]', '{"to":"drone_0","ep":1}', '{"to":"drone_0","ep":1,"i":[[1,2]]}'])
def test_malformed_intent_raises(s):
    with pytest.raises(ValueError):
        rm.decode_intent(s)


def test_report_size():
    """Full-precision floats: ~190 bytes without intent, ~460 with a 5-point intent."""
    pos, vel = (-12.345678901234567, 45.67890123456789, 3.0123456789012345), (1.2345678901234567, -0.12345678901234, 0.0012345678901234)
    intent = [[125.456 + 2 * k, -12.3456789012345 + k, 45.678901234567 + k, 3.0] for k in range(5)]
    assert len(rm.encode_report('drone_12', 1, 123, 123.456, pos, vel, [])) < 220
    assert len(rm.encode_report('drone_12', 1, 123, 123.456, pos, vel, intent)) < 500


def test_drone_index():
    assert rm.drone_index('drone_0') == 0 and rm.drone_index('drone_17') == 17
    for bad in ('Crazyflie', 'drone_', 'drone_x', 'robot_3'):
        with pytest.raises(ValueError):
            rm.drone_index(bad)


# ---------------------------------------------------------------- config
def test_radio_config_defaults_off():
    rc = rm.radio_config({})
    assert rc == {'enabled': False, 'period_s': 1.0, 'noise_m': 0.0, 'seed': 20, 'range_m': 200.0}
    assert rm.radio_config(None) == rc and rm.radio_config({'radio': None}) == rc


@pytest.mark.parametrize('bad', [{'period_s': 0}, {'noise_m': -1}, {'range_m': 0}, {'chanel': 3}])
def test_radio_config_rejects_bad_values(bad):
    with pytest.raises(ValueError):
        rm.radio_config({'radio': bad})


def test_controller_args_round_trip():
    rc = rm.radio_config({'radio': {'enabled': True, 'period_s': 0.5, 'noise_m': 1 / 3, 'seed': 22}})
    args = rm.controller_args(rc)
    assert not any('"' in a for a in args)
    assert rm.parse_controller_args(args) == {'period_s': 0.5, 'noise_m': 1 / 3, 'seed': 22}
    assert rm.parse_controller_args([]) is None


# ---------------------------------------------------------------- the sender
def fly(sender, ep, steps, t0=0.0, pos=lambda t: (t, 0.0, 1.0), act0='arm'):
    """steps basic steps of flight from Webots time t0; returns the decoded reports."""
    out = []
    for k in range(steps):
        t = t0 + k * DT
        msg = sender.step(ep, act0 if k == 0 else 'fly', t, pos(t))
        if msg is not None:
            out.append(rm.decode_report(msg))
    return out


def test_reports_once_per_period_from_the_second_step_of_flight():
    s = rm.RadioSender('drone_2', SETTINGS, DT)
    r = fly(s, 1, 1000, t0=2.004)                               # 4 s of flight
    assert [d['seq'] for d in r] == [0, 1, 2, 3]
    assert [d['t'] for d in r] == pytest.approx([2.008, 3.008, 4.008, 5.008])
    assert all(d['id'] == 'drone_2' and d['ep'] == 1 for d in r)


def test_one_release_step_then_stopped_sends_nothing():
    """The reset releases every drone with F for one basic step; SwarmEnv then parks the
    waiting ones: those must not report."""
    s = rm.RadioSender('drone_3', SETTINGS, DT)
    assert fly(s, 1, 1) == []                                   # arm, then held
    s.stop()
    assert s.seq == 0
    r = fly(s, 1, 2, t0=8.008)                                  # placed later: 2 steps of flight
    assert [(d['seq'], d['v']) for d in r] == [(0, [0.0, 0.0, 0.0])]
    assert r[0]['t'] == pytest.approx(8.012)


def test_a_drone_that_keeps_flying_reports_seq_0_within_2_steps():
    s = rm.RadioSender('drone_0', SETTINGS, DT)
    assert s.step(1, 'arm', 0.0, (0.0, 0.0, 1.0)) is None
    d = rm.decode_report(s.step(1, 'fly', DT, (0.0, 0.0, 1.0)))
    assert (d['seq'], d['t'], d['v']) == (0, DT, [0.0, 0.0, 0.0])


def test_position_is_the_gps_and_velocity_the_average_since_the_last_report():
    s = rm.RadioSender('drone_0', SETTINGS, DT)
    r = fly(s, 1, 751, pos=lambda t: (0.5 * t, -0.25 * t, 1.0 + 0.1 * t * t))
    assert r[0]['v'] == [0.0, 0.0, 0.0]                        # first report: no previous one
    for d in r:
        t = d['t']
        assert d['p'] == pytest.approx([0.5 * t, -0.25 * t, 1.0 + 0.1 * t * t], abs=1e-12)
    for a, b in zip(r, r[1:]):                                  # displacement / time since the last
        assert b['v'] == pytest.approx([(b['p'][j] - a['p'][j]) / (b['t'] - a['t']) for j in range(3)])


def test_no_reports_while_held_or_idle_and_a_new_take_off_restarts_the_period():
    s = rm.RadioSender('drone_1', SETTINGS, DT)
    r = fly(s, 1, 300)                                          # 1.2 s: reports at 0.004 and 1.004 s
    s.stop()                                                    # parked (H): no step() calls
    r += fly(s, 1, 300, t0=10.0)                                # placed: arm again
    assert [d['seq'] for d in r] == [0, 1, 2, 3]                # seq continues in the episode
    assert [d['t'] for d in r] == pytest.approx([0.004, 1.004, 10.004, 11.004])
    assert r[2]['v'] == [0.0, 0.0, 0.0]                         # first after the new take-off


def test_a_fly_after_stop_counts_as_a_take_off():
    s = rm.RadioSender('drone_1', SETTINGS, DT)
    fly(s, 1, 10)
    s.stop()
    r = fly(s, 1, 10, t0=5.0, act0='fly')
    assert len(r) == 1 and r[0]['t'] == 5.004 and r[0]['v'] == [0.0, 0.0, 0.0]


def test_new_episode_restarts_seq():
    s = rm.RadioSender('drone_1', SETTINGS, DT)
    fly(s, 1, 600)
    s.stop()
    r = fly(s, 2, 300, t0=30.0)
    assert [(d['ep'], d['seq']) for d in r] == [(2, 0), (2, 1)]


def test_intent_for_this_drone_is_reported_others_ignored():
    s = rm.RadioSender('drone_1', SETTINGS, DT)
    fly(s, 1, 1)                                                # flying, not reported yet
    s.hear(rm.encode_intent('drone_1', 1, INTENT))
    s.hear(rm.encode_intent('drone_10', 1, [[0.0, 9.0, 9.0, 9.0]]))   # not ours (prefix trap)
    r = fly(s, 1, 251, t0=0.004, act0='fly')
    assert r[0]['i'] == INTENT
    s.hear(rm.encode_intent('drone_1', 1, []))                  # climbing / landing: cleared
    assert fly(s, 1, 250, t0=1.004 + DT, act0='fly')[0]['i'] == []


def test_intent_from_another_episode_is_not_reported():
    s = rm.RadioSender('drone_1', SETTINGS, DT)
    s.hear(rm.encode_intent('drone_1', 1, INTENT))
    assert fly(s, 2, 2)[0]['i'] == []


def test_malformed_intent_raises_in_the_sender():
    with pytest.raises(ValueError):
        rm.RadioSender('drone_1', SETTINGS, DT).hear('garbage')


def _errors(ep, i, noise, n_reports=400):
    s = rm.RadioSender(f'drone_{i}', {**SETTINGS, 'noise_m': noise}, DT)
    r = fly(s, ep, 250 * (n_reports - 1) + 1, pos=lambda t: (0.0, 0.0, 1.0))
    return [d['p'][j] - (0.0, 0.0, 1.0)[j] for d in r for j in range(3)], [d['v'][j] for d in r[1:] for j in range(3)]


def test_noise_std_and_seeding():
    pe, ve = _errors(1, 0, 0.5)
    assert statistics.pstdev(pe) == pytest.approx(0.5, rel=0.1)
    assert statistics.pstdev(ve) == pytest.approx(0.1, rel=0.15)        # noise_m / 5
    assert _errors(1, 0, 0.5) == (pe, ve)                                # same (seed, ep, i): same noise
    assert _errors(2, 0, 0.5)[0] != pe and _errors(1, 1, 0.5)[0] != pe   # other episode / drone
    assert _errors(1, 0, 0.0)[0] == [0.0] * len(pe)


def test_noise_rng_is_process_independent():
    assert rm.noise_rng(20, 1, 0).random() == rm.noise_rng(20, 1, 0).random()
    assert rm.noise_rng(20, 1, 0).random() != rm.noise_rng(20, 2, 0).random()


# ---------------------------------------------------------------- RadioTelemetry
class FakeAdapter:
    def __init__(self):
        self.inbox, self.intents = [], []

    def reports(self):
        got, self.inbox = self.inbox, []
        return got

    def send_intent(self, i, intent):
        self.intents.append((i, intent))


def rep(i, seq, t, x=0.0):
    return Report(f'drone_{i}', t, np.array([x, 0.0, 3.0]), np.zeros(3), [], seq, t + 0.008)


def test_radio_telemetry_sends_only_the_intents():
    ad = FakeAdapter()
    tm = RadioTelemetry(ad, ['F1', 'F2', 'X1'])
    tm.send(1.0, {'X1': (np.ones(3), np.ones(3), []), 'F1': (np.zeros(3), np.zeros(3), np.array(INTENT))})
    assert sorted(ad.intents) == [(0, INTENT), (2, [])]
    assert all(isinstance(q, list) and all(type(v) is float for v in q) for _, it in ad.intents for q in it)


def test_radio_telemetry_maps_ids_and_keeps_the_newest():
    ad = FakeAdapter()
    tm = RadioTelemetry(ad, ['F1', 'F2'])
    ad.inbox = [rep(1, 0, 0.0), rep(0, 0, 0.1), rep(1, 1, 1.0, x=5.0)]
    got = tm.receive(1.0)
    assert [r.id for r in got] == ['F2', 'F1', 'F2']
    assert set(tm.latest) == {'F1', 'F2'} and tm.latest['F2'].pos[0] == 5.0
    assert got[0].t_received == pytest.approx(0.008)
    ad.inbox = [rep(1, 1, 1.0, x=7.0)]                          # same t_sent again: newest wins
    tm.receive(2.0)
    assert tm.latest['F2'].pos[0] == 7.0
    tm.forget('F2')
    assert set(tm.latest) == {'F1'}
    assert tm.receive(3.0) == []


def test_radio_telemetry_sent_and_lost_from_seq_gaps():
    ad = FakeAdapter()
    tm = RadioTelemetry(ad, ['F1', 'F2'])
    ad.inbox = [rep(0, 0, 0.0), rep(0, 1, 1.0), rep(0, 3, 3.0), rep(1, 0, 0.0), rep(1, 4, 4.0)]
    tm.receive(4.0)
    assert (tm.sent, tm.lost, tm.received) == (4 + 5, 1 + 3, 5)
    assert (TelemetryModel().sent, TelemetryModel().lost) == (0, 0)  # same attributes


def test_radio_telemetry_rejects_duplicate_ids_and_unknown_drones():
    with pytest.raises(ValueError):
        RadioTelemetry(FakeAdapter(), ['F1', 'F1'])
    ad = FakeAdapter()
    tm = RadioTelemetry(ad, ['F1'])
    ad.inbox = [rep(3, 0, 0.0)]
    with pytest.raises(KeyError):
        tm.receive(0.0)


def test_radio_telemetry_has_the_eval_interface():
    for name in ('send', 'receive', 'forget', 'latest', 'sent', 'lost'):
        assert hasattr(RadioTelemetry(FakeAdapter(), []), name) and hasattr(TelemetryModel(), name)


# ---------------------------------------------------------------- flag off changes nothing
def test_flag_off_drone_node_unchanged():
    for i in range(12):
        assert drone_node(i) == drone_node(i, rm.radio_config({})) == DRONE_NODE.format(i=i, x=2.0 * i)
    assert DRONE_NODE == ('DEF DRONE_{i} Crazyflie {{\n  translation {x} 0 0.015\n  name "drone_{i}"\n'
                          '  controller "cf_velocity"\n}}')


def test_flag_on_drone_node_has_devices_and_args():
    rc = rm.radio_config({'radio': {'enabled': True, 'noise_m': 0.5, 'seed': 21, 'range_m': 150}})
    s = drone_node(3, rc)
    assert s.startswith(DRONE_NODE.format(i=3, x=6.0)[:-2])   # same node, more fields
    for part in ('controllerArgs [ "--radio" "period_s=1.0" "noise_m=0.5" "seed=21" ]',
                 'name "radio_tx"', f'channel {rm.UPLINK}', 'range 150.0', 'name "radio_rx"', f'channel {rm.DOWNLINK}'):
        assert part in s


def test_flag_off_command_strings_unchanged():
    assert format_command(3, 'F', (0.1, -2.0, 1 / 3)) == '3 F 0.1 -2.0 0.3333333333333333'
    assert format_command(1, 'H', (0.0, 0.0, 0.0)) == '1 H 0.0 0.0 0.0'


def test_flag_off_adapter_has_no_radio():
    ad = WebotsL2MultiAdapter.__new__(WebotsL2MultiAdapter)   # no Webots
    ad._rx = ad._tx = None
    with pytest.raises(RuntimeError):
        ad.reports()
    with pytest.raises(RuntimeError):
        ad.send_intent(0, [])


def test_world_supervisor_devices_are_inert_by_default():
    """utm_rx / utm_tx sit in the world file; nothing enables or uses them unless the flag is on."""
    w = open(WORLD).read()
    assert 'name "utm_rx"' in w and f'channel {rm.UPLINK}' in w
    assert 'name "utm_tx"' in w and f'channel {rm.DOWNLINK}' in w
    sup = w.split('DEF SUPERVISOR')[1].split('DEF DRONES')[0]
    code = '\n'.join(ln.split('#')[0] for ln in sup.splitlines())
    assert 'physics' not in code and 'boundingObject' not in code
    tree = ast.parse(open(os.path.join(ROOT, 'envs', 'webots_l2_multi.py')).read())
    init = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == '__init__')
    guard = next(n for n in init.body if isinstance(n, ast.If) and "self.radio['enabled']" in ast.unparse(n.test))
    outside = [ast.unparse(n) for n in init.body if n is not guard]
    assert "getDevice('utm_" in ast.unparse(guard) and not any("getDevice('utm_" in u for u in outside)


def test_radio_module_is_stdlib_only():
    tree = ast.parse(open(os.path.join(ROOT, 'envs', 'radio_msg.py')).read())
    names = {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    names |= {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    assert names == {'json', 'random'}
