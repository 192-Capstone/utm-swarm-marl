"""
test_eval_layered_failures.py
-----------------------------
D5 failures in scripts/eval_layered.py: FailureFeed (what the UTM hears), fly(failures=...) and the CLI.

Run: python -m pytest tests/test_eval_layered_failures.py -q
"""

import copy
import os
import sys
import types

import numpy as np
import pytest
import yaml

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

import eval_layered as ev                                           # noqa: E402
from utm.conformance import FAILURE_STATES                          # noqa: E402
from utm.telemetry import TelemetryModel                            # noqa: E402

IDS = ['a', 'b', 'c']


class FakeEnv:
    """The slice of SwarmEnv that FailureFeed reads."""

    def __init__(self, n=3):
        self.failed = [''] * n
        self.failure_t = np.full(n, np.nan)
        self.comms_lost = np.zeros(n, bool)
        self.forced_down = np.zeros(n, bool)
        self.emergency_report = np.zeros(n, bool)
        self.landed = np.zeros(n, bool)
        self.crashed = np.zeros(n, bool)
        self.cancelled = np.zeros(n, bool)
        self.waiting = np.zeros(n, bool)
        self.failures_skipped = []
        self.adapter = types.SimpleNamespace(get_raw_state=lambda: {f'drone_{i}': {'velocity': [0.5, 0.0, -9.0]}
                                                                    for i in range(n)})


POS = np.array([[1.0, 2.0, 8.0], [3.0, 4.0, 8.0], [5.0, 6.0, 8.0]])


def feed_of(failures, **kw):
    env, tm = FakeEnv(), TelemetryModel()
    return env, tm, ev.FailureFeed(env, tm, IDS, ev.resolve_failures(failures, IDS, 3))


def test_comms_loss_is_muted_once():
    env, tm, feed = feed_of([{'flight': 'a', 't': 1.0, 'type': 'comms'}])
    calls = []
    tm.mute = lambda fid: calls.append(fid)
    for k in range(5):
        feed.step(k * 0.032, POS)
    assert calls == []                                   # not failed yet
    env.failed[0], env.comms_lost[0] = 'comms', True
    for k in range(50):
        feed.step(k * 0.032, POS)
    assert calls == ['a']


def test_one_immediate_emergency_report():
    env, tm, feed = feed_of([{'flight': 1, 't': 1.0, 'type': 'motor'}])
    env.failed[1], env.emergency_report[1] = 'motor', True
    for k in range(20):
        feed.step(2.0 + k * 0.032, POS)
    assert tm.sent == 1                                  # one report, however many steps the fall takes
    r = tm.latest['b']                                   # received at once (perfect link)
    assert r.status == 'emergency' and r.t_sent == 2.0
    assert np.allclose(r.pos, POS[1]) and np.allclose(r.vel, [0.5, 0.0, -9.0])


def test_step_says_when_it_sent_an_emergency_report():
    env, tm, feed = feed_of([{'flight': 'a', 't': 1.0, 'type': 'motor'}, {'flight': 'b', 't': 1.0, 'type': 'land'}])
    assert feed.step(1.0, POS) is False
    env.failed[:2], env.emergency_report[:2] = ['motor', 'land'], True
    assert feed.step(2.0, POS) is True and feed.step(2.032, POS) is False       # once; a forced landing sends none


def test_regular_report_is_skipped_in_the_step_of_the_emergency_one():
    env, tm, feed = feed_of([{'flight': 'a', 't': 1.0, 'type': 'motor'}])
    env.failed[0], env.emergency_report[0] = 'motor', True
    feed.step(2.0, POS)
    assert feed.reported_now(0, 2.0) and not feed.reported_now(0, 2.032) and not feed.reported_now(1, 2.0)


def test_takeoff_time_and_time_into_the_flight():
    env, tm, feed = feed_of([{'flight': 'a', 't': 9.0, 'type': 'land'}, {'flight': 'b', 't': 9.0, 'type': 'land'}])
    env.waiting[:2] = True
    feed.step(1.0, POS)
    env.waiting[0] = False                                 # a leaves its pad at 2.0 s; b never does
    feed.step(2.0, POS)
    feed.step(3.0, POS)
    env.failed[0], env.failure_t[0] = 'land', 9.25
    rows = {r['flight']: r for r in feed.results(np.zeros(3, bool), [])}
    assert rows['a']['takeoff_t'] == 2.0 and rows['a']['into_flight_s'] == 7.25
    assert rows['b']['takeoff_t'] is None and rows['b']['into_flight_s'] is None


def test_no_emergency_report_when_the_drone_cannot_send_it():
    env, tm, feed = feed_of([{'flight': 'b', 't': 1.0, 'type': 'motor'}])
    env.failed[1] = 'motor'                              # env.emergency_report False: the report never goes out
    feed.step(2.0, POS)
    assert tm.sent == 0 and not tm.latest


def test_emergency_report_follows_the_link_model():
    env, tm = FakeEnv(), TelemetryModel(delay_s=0.5)
    feed = ev.FailureFeed(env, tm, IDS, ev.resolve_failures([{'flight': 'a', 't': 1, 'type': 'motor'}], IDS, 3))
    env.failed[0], env.emergency_report[0] = 'motor', True
    feed.step(2.0, POS)
    assert not tm.latest and tm.receive(2.5)[0].status == 'emergency'


def test_landing_status():
    env, tm, feed = feed_of([{'flight': 'a', 't': 1.0, 'type': 'land'}])
    assert [feed.status(i) for i in range(3)] == ['ok', 'ok', 'ok']
    env.failed[0] = 'land'
    assert feed.status(0) == 'landing' and feed.status(1) == 'ok'
    env.landed[0] = True
    assert feed.status(0) == 'ok'


def test_resolve_failures_ids_and_indices():
    got = ev.resolve_failures([{'flight': 'b', 't': 1, 'type': 'land'}, {'flight': 2, 't': 2, 'type': 'comms'}], IDS, 3)
    assert [g['flight'] for g in got] == ['b', 'c']
    for bad in ('zz', 3, -1, True):
        with pytest.raises(ValueError):
            ev.resolve_failures([{'flight': bad, 't': 1, 'type': 'land'}], IDS, 3)
    with pytest.raises(ValueError):                      # an intruder is not a planned flight
        ev.resolve_failures([{'flight': 'x', 't': 1, 'type': 'land'}], IDS + ['x'], 3)


def test_events_and_new_events():
    env, tm, feed = feed_of([{'flight': 'a', 't': 1.0, 'type': 'land'}])
    assert feed.new_events() == []
    env.failed[0] = 'land'
    assert feed.new_events() == [('failure', 0, 'land')] and feed.new_events() == []
    env.forced_down[0] = True
    assert feed.new_events() == [('forced_down', 0, '')]


class Mon:
    def __init__(self):
        self.state = {}


def test_observe_failures_and_false_alerts():
    env, tm, feed = feed_of([{'flight': 'a', 't': 1.0, 'type': 'comms'}, {'flight': 'b', 't': 5.0, 'type': 'comms'}])
    mon = Mon()
    mon.state = {'c': 'lost_link', 'b': 'down', 'a': 'ok'}      # b: a failure state before its failure fired
    feed.observe(3.0, mon)
    env.failed[0] = 'comms'
    mon.state = {'c': 'lost_link', 'b': 'down', 'a': 'lost_link'}
    feed.observe(4.0, mon)
    feed.observe(5.0, mon)                                         # nothing new
    assert feed.transitions == {'a': [(4.0, 'lost_link')]}
    assert [(x['flight'], x['state']) for x in feed.false_alerts] == [('c', 'lost_link'), ('b', 'down')]
    assert all(x['state'] in FAILURE_STATES for x in feed.false_alerts)


def test_arrivals_exclude_forced_down_and_results_rows():
    env, tm, feed = feed_of([{'flight': 'a', 't': 1.0, 'type': 'land'}, {'flight': 'b', 't': 2.0, 'type': 'motor'},
                             {'flight': 'c', 't': 3.0, 'type': 'comms'}])
    env.failed[:2] = ['land', 'motor']
    env.failure_t[:2] = [1.0, 2.0]
    env.forced_down[0] = env.crashed[1] = True
    feed.transitions['a'] = [(4.0, 'emergency_landing')]
    replans = [{'t': 0.5, 'flight': 'x', 'why': 'intrusion by a', 'status': 'approved'},     # before a's failure: not its doing
               {'t': 1.2, 'flight': 'a', 'why': 'emergency landing: route released, column held', 'kind': 'log',
                'status': 'logged'},
               {'t': 1.2, 'flight': 'x', 'why': 'landing a', 'status': 'approved'},
               {'t': 2.5, 'flight': 'x', 'why': 'landing b', 'status': 'approved'}]
    rows = {r['flight']: r for r in feed.results(np.array([True, False, True]), replans)}
    assert rows['a']['outcome'] == 'forced_down' and rows['b']['outcome'] == 'crashed'
    assert rows['c']['outcome'] == 'arrived' and rows['c']['status'] == 'skipped'
    assert rows['c']['reason'] == 'episode ended before' and rows['a']['reason'] is None
    assert rows['a']['status'] == 'fired' and rows['a']['fired_t'] == 1.0
    assert rows['a']['e3'] == {'detected_t': 4.0, 'state': 'emergency_landing',
                               'transitions': [(4.0, 'emergency_landing')]} and rows['b']['e3'] is None
    assert [y['why'] for y in rows['a']['e4']] == ['emergency landing: route released, column held', 'landing a']


def test_skipped_reasons_and_true_state():
    env, tm, feed = feed_of([{'flight': f, 't': 5.0, 'type': 'land'} for f in IDS])
    assert [feed.true_state(i) for i in range(3)] == ['flying'] * 3
    env.crashed[0], env.landed[1], env.waiting[2] = True, True, True
    assert [feed.true_state(i) for i in range(3)] == ['crashed', 'landed', 'waiting']
    env.failures_skipped = [{'drone': 'drone_2', 't': 5.0, 'type': 'land'}]
    feed.step(6.0, POS)                                   # the env skipped drone_2's failure: noted while it waited
    env.waiting[2] = False                                # ... it takes off later; the reason stays
    reasons = {r['flight']: r['reason'] for r in feed.results(np.zeros(3, bool), [])}
    assert reasons == {'a': 'crashed before', 'b': 'arrived before', 'c': 'still on its pad when due'}
    env.crashed[:] = env.landed[:] = False
    feed2 = ev.FailureFeed(env, tm, IDS, ev.resolve_failures([{'flight': 'a', 't': 5.0, 'type': 'land'}], IDS, 3))
    assert feed2.results(np.zeros(3, bool), [])[0]['reason'] == 'episode ended before'


def test_false_alert_records_the_true_state():
    env, tm, feed = feed_of([])
    env.crashed[0] = True
    mon = Mon()
    mon.state = {'a': 'lost_link', 'b': 'lost_link'}
    feed.observe(9.0, mon)
    assert [(x['flight'], x['true_state']) for x in feed.false_alerts] == [('a', 'crashed'), ('b', 'flying')]


def test_caused_by_formats_are_pinned():
    # the 'why' strings utm/contingency.py writes for work caused by a failed flight (E4 attribution fallback)
    src = open(os.path.join(ROOT, 'utm', 'contingency.py')).read()
    for fmt in ("f'lost link {fid}'", "f'landing {fid}'", "f'intrusion by {fid}'"):
        assert fmt in src, f'contingency.py no longer writes {fmt}: update eval_layered.caused_by'
    for why in ('lost link f3', 'landing f3', 'intrusion by f3'):
        assert ev.caused_by({'flight': 'f1', 'why': why}, 'f3')
        assert not ev.caused_by({'flight': 'f1', 'why': why}, 'f4')
        assert not ev.caused_by({'flight': 'f3', 'why': why}, 'f3')        # its own re-plan is not caused by it
    assert not ev.caused_by({'flight': 'f1', 'why': 'zone z1'}, 'f1')
    assert ev.caused_by({'flight': 'f3', 'why': 'down: booking released', 'kind': 'log'}, 'f3')      # structured first
    assert not ev.caused_by({'flight': 'f1', 'why': 'landing f3', 'kind': 'log'}, 'f3')
    assert ev.caused_by({'flight': 'f1', 'why': 'lost link 7'}, 7)         # ids need not be strings


def tiny_flights():
    return [{'id': 'a', 'start': (0, 0), 'goal': (5, 0), 'no_e1': (0.0, [(0, 0, 1), (5, 0, 5), (5, 0, 1)]),
             'e1': (0.0, [(0, 0, 1), (5, 0, 5), (5, 0, 1)]), 'delay': 0.0, 'arrival': {'no_e1': 9.0, 'e1': 9.0}}]


def test_fly_refuses_webots_and_no_e1():
    fl, args = tiny_flights(), types.SimpleNamespace(simulator='webots_l2_multi', telemetry=None, risk='booking')
    f = [{'flight': 'a', 't': 3.0, 'type': 'land'}]
    with pytest.raises(ValueError, match='PyBullet'):
        ev.fly({'environment': {}}, args, [], None, fl, [], [0.0], [1.0], 'e1', 'simple', 20, failures=f)
    args.simulator = 'pybullet'
    with pytest.raises(ValueError, match='e1 or e4'):
        ev.fly({'environment': {}}, args, [], None, fl, [], [0.0], [1.0], 'no_e1', 'simple', 20, failures=f)
    with pytest.raises(ValueError, match='unknown flight'):
        ev.fly({'environment': {}}, args, [], None, fl, [], [0.0], [1.0], 'e1', 'simple', 20,
               failures=[{'flight': 'q', 't': 1.0, 'type': 'land'}])
    with pytest.raises(TypeError):                       # keyword-only: no positional call can hit it
        ev.fly({'environment': {}}, args, [], None, fl, [], [0.0], [1.0], 'e1', 'simple', 20, None, None, None, f)


class StubRec:
    frame_dt = 0.5

    def __init__(self):
        self.log = {'frames': {'t': []}}
        self.events = []

    def start(self, *a):
        pass

    def frame(self, t, pos, status):
        self.log['frames']['t'].append(round(t, 3))

    def event(self, t, kind, **kw):
        self.events.append((kind, kw))

    def zone_on(self, *a):
        pass

    def tick(self, *a):
        pass

    def replan(self, *a):
        pass

    def finish(self, *a):
        pass


def small_world(n_flights=3):
    from eval_e1_flown import plans
    from preview_route import load_map
    from preview_strategic import random_requests
    from utm.strategic_planner import StrategicPlanner
    with open(os.path.join(ROOT, 'configs/config_stage2_round7_eval_goalhold.yaml'), encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    cfg.pop('scenario_mix', None)
    cfg['environment']['collision_physics'] = False
    cfg.setdefault('obstacles', {})['enabled'] = False
    args = types.SimpleNamespace(map='city', level='medium', seed=20, simulator='pybullet', telemetry=None,
                                 risk='booking', horizon=20.0, checkpoint=None, extra_time=0.0)
    layout, bounds = load_map(args)
    sky = StrategicPlanner(layout, bounds)
    _, flights = plans(sky, random_requests(np.random.RandomState(20), sky, n_flights, window=10.0))
    return cfg, args, layout, sky, flights


def test_fly_with_failures_end_to_end():
    """A short real run (PyBullet, simple pilot): a forced landing and a comms loss in e1 and e4; the config
    given to fly() is not touched, and a later run without failures has none of the failure keys."""
    cfg, args, layout, sky, flights = small_world()
    z = [0.0] * len(flights)
    mid = [0.5 * (f['e1'][0] + f['arrival']['e1']) for f in flights]
    fails = [{'flight': flights[0]['id'], 't': mid[0], 'type': 'land'}, {'flight': 1, 't': mid[1], 'type': 'comms'}]
    before = copy.deepcopy(cfg)
    for mode in ('e1', 'e4'):
        rec = StubRec()
        r = ev.fly(cfg, args, layout, sky, flights, [], z, [1.0] * len(flights), mode, 'simple', 20, rec=rec,
                   failures=fails)
        assert cfg == before and 'failures' not in cfg['environment']
        rows = {x['flight']: x for x in r['failures']}
        assert rows[flights[0]['id']]['status'] == 'fired' and rows[flights[1]['id']]['status'] == 'fired'
        assert r['forced_down'] == 1 and rows[flights[0]['id']]['outcome'] == 'forced_down'
        kinds = [(k, kw['flight']) for k, kw in rec.events]
        assert ('failure', flights[0]['id']) in kinds and ('forced_down', flights[0]['id']) in kinds
        assert ('arrived', flights[0]['id']) not in kinds
        assert r['arrived'] <= len(flights) - 1
        assert rows[flights[1]['id']]['e3'] is not None      # silence: lost link
        assert set(r['false_alerts'][0]) == {'flight', 't', 'state'} if r['false_alerts'] else True
    r = ev.fly(cfg, args, layout, sky, flights, [], z, [1.0] * len(flights), 'e1', 'simple', 20)
    assert cfg == before and not {'failures', 'forced_down', 'false_alerts'} & set(r)


def test_motor_failure_is_detected_in_the_step_of_its_emergency_report():
    """E3 / E4 run in the step the emergency report goes out, not at the next 1 Hz tick (the fall ends before it)."""
    cfg, args, layout, sky, flights = small_world()
    z = [0.0] * len(flights)
    tick = 31 * cfg['environment'].get('control_timestep', 0.032)            # the eval's 1 Hz tick (31 control steps)
    f = flights[0]
    for mode in ('e1', 'e4'):
        fails = [{'flight': f['id'], 't': f['e1'][0] + 0.5 * (f['arrival']['e1'] - f['e1'][0]) + 0.2, 'type': 'motor'}]
        r = ev.fly(cfg, args, layout, sky, flights, [], z, [1.0] * len(flights), mode, 'simple', 20, failures=fails)
        row = r['failures'][0]
        assert row['status'] == 'fired' and row['e3']['state'] == 'down'
        det = row['e3']['detected_t']
        assert 0 <= det - row['fired_t'] <= 0.05, (mode, row)                # the report goes out within a control step
        assert abs(det / tick - round(det / tick)) > 0.02, 'pick a failure time that is not on a tick'
        if mode == 'e4':
            assert any(a['kind'] == 'log' and a['t'] == pytest.approx(det, abs=0.01) and a['why'].startswith('down')
                       for a in row['e4'])


def test_motor_failure_on_a_tick_step_is_still_detected_as_down():
    """The step of the failure is also a 1 Hz tick: the regular report of that step must not replace the emergency
    one (same t_sent), or E3 only hears silence and calls it a lost link seconds later."""
    cfg, args, layout, sky, flights = small_world()
    z = [0.0] * len(flights)
    tick = 31 * cfg['environment'].get('control_timestep', 0.032)
    f = flights[0]
    k = round((f['e1'][0] + 0.5 * (f['arrival']['e1'] - f['e1'][0])) / tick)
    for mode in ('e1', 'e4'):
        fails = [{'flight': f['id'], 't': k * tick - 0.01, 'type': 'motor'}]
        row = ev.fly(cfg, args, layout, sky, flights, [], z, [1.0] * len(flights), mode, 'simple', 20,
                     failures=fails)['failures'][0]
        assert row['e3']['state'] == 'down' and row['e3']['detected_t'] == pytest.approx(k * tick, abs=0.04), row


def test_cli_default_draws_nothing(monkeypatch, tmp_path):
    seen = []

    def boom(*a, **k):
        raise AssertionError('draw_failures called with --failures unset')
    monkeypatch.setattr(ev, 'draw_failures', boom)
    monkeypatch.setattr(ev, 'fly', lambda *a, **k: seen.append(k.get('failures', 'absent')) or _stub_result())
    out = tmp_path / 'o.json'
    monkeypatch.setattr(sys, 'argv', ['x', '--config', os.path.join(ROOT, 'configs/config_stage2_round7_eval_goalhold.yaml'),
                                      '--checkpoint', 'none', '--seeds', '20', '--flights', '3', '--modes', 'e1', 'e4',
                                      '--pilots', 'simple', '--out', str(out)])
    ev.main()
    assert seen == [None, None]
    import json
    assert 'failures' not in json.load(open(out))['args']


def _stub_result():
    return {'flights': 3, 'intruders': 0, 'arrived': 3, 'crashed': 0, 'crashed_all': 0, 'pairs_below_1.4': 0,
            'pairs_below_1.4_with_intruder': 0, 'min_separation_m': None, 'zone_entries': 0, 'zone_seconds': 0.0,
            'emergencies': [], 'cancelled': 0, 'returned': 0, 'replans': [], 'e3': {'alerts': 0},
            'close_pair_warning_s': [], 'close_pair_high_warning_s': [], 'high_alerts': 0, 'high_alerts_followed': 0,
            'alerts_followed_by_close_pass': 0}


def test_cli_failures_one_schedule_per_seed_for_every_mode(monkeypatch, tmp_path):
    seen = []
    res = _stub_result() | {'failures': [], 'forced_down': 0, 'false_alerts': []}
    monkeypatch.setattr(ev, 'fly', lambda *a, **k: seen.append((a[8], a[9], k['failures'])) or res)
    out = tmp_path / 'o.json'
    monkeypatch.setattr(sys, 'argv', ['x', '--config', os.path.join(ROOT, 'configs/config_stage2_round7_eval_goalhold.yaml'),
                                      '--checkpoint', 'none', '--seeds', '20', '--flights', '6', '--modes', 'e1', 'e4',
                                      '--pilots', 'simple', 'learned', '--failures', '1.0', '--failure-types', 'land',
                                      'comms', '--out', str(out)])
    ev.main()
    assert len(seen) == 4 and all(s[2] is seen[0][2] or s[2] == seen[0][2] for s in seen)
    assert len(seen[0][2]) == 6 and {f['type'] for f in seen[0][2]} <= {'land', 'comms'}


@pytest.mark.parametrize('extra', [['--modes', 'no_e1', 'e1'], ['--simulator', 'webots_l2_multi', '--modes', 'e1',
                                                                '--pilots', 'simple'], ['--failures', '1.5']])
def test_cli_refuses(monkeypatch, extra):
    argv = ['x', '--config', 'c', '--checkpoint', 'n', '--seeds', '20', '--failures', '0.5'] + extra
    monkeypatch.setattr(sys, 'argv', argv)
    with pytest.raises(SystemExit):
        ev.main()
