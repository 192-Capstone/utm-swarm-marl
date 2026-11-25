"""The tracking page's logic (dashboard/track_logic.js), run under node: remaining distance and ETA (checked against E1's
own profile), the smoothing, the status stepper, and loading a replay from the files export_dashboard.py and flight_db.py
write. Skipped without node."""

import json
import os
import runpy
import shutil
import subprocess
import sys

import numpy as np
import pytest

from utm import flight_db, flight_log, speeds
from utm.strategic_planner import FlightPlan, Settings, StrategicPlanner

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
NODE = shutil.which('node')
pytestmark = pytest.mark.skipif(NODE is None, reason='node is not installed')
B = ((-20, 20), (-20, 20))


def node(js, **data):
    """Runs js (with T = track_logic.js and D = data) under node; its last printed line is JSON."""
    code = (f"const T = require({json.dumps(os.path.join(ROOT, 'dashboard', 'track_logic.js'))});"
            f"const D = {json.dumps(data)}; const out = (() => {{ {js} }})(); "
            "Promise.resolve(out).then(v => console.log(JSON.stringify(v))).catch(e => { console.log(JSON.stringify({error: String(e)})); });")
    r = subprocess.run([NODE, '-e', code], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout.strip().splitlines()[-1])


SP = {'v_cruise': 1.5, 'v_climb': 0.5, 'v_descent': 0.5, 'v_descent_high': 1.0, 'slow_band_m': 3.0, 'start_z': 1.0}
ROUTE = [[0, 0, 1], [0, 0, 5], [30, 0, 5], [30, 0, 1]]           # climb 4 m, cruise 30 m, descend 4 m


def test_the_profile_matches_e1s_own_booking():
    """The time JS gives a route is the time E1 booked for it (strategic_planner.profile): climb, cruise, two-speed descent."""
    s = Settings(n_layers=3, v_climb=0.5, v_descent=0.5, v_descent_high=1.0, slow_band_m=3.0)
    sky = StrategicPlanner([], B, s)
    d = sky.submit(FlightPlan('A', (-15, 0), (15, 0), layer=1, t_earliest=0.0))
    assert d.status == 'approved'
    out = node('return T.profile(D.route, D.sp).total_time;', route=d.waypoints, sp=speeds.from_settings(s))
    assert out == pytest.approx(d.arrival - d.t_dep, abs=1e-6)
    by_hand = node('return T.profile(D.route, D.sp);', route=ROUTE, sp=SP)
    assert by_hand['total_time'] == pytest.approx(4 / 0.5 + 30 / 1.5 + (1 / 1.0 + 3 / 0.5))     # 8 + 20 + 7 s
    assert by_hand['total_len'] == pytest.approx(38.0)


def test_remaining_time_and_distance_along_the_route():
    r = node('return [T.remaining(D.route, [0, 0, 1], D.sp), T.remaining(D.route, [15, 0, 5], D.sp), '
             'T.remaining(D.route, [30, 0, 1], D.sp), T.remaining(D.route, [30, 0, 2.5], D.sp)];', route=ROUTE, sp=SP)
    start, mid, end, low = r
    assert start['time_s'] == pytest.approx(35.0) and start['dist_m'] == pytest.approx(38.0) and start['flown_m'] == 0
    assert mid['dist_m'] == pytest.approx(15 + 4) and mid['time_s'] == pytest.approx(10 + 7)     # half the cruise, the descent
    assert end['time_s'] == pytest.approx(0.0, abs=1e-9) and end['dist_m'] == pytest.approx(0.0, abs=1e-9)
    assert low['time_s'] == pytest.approx(1.5 / 0.5) and low['flown_m'] == pytest.approx(36.5)    # 1.5 m left in the slow band


def test_a_position_never_projects_back_onto_an_earlier_segment():
    loop = [[0, 0, 5], [10, 0, 5], [10, 10, 5], [0, 10, 5], [0, 0.5, 5]]          # ends next to where it starts
    r = node('return [T.project(D.r, [0, 0.25, 5], 0).seg, T.project(D.r, [0, 0.25, 5], 3).seg];', r=loop)
    assert r == [0, 3]


def test_a_climbing_drone_still_counts_the_climb_and_a_late_one_the_wait():
    c = node("const r = T.remaining(D.route, [0, 0, 3], D.sp); const e = T.makeEta();"
             "return [r.time_s, e.update(100, 12, r.time_s, 140)];", route=ROUTE, sp=SP)
    assert c[0] == pytest.approx(2 / 0.5 + 20 + 7)
    assert c[1]['arrival'] == pytest.approx(100 + 12 + 31) and c[1]['late_s'] == pytest.approx(3.0) and c[1]['left_s'] == pytest.approx(43)


def test_the_arrival_estimate_is_smoothed_and_resets_on_a_replan():
    r = node("const e = T.makeEta(); const out = [];"
             "out.push(e.update(0, 0, 50, 50).arrival);"          # first estimate: as it is, 50
             "out.push(e.update(1, 0, 52, 50).arrival);"          # raw 53, a 3 s jump: only part of it
             "out.push(e.update(2, 0, 49, 50).arrival);"          # raw 51
             "out.push(e.update(3, 0, 5, 50).arrival);"           # raw 8 ... a jump of 40 s: a re-plan, taken as it is
             "return out;")
    assert r[0] == 50 and 50 < r[1] < 53 and r[1] - 50 == pytest.approx(3 * (1 - np.exp(-1 / 4)))
    assert r[3] == pytest.approx(8.0)
    steady = node("const e = T.makeEta(); let a; for (let t = 0; t < 30; t++) a = e.update(t, 0, 100 - t, 100).arrival; return a;")
    assert steady == pytest.approx(100.0)


def test_the_eta_line_and_late_note_in_words():
    r = node("return [T.fmtLeft(80), T.fmtLeft(45), T.fmtLeft(120), T.fmtLeft(3900),"
             "T.etaLine({status: 'flying', left_s: 80}), T.etaLine({status: 'flying', left_s: 1.2}),"
             "T.etaLine({status: 'landed', clock_arr: 0, t_arr: 61}), T.etaLine({status: 'crashed', left_s: 4}),"
             "T.lateNote(34, false), T.lateNote(-12, true), T.lateNote(3, false), T.etaLine({status: 'landed', returned: true})];")
    assert r[:4] == ['1 min 20 s', '45 s', '2 min', '1 h 05 min']
    assert r[4] == 'Arriving in 1 min 20 s' and r[5] == 'Arriving now'
    assert r[6].startswith('Delivered at ') and len(r[6]) == len('Delivered at 00:00:00') and r[7] is None
    assert r[8] == {'tone': 'warn', 'text': 'Running late +34 s'} and r[9] == {'tone': 'ok', 'text': 'Arrived 12 s early'}
    assert r[10]['text'] == 'On time' and r[11] == 'Back at base'


def stage(**kw):
    base = {'status': 'flying', 'now': 100, 't_dep': 90, 'left_s': 60, 'late_s': 0, 'climbing': False, 'rerouted': False}
    return node('return T.stage(D.i);', i={**base, **kw})


def test_the_stepper_follows_the_flight():
    names = lambda s: [x['state'] for x in s['steps']]                         # noqa: E731
    assert names(stage(status='waiting', now=80)) == ['done', 'current', 'todo', 'todo', 'todo', 'todo']
    assert stage(status='flying', climbing=True)['current'] == 2
    assert stage(status='flying')['current'] == 3 and stage(status='flying', left_s=15)['current'] == 4
    done = stage(status='landed')
    assert names(done) == ['done'] * 6 and done['exception'] is None


def test_the_exceptions_are_shown_clearly():
    w = stage(status='waiting', now=80)['exception']
    assert (w['label'], w['tone']) == ('Waiting for slot', 'info') and 'takes off in 10 s' in w['detail']
    d = stage(status='waiting', now=120)['exception']
    assert (d['label'], d['tone']) == ('Delayed', 'warn')
    r = stage(status='rejected', reason='no free slot')
    assert r['exception']['label'] == 'Rejected' and r['exception']['detail'] == 'no free slot'
    assert [x['state'] for x in r['steps']][:3] == ['done', 'failed', 'todo']
    c = stage(status='crashed', reached=3)
    assert c['exception']['tone'] == 'bad' and [x['state'] for x in c['steps']] == ['done', 'done', 'done', 'done', 'failed', 'todo']
    assert stage(status='forced_down', reached=3)['exception']['label'] == 'Forced down'
    ret = stage(status='landed', returned=True)
    assert ret['exception']['label'] == 'Returned to base' and ret['steps'][5]['state'] == 'failed'
    assert stage(status='cancelled')['exception']['label'] == 'Cancelled'
    notes = stage(status='flying', late_s=34, rerouted=True)['notes']
    assert [n['label'] for n in notes] == ['Running late +34 s', 'Re-routed around a zone']
    assert stage(status='flying', late_s=3)['notes'] == []


def test_events_in_words_tell_what_happened():
    ev = [{'type': 'replan', 'text': 'f8 re-routed around the no-fly zone'}, {'type': 'zone_on', 'text': 'No-fly zone closes the area'}]
    r = node("return [T.classifyEvents(D.ev), T.classifyEvents([{type: 'replan', tag: 'back to its pad', text: 'f1 re-planned'}]),"
             "T.classifyEvents([{type: 'forced_down', text: 'f2 brought down'}]), T.classifyEvents([{type: 'zone_on', text: 'No-fly zone closes the area'}])];", ev=ev)
    assert r[0]['rerouted'] and not r[0]['returned']
    assert r[1]['returned'] and r[2]['forced'] and not r[3]['rerouted']


def test_unknown_speeds_fall_back_to_e1s_defaults_and_say_so():
    r = node("return [T.resolveSpeeds(null), T.resolveSpeeds(D.sp), T.resolveSpeeds({v_cruise: 2, v_climb: 1, v_descent: null})];", sp=SP)
    assert r[0]['approximate'] is True and r[0]['speeds']['v_cruise'] == 1.5
    assert r[1]['approximate'] is False and r[1]['speeds']['v_descent_high'] == 1.0
    assert r[2]['approximate'] is False and r[2]['speeds']['v_descent'] == 1                  # v_descent null: v_climb


# ---------------------------------------------------------------- replay data, as the exporters write it
def exported(tmp_path, with_log=True, with_speeds=True):
    """A tiny recording -> data/<scene>.json (export_dashboard.py) and data/flight_log.json (flight_db.py)."""
    sky = StrategicPlanner([], B)
    sky.submit(FlightPlan('A', (-15, 0), (15, 0), t_earliest=6.0, operator='op0'))
    d = sky.current['A']
    rec = flight_log.FlightLog({'name': 'tiny', 'seed': 1, 'map': {'map': 'city'}, 'utm': {'v_climb': 0.5, 'v_descent': 0.5}}, frame_dt=0.5)
    rec.start(sky, [], 0.032, [{'id': 'A', 'kind': 'flight', 'operator': 'op0', 'start': (-15, 0), 'goal': (15, 0), 't_dep': d.t_dep,
                                'arrival': d.arrival, 'layer': d.layer, 'route': d.waypoints}])
    n = int(d.arrival / 0.5) + 4
    for k in range(n):
        t = 0.5 * (k + 1)
        u = min(1.0, max(0.0, (t - d.t_dep) / (d.arrival - d.t_dep)))
        rec.frame(t, np.array([[-15 + 30 * u, 0.0, 1.0 + 3 * u]]), [0 if t <= d.t_dep else 1 if u < 1 else 2])
    rec.event(0.5 * n, 'arrived', flight='A')
    rec.finish({'arrived': 1, 'flights': 1, 'crashed': 0})
    rec.log['requests'] = [{'id': 'A', 'operator': 'op0', 'start': [-15.0, 0.0], 'goal': [15.0, 0.0], 't_request': 0.0, 'status': 'approved', 'reason': ''}]
    path = tmp_path / 'tiny.json.gz'
    rec.save(path)
    data = tmp_path / 'data'
    old = sys.argv
    try:
        sys.argv = ['export_dashboard.py', str(path), '--out', str(data / 'tiny.json')]
        runpy.run_path(os.path.join(ROOT, 'scripts', 'export_dashboard.py'), run_name='__main__')
    finally:
        sys.argv = old
    scene = json.loads((data / 'tiny.json').read_text(encoding='utf-8'))
    if not with_speeds:
        scene.pop('speeds')
        (data / 'tiny.json').write_text(json.dumps(scene), encoding='utf-8')
    if with_log:
        db = flight_db.connect(':memory:')
        flight_db.import_recording(db, str(path))
        flight_db.export_json(db, str(data / 'flight_log.json'))
    return data, d


def load_js(data_dir, flight='A', extra=''):
    return (f"const fs = require('fs'); const get = async (u) => {{ const p = {json.dumps(str(data_dir))} + '/' + u.replace(/^data\\//, '');"
            "if (!fs.existsSync(p)) throw new Error('404 ' + u); return JSON.parse(fs.readFileSync(p, 'utf8')); };"
            f"return T.loadReplay(get, '', {{scene: 'tiny', flight: {json.dumps(flight)}}}).then(m => {{ {extra} }});")


def test_the_export_carries_the_scenarios_speeds(tmp_path):
    data, _ = exported(tmp_path)
    scene = json.loads((data / 'tiny.json').read_text(encoding='utf-8'))
    assert scene['speeds']['v_climb'] == 0.5 and scene['speeds']['v_cruise'] == Settings().v_cruise and scene['speeds']['start_z'] == 1.0


def test_a_replay_loads_and_moves(tmp_path):
    data, d = exported(tmp_path)
    r = node(load_js(data, extra=(
        "const a = m.at(m.t0), b = m.at((m.t0 + m.t1) / 2), c = m.at(m.t1);"
        "return {outcome: m.outcome, approx: m.approximate, callsign: m.callsign, kind: m.kind, from: m.from, a: a.status, b: b.status, c: c.status,"
        "x0: a.pos[0], x1: c.pos[0], route: a.route.length, hasClock: m.clockOf !== null, t0: m.t0, t1: m.t1, n: m.events.length};")))
    assert r['outcome'] == 'Landed' and r['approx'] is False and r['callsign'] == 'OP0-A' and r['kind'] == 'flight'
    assert (r['a'], r['c']) == ('waiting', 'landed') and r['b'] == 'flying'
    assert r['x0'] == pytest.approx(-15.0, abs=0.5) and r['x1'] == pytest.approx(15.0, abs=0.5) and r['route'] >= 3
    assert r['hasClock'] and 0 <= r['t0'] < d.t_dep and r['t1'] > d.arrival and r['n'] >= 1


def test_a_replay_without_the_flight_log_or_speeds_still_loads_but_says_approximate(tmp_path):
    data, _ = exported(tmp_path, with_log=False, with_speeds=False)
    r = node(load_js(data, extra="return {outcome: m.outcome, approx: m.approximate, clock: m.clockOf, last: m.at(m.t1).status, v: m.speeds.v_cruise};"))
    assert r == {'outcome': None, 'approx': True, 'clock': None, 'last': 'landed', 'v': 1.5}


def test_a_replay_of_an_unknown_flight_is_an_error(tmp_path):
    data, _ = exported(tmp_path)
    assert 'is not in scene' in node(load_js(data, flight='Z', extra='return 1;'))['error']
    missing = node("return T.loadReplay(async () => { throw new Error('404'); }, '', {scene: 'nope', flight: 'A'});")
    assert 'error' in missing


def test_a_forced_down_flight_is_shown_as_forced_down_not_delivered(tmp_path):
    data, _ = exported(tmp_path)
    log = json.loads((data / 'flight_log.json').read_text(encoding='utf-8'))
    log['flights'][0]['status'] = 'Forced down'
    (data / 'flight_log.json').write_text(json.dumps(log), encoding='utf-8')
    r = node(load_js(data, extra="return [m.at(m.t1).status, m.outcome];"))
    assert r == ['forced_down', 'Forced down']


def test_a_logged_live_delivery_replays_from_its_track():
    tr = {'run': {'run_id': 'live-x', 'scenario': 'x', 'speed': 2.0, 'speeds': SP, 'start_epoch': 1000.0},
          'flight': {'flight_id': 'f0', 'kind': 'flight', 'operator': 'op0', 'origin': 'Depot', 'dest': 'Shop', 'status': 'Landed',
                     'sched_dep': 5.0, 'sched_arr': 30.0, 'callsign': 'OP0-00', 'drone': 'D01', 'layer': 1, 'cruise_z': 4.0,
                     'distance_m': 38.0, 'n_replans': 0},
          'events': [{'t': 5, 'type': 'takeoff', 'text': 'f0 took off'}],
          'track': [[6, 0, 0, 1], [10, 0, 0, 3], [15, 8, 0, 5], [25, 30, 0, 5], [32, 30, 0, 1]]}
    r = node("const m = T.trackModel(D.tr, null); return {a: m.at(0).status, b: m.at(20).status, c: m.at(40).status, mid: m.at(20).pos,"
             "clock: m.clockOf(10), approx: m.approximate, from: m.from, rem: T.remaining(m.at(20).route, m.at(20).pos, m.speeds).time_s};", tr=tr)
    assert (r['a'], r['b'], r['c']) == ('waiting', 'flying', 'landed') and r['clock'] == 1005.0 and r['approx'] is False
    assert r['from'] == 'Depot' and 0 < r['rem'] < 40 and r['mid'][0] == pytest.approx(19.0, abs=3)
    assert 'no track' in node("return (() => { try { T.trackModel({flight: {flight_id: 'q'}, track: [], run: {}}, null); } catch (e) { return String(e); } })();")


@pytest.mark.parametrize('operator,kind,fid', [('op0', 'flight', 'f7'), ('Hospital', 'emergency', 'emergency_0'), ('', 'intruder', 'intruder_3'),
                                                ('', 'emergency', 'emergency_12'), ('East Hub', 'flight', 'f104'), ('op1', 'flight', 'A')])
def test_the_page_names_a_flight_as_the_database_does(operator, kind, fid):
    assert node('return T.callsign(D.o, D.k, D.f);', o=operator, k=kind, f=fid) == flight_db.callsign(operator, kind, fid)


def test_an_interrupted_delivery_still_has_the_way_to_its_pad_ahead():
    tr = {'run': {'start_epoch': 0, 'speed': 1.0}, 'events': [],
          'flight': {'flight_id': 'f0', 'kind': 'flight', 'operator': 'op0', 'origin': 'Depot', 'dest': 'Shop', 'sched_dep': 5,
                     'sched_arr': 60, 'status': 'Flying', 'dest_x': 60.0, 'dest_y': 0.0},
          'track': [[6, 0, 0, 1], [10, 0, 0, 5], [15, 8, 0, 5], [25, 30, 0, 5]]}
    js = 'return T.trackModel(D.tr, null).at(20).route;'
    assert node(js, tr=tr)[-2:] == [[60, 0, 5], [60, 0, 1]]
    tr['flight']['status'] = 'Crashed'                                 # it ended there: nothing left
    assert node(js, tr=tr)[-1] == [30, 0, 5]


def test_a_refused_booking_has_no_track_only_its_reason():
    tr = {'run': {'start_epoch': 0, 'speed': 1.0}, 'track': [], 'flight': {'flight_id': 'r1', 'kind': 'flight', 'operator': 'op0', 'origin': 'Depot',
          'dest': 'Shop', 'status': 'Rejected', 'callsign': 'OP0-01'},
          'events': [{'t': 6, 'type': 'refused', 'text': 'r1: booking refused (no free slot)'}]}
    m = node('const m = T.trackModel(D.tr, null); return {rejected: m.rejected, reason: m.reason, outcome: m.outcome};', tr=tr)
    assert m == {'rejected': True, 'reason': 'no free slot', 'outcome': 'Rejected'}


def test_the_models_carry_the_map_for_the_page(tmp_path):
    data, _ = exported(tmp_path)
    r = node(load_js(data, extra="return {b: m.geo.buildings.length, bounds: m.geo.bounds, zones: Array.isArray(m.geo.zones), places: Array.isArray(m.geo.places),"
                                 "land: m.tLand, off: m.tTakeoff};"))
    assert r['bounds'] == [[-20, 20], [-20, 20]] and r['zones'] and r['places'] and r['off'] > 0 and r['land'] > r['off']


def test_a_drone_parked_off_the_map_is_on_its_pad_until_it_takes_off(tmp_path):
    """Recordings park a drone that has not left (and one removed after a crash) far off the map: the replay keeps it on its
    pad, then where it fell."""
    data, _ = exported(tmp_path)
    scene = json.loads((data / 'tiny.json').read_text(encoding='utf-8'))
    n = len(scene['frames']['t'])
    for k in range(n):                                              # off the map before take-off and after the landing
        if scene['frames']['status'][k][0] == 0 or k > n - 3:
            scene['frames']['pos'][k][0] = [100300.0, 100000.0, 50.0]
    (data / 'tiny.json').write_text(json.dumps(scene), encoding='utf-8')
    r = node(load_js(data, extra="const a = m.at(m.t0).pos, z = m.at(m.t1).pos; return {a, z, route0: m.at(m.t0).route[0]};"))
    assert r['a'][:2] == r['route0'][:2] and abs(r['z'][0]) < 5000 and abs(r['z'][0] - 15.0) < 2     # the pad; the last place on the map


def test_the_clock_follows_the_knots_then_the_pace():
    run = {'start_epoch': 1000, 'speed': 2, 'clock': [[0, 1000], [20, 1010], [30, 1020]]}      # 2x, then 1x
    r = node("const c = T.clockFn(D.run); return [c(0), c(10), c(20), c(25), c(30), c(40), c(-4)];", run=run)
    assert r == [1000, 1005, 1010, 1015, 1020, 1025, 998]                     # beyond the last knot at `speed` (2)
    r = node("const c = T.clockFn(D.run); return [c(0), c(30)];", run={'start_epoch': 500, 'speed': 2})
    assert r == [500, 515]                                                     # no knots (an import): start_epoch + t / speed
