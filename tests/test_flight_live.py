"""The live flight log: what scripts/utm_server.py writes as things happen (utm/flight_db.py LiveRecorder), reading it while it
is written (WAL), and GET /api/flightlog and /api/track through the server's own request handler. The simulation itself is
not run here; the checkpoint-dependent test at the end is skipped the way tests/test_utm_server.py is."""

import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from utm import flight_db, scenario
from utm.access import Access

ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

SC = {'name': 'live-test', 'seed': 7, 'map': {'map': 'osm-crop'}}
SPEEDS = {'v_cruise': 1.5, 'v_climb': 0.5, 'v_descent': 0.5, 'v_descent_high': 1.0, 'slow_band_m': 3.0, 'start_z': 1.0}


def flight(i=0, operator='op0', kind='flight', t_dep=10.0, arrival=40.0):
    return {'id': f'f{i}' if kind == 'flight' else f'emergency_{i}', 'kind': kind, 'operator': operator, 'from': 'Depot', 'to': 'Shop',
            'status': 'waiting', 't_dep': t_dep, 'arrival': arrival, 'layer_z': 4.0, 'slot': i, 'replans': 0,
            'route': [[-10.0, -13.0, 1.0], [-10.0, -13.0, 4.0], [8.0, -13.0, 4.0], [8.0, -13.0, 1.0]], 'pos': [-10.0, -13.0, 1.0]}


@pytest.fixture
def rec(tmp_path, monkeypatch):
    monkeypatch.setattr(flight_db.time, 'time', lambda: 1_700_000_001)       # the run "started" a second ago: no pace measured yet
    r = flight_db.LiveRecorder(str(tmp_path / 'live.db'), SC, speed=2.0, speeds=SPEEDS, wall=1_700_000_000)
    yield r
    r.close()


def rows(rec, sql, *a):
    db = flight_db.connect(rec.path)                           # a reader of its own, as the HTTP handlers use
    try:
        return [dict(r) for r in db.execute(sql, a)]
    finally:
        db.close()


def fly(rec, f, t0=12.0, t1=41.5):
    """What the server does for one delivery: the booking, the take-off, a 1 Hz track, the landing."""
    rec.booked(f, 4.0, layer=1)
    rec.event({'t': 4.0, 'type': 'booked', 'text': f"op0: {f['id']} Depot → Shop approved", 'flight': f['id'], 'node': 'op0'})
    rec.status(f['id'], 'flying', t0, f)
    for k in range(int(t1 - t0)):
        rec.track(t0 + k, [(f['id'], [-10.0 + k * 0.5, -13.0, 4.0])])
    rec.event({'t': t1, 'type': 'arrived', 'text': f"{f['id']} landed at Shop", 'flight': f['id'], 'node': 'op0'})
    rec.status(f['id'], 'landed', t1, f)


def test_a_run_row_is_written_at_start(rec):
    run = rows(rec, 'SELECT * FROM runs')[0]
    assert run['run_id'].startswith('live-live-test-') and run['source_file'] == 'live' and run['speed'] == 2.0
    assert json.loads(run['speeds'])['v_cruise'] == 1.5 and run['started_at'] == run['recorded_at']


def test_a_booking_becomes_a_flight_row_that_updates_as_it_goes(rec):
    f = flight(0)
    rec.booked(f, 4.0, layer=1)
    row = rows(rec, 'SELECT * FROM flights')[0]
    assert (row['flight_id'], row['status'], row['callsign'], row['drone'], row['origin'], row['dest']) == ('f0', 'Booked', 'OP0-00', 'D01', 'Depot', 'Shop')
    assert (row['sched_dep'], row['sched_arr'], row['t_request'], row['cruise_z'], row['layer']) == (10.0, 40.0, 4.0, 4.0, 1)
    assert (row['origin_x'], row['dest_x']) == (-10.0, 8.0) and row['act_dep'] is None
    rec.status('f0', 'flying', 12.0, f)
    row = rows(rec, 'SELECT * FROM flights')[0]
    assert (row['status'], row['act_dep'], row['dep_delay']) == ('In flight', 12.0, 2.0)
    assert [e['type'] for e in rows(rec, 'SELECT * FROM events')] == ['takeoff']


def test_a_delivery_is_logged_end_to_end(rec):
    f = flight(0)
    fly(rec, f)
    row = rows(rec, 'SELECT * FROM flights')[0]
    assert (row['status'], row['act_arr'], row['arr_delay'], row['duration_s']) == ('Landed', 41.5, 1.5, 29.5)
    assert row['distance_m'] == pytest.approx(0.5 * 28, abs=0.01)                       # 29 track points, 28 steps of 0.5 m
    assert [e['type'] for e in rows(rec, 'SELECT * FROM events ORDER BY t, seq')] == ['booked', 'takeoff', 'arrived']
    track = rows(rec, 'SELECT t, x FROM track ORDER BY t')
    assert len(track) == 29 and track[0] == {'t': 12.0, 'x': -10.0} and track[-1]['x'] == pytest.approx(4.0)


def test_replans_and_a_return_to_base_are_logged(rec):
    f = flight(0)
    rec.booked(f, 4.0)
    rec.status('f0', 'flying', 12.0, f)
    f.update(replans=2, layer_z=6.0)
    rec.replanned(f)
    rec.event({'t': 20.0, 'type': 'replan', 'text': 'f0 re-planned', 'flight': 'f0', 'tag': 'back to its pad'})
    rec.status('f0', 'landed', 30.0, f)
    row = rows(rec, 'SELECT * FROM flights')[0]
    assert (row['status'], row['n_replans'], row['cruise_z'], row['act_arr'], row['arr_delay']) == ('Returned', 2, 6.0, None, None)


def test_a_crash_and_a_cancellation(rec):
    a, b = flight(0), flight(1)
    rec.booked(a, 1.0)
    rec.booked(b, 1.0)
    rec.status('f0', 'flying', 12.0, a)
    rec.status('f0', 'crashed', 20.0, a)
    rec.status('f1', 'cancelled', 9.0, b)
    assert {r['flight_id']: r['status'] for r in rows(rec, 'SELECT * FROM flights')} == {'f0': 'Crashed', 'f1': 'Cancelled'}


def test_refused_bookings_are_rejected_rows_with_their_own_ids(rec):
    rec.booked(flight(0), 1.0)
    rec.event({'t': 6.0, 'type': 'rejected', 'text': 'op0: f1 Depot → Shop refused (no free slot)', 'flight': 'f1', 'node': 'op0',
               'reason': 'no free slot', 'frm': 'Depot', 'to': 'Shop'})
    assert rec.refused(7.0, 'op1', 'Depot', 'Shop', 'no drone free') == 'r2'              # the server reuses f1: r<n> never collides
    by = {r['flight_id']: r for r in rows(rec, 'SELECT * FROM flights')}
    assert sorted(by) == ['f0', 'r1', 'r2'] and by['r1']['status'] == 'Rejected' and by['r1']['operator'] == 'op0'
    assert by['r2']['t_request'] == 7.0
    assert by['r1']['callsign'] == 'OP0-R01' and by['r2']['callsign'] == 'OP1-R02' and by['f0']['callsign'] == 'OP0-00'   # never f1's OP0-01
    ev = rows(rec, "SELECT * FROM events WHERE flight_id = 'r1'")
    assert ev[0]['type'] == 'refused' and ev[0]['text'] == 'r1: booking refused (no free slot)'


def test_flights_left_open_are_closed_when_the_server_stops_or_died(tmp_path):
    path = str(tmp_path / 'live.db')
    r = flight_db.LiveRecorder(path, SC, wall=1_700_000_000)
    r.booked(flight(0), 1.0)
    r.booked(flight(1), 1.0)
    r.status('f1', 'flying', 12.0, flight(1))
    assert r.close() == 2                                       # f0 Booked -> Not departed, f1 In flight -> In flight at end
    assert r.close() == 0 and r.booked(flight(2), 2.0) is None          # a call after close does nothing
    db = flight_db.connect(path)
    assert {x['flight_id']: x['status'] for x in db.execute('SELECT * FROM flights')} == {'f0': 'Not departed', 'f1': 'In flight at end'}
    died = flight_db.LiveRecorder(path, SC, wall=1_700_000_100)           # a second session: its own run
    died.booked(flight(0), 1.0)
    died.db.close()                                                       # ... that dies without closing
    again = flight_db.LiveRecorder(path, SC, wall=1_700_000_200)
    st = {(x['run_id'], x['flight_id']): x['status'] for x in again.db.execute('SELECT * FROM flights')}
    assert set(st.values()) == {'Not departed', 'In flight at end'} and len({k[0] for k in st}) == 2
    assert len(list(again.db.execute('SELECT * FROM runs'))) == 3
    again.close()


def test_connections_use_wal_and_wait_for_locks(tmp_path):
    db = flight_db.connect(str(tmp_path / 'x.db'))
    assert db.execute('PRAGMA journal_mode').fetchone()[0] == 'wal'
    assert db.execute('PRAGMA busy_timeout').fetchone()[0] == flight_db.BUSY_TIMEOUT_MS == 5000


def test_the_log_can_be_read_while_the_server_writes(tmp_path):
    """A writer logging a 1 Hz track in its own transactions, readers building the flight log JSON and the track meanwhile:
    nobody gets 'database is locked', and the readers see the writer's progress."""
    r = flight_db.LiveRecorder(str(tmp_path / 'live.db'), SC, wall=1_700_000_000)
    f = flight(0)
    r.booked(f, 1.0)
    r.status('f0', 'flying', 2.0, f)
    errors, seen, stop = [], [], threading.Event()

    def writer():
        try:
            for k in range(300):
                r.track(2.0 + k, [('f0', [float(k), 0.0, 4.0])])
                r.event({'t': 2.0 + k, 'type': 'log', 'text': f'tick {k}', 'flight': 'f0'})
        except Exception as e:                                   # noqa: BLE001
            errors.append(e)
        finally:
            stop.set()

    def reader():
        db = flight_db.connect(r.path)
        try:
            while not stop.is_set():
                log = flight_db.flightlog_dict(db, active_run=r.run_id, now_t=1.0)
                seen.append(len(flight_db.track_dict(db, r.run_id, 'f0')['track']))
                assert log['live'] and log['flights'][0]['status'] == 'In flight'
        except Exception as e:                                   # noqa: BLE001
            errors.append(e)
        finally:
            db.close()

    threads = [threading.Thread(target=writer)] + [threading.Thread(target=reader) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(60)
    assert not errors                                                   # no 'database is locked' on either side
    assert len(seen) > 3 and max(seen) > 0 and min(seen) < max(seen)    # the readers ran during the writes and saw them grow
    assert len(rows(r, 'SELECT * FROM track')) == 300
    r.close()


def test_the_flight_log_json_marks_the_live_run_and_scales_the_clock(rec):
    fly(rec, flight(0))
    rec.refused(9.0, 'op1', 'Depot', 'Shop', 'no drone free')
    db = flight_db.connect(rec.path)
    log = flight_db.flightlog_dict(db, active_run=rec.run_id, now_t=50.0)
    assert log['live'] and log['active_run'] == rec.run_id and log['now_t'] == 50.0
    run = log['runs'][0]
    assert run['speed'] == 2.0 and run['speeds']['v_climb'] == 0.5 and run['start_epoch'] == 1_700_000_000
    assert {f['flight_id']: f['status'] for f in log['flights']} == {'f0': 'Landed', 'r1': 'Rejected'}
    assert 'live' not in flight_db.flightlog_dict(db)               # a static export has no live flag
    t = flight_db.track_dict(db, 'live-test', 'f0')                 # the run by scenario name
    assert t['flight']['status'] == 'Landed' and len(t['track']) == 29 and t['run']['speeds']['v_cruise'] == 1.5
    assert [e['type'] for e in t['events']] == ['booked', 'takeoff', 'arrived']
    with pytest.raises(KeyError):
        flight_db.track_dict(db, rec.run_id, 'nope')
    db.close()


# ---------------------------------------------------------------- the HTTP endpoints, through the server's handler
class StubLive:
    """What the handler uses of LiveUTM, without a simulation."""

    def __init__(self, rec, path):
        self.rec, self.db_path, self.snap, self.notes = rec, path, {'t': 33.0}, []

    def scene(self):
        return {'name': 'stub'}

    def state(self, since=0):
        return {'t': 33.0, 'flights': [], 'events': [], 'n_events': 0}

    def request(self, operator, to, frm=None, pad=None):
        return {'status': 'rejected', 'reason': 'no drone free'}

    def emergency(self, operator, frm, to, layer=None):
        return {'status': 'rejected', 'reason': 'unknown place'}

    def note_reply(self, reply, body):
        self.notes.append((reply, body))


@pytest.fixture
def server(rec):
    import utm_server
    fly(rec, flight(0))
    live = StubLive(rec, rec.path)
    srv = ThreadingHTTPServer(('127.0.0.1', 0), utm_server.handler(live, os.path.join(ROOT, 'dashboard'), Access(no_auth=True)))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield live, f'http://127.0.0.1:{srv.server_address[1]}'
    srv.shutdown()


def get(url):
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_api_flightlog_serves_the_flight_log_with_the_live_run(server):
    live, url = server
    code, log = get(url + '/api/flightlog')
    assert code == 200 and log['live'] is True and log['active_run'] == live.rec.run_id and log['now_t'] == 33.0
    assert [f['flight_id'] for f in log['flights']] == ['f0'] and log['flights'][0]['status'] == 'Landed'
    assert log['runs'][0]['speed'] == 2.0


def test_api_track_serves_a_flights_history_and_track(server):
    live, url = server
    code, t = get(url + '/api/track?flight=f0')                            # the live run is the default
    assert code == 200 and len(t['track']) == 29 and t['flight']['callsign'] == 'OP0-00'
    assert get(url + f'/api/track?run={live.rec.run_id}&flight=f0')[0] == 200
    code, err = get(url + '/api/track?flight=nope')
    assert code == 404 and 'no flight' in err['reason']


def test_the_endpoints_are_404_with_no_db_and_static_files_still_serve(server):
    live, url = server
    live.db_path = None
    assert get(url + '/api/flightlog')[0] == 404 and get(url + '/api/track?flight=f0')[0] == 404
    with urllib.request.urlopen(url + '/flight_log.html', timeout=10) as r:      # the dashboard pages are still served
        assert r.status == 200


def test_a_refusal_that_never_became_a_flight_is_noted_for_the_log(server):
    live, url = server
    body = json.dumps({'operator': 'op0', 'to': 'Shop'}).encode()
    with urllib.request.urlopen(urllib.request.Request(url + '/api/request', data=body, method='POST'), timeout=10) as r:
        assert json.loads(r.read()) == {'status': 'rejected', 'reason': 'no drone free'}
    assert live.notes == [({'status': 'rejected', 'reason': 'no drone free'}, {'operator': 'op0', 'to': 'Shop'})]


# ---------------------------------------------------------------- the real server (needs the checkpoint)
SCENE = {'name': 'live-db-test', 'map': {'map': 'osm-crop'},
         'places': {'Depot': {'kind': 'hub', 'operator': 'op0', 'pads': [[-10.0, -13.0], [-7.0, -13.0]]},
                    'Shop': {'kind': 'drop', 'at': [8.0, -13.0]}},
         'requests': [{'t': 0, 'operator': 'op0', 'to': 'Shop'}],
         'sim': {'checkpoint': '../utm-swarm-marl/checkpoints/stage2_r1_crowd_rl_r7v3ft_s0_lab/checkpoint_final.pth'}}


def test_the_live_utm_writes_the_flight_log(tmp_path):
    import run_scenario
    sc = scenario.load(SCENE)
    try:
        run_scenario.checkpoint_path(sc['sim']['checkpoint'])
    except FileNotFoundError:
        pytest.skip('the shipped checkpoint is not on this machine')
    import utm_server
    live = utm_server.LiveUTM(sc, slots=2, db=str(tmp_path / 'live.db'))
    assert live.scene()['speeds']['v_cruise'] == 1.5
    assert live.request('op0', 'Shop')['status'] == 'approved'
    live.note_reply(live.request('op0', 'Nowhere'), {'operator': 'op0', 'to': 'Nowhere'})
    f = live.flights['f0']
    for _ in range(int(120 / live.dt)):
        live.step()
        if f['status'] == 'landed':
            break
    live.close()
    db = flight_db.connect(str(tmp_path / 'live.db'))
    by = {r['flight_id']: dict(r) for r in db.execute('SELECT * FROM flights')}
    assert by['f0']['status'] == 'Landed' and by['f0']['act_dep'] is not None and by['f0']['distance_m'] > 10
    assert by['r1']['status'] == 'Rejected'
    assert db.execute("SELECT count(*) FROM track WHERE flight_id = 'f0'").fetchone()[0] > 10
    assert db.execute("SELECT count(*) FROM events WHERE flight_id = 'f0' AND type = 'takeoff'").fetchone()[0] == 1


def test_the_runs_speed_follows_the_pace_the_simulation_really_runs_at(tmp_path, monkeypatch):
    """Asked for 2x, the server fell behind: 30 simulated seconds took 20 wall seconds (1.5x). runs.speed starts as the asked
    pace and becomes the measured one, so clock time = started_at + t / speed stays near the wall clock."""
    wall = 1_700_000_000
    r = flight_db.LiveRecorder(str(tmp_path / 'live.db'), SC, speed=2.0, speeds=SPEEDS, wall=wall)
    speed = lambda: rows(r, 'SELECT speed FROM runs')[0]['speed']          # noqa: E731
    r.booked(flight(0), 1.0)
    monkeypatch.setattr(flight_db.time, 'time', lambda: wall + 3)
    r.track(6.0, [])                                                       # too early to say anything
    assert speed() == 2.0
    monkeypatch.setattr(flight_db.time, 'time', lambda: wall + 20)
    r.track(30.0, [])
    assert speed() == pytest.approx(1.5)
    monkeypatch.setattr(flight_db.time, 'time', lambda: wall + 22)
    r.track(33.0, [])                                                      # within 5 s of the last measurement: unchanged
    assert speed() == pytest.approx(1.5)
    monkeypatch.setattr(flight_db.time, 'time', lambda: wall + 40)
    r.t_last = 60.0
    r.close()                                                              # the final measurement
    db = flight_db.connect(r.path)
    assert db.execute('SELECT speed FROM runs').fetchone()[0] == pytest.approx(1.5)
    db.close()


def test_the_clock_keeps_when_each_simulated_second_really_happened(tmp_path, monkeypatch):
    """The pace varies with the load (fast while idle, slow while flying), so one speed cannot give the clock: the recorder
    keeps (t, wall) knots, thinned for the pages, with the last one always kept."""
    wall = 1_700_000_000
    r = flight_db.LiveRecorder(str(tmp_path / 'live.db'), SC, speed=2.0, speeds=SPEEDS, wall=wall)
    now = wall
    for t in range(1, 31):                                                 # 2x for 10 s, then 1x
        now = wall + (t / 2 if t <= 20 else 10 + (t - 20))
        monkeypatch.setattr(flight_db.time, 'time', lambda now=now: now)
        r.track(float(t), [])
    knots = r.clock()
    assert knots[0] == [1.0, wall + 0.5] and knots[-1] == [30.0, wall + 20]
    assert all(b[0] - a[0] >= 10 or b is knots[-1] for a, b in zip(knots, knots[1:]))
    assert flight_db.flightlog_dict(r.db)['runs'][0]['clock'] == knots
    r.close()
