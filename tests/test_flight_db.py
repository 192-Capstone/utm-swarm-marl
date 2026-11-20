"""The flight log database (utm/flight_db.py): import, idempotence, statuses, delays, the dashboard export."""

import json
import sqlite3
import sys

import numpy as np
import pytest

from utm import flight_db, flight_log

sys.path.insert(0, 'scripts')


def drone(i, kind='flight', operator='op0', t_dep=2.0, arrival=20.0):
    return {'id': i, 'kind': kind, 'operator': operator, 'start': [float(len(i)), 0.0], 'goal': [10.0, 5.0],
            't_dep': t_dep, 'arrival': arrival, 'layer': 1, 'route': [[0, 0, 0.5], [5, 0, 3.0], [10, 5, 0.5]]}


def recording(tmp_path, requests=True, name='run.json.gz', places=None):
    """f0 lands 2 s late, f1 crashes, f2 is forced down, f3 is cancelled, f4 is still flying, f5 is refused
    (a request only), a bird flies by, an intruder crosses f0 and a no-fly zone closes."""
    ids = ['f0', 'f1', 'f2', 'f3', 'f4', 'intruder_0', 'bird_0']
    kinds = ['flight'] * 5 + ['intruder', 'bird']
    rec = flight_log.FlightLog({'name': 'tiny', 'seed': 3, 'map': {'map': 'city'}, **({'places': places} if places else {})})
    rec.log['drones'] = [drone(i, k, 'op1' if i == 'f1' else 'op0') for i, k in zip(ids, kinds)]
    rec.log['drones'][5].update(t_dep=None, arrival=None)
    for k in range(40):                                          # 0.5 s frames, 20 s
        t = 0.5 * (k + 1)
        pos = np.zeros((7, 3))
        pos[:, 1] = np.arange(7) * 20.0                          # lanes far apart ...
        pos[0] = [t, 0.0, 2.0]                                   # f0 flies along x
        pos[5] = [t, 1.5, 2.0]                                   # ... but the intruder passes f0 at 1.5 m
        pos[6] = [t, 0.2, 2.0]                                   # the bird passes closer still: not counted
        status = [1 if t > 2.5 else 0, 3 if t > 5 else 1, 2 if t > 8 else 1, 4 if t > 3 else 0, 1, 1, 1]
        status[0] = 2 if t > 12 else status[0]
        rec.frame(t, pos, status)
    rec.event(12.0, 'arrived', flight='f0')
    rec.event(5.0, 'crashed', flight='f1')
    rec.event(8.0, 'failure', flight='f2', failure_type='land')
    rec.event(8.0, 'forced_down', flight='f2')
    rec.event(3.0, 'replan', flight='f3', why='zone_0 -> cancelled', status='approved', reason='x')
    rec.event(4.0, 'replan', flight='f0', why='late', status='approved', reason='ok')
    rec.event(4.0, 'alert', flight='f0', state='late', risk='high', other=None)
    rec.event(6.0, 'zone_on', zone='zone_0')
    rec.event(6.0, 'alert', flight='bird_0', state='unknown', risk='high', other=None)
    if requests:
        rec.log['requests'] = [
            {'id': 'f0', 'operator': 'op0', 'start': [2.0, 0.0], 'goal': [10.0, 5.0], 't_request': 0.0, 'status': 'approved', 'reason': ''},
            {'id': 'f5', 'operator': 'op1', 'start': [3.0, 3.0], 'goal': [10.0, 5.0], 't_request': 1.5, 'status': 'rejected',
             'reason': 'no free slot'}]
    rec.finish({'arrived': 1})
    path = tmp_path / name
    rec.save(path)
    return path


@pytest.fixture
def db(tmp_path):
    d = flight_db.connect(str(tmp_path / 'flights.db'))
    flight_db.import_recording(d, recording(tmp_path))
    return d


def by_id(db):
    return {r['flight_id']: r for r in db.execute('SELECT * FROM flights')}


def test_import_twice_does_not_duplicate(tmp_path, db):
    n = (db.execute('SELECT count(*) FROM flights').fetchone()[0], db.execute('SELECT count(*) FROM events').fetchone()[0])
    flight_db.import_recording(db, recording(tmp_path, name='copy.json.gz'))        # the same run, another file
    assert db.execute('SELECT count(*) FROM runs').fetchone()[0] == 1
    assert (db.execute('SELECT count(*) FROM flights').fetchone()[0], db.execute('SELECT count(*) FROM events').fetchone()[0]) == n


def test_statuses_follow_the_precedence(db):
    f = by_id(db)
    assert {k: v['status'] for k, v in f.items()} == {
        'f0': 'Landed', 'f1': 'Crashed', 'f2': 'Forced down', 'f3': 'Cancelled', 'f4': 'In flight at end',
        'intruder_0': 'In flight at end', 'f5': 'Rejected'}
    assert 'bird_0' not in f                                                         # a bird is not a flight


def test_times_delays_and_counts(db):
    f0 = by_id(db)['f0']
    assert (f0['sched_dep'], f0['act_dep'], f0['sched_arr'], f0['act_arr']) == (2.0, 3.0, 20.0, 12.0)
    assert (f0['dep_delay'], f0['arr_delay']) == (1.0, -8.0)
    assert (f0['n_replans'], f0['n_alerts'], f0['origin'], f0['callsign']) == (1, 1, 'P01', 'OP0-00')
    assert f0['distance_m'] == pytest.approx(9.0 - 0.5, abs=1.0) and f0['cruise_z'] == 3.0
    assert f0['min_sep_m'] == pytest.approx(1.5) and f0['min_sep_with'] == 'intruder_0'   # not the bird
    f1 = by_id(db)['f1']
    assert f1['act_arr'] is None and f1['arr_delay'] is None


def test_intruder_has_no_schedule_and_refused_request_no_times(db):
    f = by_id(db)
    assert f['intruder_0']['kind'] == 'intruder' and f['intruder_0']['sched_dep'] is None
    r = f['f5']
    assert r['status'] == 'Rejected' and r['t_request'] == 1.5 and r['act_dep'] is None and r['sched_dep'] is None
    text = [e['text'] for e in db.execute("SELECT * FROM events WHERE flight_id = 'f5'")]
    assert text == ['f5: booking refused (no free slot)']


def test_old_recordings_without_requests_import(tmp_path):
    d = flight_db.connect(':memory:')
    flight_db.import_recording(d, recording(tmp_path, requests=False))
    assert 'f5' not in by_id(d) and by_id(d)['f0']['status'] == 'Landed'


def test_history_and_queries(db):
    row, events = flight_db.flight_history(db, 'tiny', 'OP0-02')                    # by callsign
    assert row['flight_id'] == 'f2'
    assert [e['type'] for e in events] == ['takeoff', 'failure', 'forced_down']
    assert 'f2: forced landing ordered' in [e['text'] for e in events]
    assert [r['flight_id'] for r in flight_db.query_flights(db, status='crashed')] == ['f1']
    assert {r['operator'] for r in flight_db.query_flights(db, operator='op1')} == {'op1'}
    with pytest.raises(KeyError):
        flight_db.flight_history(db, 'tiny', 'nope')


def test_export_json(tmp_path, db):
    out = tmp_path / 'data' / 'flight_log.json'
    assert flight_db.export_json(db, str(out)) == (1, 7)
    j = json.loads(out.read_text(encoding='utf-8'))
    assert j['schema'] == flight_db.EXPORT_SCHEMA and j['runs'][0]['scenario'] == 'tiny' and j['runs'][0]['start_epoch'] > 0
    assert [e['text'] for e in j['runs'][0]['events']] == ['No-fly zone closes the area']      # run-level events
    f0 = next(f for f in j['flights'] if f['flight_id'] == 'f0')
    assert f0['status'] == 'Landed' and [e['t'] for e in f0['events']] == sorted(e['t'] for e in f0['events'])
    assert any(e['type'] == 'takeoff' for e in f0['events'])


def test_a_schema_mismatch_is_refused(tmp_path):
    import gzip
    p = tmp_path / 'old.json.gz'
    with gzip.open(p, 'wt') as f:
        json.dump({'schema': 'utm-flight-log/1'}, f)
    with pytest.raises(ValueError):
        flight_db.import_recording(sqlite3.connect(':memory:'), str(p))


def test_pads_are_coded_by_position_with_their_coordinates(db):
    f = by_id(db)
    assert (f['f0']['origin'], f['f0']['origin_x'], f['f0']['origin_y']) == ('P01', 2.0, 0.0)
    assert (f['f0']['dest'], f['f0']['dest_x'], f['f0']['dest_y']) == ('P03', 10.0, 5.0)
    assert f['f5']['origin'] == 'P02' and f['f5']['dest'] == 'P03'            # a refused request's pads too
    assert f['intruder_0']['origin'] == '(10, 0)'                              # an intruder's ends are not pads


def test_named_places_are_used_where_the_scenario_has_them(tmp_path):
    places = {'Depot': {'kind': 'hub', 'operator': 'op0', 'pads': [[3.0, 3.0], [2.0, 0.0]]},
              'Market': {'kind': 'drop', 'at': [10.0, 5.0]}}
    d = flight_db.connect(':memory:')
    flight_db.import_recording(d, recording(tmp_path, places=places))
    f = by_id(d)
    assert (f['f0']['origin'], f['f0']['dest'], f['f5']['origin']) == ('Depot pad 2', 'Market', 'Depot pad 1')


def test_drone_slot_duration_and_filing_time(db):
    f = by_id(db)
    assert (f['f0']['drone'], f['f4']['drone'], f['f5']['drone']) == ('D01', 'D05', None)
    assert f['f0']['duration_s'] == 9.0 and f['f1']['duration_s'] is None      # act_arr 12 - act_dep 3; a crash has none
    assert f['f0']['t_request'] == 0.0 and f['f5']['t_request'] == 1.5


def test_a_database_from_an_older_version_gets_the_new_columns(tmp_path):
    path = str(tmp_path / 'old.db')
    old = sqlite3.connect(path)
    old.execute('CREATE TABLE flights (run_id TEXT NOT NULL, flight_id TEXT NOT NULL, callsign TEXT, PRIMARY KEY (run_id, flight_id))')
    old.commit()
    old.close()
    d = flight_db.connect(path)
    flight_db.import_recording(d, recording(tmp_path))
    assert by_id(d)['f0']['drone'] == 'D01'


def test_the_start_clock_is_the_file_time_minus_the_run_length(db):
    import datetime
    r = db.execute('SELECT * FROM runs').fetchone()
    gap = datetime.datetime.fromisoformat(r['recorded_at']) - datetime.datetime.fromisoformat(r['started_at'])
    assert gap.total_seconds() == pytest.approx(20.0, abs=1.0)                  # the last frame is at 20 s


def late_recording(tmp_path):
    """One flight booked to leave at 10 s and land at 14 s that the scenario's disruptions hold back 8 s: the recording
    has t_dep 18 (booked + lateness) but E1's booking window, which starts pad_before (2 s) before the booked take-off
    and ends pad_after (10 s) after the booked arrival. It flies 18 -> 22 s."""
    rec = flight_log.FlightLog({'name': 'late', 'seed': 1, 'map': {'map': 'city'}})
    rec.log['drones'] = [drone('A', t_dep=18.0, arrival=14.0)]
    rec.log['bookings'] = {'A': [['c0', 1, 8.0, 24.0], ['c1', 1, 12.0, 24.0]]}
    for k in range(50):
        t = 0.5 * (k + 1)
        rec.frame(t, np.array([[min(t, 22.0), 0.0, 1.0]]), [0 if t < 18 else 1 if t < 22 else 2])
    rec.event(22.0, 'arrived', flight='A')
    rec.finish({})
    path = tmp_path / 'late.json.gz'
    rec.save(path)
    return path


def test_the_schedule_is_e1s_booking_and_injected_lateness_is_a_departure_delay(tmp_path):
    d = flight_db.connect(':memory:')
    flight_db.import_recording(d, late_recording(tmp_path))
    a = by_id(d)['A']
    assert a['sched_dep'] == pytest.approx(10.0) and a['sched_arr'] == 14.0      # not the recorded t_dep of 18
    assert a['sched_dep'] <= a['sched_arr']
    assert a['act_dep'] == pytest.approx(18.0, abs=0.6)
    assert a['dep_delay'] == pytest.approx(8.0, abs=0.6)                          # the injected lateness
    assert a['arr_delay'] == pytest.approx(8.0, abs=0.01) and a['status'] == 'Landed'
    assert a['duration_s'] == pytest.approx(4.0, abs=0.6)                         # flown about as planned (4 s)


def test_an_emergency_flight_with_no_named_operator_takes_the_site_it_leaves_from(tmp_path):
    rec = flight_log.FlightLog({'name': 'em', 'places': {'City Hospital': {'kind': 'hospital', 'at': [1.0, 1.0]}}})
    e = drone('emergency_0', 'emergency', 'emergency')
    e['start'] = [1.0, 1.0]
    rec.log['drones'] = [e]
    rec.frame(1.0, np.array([[1.0, 1.0, 0.5]]), [0])
    rec.finish({})
    path = tmp_path / 'em.json.gz'
    rec.save(path)
    d = flight_db.connect(':memory:')
    flight_db.import_recording(d, path)
    row = by_id(d)['emergency_0']
    assert (row['operator'], row['callsign']) == ('City Hospital', 'CIT-E00')
