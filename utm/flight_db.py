"""
flight_db.py
------------
A persistent flight log across scenario runs, like an airport's departures board plus per-flight history:
one SQLite file (stdlib sqlite3) filled from the recordings of utm/flight_log.py. Read by the dashboard
(dashboard/flight_log.html, through export_json) and by scripts/flight_db.py.

    runs     one row per recording (run_id = scenario name + a hash of the recording's content)
    flights  one row per flight / emergency flight / intruder, and per refused booking request
    events   the run's events in plain words (the wording of scripts/render_scenario.py), plus take-offs
             and refusals

Importing is idempotent: the same recording again replaces that run's rows. A run's start clock is
runs.started_at: the recording's file time minus its last frame's time (the recording keeps no wall-clock
duration); clock time of an event = that + sim time. Drone kinds this module does not
know are kept as they are (no schedule); 'bird' is not a flight and is skipped. Recordings without the
optional top-level 'requests' key (scripts/run_scenario.py adds it) simply have no Rejected rows.

Status, by precedence: Crashed > Forced down > Returned > Landed > Cancelled > Rejected > In flight at end
(a flight that never left its pad and was not cancelled: Not departed). Returned = E4 sent it back to its pad.
"""

import datetime
import gzip
import functools
import hashlib
import json
import os
import re
import sqlite3
import sys
import threading
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'scripts'))

from render_scenario import STATE_WORDS, describe  # noqa: E402
from utm import flight_log, speeds, strategic_planner  # noqa: E402

DEFAULT_DB = os.path.join('eval_results', 'flight_log.db')
EXPORT_SCHEMA = 'utm-flight-log-db/1'
EMERGENCY_OPERATOR = 'Hospital'                   # who files an emergency flight when the scenario names no operator
SCHEDULED = ('flight', 'emergency')               # kinds with a booked schedule
SKIPPED = ('bird',)                               # not flights
ALERT_WORDS = {'lost_link': 'lost its radio link', 'down': 'is down', 'emergency_landing': 'is making an emergency landing'}
FAILURES = {'motor': 'motor failure', 'land': 'forced landing ordered', 'comms': 'lost its radio link'}

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY, scenario TEXT, map TEXT, seed INTEGER, source_file TEXT,
    recorded_at TEXT, summary TEXT, started_at TEXT);
CREATE TABLE IF NOT EXISTS events (
    run_id TEXT NOT NULL, seq INTEGER, t REAL, flight_id TEXT, type TEXT, text TEXT,
    PRIMARY KEY (run_id, seq));
CREATE INDEX IF NOT EXISTS events_flight ON events (run_id, flight_id);
CREATE TABLE IF NOT EXISTS track (
    run_id TEXT NOT NULL, flight_id TEXT NOT NULL, t REAL NOT NULL, x REAL, y REAL, z REAL,
    PRIMARY KEY (run_id, flight_id, t));
CREATE TABLE IF NOT EXISTS clock (
    run_id TEXT NOT NULL, t REAL NOT NULL, wall REAL NOT NULL,
    PRIMARY KEY (run_id, t));
"""
RUN_COLUMNS = (('started_at', 'TEXT'), ('speed', 'REAL'), ('speeds', 'TEXT'))   # added after the first version
FLIGHT_SCHEMA = (('run_id', 'TEXT NOT NULL'), ('flight_id', 'TEXT NOT NULL'), ('callsign', 'TEXT'), ('drone', 'TEXT'),
                 ('operator', 'TEXT'), ('kind', 'TEXT'), ('origin', 'TEXT'), ('dest', 'TEXT'),
                 ('origin_x', 'REAL'), ('origin_y', 'REAL'), ('dest_x', 'REAL'), ('dest_y', 'REAL'),
                 ('layer', 'INTEGER'), ('cruise_z', 'REAL'), ('t_request', 'REAL'),
                 ('sched_dep', 'REAL'), ('act_dep', 'REAL'), ('sched_arr', 'REAL'), ('act_arr', 'REAL'),
                 ('dep_delay', 'REAL'), ('arr_delay', 'REAL'), ('duration_s', 'REAL'), ('status', 'TEXT'),
                 ('n_replans', 'INTEGER'), ('n_alerts', 'INTEGER'), ('distance_m', 'REAL'),
                 ('min_sep_m', 'REAL'), ('min_sep_with', 'TEXT'))
FLIGHT_COLS = tuple(c for c, _ in FLIGHT_SCHEMA)


BUSY_TIMEOUT_MS = 5000


def connect(path=DEFAULT_DB, check_same_thread=True):
    """Opens the database (and creates or upgrades its tables). Every connection uses WAL and waits up to 5 s for a
    lock, so the live server can write while its HTTP handlers (and the CLI) read: a reader never blocks the writer."""
    if path != ':memory:' and os.path.dirname(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    db = sqlite3.connect(path, timeout=BUSY_TIMEOUT_MS / 1000, check_same_thread=check_same_thread)
    db.row_factory = sqlite3.Row
    db.execute(f'PRAGMA busy_timeout = {BUSY_TIMEOUT_MS}')
    db.execute('PRAGMA journal_mode = WAL')
    db.executescript(SCHEMA_SQL)
    cols = ', '.join(f'{c} {t}' for c, t in FLIGHT_SCHEMA)
    db.execute(f'CREATE TABLE IF NOT EXISTS flights ({cols}, PRIMARY KEY (run_id, flight_id))')
    for table, schema in (('flights', FLIGHT_SCHEMA), ('runs', RUN_COLUMNS)):
        have = {r['name'] for r in db.execute(f'PRAGMA table_info({table})')}
        for c, t in schema:                               # a database made by an older version: add what is new
            if c not in have:
                db.execute(f'ALTER TABLE {table} ADD COLUMN {c} {t.replace(" NOT NULL", "")}')
    return db


def _read(path):
    """(log, content hash): the recording, and a hash of its uncompressed JSON (a gzip header carries a time)."""
    with gzip.open(path, 'rb') as f:
        raw = f.read()
    log = json.loads(raw.decode('utf-8'))
    if log.get('schema') != flight_log.SCHEMA:
        raise ValueError(f"{path}: schema {log.get('schema')!r}, expected {flight_log.SCHEMA!r}")
    return log, hashlib.sha1(raw).hexdigest()[:8]


def callsign(operator, kind, flight_id):
    """Short and stable: operator prefix + number (OP0-07); E after the prefix for an emergency flight
    (HOS-E00); INT-00 for an intruder. The number is the one in the flight's id."""
    m = re.search(r'(\d+)\s*$', str(flight_id))
    num = f'{int(m.group(1)):02d}' if m else str(flight_id)
    prefix = re.sub(r'[^A-Za-z0-9]', '', str(operator or '')).upper()[:3]
    if kind == 'intruder':
        return f'INT-{num}'
    if kind == 'emergency':
        return f'{prefix or "EMG"}-E{num}'
    return f'{prefix or "FLT"}-{num}'


class _Pads:
    """Names for the pads of one run: the scenario's place name ('West Hub pad 2'; a single-pad place such as a
    drop point or the hospital: its name), else a short code (P01, P02, ...) numbered by position (x, then y),
    so the same layout always gives the same codes. Intruders' ends are not pads: plain coordinates."""

    def __init__(self, sc, log):
        self.sites = []
        for name, p in (sc.get('places') or {}).items():
            pads = p['pads'] if p.get('kind') == 'hub' else [p['at']]
            self.sites.append((name, pads, len(pads) > 1))
        ends = [d[k] for d in log['drones'] if d.get('kind') in SCHEDULED for k in ('start', 'goal') if d.get(k)]
        ends += [r[k] for r in log.get('requests', []) for k in ('start', 'goal') if r.get(k)]
        free = sorted({(round(x, 1), round(y, 1)) for x, y in ends if not self.named(x, y)})
        self.codes = {xy: f'P{k + 1:02d}' for k, xy in enumerate(free)}

    def named(self, x, y):
        for name, pads, several in self.sites:
            for k, q in enumerate(pads):
                if np.hypot(x - q[0], y - q[1]) < 0.5:
                    return f'{name} pad {k + 1}' if several else name
        return None

    def __call__(self, xy, pad=True):
        """(label, x, y) of an end point."""
        if xy is None:
            return None, None, None
        x, y = round(float(xy[0]), 1), round(float(xy[1]), 1)
        label = self.named(x, y) if pad else None
        return label or (self.codes.get((x, y)) if pad else None) or f'({round(x)}, {round(y)})', x, y


def _iso(epoch):
    return datetime.datetime.fromtimestamp(epoch, datetime.timezone.utc).isoformat(timespec='seconds')


def _row(**kw):
    return {c: kw.get(c) for c in FLIGHT_COLS}


def _cruise_z(route):
    return round(float(max(p[2] for p in route)), 2) if route else None


def _kinematics(log, keep):
    """Per drone (index in log['drones']): first flying time, distance flown, closest approach to another
    drone (m, drone index), from the frames. keep: indices of drones that count (not birds)."""
    fr = log['frames']
    n = len(log['drones'])
    out = {i: {'act_dep': None, 'dist': 0.0, 'sep': None, 'with': None, 'landed_t': None} for i in range(n)}
    if not fr['t']:
        return out
    t = np.asarray(fr['t'], dtype=float)
    pos = np.asarray(fr['pos'], dtype=float).reshape(len(t), n, 3)
    st = np.asarray(fr['status'], dtype=int).reshape(len(t), n)
    for i in range(n):
        fly = np.flatnonzero(st[:, i] == flight_log.FLYING)
        if len(fly):
            out[i]['act_dep'] = float(t[fly[0]])
        both = (st[1:, i] == flight_log.FLYING) & (st[:-1, i] == flight_log.FLYING)
        out[i]['dist'] = float(np.linalg.norm(np.diff(pos[:, i], axis=0), axis=1)[both].sum())
        down = np.flatnonzero(st[:, i] == flight_log.LANDED)
        if len(down):
            out[i]['landed_t'] = float(t[down[0]])
    keep = sorted(keep)
    if len(keep) > 1:
        air = st[:, keep] == flight_log.FLYING
        for a in range(0, len(t), 256):
            p = pos[a:a + 256][:, keep]
            d = np.linalg.norm(p[:, :, None] - p[:, None], axis=3)
            ok = air[a:a + 256][:, :, None] & air[a:a + 256][:, None] & ~np.eye(len(keep), dtype=bool)
            d = np.where(ok, d, np.inf)
            low = d.min(axis=0)                           # per drone, over this chunk's frames
            who = d.min(axis=0).argmin(axis=1)
            for k, i in enumerate(keep):
                m = float(low[k, who[k]])
                if np.isfinite(m) and (out[i]['sep'] is None or m < out[i]['sep']):
                    out[i]['sep'], out[i]['with'] = m, keep[who[k]]
    return out


def _event_text(e, home):
    t, f = e['type'], e.get('flight')
    if t == 'replan' and e.get('status') == 'logged':      # E4 only noted it (an emergency landing): its own words
        return f"{f}: {e.get('why', 'logged')}"
    if t == 'alert':
        words = ALERT_WORDS.get(e['state']) or STATE_WORDS.get(e['state'], e['state'])
        return f"Alert: {f} {words}" if e['risk'] == 'high' else f"{f} {words} (risk {e['risk']})"
    text = describe(e)
    if text:
        return text
    if t == 'arrived':
        return f'{f} back on its pad (returned to base)' if f in home else f'{f} landed at its destination'
    if t == 'crashed':
        return f'{f} crashed'
    if t == 'forced_down':
        return f'{f} brought down (forced landing)'
    if t == 'failure':
        return f"{f}: {FAILURES.get(e.get('failure_type'), e.get('failure_type') or 'failure')}"
    if t == 'replan':
        return f"{f} re-plan {e.get('status', '')} ({e.get('why', '')})"
    return t.replace('_', ' ')


def _status(kind, ev, last_status, act_dep, cancelled, refused):
    """ev: the flight's event types (set); last_status: its last frame status code."""
    if 'crashed' in ev or last_status == flight_log.CRASHED:
        return 'Crashed'
    if 'forced_down' in ev:
        return 'Forced down'
    if 'returned' in ev:
        return 'Returned'
    if 'arrived' in ev or last_status == flight_log.LANDED:
        return 'Landed'
    if cancelled or last_status == flight_log.CANCELLED:
        return 'Cancelled'
    if refused:
        return 'Rejected'
    return 'In flight at end' if act_dep is not None else 'Not departed'


def import_recording(db, path):
    """Imports one recording; returns (run_id, n_flights). Replaces the rows of the same run."""
    log, digest = _read(path)
    sc = log.get('scenario') or {}
    name = sc.get('name') or os.path.basename(path).replace('.json.gz', '')
    run_id = f'{name}-{digest}'
    mtime = os.path.getmtime(path)
    length = log['frames']['t'][-1] if log['frames']['t'] else 0.0       # how long the run was (sim s)
    recorded, started = _iso(mtime), _iso(mtime - length)
    drones = log['drones']
    place = _Pads(sc, log)
    pad_before = float((sc.get('utm') or {}).get('pad_before', strategic_planner.Settings().pad_before))
    keep = {i for i, d in enumerate(drones) if d.get('kind') not in SKIPPED}
    kin = _kinematics(log, keep)
    index = {d['id']: i for i, d in enumerate(drones)}
    events, seen = [], set()
    for e in log['events']:                               # the same alert twice in one instant is one alert
        key = (e['t'], e.get('flight'), e.get('state'), e.get('risk')) if e['type'] == 'alert' else None
        if key is None or key not in seen:
            events.append(e)
            seen.add(key)
    by = {}                                               # flight id -> its events
    for e in events:
        if e.get('flight'):
            by.setdefault(e['flight'], []).append(e)
    home = {f for f, es in by.items() for e in es
            if e['type'] == 'replan' and e.get('status') == 'approved' and e.get('reason') == 'return to base'}
    refused = {r['id']: r for r in log.get('requests', []) if r.get('status') != 'approved'}
    last = log['frames']['status'][-1] if log['frames']['status'] else [None] * len(drones)
    requested = {r['id']: r for r in log.get('requests', [])}

    rows, ev_rows = [], []
    for i, d in enumerate(drones):
        if i not in keep:
            continue
        f, kind = d['id'], d.get('kind', '')
        es = by.get(f, [])
        types = {e['type'] for e in es}
        if f in home:
            types.add('returned')
        k = kin[i]
        sched = kind in SCHEDULED
        t_dep, arr = (d.get('t_dep'), d.get('arrival')) if sched else (None, None)
        if kind == 'flight' and log['bookings'].get(f):   # E1's booked departure, not the recorded one (it has the injected lateness)
            t_dep = min(w[2] for w in log['bookings'][f]) + pad_before
        if kind == 'emergency':                           # its booking is the one E4 approved when it was filed
            ok = [e for e in es if e['type'] == 'replan' and e.get('why') == 'emergency' and e.get('status') == 'approved']
            if ok:
                t_dep, arr = ok[0].get('t_dep'), ok[0].get('arrival')
        arrived = next((e['t'] for e in es if e['type'] == 'arrived' and f not in home), None)
        if arrived is None and 'arrived' not in types and 'returned' not in types and 'forced_down' not in types \
                and 'crashed' not in types and k['landed_t'] is not None:
            arrived = k['landed_t']                       # landed with no event (old recordings)
        cancelled = any(e['type'] == 'replan' and str(e.get('why', '')).endswith('-> cancelled') for e in es)
        emerg_refused = kind == 'emergency' and any(
            e['type'] == 'replan' and e.get('why') == 'emergency' and e.get('status') != 'approved' for e in es)
        status = _status(kind, types, last[i], k['act_dep'], cancelled, emerg_refused)
        n_replans = sum(1 for e in es if e['type'] == 'replan' and e.get('status') == 'approved' and e.get('why') != 'emergency')
        n_alerts = sum(1 for e in es if e['type'] == 'alert')
        other = drones[k['with']]['id'] if k['with'] is not None else None
        dep_delay = None if (k['act_dep'] is None or t_dep is None) else round(k['act_dep'] - float(t_dep), 2)
        arr_delay = None if (arrived is None or arr is None or status != 'Landed') else round(arrived - float(arr), 2)
        route = d.get('route') or []
        origin, goal = place(d.get('start'), sched), place(d.get('goal'), sched)
        filed = requested.get(f, {}).get('t_request')
        if kind == 'emergency':                           # filed when E4 was asked for its booking
            filed = next((e['t'] for e in es if e['type'] == 'replan' and e.get('why') == 'emergency'), None)
        act_arr = arrived if status == 'Landed' else None
        operator = d.get('operator') or ''
        if kind == 'emergency' and operator in ('', 'emergency'):     # the scenario named no operator: the emergency
            operator = EMERGENCY_OPERATOR                              # operator, as ecity_rooftops and the live server call it
        rows.append(_row(
            run_id=run_id, flight_id=f, callsign=callsign(operator, kind, f), drone=f'D{i + 1:02d}',
            operator=operator, kind=kind, origin=origin[0], origin_x=origin[1], origin_y=origin[2],
            dest=goal[0], dest_x=goal[1], dest_y=goal[2], layer=d.get('layer'), cruise_z=_cruise_z(route),
            t_request=filed, sched_dep=t_dep, act_dep=k['act_dep'], sched_arr=arr, act_arr=act_arr,
            dep_delay=dep_delay, arr_delay=arr_delay,
            duration_s=None if act_arr is None or k['act_dep'] is None else round(act_arr - k['act_dep'], 2),
            status=status, n_replans=n_replans, n_alerts=n_alerts, distance_m=round(k['dist'], 2),
            min_sep_m=None if k['sep'] is None else round(k['sep'], 3), min_sep_with=other))
        if k['act_dep'] is not None:
            ev_rows.append((k['act_dep'], f, 'takeoff', f'{f} took off'))
    for r in refused.values():
        f = r['id']
        origin, goal = place(r.get('start')), place(r.get('goal'))
        rows.append(_row(
            run_id=run_id, flight_id=f, callsign=callsign(r.get('operator'), 'flight', f), operator=r.get('operator') or '',
            kind='flight', origin=origin[0], origin_x=origin[1], origin_y=origin[2], dest=goal[0], dest_x=goal[1],
            dest_y=goal[2], t_request=r.get('t_request'), status='Rejected', n_replans=0, n_alerts=0, distance_m=0.0))
        ev_rows.append((float(r.get('t_request') or 0.0), f, 'refused',
                        f"{f}: booking refused" + (f" ({r['reason']})" if r.get('reason') else '')))
    for e in events:
        if e.get('flight') in index and index[e['flight']] not in keep:
            continue                                      # a bird's event
        ev_rows.append((e['t'], e.get('flight'), e['type'], _event_text(e, home)))
    ev_rows.sort(key=lambda r: r[0])

    with db:                                              # one transaction: replace the run
        for table in ('events', 'flights', 'runs'):
            db.execute(f'DELETE FROM {table} WHERE run_id = ?', (run_id,))
        db.execute('INSERT INTO runs (run_id, scenario, map, seed, source_file, recorded_at, summary, started_at, speed, speeds) '
                   'VALUES (?,?,?,?,?,?,?,?,?,?)',
                   (run_id, name, json.dumps(sc.get('map')), sc.get('seed'), os.path.abspath(path), recorded,
                    json.dumps(log.get('summary') or {}), started, 1.0, json.dumps(speeds.from_scenario(sc.get('utm')))))
        db.executemany(f'INSERT INTO flights ({",".join(FLIGHT_COLS)}) VALUES ({",".join(":" + c for c in FLIGHT_COLS)})', rows)
        db.executemany('INSERT INTO events VALUES (?,?,?,?,?,?)',
                       [(run_id, n, t, f, ty, tx) for n, (t, f, ty, tx) in enumerate(ev_rows)])
    return run_id, len(rows)


def find_run(db, key):
    """A run by id, id prefix or scenario name (the latest recording of that name)."""
    r = db.execute('SELECT * FROM runs WHERE run_id = ?', (key,)).fetchone() \
        or db.execute('SELECT * FROM runs WHERE scenario = ? ORDER BY recorded_at DESC', (key,)).fetchone() \
        or db.execute("SELECT * FROM runs WHERE run_id LIKE ? ORDER BY recorded_at DESC", (key + '%',)).fetchone()
    if r is None:
        raise KeyError(f'no run {key!r}')
    return r


def query_flights(db, operator=None, status=None, run=None):
    sql, args = 'SELECT f.*, r.scenario FROM flights f JOIN runs r USING (run_id) WHERE 1', []
    if operator:
        sql += ' AND f.operator = ?'
        args.append(operator)
    if status:
        sql += ' AND lower(f.status) = lower(?)'
        args.append(status)
    if run:
        sql += ' AND f.run_id = ?'
        args.append(find_run(db, run)['run_id'])
    return db.execute(sql + ' ORDER BY r.recorded_at, f.run_id, COALESCE(f.sched_dep, f.t_request, 1e9), f.flight_id', args).fetchall()


def flight_history(db, run, flight):
    """(flight row, its events, the run's events with no flight) for a flight id or callsign."""
    r = find_run(db, run)
    row = db.execute('SELECT * FROM flights WHERE run_id = ? AND (flight_id = ? OR lower(callsign) = lower(?))',
                     (r['run_id'], flight, flight)).fetchone()
    if row is None:
        raise KeyError(f'no flight {flight!r} in {r["run_id"]}')
    own = db.execute('SELECT * FROM events WHERE run_id = ? AND flight_id = ? ORDER BY t, seq',
                     (r['run_id'], row['flight_id'])).fetchall()
    return row, own


def clock_points(db, run_id, step=10.0):
    """A live run's clock: [[sim t, wall epoch seconds]], about one knot per `step` sim seconds (and the last one). The
    server's pace varies with its load, so the pages interpolate these instead of assuming one speed. Empty for an import."""
    rows = [[r['t'], r['wall']] for r in db.execute('SELECT t, wall FROM clock WHERE run_id = ? ORDER BY t', (run_id,))]
    out = []
    for p in rows:
        if not out or p[0] - out[-1][0] >= step:
            out.append(p)
    if rows and out[-1] is not rows[-1]:
        out.append(rows[-1])
    return out


def flightlog_dict(db, active_run=None, now_t=None):
    """What dashboard/flight_log.html reads (data/flight_log.json, or GET /api/flightlog of the live server): the runs,
    the flights with their histories, and each run's events that belong to no flight (the no-fly zone closing).
    A run has `speed` (sim seconds per wall second: clock time = start_epoch + t / speed) and `speeds` (E1's flight
    profile, for the tracking page's ETA). active_run / now_t: the live server's current run and its sim time."""
    runs = []
    for r in db.execute('SELECT * FROM runs ORDER BY recorded_at'):
        start = datetime.datetime.fromisoformat(r['started_at'] or r['recorded_at']).timestamp()
        runs.append({'run_id': r['run_id'], 'scenario': r['scenario'], 'map': json.loads(r['map'] or 'null'),
                     'seed': r['seed'], 'source': os.path.basename(r['source_file']), 'recorded_at': r['recorded_at'],
                     'started_at': r['started_at'], 'start_epoch': start, 'speed': r['speed'] or 1.0,
                     'speeds': json.loads(r['speeds']) if r['speeds'] else None,
                     'clock': clock_points(db, r['run_id']),
                     'events': [{'t': e['t'], 'type': e['type'], 'text': e['text']} for e in db.execute(
                         'SELECT * FROM events WHERE run_id = ? AND flight_id IS NULL ORDER BY t, seq', (r['run_id'],))]})
    hist = {}
    for e in db.execute('SELECT * FROM events WHERE flight_id IS NOT NULL ORDER BY run_id, t, seq'):
        hist.setdefault((e['run_id'], e['flight_id']), []).append({'t': e['t'], 'type': e['type'], 'text': e['text']})
    flights = []
    for f in db.execute('SELECT * FROM flights ORDER BY run_id, COALESCE(sched_dep, t_request, 1e9), flight_id'):
        d = {k: f[k] for k in f.keys()}
        d['events'] = hist.get((f['run_id'], f['flight_id']), [])
        flights.append(d)
    out = {'schema': EXPORT_SCHEMA, 'generated': datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),
           'runs': runs, 'flights': flights}
    if active_run is not None:
        out.update(live=True, active_run=active_run, now_t=now_t)
    return out


def export_json(db, path):
    """Writes flightlog_dict to path (dashboard/data/flight_log.json)."""
    out = flightlog_dict(db)
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(out, fh, separators=(',', ':'))
    return len(out['runs']), len(out['flights'])


def track_dict(db, run, flight):
    """GET /api/track: one flight's row, its history and its 1 Hz position track (t, x, y, z). run: a run id (a
    prefix or a scenario name also work)."""
    r = find_run(db, run)
    row, events = flight_history(db, r['run_id'], flight)
    pts = [[p['t'], p['x'], p['y'], p['z']] for p in db.execute(
        'SELECT t, x, y, z FROM track WHERE run_id = ? AND flight_id = ? ORDER BY t', (r['run_id'], row['flight_id']))]
    return {'run': {'run_id': r['run_id'], 'scenario': r['scenario'], 'seed': r['seed'], 'speed': r['speed'] or 1.0,
                    'speeds': json.loads(r['speeds']) if r['speeds'] else None, 'clock': clock_points(db, r['run_id']),
                    'start_epoch': datetime.datetime.fromisoformat(r['started_at'] or r['recorded_at']).timestamp()},
            'flight': {k: row[k] for k in row.keys()},
            'events': [{'t': e['t'], 'type': e['type'], 'text': e['text']} for e in events], 'track': pts}


# ---------------------------------------------------------------------------------------------------- live logging
LIVE_STATUS = {'waiting': 'Booked', 'flying': 'In flight', 'landed': 'Landed', 'crashed': 'Crashed', 'cancelled': 'Cancelled'}
STALE_STATUS = {'Booked': 'Not departed', 'In flight': 'In flight at end'}      # what a flight that never finished becomes


def close_stale(db):
    """Flights of live runs still Booked / In flight (their server stopped, or died): closed as Not departed / In
    flight at end. Returns how many."""
    n = 0
    with db:
        for old, new in STALE_STATUS.items():
            n += db.execute("UPDATE flights SET status = ? WHERE status = ? AND run_id IN "
                            "(SELECT run_id FROM runs WHERE source_file = 'live')", (new, old)).rowcount
    return n


def _open(fn):
    """A LiveRecorder call after close() does nothing (the simulation thread may still be finishing its step)."""
    @functools.wraps(fn)
    def wrapper(self, *a, **k):
        return None if self.closed else fn(self, *a, **k)
    return wrapper


class LiveRecorder:
    """What scripts/utm_server.py writes as things happen. One run per server session (run_id
    live-<scenario>-<start time>, source_file 'live'); the flight row is written at booking and updated as it goes
    (Booked -> In flight -> Landed / Crashed / Cancelled / Returned), its events go to `events` (plus a take-off), a
    1 Hz position track to `track`, and a refused booking is a Rejected row with id r<n> (the server reuses f<n> for
    those). Own connection (WAL, 5 s busy timeout), used from the simulation thread; readers open theirs."""

    def __init__(self, path, sc, speed=1.0, speeds=None, wall=None):
        self.path = path
        self.db = connect(path, check_same_thread=False)
        self.lock = threading.RLock()
        close_stale(self.db)                              # an earlier server that did not close its flights
        wall = time.time() if wall is None else wall
        base = f"live-{sc.get('name', 'scenario')}-{time.strftime('%Y%m%d-%H%M%S', time.localtime(wall))}"
        self.run_id, k = base, 1
        while self.db.execute('SELECT 1 FROM runs WHERE run_id = ?', (self.run_id,)).fetchone():
            k += 1
            self.run_id = f'{base}-{k}'
        with self.db:
            self.db.execute('INSERT INTO runs (run_id, scenario, map, seed, source_file, recorded_at, summary, started_at, speed, speeds) '
                            'VALUES (?,?,?,?,?,?,?,?,?,?)',
                            (self.run_id, sc.get('name'), json.dumps(sc.get('map')), sc.get('seed'), 'live', _iso(wall),
                             json.dumps({'live': True}), _iso(wall), float(speed), json.dumps(speeds) if speeds else None))
        self.seq, self.n_refused, self.closed = 0, 0, False
        self.wall0, self.t_last, self.pace_at = wall, 0.0, wall                        # to measure the pace the simulation really runs at
        self.rows = {}                                    # flight id -> {act_dep, last, dist, returned}

    # -- rows
    def _insert(self, **kw):
        row = _row(run_id=self.run_id, **kw)
        cols = ','.join(FLIGHT_COLS)
        self.db.execute(f'INSERT OR REPLACE INTO flights ({cols}) VALUES ({",".join(":" + c for c in FLIGHT_COLS)})', row)

    def _update(self, fid, **kw):
        sets = ', '.join(f'{k} = :{k}' for k in kw)
        self.db.execute(f'UPDATE flights SET {sets} WHERE run_id = :run_id AND flight_id = :fid',
                        {**kw, 'run_id': self.run_id, 'fid': fid})

    def _event(self, t, fid, kind, text):
        self.db.execute('INSERT INTO events VALUES (?,?,?,?,?,?)', (self.run_id, self.seq, round(float(t), 2), fid, kind, text))
        self.seq += 1

    # -- what the server reports
    @_open
    def booked(self, f, t_request, layer=None):
        """f: the server's flight dict (id, kind, operator, from, to, route, t_dep, arrival, layer_z, slot)."""
        a, b = f['route'][0], f['route'][-1]
        with self.lock, self.db:
            self._insert(flight_id=f['id'], callsign=callsign(f['operator'], f['kind'], f['id']), drone=f"D{f['slot'] + 1:02d}",
                         operator=f['operator'], kind=f['kind'], origin=f['from'], dest=f['to'],
                         origin_x=round(a[0], 1), origin_y=round(a[1], 1), dest_x=round(b[0], 1), dest_y=round(b[1], 1),
                         layer=layer, cruise_z=f['layer_z'], t_request=round(float(t_request), 2), sched_dep=f['t_dep'],
                         sched_arr=f['arrival'], status='Booked', n_replans=0, n_alerts=0, distance_m=0.0)
            self.rows[f['id']] = {'act_dep': None, 'last': None, 'dist': 0.0, 'returned': False}

    @_open
    def refused(self, t, operator, frm, to, reason):
        """A refused booking: a Rejected row (id r<n>) and its event. Returns the id."""
        with self.lock, self.db:
            self.n_refused += 1
            fid = f'r{self.n_refused}'
            self._insert(flight_id=fid, callsign=callsign(operator, 'flight', fid).replace('-', '-R', 1), operator=operator or '', kind='flight',
                         origin=frm, dest=to, t_request=round(float(t), 2), status='Rejected', n_replans=0, n_alerts=0,
                         distance_m=0.0)
            self._event(t, fid, 'refused', f'{fid}: booking refused' + (f' ({reason})' if reason else ''))
        return fid

    @_open
    def event(self, e):
        """One of the server's events ({t, type, text, flight?, ...}). A refusal by E1 becomes a Rejected row."""
        if e['type'] == 'rejected':
            return self.refused(e['t'], e.get('node'), e.get('frm'), e.get('to'), e.get('reason'))
        fid = e.get('flight')
        with self.lock, self.db:
            if e.get('tag') == 'back to its pad' and fid in self.rows:
                self.rows[fid]['returned'] = True
            self._event(e['t'], fid, e['type'], e['text'])
            if e['type'] == 'arrived' and fid in self.rows:
                self._update(fid, act_arr=round(e['t'], 2))

    @_open
    def status(self, fid, s, t, f):
        """The simulator's status of flight fid changed to s (waiting / flying / landed / crashed / cancelled)."""
        st = self.rows.get(fid)
        if st is None:
            return
        with self.lock, self.db:
            name = 'Returned' if s == 'landed' and st['returned'] else LIVE_STATUS[s]
            kw = {'status': name}
            if s == 'flying' and st['act_dep'] is None:
                st['act_dep'] = round(float(t), 2)
                kw.update(act_dep=st['act_dep'], dep_delay=round(st['act_dep'] - f['t_dep'], 2))
                self._event(t, fid, 'takeoff', f'{fid} took off')
            if s == 'landed' and name == 'Landed':
                row = self.db.execute('SELECT act_arr FROM flights WHERE run_id = ? AND flight_id = ?',
                                      (self.run_id, fid)).fetchone()
                arr = row['act_arr'] if row and row['act_arr'] is not None else round(float(t), 2)
                kw.update(act_arr=arr, arr_delay=round(arr - f['arrival'], 2))
                if st['act_dep'] is not None:
                    kw['duration_s'] = round(arr - st['act_dep'], 2)
            self._update(fid, **kw)

    @_open
    def replanned(self, f):
        """E4 re-planned flight f (the server's dict, already updated): its re-plan count and height."""
        if f['id'] in self.rows:
            with self.lock, self.db:
                self._update(f['id'], n_replans=f['replans'], cruise_z=f['layer_z'])

    @_open
    def track(self, t, positions):
        """positions: [(flight id, [x, y, z])] of the flights flying now: one track point each and the distance flown. Called
        once per simulated second, it also keeps runs.speed at the pace the simulation really runs at (the server falls behind
        when it is asked for more than the machine can do), so clock time = started_at + t / speed stays close to the wall clock."""
        self.t_last = float(t)
        self._measure_pace()
        with self.lock, self.db:
            self.db.execute('INSERT OR REPLACE INTO clock VALUES (?,?,?)', (self.run_id, round(float(t), 2), round(time.time(), 3)))
            for fid, p in positions:
                st = self.rows.get(fid)
                if st is None:
                    continue
                self.db.execute('INSERT OR REPLACE INTO track VALUES (?,?,?,?,?,?)',
                                (self.run_id, fid, round(float(t), 2), p[0], p[1], p[2]))
                if st['last'] is not None:
                    st['dist'] += float(np.linalg.norm(np.subtract(p, st['last'])))
                    self._update(fid, distance_m=round(st['dist'], 2))
                st['last'] = list(p)

    def clock(self):
        """The run's clock so far (clock_points), for the scene a page loads."""
        with self.lock:
            return clock_points(self.db, self.run_id)

    def pace(self):
        """Simulated seconds per wall second since the run began (None in the first 5 s): the pace stored as runs.speed."""
        now = time.time()
        if now - self.wall0 < 5.0 or self.t_last <= 0:
            return None
        return round(self.t_last / (now - self.wall0), 4)

    def _measure_pace(self, force=False):
        now = time.time()
        if not force and now - self.pace_at < 5.0:
            return
        pace = self.pace()
        if pace is None:
            return
        self.pace_at = now
        with self.lock, self.db:
            self.db.execute('UPDATE runs SET speed = ? WHERE run_id = ?', (pace, self.run_id))

    def close(self):
        """The server stops: flights still Booked / In flight are closed (see close_stale). Returns how many."""
        with self.lock:
            if self.closed:
                return 0
            self._measure_pace(force=True)
            n = close_stale(self.db)
            self.closed = True
            self.db.close()
        return n
