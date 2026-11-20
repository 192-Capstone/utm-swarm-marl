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
import hashlib
import json
import os
import re
import sqlite3
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'scripts'))

from render_scenario import STATE_WORDS, describe  # noqa: E402
from utm import flight_log, strategic_planner  # noqa: E402

DEFAULT_DB = os.path.join('eval_results', 'flight_log.db')
EXPORT_SCHEMA = 'utm-flight-log-db/1'
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
"""
FLIGHT_SCHEMA = (('run_id', 'TEXT NOT NULL'), ('flight_id', 'TEXT NOT NULL'), ('callsign', 'TEXT'), ('drone', 'TEXT'),
                 ('operator', 'TEXT'), ('kind', 'TEXT'), ('origin', 'TEXT'), ('dest', 'TEXT'),
                 ('origin_x', 'REAL'), ('origin_y', 'REAL'), ('dest_x', 'REAL'), ('dest_y', 'REAL'),
                 ('layer', 'INTEGER'), ('cruise_z', 'REAL'), ('t_request', 'REAL'),
                 ('sched_dep', 'REAL'), ('act_dep', 'REAL'), ('sched_arr', 'REAL'), ('act_arr', 'REAL'),
                 ('dep_delay', 'REAL'), ('arr_delay', 'REAL'), ('duration_s', 'REAL'), ('status', 'TEXT'),
                 ('n_replans', 'INTEGER'), ('n_alerts', 'INTEGER'), ('distance_m', 'REAL'),
                 ('min_sep_m', 'REAL'), ('min_sep_with', 'TEXT'))
FLIGHT_COLS = tuple(c for c, _ in FLIGHT_SCHEMA)


def connect(path=DEFAULT_DB):
    if path != ':memory:' and os.path.dirname(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    db = sqlite3.connect(path)
    db.row_factory = sqlite3.Row
    db.executescript(SCHEMA_SQL)
    cols = ', '.join(f'{c} {t}' for c, t in FLIGHT_SCHEMA)
    db.execute(f'CREATE TABLE IF NOT EXISTS flights ({cols}, PRIMARY KEY (run_id, flight_id))')
    have = {r['name'] for r in db.execute('PRAGMA table_info(flights)')}
    if 'started_at' not in {r['name'] for r in db.execute('PRAGMA table_info(runs)')}:
        db.execute('ALTER TABLE runs ADD COLUMN started_at TEXT')
    for c, t in FLIGHT_SCHEMA:                            # a database made by an older version: add what is new
        if c not in have:
            db.execute(f'ALTER TABLE flights ADD COLUMN {c} {t.replace(" NOT NULL", "")}')
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
        if kind == 'emergency' and operator in ('', 'emergency'):     # the scenario named no operator: the site it leaves from
            site = place.named(origin[1], origin[2])
            operator = site if site and ' pad ' not in site else operator
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
        db.execute('INSERT INTO runs (run_id, scenario, map, seed, source_file, recorded_at, summary, started_at) '
                   'VALUES (?,?,?,?,?,?,?,?)',
                   (run_id, name, json.dumps(sc.get('map')), sc.get('seed'), os.path.abspath(path), recorded,
                    json.dumps(log.get('summary') or {}), started))
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


def export_json(db, path):
    """dashboard/data/flight_log.json: the runs, the flights with their histories, and each run's events
    that belong to no flight (the no-fly zone closing)."""
    runs = []
    for r in db.execute('SELECT * FROM runs ORDER BY recorded_at'):
        start = datetime.datetime.fromisoformat(r['started_at'] or r['recorded_at']).timestamp()
        runs.append({'run_id': r['run_id'], 'scenario': r['scenario'], 'map': json.loads(r['map'] or 'null'),
                     'seed': r['seed'], 'source': os.path.basename(r['source_file']), 'recorded_at': r['recorded_at'], 'started_at': r['started_at'],
                     'start_epoch': start,
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
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(out, fh, separators=(',', ':'))
    return len(runs), len(flights)
