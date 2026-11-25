"""Per-operator access (utm/access.py) through the live server's own request handler: login and logout, 401 / 403, each
operator's view of /api/state, /api/flightlog, /api/track and the bookings, the admin's view, the dashboard's data/ folder
and --no-auth. The simulation is not run: a small stand-in for LiveUTM, a real flight log database."""

import http.client
import json
import os
import sys
import threading
from http.server import ThreadingHTTPServer

import pytest

from utm import access as acc
from utm import flight_db

ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

SC = {'name': 'live-test', 'seed': 7, 'map': {'map': 'osm-crop'}}
SPEEDS = {'v_cruise': 1.5, 'v_climb': 0.5, 'v_descent': 0.5, 'v_descent_high': 1.0, 'slow_band_m': 3.0, 'start_z': 1.0}
CODES = {'admin': 'a-pass', 'op0': 'zero-pass', 'op1': 'one-pass', 'Hospital': 'h-pass'}


def flight(i, operator):
    return {'id': f'f{i}', 'kind': 'flight', 'operator': operator, 'from': 'Depot', 'to': 'Shop', 'status': 'waiting', 't_dep': 10.0,
            'arrival': 40.0, 'layer_z': 4.0, 'slot': i, 'replans': 0, 'pos': [-10.0, -13.0, 1.0],
            'route': [[-10.0, -13.0, 1.0], [-10.0, -13.0, 4.0], [8.0, -13.0, 4.0], [8.0, -13.0, 1.0]]}


class FakeLive:
    """What the handler uses of LiveUTM."""

    def __init__(self, db_path):
        self.rec = flight_db.LiveRecorder(db_path, SC, speed=1.0, speeds=SPEEDS)
        self.db_path, self.hubs, self.snap = db_path, {'op0': 'Depot', 'op1': 'Depot 2'}, {'t': 5.0}
        self.flights = [flight(0, 'op0'), flight(1, 'op1'), flight(2, 'op0')]
        self.events, self.booked = [], []
        for f in self.flights:
            self.rec.booked(f, 1.0, layer=1)
            self.events.append({'t': 1.0, 'type': 'booked', 'text': f"{f['operator']}: {f['id']} approved", 'flight': f['id'],
                                'node': f['operator']})
            self.rec.event(self.events[-1])
        self.events.append({'t': 2.0, 'type': 'log', 'text': 'f1: something', 'flight': 'f1'})   # no node: found by its flight

    def scene(self):
        return {'name': 'live-test', 'operators': ['op0', 'op1', 'Hospital'], 'places': []}

    def state(self, since=0):
        return {'t': 5.0, 'free': 3, 'flights': [dict(f) for f in self.flights], 'events': self.events[since:], 'n_events': len(self.events)}

    def request(self, operator, to, frm=None, pad=None):
        self.booked.append(('request', operator, to))
        return {'flight': 'f9', 'status': 'approved', 'from': 'Depot', 'to': to}

    def emergency(self, operator, frm, to, layer=None):
        self.booked.append(('emergency', operator, to))
        return {'flight': 'emergency_0', 'status': 'approved', 'from': frm, 'to': to}

    def note_reply(self, reply, body):
        pass


@pytest.fixture
def srv(tmp_path):
    import utm_server
    root = tmp_path / 'dash'
    (root / 'data').mkdir(parents=True)
    (root / 'index.html').write_text('<html>page</html>')
    (root / 'data' / 'flight_log.json').write_text('{"flights": ["every operator"]}')
    (root / 'data' / 'ecity_demo.json').write_text('{}')
    live = FakeLive(str(tmp_path / 'live.db'))
    access = acc.Access({r: acc.hash_passcode(p, iterations=1000) for r, p in CODES.items()})
    httpd = ThreadingHTTPServer(('127.0.0.1', 0), utm_server.handler(live, str(root), access))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    live.port, live.access, live.httpd = httpd.server_address[1], access, httpd
    yield live
    httpd.shutdown()
    httpd.server_close()
    live.rec.close()


class Client:
    def __init__(self, live):
        self.port, self.cookie = live.port, None

    def call(self, method, path, body=None, headers=None):
        c = http.client.HTTPConnection('127.0.0.1', self.port, timeout=10)
        h = dict(headers or {})
        if self.cookie:
            h['Cookie'] = self.cookie
        c.request(method, path, json.dumps(body) if body is not None else None, h)
        r = c.getresponse()
        raw = r.read()
        sc = r.getheader('Set-Cookie')
        if sc:
            self.cookie = sc.split(';')[0] if not sc.split(';')[0].endswith('=') else None
        try:
            return r.status, json.loads(raw)
        except ValueError:
            return r.status, raw.decode(errors='replace')

    def get(self, path):
        return self.call('GET', path)

    def post(self, path, body):
        return self.call('POST', path, body)

    def login(self, role):
        st, _ = self.post('/api/login', {'role': role, 'passcode': CODES[role]})
        assert st == 200
        return self


def as_(live, role):
    return Client(live).login(role)


# ------------------------------------------------------------------ login and logout
def test_login_sets_an_httponly_session_and_logout_ends_it(srv):
    c = http.client.HTTPConnection('127.0.0.1', srv.port)
    c.request('POST', '/api/login', json.dumps({'role': 'op0', 'passcode': 'zero-pass'}))
    r = c.getresponse()
    assert r.status == 200 and json.loads(r.read()) == {'role': 'op0', 'admin': False, 'no_auth': False}
    cookie = r.getheader('Set-Cookie')
    assert 'HttpOnly' in cookie and 'SameSite=Strict' in cookie and cookie.startswith('utm_session=')
    me = Client(srv)
    me.cookie = cookie.split(';')[0]
    assert me.get('/api/me')[1]['role'] == 'op0'
    assert me.post('/api/logout', {})[0] == 200
    me.cookie = cookie.split(';')[0]                           # the old token is dead
    assert me.get('/api/me')[0] == 401 and me.get('/api/state')[0] == 401


@pytest.mark.parametrize('role,code', [('op0', 'wrong'), ('op0', ''), ('op0', 'one-pass'), ('nobody', 'zero-pass'), ('', '')])
def test_a_wrong_passcode_or_role_is_refused(srv, role, code):
    c = Client(srv)
    st, body = c.post('/api/login', {'role': role, 'passcode': code})
    assert st == 401 and c.cookie is None and c.get('/api/state')[0] == 401


def test_sessions_expire(srv):
    t = [1000.0]
    srv.access.clock = lambda: t[0]
    c = as_(srv, 'op0')
    assert c.get('/api/state')[0] == 200
    t[0] += acc.SESSION_S + 1
    assert c.get('/api/state')[0] == 401


def test_the_scene_is_public_and_lists_the_roles(srv):
    st, scene = Client(srv).get('/api/scene')
    assert st == 200 and scene['roles'] == ['admin', 'Hospital', 'op0', 'op1']


# ------------------------------------------------------------------ 401 without a session
@pytest.mark.parametrize('method,path', [('GET', '/api/state'), ('GET', '/api/flightlog'), ('GET', '/api/track?flight=f0'),
                                         ('POST', '/api/request'), ('POST', '/api/emergency'), ('GET', '/data/flight_log.json')])
def test_no_session_is_401(srv, method, path):
    assert Client(srv).call(method, path, {} if method == 'POST' else None)[0] == 401
    assert srv.booked == []


def test_the_pages_are_served_without_a_session(srv):
    assert Client(srv).get('/index.html')[0] == 200


# ------------------------------------------------------------------ an operator sees only its own
def test_state_is_filtered_per_operator(srv):
    st, s = as_(srv, 'op0').get('/api/state')
    assert st == 200 and [f['id'] for f in s['flights']] == ['f0', 'f2']
    assert all(e.get('node') == 'op0' or e.get('flight') in ('f0', 'f2') for e in s['events']) and len(s['events']) == 2
    s1 = as_(srv, 'op1').get('/api/state')[1]
    assert [f['id'] for f in s1['flights']] == ['f1'] and len(s1['events']) == 2     # its booking, and the log line of its flight
    assert s1['n_events'] == s['n_events'] == 4                                      # the cursor is the server's


def test_the_flight_log_is_filtered_per_operator(srv):
    st, log = as_(srv, 'op0').get('/api/flightlog')
    assert st == 200 and {f['flight_id'] for f in log['flights']} == {'f0', 'f2'} and {f['operator'] for f in log['flights']} == {'op0'}
    assert all(f['events'] and all('f1' not in e['text'] for e in f['events']) for f in log['flights'])
    log1 = as_(srv, 'op1').get('/api/flightlog')[1]
    assert {f['flight_id'] for f in log1['flights']} == {'f1'}
    assert 'op0' not in json.dumps(log1['flights'])


def test_an_operator_cannot_open_anothers_track(srv):
    a, b = as_(srv, 'op0'), as_(srv, 'op1')
    assert a.get('/api/track?flight=f0')[0] == 200
    st, body = b.get('/api/track?flight=f0')
    assert st == 403 and 'track' not in body
    run = srv.rec.run_id
    assert b.get(f'/api/track?run={run}&flight=f0')[0] == 403 and b.get(f'/api/track?run={run}&flight=f2')[0] == 403
    assert b.get('/api/track?flight=nope')[0] == 403                   # unknown looks the same: no probing
    assert b.get('/api/track?flight=f1')[0] == 200


def test_an_operator_books_only_as_itself(srv):
    a = as_(srv, 'op0')
    assert a.post('/api/request', {'operator': 'op1', 'to': 'Shop'})[0] == 403 and srv.booked == []
    assert a.post('/api/request', {'operator': 'op0', 'to': 'Shop'})[0] == 200
    assert a.post('/api/request', {'to': 'Shop'})[0] == 200
    assert srv.booked == [('request', 'op0', 'Shop')] * 2
    assert a.post('/api/emergency', {'from': 'Clinic', 'to': 'Shop'})[0] == 403          # a delivery operator files no emergency
    h = as_(srv, 'Hospital')
    assert h.post('/api/emergency', {'operator': 'op0', 'from': 'Clinic', 'to': 'Shop'})[0] == 403
    assert h.post('/api/emergency', {'from': 'Clinic', 'to': 'Shop'})[0] == 200 and srv.booked[-1] == ('emergency', 'Hospital', 'Shop')


# ------------------------------------------------------------------ the admin sees everything
def test_admin_sees_everything_and_books_for_any_operator(srv):
    c = as_(srv, 'admin')
    assert c.get('/api/me')[1] == {'role': 'admin', 'admin': True, 'no_auth': False}
    assert [f['id'] for f in c.get('/api/state')[1]['flights']] == ['f0', 'f1', 'f2']
    assert len(c.get('/api/state')[1]['events']) == 4
    assert {f['operator'] for f in c.get('/api/flightlog')[1]['flights']} == {'op0', 'op1'}
    assert all(c.get(f'/api/track?flight=f{i}')[0] == 200 for i in range(3))
    assert c.post('/api/request', {'operator': 'op1', 'to': 'Shop'})[0] == 200 and srv.booked == [('request', 'op1', 'Shop')]
    assert c.get('/api/track?flight=nope')[0] == 404


# ------------------------------------------------------------------ the data/ folder
@pytest.mark.parametrize('path', ['/data/flight_log.json', '/data/ecity_demo.json', '/DATA/flight_log.json', '/data%2Fflight_log.json',
                                  '/data%5Cflight_log.json', '//data/flight_log.json', '/./data/flight_log.json', '/x/../data/flight_log.json',
                                  '/data/', '/data./flight_log.json', '/data/flight_log.json::$DATA', '/%2e%2e/dash/data/flight_log.json'])
def test_an_operator_cannot_read_the_data_folder(srv, path):
    st, body = as_(srv, 'op0').get(path)
    assert st in (403, 404) and 'every operator' not in str(body)
    assert Client(srv).get(path)[0] in (401, 404)


def test_op0_cannot_fetch_flight_log_json(srv):
    st, body = as_(srv, 'op0').get('/data/flight_log.json')
    assert st == 403 and 'every operator' not in str(body)
    assert Client(srv).get('/data/flight_log.json')[0] == 401
    st, body = as_(srv, 'admin').get('/data/flight_log.json')
    assert st == 200 and body == {'flights': ['every operator']}


# ------------------------------------------------------------------ cross-site and --no-auth
def test_a_post_from_another_origin_is_refused(srv):
    c = as_(srv, 'op0')
    st, _ = c.call('POST', '/api/request', {'to': 'Shop'}, {'Origin': 'http://evil.example'})
    assert st == 403 and srv.booked == []
    assert c.call('POST', '/api/request', {'to': 'Shop'}, {'Origin': f'http://127.0.0.1:{srv.port}', 'Host': f'127.0.0.1:{srv.port}'})[0] == 200


def test_no_auth_makes_everyone_the_admin(tmp_path):
    import utm_server
    live = FakeLive(str(tmp_path / 'n.db'))
    httpd = ThreadingHTTPServer(('127.0.0.1', 0), utm_server.handler(live, str(tmp_path), acc.Access(no_auth=True)))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        live.port = httpd.server_address[1]
        c = Client(live)
        assert c.get('/api/me')[1] == {'role': 'admin', 'admin': True, 'no_auth': True}
        assert len(c.get('/api/state')[1]['flights']) == 3 and c.get('/api/scene')[1]['no_auth'] is True
    finally:
        httpd.shutdown()
        httpd.server_close()
        live.rec.close()


# ------------------------------------------------------------------ the module
def test_hashes_are_salted_and_checked():
    h1, h2 = acc.hash_passcode('x'), acc.hash_passcode('x')
    assert h1 != h2 and acc.check_passcode('x', h1) and not acc.check_passcode('y', h1) and not acc.check_passcode('x', 'junk')


def test_the_shipped_demo_file_loads_with_its_documented_passcodes():
    a = acc.Access.load(acc.DEFAULT_FILE)
    for role, code in {'admin': 'admin-demo', 'op0': 'op0-demo', 'op1': 'op1-demo', 'Hospital': 'hospital-demo'}.items():
        assert a.login(role, code) and not a.login(role, 'nope')


def test_an_unknown_role_costs_one_hash_like_a_wrong_passcode(monkeypatch):
    a, real, calls = acc.Access.load(acc.DEFAULT_FILE), acc.hashlib.pbkdf2_hmac, []

    def counted(*args, **kw):
        calls.append(args[3])                                # the iterations
        return real(*args, **kw)
    monkeypatch.setattr(acc.hashlib, 'pbkdf2_hmac', counted)
    assert a.login('op0', 'nope') is None
    known, calls[:] = list(calls), []
    assert a.login('nobody', 'nope') is None
    assert calls == known == [acc.ITERATIONS]


def test_public_static_is_strict():
    ok = ['/', '/index.html', '/operator.html', '/track_logic.js', '/auth.js']
    bad = ['/data/flight_log.json', '/data', '/a/b.html', '/..%2fx.html', '/x.json', '/.hidden.html', '/x.html::$DATA', '/\\data\\x.js']
    assert all(acc.public_static(p) for p in ok) and not any(acc.public_static(p) for p in bad)


def test_set_passcode_rewrites_the_file_and_keeps_the_header(tmp_path):
    p = tmp_path / 'access.yaml'
    p.write_text('# DEMO\nadmin: admin\nroles:\n  admin: x\n')
    acc.set_passcode(str(p), 'op7', 'seven')
    a = acc.Access.load(str(p))
    assert p.read_text().startswith('# DEMO') and a.login('op7', 'seven') and 'op7' in a.role_names(['op7'])
