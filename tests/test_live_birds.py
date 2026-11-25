"""Birds in the live UTM (scripts/utm_server.py, envs/birds.py BirdPool): POST /api/bird is the admin's; a bird added while the
server runs is in /api/state and seen by the drones' detector; the bird guard takes over for a bird aimed at a flight; the
events go to the flight log in plain words; a hit is a crash; operators see only the birds near their own drones; with
birds off nothing changes."""

import http.client
import json
import os
import sys
import threading
from http.server import ThreadingHTTPServer

import numpy as np
import pytest

ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
sys.path.insert(0, os.path.dirname(__file__))

from envs import bird_detector, birds  # noqa: E402
from utm import access as acc  # noqa: E402
from utm import flight_db, scenario  # noqa: E402

import test_access as ta  # noqa: E402

BOUNDS = [[-30.0, 30.0], [-30.0, 30.0]]
DT = 0.032


# ------------------------------------------------------------------ the pool, the detector, the access filter (no simulator)
def pool(n=3):
    return birds.BirdPool(n, DT, 0.12, BOUNDS, margin_m=5.0, max_age_s=60.0)


def straight(p0, v):
    return birds.Bird.straight(p0, v, 0.12, DT)


def test_a_pool_starts_parked_and_launches_into_a_free_slot():
    p = pool(2)
    assert not p.active().any() and np.all(p.positions()[:, 0] > 1e3)               # parked far away
    b = straight([0, 0, 5], [1, 0, 0])
    assert p.launch(b) == 0 and p.launch(straight([0, 5, 5], [0, 0, 0])) == 1 and p.launch(b) is None   # full
    for _ in range(int(2 / DT)):
        p.step()
    assert p.positions()[0, 0] == pytest.approx(2.0, abs=0.1)                       # flies from where it was launched
    assert p.positions()[1, 1] == pytest.approx(5.0)


def test_a_bird_that_left_the_area_is_parked_again_and_its_slot_is_free():
    p = pool(1)
    p.launch(straight([28, 0, 5], [3, 0, 0]))
    for _ in range(int(2 / DT)):                                                    # inside the bounds + 5 m margin: stays
        p.step()
    assert p.active()[0]
    for _ in range(int(3 / DT)):                                                    # x = 28 + 3 * 5 s = 43 > 35: gone
        p.step()
    assert not p.active()[0] and p.free_slots() == [0]
    assert p.launch(straight([0, 0, 5], [0, 0, 0])) == 0


def test_a_bird_is_parked_after_its_maximum_age():
    p = birds.BirdPool(1, DT, 0.12, BOUNDS, max_age_s=2.0)
    p.launch(straight([0, 0, 5], [0, 0, 0]))
    for _ in range(int(2.2 / DT)):
        p.step()
    assert not p.active()[0]


def test_a_bird_aimed_with_fold_off_flies_through_and_away():
    cfg = {'speed': [3.0, 3.0], 'radius': 0.12, 'vz_max': 0.0, 'circle_radius': [2, 4], 'max_turn_rate': 0.35}
    bird = birds.Bird.planned(np.random.default_rng(0), cfg, [0.0, 0.0, 5.0], 4.0, DT, 'cross', (1.0, 9.0),
                              ((-10, 10), (-10, 10)), fold=False, heading=0.0)
    k = int(round(4.0 / DT))
    assert np.allclose(bird.position(k), [0, 0, 5], atol=0.2)                       # at the aim point at the aim time
    assert bird.position(k + int(10 / DT))[0] == pytest.approx(30.0, abs=0.5)       # carried on, not reflected at the bounds


def test_the_detector_forgets_a_slot_and_idles():
    cfg = bird_detector.validate({'enabled': True, 'mode': 'realistic', 'range_m': 12, 'fov_deg': 360, 'tracker': 'kalman',
                                  'birds_in_slots': False})
    det = bird_detector.BirdDetector(cfg, 2, 3, DT, np.random.default_rng(0))
    drones = np.array([[0.0, 0, 5], [20.0, 0, 5]])
    pos = np.array([[3.0, 0, 5], [1e4, 1e4, 50], [1e4, 1e4, 50]])
    for k in range(40):
        known, _, _ = det.step(k, drones, np.zeros((2, 3)), pos, np.zeros((3, 3)))
    assert known[0, 0] and not known[1, 0] and not known[:, 1:].any()              # tracked by the near drone only
    det.forget(0)
    assert det.hits[:, 0].sum() == 0
    known, _, _ = det.idle()
    assert not known.any()


def test_an_operator_sees_only_the_birds_near_its_own_drones():
    state = {'t': 1.0, 'free': 1, 'events': [], 'n_events': 0,
             'flights': [{'id': 'f0', 'operator': 'op0', 'status': 'flying', 'pos': [0, 0, 5]},
                         {'id': 'f1', 'operator': 'op1', 'status': 'flying', 'pos': [50, 0, 5]}],
             'birds': [{'id': 'b1', 'pos': [3, 0, 5]}, {'id': 'b2', 'pos': [52, 0, 5]}, {'id': 'b3', 'pos': [100, 0, 5]}]}
    assert [b['id'] for b in acc.filter_state(state, 'op0')['birds']] == ['b1']
    assert [b['id'] for b in acc.filter_state(state, 'op1')['birds']] == ['b2']
    assert len(acc.filter_state(state, 'admin', admin=True)['birds']) == 3
    assert 'birds' not in acc.filter_state({k: v for k, v in state.items() if k != 'birds'}, 'op0')   # birds off: no key


# ------------------------------------------------------------------ POST /api/bird through the request handler
class BirdLive(ta.FakeLive):
    def __init__(self, db_path):
        super().__init__(db_path)
        self.added = []

    def add_bird(self, flight=None, place=None, at=None, speed=2.0, count=1, who='admin', path='cross'):
        if flight == 'nope':
            raise ValueError('nope is not on its pad or flying')
        if count == 99:
            return 409, {'status': 'error', 'reason': 'too many birds flying'}
        self.added.append((flight, place, at, speed, count))
        return 200, {'status': 'ok', 'added': count, 'text': f'bird added near {flight or place} ({who})'}


@pytest.fixture
def srv(tmp_path):
    import utm_server
    root = tmp_path / 'dash'
    root.mkdir()
    live = BirdLive(str(tmp_path / 'live.db'))
    access = acc.Access({r: acc.hash_passcode(p, iterations=1000) for r, p in ta.CODES.items()})
    httpd = ThreadingHTTPServer(('127.0.0.1', 0), utm_server.handler(live, str(root), access))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    live.port, live.access = httpd.server_address[1], access
    yield live
    httpd.shutdown()
    httpd.server_close()
    live.rec.close()


def call(srv, role, body):
    c = ta.Client(srv)
    if role:
        c.login(role)
    return c.post('/api/bird', body)


def test_api_bird_is_for_the_admin_only(srv):
    assert call(srv, None, {'flight': 'f0'})[0] == 401                              # no session
    st, reply = call(srv, 'op0', {'flight': 'f0'})
    assert st == 403 and srv.added == []                                            # an operator, even for its own flight
    st, reply = call(srv, 'Hospital', {'flight': 'f0'})
    assert st == 403
    st, reply = call(srv, 'admin', {'flight': 'f0', 'speed': 2.5, 'count': 2})
    assert st == 200 and srv.added == [('f0', None, None, 2.5, 2)] and 'bird added near f0 (admin)' in reply['text']


def test_api_bird_errors(srv):
    assert call(srv, 'admin', {'flight': 'nope'})[0] == 400                          # not flying: the server's reason
    assert call(srv, 'admin', {'flight': 'f0', 'count': 99})[0] == 409               # the pool is full


# ------------------------------------------------------------------ the live simulation
CKPTS = ['../utm-swarm-marl/checkpoints/stage2_r1_crowd_rl_r7v3ft_s0/checkpoint_final.pth',
         '../utm-swarm-marl/checkpoints/stage2_r1_crowd_rl_r7v3ft_s0_lab/checkpoint_final.pth',
         'checkpoints/stage2_r1_crowd_rl_r7v3ft_s0/checkpoint_final.pth']


def scene():
    found = next((c for c in CKPTS if os.path.isfile(os.path.join(ROOT, c))), None)
    if found is None:
        pytest.skip('the shipped checkpoint is not on this machine')
    return scenario.load({**ta_scene(), 'sim': {'checkpoint': os.path.abspath(os.path.join(ROOT, found))}})


def ta_scene():
    return {'name': 'live-birds-test', 'map': {'map': 'osm-crop'},
            'places': {'Depot': {'kind': 'hub', 'operator': 'op0', 'pads': [[-10.0, -13.0], [-7.0, -13.0]]},
                       'Shop': {'kind': 'drop', 'at': [8.0, -13.0]}, 'Clinic': {'kind': 'hospital', 'at': [-13.0, 15.0]}},
            'requests': [{'t': 0, 'operator': 'op0', 'to': 'Shop'}]}


def make(birds_on, db=None):
    import utm_server
    return utm_server.LiveUTM(scene(), slots=2, db=db, birds=birds_on)


@pytest.fixture(scope='module')
def live():
    return make(True)


def fly_until(live, fid, status, max_s=90.0):
    for _ in range(int(max_s / live.dt)):
        live.step()
        if live.flights[fid]['status'] == status:
            return True
    return False


def test_the_final_configuration_is_what_the_live_server_runs(live):
    b = live.env.bird_cfg
    d, g = b['detector'], b['guard']
    assert (d['fov_deg'], d['range_m'], d['tracker'], d['kalman_q'], d['share'], d['birds_in_slots']) == (360.0, 12.0, 'kalman', 2.0, False, False)
    assert g['enabled'] and g['layer_spacing_m'] == 2.0
    assert live.pool is not None and len(live.pool) == 12 and live.env.bird_slots is False


def test_a_bird_added_while_running_is_in_the_state_and_seen_by_the_detector_and_the_guard_takes_over(live):
    r = live.request('op0', 'Shop')
    fid = r['flight']
    assert r['status'] == 'approved' and fly_until(live, fid, 'flying')
    for _ in range(int(3 / live.dt)):
        live.step()
    code, reply = live.add_bird(flight=fid, speed=2.0)
    assert code == 200 and reply['text'] == f'bird added near {fid} (admin)'
    slot, i = next(iter(live.bird_name)), live.slot[fid]
    seen = guarded = False
    for _ in range(int(40 / live.dt)):
        live.step()
        if live.snap.get('birds') and not seen:
            assert live.snap['birds'][0]['id'] == 'b1' and len(live.snap['birds'][0]['pos']) == 3
        known = live.env._bird_tracks[0]
        seen |= bool(known[i, slot])                                                 # the drone's detector has a track of it
        guarded |= bool(live.guard.active[i])
        if live.flights[fid]['status'] != 'flying':
            break
    assert seen and guarded
    assert any(e['type'] == 'bird_added' and e['flight'] == fid for e in live.events)
    assert any(e['text'].startswith(f'{fid} avoided a bird (closest ') for e in live.events) or live.flights[fid]['status'] == 'crashed'
    assert fly_until(live, fid, 'landed') or live.flights[fid]['status'] in ('landed', 'crashed')


def test_the_snapshot_lists_birds_and_the_guard_and_birds_are_retired(live):
    for _ in range(int(60 / live.dt)):
        live.step()
    live.step()
    assert not live.pool.active().any() and live.bird_name == {}                      # retired after leaving the area / 60 s
    assert live.snap.get('birds') == [] or 'birds' in live.snap
    assert all('guard' in f for f in live.snap['flights'])


def test_a_bird_sent_to_a_place_or_a_point_and_bad_requests(live):
    assert live.add_bird(place='Clinic', count=2)[1]['text'] == '2 birds added near Clinic (admin)'
    assert live.add_bird(at=[0.0, 0.0])[1]['text'] == 'bird added at (0, 0) (admin)'
    assert len(live.bird_name) == 3
    with pytest.raises(ValueError):
        live.add_bird(place='Nowhere')
    with pytest.raises(ValueError):
        live.add_bird(flight='f99')
    with pytest.raises(ValueError):
        live.add_bird(place='Clinic', speed=5.0)
    with pytest.raises(ValueError):
        live.add_bird(place='Clinic', count=5)
    with pytest.raises(ValueError):
        live.add_bird()                                                               # no target
    with pytest.raises(ValueError):
        live.add_bird(place='Clinic', path='zigzag')
    assert live.add_bird(place='Clinic', count=4, path='circle')[1]['text'] == '4 circling birds added near Clinic (admin)' and live.add_bird(place='Clinic', count=4)[0] == 200   # 11 flying
    code, reply = live.add_bird(place='Clinic', count=4)
    assert code == 409 and 'at most 12' in reply['reason']                            # the pool is full


def test_a_hit_is_a_crash_in_plain_words_and_the_events_reach_the_flight_log(tmp_path):
    lv = make(True, db=str(tmp_path / 'log.db'))
    fid = lv.request('op0', 'Shop')['flight']
    assert fly_until(lv, fid, 'flying')
    for _ in range(int(3 / lv.dt)):
        lv.step()
    lv.add_bird(flight=fid)
    i = lv.slot[fid]
    p = lv.env.adapter.get_positions()[i]
    lv.env.launch_bird(birds.Bird.straight(p, [0, 0, 0], 0.12, lv.dt))               # a bird right on the drone
    lv.step()
    lv.step()
    assert lv.flights[fid]['status'] == 'crashed'
    assert any(e['type'] == 'crashed' and e['text'] == f'{fid} hit a bird' for e in lv.events)
    lv.close()
    db = flight_db.connect(str(tmp_path / 'log.db'))
    texts = [r['text'] for r in db.execute('SELECT text FROM events WHERE flight_id = ? ORDER BY seq', (fid,))]
    assert f'bird added near {fid} (admin)' in texts and f'{fid} hit a bird' in texts
    assert db.execute('SELECT status FROM flights WHERE flight_id = ?', (fid,)).fetchone()['status'] == 'Crashed'


def test_an_avoidance_is_logged_with_its_closest_distance(tmp_path):
    lv = make(True, db=str(tmp_path / 'log.db'))
    fid = lv.request('op0', 'Shop')['flight']
    assert fly_until(lv, fid, 'flying')
    for _ in range(int(3 / lv.dt)):
        lv.step()
    lv.add_bird(flight=fid)
    for _ in range(int(40 / lv.dt)):
        lv.step()
    lv.close()
    db = flight_db.connect(str(tmp_path / 'log.db'))
    texts = [r['text'] for r in db.execute('SELECT text FROM events WHERE flight_id = ? ORDER BY seq', (fid,))]
    assert any(t.startswith(f'{fid} avoided a bird (closest ') and t.endswith(' m)') for t in texts) or f'{fid} hit a bird' in texts


def test_the_display_flag_outlasts_the_guard_by_the_merge_window():
    import utm_server
    live = make(True)
    fid = live.request('op0', 'Shop')['flight']
    assert fly_until(live, fid, 'flying')
    for _ in range(int(3 / live.dt)):
        live.step()
    live.add_bird(flight=fid)
    i = live.slot[fid]
    last_raw = None
    held = early_off = False
    for _ in range(int(40 / live.dt)):
        live.step()
        if live.guard.active is not None and live.guard.active[i]:
            last_raw = live.t
        snap = next(f for f in live.snap['flights'] if f['id'] == fid)
        if last_raw is not None and not live.guard.active[i]:
            if snap['guard']:
                held = True
            elif live.t - last_raw < utm_server.AVOID_QUIET_S - 0.3:       # (the snapshot is ~0.1 s old)
                early_off = True
    assert last_raw is not None and held and not early_off


def test_birds_off_changes_nothing_and_adds_nothing():
    off, on = make(False), make(True)
    assert off.pool is None and off.guard is None and 'birds' not in off.state()
    assert off.add_bird(flight='f0')[0] == 404 and off.scene()['birds']['on'] is False
    assert on.scene()['birds']['on'] is True
    for lv in (off, on):
        lv.request('op0', 'Shop')
    for _ in range(int(12 / off.dt)):
        off.step()
        on.step()
    assert on.flights['f0']['status'] == off.flights['f0']['status'] == 'flying'
    assert np.allclose(on.env.adapter.get_positions(), off.env.adapter.get_positions())     # birds on, none flying: the same flight
    assert off.snap['flights'][0].keys() >= {'id'} and 'guard' not in off.snap['flights'][0]
