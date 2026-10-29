"""E5 phase 3: the live UTM (scripts/utm_server.py) on a small scene, without the HTTP layer."""

import os
import sys

import pytest

ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

from utm import scenario  # noqa: E402

SCENE = {'name': 'live-test', 'map': {'map': 'osm-crop'},
         'places': {'Depot': {'kind': 'hub', 'operator': 'op0', 'pads': [[-10.0, -13.0], [-7.0, -13.0]]},
                    'Shop': {'kind': 'drop', 'at': [8.0, -13.0]}, 'Clinic': {'kind': 'hospital', 'at': [-13.0, 15.0]}},
         'requests': [{'t': 0, 'operator': 'op0', 'to': 'Shop'}],
         'sim': {'checkpoint': '../utm-swarm-marl/checkpoints/stage2_r1_crowd_rl_r7v3ft_s0_lab/checkpoint_final.pth'}}


@pytest.fixture(scope='module')
def live():
    import run_scenario
    sc = scenario.load(SCENE)
    try:
        run_scenario.checkpoint_path(sc['sim']['checkpoint'])
    except FileNotFoundError:
        pytest.skip('the shipped checkpoint is not on this machine')
    import utm_server
    return utm_server.LiveUTM(sc, slots=2)


def test_a_booking_flies_lands_and_frees_its_drone(live):
    r = live.request('op0', 'Shop')
    assert r['status'] == 'approved' and r['from'] == 'Depot' and len(live.free) == 1
    f = live.flights[r['flight']]
    assert f['route'][0][:2] in ([-10.0, -13.0], [-7.0, -13.0]) and f['route'][-1][:2] == [8.0, -13.0]
    for _ in range(int(120 / live.dt)):
        live.step()
        if f['status'] == 'landed':
            break
    assert f['status'] == 'landed' and len(live.free) == 2
    assert [e['type'] for e in live.events] == ['booked', 'arrived']
    st = live.state()
    assert st['n_events'] == 2 and live.state(since=1)['events'][0]['type'] == 'arrived'


def test_bookings_are_refused_with_a_reason(live):
    assert live.request('op0', 'Nowhere')['status'] == 'rejected'
    assert live.request('op9', 'Shop')['status'] == 'rejected'          # no hub for op9
    live.free.clear()
    try:
        assert live.request('op0', 'Shop') == {'status': 'rejected', 'reason': 'no drone free'}
    finally:
        live.free.extend(range(live.n))


def test_the_scene_lists_places_and_client_nodes(live):
    s = live.scene()
    assert {p['name'] for p in s['places']} == {'Depot', 'Shop', 'Clinic'}
    assert s['operators'][0] == 'op0' and s['slots'] == 2 and s['buildings']
