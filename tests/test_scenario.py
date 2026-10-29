"""E6: scenario files (utm/scenario.py) and the flight log (utm/flight_log.py)."""

import numpy as np
import pytest

from utm import flight_log, scenario
from utm.strategic_planner import FlightPlan, StrategicPlanner

B = ((-20, 20), (-20, 20))


def test_the_demo_scenario_loads_with_defaults_filled_in():
    sc = scenario.load('scenarios/ecity_demo.yaml')
    assert sc['map'] == {'map': 'osm-crop'} and sc['utm']['mode'] == 'e4' and sc['sim']['pilot'] == 'learned'
    assert sc['zones'][0]['polygon'] == [[-6, 5], [0, 5], [0, 11], [-6, 11]] and 'zone' not in sc
    assert sc['emergencies'][0]['t'] == 25


def test_zones_and_emergencies():
    sc = scenario.load({'map': {'map': 'city'}, 'zone': {'at': 5}, 'zones': [{'at': 9, 'polygon': [[0, 0], [1, 0], [1, 1]]}],
                        'emergencies': [{'t': 20, 'start': [0, 0], 'goal': [5, 5]}]})
    assert [z['at'] for z in sc['zones']] == [5, 9]          # zone: shorthand, first
    assert sc['emergencies'][0]['t'] == 20


@pytest.mark.parametrize('bad', [
    {'map': {'map': 'moon'}},
    {'map': {'map': 'city'}, 'seed': 12},
    {'map': {'map': 'city'}, 'utm': {'mode': 'e9'}},
    {'map': {'map': 'city'}, 'zone': {'place': 'busiest'}},
    {'map': {'map': 'city'}, 'intruders': [{'t': 3}]},
    {'map': {'map': 'city'}, 'utm': {'telemetry': [1.0]}},
    {'map': {'map': 'city'}, 'emergencies': [{'t': 3, 'start': [0, 0]}]},
    {'map': {'map': 'city'}, 'emergencies': [{'t': 3, 'start': [0, 0], 'goal': [1, 1], 'priority': 0}]},
    {'map': {'map': 'city'}, 'utm': {'mode': 'no_e1'}, 'emergencies': [{'t': 3, 'start': [0, 0], 'goal': [1, 1]}]},
])
def test_bad_scenarios_are_refused(bad):
    with pytest.raises(ValueError):
        scenario.load(bad)


def test_flight_log_round_trip(tmp_path):
    sky = StrategicPlanner([], B)
    sky.submit(FlightPlan('A', (-15, 0), (15, 0), t_earliest=0.0, operator='op0'))
    d = sky.current['A']
    rec = flight_log.FlightLog({'name': 't'}, frame_dt=0.5)
    rec.start(sky, [], 0.032, [{'id': 'A', 'kind': 'flight', 'operator': 'op0', 'start': (-15, 0), 'goal': (15, 0),
                                't_dep': d.t_dep, 'arrival': d.arrival, 'layer': d.layer, 'route': d.waypoints}])
    rec.frame(0.5, np.array([[-15.0, 0.0, 1.0]]), [1])
    rec.zone_on(5.0, 'nothing')
    rec.replan(6.0, 'A', 'late', d)
    rec.finish({'arrived': 1, 'replans': []})
    path = tmp_path / 'run.json.gz'
    rec.save(path)
    log = flight_log.read(path)
    assert log['schema'] == flight_log.SCHEMA and log['drones'][0]['operator'] == 'op0'
    assert log['bookings']['A'] and all(c in log['cells'] for c, *_ in log['bookings']['A'])
    assert len(log['cells'][log['bookings']['A'][0][0]]) == 6                 # a hexagon's corners
    assert log['frames']['pos'] == [[[-15.0, 0.0, 1.0]]] and log['frames']['status'] == [[1]]
    assert [e['type'] for e in log['events']] == ['zone_on', 'replan'] and log['events'][1]['route']
    assert log['summary'] == {'arrived': 1}


def test_a_small_scenario_flies_end_to_end(tmp_path):
    """The whole pipeline on a tiny scenario: E-City, 4 flights, the simple pilot (no model), a zone,
    an intruder and an emergency; checks the results, E7 and the recording."""
    import sys
    sys.path.insert(0, 'scripts')
    import run_scenario
    sc = scenario.load({
        'name': 'tiny', 'seed': 20, 'map': {'map': 'osm-crop'},
        'demand': {'flights': 4, 'window': 10},
        'intruders': [{'t': 4, 'start': [20, 8], 'goal': [-20, 6]}],
        'zones': [{'at': 6, 'for': 20, 'polygon': [[-6, 5], [0, 5], [0, 11], [-6, 11]]}],
        'emergencies': [{'t': 8, 'start': [2, 20], 'goal': [7, 8]}],
        'sim': {'pilot': 'simple'}})
    path = tmp_path / 'tiny.json.gz'
    out, rec = run_scenario.run(sc, record=str(path))
    assert out['approved'] == out['flights'] >= 3
    assert out['emergencies'][0]['status'] == 'approved'
    assert out['e7']['throughput_per_h'] > 0 and 0 < out['e7']['fairness_jain'] <= 1
    log = flight_log.read(path)
    kinds = [d['kind'] for d in log['drones']]
    assert kinds.count('flight') == out['flights'] and kinds.count('intruder') == 1 and kinds.count('emergency') == 1
    assert log['zones'][0]['t_on'] is not None and log['summary']['e7']
    assert {e['type'] for e in log['events']} >= {'zone_on', 'replan'}
    assert all(s in (flight_log.LANDED, flight_log.CRASHED, flight_log.CANCELLED) for s in log['frames']['status'][-1])


def _tiny(**extra):
    return scenario.load({
        'name': 'tiny', 'seed': 20, 'map': {'map': 'osm-crop'},
        'demand': {'flights': 3, 'window': 10},
        'emergencies': [{'t': 4, 'start': [2, 20], 'goal': [7, 8]}],
        'sim': {'pilot': 'simple'}, **extra})


def test_a_refused_emergency_is_cancelled_and_the_run_does_not_wait_for_it(tmp_path, monkeypatch):
    """#37 review: an emergency E1 refuses never takes off; it ends cancelled and the run stops
    when the others are done, not at max_steps."""
    import sys
    sys.path.insert(0, 'scripts')
    import run_scenario
    from utm.contingency import ContingencyManager
    from utm.strategic_planner import Decision

    def refuse(self, fp, t, positions):
        return [{'t': t, 'flight': fp.id, 'why': 'emergency', 'decision': Decision(fp.id, 'rejected', reason='test')}]
    monkeypatch.setattr(ContingencyManager, 'emergency', refuse)
    path = tmp_path / 'refused.json.gz'
    out, rec = run_scenario.run(_tiny(), record=str(path))
    assert out['emergencies'][0]['status'] == 'rejected'
    log = flight_log.read(path)
    em = [k for k, d in enumerate(log['drones']) if d['kind'] == 'emergency'][0]
    assert log['frames']['status'][-1][em] == flight_log.CANCELLED
    assert log['frames']['t'][-1] < 150          # max_steps here is several minutes


def test_arrivals_are_events_and_no_frame_is_recorded_twice(tmp_path):
    """#37 review: 'arrived' events (one per landed drone) and frames.t strictly increasing."""
    import sys
    sys.path.insert(0, 'scripts')
    import run_scenario
    path = tmp_path / 'tiny.json.gz'
    out, rec = run_scenario.run(_tiny(), record=str(path))
    log = flight_log.read(path)
    arrived = [e['flight'] for e in log['events'] if e['type'] == 'arrived']
    assert len(arrived) == len(set(arrived)) >= out['arrived']
    t = log['frames']['t']
    assert all(a < b for a, b in zip(t, t[1:]))


def test_checkpoint_path_is_found_from_the_repo_root_or_named_clearly(tmp_path, monkeypatch):
    import sys
    sys.path.insert(0, 'scripts')
    import run_scenario
    monkeypatch.chdir(tmp_path)
    assert run_scenario.checkpoint_path('scripts/run_scenario.py').endswith('run_scenario.py')   # repo-relative
    with pytest.raises(FileNotFoundError, match='checkpoint not found'):
        run_scenario.checkpoint_path('no/such.pth')
