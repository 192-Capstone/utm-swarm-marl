"""E7 system metrics (utm/metrics.py)."""

import pytest

from utm.metrics import jain, safety_per_flight_hour, system_metrics
from utm.strategic_planner import FlightPlan, StrategicPlanner

B = ((-20, 20), (-20, 20))


def test_jain_index():
    assert jain([50, 50, 50]) == pytest.approx(1.0)
    assert jain([100, 0, 0, 0]) == pytest.approx(0.25)
    assert jain([]) == 1.0


def test_system_metrics_on_two_operators_sharing_one_corridor():
    sky = StrategicPlanner([], B)
    plans = [FlightPlan('a', (-15, 0), (15, 0), t_earliest=0.0, operator='op0'),
             FlightPlan('b', (-15, 0.5), (15, 0.5), t_earliest=0.0, operator='op1')]
    for p in plans:
        sky.submit(p)
    m = system_metrics(sky, plans, duration_s=60.0)
    d = {x.id: x for x in sky.decisions}
    assert m['approved'] == 2 and m['throughput_per_h'] == pytest.approx(120.0)
    assert m['flight_s'] == pytest.approx(sum(x.arrival - x.t_dep for x in d.values()))
    assert d['a'].delay == 0.0 and d['b'].delay > 0.0          # b came second and waits
    assert m['per_operator']['op1']['delay_mean_s'] == pytest.approx(d['b'].delay)
    assert m['delay_gap_s'] == pytest.approx(d['b'].delay)
    assert m['fairness_jain'] == pytest.approx(1.0)            # both approved


def test_safety_per_flight_hour():
    s = safety_per_flight_hour(close_passes=3, crashes=1, flight_s=1800.0)
    assert s == {'flight_h': 0.5, 'close_passes_per_h': 6.0, 'crashes_per_h': 2.0}
    with pytest.raises(ValueError):
        safety_per_flight_hour(1, 0, 0.0)
