"""
metrics.py
----------
E7: system-level metrics for the UTM layer, from E1's decisions (and, for safety, what was flown).

    m = system_metrics(planner, plans, duration_s=120)     # after every plan was submitted to E1
    safety_per_flight_hour(close_passes=3, crashes=0, flight_s=m['flight_s'])

    throughput        approved flights per hour of demand (departures spread over duration_s)
    delay             E1 delay of approved flights (t_dep - t_earliest): mean, p95
    flight hours      planned airborne time of approved flights (arrival - t_dep), summed
    per operator      requests, approved %, mean delay
    fairness          Jain's index of the operators' approved % (1 = all equal, 1/n = one operator
                      gets everything) and the gap between the most and least delayed operator
    safety            close passes and crashes per flight hour (needs a flown run)

E1 is first come, first served and never looks at the operator, so unfairness can only come from
where and when an operator's flights are (e.g. a hub in a busy area).
"""

import numpy as np


def jain(values):
    """Jain's fairness index of non-negative values: 1 if all equal, 1/n if one has everything."""
    x = np.asarray(values, dtype=float)
    if len(x) == 0 or not np.any(x):
        return 1.0
    return float(x.sum() ** 2 / (len(x) * (x ** 2).sum()))


def system_metrics(planner, plans, duration_s):
    """E7 metrics for `plans` (FlightPlans, all submitted to `planner`) whose departures were spread
    over `duration_s` s."""
    dec = {d.id: d for d in planner.decisions}
    ok = [dec[p.id] for p in plans if p.id in dec and dec[p.id].status == 'approved']
    delays = np.array([d.delay for d in ok]) if ok else np.zeros(0)
    flight_s = float(sum(d.arrival - d.t_dep for d in ok))
    ops = {}
    for p in plans:
        d = dec.get(p.id)
        o = ops.setdefault(p.operator, {'requests': 0, 'approved': 0, 'delays': []})
        o['requests'] += 1
        if d is not None and d.status == 'approved':
            o['approved'] += 1
            o['delays'].append(d.delay)
    per_op = {k: {'requests': o['requests'], 'approved_pct': 100.0 * o['approved'] / o['requests'],
                  'delay_mean_s': float(np.mean(o['delays'])) if o['delays'] else None}
              for k, o in sorted(ops.items())}
    op_delays = [o['delay_mean_s'] for o in per_op.values() if o['delay_mean_s'] is not None]
    return {
        'requests': len(plans),
        'approved': len(ok),
        'approved_pct': 100.0 * len(ok) / len(plans) if plans else 0.0,
        'throughput_per_h': 3600.0 * len(ok) / duration_s,
        'delay_mean_s': float(delays.mean()) if ok else 0.0,
        'delay_p95_s': float(np.percentile(delays, 95)) if ok else 0.0,
        'flight_s': flight_s,
        'per_operator': per_op,
        'fairness_jain': jain([o['approved_pct'] for o in per_op.values()]),
        'delay_gap_s': float(max(op_delays) - min(op_delays)) if op_delays else 0.0,
    }


def safety_per_flight_hour(close_passes, crashes, flight_s):
    """Close passes and crashes per flight hour (flight_s: airborne seconds flown)."""
    h = flight_s / 3600.0
    if h <= 0:
        raise ValueError('flight_s must be > 0')
    return {'flight_h': h, 'close_passes_per_h': close_passes / h, 'crashes_per_h': crashes / h}
