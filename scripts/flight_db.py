"""
flight_db.py
------------
The flight log database (utm/flight_db.py): import scenario recordings, look flights up, export for the dashboard.

    python scripts/flight_db.py import eval_results/scenarios/*.json.gz
    python scripts/flight_db.py list --operator op0 --status Landed
    python scripts/flight_db.py show ecity_demo f3
    python scripts/flight_db.py export            # dashboard/data/flight_log.json

--db sets the database file (default eval_results/flight_log.db).
"""

import argparse
import datetime
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from utm import flight_db as fdb  # noqa: E402


def clock(run, t):
    """HH:MM:SS (local time) of a sim time t on the run's start clock."""
    if t is None:
        return '--:--:--'
    start = datetime.datetime.fromisoformat(run['started_at'] or run['recorded_at']).astimezone()
    return (start + datetime.timedelta(seconds=float(t))).strftime('%H:%M:%S')


def num(v, d=1):
    return '-' if v is None else f'{v:.{d}f}'


def cmd_import(a, db):
    for path in a.recordings:
        run_id, n = fdb.import_recording(db, path)
        print(f'{path}: run {run_id}, {n} flights')


def cmd_list(a, db):
    rows = fdb.query_flights(db, a.operator, a.status, a.run)
    print(f"{'run':<22} {'callsign':<9} {'flight':<12} {'operator':<10} {'from -> to':<34} {'sched dep':>9} {'act dep':>8} "
          f"{'dep delay':>9} {'arr delay':>9}  {'status':<16} {'rp':>2} {'al':>2}")
    runs = {}
    for r in rows:
        run = runs.setdefault(r['run_id'], fdb.find_run(db, r['run_id']))
        route = f"{r['origin'] or '?'} -> {r['dest'] or '?'}"
        print(f"{r['run_id']:<22} {r['callsign']:<9} {r['flight_id']:<12} {r['operator']:<10} {route:<34} "
              f"{clock(run, r['sched_dep']):>9} {clock(run, r['act_dep']):>8} {num(r['dep_delay']):>9} "
              f"{num(r['arr_delay']):>9}  {r['status']:<16} {r['n_replans']:>2} {r['n_alerts']:>2}")
    print(f'{len(rows)} flights')


def cmd_show(a, db):
    f, events = fdb.flight_history(db, a.run, a.flight)
    run = fdb.find_run(db, a.run)
    print(f"{f['callsign']}  ({f['flight_id']}, {f['kind']}, operator {f['operator'] or '-'})  run {f['run_id']}")
    print(f"  {f['origin']} -> {f['dest']}   layer {f['layer']}, cruise {num(f['cruise_z'], 2)} m")
    print(f"  scheduled dep {clock(run, f['sched_dep'])} (T+{num(f['sched_dep'])} s)   actual dep {clock(run, f['act_dep'])} "
          f"(T+{num(f['act_dep'])} s)   delay {num(f['dep_delay'])} s")
    print(f"  scheduled arr {clock(run, f['sched_arr'])} (T+{num(f['sched_arr'])} s)   actual arr {clock(run, f['act_arr'])} "
          f"(T+{num(f['act_arr'])} s)   delay {num(f['arr_delay'])} s")
    print(f"  status {f['status']}   re-plans {f['n_replans']}   alerts {f['n_alerts']}   flown {num(f['distance_m'])} m")
    if f['min_sep_m'] is not None:
        print(f"  closest approach {f['min_sep_m']:.2f} m to {f['min_sep_with']}")
    print('  history:')
    for e in events:
        print(f"    {clock(run, e['t'])}  T+{e['t']:7.1f} s  {e['text']}")


def cmd_export(a, db):
    n_runs, n_flights = fdb.export_json(db, a.out)
    print(f'wrote {a.out}: {n_runs} runs, {n_flights} flights')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--db', default=fdb.DEFAULT_DB)
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('import', help='add recordings (re-importing one replaces its rows)')
    p.add_argument('recordings', nargs='+')
    p.set_defaults(fn=cmd_import)
    p = sub.add_parser('list', help='the departures board, as text')
    p.add_argument('--operator')
    p.add_argument('--status', help='Landed, Crashed, Forced down, Returned, Cancelled, Rejected, In flight at end, Not departed')
    p.add_argument('--run', help='run id, id prefix or scenario name')
    p.set_defaults(fn=cmd_list)
    p = sub.add_parser('show', help="one flight's history")
    p.add_argument('run')
    p.add_argument('flight', help='flight id or callsign')
    p.set_defaults(fn=cmd_show)
    p = sub.add_parser('export', help='write the JSON the dashboard page loads')
    p.add_argument('--out', default=os.path.join('dashboard', 'data', 'flight_log.json'))
    p.set_defaults(fn=cmd_export)
    a = ap.parse_args()
    db = fdb.connect(a.db)
    try:
        a.fn(a, db)
    except (KeyError, ValueError, OSError) as e:
        sys.exit(f'error: {e}')


if __name__ == '__main__':
    main()
