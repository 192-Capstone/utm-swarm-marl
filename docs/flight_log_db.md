# Flight log database

A persistent log of every flight across scenario runs, like an airport departures board with a history per flight.
One SQLite file (default `eval_results/flight_log.db`, git-ignored), filled from scenario recordings
(`scripts/run_scenario.py --record`, `utm/flight_log.py`). Code: `utm/flight_db.py`, `scripts/flight_db.py`.
Dashboard: `dashboard/flight_log.html`.

## Use

```
python scripts/run_scenario.py scenarios/ecity_failures.yaml --record eval_results/scenarios/ecity_failures.json.gz
python scripts/flight_db.py import eval_results/scenarios/*.json.gz     # again later: replaces that run's rows
python scripts/flight_db.py list --operator op0 --status Landed         # also --run <scenario or run id>
python scripts/flight_db.py show ecity_failures f6                      # a flight's history (id or callsign)
python scripts/flight_db.py export                                      # dashboard/data/flight_log.json
cd dashboard && python -m http.server 8000                              # then open http://localhost:8000/flight_log.html
```

`--db path` (before the command) picks another database file. The exported `flight_log.json` is generated and
git-ignored. If it is missing the page shows an empty state. The page works from the exported file only, so the
database is not needed to view it.

Direct SQL works too: `sqlite3 eval_results/flight_log.db "select callsign, status from flights"`.

## What is stored

A run is identified by `<scenario name>-<hash of the recording's content>`. Importing the same recording again
(even a copy) replaces that run's rows, so nothing is duplicated.

**runs**: `run_id`, `scenario`, `map` (JSON), `seed`, `source_file`, `recorded_at` (the recording file's time, ISO UTC),
`started_at` (the run's start clock, ISO UTC), `summary` (the run's results, JSON).

**Start clock.** The recording keeps no wall-clock duration, so `started_at` = the file time (`recorded_at`) minus the
**last frame's sim time**. Clock time of any event = `started_at` + its sim time. This treats the simulation as real time
(a run that took 70 s to compute is placed as if it lasted its simulated length), which is what a board wants.

**flights**: one row per flight, emergency flight or intruder, and one per refused booking request. Key `(run_id, flight_id)`.

| column | meaning |
|---|---|
| `callsign` | operator prefix + the number in the flight id: `OP0-07`; `HOS-E00` for an emergency flight; `INT-00` for an intruder |
| `operator`, `kind` | `kind` is `flight`, `emergency` or `intruder` (other kinds are kept as they are; `bird` is skipped) |
| `drone` | the drone slot that flew it, like a tail number: `D01` is the first drone in the recording (none for a refused request) |
| `origin`, `dest` | the scenario's place name (a hub pad as `West Hub pad 2`; a one-pad place such as a drop point or the hospital: its name), else a pad code `P01`, `P02`, ... numbered by position (x, then y) within the run, so one layout always gets the same codes. An intruder's ends are not pads: plain `(x, y)` |
| `origin_x`, `origin_y`, `dest_x`, `dest_y` | the pad coordinates (local m), shown as a tooltip and in the detail panel |
| `layer`, `cruise_z` | cruise layer, highest point of the route (m) |
| `t_request` | when the booking was filed: the request's time (only in recordings with `requests`); an emergency flight: when it was filed |
| `sched_dep`, `sched_arr` | booked take-off and arrival (sim s). An emergency flight: its booking when E4 cleared it. Intruders: none |
| `act_dep` | first frame in which the drone is flying |
| `act_arr` | its `arrived` event (only for status Landed) |
| `dep_delay`, `arr_delay` | actual minus scheduled (s) |
| `duration_s` | `act_arr` - `act_dep` (s), for a landed flight |
| `status` | see below |
| `n_replans`, `n_alerts` | approved E4 re-plans of this flight (not the emergency filing); E3 alerts about it |
| `distance_m` | path length flown, from the frames |
| `min_sep_m`, `min_sep_with` | closest approach to any other airborne drone (intruders count, birds do not) and which |

**events**: `run_id`, `seq`, `t` (sim s), `flight_id` (null for run-wide events such as the no-fly zone closing),
`type`, `text`. The text uses the wording of `scripts/render_scenario.py` (the dashboard's feed); events that wording
leaves out (arrived, crashed, forced down, failure, low alerts) get plain words here. `takeoff` and `refused` events are
added from the frames and from the refused requests.

## Status

By precedence, first match wins: **Crashed** > **Forced down** > **Returned** (E4 sent it back to its pad; it did
not reach its destination) > **Landed** (an `arrived` event, which leaves out forced-down drones) > **Cancelled**
(E4 cancelled it, or frame status 4) > **Rejected** (E1 refused the booking: the flight never flew) > **In flight at end**.
A flight that never left its pad and was not cancelled is **Not departed**.

Rejected rows come from the recording's top-level `requests` key, which `scripts/run_scenario.py` adds (every booking
request with E1's decision and reason). Older recordings without it import fine, with no Rejected rows.

## The board (`dashboard/flight_log.html`)

Columns: Flight (callsign), Drone, Operator, From → To (coordinates on hover), Layer (`L2 · 4 m`), Sched / Actual dep,
Sched / Actual arr, Duration, Delay (the arrival delay of a landing, else the departure delay), Status, Re-plans, Alerts.
Filters for operator, status and run; text search; click a header to sort. Clicking a row opens the flight's history in
a drawer over the board (close with the x or Esc). **Download CSV** saves the rows now shown (filters and sort applied).

The Status column shows the outcome plus, for a landing, its punctuality, as an airline board does: **Landed · On time**
(green; arrival within +/-`ON_TIME_S` = 5 s of the booked arrival, a constant at the top of the page script), **Landed ·
Delayed +34 s** (amber), **Landed · Early -8 s** (green). Crashed and Forced down are red, Returned amber, Cancelled and
Rejected grey. The database keeps the plain outcome (`Landed`); the punctuality is worked out by the page.

## Export

`flight_log.json`: `{schema: "utm-flight-log-db/1", generated, runs: [{run_id, scenario, map, seed, source, recorded_at,
started_at, start_epoch, events: [run-wide events]}], flights: [flights columns + events]}`. Times in it are sim seconds;
the page adds `start_epoch` (the start clock). The "Replay" link (shown when `data/<scenario>.json` exists: run
`python scripts/export_dashboard.py <recording>` for it) on the page opens `index.html#<scenario>` for the scenes listed in `SCENES` there
(keep the list in `flight_log.html` in step). `index.html` reads only the scene from the URL, so the link cannot yet
jump to a flight or time.
