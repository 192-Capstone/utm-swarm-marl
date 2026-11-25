# Tracking a delivery

`dashboard/track.html` follows one delivery drone the way a delivery app follows its partner: a map, a big ETA, a status
stepper, the details and the updates. Phone first (about 390 px wide), two columns from 900 px.

| URL | what it shows | needs |
|---|---|---|
| `track.html?flight=f3` | the flight live, as the UTM server flies it | `scripts/utm_server.py` serving the page, and a login |
| `track.html?scene=ecity_demo&flight=f3` | a replay of a recorded run, with play / pause, a scrub bar and 1-16x speed | `dashboard/data/<scene>.json` (`python scripts/export_dashboard.py <recording>`); `data/flight_log.json` adds the outcome, the full history and the clock |
| `track.html?run=<run id>&flight=f3` | a delivery the live server logged, replayed from its 1 Hz track (also a refused booking: only its reason) | the live server (`GET /api/track`) |

Replay works without the server (`cd dashboard && python -m http.server`). Links: a **Track** button on every flight in
`flight_log.html` (and in its detail drawer), and a **Track** link per booked flight in `operator.html`.

## The demo command
```
python scripts/utm_server.py eval_results/demo/rooftops_demo.yaml          # then http://localhost:8770/login.html
```
It writes to the **main flight log**, `eval_results/flight_log.db` (the default; `--db FILE` for another, `--no-db` for none), so the
recorded runs (imported with `scripts/flight_db.py import`) and the live flights are in one log. Birds are on (see
`docs/birds_live.md`; `--no-birds` turns them off). Log in as `admin` (`admin-demo`) or an operator (`op0-demo`, `op1-demo`,
`hospital-demo`, docs/access.md).

## Who may track what
With the UTM server, tracking needs a login (`login.html`, see `docs/access.md`). An operator can track **only its own
flights**: `track.html` for another operator's flight shows **This is not your flight**, and `GET /api/track` answers 403 (also for
an unknown id). The admin can track any flight. Replays of recorded scenes (`?scene=`) read `data/*.json`, which holds every
operator's flights and is therefore admin-only; an operator tracks its live flights, and logged live ones with `?run=`.
Without the server (static mode) there is no login: the page is the admin's offline view and says so in a banner.

## What is on the screen
- **Map:** top-down, building footprints, no-fly zones (red while closed), the route (flown part solid, rest dashed), the
  drone with its heading, pick-up (P) and drop-off (D) pins with their place names. The view fits the route and does not
  zoom in again when a re-plan shortens it.
- **ETA:** "Arriving in 1 min 20 s"; afterwards "Delivered at 16:05:21". While the drone waits on its pad the ETA includes the
  wait ("Waiting for slot: takes off in 25 s"). A note compares the arrival with the booking: "Running late +34 s", "On time",
  "Arrived 21 s late".
- **Stepper:** Booked, Approved, Taken off, On the way, Arriving (the last 20 s), Delivered. Exceptions are shown as a coloured
  banner and a failed step: **Waiting for slot**, **Delayed** (still on its pad 5 s after its slot), **Re-routed around a zone**,
  **Returned to base**, **Forced down**, **Crashed**, **Cancelled**, **Rejected** (with E1's reason). Forced down needs a
  recording with failures; the live server cannot inject failures.
- **Details:** drone, operator, from / to, layer and height, height now, speed, distance left and flown, booked take-off and
  arrival. **Updates:** the events in the flight log's plain words, newest first.

## How the ETA is worked out (`dashboard/track_logic.js`)
One implementation for the live view and the replays, run under `node` by `tests/test_tracking.py`.
1. **Time left** follows E1's own flight profile (`utm/strategic_planner.py` `profile()`): climb at `v_climb`, cruise at
   `v_cruise`, descent at `v_descent_high` above `slow_band_m` over the pad and `v_descent` below it. The drone's position is
   projected onto the current route (never back onto an earlier segment), and the time of what remains is summed, so the climb
   still to do and the descent over the goal count. The test compares it with E1's booked time for the same route.
2. **Arrival estimate** = now + the wait for take-off + the time left, smoothed with an exponential average (4 s) so the line
   does not flicker, and taken as it is when it jumps by more than 30 s (a re-plan).
3. **Late** = the estimate minus the booked arrival; over 5 s it shows as "Running late".

The speeds are the scenario's: `export_dashboard.py` writes them into the scene JSON (`speeds`), the flight log keeps them per
run (`runs.speeds`), and the live server serves them in `/api/scene`. If none is known the page uses E1's defaults and shows
**ETA approximate (default speeds)**.

## Live details
The page polls `/api/state` four times a second and eases the drone between polls. Its clock times map simulation seconds to the
wall clock from the server's `clock` knots (see flight_log_db.md), so they stay right when the server runs faster than real time (`--speed`) or falls behind. The server keeps
every flight in memory, so a delivered flight can still be opened; after a restart open it with `?run=<run id>`.

**Birds:** while a bird is near, the map shows it (a red dot with a short trail) and a banner **Avoiding a bird** is shown while the
flight's bird guard is active. Birds are environment, not anyone's data: an operator sees the birds within 15 m of its own flying
drones. The flight's updates then read "bird added near f3 (admin)", "f3 avoided a bird (closest 1.2 m)" or "f3 hit a bird"
(and the flight is Crashed).

Recordings park a drone that has not left, or was removed after a crash, far off the map: a replay shows it on its pad until
take-off and where it fell afterwards.
