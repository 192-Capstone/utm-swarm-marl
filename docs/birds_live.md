# Birds in the live system

The live UTM server (`scripts/utm_server.py`) runs the bird work of `docs/birds_report.md` in the live simulation: kinematic birds that
drones never see directly, the onboard detector and the bird guard, with the final configuration (`LIVE_BIRDS` in the server):
a 360 degree view, 12 m range, Kalman tracker (q = 2), the guard on, shared tracks off, birds not in the neighbour slots, guard layer
spacing 2 m. On by default; `python scripts/utm_server.py <scenario> --no-birds` turns it all off (no birds, detector, guard or
`/api/bird`, and the simulation behaves exactly as before).

## Adding birds while it runs
`POST /api/bird` (**admin only**: an operator gets 403, no session 401):

| field | meaning |
|---|---|
| `flight` | a flight on its pad or flying: each bird is aimed to cross that flight's route ahead of it, at the point the drone reaches 2-16 s from now (half of what is left at the most, by E1's flight profile), timed to be there with it, coming from the side |
| `place` / `at: [x, y(, z)]` | instead of a flight: the bird is at that place or point 5 s from now |
| `speed` | 1-3 m/s (default 2) |
| `count` | 1-4 (default 1); several birds cross the same point 0.8 s apart |

The reply is `{status, added, text}`; 400 for a bad request (unknown place, a flight that is not flying, speed or count out of range), 409
when too many birds are flying, 404 with `--no-birds`.

**How it is built.** The simulator never restarts, so the birds are a fixed *pool* of 12 slots (`envs/birds.py` `BirdPool`): kinematic
PyBullet bodies, created once and parked far away. Adding a bird sets a free slot flying from now (`SwarmEnv.launch_bird`); the drones'
detector sees it like any bird (a new slot starts with no tracks). A bird flies straight through the area (it is not reflected at the
map's edge) and its slot is parked again once it is past its aim time and more than 5 m outside the map, or after 60 s. With no bird
flying the detector does no work (`BirdDetector.idle`) and the guard hands the command back at once (`skip_idle`).

## What the consoles show
- `index.html#live` (3D console): birds as red spheres with a short trail; a drone is **orange** while its bird guard is active and for 2 s after it last released (the avoidance merge window; display only); the admin
  has an **Add bird** panel (a flying flight or a place, speed, count).
- `track.html`: the nearby birds on the map and an **Avoiding a bird** banner while the guard is active for that flight and for 2 s after it last released. An operator sees
  the birds within 15 m of its own drones.
- The flight log: "bird added near f3 (admin)", "f3 avoided a bird (closest 1.2 m)", "f3 hit a bird" (the flight is Crashed).

`/api/state` has `birds` (`id`, `pos`, `trail`, `r`) and, per flight, `guard` (the guard is active, or was within the last 2 s: display only); `/api/scene` has `birds: {on, pool, per_call, speed}`.

## Cost
Measured on the demo scenario (`eval_results/demo/rooftops_demo.yaml`, 16 slots, this laptop), mean ms per control step; 32 ms is real
time at `--speed 1`, so the last column is how fast the server could run:

| | `--no-birds` | birds on |
|---|---|---|
| idle | 17.1-17.6 ms (1.9x) | 18.1-18.2 ms (1.8x) |
| 5 flights flying | 19.9-20.0 ms (1.6x) | 20.8-20.9 ms (1.5x) |
| 5 flying + 4 birds | - | 21.8-22.0 ms (1.45x) |

Birds cost 5-10 %, no lag at `--speed 1`. (Measured after two savings: parked birds are not moved in PyBullet, and with nothing tracked
the guard does no per-drone work; before the second one the guard alone took ~2.3 ms of every step.)
