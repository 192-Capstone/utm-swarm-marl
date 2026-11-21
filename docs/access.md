# Access: who sees what

The live UTM server (`scripts/utm_server.py`) gives each delivery operator its own view, the way a delivery app shows a
company only its own partners and orders, and a central **admin** sees everything.

> **Demo-grade security. Not for production.** No HTTPS (passcodes and the session cookie cross the network in clear text),
> simple shared passcodes, sessions in memory, no lockout or audit log. The server listens on localhost only. It is there to
> show the separation between operators, not to protect real data.

## Roles
| Role | Is | May |
|---|---|---|
| `admin` | the central UTM | see every flight, event and track; book for any operator (the `operator` in the request); open the recorded scenes (`data/*.json`) |
| `op0`, `op1`, ... | a delivery operator of the scenario | see only its own flights and their events; book only as itself; track only its own flights |
| `Hospital` | the emergency operator | the same, and files emergency flights (`/api/emergency`); a delivery operator cannot |

The roles that can log in are the admin plus the scenario operators that have a passcode in the access file.

## Passcodes
`deployment/access.yaml` holds one **salted PBKDF2-SHA256 hash** per role (never the passcode). The shipped file is a **demo
file** with public passcodes:

| Role | Demo passcode |
|---|---|
| `admin` | `admin-demo` |
| `op0` | `op0-demo` |
| `op1` | `op1-demo` |
| `Hospital` | `hospital-demo` |

Change or add one (asks for the passcode, or give it as the third word; the leading comment of the file is kept):

```bash
python -m utm.access set op0                # prompts
python -m utm.access set op2 my-passcode    # a new operator
python -m utm.access hash my-passcode       # just print a hash line to paste into the file
```

Another file: `python scripts/utm_server.py <scenario> --access path/to/access.yaml`. **Without the access file the server
refuses to start.** To run with no login at all use `--no-auth` (below).

## Login
`dashboard/login.html`: choose a role, enter the passcode. `POST /api/login {role, passcode}` replies with the role and sets the
`utm_session` cookie (`HttpOnly`, `SameSite=Strict`, 8 hours). A wrong role or passcode is `401` (after a short delay).
`POST /api/logout` ends the session; `GET /api/me` says who is logged in. Sessions live in the server's memory: restarting the
server logs everyone out. Every page shows **Logged in as ...** and a **Log out** button, and goes to the login page when the
session ends.

## What is enforced where
The **server** decides. The pages only follow; a hand-typed URL or `curl` gets the same answers.

| Request | No session | Operator | Admin |
|---|---|---|---|
| `GET /api/scene` (map, places, operators) | public | public | public |
| `GET /api/state` | 401 | own flights and the events of them (or addressed to it) | all |
| `GET /api/flightlog` | 401 | own flights with their events; the runs they are in; no run-wide events | all |
| `GET /api/track?...` | 401 | own flight; **403** for another operator's flight (and for an unknown id, so ids cannot be probed) | any |
| `POST /api/request` | 401 | books as itself; `operator` for someone else is **403** | books for the `operator` given |
| `POST /api/emergency` | 401 | only the emergency operator, as itself | any operator |
| `GET /data/*.json` (every operator's flights) | 401 | **403** | served |
| the pages (`*.html`, `*.js`) | served | served | served |

- Static files: for anyone who is not the admin, only the dashboard's top-level `.html` / `.js` / `.css` / `.ico` files are
  served; anything else (the whole `data/` folder, however the path is spelled) is 401 without a session and 403 for an
  operator. The pages hold no data: it all comes from the API.
- A `POST` from a page of another site (an `Origin` that is not this server) is refused.
- Live 3D console (`index.html#live`): the server's `/api/state` is already filtered, so an operator sees only its own drones,
  routes and events. The recorded scenes in that console read `data/*.json` and are for the admin; an operator is sent to `#live`.
- Not hidden: the number of free drones (`free` in `/api/state`), and flight ids named in the text of one of your own events
  (e.g. "displaced by f5"), since those events concern your flight. The public `/api/scene` lists all operators and places.
- A flight from a recorded run in the database (an import) shows in the operator's flight log but cannot be replayed by it:
  replays read `data/<scene>.json`, which is admin-only.

## The pages
- `operator.html`: the logged-in operator, no switching; the admin picks any operator to book for.
- `flight_log.html`: the operator's flights only; the admin has everything and the operator filter.
- `track.html`: an operator opening another's flight gets **This is not your flight**.
- `index.html#live`: as above.

## Static mode (no server)
Pages opened without the server (`file://`, or `cd dashboard && python -m http.server`) read `data/*.json`; there is no login
and nothing to enforce. They show a banner, **Offline view: no login, you see everything, as the admin does**, and behave as the
admin's view. Do not publish `dashboard/data/*.json` to anyone who should not see every operator.

## `--no-auth`
`python scripts/utm_server.py <scenario> --no-auth`: no login, everyone is the admin. A loud warning is printed when the server
starts and every page shows a red banner. For a quick demo only.

## Tests
`tests/test_access.py` runs the server's own request handler with a stand-in simulation: login and logout, wrong passcodes,
401 without a session, operator A refused B's flights, track and bookings (403), per-operator flight log and state, the
admin's full view, `op0` refused `/data/flight_log.json`, cross-site posts, `--no-auth`, and the hashing.
