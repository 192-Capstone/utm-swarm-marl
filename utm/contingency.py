"""
contingency.py
--------------
E4: contingency handling. E3 (utm/conformance.py) detects drones that leave their booking; E4
changes the bookings so that everyone is safe again. The rules (agreed with the team):

    alert                       reaction
    intrusion (inside another   if it is another flight's air: re-plan that other flight first
      flight's booking or an      (it is on its plan and has more options), and this drone only
      active zone)                if that fails; inside a zone: re-plan this drone
    unknown (a drone with no    book a protective volume around it (owner 'intruder:<id>', the
      plan: an intruder)          next `intruder_horizon` s), and re-plan every flight whose booking
                                  it now clashes with
    late / off_route            re-plan this drone, but only when E3 rates the alert high risk
                                  (another booking near it soon) and it has lasted `persist` s
                                  (E3's own 1 s debounce comes first): a drone that is merely off
                                  its plan with nobody around is left alone
    early / overstay            nothing (inside the safety margins / landing, not a hazard)
    a zone added mid-flight     book it from now (D2 rules), re-plan every flight it clashes with
    an emergency request        StrategicPlanner.submit with priority > 0 takes the air it needs;
                                  re-plan the airborne flights it displaced

Re-planning an airborne drone books a new plan from its current position (StrategicPlanner.
replan): the first `grace` s are not checked (they are already happening; the onboard policy
handles them); if its goal can't be reached it returns to its start pad. If no new plan fits
an airborne drone, it keeps its old booking: it is never left without one. A flight still on the
ground is filed again instead, waiting up to `refile_wait` s (longer than a new request may: a
temporary zone or an intruder passes, and the busy sky leaves few slots); for a temporary zone
with a known end, until the zone ends and `refile_wait` s more, but never more than `max_wait` s
(the operator's limit: past it the trip is not worth flying). It is cancelled (its booking
released, it never takes off) only if that fails. A flight is re-planned at most once every
`cooldown` s.

    e4 = ContingencyManager(planner, monitor)
    actions = e4.step(t, positions)            # positions: {flight id: (x, y, z)} of airborne drones
    actions += e4.add_zone(zone, t, positions)
    actions += e4.emergency(flight_plan, t, positions)
    # each action: {'t', 'flight', 'why', 'decision'}; an approved decision carries the new
    # departure time and waypoints to send to that drone; 'why' ending in '-> cancelled' means
    # the flight must not take off

Drone failures (opt-in: ContingencyManager(planner, monitor, failures=True); the monitor must have
failures=True too, E3: utm/conformance.py; off, nothing below runs). A drone in a failure state
can no longer follow a new route (SwarmEnv.set_route refuses it), so it is never re-planned
(_replan refuses it) and the others are moved instead:

    alert                       reaction
    lost_link                   it can't hear: keep its booking ("continue as filed") and book, once,
                                  at onset, a protective volume 'lostlink:<id>': LOSTLINK_RADIUS_M
                                  around where its booking has the drone, along the whole remaining
                                  route up to its booked arrival + LOSTLINK_RELEASE_MARGIN_S, on the
                                  layers it is in or passing between, each cell's window widened by
                                  max(LOSTLINK_TIME_SLACK_S, how late/early its last report was);
                                  plus the air around its last heard position for `intruder_horizon`
                                  s. Clashing flights are re-planned then, once. Everything is
                                  released after the booked arrival + margin, or when the link is
                                  restored (the monitor goes back to ok)
    down                        release its remaining booking (and any volumes), log it
    emergency_landing           release its route booking and book a column 'landing:<id>' under
                                  its last position on every layer down to the floor, for the
                                  descent time + LANDING_HOLD_S; clashing flights are re-planned
Volumes are owned by exactly one live state: lost_link -> emergency_landing -> down hands over
(each handler drops the volumes of the one before) and nothing is left behind. These actions
are log entries only: {'kind': 'log', 'decision': Decision(status 'logged')} (they change no
route and are not rejections: a consumer counting re-plans skips kind == 'log'); the re-plans
they cause are ordinary actions.
"""

import numpy as np

from utm.airspace_zones import ZONE_PREFIX, zone_keys
from utm.conformance import FAILURE_STATES
from utm.strategic_planner import Decision, FlightPlan

IMMEDIATE = ('intrusion',)
PERSISTENT = ('late', 'off_route')            # re-planned only when E3 rates them high risk

LOSTLINK_RADIUS_M = 4.2           # m: 2 x the safety radius; the drone is not steered, so its track is unsure
LOSTLINK_RELEASE_MARGIN_S = 7.0   # s after its booked arrival (Settings.pad_after: p99 of settling)
LANDING_HOLD_S = 7.0              # s a landing column is held after the descent (same settling margin)
LOSTLINK_TIME_SLACK_S = 5.0       # s: each cell of the volume is held this long before and after its booked time
LOSTLINK_STEP_S = 1.0             # s between the booked positions the volume is built from


def _merge_windows(intervals):
    out = []
    for lo, hi in sorted(intervals):
        if out and lo <= out[-1][1]:
            out[-1][1] = max(out[-1][1], hi)
        else:
            out.append([lo, hi])
    return [tuple(iv) for iv in out]


def cells_within(planner, x, y, radius):
    """H3 cells that come within `radius` of (x, y) (StrategicPlanner.cells_near for any radius)."""
    sky = planner.sky
    reach = radius + planner._edge
    out = []
    for c in sky.disk(sky.cell(x, y), sky.k_for_radius(radius)):
        if c not in planner._centres:
            planner._centres[c] = sky.center(c)
        cx, cy = planner._centres[c]
        if (cx - x) ** 2 + (cy - y) ** 2 <= reach * reach:
            out.append(c)
    return out


class ContingencyManager:
    def __init__(self, planner, monitor=None, persist=3.0, grace=2.0, cooldown=5.0, intruder_horizon=10.0,
                 refile_wait=180.0, max_wait=1800.0, failures=False, landing_descent_ms=0.5):
        if failures and (monitor is None or not getattr(monitor, 'failures', False)):
            raise ValueError('failures=True needs a ConformanceMonitor with failures=True')
        if landing_descent_ms <= 0:
            raise ValueError('landing_descent_ms must be > 0')
        self.p, self.mon = planner, monitor
        self.failures = bool(failures)
        # m/s: a forced landing's descent (envs/disturbances.py land_vz, 0.5 by default; swarm_env.py
        # clips the commanded vz to it); the column is held for the slower of this and v_climb
        self.landing_descent_ms = float(landing_descent_ms)
        self._owners = {}                 # lostlink: / landing: volume owner now in the ledger -> its flight id
        self._lost_done = {}              # lost-link flights whose volume was released after their arrival
        self._handled = {}                # flight -> failure state whose one-off handler has run
        self._landing_end = {}            # flight -> when its landing column ends
        self.persist, self.grace, self.cooldown, self.horizon = persist, grace, cooldown, intruder_horizon
        self.refile_wait, self.max_wait = refile_wait, max_wait
        self.last = {}                    # flight id -> time of its last re-plan
        self.actions = []

    # ------------------------------------------------------------------ helpers
    def _airborne(self, fid, t, positions):
        return fid in positions

    def _failed(self, fid):
        return self.failures and self.mon.state.get(fid) in FAILURE_STATES

    def _replan(self, fid, t, positions, why, wait=None):
        if self._failed(fid):
            return None                             # it can't follow a new route (lost link, down, landing)
        if t - self.last.get(fid, -1e9) < self.cooldown:
            return None
        if self._airborne(fid, t, positions):
            self.last[fid] = t
            d = self.p.replan(fid, positions[fid], t, self.grace)
        else:                                       # not flying: waiting to take off, or done
            booked = self.p.current.get(fid)
            if booked is None or t >= booked.arrival:
                return None                         # already landed (or cancelled): nothing to do
            self.last[fid] = t
            p = self.p.plans[fid]
            self.p.release(fid, -float('inf'))
            d = self.p._submit(FlightPlan(p.id, p.start, p.goal, p.layer, max(p.t_earliest, t), p.priority,
                                          p.operator), max_delay=self.refile_wait if wait is None else wait)
            if d.status != 'approved':                 # can't fly safely any more: cancel it
                self.p.current.pop(fid, None)
                why += ' -> cancelled'
        if d.status == 'approved' and self.mon is not None:
            self.mon.plans[fid] = d                 # E3 now checks against the new booking
            self.mon.state.pop(fid, None)
            self.mon._pending.pop(fid, None)
        a = {'t': float(t), 'flight': fid, 'why': why, 'decision': d}
        self.actions.append(a)
        return a

    def _future_clashes(self, owner, t):
        """Flights whose booking (from t on) shares a key and a time with `owner`'s."""
        mine = {}
        for key, rows in self.p.ledger.items():
            ivs = [(lo, hi) for lo, hi, o in rows if o == owner and hi > t]
            if ivs:
                mine[key] = ivs
        out = set()
        for key, ivs in mine.items():
            for lo2, hi2, o in self.p.ledger[key]:
                if o != owner and o in self.p.plans and hi2 > t and any(lo < hi2 and lo2 < hi for lo, hi in ivs):
                    out.add(o)
        return sorted(out)

    # ------------------------------------------------------------------ alerts
    def step(self, t, positions, paths=None, reports=None):
        """Run E3 on the positions (if a monitor is attached; paths: the predicted paths for its
        risk='intent', E2; reports: {flight: Report} for its drone failures) and react to its
        alerts. Drones in a failure state are not in `positions` (a lost link) or are not
        re-planned; they are handled first, so the others are re-planned around them."""
        if self.mon is not None:
            if self.failures:
                self.mon.check(t, positions, paths, reports)
            else:
                self.mon.check(t, positions, paths)
        out = self._failures(t, positions) if self.failures else []
        for fid, pos in positions.items():
            if fid not in self.p.plans:             # unknown: an intruder
                out += self._intruder(fid, pos, t, positions)
                continue
            state = self.mon.state.get(fid, 'ok') if self.mon is not None else 'ok'
            risk = self.mon.risk.get(fid, '') if self.mon is not None else ''
            if state in IMMEDIATE:
                out += self._intrusion(fid, t, positions)
            elif state in PERSISTENT and risk == 'high' and t - self._since(fid, state) >= self.persist:
                a = self._replan(fid, t, positions, state)
                if a:
                    out.append(a)
        return out

    def _since(self, fid, state):
        """When the drone's current alert state started: the first of its latest run of events in
        that state (a risk change within the same state doesn't restart the clock)."""
        since = float('inf')
        for e in reversed(self.mon.events):
            if e['flight'] != fid:
                continue
            if e['state'] != state:
                break
            since = e['t']
        return since

    def _intrusion(self, fid, t, positions):
        """A drone inside air booked by another flight: re-plan that flight first, then this one."""
        other = None
        for e in reversed(self.mon.events if self.mon is not None else ()):
            if e['flight'] == fid:
                other = e['other'] if e['state'] == 'intrusion' else None
                break
        out = []
        if self._failed(fid):                       # the intruder can't be moved: the other flight can
            if other in self.p.plans and not self._failed(other):
                a = self._replan(other, t, positions, f'intrusion by {fid}')
                return [a] if a else []
            return [self._note(t, fid, f'intrusion by failed {fid}: neither flight re-planned'
                               if other in self.p.plans else f'intrusion by failed {fid}: nobody to re-plan')]
        if other in self.p.plans:
            a = self._replan(other, t, positions, f'intrusion by {fid}')
            if a:
                out.append(a)
                if a['decision'].status == 'approved':
                    return out
        a = self._replan(fid, t, positions, 'intrusion')
        if a:
            out.append(a)
        return out

    def _protect(self, owner, volume, t, positions, why, skip=None):
        """Book `volume` ({key: [(t0, t1)]}) as `owner`, replacing what it booked before (it moves
        with the drone), and re-plan every flight whose booking now clashes with it (not `skip`)."""
        self.p.release(owner, -float('inf'))
        self.p._book(owner, volume)
        out = []
        for fid in self._future_clashes(owner, t):
            if fid == skip:
                continue
            a = self._replan(fid, t, positions, why)
            if a:
                out.append(a)
        return out

    def _intruder(self, iid, pos, t, positions):
        layer = self.p.layer_of(pos[2])
        volume = {(c, layer): [(t, t + self.horizon)] for c in self.p.cells_near(pos[0], pos[1])}
        return self._protect('intruder:' + str(iid), volume, t, positions, f'intruder {iid}')

    # ------------------------------------------------------------------ drone failures
    @staticmethod
    def _owner(kind, fid):
        """Ledger owner of a failure volume (flight ids may be any type: the name uses str(fid))."""
        return kind + ':' + str(fid)

    def _note(self, t, fid, why):
        """A log entry: no route changes and it is not a rejection (kind 'log', Decision 'logged')."""
        a = {'t': float(t), 'flight': fid, 'why': why, 'kind': 'log',
             'decision': Decision(fid, 'logged', reason=why)}
        self.actions.append(a)
        return a

    def _unbook(self, owner):
        if owner in self._owners:
            self.p.release(owner, -float('inf'))
            del self._owners[owner]

    def _failures(self, t, positions):
        """Handle every drone the monitor holds in a failure state, and drop the volumes of those
        that no longer are (link restored)."""
        out = []
        failed = {f: s for f, s in self.mon.state.items() if s in FAILURE_STATES}
        # insertion order, never sorted: flight ids need not be comparable
        for fid in dict.fromkeys([*failed, *self._owners.values(), *self._handled, *self._lost_done]):
            state = failed.get(fid)
            if state is None:                       # ok again: nothing of its is left booked
                self._unbook(self._owner('lostlink', fid))
                self._unbook(self._owner('landing', fid))
                self._handled.pop(fid, None)
                self._landing_end.pop(fid, None)
                self._lost_done.pop(fid, None)
            elif state == 'lost_link':
                out += self._lost_link(fid, t, positions)
            elif state == 'emergency_landing':
                out += self._emergency_landing(fid, t, positions)
            else:
                out += self._down(fid, t)
        return out

    def _cells_at(self, x, y, layers, window, volume, radius=LOSTLINK_RADIUS_M):
        for c in cells_within(self.p, x, y, radius):
            for layer in layers:
                volume.setdefault((c, layer), []).append(window)

    def _lost_link(self, fid, t, positions):
        """Keep its booking; at onset hold the air along its whole remaining booked route (once)."""
        d, last = self.mon.booked(fid), self.mon.last_report(fid)
        owner = self._owner('lostlink', fid)
        if d is None or last is None or fid in self._lost_done:
            return []
        self._unbook(self._owner('landing', fid))
        end = d.arrival + LOSTLINK_RELEASE_MARGIN_S
        if t > end:                                 # it should be down by now: free the air
            self._unbook(owner)
            self._lost_done[fid] = t
            self._handled.pop(fid, None)
            return [self._note(t, fid, 'lost link: released after its booked arrival')]
        if owner in self._owners:
            return []                               # booked at onset; nothing moves
        self._handled[fid] = 'lost_link'
        # how far off its booking it was at its last report (late +, early -); the volume is wide in time by that
        slack = max(LOSTLINK_TIME_SLACK_S, abs(last.t_sent - self.mon.booked_time_at(fid, last.pos)))
        p, volume = self.p, {}
        start = max(t - slack, d.t_dep)             # where a drone `slack` late is now, onwards
        taus = list(np.arange(start, end, LOSTLINK_STEP_S)) or [start]
        for lo in taus:
            hi = min(lo + LOSTLINK_STEP_S, end)
            a, b = self.mon.booked_at(fid, lo), self.mon.booked_at(fid, hi)
            layers = range(p.layer_of(min(a[2], b[2])), p.layer_of(max(a[2], b[2])) + 1)
            self._cells_at(a[0], a[1], layers, (lo - slack, hi + slack), volume)
        self._cells_at(b[0], b[1], layers, (end - slack, end + slack), volume)
        x, y, z = (float(v) for v in last.pos)
        self._cells_at(x, y, [p.layer_of(z)], (t, t + self.horizon), volume)      # and where it was last heard
        self._owners[owner] = fid
        out = [self._note(t, fid, 'lost link: keeps its booking, airspace held')]
        return out + self._protect(owner, {k: _merge_windows(v) for k, v in volume.items()}, t, positions,
                                   f'lost link {fid}', skip=fid)

    def _emergency_landing(self, fid, t, positions):
        """Release its route; hold a column under it on every layer down to the floor."""
        owner = self._owner('landing', fid)
        if self._handled.get(fid) == 'emergency_landing':
            if t > self._landing_end.get(fid, -np.inf):
                self._unbook(owner)                 # the column has served its time
            return []
        last = self.mon.last_report(fid)
        if last is None:
            return []
        self._unbook(self._owner('lostlink', fid))
        self.p.release(fid, t)
        x, y, z = (float(v) for v in last.pos)
        s = self.p.s
        descent = max(0.0, z - s.layer_floor) / min(s.v_climb, self.landing_descent_ms)
        self._landing_end[fid] = t + descent + LANDING_HOLD_S
        volume = {}
        self._cells_at(x, y, range(0, self.p.layer_of(z) + 1), (t, self._landing_end[fid]), volume, s.radius)
        self._handled[fid] = 'emergency_landing'
        self._owners[owner] = fid
        out = [self._note(t, fid, 'emergency landing: route released, column held')]
        return out + self._protect(owner, {k: _merge_windows(v) for k, v in volume.items()}, t, positions,
                                   f'landing {fid}', skip=fid)

    def _down(self, fid, t):
        """It is falling or fell: its booking and any volumes of earlier states are released."""
        if self._handled.get(fid) == 'down':
            return []
        self._unbook(self._owner('lostlink', fid))
        self._unbook(self._owner('landing', fid))
        self._landing_end.pop(fid, None)
        self.p.release(fid, t)
        self._handled[fid] = 'down'
        return [self._note(t, fid, 'down: booking released')]

    # ------------------------------------------------------------------ zones and emergencies
    def add_zone(self, zone, t, positions):
        """A no-fly zone that appears now (D5): book it from t (D2 rules: permanent red zones are
        kept off by booking them until forever), route new plans around it, and re-plan every
        flight it clashes with."""
        owner = ZONE_PREFIX + zone.id
        keys = zone_keys(zone, self.p.sky, self.p.s.n_layers)
        windows = [(t, float('inf'))] if zone.permanent else [(max(t, lo), hi) for lo, hi in zone.active if hi > t]
        self.p._book(owner, {k: windows for k in keys})
        self.p.zone_keys[owner] = keys
        if zone.restriction == 'authorisation':
            for fid in zone.authorised:
                self.p.exempt.setdefault(fid, set()).add(owner)
        wait = None                                 # temporary zone: a grounded flight may wait it out
        if not zone.permanent and windows:
            end = max(hi for _, hi in windows)
            wait = min(self.max_wait, max(self.refile_wait, end - t + self.refile_wait))
        out = []
        for fid in self._future_clashes(owner, t):
            if owner in self.p.exempt.get(fid, ()):
                continue
            a = self._replan(fid, t, positions, f'zone {zone.id}', wait)
            if a:
                out.append(a)
        return out

    def emergency(self, fp, t, positions):
        """File an emergency request (priority > 0) now; re-plan the airborne flights it displaced."""
        if fp.priority <= 0:
            raise ValueError('an emergency request needs priority > 0')
        d = self.p.submit(fp, now=t)
        out = [{'t': float(t), 'flight': fp.id, 'why': 'emergency', 'decision': d}]
        self.actions.append(out[0])
        for fid in d.displaced:
            self.last.pop(fid, None)               # displaced: re-plan now, whatever the cooldown
            a = self._replan(fid, t, positions, f'displaced by {fp.id}')
            if a:
                out.append(a)
        return out
