"""
conformance.py
--------------
E3: conformance monitoring. Watches flown positions against the bookings E1 approved and raises
an alert when a drone leaves its plan.

    mon = ConformanceMonitor(planner)                  # after E1 has decided (utm/strategic_planner.py)
    events = mon.check(t, {'f1': (x, y, z), ...})      # airborne drones only; call ~1 Hz or faster
    mon.summary()

Each airborne drone is in one state, from its current key (H3 cell, layer) and the time:
    ok          the key is in its own booking at a booked time
    late        the key is in its own booking, but its window there has passed (or not reached
    early       yet: early); in its corridor, off its timing
    off_route   the key is not in its own booking at all
    intrusion   the key is booked by another flight at this time, or by an active no-fly zone
                (D2: 'zone:<id>', or a permanent red zone: 'blocked'); zones the flight is
                authorised for don't count
    overstay    still airborne over its goal pad after the pad booking ended
    unknown     an airborne drone without an approved plan (intruder)
A state counts only after it has lasted `debounce` seconds (1 s): a single noisy sample raises
no alarm.

Each alert also gets a risk level, because "off its plan" is not the same as "about to meet
someone" (under disruptions only ~1 in 8 off-plan alerts was followed by a close pass):
    high   intrusion; or another flight (or a zone) has air booked within the drone's safety
           radius of where it actually is, at some time in the next `risk_horizon` s (10 s):
           a conflict is possible ('other' names it)
    low    off plan, but nobody else is booked around it soon
With risk='intent' (E2) the risk comes from where the drones are going instead of the bookings:
check() also gets every airborne drone's predicted path (utm/intent.py: its shared intent, or its
last position and velocity) and an alert is high if it is an intrusion, or the drone's path comes
within the safety radius of another drone's path horizontally and within `vertical_sep` (1 m,
half a layer) vertically, as E1 separates drones, or enters a no-fly zone, in the next
`risk_horizon` s. A drone that is late with nobody heading its way is then low; two drones on
different layers crossing one above the other are not a conflict.
A change of risk with the same state (low -> high as a late drone nears traffic) is an event
too. Every confirmed change is an event {t, flight, state, risk, key, other}; the dashboard
(Module 4) shows people the high-risk ones, and E4 (contingency) reacts to all of them.

Drone failures (opt-in: ConformanceMonitor(planner, failures=True); off, nothing below runs and
the output is as before). The UTM judges them only from what the drones report, so check() also
gets reports ({flight: Report}, e.g. tm.latest; utm/telemetry.py): status, vel and the time sent.
Three more states, for flights with a plan that have been heard at least once (pass the
telemetry's GPS noise as noise_m=tm.noise; it only widens the arrival test below):
    lost_link          no report for LINK_TIMEOUT_S (that is its debounce). Risk is judged at the
                       position its booking gives for now: it is assumed to fly on as filed.
    down               the drone reported status 'emergency', or its last vz is below DOWN_VZ
                       (healthy drones are limited to 2 m/s). Confirmed at once.
    emergency_landing  status 'landing' or 'landed' away from the goal pad, or descending (vz < LANDING_VZ in every report for
                       LANDING_DEBOUNCE_S) below the height of its CURRENT booking
                       (planner.current[fid], so a re-plan that moves it down a layer is not a
                       landing), away from its own goal pad. The descent test only fires once the
                       drone is HEIGHT_MARGIN_M (3 m) below its booked height: from the low layers
                       (6-8 m) that is below about 3-5 m, so a forced landing from lower down is
                       caught by its 'landing' status, not by the descent.
Precedence is down > emergency_landing > lost_link. down and emergency_landing are sticky; a
lost_link drone whose reports resume goes back to ok. A flight whose last reports (their mean, up
to 3) were on its goal pad (z within CLIMB_DONE_M + FINAL_DESCENT_M of it: a drone whose 'landed'
report was lost, on its last metres down), or close(fid) was called, has arrived: it going quiet is
not a lost link. Status 'landed' (on the ground, like Remote ID's on-ground status) settles it
at once: within the goal tolerance (xy: the safety radius + 3 noise_m, for a single report) the flight
is closed; away from the goal it is an unplanned landing: emergency_landing is latched (unless it is
already down) and the drone is never lost_link afterwards.
The monitor keeps the last report of each flight itself, so a drone the eval dropped from
tm.latest (forget after LINK_TIMEOUT_S) still counts as silent, not as gone.
Known limit: from silence alone a motor failure looks like comms loss and is handled as lost_link
(safe: it holds its airspace until its booked arrival); only a report tells them apart.
"""

from collections import deque

import numpy as np

from utm.intent import VERTICAL_SEP, first_conflict
from utm.telemetry import LINK_TIMEOUT_S, silent

STATES = ('ok', 'late', 'early', 'off_route', 'intrusion', 'overstay', 'unknown')
FAILURE_STATES = ('lost_link', 'down', 'emergency_landing')      # only with failures=True
_RANK = {'lost_link': 1, 'emergency_landing': 2, 'down': 3}      # precedence: higher wins

# Thresholds against the telemetry noise (utm/telemetry.py TelemetryModel(noise_m): position error N(0, noise_m)
# per axis, velocity error N(0, noise_m / 5)); the values are for noise_m = 1 (z sd 1 m, vz sd 0.2 m/s).
REF_NOISE_M = 1.0                 # m: the GPS noise the constants below are set for
CLOSE_REPORTS = 3                 # the arrival test averages this many of a flight's last reports
DOWN_VZ = -3.0                    # m/s: a fall. Healthy |vz| <= 2, so -2 - 5 vz sd = -3.0 (rounded down to 0.5)
LANDING_VZ = -0.2                 # m/s: descending
LANDING_DEBOUNCE_S = 2.0          # s of descent below the booking before it is a landing
HEIGHT_MARGIN_M = 3.0             # m below the booked height: 3 z sd (a layer is 2 m, so it only
                                  # works above the lowest layers; below them a landing is reported)
FINAL_DESCENT_M = 1.5             # m: on top of CLIMB_DONE_M, the last metres of a landing in which the 'landed'
                                  # report may not have been sent (or was lost) before the drone goes quiet
CLIMB_DONE_M = 0.6                # m: as envs/swarm_env.py (tests check it; utm/ must not import torch)


class ConformanceMonitor:
    def __init__(self, planner, debounce=1.0, risk_horizon=10.0, risk='booking', vertical_sep=VERTICAL_SEP,
                 failures=False, noise_m=REF_NOISE_M, blind_margin=0.0):
        if risk not in ('booking', 'intent'):
            raise ValueError("risk must be 'booking' or 'intent'")
        self.p = planner
        self.debounce = float(debounce)
        self.risk_horizon = float(risk_horizon)
        self.risk_mode = risk
        self.vertical_sep = float(vertical_sep)   # m: risk='intent' conflict height (half a layer gap)
        self.paths = {}                   # flight -> predicted path (risk='intent'), from check()
        # risk='intent': a drone with no flight plan (an intruder) shares no intent and is predicted from
        # its velocity alone, so pairs with one get blind_margin m more horizontal radius (E2 follow-up,
        # option B, small version; 0 = as before). The caller sets `blind` (ids with no plan) before check().
        self.blind_margin = float(blind_margin)
        self.blind = set()
        self.plans = {d.id: d for d in planner.decisions if d.status == 'approved'}
        self.state = {}                   # flight -> confirmed state
        self.risk = {}                    # flight -> confirmed risk ('high' / 'low'; '' when ok)
        self._pending = {}                # flight -> (state, risk, since t, key, other)
        self.events = []
        self.failures = bool(failures)    # detect lost_link / down / emergency_landing (see above)
        self._last = {}                   # flight -> newest Report heard (kept after tm.forget)
        self._latched = {}                # flight -> (down | emergency_landing, since t): sticky
        self._descent = {}                # flight -> t_sent of the first report of its current descent
        self.noise_m = float(noise_m)     # m: the telemetry's GPS noise per axis (TelemetryModel.noise): arrival test
        self._hist = {}                   # flight -> its last CLOSE_REPORTS reports, oldest first
        self._closed = set()              # flights that have arrived (close())
        self._profiles = {}               # flight -> (decision, profile): booked_at's cache

    def key_of(self, pos):
        x, y, z = (float(v) for v in pos)
        return self.p.sky.cell(x, y), self.p.layer_of(z)

    def classify(self, fid, pos, t):
        """(state, key, other) for one drone right now, without debouncing."""
        key = self.key_of(pos)
        d = self.plans.get(fid)
        if d is None:
            return 'unknown', key, None
        if key in self.p.blocked:
            return 'intrusion', key, 'blocked'
        skip = self.p.exempt.get(fid, ())
        for lo, hi, owner in self.p.ledger.get(key, ()):
            if owner != fid and owner not in skip and lo <= t < hi:
                return 'intrusion', key, owner
        ivs = d.reservation.get(key)
        if not ivs:
            return 'off_route', key, None
        if any(lo <= t < hi for lo, hi in ivs):
            return 'ok', key, None
        goal = np.asarray(d.waypoints[-1][:2])
        if t > d.arrival and np.hypot(*(np.asarray(pos[:2], dtype=float) - goal)) <= self.p.s.radius:
            return 'overstay', key, None
        return ('early' if t < min(lo for lo, _ in ivs) else 'late'), key, None

    def assess(self, fid, pos, t, state, other):
        """(risk, other) of an alert: 'high' if it is an intrusion, or if air within the safety
        radius of the drone's actual position is booked by another flight or a zone during
        [t, t + risk_horizon]; else 'low'."""
        if state == 'ok':
            return '', None
        if state == 'intrusion':
            return 'high', other
        if self.risk_mode == 'intent' and fid in self.paths:
            return self._assess_intent(fid, t)
        layer = self.p.layer_of(float(pos[2]))
        skip = set(self.p.exempt.get(fid, ())) | {fid}
        until = t + self.risk_horizon
        for c in self.p.cells_near(float(pos[0]), float(pos[1])):
            key = (c, layer)
            if key in self.p.blocked:
                return 'high', 'blocked'
            for lo, hi, owner in self.p.ledger.get(key, ()):
                if owner not in skip and lo < until and t < hi:
                    return 'high', owner
        return 'low', None

    def _assess_intent(self, fid, t):
        """E2 risk: another drone's predicted path within the safety radius of this one's horizontally
        and within vertical_sep vertically, or this one's path through an active no-fly zone, in
        [t, t + risk_horizon]."""
        times, path = self.paths[fid]
        keep = times <= t + self.risk_horizon + 1e-9
        skip = set(self.p.exempt.get(fid, ()))
        for other, theirs in self.paths.items():
            r = self.p.s.radius + (self.blind_margin if fid in self.blind or other in self.blind else 0.0)
            if other != fid and first_conflict((times[keep], path[keep]), (theirs[0][keep], theirs[1][keep]),
                                               r, self.vertical_sep) is not None:
                return 'high', other
        for tk, q in zip(times[keep], path[keep]):
            key = self.key_of(q)
            if key in self.p.blocked:
                return 'high', 'blocked'
            for lo, hi, owner in self.p.ledger.get(key, ()):
                if str(owner).startswith('zone:') and owner not in skip and lo <= tk < hi:
                    return 'high', owner
        return 'low', None

    # ------------------------------------------------------------------ drone failures
    def booked(self, fid):
        """The flight's current booking: the newest approved decision the planner holds, or None
        (never approved, or cancelled: a rejected re-plan leaves the old booking in place)."""
        d = self.p.current.get(fid)
        return d if d is not None and d.status == 'approved' else None

    def last_report(self, fid):
        """The newest report heard from the flight (kept after the eval forgets it), or None."""
        return self._last.get(fid)

    def _profile(self, fid):
        d = self.booked(fid)
        if d is None:
            return None, None
        cached = self._profiles.get(fid)
        if cached is None or cached[0] is not d:          # keyed on the decision object
            route = [(x, y) for x, y, _ in d.waypoints[1:-1]]
            prof, _ = self.p.profile(route, d.layer, d.waypoints[0][2])
            cached = self._profiles[fid] = (d, np.asarray(prof, dtype=float))
        return d, cached[1]

    def booked_at(self, fid, t):
        """(x, y, z) where the flight's current booking has the drone at time t: on its start pad
        before departure, at its goal after arrival. None if it has no booking."""
        d, prof = self._profile(fid)
        if d is None:
            return None
        tau = t - d.t_dep
        return tuple(float(np.interp(tau, prof[:, 0], prof[:, j])) for j in (1, 2, 3))

    def booked_time_at(self, fid, pos):
        """The time at which the flight's booking is nearest (3D) to pos; None if it has no booking.
        A report's t_sent minus this is how late (+) or early (-) the drone was."""
        d, prof = self._profile(fid)
        if d is None:
            return None
        k = int(np.argmin(np.linalg.norm(prof[:, 1:] - np.asarray(pos, dtype=float), axis=1)))
        return float(d.t_dep + prof[k, 0])

    def close(self, fid):
        """The flight has arrived (landed on its goal pad): it going quiet is not a lost link."""
        self._closed.add(fid)

    def _closed_flight(self, fid, d):
        """Arrived: close() was called, or the mean of its last (up to 3) reports is on its goal pad:
        xy within the safety radius, z at the pad (or within its last FINAL_DESCENT_M) or the last report sent within LINK_TIMEOUT_S of
        the booked arrival. One report is not enough: with GPS noise it can land outside the pad
        for a drone that has settled on it; the mean of n has 1/sqrt(n) of the error, and the
        tolerance is widened by 3 of those."""
        if fid in self._closed:
            return True
        hist = self._hist.get(fid)
        if not hist:
            return False
        n = len(hist)
        tol = 3.0 * self.noise_m / np.sqrt(n)
        mean = np.mean([np.asarray(r.pos, dtype=float) for r in hist], axis=0)
        goal = d.waypoints[-1]
        if np.hypot(*(mean[:2] - np.asarray(goal[:2]))) > self.p.s.radius + tol:
            return False
        return bool(mean[2] <= goal[2] + CLIMB_DONE_M + FINAL_DESCENT_M + tol or hist[-1].t_sent >= d.arrival - LINK_TIMEOUT_S)

    def _latch(self, fid, state, since):
        cur = self._latched.get(fid)
        if cur is None or _RANK[state] > _RANK[cur[0]]:
            self._latched[fid] = (state, float(since))

    def _hear(self, fid, r, d):
        """Take one new report into account: sticky down / emergency_landing."""
        vz = float(r.vel[2])
        if r.status == 'emergency' or vz < DOWN_VZ:
            self._latch(fid, 'down', r.t_sent)
        if r.status == 'landing':
            self._latch(fid, 'emergency_landing', r.t_sent)
        goal = d.waypoints[-1]
        if r.status == 'landed':                          # on the ground: settled at once
            off = np.hypot(*(np.asarray(r.pos[:2], dtype=float) - np.asarray(goal[:2])))
            if off <= self.p.s.radius + 3.0 * self.noise_m:
                self._closed.add(fid)
            else:
                self._latch(fid, 'emergency_landing', r.t_sent)      # keeps down: it ranks higher
        floor = None
        booked = self.booked_at(fid, r.t_sent)
        if booked is not None:
            floor = min(booked[2], self.p.layer_z(d.layer)) - HEIGHT_MARGIN_M
        away = np.hypot(*(np.asarray(r.pos[:2], dtype=float) - np.asarray(goal[:2]))) > self.p.s.radius
        if floor is not None and away and vz < LANDING_VZ and float(r.pos[2]) < floor:
            since = self._descent.setdefault(fid, r.t_sent)
            if r.t_sent - since >= LANDING_DEBOUNCE_S:
                self._latch(fid, 'emergency_landing', since)
        else:
            self._descent.pop(fid, None)

    def _check_failures(self, t, reports):
        new = []
        for fid, r in (reports or {}).items():
            d = self.booked(fid)
            if d is None or (fid in self._last and r.t_sent <= self._last[fid].t_sent):
                continue                                  # no booking (an intruder), or an old report
            self._last[fid] = r
            self._hist.setdefault(fid, deque(maxlen=CLOSE_REPORTS)).append(r)
            self._hear(fid, r, d)
        quiet = set(silent(self._last, t))
        for fid, r in self._last.items():
            d = self.booked(fid)
            if d is None:
                continue                                  # its booking is gone (cancelled)
            latched = self._latched.get(fid)
            if latched is not None:
                s, since = latched
                pos = r.pos
                other = None
                risk = 'high'
            elif fid in quiet and not self._closed_flight(fid, d):
                s, since = 'lost_link', r.t_sent
                pos = self.booked_at(fid, t)
                risk, other = self.assess(fid, pos, t, s, None)
            else:
                if self.state.get(fid) == 'lost_link':    # reports resumed
                    s, risk, since, pos, other = 'ok', '', t, r.pos, None
                else:
                    continue
            if (s, risk) == (self.state.get(fid, 'ok'), self.risk.get(fid, '')):
                continue
            self.state[fid], self.risk[fid] = s, risk
            self._pending.pop(fid, None)
            ev = {'t': float(since), 'flight': fid, 'state': s, 'risk': risk, 'key': self.key_of(pos), 'other': other}
            self.events.append(ev)
            new.append(ev)
        return new

    def check(self, t, positions, paths=None, reports=None):
        """positions: {flight id: (x, y, z)} of the airborne drones; paths (risk='intent'): {flight
        id: (times, positions)} predicted from now on, on one time grid; reports (failures=True):
        {flight id: Report}, the newest the UTM has heard from each drone (tm.latest). Returns the
        new events."""
        if paths is not None:
            self.paths = paths
        new = self._check_failures(t, reports) if self.failures else []
        for fid, pos in positions.items():
            if self.failures and self.state.get(fid) in FAILURE_STATES:
                continue                                  # a failed drone is not judged by its plan
            s, key, other = self.classify(fid, pos, t)
            r, why = self.assess(fid, pos, t, s, other)
            other = other if other is not None else why
            cur = (self.state.get(fid, 'ok'), self.risk.get(fid, ''))
            if (s, r) == cur:
                self._pending.pop(fid, None)
                continue
            pend = self._pending.get(fid)
            if pend is None or pend[:2] != (s, r):
                self._pending[fid] = (s, r, t, key, other)
                pend = self._pending[fid]
            if t - pend[2] >= self.debounce:
                self.state[fid], self.risk[fid] = s, r
                self._pending.pop(fid, None)
                ev = {'t': float(pend[2]), 'flight': fid, 'state': s, 'risk': r, 'key': key, 'other': pend[4]}
                self.events.append(ev)
                new.append(ev)
        return new

    def summary(self):
        alerts = [e for e in self.events if e['state'] != 'ok']
        return {'alerts': len(alerts),
                'high_risk': sum(e['risk'] == 'high' for e in alerts),
                'by_state': {s: sum(e['state'] == s for e in alerts)
                             for s in STATES + (FAILURE_STATES if self.failures else ()) if s != 'ok'},
                'flights_alerted': len({e['flight'] for e in alerts})}
