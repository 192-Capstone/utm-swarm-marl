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
"""

import numpy as np

from utm.intent import VERTICAL_SEP, first_conflict

STATES = ('ok', 'late', 'early', 'off_route', 'intrusion', 'overstay', 'unknown')


class ConformanceMonitor:
    def __init__(self, planner, debounce=1.0, risk_horizon=10.0, risk='booking', vertical_sep=VERTICAL_SEP):
        if risk not in ('booking', 'intent'):
            raise ValueError("risk must be 'booking' or 'intent'")
        self.p = planner
        self.debounce = float(debounce)
        self.risk_horizon = float(risk_horizon)
        self.risk_mode = risk
        self.vertical_sep = float(vertical_sep)   # m: risk='intent' conflict height (half a layer gap)
        self.paths = {}                   # flight -> predicted path (risk='intent'), from check()
        self.plans = {d.id: d for d in planner.decisions if d.status == 'approved'}
        self.state = {}                   # flight -> confirmed state
        self.risk = {}                    # flight -> confirmed risk ('high' / 'low'; '' when ok)
        self._pending = {}                # flight -> (state, risk, since t, key, other)
        self.events = []

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
            if other != fid and first_conflict((times[keep], path[keep]), (theirs[0][keep], theirs[1][keep]),
                                               self.p.s.radius, self.vertical_sep) is not None:
                return 'high', other
        for tk, q in zip(times[keep], path[keep]):
            key = self.key_of(q)
            if key in self.p.blocked:
                return 'high', 'blocked'
            for lo, hi, owner in self.p.ledger.get(key, ()):
                if str(owner).startswith('zone:') and owner not in skip and lo <= tk < hi:
                    return 'high', owner
        return 'low', None

    def check(self, t, positions, paths=None):
        """positions: {flight id: (x, y, z)} of the airborne drones; paths (risk='intent'): {flight
        id: (times, positions)} predicted from now on, on one time grid. Returns the new events."""
        if paths is not None:
            self.paths = paths
        new = []
        for fid, pos in positions.items():
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
                'by_state': {s: sum(e['state'] == s for e in alerts) for s in STATES if s != 'ok'},
                'flights_alerted': len({e['flight'] for e in alerts})}
