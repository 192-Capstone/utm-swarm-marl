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
no alarm. Every confirmed change is an event {t, flight, state, key, other}; the list is what the
dashboard (Module 4) shows and what E4 (contingency) will react to.
"""

import numpy as np

STATES = ('ok', 'late', 'early', 'off_route', 'intrusion', 'overstay', 'unknown')


class ConformanceMonitor:
    def __init__(self, planner, debounce=1.0):
        self.p = planner
        self.debounce = float(debounce)
        self.plans = {d.id: d for d in planner.decisions if d.status == 'approved'}
        self.state = {}                   # flight -> confirmed state
        self._pending = {}                # flight -> (state, since t, key, other)
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

    def check(self, t, positions):
        """positions: {flight id: (x, y, z)} of the airborne drones. Returns the new events."""
        new = []
        for fid, pos in positions.items():
            s, key, other = self.classify(fid, pos, t)
            cur = self.state.get(fid, 'ok')
            if s == cur:
                self._pending.pop(fid, None)
                continue
            pend = self._pending.get(fid)
            if pend is None or pend[0] != s:
                self._pending[fid] = (s, t, key, other)
                pend = self._pending[fid]
            if t - pend[1] >= self.debounce:
                self.state[fid] = s
                self._pending.pop(fid, None)
                ev = {'t': float(pend[1]), 'flight': fid, 'state': s, 'key': key, 'other': other}
                self.events.append(ev)
                new.append(ev)
        return new

    def summary(self):
        alerts = [e for e in self.events if e['state'] != 'ok']
        return {'alerts': len(alerts),
                'by_state': {s: sum(e['state'] == s for e in alerts) for s in STATES if s != 'ok'},
                'flights_alerted': len({e['flight'] for e in alerts})}
