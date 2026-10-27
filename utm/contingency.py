"""
contingency.py
--------------
E4: contingency handling. E3 (utm/conformance.py) detects drones that leave their booking; E4
changes the bookings so that everyone is safe again. The rules (agreed with the team):

    alert                       reaction
    intrusion (inside another   re-plan this drone now, from where it is
      flight's booking or an
      active zone)
    unknown (a drone with no    book a protective volume around it (owner 'intruder:<id>', the
      plan: an intruder)          next `intruder_horizon` s), and re-plan every flight whose booking
                                  it now clashes with
    late / off_route            re-plan this drone, but only once the alert has lasted `persist` s
                                  (E3's own 1 s debounce comes first), so small wobbles don't
                                  reshuffle everyone's bookings
    early / overstay            nothing (inside the safety margins / landing, not a hazard)
    a zone added mid-flight     book it from now (D2 rules), re-plan every flight it clashes with
    an emergency request        StrategicPlanner.submit with priority > 0 takes the air it needs;
                                  re-plan the airborne flights it displaced

Re-planning an airborne drone books a new plan from its current position (StrategicPlanner.
replan): the first `grace` s are not checked (they are already happening; the onboard policy
handles them); if its goal can't be reached it returns to its start pad. A flight still on the
ground is filed again instead, and cancelled (its booking released, it never takes off) if that
fails. If no new plan fits an airborne drone, it keeps its old booking: it is never left without
one. A flight is re-planned at most once every `cooldown` s.

    e4 = ContingencyManager(planner, monitor)
    actions = e4.step(t, positions)            # positions: {flight id: (x, y, z)} of airborne drones
    actions += e4.add_zone(zone, t, positions)
    actions += e4.emergency(flight_plan, t, positions)
    # each action: {'t', 'flight', 'why', 'decision'}; an approved decision carries the new
    # departure time and waypoints to send to that drone; 'why' ending in '-> cancelled' means
    # the flight must not take off

Lost link (no position from a drone) is not handled yet (stretch goal).
"""

from utm.airspace_zones import ZONE_PREFIX, zone_keys
from utm.strategic_planner import FlightPlan

IMMEDIATE = ('intrusion',)
PERSISTENT = ('late', 'off_route')


class ContingencyManager:
    def __init__(self, planner, monitor=None, persist=3.0, grace=2.0, cooldown=5.0, intruder_horizon=10.0):
        self.p, self.mon = planner, monitor
        self.persist, self.grace, self.cooldown, self.horizon = persist, grace, cooldown, intruder_horizon
        self.last = {}                    # flight id -> time of its last re-plan
        self.actions = []

    # ------------------------------------------------------------------ helpers
    def _airborne(self, fid, t, positions):
        return fid in positions

    def _replan(self, fid, t, positions, why):
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
                                          p.operator))
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
    def step(self, t, positions):
        """Run E3 on the positions (if a monitor is attached) and react to its alerts."""
        if self.mon is not None:
            self.mon.check(t, positions)
        out = []
        for fid, pos in positions.items():
            if fid not in self.p.plans:             # unknown: an intruder
                out += self._intruder(fid, pos, t, positions)
                continue
            state = self.mon.state.get(fid, 'ok') if self.mon is not None else 'ok'
            if state in IMMEDIATE:
                a = self._replan(fid, t, positions, state)
            elif state in PERSISTENT and t - self._since(fid, state) >= self.persist:
                a = self._replan(fid, t, positions, state)
            else:
                a = None
            if a:
                out.append(a)
        return out

    def _since(self, fid, state):
        """When the drone's current alert state started (E3's confirmed event time)."""
        for e in reversed(self.mon.events):
            if e['flight'] == fid:
                return e['t'] if e['state'] == state else float('inf')
        return float('inf')

    def _intruder(self, iid, pos, t, positions):
        owner = 'intruder:' + str(iid)
        self.p.release(owner, -float('inf'))       # move the protective volume with the drone
        layer = self.p.layer_of(pos[2])
        self.p._book(owner, {(c, layer): [(t, t + self.horizon)] for c in self.p.cells_near(pos[0], pos[1])})
        out = []
        for fid in self._future_clashes(owner, t):
            a = self._replan(fid, t, positions, f'intruder {iid}')
            if a:
                out.append(a)
        return out

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
        out = []
        for fid in self._future_clashes(owner, t):
            if owner in self.p.exempt.get(fid, ()):
                continue
            a = self._replan(fid, t, positions, f'zone {zone.id}')
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
