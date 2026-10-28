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

Lost link (no position from a drone) is not handled yet (stretch goal).
"""

from utm.airspace_zones import ZONE_PREFIX, zone_keys
from utm.strategic_planner import FlightPlan

IMMEDIATE = ('intrusion',)
PERSISTENT = ('late', 'off_route')            # re-planned only when E3 rates them high risk


class ContingencyManager:
    def __init__(self, planner, monitor=None, persist=3.0, grace=2.0, cooldown=5.0, intruder_horizon=10.0,
                 refile_wait=180.0, max_wait=1800.0):
        self.p, self.mon = planner, monitor
        self.persist, self.grace, self.cooldown, self.horizon = persist, grace, cooldown, intruder_horizon
        self.refile_wait, self.max_wait = refile_wait, max_wait
        self.last = {}                    # flight id -> time of its last re-plan
        self.actions = []

    # ------------------------------------------------------------------ helpers
    def _airborne(self, fid, t, positions):
        return fid in positions

    def _replan(self, fid, t, positions, why, wait=None):
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
