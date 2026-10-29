"""
telemetry.py
------------
What the UTM hears from the drones (E2): each airborne drone sends a report about once per
second, as real drones do over the network (Network Remote ID):

    {id, seq, t_sent, pos (x, y, z), vel (vx, vy, vz), intent [[t, x, y, z], ...], status}

seq counts each drone's reports (set by the drone: gaps are lost reports); t_received is set by
the receiver when the report arrives (t_received - t_sent: the delay).

intent is the drone's planned path for the next ~10 s (utm/intent.py route_intent); empty for a
drone that has none to share (an intruder, tracked by surveillance: position and velocity only).

status is the drone's own state, as Remote ID's emergency status (D5): 'ok' (default), 'emergency'
(a motor failure: it is going down) or 'landing' (a forced landing where it is).

TelemetryModel turns the simulator's true states into the reports the UTM receives:

    tm = TelemetryModel(noise_m=1.0, delay_s=0.5, loss=0.05, seed=20)
    tm.send(t, {'f1': (pos, vel, intent), ...})       # every report period
    for r in tm.receive(t): ...                       # what has arrived by time t
    tm.latest                                         # id -> newest report received
    tm.mute(fid) / tm.unmute(fid)                     # comms loss (D5): its reports are not sent

    noise_m   GPS error: Gaussian, this std per axis (m) on the position; a fifth of it (m/s) on the
              velocity. The intent is the drone's own plan, sent as is.
    delay_s   network delay: a report arrives delay_s after it was sent
    loss      probability that a report is lost

A state may carry a fourth item, the report's status: (pos, vel, intent, status). A muted drone
(comms loss) still counts its seq (so a gap shows when it comes back) but nothing goes out: it
is not in sent or lost, only in muted (the number of reports not sent). Muting draws no random
numbers, so the other drones' noise and losses stay as they were.

Limitation: the GPS error is drawn afresh for every report; real GPS error drifts slowly (it stays
off by about the same amount for a while), which a predictor finds harder to average out.

Defaults are a perfect link (no noise, no delay, no loss): the UTM then sees the true positions,
as before E2. In Webots the reports can instead come from the drones themselves over a simulated
radio (Emitter -> Receiver); they use the same format, and receive() / latest work the same.

RadioTelemetry is that radio (option 3: envs/webots_l2_multi.py with webots.radio.enabled,
envs/radio_msg.py), with TelemetryModel's interface, so the eval switches with one line:

    tm = RadioTelemetry(env.adapter, ids)    # ids[i]: the flight id of drone_i

    send(t, states)   sends each drone in states its intent (UTM -> drone, ~ once per second as
                      the eval calls it); the true pos / vel in states are IGNORED: the drones
                      report their own GPS, on their own clock (once per period while flying)
    receive(t)        the reports that arrived since the last call (the adapter stamps
                      t_received each control step, so t is not used), ids mapped to flight ids;
                      updates latest
    sent, lost        from the drones' seq numbers: sent = sum over drones of (highest seq heard
                      + 1), lost = sent - received. Blind spot: a lost LAST report (or all of a
                      drone's reports) is not seen, so lost can be undercounted by up to one
                      report per drone.
"""

from dataclasses import dataclass, field

import numpy as np

LINK_TIMEOUT_S = 3.0      # s without a report before the UTM drops a drone (three report periods)


def silent(latest, t, timeout=LINK_TIMEOUT_S):
    """Drones in latest (id -> newest report) whose newest report was sent more than timeout s before
    t: the UTM's only sign that one has landed, crashed or lost its link (it cannot see the truth)."""
    return [fid for fid, r in latest.items() if t - r.t_sent > timeout]


@dataclass
class Report:
    id: str
    t_sent: float
    pos: np.ndarray
    vel: np.ndarray
    intent: list = field(default_factory=list)        # [[t, x, y, z], ...]
    seq: int = 0                                      # per drone, from 0
    t_received: float = None
    status: str = 'ok'                                # 'ok' | 'emergency' | 'landing' (D5)


class TelemetryModel:
    def __init__(self, noise_m=0.0, delay_s=0.0, loss=0.0, seed=0):
        if noise_m < 0 or delay_s < 0 or not 0.0 <= loss < 1.0:
            raise ValueError('noise_m and delay_s must be >= 0, loss in [0, 1)')
        self.noise, self.delay, self.loss = float(noise_m), float(delay_s), float(loss)
        self.rng = np.random.RandomState(seed)
        self._queue = []                  # (arrival time, report)
        self.latest = {}                  # id -> newest report received (by t_sent)
        self.sent = self.lost = 0
        self._seq = {}                    # id -> next sequence number
        self.muted = 0                    # reports not sent: their drone had lost its link
        self._muted = set()

    def mute(self, fid):
        """Comms loss (D5): fid's reports are no longer sent."""
        self._muted.add(fid)

    def unmute(self, fid):
        self._muted.discard(fid)

    def send(self, t, states):
        """states: {id: (pos, vel, intent) or (pos, vel, intent, status)} at time t (the true values)."""
        for fid, state in states.items():
            pos, vel, intent = state[:3]
            status = state[3] if len(state) > 3 else 'ok'
            seq = self._seq.get(fid, 0)
            self._seq[fid] = seq + 1
            if fid in self._muted:
                self.muted += 1
                continue
            self.sent += 1
            if self.loss and self.rng.rand() < self.loss:
                self.lost += 1
                continue
            p = np.asarray(pos, dtype=float) + (self.rng.normal(0.0, self.noise, 3) if self.noise else 0.0)
            v = np.asarray(vel, dtype=float) + (self.rng.normal(0.0, self.noise / 5, 3) if self.noise else 0.0)
            self._queue.append((t + self.delay, Report(fid, float(t), p, v, [list(map(float, q)) for q in intent], seq,
                                                         status=str(status))))

    def receive(self, t):
        """The reports that have arrived by time t (oldest first); also updates latest."""
        got = []
        for a, r in self._queue:
            if a <= t + 1e-9:
                r.t_received = float(a)
                got.append(r)
        self._queue = [(a, r) for a, r in self._queue if a > t + 1e-9]
        _update_latest(self.latest, got)
        return got

    def forget(self, fid):
        """Drop a drone that has landed or crashed (it stops reporting)."""
        self.latest.pop(fid, None)


def _update_latest(latest, got):
    for r in got:
        if r.id not in latest or r.t_sent >= latest[r.id].t_sent:
            latest[r.id] = r


class RadioTelemetry:
    """The UTM's side of the Webots radio, as a drop-in for TelemetryModel. adapter: anything
    with reports() (Reports with ids drone_i) and send_intent(i, intent); ids[i] is drone_i's
    flight id."""

    def __init__(self, adapter, ids):
        self.adapter = adapter
        self.ids = list(ids)
        if len(set(self.ids)) != len(self.ids):
            raise ValueError('RadioTelemetry: flight ids must be unique')
        self._index = {fid: i for i, fid in enumerate(self.ids)}
        self._fid = {f'drone_{i}': fid for i, fid in enumerate(self.ids)}
        self.latest = {}
        self.received = 0
        self._max_seq = {}                # flight id -> highest seq heard

    def send(self, t, states):
        """states: {flight id: (pos, vel, intent)}: each drone is sent its intent (pos, vel unused)."""
        for fid, (_, _, intent) in states.items():
            self.adapter.send_intent(self._index[fid], [list(map(float, q)) for q in intent])

    def receive(self, t):
        """The reports that arrived since the last call (oldest first); also updates latest."""
        got = []
        for r in self.adapter.reports():
            fid = self._fid[r.id]
            got.append(Report(fid, r.t_sent, r.pos, r.vel, r.intent, r.seq, r.t_received,
                              getattr(r, 'status', 'ok')))
            self._max_seq[fid] = max(self._max_seq.get(fid, -1), r.seq)
        self.received += len(got)
        _update_latest(self.latest, got)
        return got

    def forget(self, fid):
        """Drop a drone that has landed or crashed (it stops reporting)."""
        self.latest.pop(fid, None)

    @property
    def sent(self):
        return sum(s + 1 for s in self._max_seq.values())

    @property
    def lost(self):
        return self.sent - self.received
