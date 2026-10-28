"""
telemetry.py
------------
What the UTM hears from the drones (E2): each airborne drone sends a report about once per
second, as real drones do over the network (Network Remote ID):

    {id, seq, t_sent, pos (x, y, z), vel (vx, vy, vz), intent [[t, x, y, z], ...]}

seq counts each drone's reports (set by the drone: gaps are lost reports); t_received is set by
the receiver when the report arrives (t_received - t_sent: the delay).

intent is the drone's planned path for the next ~10 s (utm/intent.py route_intent); empty for a
drone that has none to share (an intruder, tracked by surveillance: position and velocity only).

TelemetryModel turns the simulator's true states into the reports the UTM receives:

    tm = TelemetryModel(noise_m=1.0, delay_s=0.5, loss=0.05, seed=20)
    tm.send(t, {'f1': (pos, vel, intent), ...})       # every report period
    for r in tm.receive(t): ...                       # what has arrived by time t
    tm.latest                                         # id -> newest report received

    noise_m   GPS error: Gaussian, this std per axis (m) on the position; a fifth of it (m/s) on the
              velocity. The intent is the drone's own plan, sent as is.
    delay_s   network delay: a report arrives delay_s after it was sent
    loss      probability that a report is lost

Limitation: the GPS error is drawn afresh for every report; real GPS error drifts slowly (it stays
off by about the same amount for a while), which a predictor finds harder to average out.

Defaults are a perfect link (no noise, no delay, no loss): the UTM then sees the true positions,
as before E2. In Webots the reports can instead come from the drones themselves over a simulated
radio (Emitter -> Receiver); they use the same format, and receive() / latest work the same.
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

    def send(self, t, states):
        """states: {id: (pos, vel, intent)} at time t (the true values)."""
        for fid, (pos, vel, intent) in states.items():
            self.sent += 1
            seq = self._seq.get(fid, 0)
            self._seq[fid] = seq + 1
            if self.loss and self.rng.rand() < self.loss:
                self.lost += 1
                continue
            p = np.asarray(pos, dtype=float) + (self.rng.normal(0.0, self.noise, 3) if self.noise else 0.0)
            v = np.asarray(vel, dtype=float) + (self.rng.normal(0.0, self.noise / 5, 3) if self.noise else 0.0)
            self._queue.append((t + self.delay, Report(fid, float(t), p, v, [list(map(float, q)) for q in intent], seq)))

    def receive(self, t):
        """The reports that have arrived by time t (oldest first); also updates latest."""
        got = []
        for a, r in self._queue:
            if a <= t + 1e-9:
                r.t_received = float(a)
                got.append(r)
        self._queue = [(a, r) for a, r in self._queue if a > t + 1e-9]
        for r in got:
            if r.id not in self.latest or r.t_sent >= self.latest[r.id].t_sent:
                self.latest[r.id] = r
        return got

    def forget(self, fid):
        """Drop a drone that has landed or crashed (it stops reporting)."""
        self.latest.pop(fid, None)
