"""
radio_msg.py
------------
E2 option 3: the drones report to the UTM over a simulated Webots radio (Emitter -> Receiver)
instead of the supervisor reading their true positions. Standard library only, so the same code
runs in each drone's controller (webots/controllers/cf_velocity) and in the supervisor
(envs/webots_l2_multi.py).

Channels (radio type, not blocked by buildings, no bandwidth limit):
    UPLINK    drones -> UTM: every drone's Emitter; the supervisor's Receiver "utm_rx"
    DOWNLINK  UTM -> drones: the supervisor's Emitter "utm_tx"; every drone's Receiver. One
              shared channel: each intent message names its drone, the others ignore it.

Report (drone -> UTM), compact JSON, the fields of utm/telemetry.py Report:
    {"id": "drone_3", "ep": 1, "seq": 0, "t": 12.344, "p": [x, y, z], "v": [vx, vy, vz],
     "i": [[t, x, y, z], ...]}
    t   Webots time of the GPS read (the supervisor turns it into episode time)
    p   the drone's own GPS + Gaussian noise (noise_m per axis; the proto's GPS has no error)
    v   displacement of the TRUE GPS since the drone's previous report / the time between them
        (as scripts/eval_layered.py computes it for TelemetryModel), + noise_m / 5 per axis.
        The first report after take-off has no previous one: v = 0 (as eval_layered).
    i   the newest intent the drone heard in this episode ([] if none)
    About 190 bytes without intent, 460 with a 5-point intent (floats in full: an intent comes
    back exactly as sent).
Intent (UTM -> drone): {"to": "drone_3", "ep": 1, "i": [[t, x, y, z], ...]}

RadioSender (one per drone controller) decides when the drone reports: once it has flown (F)
for MIN_FLIGHT_STEPS = 2 basic steps (4 ms after take-off) and then every period_s while it flies;
never while held (H) or idle, so parked drones are silent. The 2 steps: at every reset all drones
are released with F for one basic step, and SwarmEnv parks the waiting ones right after; such a
drone must not report (its report would be lost out of range and counted as lost). seq counts the drone's reports in each episode from 0. The noise is drawn from a
generator seeded by (seed, episode, drone index): the same run gives the same noise, and each
episode gets its own.
"""

import json
import random

UPLINK = 100
DOWNLINK = 101
MIN_FLIGHT_STEPS = 2                # basic steps of flight before the first report
DEFAULTS = {'enabled': False, 'period_s': 1.0, 'noise_m': 0.0, 'seed': 20, 'range_m': 200.0}
_SEP = (',', ':')


def radio_config(webots_cfg):
    """webots.radio from the config, with defaults; checks the values."""
    rc = dict(DEFAULTS)
    rc.update((webots_cfg or {}).get('radio') or {})
    unknown = set(rc) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"webots.radio: unknown keys {sorted(unknown)}")
    rc['enabled'] = bool(rc['enabled'])
    rc['period_s'], rc['noise_m'], rc['range_m'] = float(rc['period_s']), float(rc['noise_m']), float(rc['range_m'])
    rc['seed'] = int(rc['seed'])
    if rc['period_s'] <= 0 or rc['noise_m'] < 0 or not (rc['range_m'] > 0 or rc['range_m'] == -1):
        raise ValueError('webots.radio: period_s must be > 0, noise_m >= 0, range_m > 0 (or -1: infinite)')
    return rc


SENDER_KEYS = ('period_s', 'noise_m', 'seed')


def controller_args(rc):
    """controllerArgs for a drone: the settings its RadioSender needs (no quotes inside, so they
    go into the node string as they are)."""
    return ['--radio'] + [f"{k}={rc[k]!r}" for k in SENDER_KEYS]


def parse_controller_args(argv):
    """The radio settings from a controller's argv (sys.argv[1:]), or None: no radio."""
    if '--radio' not in argv:
        return None
    kv = dict(a.split('=', 1) for a in argv[argv.index('--radio') + 1:] if '=' in a)
    if set(kv) != set(SENDER_KEYS):
        raise ValueError(f"radio controller args {argv!r}: expected {SENDER_KEYS}")
    return {'period_s': float(kv['period_s']), 'noise_m': float(kv['noise_m']), 'seed': int(kv['seed'])}


def drone_index(name):
    """'drone_3' -> 3"""
    prefix, _, k = name.rpartition('_')
    if prefix != 'drone' or not k.isdigit():
        raise ValueError(f"not a drone name: {name!r}")
    return int(k)


def _floats(v, n=None):
    out = [float(x) for x in v]
    if n is not None and len(out) != n:
        raise ValueError(f"expected {n} numbers, got {len(out)}")
    return out


def _intent(intent):
    return [_floats(q, 4) for q in intent]


def encode_report(rid, ep, seq, t, pos, vel, intent):
    return json.dumps({'id': str(rid), 'ep': int(ep), 'seq': int(seq), 't': float(t),
                       'p': _floats(pos, 3), 'v': _floats(vel, 3), 'i': _intent(intent)}, separators=_SEP)


def decode_report(s):
    """JSON -> dict with id, ep, seq, t, p, v, i (checked). Raises ValueError if malformed."""
    try:
        d = json.loads(s)
        return {'id': str(d['id']), 'ep': int(d['ep']), 'seq': int(d['seq']), 't': float(d['t']),
                'p': _floats(d['p'], 3), 'v': _floats(d['v'], 3), 'i': _intent(d['i'])}
    except (KeyError, TypeError, ValueError) as e:            # json.JSONDecodeError is a ValueError
        raise ValueError(f"bad radio report {s!r}: {e}") from None


def encode_intent(to, ep, intent):
    return json.dumps({'to': str(to), 'ep': int(ep), 'i': _intent(intent)}, separators=_SEP)


def decode_intent(s):
    """JSON -> (to, ep, intent). Raises ValueError if malformed."""
    try:
        d = json.loads(s)
        return str(d['to']), int(d['ep']), _intent(d['i'])
    except (KeyError, TypeError, ValueError) as e:
        raise ValueError(f"bad radio intent {s!r}: {e}") from None


def noise_rng(seed, ep, i):
    """The noise generator of drone i in episode ep (a str seed is hashed with SHA-512:
    the same in every process, unlike hash())."""
    return random.Random(f"{int(seed)}:{int(ep)}:{int(i)}")


class RadioSender:
    """The reporting side of one drone (its controller calls this every basic step)."""

    def __init__(self, name, settings, dt):
        self.name, self.index = name, drone_index(name)
        self.period_steps = max(1, int(round(float(settings['period_s']) / dt)))
        self.noise = float(settings['noise_m'])
        self.seed = int(settings['seed'])
        self.ep = None                  # episode of the reports so far
        self.seq = 0
        self.rng = None
        self.flying = False
        self.k = 0                      # basic steps since take-off
        self.last = None                # (t, true GPS) of the previous report since take-off
        self.intent_ep, self.intent = None, []

    def hear(self, s):
        """An intent message from the UTM: kept if it is for this drone."""
        to, ep, intent = decode_intent(s)
        if to == self.name:
            self.intent_ep, self.intent = ep, intent

    def stop(self):
        """Held (H) or idle: no reports; the next F is a new take-off."""
        self.flying = False

    def step(self, ep, act, t, gps):
        """One basic step of flight (act 'arm' or 'fly', ControllerState) with the GPS read at
        Webots time t. Returns the report to send now (JSON), or None."""
        if ep != self.ep:                                       # new episode: seq and noise afresh
            self.ep, self.seq = ep, 0
            self.rng = noise_rng(self.seed, ep, self.index)
        if act == 'arm' or not self.flying:
            self.flying, self.k, self.last = True, 0, None
        # first report on the 2nd basic step of flight (MIN_FLIGHT_STEPS), then every period: the
        # reset's single release step (F, then parked at once) never produces a report
        k = self.k - (MIN_FLIGHT_STEPS - 1)
        due = k >= 0 and k % self.period_steps == 0
        self.k += 1
        if not due:
            return None
        g = _floats(gps, 3)
        if self.last is None:
            v = [0.0, 0.0, 0.0]
        else:
            t0, g0 = self.last
            v = [(g[j] - g0[j]) / (t - t0) for j in range(3)]
        self.last = (t, g)
        if self.noise:
            p = [g[j] + self.rng.gauss(0.0, self.noise) for j in range(3)]
            v = [v[j] + self.rng.gauss(0.0, self.noise / 5) for j in range(3)]
        else:
            p = g
        msg = encode_report(self.name, ep, self.seq, t, p, v, self.intent if self.intent_ep == ep else [])
        self.seq += 1
        return msg
