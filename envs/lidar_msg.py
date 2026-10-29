"""
lidar_msg.py
------------
C3: the drone side of the Webots Lidar (webots.lidar, envs/webots_lidar.py). Standard library
only, so the same code runs in each drone's controller (webots/controllers/cf_velocity) and in
the supervisor (envs/webots_l2_multi.py).

When to scan. With the Lidar on, the supervisor appends a scan tag "L<k>/<n>" to every command it
writes (format_command(...) + ' ' + format_tag(k, n)): "the supervisor reads the scan n basic
steps after writing this". A controller first sees a new command one basic step after it was
written (j = 1), so ScanTimer switches the Lidar on at j = n - 2 and reads it at j = n - 1: the
image shows the scene 4 ms (one basic step) before the supervisor reads the state, and the packet
sent then arrives exactly when the supervisor's step returns. The Lidar renders once per tag
(one basic step in n), and is off otherwise. At a reset the held drones get the tag too, so the
first observation has a scan (taken on the spawn). Commands without a tag (parked or idle drones,
a just-placed drone's first step, or the Lidar off) scan nothing.

Scan packet (drone -> supervisor, channel LIDAR_CHANNEL), raw bytes, not JSON:
    HEADER  little-endian: drone index, episode, tag k (int32 x 3), t (Webots time of the image),
            GPS x y z, IMU roll pitch yaw, cpu (seconds the controller spent copying the image out
            and switching the Lidar off; packing and sending are not included) (float64 x 8)
    ranges  layers x horizontal_res float32, the Lidar's range image as Webots gives it (layer 0
            = up, column 0 = 180 deg, going clockwise); inf = nothing in range
"""

import struct

LIDAR_CHANNEL = 102
ARG = '--lidar'                                            # controllerArgs flag
HEADER = struct.Struct('<3i8d')


def format_tag(k, n):
    return f'L{int(k)}/{int(n)}'


def split_tag(s):
    """A command string with the Lidar on -> (command without the tag, (k, n) or None).
    Raises ValueError on a malformed tag."""
    head, _, last = s.strip().rpartition(' ')
    if not last.startswith('L'):
        return s, None
    try:
        k, n = last[1:].split('/')
        k, n = int(k), int(n)
    except ValueError:
        raise ValueError(f"bad scan tag {last!r} in {s!r}: expected 'L<k>/<n>'") from None
    if n < 3:
        raise ValueError(f"bad scan tag {last!r}: n must be >= 3 (Lidar on at n-2, read at n-1)")
    return head, (k, n)


class ScanTimer:
    """Per drone controller: call step(tag) once per basic step with split_tag's tag;
    returns 'enable', 'read' or None."""

    def __init__(self):
        self.tag, self.j = None, 0

    def step(self, tag):
        if tag is None:
            self.tag, self.j = None, 0
            return None
        if tag != self.tag:
            self.tag, self.j = tag, 1
        else:
            self.j += 1
        n = tag[1]
        if self.j == n - 2:
            return 'enable'
        if self.j == n - 1:
            return 'read'
        return None


def encode_scan(idx, ep, k, t, pos, rpy, cpu, ranges):
    """ranges: the raw float32 bytes of the range image."""
    return HEADER.pack(int(idx), int(ep), int(k), float(t), *(float(v) for v in pos),
                       *(float(v) for v in rpy), float(cpu)) + bytes(ranges)


def decode_scan(b):
    """bytes -> dict(idx, ep, k, t, pos, rpy, cpu, ranges (bytes)). Raises ValueError if too short."""
    if len(b) < HEADER.size or (len(b) - HEADER.size) % 4:
        raise ValueError(f"bad scan packet: {len(b)} bytes")
    h = HEADER.unpack_from(b)
    return {'idx': h[0], 'ep': h[1], 'k': h[2], 't': h[3], 'pos': h[4:7], 'rpy': h[7:10],
            'cpu': h[10], 'ranges': b[HEADER.size:]}
