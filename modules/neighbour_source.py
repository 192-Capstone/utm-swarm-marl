"""
neighbour_source.py
-------------------
Answers one question for the observation processor: "what does each drone
currently know about the other drones?" The policy never sees where that
information comes from, so the source can be swapped without touching the
filtering, top-K selection or the networks.

    TruthNeighbourSource : the simulator's exact positions and velocities.
                           What we use in PyBullet today.

Planned (see docs/neighbour_information_design.md):
    a degraded source    : same, but updated only every N steps, with noise
                           and lost messages (robustness check)
    a Webots source      : the latest message each drone actually received
                           over the Webots Emitter/Receiver radio

Every source returns, for each observing drone i and other drone j:
    known_pos[i, j]  (3,)  where i believes j is
    known_vel[i, j]  (3,)  how fast i believes j is moving
    known[i, j]      bool  whether i has any information about j at all
"""

import numpy as np


class TruthNeighbourSource:
    def observe(self, positions: np.ndarray, velocities: np.ndarray):
        n = positions.shape[0]
        known_pos = np.broadcast_to(positions[None, :, :], (n, n, 3))
        known_vel = np.broadcast_to(velocities[None, :, :], (n, n, 3))
        known = ~np.eye(n, dtype=bool)
        return known_pos, known_vel, known
