"""
neighbour_index.py
------------------
Which other drones could be within a drone's sensing radius? (task B5)

The observation processor keeps each drone's K nearest neighbours within the
sensing radius. Checking every other drone costs N^2 work per step; a spatial
index returns a small candidate set per drone (every drone that COULD be in
range, maybe a few more), and the processor then applies exactly the same
distance, radius and nearest-K rules to those candidates, so the neighbours it
picks are identical.

    brute  every other drone (the reference, N^2)
    grid   square cells of side = radius in the horizontal plane; candidates are the
           drones in the 3x3 cells around the drone's own cell
    h3     H3 hexagonal cells (envs/airspace.py) at a resolution near the radius;
           candidates are the drones in grid_disk(cell, k) with k from k_for_radius

Both indexes are exact supersets: a drone within `radius` (3D) is within `radius`
horizontally, so it always lies in the cells searched.

Which to use (benchmark, constant density, candidates only): both cost about the same per
drone at any swarm size (1,000 to 100,000 drones), and the grid is about 4x faster than h3,
which first converts every position to lat/lng. So the design is a hybrid:
    grid  for the high-frequency neighbour search inside the simulation (every control step,
          local metres only)
    h3    for the UTM layer (envs/airspace.py): reservations, no-fly zones, conformance,
          demand and dashboards, where a shared, global, hierarchical cell id matters and
          queries are far less frequent
'h3' here exists to show both give identical neighbours and to compare their cost.

Config: observation.neighbor_index: brute (default) | grid | h3
        observation.h3_resolution (optional, default: chosen from the radius)
"""

import math
from collections import defaultdict

import numpy as np

KINDS = ('brute', 'grid', 'h3')


class NeighbourIndex:
    def __init__(self, kind, radius, h3_resolution=None):
        if kind not in KINDS:
            raise ValueError(f"neighbor_index must be one of {KINDS}, got {kind!r}")
        if kind != 'brute' and radius is None:
            raise ValueError("neighbor_index grid / h3 needs observation.sensing_radius")
        self.kind = kind
        self.radius = None if radius is None else float(radius)
        if kind == 'h3':
            from envs.airspace import Airspace
            res = h3_resolution if h3_resolution is not None else self._resolution_for(self.radius)
            self.airspace = Airspace(resolution=res)
            self.k = self.airspace.k_for_radius(self.radius)

    @staticmethod
    def _resolution_for(radius):
        """The finest H3 resolution whose average hexagon edge is still >= the radius / 3,
        so the searched disk stays about three rings wide (30 m -> resolution 12, 10.8 m
        edges: in benchmarks a little faster than 11, with a third of the candidates)."""
        import h3
        for res in range(15, -1, -1):
            if h3.average_hexagon_edge_length(res, 'm') >= radius / 3:
                return res
        return 0

    def candidates(self, positions):
        """For each drone i, a sorted int array of the other drones that may be within the
        radius. positions: (N, 3+) array of the broadcast positions."""
        n = len(positions)
        if self.kind == 'brute' or n <= 1:
            idx = np.arange(n)
            return [np.delete(idx, i) for i in range(n)]
        if self.kind == 'grid':
            cells = np.floor(np.asarray(positions, dtype=float)[:, :2] / self.radius).astype(np.int64)
            buckets = defaultdict(list)
            for i, (cx, cy) in enumerate(cells.tolist()):
                buckets[(cx, cy)].append(i)
            near = {}
            for (cx, cy) in buckets:
                found = []
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        found.extend(buckets.get((cx + dx, cy + dy), ()))
                near[(cx, cy)] = np.array(sorted(found), dtype=np.int64)
            return [near[tuple(c)][near[tuple(c)] != i] for i, c in enumerate(cells.tolist())]
        # h3
        ids = self.airspace.cells(np.asarray(positions, dtype=float)[:, :2])
        buckets = defaultdict(list)
        for i, c in enumerate(ids):
            buckets[c].append(i)
        near = {}
        for c in buckets:
            found = []
            for d in self.airspace.disk(c, self.k):
                found.extend(buckets.get(d, ()))
            near[c] = np.array(sorted(found), dtype=np.int64)
        return [near[c][near[c] != i] for i, c in enumerate(ids)]
