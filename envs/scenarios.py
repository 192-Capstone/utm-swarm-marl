"""
scenarios.py
------------
Spawn/goal generators for structured multi-drone scenarios, used both for
training (SwarmEnv's scenario mix) and for evaluation (record_rollout.py,
eval_suite.py, baseline_pcontroller.py).

Each generator takes an RNG (np.random.RandomState, or the np.random module
itself) and the number of drones, and returns two dicts:
    spawns = {'drone_i': [x, y, z]},  goals = {'drone_i': [x, y, z]}

The order of RNG calls in each generator is fixed. Changing it changes every
scenario drawn from a given seed, which breaks reproducibility of the
evaluation results, so don't reorder them.

    headon   : drones evenly spaced on a circle (radius 3-5 m), same height,
               each flying to the exact opposite point. All straight paths
               meet at the centre at the same time.
    crossing : like headon, with random jitter in angle (+-15 deg) and radius
               (x0.8-1.2), radius 2.5-6 m, so paths cross near the centre
               but not exactly at one point.
    swap     : random spawns in a 20 x 20 m area (heights 1-5 m, at least 5 m
               apart). Each drone's goal is another drone's spawn point, no
               drone keeps its own.
    flythrough : drone_0 hovers on its goal at the centre; every other drone
               flies straight through that point (training version of the
               intruder / blocker evaluation tests).

With collision physics off, head-on and crossing are real collision courses
for a straight-flying controller. Swap with few drones rarely produces real
conflicts, since random heights and timings mean drones usually pass each
other at a distance.
"""

import numpy as np


def headon(rng, n_agents, r_min=3.0, r_max=5.0):
    R = rng.uniform(r_min, r_max)
    z = rng.uniform(2.0, 3.0)
    base_angle = rng.uniform(0, 360)    # only the overall orientation is random

    angles = base_angle + np.linspace(0, 360, n_agents, endpoint=False)
    spawns, goals = {}, {}
    for i, ang in enumerate(angles):
        rad = np.radians(ang)
        spawns[f'drone_{i}'] = [R * np.cos(rad), R * np.sin(rad), z]
        goals[f'drone_{i}'] = [-R * np.cos(rad), -R * np.sin(rad), z]
    return spawns, goals


def crossing(rng, n_agents, r_min=2.5, r_max=6.0, shared_altitude=True):
    R = rng.uniform(r_min, r_max)
    base_angle = rng.uniform(0, 360)
    jitter = rng.uniform(-15, 15, size=n_agents)
    radius_jitter = rng.uniform(0.8, 1.2, size=n_agents)
    shared_z = rng.uniform(1.5, 3.5)
    z_vals = np.full(n_agents, shared_z) if shared_altitude else rng.uniform(1.5, 3.5, size=n_agents)

    angles = base_angle + np.linspace(0, 360, n_agents, endpoint=False) + jitter
    spawns, goals = {}, {}
    for i, ang in enumerate(angles):
        rad = np.radians(ang)
        r = R * radius_jitter[i]
        spawns[f'drone_{i}'] = [r * np.cos(rad), r * np.sin(rad), float(z_vals[i])]
        goals[f'drone_{i}'] = [-r * np.cos(rad), -r * np.sin(rad), float(z_vals[i])]
    return spawns, goals


def swap(rng, n_agents, reach=10.0, min_spawn_sep=5.0):
    def rand_xyz():
        return np.array([rng.uniform(-reach, reach), rng.uniform(-reach, reach),
                         rng.uniform(1.0, 5.0)])

    spawns_arr = []
    for _ in range(n_agents):
        cand = rand_xyz()
        tries = 0
        while any(np.linalg.norm(cand - s) < min_spawn_sep for s in spawns_arr) and tries < 50:
            cand = rand_xyz()
            tries += 1
        spawns_arr.append(cand)

    perm = rng.permutation(n_agents)
    while any(perm == np.arange(n_agents)):     # no drone keeps its own spot
        perm = rng.permutation(n_agents)

    spawns = {f'drone_{i}': spawns_arr[i].tolist() for i in range(n_agents)}
    goals = {f'drone_{i}': spawns_arr[perm[i]].tolist() for i in range(n_agents)}
    return spawns, goals


def flythrough(rng, n_agents, r_min=3.0, r_max=5.0):
    """drone_0 starts on its own goal at the centre and just has to hover
    there. Every other drone flies a straight line THROUGH that point, from
    one side of a circle to the opposite side, at random angles and slightly
    different distances (so they don't all arrive at once). The training
    version of the intruder/blocker tests: the hovering drone must move out of
    the way, or the others must route around it."""
    centre = np.array([rng.uniform(-3, 3), rng.uniform(-3, 3), rng.uniform(2.0, 3.0)])
    base_angle = rng.uniform(0, 180)
    R = rng.uniform(r_min, r_max)
    radius_jitter = rng.uniform(0.8, 1.2, size=n_agents)
    angle_jitter = rng.uniform(-15, 15, size=n_agents)

    spawns = {'drone_0': centre.tolist()}
    goals = {'drone_0': centre.tolist()}
    n_movers = max(n_agents - 1, 1)
    for i in range(1, n_agents):
        ang = np.radians(base_angle + (i - 1) * 180.0 / n_movers + angle_jitter[i])
        u = np.array([np.cos(ang), np.sin(ang), 0.0])
        r = R * radius_jitter[i]
        spawns[f'drone_{i}'] = (centre - r * u).tolist()
        goals[f'drone_{i}'] = (centre + r * u).tolist()
    return spawns, goals


GENERATORS = {'headon': headon, 'crossing': crossing, 'swap': swap, 'flythrough': flythrough}
