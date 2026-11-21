"""
test_bird_guard.py
------------------
Stage 3a+: the bird guard (modules/bird_guard.py), the shared CPA function, birds_in_slots and the random
crossing spawn mode.

  A. cpa() is the teacher's arithmetic, bit for bit (float32 and float64); known geometry.
  B. Config: off by default; the guard needs the detector; bad settings refused.
  C. Threat: a bird on a collision course is a threat, a bird passing wide is not (output untouched).
  D. Evasion raises the predicted miss, stays inside the altitude band (layered and unlayered, floor),
     avoids a wall seen in the LiDAR ring and drones nearby (the drones' shared states, not the slots).
  E. Hysteresis: stays active while a bird is closing with a miss under the release threshold, releases
     only after release_steps once the bird is opening.
  F. End to end: the guard uses tracks, never truth (an undetected bird triggers nothing and hits); with the
     oracle detector it keeps a head-on bird away; no new crashes.
  G. birds_in_slots false leaves the neighbour slots exactly as without birds.
  H. Random crossing spawn: deterministic per seed, inside the arena, at random times.

Run: python -m pytest tests/test_bird_guard.py -q
"""

import os
import sys
import types

import numpy as np
import pytest
import yaml

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, ROOT)

from baselines.repulsion_controller import cpa                      # noqa: E402
from envs import birds as B                                         # noqa: E402
from envs import lidar as lidar_mod                                 # noqa: E402
from modules import bird_guard as G                                 # noqa: E402
from tests.test_birds import make_env, DT, SP, GL, STILL            # noqa: E402

CONFIG = os.path.join(ROOT, 'configs', 'config_stage2_round7_eval_goalhold.yaml')
TAU = 0.4


def guard_cfg(**guard):
    cfg = yaml.safe_load(open(CONFIG, encoding='utf-8'))
    cfg['environment']['birds'] = {'enabled': True, 'count': 1, 'detector': {'enabled': True, 'mode': 'oracle'},
                                   'guard': {'enabled': True, **guard}}
    return cfg


class Const:
    """An inner policy that always commands `cmd` (the policy + goal-hold output the guard sees)."""
    def __init__(self, cmd):
        self.cmd = cmd

    def act(self, obs, env, n):
        return {f'drone_{i}': list(self.cmd) for i in range(n)}


class FakeEnv:
    """Just what the guard reads: the noisy state, the tracks, the ring, the target height."""
    lidar_range = 6.0

    def __init__(self, pos, vel, birds, goal_z=2.0, ring=None, episode=1):
        pos, vel = np.atleast_2d(np.asarray(pos, float)), np.atleast_2d(np.asarray(vel, float))
        n = len(pos)
        self.episode_index = episode
        self.last_obs_raw = {f'drone_{i}': {'position': pos[i].tolist(), 'velocity': vel[i].tolist(),
                                            'lidar_ring': (np.full(32, 6.0) if ring is None else ring).tolist()}
                             for i in range(n)}
        self.set_birds(birds, n)
        self.adapter = types.SimpleNamespace(goals={f'drone_{i}': np.array([50.0, 0.0, goal_z]) for i in range(n)},
                                             _heading={f'drone_{i}': 0.0 for i in range(n)})

    def set_birds(self, birds, n=1):
        """birds: [(pos, vel)] tracked by every drone."""
        b = len(birds)
        self._bird_tracks = (np.ones((n, b), dtype=bool), np.array([[p for p, _ in birds]] * n, float).reshape(n, b, 3),
                             np.array([[v for _, v in birds]] * n, float).reshape(n, b, 3))


def run_guard(pos, vel, birds, base=(1.5, 0.0, 0.0, 0.0), n=None, **kw):
    ring = kw.pop('ring', None)
    goal_z = kw.pop('goal_z', 2.0)
    guard = G.BirdGuardPolicy(Const(base), guard_cfg(**kw))
    env = FakeEnv(pos, vel, birds, goal_z=goal_z, ring=ring)
    out = guard.act(None, env, len(np.atleast_2d(pos)))
    return guard, env, out


def miss_of(cmd, pos, vel, bird):
    """The predicted miss of a drone flying at `cmd` against a bird (same look-ahead as the guard)."""
    p = (np.asarray(pos, float) + np.asarray(vel, float) * TAU) - (np.asarray(bird[0], float) + np.asarray(bird[1], float) * TAU)
    return float(np.linalg.norm(cpa(p, np.asarray(cmd[:3], float) - np.asarray(bird[1], float), 3.0)[1]))


# ---------------------------------------------------------------- A -------
def test_cpa_is_the_teachers_arithmetic_bit_for_bit():
    rng = np.random.default_rng(1)
    for dt in (np.float32, np.float64):
        for _ in range(500):
            p, rv = rng.uniform(-5, 5, 3).astype(dt), rng.uniform(-3, 3, 3).astype(dt)
            speed2 = float(rv @ rv)                                   # the expression the teacher always used
            t_old = float(np.clip(-(p @ rv) / speed2, 0.0, 3.0)) if speed2 > 1e-6 else 0.0
            c_old = p + rv * t_old
            t, c = cpa(p, rv, 3.0)
            assert t == t_old and c.dtype == c_old.dtype and np.array_equal(c, c_old)


def test_cpa_known_geometry():
    t, c = cpa(np.array([-4.0, 0, 0]), np.array([2.0, 0, 0]), 3.0)       # head on: closest in 2 s, miss 0
    assert abs(t - 2.0) < 1e-9 and np.allclose(c, 0)
    t, c = cpa(np.array([-4.0, 1.0, 0]), np.array([2.0, 0, 0]), 3.0)     # passes 1 m to the side
    assert abs(t - 2.0) < 1e-9 and abs(np.linalg.norm(c) - 1.0) < 1e-9
    t, c = cpa(np.array([4.0, 0, 0]), np.array([2.0, 0, 0]), 3.0)        # moving apart: closest now
    assert t == 0.0 and np.allclose(c, [4.0, 0, 0])
    t, c = cpa(np.array([1.0, 0, 0]), np.zeros(3), 3.0)                  # same velocity: never closer
    assert t == 0.0


# ---------------------------------------------------------------- B -------
def test_config():
    assert G.validate(None) is None and G.validate({'enabled': False}) is None
    assert not G.guard_enabled(yaml.safe_load(open(CONFIG, encoding='utf-8')))        # off by default
    with pytest.raises(ValueError):
        B.validate_block({'enabled': True, 'count': 1, 'guard': {'enabled': True}})   # no detector
    for bad in ({'enabled': True, 'miss_m': 0}, {'enabled': True, 'release_m': 0.5}, {'enabled': True, 'bogus': 1},
                {'enabled': True, 'layer_frac': 0.9}):
        with pytest.raises(ValueError):
            G.validate(bad)
    assert G.band_half_width(G.validate({'enabled': True})) == 1.5
    assert abs(G.band_half_width(G.validate({'enabled': True, 'layer_spacing_m': 2.0})) - 0.9) < 1e-12
    assert G.band_half_width(G.validate({'enabled': True, 'layer_spacing_m': 10.0})) == 1.5    # min(1.5, 4.5)


# ---------------------------------------------------------------- C -------
HEAD_ON = ([4.0, 0.0, 2.0], [-1.0, 0.0, 0.0])                            # a bird 4 m ahead, flying at the drone


def test_threat_detected_and_evasion_raises_the_miss():
    guard, env, out = run_guard([0, 0, 2.0], [1.5, 0, 0], [HEAD_ON])
    assert guard.active[0] and guard.guard_stats['activations'] == 1
    cmd = out['drone_0']
    base_miss = miss_of([1.5, 0, 0], [0, 0, 2.0], [1.5, 0, 0], HEAD_ON)
    new_miss = miss_of(cmd, [0, 0, 2.0], [1.5, 0, 0], HEAD_ON)
    assert base_miss < 0.2 and new_miss > 1.0 and new_miss > base_miss + 0.8
    assert max(abs(x) for x in cmd[:3]) <= 2.0 + 1e-9                    # capped at 2 m/s per axis


def test_a_bird_passing_wide_is_no_threat_and_the_command_is_untouched():
    guard, env, out = run_guard([0, 0, 2.0], [1.5, 0, 0], [([4.0, 3.5, 2.0], [-1.0, 0.0, 0.0])])
    assert not guard.active[0] and out['drone_0'] == [1.5, 0.0, 0.0, 0.0]


def test_a_receding_bird_is_no_threat():
    guard, env, out = run_guard([0, 0, 2.0], [1.5, 0, 0], [([-2.0, 0.0, 2.0], [-1.0, 0.0, 0.0])])
    assert not guard.active[0] and out['drone_0'] == [1.5, 0.0, 0.0, 0.0]


def test_no_tracks_no_action():
    guard = G.BirdGuardPolicy(Const((1.5, 0, 0, 0)), guard_cfg())
    env = FakeEnv([0, 0, 2.0], [1.5, 0, 0], [])
    env._bird_tracks = None                                              # no detector this episode
    assert guard.act(None, env, 1)['drone_0'] == [1.5, 0, 0, 0] and not guard.active[0]


# ---------------------------------------------------------------- D -------
def test_altitude_band_unlayered_layered_and_floor():
    # a bird straight at the drone; sideways is free, so make the vertical choice the only good one by walling both sides
    ring = np.full(32, 6.0)
    ring[[4, 5, 6, 7, 8, 9, 10, 11, 12]] = 0.5                           # left side (ray 8 = +90 deg) blocked
    ring[[20, 21, 22, 23, 24, 25, 26, 27, 28]] = 0.5                     # right side (ray 24 = -90 deg) blocked
    # at the top of the band: no upward evasion
    _, _, out = run_guard([0, 0, 2.0 + 1.5], [1.5, 0, 0], [([4.0, 0, 3.5], [-1.0, 0, 0])], ring=ring, goal_z=2.0)
    assert out['drone_0'][2] <= 1e-9
    # at the bottom: no downward evasion
    _, _, out = run_guard([0, 0, 2.0 - 1.5], [1.5, 0, 0], [([4.0, 0, 0.5], [-1.0, 0, 0])], ring=ring, goal_z=2.0)
    assert out['drone_0'][2] >= -1e-9
    # layered scenario: half-width 0.9, so 1.0 m above the target is already outside it
    _, _, out = run_guard([0, 0, 3.0], [1.5, 0, 0], [([4.0, 0, 3.0], [-1.0, 0, 0])], ring=ring, goal_z=2.0, layer_spacing_m=2.0)
    assert out['drone_0'][2] <= 1e-9
    # the unlayered band still lets the same drone climb (it is 1.0 m above the target, 1.5 m half-width)
    _, _, out = run_guard([0, 0, 3.0], [1.5, 0, 0], [([4.0, 0, 3.0], [-1.0, 0, 0])], ring=ring, goal_z=2.0)
    assert out['drone_0'][2] > 0.3
    # the floor: a drone on the floor never evades downwards
    _, _, out = run_guard([0, 0, 0.8], [1.5, 0, 0], [([4.0, 0, 0.8], [-1.0, 0, 0])], ring=ring, goal_z=1.0)
    assert out['drone_0'][2] >= -1e-9


def test_a_wall_in_the_ring_is_not_evaded_into():
    ring = np.full(32, 6.0)
    ring[22:27] = 0.6                                                    # a wall on the right (-y) at 0.6 m
    guard, env, out = run_guard([0, 0, 2.0], [1.5, 0, 0], [HEAD_ON], ring=ring)
    cmd = out['drone_0']
    assert cmd[1] > -0.3                                                 # not into the wall (it is pushed away from it)
    assert miss_of(cmd, [0, 0, 2.0], [1.5, 0, 0], HEAD_ON) > 0.9         # and still avoids the bird (left or vertical)


def test_a_drone_beside_is_not_evaded_into_and_slots_are_not_read():
    # the right-hand side (-y) is where a plain sideways dodge would go; a drone flies there
    pos = np.array([[0.0, 0.0, 2.0], [0.0, -1.1, 2.0]])
    vel = np.array([[1.5, 0, 0], [1.5, 0, 0]])
    guard = G.BirdGuardPolicy(Const((1.5, 0, 0, 0)), guard_cfg())
    env = FakeEnv(pos, vel, [HEAD_ON], episode=7)
    out = guard.act(None, env, 2)                                        # obs=None: the guard never reads the slots
    cmd = out['drone_0']
    assert guard.active[0] and cmd[1] > 0.3                              # dodges to the free side
    assert miss_of(cmd, pos[0], vel[0], HEAD_ON) > 0.9


# ---------------------------------------------------------------- E -------
def test_hysteresis():
    guard = G.BirdGuardPolicy(Const((1.5, 0, 0, 0)), guard_cfg())
    env = FakeEnv([0, 0, 2.0], [1.5, 0, 0], [HEAD_ON])
    guard.act(None, env, 1)
    assert guard.active[0]
    # the bird now passes at a predicted 1.2 m: not a threat (>= 1.0) but still closing under the release threshold
    env.set_birds([([4.0, 1.2, 2.0], [-1.0, 0.0, 0.0])])
    for _ in range(20):
        guard.act(None, env, 1)
        assert guard.active[0]                                           # stays active
    # the bird has passed (opening): released only after release_steps (5) quiet steps
    env.set_birds([([-3.0, 0.0, 2.0], [-1.0, 0.0, 0.0])])
    states = []
    for _ in range(8):
        guard.act(None, env, 1)
        states.append(bool(guard.active[0]))
    assert states == [True, True, True, True, False, False, False, False]
    # and a threat that returns re-activates it
    env.set_birds([HEAD_ON])
    guard.act(None, env, 1)
    assert guard.active[0] and guard.guard_stats['activations'] == 2


def test_a_dropped_track_while_latched_does_nothing_and_releases():
    guard = G.BirdGuardPolicy(Const((1.5, 0, 0, 0)), guard_cfg())
    env = FakeEnv([0, 0, 2.0], [1.5, 0, 0], [HEAD_ON])
    guard.act(None, env, 1)
    assert guard.active[0]
    env.set_birds([])                                            # the detector lost the bird (coasting ended)
    states = []
    for _ in range(7):
        out = guard.act(None, env, 1)
        assert out['drone_0'] == [1.5, 0.0, 0.0, 0.0]            # no evasion without a track (and no crash)
        states.append(bool(guard.active[0]))
    assert states == [True, True, True, True, False, False, False]


def test_a_new_episode_clears_the_state():
    guard = G.BirdGuardPolicy(Const((1.5, 0, 0, 0)), guard_cfg())
    guard.act(None, FakeEnv([0, 0, 2.0], [1.5, 0, 0], [HEAD_ON], episode=1), 1)
    assert guard.active[0]
    guard.act(None, FakeEnv([0, 0, 2.0], [1.5, 0, 0], [([4.0, 3.5, 2.0], [-1, 0, 0])], episode=2), 1)
    assert not guard.active[0]


# ---------------------------------------------------------------- F -------
def cruise_env(detector, guard=True, birds_in_slots=False, layout=None):
    cfg_b = {'enabled': True, 'count': 1, 'detector': {**detector, 'enabled': True, 'birds_in_slots': birds_in_slots},
             **({'guard': {'enabled': True}} if guard else {})}
    env, full = make_env(noise=False, birds=cfg_b)
    return env, full


class Cruise:
    """A stand-in for policy + goal-hold: fly to the goal at 1.5 m/s (the guard logic is what is tested)."""
    def act(self, obs, env, n):
        out = {}
        for i in range(n):
            err = env.goals[i] - env.adapter.get_positions()[i]
            d = np.linalg.norm(err)
            out[f'drone_{i}'] = (list(err / d * min(1.5, 1.2 * d)) if d > 1e-6 else [0, 0, 0]) + [0.0]
        return out


def head_on_run(detector, guard):
    env, full = cruise_env(detector, guard=guard)
    pol = G.BirdGuardPolicy(Cruise(), full) if guard else Cruise()
    t = 12.0 / (1.5 + 2.0)
    bird = B.BirdField([B.Bird.straight([1.5 * t + 2.0 * t, 0.0, 2.0], [-2.0, 0.0, 0.0], 0.12, DT)], DT)
    obs = env.start_episode({'drone_0': [0.0, 0.0, 2.0]}, {'drone_0': [12.0, 0.0, 2.0]}, obstacles=[], birds=bird)
    dmin, active = 9e9, False
    for _ in range(600):
        obs, _, done, info = env.step(pol.act(obs, env, 1))
        dmin = min(dmin, float(info['bird_distance'][0][0]))
        active |= bool(guard and pol.active[0])
        if done:
            break
    env.close()
    return dmin, info['reason'], active


def test_the_guard_keeps_a_head_on_bird_away_and_uses_tracks_only():
    d0, reason0, _ = head_on_run({'mode': 'oracle'}, guard=False)
    assert reason0 == 'bird' and d0 < 0.3                                # without a guard the cruise hits the bird
    d1, reason1, active = head_on_run({'mode': 'oracle'}, guard=True)
    assert active and reason1 != 'bird' and d1 > 0.6                     # with the oracle's tracks it does not
    # a detector that never detects (p = 0): the guard has no tracks, never activates, and the bird hits
    d2, reason2, active2 = head_on_run({'mode': 'realistic', 'p_near': 0.0, 'p_far': 0.0}, guard=True)
    assert not active2 and reason2 == 'bird'


# ---------------------------------------------------------------- G -------
def test_birds_in_slots_false_leaves_the_slots_alone():
    def run(detector):
        cfg = {'enabled': True, 'count': 1, **({'detector': detector} if detector else {})}
        env, _ = make_env(n=2, birds=cfg)
        np.random.seed(7)
        bird = B.BirdField([B.Bird.straight([3.0, 0.5, 2.0], [-1.0, 0, 0], 0.12, DT)], DT)      # close to drone_0
        env.start_episode({'drone_0': [0, 0, 2.0], 'drone_1': [0, 2.0, 2.0]},
                          {'drone_0': [4, 0, 2.0], 'drone_1': [4, 2.0, 2.0]}, obstacles=[], birds=bird)
        out = []
        for _ in range(20):
            obs, _, _, info = env.step({'drone_0': [1, 0, 0, 0], 'drone_1': [1, 0, 0, 0]})
            out.append(np.concatenate([obs['neighbor_states'].numpy().ravel(), obs['neighbor_mask'].numpy().ravel()]))
        env.close()
        return np.array(out), info
    plain, _ = run(None)
    off, info_off = run({'enabled': True, 'mode': 'oracle', 'birds_in_slots': False})
    on, _ = run({'enabled': True, 'mode': 'oracle', 'birds_in_slots': True})
    assert np.array_equal(plain, off) and any(any(r) for r in info_off['bird_tracked'])    # tracked, but not in the slots
    assert not np.array_equal(plain, on)                                                    # in the slots they do show


# ---------------------------------------------------------------- H -------
def test_random_crossing_spawn():
    paths = [np.array([[0.0, 0.0, 2.0], [8.0, 0.0, 2.0]]), np.array([[0.0, 4.0, 2.0], [8.0, 4.0, 2.0]])]
    cfg = B.validate_block({'enabled': True, 'count': 4, 'seed': 3, 'spawn': 'random'})
    a, b = B.BirdField.generate(cfg, paths, DT), B.BirdField.generate(cfg, paths, DT)
    for k in (0, 50, 400):
        assert np.array_equal(a.positions(k), b.positions(k))              # deterministic per seed
    other = B.BirdField.generate(B.validate_block({'enabled': True, 'count': 4, 'seed': 4, 'spawn': 'random'}), paths, DT)
    assert not np.allclose(a.positions(100), other.positions(100))
    t_end = 8.0 / 2.0 + 1.0
    for bird in a.birds:
        assert bird.mode == 'cross' and 1.0 <= bird.t_aim <= t_end + 0.1   # random times during the flight
        assert -2.0 - 1e-6 <= bird.aim[0] <= 10.0 + 1e-6 and -2.0 - 1e-6 <= bird.aim[1] <= 6.0 + 1e-6   # inside the arena
    timed = B.BirdField.generate(B.validate_block({'enabled': True, 'count': 4, 'seed': 3}), paths, DT)
    assert not np.allclose(timed.positions(100), a.positions(100))         # a different mode
    with pytest.raises(ValueError):
        B.validate_block({'spawn': 'sideways'})
