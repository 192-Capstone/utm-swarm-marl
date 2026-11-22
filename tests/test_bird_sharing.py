"""
test_bird_sharing.py
--------------------
Stage 3a++: shared bird tracks (detector.share), the Kalman tracker (detector.tracker), the guard's brake
candidates (guard.brake) and the assumption behind eval_suite --episode-offset.

  A. Config: new options off by default; bad settings refused.
  B. Sharing: a drone that cannot see a bird gets the track from a neighbour after the comms latency, extrapolated;
     comms range is respected; duplicates of one bird are merged, different birds are kept; off = unchanged.
  C. Kalman tracker: estimates a bird's position and velocity, coasts and drops like the alpha-beta tracker.
  D. Brake: hovering is chosen when sideways and vertical evasion are blocked; off = unchanged; bad scales refused.
  E. Episode offset: flying an episode draws nothing from the global random stream, so skipping the flights of the
     first N episodes (only generating them) leaves the following episodes exactly as in the longer run.

Run: python -m pytest tests/test_bird_sharing.py -q
"""

import os
import sys

import numpy as np
import pytest
import yaml

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, ROOT)

from envs import birds as B                                         # noqa: E402
from envs import bird_detector as BD                                # noqa: E402
from modules import bird_guard as G                                 # noqa: E402
from tests.test_birds import make_env, DT, STILL                    # noqa: E402
from tests.test_bird_guard import Const, FakeEnv, guard_cfg         # noqa: E402

SURE = dict(enabled=True, p_near=1.0, p_far=1.0, sigma_m=0.0, latency_steps=0, confirm_hits=1, coast_s=0.5)


def det(n=1, b=1, headings=None, seed=0, **kw):
    cfg = BD.validate({**SURE, **kw})
    return BD.BirdDetector(cfg, n, b, DT, np.random.default_rng(seed), headings=headings)


def step(d, k, drones, birds, dvel=None, bvel=None):
    n, b = len(drones), len(birds)
    return d.step(k, drones, dvel if dvel is not None else [[0.0, 0, 0]] * n, birds, bvel if bvel is not None else [[0.0, 0, 0]] * b)


# ---------------------------------------------------------------- A -------
def test_config_defaults_and_errors():
    cfg = BD.validate({'enabled': True})
    assert cfg['tracker'] == 'alphabeta' and cfg['share'] is False and cfg['share_range_m'] == 30.0
    assert cfg['share_latency_steps'] == 3 and cfg['share_gate_m'] == 1.0
    assert BD.validate({'enabled': True, 'tracker': 'kalman', 'share': True})['share'] is True
    for bad in ({'tracker': 'particle'}, {'kalman_q': 0}, {'share_range_m': 0}, {'share_gate_m': -1}, {'share_latency_steps': -1}):
        with pytest.raises(ValueError):
            BD.validate({'enabled': True, **bad})


# ---------------------------------------------------------------- B -------
A_POS, B_POS = [0.0, 0.0, 2.0], [0.0, 5.0, 2.0]
HEAD = [[1.0, 0.0], [0.0, 1.0]]                          # A looks along +x, B along +y (blind to a bird at +x)


def test_a_blind_drone_gets_the_neighbours_track_after_the_comms_latency():
    d = det(n=2, b=1, headings=HEAD, fov_deg=60.0, share=True, share_latency_steps=3)
    bird = [[4.0, 0.0, 2.0]]
    seen_b = []
    for k in range(8):
        known, pos, vel = step(d, k, [A_POS, B_POS], bird)
        assert known[0].any()                                        # A sees it from the start
        seen_b.append(bool(known[1].any()))
    assert seen_b == [False, False, False, True, True, True, True, True]          # B: after the 3-step latency
    assert np.allclose(pos[1][known[1]][0], [4.0, 0.0, 2.0], atol=1e-9)
    # without sharing B never gets it
    d = det(n=2, b=1, headings=HEAD, fov_deg=60.0)
    for k in range(8):
        known, _, _ = step(d, k, [A_POS, B_POS], bird)
    assert known[0].any() and not known[1].any()


def test_shared_tracks_are_extrapolated_over_the_latency_and_age():
    d = det(n=2, b=1, headings=HEAD, fov_deg=60.0, share=True, share_latency_steps=5, coast_s=1.0)
    err = []
    for k in range(60):
        bird = [[4.0, 0.0 + 2.0 * k * DT, 2.0]]                       # flying +y at 2 m/s, then out of B's reach? no: in range
        known, pos, vel = step(d, k, [A_POS, [0.0, -4.0, 2.0]], bird, bvel=[[0.0, 2.0, 0.0]])
        if k > 20 and known[1].any():
            err.append(np.linalg.norm(pos[1][known[1]][0] - np.array(bird[0])))
    assert len(err) > 30 and max(err) < 0.2                          # moved on by the bird's velocity, not stale by 5 steps


def test_comms_range_is_respected():
    d = det(n=2, b=1, headings=HEAD, fov_deg=60.0, share=True, share_range_m=30.0)
    for k in range(8):
        known, _, _ = step(d, k, [A_POS, [0.0, 40.0, 2.0]], [[4.0, 0.0, 2.0]])
    assert known[0].any() and not known[1].any()                      # 40 m away: out of range
    d = det(n=2, b=1, headings=HEAD, fov_deg=60.0, share=True, share_range_m=45.0)
    for k in range(8):
        known, _, _ = step(d, k, [A_POS, [0.0, 40.0, 2.0]], [[4.0, 0.0, 2.0]])
    assert known[1].any()


def test_duplicates_are_merged_and_different_birds_are_kept():
    # both drones see the same bird: one track each, not two
    d = det(n=2, b=1, headings=[[1.0, 0.0], [1.0, 0.0]], fov_deg=360.0, share=True)
    for k in range(8):
        known, pos, _ = step(d, k, [A_POS, B_POS], [[4.0, 2.0, 2.0]])
    assert known[0].sum() == 1 and known[1].sum() == 1
    # two birds 6 m apart: both tracks kept (more than the 1 m gate apart)
    d = det(n=2, b=2, headings=[[1.0, 0.0], [1.0, 0.0]], fov_deg=360.0, share=True)
    for k in range(8):
        known, pos, _ = step(d, k, [A_POS, B_POS], [[4.0, 2.0, 2.0], [4.0, 8.0, 2.0]])
    assert known[0].sum() == 2 and known[1].sum() == 2


def test_the_gate_is_what_decides_duplicates_and_own_tracks_win_ties():
    # one bird, A's and B's estimates differ by noise: with a wide gate they merge, with a tiny one they do not
    kw = dict(n=2, b=1, headings=[[1.0, 0.0], [1.0, 0.0]], fov_deg=360.0, share=True, sigma_m=0.3, p_near=1.0, p_far=1.0)
    wide, tiny = det(share_gate_m=2.0, **kw), det(share_gate_m=1e-6, **kw)
    for k in range(40):
        kw_, _, _ = step(wide, k, [A_POS, B_POS], [[4.0, 2.0, 2.0]])
        kt, _, _ = step(tiny, k, [A_POS, B_POS], [[4.0, 2.0, 2.0]])
    assert kw_[0].sum() == 1 and kt[0].sum() > 1


def test_sharing_off_is_unchanged():
    a, b = det(n=2, b=1, headings=HEAD, fov_deg=120.0), det(n=2, b=1, headings=HEAD, fov_deg=120.0, share=False)
    for k in range(20):
        ra = step(a, k, [A_POS, B_POS], [[4.0, 2.0, 2.0]])
        rb = step(b, k, [A_POS, B_POS], [[4.0, 2.0, 2.0]])
        assert all(np.array_equal(x, y) for x, y in zip(ra, rb)) and ra[0].shape == (2, 1)       # (drone, bird) layout


def test_sharing_in_the_env_and_the_tracked_statistic():
    cfg = {'enabled': True, 'count': 1, 'detector': {**SURE, 'fov_deg': 60.0, 'share': True, 'birds_in_slots': False}}
    env, _ = make_env(n=2, noise=False, birds=cfg)
    still = B.BirdField([B.Bird.straight([4.0, 0.0, 2.0], [0, 0, 0], 0.12, DT)], DT)
    env.start_episode({'drone_0': [0.0, 0.0, 2.0], 'drone_1': [0.0, 5.0, 2.0]},
                      {'drone_0': [1.0, 0.0, 2.0], 'drone_1': [0.0, 6.0, 2.0]}, obstacles=[], birds=still)
    tracked = []
    for _ in range(8):
        _, _, _, info = env.step({'drone_0': [0.2, 0, 0, 0], 'drone_1': [0, 0.2, 0, 0]})      # drone_1 looks along +y
        tracked.append(info['bird_tracked'])
    assert tracked[-1][0][0] and tracked[-1][1][0]                                          # both have it, one via sharing
    assert env._bird_tracks[0].shape[1] == 2                                                # merged layout: n_drones * n_birds
    env.close()


# ---------------------------------------------------------------- C -------
def run_tracker(tracker, sigma=0.1, steps=110, seed=3, **kw):
    d = det(sigma_m=sigma, p_near=0.9, p_far=0.9, latency_steps=3, coast_s=0.5, confirm_hits=2, seed=seed,
            tracker=tracker, headings=[[1.0, 0.0]], **kw)
    ep, ev = [], []
    for k in range(steps):
        bird = np.array([3.0 + 1.5 * k * DT, 1.0, 2.0])
        known, pos, vel = step(d, k, [[0, 0, 2.0]], [bird], bvel=[[1.5, 0, 0]])
        if k > 30 and known[0, 0]:
            ep.append(np.linalg.norm(pos[0, 0] - bird))
            ev.append(np.linalg.norm(vel[0, 0] - [1.5, 0, 0]))
    return np.mean(ep), np.mean(ev), len(ep)


def test_kalman_tracker_estimates_position_and_velocity():
    ep, ev, n = run_tracker('kalman')
    assert n > 60 and ep < 0.2 and ev < 0.6


def test_kalman_coasts_then_drops_and_restarts():
    d = det(coast_s=0.5, confirm_hits=1, tracker='kalman', headings=[[1.0, 0.0]])
    for k in range(20):
        assert step(d, k, [[0, 0, 2.0]], [[4.0, 0, 2.0]])[0][0, 0]
    coast = int(round(0.5 / DT))
    flags = [bool(step(d, 20 + i, [[0, 0, 2.0]], [[20.0, 0, 2.0]])[0][0, 0]) for i in range(coast + 5)]
    assert all(flags[:coast]) and not any(flags[coast + 1:])
    assert step(d, 60, [[0, 0, 2.0]], [[4.0, 0, 2.0]])[0][0, 0]


def test_both_trackers_are_deterministic_per_seed():
    for tr in ('alphabeta', 'kalman'):
        assert run_tracker(tr, seed=5) == run_tracker(tr, seed=5)


# ---------------------------------------------------------------- D -------
# a bird crossing in front of the drone (x = 2, from y = -3 at 1.5 m/s): hovering lets it pass (miss 1.4 m), nothing
# else is allowed: the altitude band is closed in both directions and the ring sees walls at 1.05 m all around
CROSSING = ([2.0, -3.0, 0.86], [0.0, 1.5, 0.0])


def brake_run(**guard):
    ring = np.full(32, 1.05)                                   # far enough for no push-away (obs_close 1.0), near enough
    cfg = guard_cfg(climb_m=0.05, z_floor_m=0.86, **guard)     # to limit every sideways evasion to a 1.05 m score
    from modules.bird_guard import BirdGuardPolicy
    pol = BirdGuardPolicy(Const((1.5, 0, 0, 0)), cfg)
    env = FakeEnv([0, 0, 0.86], [1.5, 0, 0], [CROSSING], goal_z=0.8, ring=ring)   # band [0.86, 0.85]: closed
    return pol, pol.act(None, env, 1)['drone_0']


def test_brake_hovers_when_nothing_else_is_open():
    pol, no_brake = brake_run()
    assert pol.active[0] and np.linalg.norm(no_brake[:3]) > 0.3          # without brake it has to keep moving
    pol, hover = brake_run(brake=[0.0, 0.5])
    assert pol.active[0] and np.allclose(hover[:3], 0.0, atol=1e-9)       # with brake: hover (scale 0)


def test_brake_off_is_the_default_and_changes_nothing():
    _, a = brake_run()
    _, b = brake_run(brake=[])
    assert a == b
    assert G.validate({'enabled': True})['brake'] == []
    with pytest.raises(ValueError):
        G.validate({'enabled': True, 'brake': [1.0]})
    with pytest.raises(ValueError):
        G.validate({'enabled': True, 'brake': [-0.1]})


def test_brake_does_not_replace_a_better_evasion():
    # open sky: a sideways dodge scores far higher than hovering, so the brake scales are not chosen
    pol = G.BirdGuardPolicy(Const((1.5, 0, 0, 0)), guard_cfg(brake=[0.0, 0.5]))
    env = FakeEnv([0, 0, 2.0], [1.5, 0, 0], [([4.0, 0.0, 2.0], [-1.0, 0.0, 0.0])])      # head-on: hovering does not help
    cmd = pol.act(None, env, 1)['drone_0']
    assert np.linalg.norm(cmd[:3]) > 0.5 and (abs(cmd[1]) > 0.3 or abs(cmd[2]) > 0.3)


# ---------------------------------------------------------------- E -------
def test_flying_an_episode_draws_nothing_from_the_global_random_stream():
    """eval_suite --episode-offset generates the skipped episodes without flying them; that is exact only if flying
    consumes no global random numbers (deterministic policy)."""
    cfg = {'enabled': True, 'count': 2, 'spawn': 'random',
           'detector': {'enabled': True, 'share': True, 'tracker': 'kalman'}, 'guard': {'enabled': True, 'brake': [0.0]}}

    def second_episode(fly):
        env, _ = make_env(n=2, birds=cfg)
        np.random.seed(11)
        sp = {'drone_0': [0.0, 0.0, 2.0], 'drone_1': [0.0, 3.0, 2.0]}
        gl = {'drone_0': [6.0, 0.0, 2.0], 'drone_1': [6.0, 3.0, 2.0]}
        env.start_episode(sp, gl, obstacles=[])
        if fly:
            for _ in range(60):
                env.step({'drone_0': [1.0, 0, 0, 0], 'drone_1': [1.0, 0, 0, 0]})
        env.start_episode(sp, gl, obstacles=[])
        out = (np.array(env.adapter._alpha), np.array(env.adapter._delay), env.birds.positions(40), np.random.rand())
        env.close()
        return out
    a, b = second_episode(True), second_episode(False)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
