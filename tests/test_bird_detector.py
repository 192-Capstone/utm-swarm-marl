"""
test_bird_detector.py
---------------------
Stage 3a: the onboard bird detector + tracker (envs/bird_detector.py) and its feed into the policy's
neighbour slots.

  A. Config: off by default, bad settings refused.
  B. Detection respects range, field of view and probability; the oracle sees everything in range.
  C. Latency, coasting and dropping of tracks; the tracker's position / velocity estimate.
  D. Occlusion: a bird behind a building is not detected, the same bird in the open is (realistic only).
  E. Tracked birds appear in the neighbour slots (units, mixing with drones); detector off = unchanged.
  F. The goal-hold guard reacts to a tracked bird.

Run: python -m pytest tests/test_bird_detector.py -q
"""

import os
import sys

import numpy as np
import pytest

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, ROOT)

from envs import birds as B                                         # noqa: E402
from envs import bird_detector as BD                                # noqa: E402
from modules.goal_hold import GoalHoldPolicy                        # noqa: E402
from tests.test_birds import make_env, DT, SP, GL, STILL, Zero      # noqa: E402

SURE = dict(enabled=True, p_near=1.0, p_far=1.0, sigma_m=0.0, latency_steps=0, confirm_hits=1, coast_s=0.0)


def det(rng_seed=0, n=1, b=1, headings=None, blocked=None, **kw):
    cfg = BD.validate({**SURE, **kw})
    return BD.BirdDetector(cfg, n, b, DT, np.random.default_rng(rng_seed), headings=headings, blocked=blocked)


def look(d, k, drone, bird, vel=(0, 0, 0), bvel=(0, 0, 0)):
    return d.step(k, [drone], [vel], [bird], [bvel])


# ---------------------------------------------------------------- A -------
def test_config_off_by_default_and_bad_settings():
    assert BD.validate(None) is None and BD.validate({'enabled': False}) is None
    assert B.config({'environment': {'birds': {'enabled': True, 'count': 1}}})['detector'] is None
    assert B.config({'environment': {'birds': {'enabled': True, 'count': 1,
                                               'detector': {'enabled': True}}}})['detector']['range_m'] == 8.0
    for bad in ({'enabled': True, 'mode': 'x'}, {'enabled': True, 'range_m': 0}, {'enabled': True, 'fov_deg': 400},
                {'enabled': True, 'p_near': 0.4, 'p_far': 0.6}, {'enabled': True, 'bogus': 1}):
        with pytest.raises(ValueError):
            BD.validate(bad)


# ---------------------------------------------------------------- B -------
def test_range_and_fov():
    d = det(range_m=8.0, fov_deg=120.0, headings=[[1.0, 0.0]])
    assert look(d, 0, [0, 0, 2], [7.0, 0, 2])[0][0, 0]                  # ahead, in range
    d = det(range_m=8.0, fov_deg=120.0, headings=[[1.0, 0.0]])
    assert not look(d, 0, [0, 0, 2], [9.0, 0, 2])[0][0, 0]              # out of range
    d = det(range_m=8.0, fov_deg=120.0, headings=[[1.0, 0.0]])
    assert not look(d, 0, [0, 0, 2], [-5.0, 0, 2])[0][0, 0]             # behind
    d = det(range_m=8.0, fov_deg=120.0, headings=[[1.0, 0.0]])
    assert look(d, 0, [0, 0, 2], [3.0, 4.0, 2])[0][0, 0]                # 53 deg off the heading: inside +-60
    d = det(range_m=8.0, fov_deg=120.0, headings=[[1.0, 0.0]])
    assert not look(d, 0, [0, 0, 2], [2.0, 4.0, 2])[0][0, 0]            # 63 deg: outside
    d = det(range_m=8.0, fov_deg=360.0, headings=[[1.0, 0.0]])
    assert look(d, 0, [0, 0, 2], [-5.0, 0, 2])[0][0, 0]                 # a 360 degree rig sees behind


def test_fov_follows_the_velocity_direction():
    d = det(fov_deg=90.0, headings=[[1.0, 0.0]])
    assert not look(d, 0, [0, 0, 2], [0.0, 5.0, 2], vel=(0, 1.0, 0))[0][0, 0] is False   # flying +y: sees +y
    d = det(fov_deg=90.0, headings=[[1.0, 0.0]])
    assert not look(d, 0, [0, 0, 2], [5.0, 0.0, 2], vel=(0, 1.0, 0))[0][0, 0]           # now blind to +x


def test_detection_probability():
    d = det(p_near=0.5, p_far=0.5, headings=[[1.0, 0.0]])
    hits = sum(bool(look(d, k, [0, 0, 2], [4.0, 0, 2])[0][0, 0]) for k in range(3000))
    assert 0.45 < hits / 3000 < 0.55
    d = det(p_near=0.9, p_far=0.5, range_m=8.0, headings=[[1.0, 0.0]])
    near = sum(bool(look(d, k, [0, 0, 2], [0.5, 0, 2])[0][0, 0]) for k in range(3000)) / 3000
    d = det(p_near=0.9, p_far=0.5, range_m=8.0, headings=[[1.0, 0.0]])
    far = sum(bool(look(d, k, [0, 0, 2], [7.5, 0, 2])[0][0, 0]) for k in range(3000)) / 3000
    assert near > 0.8 and far < 0.6 and near > far + 0.2


def test_oracle_sees_everything_in_range_exactly():
    d = det(mode='oracle', range_m=8.0, fov_deg=60.0, latency_steps=3, sigma_m=0.5, p_near=0.1, p_far=0.1)
    known, pos, vel = look(d, 0, [0, 0, 2], [-5.0, 1.0, 3.0], bvel=(1.0, 0, 0))
    assert known[0, 0] and np.array_equal(pos[0, 0], [-5.0, 1.0, 3.0]) and np.array_equal(vel[0, 0], [1.0, 0, 0])
    assert not look(d, 1, [0, 0, 2], [9.0, 0, 2])[0][0, 0]              # the same 8 m range


# ---------------------------------------------------------------- C -------
def test_latency_then_extrapolation():
    d = det(latency_steps=3, coast_s=1.0)
    bird = lambda k: np.array([4.0 + 2.0 * k * DT, 0.0, 2.0])
    first = None
    for k in range(40):
        known, pos, _ = look(d, k, [0, 0, 2], bird(k), bvel=(2.0, 0, 0))
        if known[0, 0] and first is None:
            first = k
        if first is not None and k > first + 10:
            assert np.linalg.norm(pos[0, 0] - bird(k)) < 0.15           # extrapolated to now, despite the delay
    assert first == 3                                                   # the first measurement arrives 3 steps late


def test_track_coasts_then_drops():
    d = det(coast_s=0.5, confirm_hits=1)
    for k in range(20):
        assert look(d, k, [0, 0, 2], [4.0, 0, 2])[0][0, 0]
    coast = int(round(0.5 / DT))
    flags = [bool(look(d, 20 + i, [0, 0, 2], [20.0, 0, 2])[0][0, 0]) for i in range(coast + 5)]   # out of range now
    assert all(flags[:coast]) and not any(flags[coast + 1:])            # coasts ~0.5 s, then drops
    assert look(d, 60, [0, 0, 2], [4.0, 0, 2])[0][0, 0]                 # and a new detection starts a new track


def test_tracker_estimates_position_and_velocity():
    d = det(sigma_m=0.1, p_near=0.9, p_far=0.9, latency_steps=3, coast_s=0.5, confirm_hits=2, seed=3)
    errs_p, errs_v = [], []
    for k in range(110):                                                # the bird stays within the 8 m range
        bird = np.array([3.0 + 1.5 * k * DT, 1.0, 2.0])
        known, pos, vel = look(d, k, [0, 0, 2], bird, bvel=(1.5, 0, 0))
        if k > 30 and known[0, 0]:
            errs_p.append(np.linalg.norm(pos[0, 0] - bird))
            errs_v.append(np.linalg.norm(vel[0, 0] - [1.5, 0, 0]))
    assert len(errs_p) > 60 and np.mean(errs_p) < 0.2 and np.mean(errs_v) < 0.6


def test_same_seed_same_tracks():
    out = []
    for _ in range(2):
        d = det(sigma_m=0.1, p_near=0.7, p_far=0.7, seed=5)
        out.append(np.array([look(d, k, [0, 0, 2], [4.0, 0.5, 2])[1][0, 0] for k in range(50)]))
    assert np.array_equal(out[0], out[1])


# ---------------------------------------------------------------- D -------
BUILDING = {'type': 'box', 'center': (3.0, 0.0), 'half': (0.5, 1.0), 'height': 8.0, 'yaw': 0.0}


def tracked_by(mode, layout, bird_x=6.0, steps=40):
    cfg = {'enabled': True, 'count': 1, 'detector': {**SURE, 'mode': mode, 'fov_deg': 360.0}}
    env, _ = make_env(noise=False, birds=cfg)
    still = B.BirdField([B.Bird.straight([bird_x, 0.0, 2.0], [0.0, 0.0, 0.0], 0.12, DT)], DT)
    env.start_episode(SP, GL, obstacles=layout, birds=still)
    seen = False
    for _ in range(steps):
        _, _, _, info = env.step({'drone_0': STILL})
        seen |= bool(info['bird_tracked'][0][0])
    env.close()
    return seen


def test_occlusion_hides_a_bird_behind_a_building():
    assert tracked_by('realistic', [])                                  # the same bird in the open is detected
    assert not tracked_by('realistic', [BUILDING])                      # behind the building it is not
    assert tracked_by('oracle', [BUILDING])                             # the oracle ignores occlusion (upper bound)


def test_occlusion_ignores_drones_and_birds_in_the_way():
    env, _ = make_env(n=2, noise=False, birds={'enabled': True, 'count': 1, 'detector': {**SURE, 'fov_deg': 360.0}})
    sp = {'drone_0': [0.0, 0.0, 2.0], 'drone_1': [3.0, 0.0, 2.0]}      # drone_1 stands between drone_0 and the bird
    gl = {'drone_0': [0.0, 0.0, 2.0], 'drone_1': [3.0, 0.0, 2.0]}
    still = B.BirdField([B.Bird.straight([6.0, 0.0, 2.0], [0, 0, 0], 0.12, DT)], DT)
    env.start_episode(sp, gl, obstacles=[BUILDING_FAR := {'type': 'box', 'center': (3.0, 8.0), 'half': (0.5, 0.5),
                                                         'height': 8.0, 'yaw': 0.0}], birds=still)
    seen = False
    for _ in range(10):
        _, _, _, info = env.step({'drone_0': STILL, 'drone_1': STILL})
        seen |= bool(info['bird_tracked'][0][0])
    assert seen
    env.close()


# ---------------------------------------------------------------- E -------
def test_a_tracked_bird_is_a_neighbour_slot():
    cfg = {'enabled': True, 'count': 1, 'detector': {'enabled': True, 'mode': 'oracle'}}
    env, _ = make_env(n=2, noise=False, birds=cfg)
    sp = {'drone_0': [0.0, 0.0, 2.0], 'drone_1': [0.0, 4.0, 2.0]}
    gl = {'drone_0': [0.0, 0.0, 2.0], 'drone_1': [0.0, 4.0, 2.0]}
    bird = B.BirdField([B.Bird.straight([3.0, 0.0, 2.0], [0.0, 1.0, 0.0], 0.12, DT)], DT)   # 1 m/s towards +y
    env.start_episode(sp, gl, obstacles=[], birds=bird)
    obs, _, _, _ = env.step({'drone_0': STILL, 'drone_1': STILL})
    ns, nm = obs['neighbor_states'].numpy(), obs['neighbor_mask'].numpy()
    # drone_0 slots: the bird (about 3 m away) first, then drone_1 (4 m)
    t = env.birds.k * DT
    assert not nm[0, 0] and not nm[0, 1] and nm[0, 2]
    assert np.allclose(ns[0, 0, :3], np.array([3.0, 1.0 * t, 0.0]) / 3.0, atol=0.02)    # relative position / 3 m
    assert np.allclose(ns[0, 0, 3:], np.array([0.0, 1.0, 0.0]) / 2.0, atol=0.02)        # relative velocity / 2 m/s
    assert np.allclose(ns[0, 1, :3], np.array([0.0, 4.0, 0.0]) / 3.0, atol=0.02)
    env.close()


def test_detector_off_or_nothing_in_range_leaves_the_inputs_unchanged():
    def run(detector):
        cfg = {'enabled': True, 'count': 1, **({'detector': detector} if detector else {})}
        env, _ = make_env(n=2, birds=cfg)
        np.random.seed(7)
        far = B.BirdField([B.Bird.straight([40.0, 40.0, 2.0], [0, 1.0, 0], 0.12, DT)], DT)
        env.start_episode({'drone_0': [0, 0, 2.0], 'drone_1': [0, 2.0, 2.0]},
                          {'drone_0': [4, 0, 2.0], 'drone_1': [4, 2.0, 2.0]}, obstacles=[], birds=far)
        out = []
        for _ in range(40):
            obs, _, _, info = env.step({'drone_0': [1, 0, 0, 0], 'drone_1': [1, 0, 0, 0]})
            out.append(np.concatenate([obs['local_fixed'].numpy().ravel(), obs['neighbor_states'].numpy().ravel(),
                                       obs['neighbor_mask'].numpy().ravel()]))
        env.close()
        return np.array(out), info
    a, info_a = run(None)
    b, info_b = run({'enabled': True})
    assert 'bird_tracked' not in info_a and 'bird_tracked' in info_b
    assert np.array_equal(a, b)                      # the detector seeing nothing changes nothing, bit for bit


# ---------------------------------------------------------------- F -------
def hold_cmd(detector):
    cfg = {'enabled': True, 'count': 1, **({'detector': detector} if detector else {})}
    env, full = make_env(noise=False, birds=cfg)
    # a bird 0.7 m straight above a drone that is 0.2 m from its goal: the horizontal ring cannot see it
    bird = B.BirdField([B.Bird.straight([0.0, 0.0, 2.7], [0, 0, 0], 0.12, DT)], DT)
    obs = env.start_episode({'drone_0': [0.0, 0.0, 2.0]}, {'drone_0': [0.2, 0.0, 2.0]}, obstacles=[], birds=bird)
    pol = GoalHoldPolicy(Zero(), full)
    for _ in range(3):
        act = pol.act(obs, env, 1)
        obs, _, _, _ = env.step(act)
    guarded = bool(pol.guarded[0])
    env.close()
    return guarded, np.array(act['drone_0'][:3])


def test_goal_hold_guard_reacts_to_a_tracked_bird():
    guarded, cmd = hold_cmd(None)
    assert not guarded and np.linalg.norm(cmd) > 0.05               # ring blind to it: goal-hold drives the drone
    guarded, cmd = hold_cmd({'enabled': True, 'mode': 'oracle'})
    assert guarded and np.allclose(cmd, 0.0)                        # tracked within 0.9 m: the policy keeps control
