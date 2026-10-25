"""
test_rotated_boxes.py
---------------------
Optional 'yaw' on box obstacles (envs/scenario_manager.py): a rotated block.

  A. No 'yaw' and yaw = 0 give exactly the old axis-aligned results.
  B. Rotating a box and a point together about the box centre leaves the distance unchanged.
  C. The teacher's closest_point lies on the rotated surface, at the reported distance.
  D. PyBullet builds the box rotated: horizontal LiDAR-style rays hit the analytic surface.
  E. A point inside the axis-aligned bounding box but outside the rotated box is free.
  F. The rule teacher flies around a rotated wide building and reaches its goal.

Run: python -m pytest tests/test_rotated_boxes.py -q
"""

import os
import sys

import numpy as np
import pybullet as p
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from envs.scenario_manager import surface_distance, collides
from baselines.repulsion_controller import closest_point, RepulsionController

BOX = {'type': 'box', 'center': (1.0, -2.0), 'half': (1.5, 0.4), 'height': 8.0}


def rot(v, a, c=(0.0, 0.0)):
    c = np.asarray(c, dtype=float)
    d = np.asarray(v, dtype=float) - c
    return c + np.array([np.cos(a) * d[0] - np.sin(a) * d[1], np.sin(a) * d[0] + np.cos(a) * d[1]])


def test_no_yaw_is_unchanged():
    rng = np.random.default_rng(0)
    zero = dict(BOX, yaw=0.0)
    for _ in range(200):
        q = rng.uniform(-5, 5, 2)
        assert surface_distance(BOX, q) == surface_distance(zero, q)
        assert np.allclose(closest_point(BOX, q), closest_point(zero, q))


def test_rotation_invariance():
    rng = np.random.default_rng(1)
    for a in (0.3, 0.785398, 1.2, -2.0):
        turned = dict(BOX, yaw=a)
        for _ in range(200):
            q = rng.uniform(-5, 5, 2)
            assert abs(surface_distance(BOX, q) - surface_distance(turned, rot(q, a, BOX['center']))) < 1e-9


def test_closest_point_on_rotated_surface():
    rng = np.random.default_rng(2)
    turned = dict(BOX, yaw=0.6)
    for _ in range(300):
        q = rng.uniform(-5, 5, 2)
        cp = closest_point(turned, q)
        assert abs(surface_distance(turned, cp)) < 1e-9                 # on the surface
        d = surface_distance(turned, q)
        if d > 0:
            assert abs(np.linalg.norm(q - cp) - d) < 1e-9               # at the reported distance


def test_pybullet_body_is_rotated():
    from envs.pybullet_adapter import PyBulletAdapter
    config = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), '..', 'configs',
                                              'config_stage2_ring_wider_teacher.yaml'), encoding='utf-8'))
    config['environment']['n_agents'] = 1
    adapter = PyBulletAdapter(config)
    try:
        turned = dict(BOX, center=(0.0, 0.0), yaw=0.5)
        adapter.reset_world({'drone_0': [5.0, 5.0, 2.0]}, {'drone_0': [6.0, 5.0, 2.0]}, [turned])
        for ang in np.linspace(0, 2 * np.pi, 24, endpoint=False):
            start = np.array([6 * np.cos(ang), 6 * np.sin(ang)])
            hit = p.rayTest([*start, 2.0], [0.0, 0.0, 2.0], physicsClientId=adapter.client)[0]
            assert hit[0] == adapter.obstacle_ids[0]
            along = hit[2] * 6.0                                            # distance from start to the hit
            # analytic: march along the ray to where the surface distance reaches 0
            ts = np.linspace(0, 6, 6001)
            pts = start[None, :] * (1 - ts[:, None] / 6)
            first = ts[np.argmax([surface_distance(turned, q) <= 0 for q in pts])]
            assert abs(along - first) < 0.01
    finally:
        adapter.close()


def test_inside_aabb_outside_rotated_box_is_free():
    turned = {'type': 'box', 'center': (0.0, 0.0), 'half': (2.0, 0.3), 'height': 8.0, 'yaw': np.pi / 4}
    corner = np.array([1.2, -1.2, 2.0])            # inside the AABB (|x|, |y| <= 1.63), far from the diagonal bar
    assert not collides([turned], corner)
    assert collides([turned], np.array([1.0, 1.0, 2.0]))                    # on the bar's axis


def test_teacher_flies_around_rotated_building():
    from envs.swarm_env import SwarmEnv
    config = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), '..', 'configs',
                                              'config_stage2_ring_wider_teacher.yaml'), encoding='utf-8'))
    config['environment']['n_agents'] = 1
    env = SwarmEnv(config)
    teacher = RepulsionController(config)
    try:
        for yaw in (0.4, 0.9):
            building = {'type': 'box', 'center': (0.0, 0.0), 'half': (1.1, 1.0), 'height': 8.0, 'yaw': yaw}
            obs = env.start_episode({'drone_0': [-5.0, 0.1, 2.5]}, {'drone_0': [5.0, 0.0, 2.5]}, obstacles=[building])
            done, info = False, {}
            for _ in range(config['environment']['max_steps']):
                obs, _, done, info = env.step(teacher.act(obs, env, 1))
                if done:
                    break
            assert info.get('reason') == 'success', (yaw, info.get('reason'))
    finally:
        env.close()
