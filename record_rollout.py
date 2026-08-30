"""
record_rollout.py — Record one deterministic rollout of a trained checkpoint
as an MP4, using PyBullet's GUI window + built-in video logger.

Usage:
    python record_rollout.py --config configs/config_3agent_review.yaml \
        --checkpoint checkpoints_review_3agent/checkpoint_3agent_review_seed1_ep500.pth \
        --out rollout_3agent.mp4

Requires a real display (DISPLAY set) and ffmpeg on PATH — PyBullet's
STATE_LOGGING_VIDEO_MP4 shells out to ffmpeg internally.

Restores the checkpoint's saved log_std ceiling/floor and observation
normalizer exactly like eval_checkpoint.py, so the recorded behavior matches
the actor's real trained behavior — not a mis-normalized approximation.
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
import yaml
import pybullet as p

sys.path.insert(0, os.path.dirname(__file__))

from training.networks.actor_network import ActorNetwork
from envs.swarm_env import SwarmEnv


def unpack_obs(obs_dict, device):
    return (
        obs_dict['local_fixed'].to(device),
        obs_dict['neighbor_states'].to(device),
        obs_dict['neighbor_mask'].to(device),
        obs_dict['global_state'].to(device),
    )


def active_action_dims(curriculum_stage=1):
    return 3 if curriculum_stage < 3 else 4


def apply_yaw_mask(actions, n_active_dims):
    if n_active_dims >= actions.shape[-1]:
        return actions
    masked = actions.clone()
    masked[:, n_active_dims:] = 0.0
    return masked


def actions_to_dict(actions, n_agents):
    arr = actions.detach().cpu().numpy()
    return {f'drone_{i}': arr[i].tolist() for i in range(n_agents)}


def disable_all_collision_physics(env, n_agents):
    """Disable PyBullet's native rigid-body collision response between every
    pair of drones. Without this, drones are real dynamic bodies (mass=1.0,
    box collision shapes) and PyBullet's own physics engine passively
    prevents true interpenetration below ~0.3m (combined box half-extents)
    regardless of what velocity command any controller issues — confirmed
    directly: a naive P-controller's apparent '0.28m safety margin' in every
    earlier stress test vanished entirely once this was disabled, and it
    then genuinely collided (~0.11-0.16m separation, true overlap) in a
    meaningful fraction of trials. Any collision-avoidance comparison that
    doesn't disable this is partly measuring the physics engine, not either
    policy's actual behavior."""
    for i in range(n_agents):
        for j in range(i + 1, n_agents):
            p.setCollisionFilterPair(
                env.adapter.drone_ids[i], env.adapter.drone_ids[j], -1, -1, enableCollision=0
            )


def _apply_scenario(env, spawns, goals, n_agents):
    """Shared plumbing for any hand-constructed spawn/goal scenario: push
    the positions into the adapter, then reset all the per-episode tracking
    state SwarmEnv.reset() would normally initialize (settle counters,
    latched-success flags, distance bookkeeping) so step() behaves exactly
    as it would after a real reset."""
    env.adapter.reset_world(spawns, goals)
    env.goals = np.array([goals[f'drone_{i}'] for i in range(n_agents)], dtype=np.float32)
    positions = env.adapter.get_positions()
    env.prev_dist = np.linalg.norm(positions - env.goals, axis=1)
    env.prev_actions = np.zeros((n_agents, 4), dtype=np.float32)
    env.drone_reached_goal = [False] * n_agents
    env.settle_counter = np.zeros(n_agents, dtype=np.int64)
    env.best_settle_counter = np.zeros(n_agents, dtype=np.int64)
    env.ever_entered_goal = [False] * n_agents
    env.initial_dist = env.prev_dist.copy()
    env.min_dist_ever = env.prev_dist.copy()
    env.episode_manager.reset()

    raw = env.adapter.get_raw_state()
    return env._build_obs(raw)


def setup_crossing_scenario(env, rng, n_agents, r_min=2.5, r_max=6.0, shared_altitude=True):
    """Force a genuine path-crossing scenario: agents placed on a circle,
    each one's goal is the diametrically OPPOSITE point — all straight-line
    paths cross near the shared center. Randomized radius + per-drone
    angular/radial jitter so no two calls produce an identical geometry.

    shared_altitude=True puts every drone at the SAME z — verified via a
    30-trial batch at forced-identical altitude that avoidance is genuinely
    lateral (100% success, 0% collision, min separation 0.289m), not an
    artifact of drones incidentally flying at different heights. Also makes
    the demo visually unambiguous: with independently random per-drone z
    (the old default), a visible near-pass can look like a collision from
    some camera angles even when the drones are meters apart in altitude —
    same-altitude removes that ambiguity entirely.

    Note on visual near-misses even at same altitude: the collision_radius
    used by the sim (0.2m, center-to-center) is smaller than the rendered
    drone body (0.15m half-extent, i.e. 0.3m wide) — so a near-miss with
    centers between 0.2m and 0.3m apart is NOT a collision by the physics/
    reward logic, but the rendered boxes visually overlap on screen. Some
    of the 50-trial batch's successful, non-colliding runs had min
    separation as low as 0.26-0.29m, squarely in that "looks like a
    collision but isn't" zone. Filter for a larger min-separation margin at
    the call site (see --min-demo-separation) to avoid recording one of those.
    """
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
        spawn = np.array([r * np.cos(rad), r * np.sin(rad), z_vals[i]])
        goal = np.array([-r * np.cos(rad), -r * np.sin(rad), z_vals[i]])
        spawns[f'drone_{i}'] = spawn.tolist()
        goals[f'drone_{i}'] = goal.tolist()

    return _apply_scenario(env, spawns, goals, n_agents)


def setup_swap_scenario(env, rng, n_agents, reach=10.0, min_spawn_sep=5.0):
    """Scatter agents at fully random 3D positions (no circle symmetry),
    then assign each one's goal to ANOTHER agent's exact spawn point via a
    derangement (a permutation with no fixed points, so no drone's goal is
    its own spawn) — guarantees genuine, non-trivial crossing paths across
    varied distances and altitudes, without an artificial formation.

    Verified via a 40-trial batch: 100% success, 0% collision, min
    separation as low as 0.266m (same 'looks-worse-than-it-is' zone as the
    crossing scenario — still filter with --min-demo-separation), mean
    travel distance 11.53m (max 16.53m) — much more visually dramatic than
    the crossing scenario's 2-6m radius flights.
    """
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
    while any(perm == np.arange(n_agents)):  # derangement: no drone's goal is its own spawn
        perm = rng.permutation(n_agents)

    spawns = {f'drone_{i}': spawns_arr[i].tolist() for i in range(n_agents)}
    goals = {f'drone_{i}': spawns_arr[perm[i]].tolist() for i in range(n_agents)}

    return _apply_scenario(env, spawns, goals, n_agents)


def setup_headon_scenario(env, rng, n_agents, r_min=3.0, r_max=5.0):
    """Maximum-drama version of the crossing scenario: NO jitter at all —
    an exact symmetric formation (equilateral for 3 agents) on a circle,
    same radius, same altitude, goals at the exact diametrically opposite
    point. All agents are on paths that would collide dead-center at the
    same instant if flown at matched constant speed — a genuine 'this WILL
    collide unless something changes' setup, not just a mild proximity dip.

    Verified via a 15-trial batch: 100% success, 0% collision, but mean
    minimum separation only 0.49m (vs 0.85-3.57m for the jittered crossing/
    swap scenarios) — several trials land in the 0.28-0.5m range (genuine
    close call) and others in 0.6-0.8m (clear, visible swerve). Use
    --min-demo-separation ~0.35 (not the default 0.5) to keep the dramatic
    tight calls without dropping into the <0.3m visually-ambiguous zone.
    """
    R = rng.uniform(r_min, r_max)
    z = rng.uniform(2.0, 3.0)
    base_angle = rng.uniform(0, 360)  # only the overall orientation is randomized

    angles = base_angle + np.linspace(0, 360, n_agents, endpoint=False)
    spawns, goals = {}, {}
    for i, ang in enumerate(angles):
        rad = np.radians(ang)
        spawn = np.array([R * np.cos(rad), R * np.sin(rad), z])
        goal = np.array([-R * np.cos(rad), -R * np.sin(rad), z])
        spawns[f'drone_{i}'] = spawn.tolist()
        goals[f'drone_{i}'] = goal.tolist()

    return _apply_scenario(env, spawns, goals, n_agents)


def main():
    parser = argparse.ArgumentParser(description="Record one deterministic rollout as MP4")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", default="rollout.mp4")
    parser.add_argument("--mode", choices=["deterministic", "stochastic"], default="deterministic")
    parser.add_argument("--max-attempts", type=int, default=20,
                         help="Re-roll spawn/goal scenario up to this many times "
                              "if the episode doesn't reach success (or, for --scenario "
                              "crossing, doesn't also clear --min-demo-separation) — for "
                              "a review video we want to show a clean, unambiguous "
                              "successful rollout, not just any success.")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--slowmo", type=float, default=3.0,
                         help="Playback speed multiplier for the recording — 1.0 is "
                              "real-time (control_timestep per step), higher slows it "
                              "down for visibility. A raw episode is only ~2-3 seconds "
                              "of sim time, too fast to see settlement clearly at 1x.")
    parser.add_argument("--scenario", choices=["random", "crossing", "swap", "headon"],
                         default="random",
                         help="'random' uses the config's own goal_curriculum sampling "
                              "(independently random per-drone goals — verified to rarely "
                              "produce real inter-drone proximity, min separation typically "
                              "several meters). 'crossing' forces all agents' paths through "
                              "a shared center with jitter (100%% success / 0%% collision "
                              "over 50 trials, mean sep 0.85m, but visually modest 2-6m "
                              "radius). 'swap' scatters agents at fully random 3D positions, "
                              "each sent to another agent's exact spawn point — longest and "
                              "most varied flights (100%% / 0%% over 40 trials, mean travel "
                              "11.5m, max 16.5m). 'headon' is ZERO-jitter exact symmetric "
                              "crossing — all agents would collide dead-center at the same "
                              "instant if undriven; the most dramatic genuine near-miss "
                              "(100%% / 0%% over 15 trials, but mean separation only 0.49m — "
                              "use --min-demo-separation ~0.35 with this one).")
    parser.add_argument("--seed", type=int, default=None,
                         help="Seed for the crossing/swap/headon scenario's randomized geometry.")
    parser.add_argument("--camera-pause", type=float, default=0.0,
                         help="Seconds to pause after auto-framing the camera for this "
                              "attempt, before recording starts — gives you a window to "
                              "manually drag/zoom/pan (left-drag rotates, scroll zooms, "
                              "ctrl/middle-drag pans) to a better angle. The auto-frame "
                              "still resets on the NEXT attempt if this one is rejected. "
                              "Note: manual mid-episode camera control has been reported "
                              "unreliable in some remote-display setups — the script only "
                              "sets the camera once per attempt, so if dragging still "
                              "doesn't respond, it's an environment/display issue, not "
                              "something this script is fighting. --camera-mode topdown "
                              "avoids needing manual control at all for this use case.")
    parser.add_argument("--camera-mode", choices=["angled", "topdown"], default="angled",
                         help="'angled' is the default 45deg-yaw/-35deg-pitch view. "
                              "'topdown' looks straight down (pitch ~-90deg) — removes "
                              "the z-axis from the visual entirely, showing pure lateral "
                              "(x,y) motion. Use this to make it visually undeniable that "
                              "avoidance in the crossing/headon scenarios (same altitude "
                              "for all drones, zero vertical motion by construction) is "
                              "genuinely lateral, not a z-axis trick.")
    parser.add_argument("--min-demo-separation", type=float, default=0.5,
                         help="Reject an attempt (even if it succeeded) if the minimum "
                              "inter-drone distance dropped below this during the "
                              "episode. Rendered drone bodies are 0.3m wide but the "
                              "sim's collision_radius is only 0.2m, so a 'successful, "
                              "non-colliding' episode can still show the boxes visually "
                              "overlapping if separation falls in that 0.2-0.3m gap. "
                              "0.5m keeps a clean visible margin between drones on video.")
    parser.add_argument("--show-coords", action="store_true", default=True,
                         help="Overlay each drone's live (x,y,z) position as floating "
                              "text in the GUI/recording, and print a position/distance "
                              "table to the terminal every --print-every steps.")
    parser.add_argument("--no-show-coords", dest="show_coords", action="store_false")
    parser.add_argument("--print-every", type=int, default=10,
                         help="Terminal coordinate printout interval, in control steps.")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device(args.device) if args.device else torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'
    )

    actor = ActorNetwork(config).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state_dict = ckpt['actor_state_dict'] if 'actor_state_dict' in ckpt else ckpt
    actor.load_state_dict(state_dict)
    actor.eval()

    if 'log_std_ceiling' in ckpt:
        actor.set_log_std_ceiling(ckpt['log_std_ceiling'])
    if 'log_std_floor' in ckpt:
        actor.set_log_std_floor(ckpt['log_std_floor'])

    n_active = active_action_dims(ckpt.get('curriculum_stage', 1))
    n_agents = config['environment']['n_agents']
    max_steps = config['environment']['max_steps']
    deterministic = (args.mode == "deterministic")
    control_timestep = config['environment'].get('control_timestep', 0.032)
    step_sleep = control_timestep * args.slowmo

    # A purely vertical label offset renders directly on top of the drone's
    # dot when the camera looks straight down — offset horizontally instead
    # for topdown mode so the label sits legibly beside it.
    label_offset = np.array([0.4, 0, 0]) if args.camera_mode == "topdown" else np.array([0, 0, 0.3])
    label_dup_offset = label_offset + np.array([0.015, 0, 0])

    print(f"Loaded checkpoint (episode {ckpt.get('episode_count', ckpt.get('episode', '?'))}), "
          f"mode={args.mode}, n_agents={n_agents}")

    env = SwarmEnv(config, device=device, gui=True)

    # PyBullet's default GUI has debug sliders/panels cluttering the view
    # and a default camera that isn't pointed anywhere near the drones —
    # world_size is 50m but the actual action happens in a small region
    # near the spawn/goal positions. Hide the panels; the camera itself is
    # framed per-attempt below, once we know where this episode's spawns
    # and goals actually are.
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

    if 'normalizer_mean' in ckpt:
        norm = env.obs_processor.normalizer
        norm.mean  = ckpt['normalizer_mean']
        norm.var   = ckpt['normalizer_var']
        norm.count = ckpt['normalizer_count']
        norm.freeze()
        print(f"Restored and froze observation normalizer (count={norm.count}).")
    else:
        print("WARNING: checkpoint has no saved normalizer state — "
              "recorded behavior may not exactly match the trained policy.")

    # Give the GUI window a moment to open and settle before recording starts
    time.sleep(1.0)

    rng = np.random.RandomState(args.seed)
    success = False
    goal_marker_ids = []
    coord_label_ids = [None] * n_agents      # PyBullet text item ids, per drone
    coord_label_dup_ids = [None] * n_agents  # faux-bold: same text redrawn at a tiny
                                               # offset — PyBullet's debug text has no
                                               # real bold/weight parameter at all
    for attempt in range(args.max_attempts):
        if args.scenario == "crossing":
            obs = setup_crossing_scenario(env, rng, n_agents)
        elif args.scenario == "swap":
            obs = setup_swap_scenario(env, rng, n_agents)
        elif args.scenario == "headon":
            obs = setup_headon_scenario(env, rng, n_agents)
        else:
            obs = env.reset()
        lf, ns, nm, gs = unpack_obs(obs, device)
        info = {}

        # Frame the camera on THIS episode's actual spawn+goal positions —
        # they're randomized each reset, and the default PyBullet camera
        # isn't pointed at any of them. Distance is sized to the spread of
        # points so the whole scene fits regardless of where they landed.
        positions = env.adapter.get_positions()
        all_points = np.vstack([positions, env.goals])
        center = all_points.mean(axis=0)
        spread = float(np.max(np.linalg.norm(all_points - center, axis=1)))
        cam_distance = max(spread * 2.5, 4.0)
        pitch = -89.9 if args.camera_mode == "topdown" else -35
        p.resetDebugVisualizerCamera(
            cameraDistance=cam_distance,
            cameraYaw=45,
            cameraPitch=pitch,
            cameraTargetPosition=center.tolist(),
        )

        # The goals themselves are invisible by default — only the drone
        # bodies render. Add a translucent green marker sphere at each goal
        # (visual-only: no collision shape, mass=0, not part of
        # env.adapter.drone_ids) so the video actually shows what each
        # drone is flying toward and visibly stopping inside.
        for gid in goal_marker_ids:
            p.removeBody(gid)
        goal_marker_ids = []
        for goal_pos in env.goals:
            vis_shape = p.createVisualShape(
                p.GEOM_SPHERE, radius=0.5, rgbaColor=[0.1, 0.9, 0.2, 0.35]
            )
            marker_id = p.createMultiBody(
                baseMass=0, baseVisualShapeIndex=vis_shape,
                basePosition=goal_pos.tolist(),
            )
            goal_marker_ids.append(marker_id)

        # Live coordinate labels — floating text above each drone, updated
        # in place every step (replaceItemUniqueId avoids spamming the
        # debug-item list with a new label each frame). Visible both in the
        # live GUI and in the recorded video, since it's part of the scene.
        if args.show_coords:
            positions = env.adapter.get_positions()
            for i in range(n_agents):
                text = f"d{i}: ({positions[i][0]:+.2f}, {positions[i][1]:+.2f}, {positions[i][2]:+.2f})"
                label_pos = (positions[i] + label_offset).tolist()
                dup_pos = (positions[i] + label_dup_offset).tolist()
                coord_label_ids[i] = p.addUserDebugText(
                    text, label_pos, textColorRGB=[0, 0, 0], textSize=1.6,
                )
                coord_label_dup_ids[i] = p.addUserDebugText(
                    text, dup_pos, textColorRGB=[0, 0, 0], textSize=1.6,
                )

        if args.camera_pause > 0:
            print(f"  Camera auto-framed — {args.camera_pause:.0f}s to manually adjust "
                  f"(left-drag rotate, scroll zoom, ctrl/middle-drag pan)...")
            time.sleep(args.camera_pause)

        # Restart logging fresh each attempt — starting a new log at the
        # same path overwrites the previous attempt's file, so only the
        # LAST attempt's rollout survives on disk (which is the successful
        # one, since we break out of the loop as soon as one succeeds).
        log_id = p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, args.out)
        print(f"Recording attempt {attempt + 1}/{args.max_attempts} -> {args.out}")

        ep_min_sep = float('inf')
        for step in range(max_steps):
            with torch.no_grad():
                actions, _ = actor.get_action(
                    lf, ns, nm, deterministic=deterministic, active_dims=n_active
                )
            masked = apply_yaw_mask(actions, n_active)
            next_obs, _, done, info = env.step(actions_to_dict(masked, n_agents))
            sep = info.get('min_inter_drone_distance', None)
            if sep is not None:
                ep_min_sep = min(ep_min_sep, sep)

            if args.show_coords:
                positions = env.adapter.get_positions()
                per_agent_dist = info.get('per_agent_min_dist', None)
                for i in range(n_agents):
                    text = f"d{i}: ({positions[i][0]:+.2f}, {positions[i][1]:+.2f}, {positions[i][2]:+.2f})"
                    label_pos = (positions[i] + label_offset).tolist()
                    dup_pos = (positions[i] + label_dup_offset).tolist()
                    coord_label_ids[i] = p.addUserDebugText(
                        text, label_pos, textColorRGB=[0, 0, 0], textSize=1.6,
                        replaceItemUniqueId=coord_label_ids[i],
                    )
                    coord_label_dup_ids[i] = p.addUserDebugText(
                        text, dup_pos, textColorRGB=[0, 0, 0], textSize=1.6,
                        replaceItemUniqueId=coord_label_dup_ids[i],
                    )
                if step % args.print_every == 0:
                    print(f"    step {step:4d} | " + " | ".join(
                        f"d{i}=({positions[i][0]:+.2f},{positions[i][1]:+.2f},{positions[i][2]:+.2f})"
                        for i in range(n_agents)
                    ) + (f" | min_sep={sep:.2f}m" if sep is not None else ""))

            # Pace to (slowed) real-time — env.step() otherwise runs as fast
            # as the CPU/GPU allow, which is correct for training but means
            # a whole episode finishes in well under a second of wall clock,
            # giving the video logger almost no frames to capture (observed:
            # 14 frames / 0.18s for a 72-step episode with no sleep at all).
            time.sleep(step_sleep)
            if done:
                break
            lf, ns, nm, gs = unpack_obs(next_obs, device)

        p.stopStateLogging(log_id)
        success = info.get('success', False)
        sep_str = 'N/A' if ep_min_sep == float('inf') else f'{ep_min_sep:.3f}m'
        # min_inter_drone_distance is None for n_agents==1 — the separation
        # filter doesn't apply there, only to success.
        sep_ok = (ep_min_sep == float('inf')) or (ep_min_sep >= args.min_demo_separation)
        print(f"  attempt {attempt + 1}: success={success}, "
              f"drones_at_goal={info.get('drones_at_goal')}/{n_agents}, "
              f"min_dist={info.get('min_dist_to_goal', float('nan')):.3f}, "
              f"min_inter_drone_sep={sep_str}"
              f"{'' if sep_ok else ' (BELOW min-demo-separation, rejecting)'}")

        if success and sep_ok:
            print(f"\nRecorded a successful, clean-margin rollout to {args.out}")
            break
    else:
        print(f"\nWARNING: no attempt in {args.max_attempts} both succeeded AND kept "
              f"separation >= {args.min_demo_separation}m — {args.out} contains the "
              f"LAST attempt's rollout, which does not meet one of those criteria. "
              f"Try increasing --max-attempts or lowering --min-demo-separation.")

    env.close()


if __name__ == "__main__":
    main()
