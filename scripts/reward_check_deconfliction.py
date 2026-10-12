"""
reward_check_deconfliction.py — does our reward actually make avoiding pay?

Evaluation only, no model. Two scripted drones, collision physics off, the
3-agent training config's reward weights. Each situation is played out twice:
  A: the drones ignore each other and crash
  B: one drone avoids (sidestep / swerve / wait)
Both versions are identical up to the moment the avoiding drone deviates
(the "decision point"). From that point on we add up each drone's reward the
way training sees it: per drone, discounted by gamma per 32 ms step, with the
episode ending at the first collision.

If B doesn't clearly beat A for every drone, training has little reason to
learn avoidance and the reward needs fixing first.

    python scripts/reward_check_deconfliction.py
    python scripts/reward_check_deconfliction.py --config configs/config_3agent_deconfliction_v2.yaml
"""

import os
import sys

import numpy as np
import yaml

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)

from envs.swarm_env import SwarmEnv
from record_rollout import disable_all_collision_physics, _apply_scenario

# Reward weights come from this config. Override: --config configs/<file>.yaml
CONFIG = os.path.join(ROOT, "configs", "config_3agent_review.yaml")
if "--config" in sys.argv:
    CONFIG = os.path.join(ROOT, sys.argv[sys.argv.index("--config") + 1])
COMPONENTS = ["goal_progress", "goal_entry", "goal_reached", "settle_counter_progress",
              "goal_hold", "proximity_penalty", "drone_collision", "step_penalty"]


def make_env():
    with open(CONFIG) as f:
        cfg = yaml.safe_load(f)
    cfg["environment"]["n_agents"] = 2
    env = SwarmEnv(cfg)
    disable_all_collision_physics(env, 2)
    # Keep a copy of the per-drone reward breakdown from every step
    original = env.reward_engine.compute_rewards
    def recording(data):
        out = original(data)
        env._last_components = {k: np.array(v, dtype=float) for k, v in out.items()}
        return out
    env.reward_engine.compute_rewards = recording
    return env, cfg


def toward(pos, target, vmax):
    err = np.asarray(target) - pos
    d = np.linalg.norm(err)
    return np.zeros(3) if d < 1e-6 else err / d * min(vmax, 3.0 * d)


def play(env, spawns, goals, controller, max_steps):
    """Run one episode. controller(step, pos, state) -> (vel0, vel1, decided)
    where `decided` marks steps from the decision point onwards."""
    _apply_scenario(env, {k: list(v) for k, v in spawns.items()},
                    {k: list(v) for k, v in goals.items()}, 2)
    state, log, t0 = {}, [], None
    info, reason = {}, "timeout"
    for step in range(max_steps):
        pos = env.adapter.get_positions()
        v0, v1, decided = controller(step, pos, state)
        if decided and t0 is None:
            t0 = step
        _, _, done, info = env.step({"drone_0": list(v0) + [0.0], "drone_1": list(v1) + [0.0]})
        log.append({k: env._last_components[k].copy() for k in COMPONENTS + ["total"]})
        if done:
            reason = info.get("reason") or "timeout"
            break
    return log, (t0 if t0 is not None else 0), reason, info


def returns_from(log, t0, gamma):
    """Per-drone discounted return from t0, plus undiscounted component sums."""
    disc = np.zeros(2)
    comp = {k: np.zeros(2) for k in COMPONENTS}
    for t, r in enumerate(log[t0:]):
        disc += (gamma ** t) * r["total"]
        for k in COMPONENTS:
            comp[k] += r[k]
    return disc, comp


def report(name, env, cfg, spawns, goals, ctrl_a, ctrl_b, who_avoids):
    gamma = cfg["training"]["gamma"]
    max_steps = cfg["environment"]["max_steps"]
    log_a, t0_a, why_a, info_a = play(env, spawns, goals, ctrl_a, max_steps)
    log_b, t0_b, why_b, info_b = play(env, spawns, goals, ctrl_b, max_steps)
    t0 = t0_b                     # where B starts doing something different
    g_a, c_a = returns_from(log_a, t0, gamma)
    g_b, c_b = returns_from(log_b, t0, gamma)

    print(f"\n{'=' * 78}\n{name}\n{'=' * 78}")
    print(f"decision point: step {t0} ({t0 * 0.032:.2f} s).  drone {who_avoids} is the one that avoids in B")
    print(f"A: ended by {why_a} at step {len(log_a)}, settled {info_a.get('drone_reached_goal')}")
    print(f"B: ended by {why_b} at step {len(log_b)}, settled {info_b.get('drone_reached_goal')}")
    print(f"\n{'':26s}{'drone 0':>22s}{'drone 1':>22s}")
    print(f"{'':26s}{'A crash':>11s}{'B avoid':>11s}{'A crash':>11s}{'B avoid':>11s}")
    for k in COMPONENTS:
        print(f"  {k:24s}{c_a[k][0]:+11.2f}{c_b[k][0]:+11.2f}{c_a[k][1]:+11.2f}{c_b[k][1]:+11.2f}")
    print(f"  {'-' * 68}")
    print(f"  {'discounted return':24s}{g_a[0]:+11.2f}{g_b[0]:+11.2f}{g_a[1]:+11.2f}{g_b[1]:+11.2f}")
    verdict = ["avoid wins" if g_b[i] > g_a[i] else "CRASH WINS" for i in range(2)]
    print(f"  {'B - A':24s}{g_b[0] - g_a[0]:+22.2f}{g_b[1] - g_a[1]:+22.2f}")
    print(f"  {'':24s}{verdict[0]:>22s}{verdict[1]:>22s}")
    return g_b - g_a


def main():
    env, cfg = make_env()
    print(f"reward weights from {os.path.relpath(CONFIG, ROOT)}: lambda_prox={cfg['reward']['lambda_prox']}, "
          f"r_danger={cfg['reward']['r_danger']}, collision={cfg['reward']['drone_collision_penalty']}")
    z = 2.5
    results = {}

    # 1. Intruder: drone 0 hovers settled on its goal, drone 1 flies straight
    #    through that point. In B, drone 0 steps 1 m sideways when drone 1 is
    #    2.5 m away and returns once it has passed.
    for speed in (1.0, 2.0):
        G = np.array([0.0, 0.0, z])
        spawns = {"drone_0": G, "drone_1": np.array([-6.0, 0.0, z])}
        goals = {"drone_0": G, "drone_1": np.array([6.0, 0.0, z])}

        def intruder(step, pos):
            return toward(pos[1], goals["drone_1"], speed)

        def ctrl_a(step, pos, s):
            return toward(pos[0], G, 1.0), intruder(step, pos), False

        def ctrl_b(step, pos, s):
            if "phase" not in s:
                s["phase"] = "hold"
            if s["phase"] == "hold" and pos[1][0] > -2.5:
                s["phase"] = "aside"
            elif s["phase"] == "aside" and pos[1][0] > 1.5:
                s["phase"] = "back"
            target = G + np.array([0.0, 1.0, 0.0]) if s["phase"] == "aside" else G
            return toward(pos[0], target, 1.0), intruder(step, pos), s["phase"] != "hold"

        results[f"intruder {speed:g} m/s"] = report(
            f"1. INTRUDER at {speed:g} m/s: settled drone stays put (A) vs steps 1 m aside (B)",
            env, cfg, spawns, goals, ctrl_a, ctrl_b, who_avoids=0)

    # 2. Head-on: both fly at each other at 1.5 m/s. In B, drone 1 swerves
    #    1.2 m to the side once they are 4 m apart.
    S0, S1 = np.array([-4.0, 0.0, z]), np.array([4.0, 0.0, z])
    spawns, goals = {"drone_0": S0, "drone_1": S1}, {"drone_0": S1, "drone_1": S0}

    def ctrl_a(step, pos, s):
        return toward(pos[0], S1, 1.5), toward(pos[1], S0, 1.5), False

    def ctrl_b(step, pos, s):
        if "phase" not in s:
            s["phase"] = "straight"
        sep = np.linalg.norm(pos[0] - pos[1])
        if s["phase"] == "straight" and sep < 4.0:
            s["phase"] = "swerve"
        elif s["phase"] == "swerve" and pos[1][0] < pos[0][0] - 1.0:
            s["phase"] = "home"
        waypoint = np.array([0.0, 1.2, z]) if s["phase"] == "swerve" else S0
        return toward(pos[0], S1, 1.5), toward(pos[1], waypoint, 1.5), s["phase"] != "straight"

    results["head-on"] = report(
        "2. HEAD-ON at 1.5 m/s: both fly straight (A) vs drone 1 swerves 1.2 m (B)",
        env, cfg, spawns, goals, ctrl_a, ctrl_b, who_avoids=1)

    # 3. Crossing at 90 degrees, timed to meet in the middle. In B, drone 1
    #    stops and waits until drone 0 has passed, then carries on.
    S0, G0 = np.array([-4.0, 0.0, z]), np.array([4.0, 0.0, z])
    S1, G1 = np.array([0.0, -4.0, z]), np.array([0.0, 4.0, z])
    spawns, goals = {"drone_0": S0, "drone_1": S1}, {"drone_0": G0, "drone_1": G1}

    def ctrl_a(step, pos, s):
        return toward(pos[0], G0, 1.5), toward(pos[1], G1, 1.5), False

    def ctrl_b(step, pos, s):
        if "phase" not in s:
            s["phase"] = "go"
        if s["phase"] == "go" and pos[1][1] > -1.5 and pos[0][0] < 1.0:
            s["phase"], s["hold"] = "wait", pos[1].copy()
        elif s["phase"] == "wait" and pos[0][0] > 1.0:
            s["phase"] = "resume"
        v1 = toward(pos[1], s["hold"], 1.5) if s["phase"] == "wait" else toward(pos[1], G1, 1.5)
        return toward(pos[0], G0, 1.5), v1, s["phase"] != "go"

    results["crossing"] = report(
        "3. CROSSING at 1.5 m/s: both keep going (A) vs drone 1 waits for drone 0 to pass (B)",
        env, cfg, spawns, goals, ctrl_a, ctrl_b, who_avoids=1)

    env.close()
    print(f"\n{'=' * 78}\nSUMMARY: discounted return of avoiding minus crashing (positive = avoiding pays)")
    for k, d in results.items():
        print(f"  {k:20s} drone 0 {d[0]:+8.2f}   drone 1 {d[1]:+8.2f}")


if __name__ == "__main__":
    main()
