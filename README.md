# UTM Swarm MARL

Autonomous traffic management for low-altitude drone swarms, learned with multi-agent
reinforcement learning.

This project trains a fleet of drones to navigate shared urban airspace on their own —
reaching their destinations while avoiding each other and static obstacles — without a
central air-traffic controller telling each drone what to do at runtime. Every drone runs
the same lightweight policy on its own onboard state and a local view of its neighbours.
Coordination is something the swarm *learns*, not something we hand-code.

The long-term motivation is the coming wave of delivery and inspection drones over dense
cities. Once several operators are flying hundreds of drones through the same low-altitude
corridors, human-style air-traffic control does not scale. We are building a proof of
concept for the intelligence layer that could.

---

## The approach

We use **MAPPO** (Multi-Agent Proximal Policy Optimization) under the **CTDE** paradigm —
Centralized Training, Decentralized Execution:

- **During training**, a centralized *critic* gets to see the whole swarm's state. That
  richer picture gives the learning signal something a single drone could never observe on
  its own, which makes training more stable.
- **At execution time**, the critic is thrown away. Each drone runs only its decentralized
  *actor*, using its own sensors and a local view of nearby drones. This is what makes the
  system deployable and scalable — no drone depends on a central server to decide its next
  move.

Two design choices are worth calling out:

- **Attention over neighbours.** Instead of feeding the policy a fixed-size list of "the K
  nearest drones" (which breaks the moment the swarm grows), each drone runs a small
  attention module that compresses however many neighbours it sees into a fixed-length
  summary. The network never has to be rebuilt as the swarm scales, and the policy learns
  to pay more attention to the drones that actually matter — the ones on a collision course
  — and ignore the ones flying safely away.
- **Curriculum learning.** We do not drop the swarm straight into a hard city. Training
  ramps up in stages: empty space first, then sparse obstacles, then a dense urban grid,
  then a real city map. Each stage only unlocks once the swarm is reliably succeeding at the
  previous one.

---

## Architecture

The system is organized into four modules. Data flows in a closed loop: the simulator
produces state, the policy acts, the world responds, and the learner updates.

**Module 1 — Environment & World Generation.**
`SwarmEnv` is the RL-facing wrapper that everything trains against. Underneath it sits a
simulator adapter (a thin driver that talks to the physics engine), a scenario manager that
procedurally lays out drones, goals, and obstacles, and an episode manager that decides when
an episode ends (goal reached, collision, or timeout) and tracks curriculum progression.
The adapter is deliberately swappable — the RL engine does not care whether PyBullet or
Webots is underneath, as long as the adapter emits the same state contract.

**Module 2 — State Processing & Reward Engine.**
The observation processor turns raw simulator readings into the exact tensors the networks
expect: a per-drone local observation, the relative states of nearby drones, and a global
state for the critic. The reward engine scores every step across seven components — goal
progress, goal completion, obstacle and drone collisions, a time penalty, an action-
smoothness penalty (so the policy does not learn to jerk the motors), and a proximity
penalty that keeps drones out of each other's downwash.

**Module 3 — MAPPO Training Engine.**
The decentralized actor and centralized critic, a rollout buffer that stores transitions and
computes advantages (GAE), and the PPO optimizer that runs the clipped-surrogate update. The
trainer orchestrates the whole loop — collecting episodes, updating the networks, saving
checkpoints, and advancing the curriculum.

**Module 4 — Evaluation & Deployment.**
The trained policy is served to drone "client nodes" that load the weights and run inference,
with an analytics layer for metrics and reporting. This module is future work.

A full architecture diagram lives in `../diagrams/`.

---

## Repository layout

```
utm-swarm-marl/
├── configs/
│   └── default_config.yaml     # single source of truth for every hyperparameter
├── envs/
│   ├── episode_manager.py      # episode lifecycle + curriculum gating   [done]
│   ├── swarm_env.py            # RL environment wrapper                   [in progress]
│   ├── pybullet_adapter.py     # PyBullet simulator driver               [in progress]
│   ├── webots_adapter.py       # Webots simulator driver                 [later]
│   └── scenario_manager.py     # procedural world generation             [in progress]
├── modules/
│   ├── observation_processor.py  # raw state -> network tensors          [done]
│   └── reward_engine.py          # 7-component reward                    [done]
├── training/
│   ├── networks/
│   │   ├── actor_network.py    # decentralized policy + attention        [done]
│   │   └── critic_network.py   # centralized value function              [done]
│   ├── rollout_buffer.py       # transition storage + GAE                [done]
│   ├── ppo_optimizer.py        # clipped-surrogate PPO update            [done]
│   └── mappo_trainer.py        # training loop orchestrator              [done]
├── deployment/                 # policy server / client / analytics      [later]
├── tests/                      # per-module checks
└── docs/
    └── data_contract.md        # the observation/action/reward schema
```

---

## Current status

The learning engine is built. The actor, critic, rollout buffer, PPO optimizer, observation
processor, reward engine, and episode manager are all implemented, and the full training loop
runs end-to-end on synthetic data (`python training/mappo_trainer.py`) with stable PPO
diagnostics (bounded KL, sensible clip fractions, no NaNs).

What is not done yet is the bridge from a real simulator into that engine — the `SwarmEnv`
wrapper and a simulator adapter. That is the current focus. The immediate milestone is a
first real training run on **Curriculum Stage 1** (empty sandbox, basic navigation), trained
in PyBullet for speed. Webots is reserved for later high-fidelity validation, since the two
simulators share the same state contract and a policy trained in one can be transferred to
the other.

No trained checkpoints exist yet — producing the first one is the near-term goal.

---

## Curriculum

| Stage | Environment            | Action space          | What the swarm learns              |
|-------|------------------------|-----------------------|------------------------------------|
| 1     | Empty sandbox          | Vx, Vy, Vz            | Basic goal-reaching navigation     |
| 2     | Sparse obstacles       | Vx, Vy, Vz            | Obstacle avoidance                 |
| 3     | Dense urban grid       | Vx, Vy, Vz, Yaw rate  | Tight coordination + rotation      |
| 4     | Real OSM city map      | Vx, Vy, Vz, Yaw rate  | Final validation at city scale     |

Yaw is intentionally masked off until Stage 3 so the policy is not overwhelmed by a 4D action
space before it can navigate.

---

## Getting started

Requires Python 3.10+.

```bash
pip install -r requirements.txt
```

Run the module checks:

```bash
python tests/test_actor_network.py
python tests/test_episode_manager.py
python tests/test_reward_engine.py
```

Run the training loop on synthetic data (no simulator needed — verifies the engine
end-to-end):

```bash
python training/mappo_trainer.py
```

Every hyperparameter lives in `configs/default_config.yaml`. Change it there and it changes
everywhere — nothing is hard-coded in the modules.

---

## Roadmap

- [ ] `SwarmEnv` + PyBullet adapter — the integration bridge
- [ ] First Stage-1 training run and checkpoint
- [ ] Evaluation script (success rate, collisions, path efficiency)
- [ ] Stages 2–4 curriculum training
- [ ] H3 hexagonal indexing for O(1) neighbour lookup at large swarm sizes
- [ ] Webots high-fidelity validation and sim-to-sim transfer
- [ ] Deployment layer (policy server + client inference)

---

## Tech stack

PyTorch (networks and training), NumPy, PyYAML (config), Weights & Biases (optional training
logs), Matplotlib (plots). PyBullet for early-stage training; Webots for high-fidelity
validation. OSMnx is used to pull real building footprints for the Stage-4 city map.

Training runs on the lab GPU system; the engine also runs on CPU for local testing.
