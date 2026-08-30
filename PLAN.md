# Project Context: UTM Swarm MARL

## Overview
This project implements a Multi-Agent Reinforcement Learning (MARL) system for coordinated DJI Mavic 2 Pro drone swarm control. 
* **Simulator:** Webots R2025a (via Python API).
* **Algorithm:** MAPPO (Multi-Agent Proximal Policy Optimization).
* **Control Strategy:** Hierarchical. The Actor network outputs velocity commands; a custom `WebotsAdapter` uses the Supervisor API (`setVelocity`) to bypass low-level motor PID tuning (Stage 1 training).

## Current Project State
* **Phase A (Simulator Foundation):** Complete. Single-agent Webots adapter is working with a custom P-Controller, active stabilization, and gravity compensation. Teammates are currently scaling this to 3 drones and building the `SwarmEnv` Gymnasium wrapper.
* **Phase B (MAPPO Engine):** Complete. The pure PyTorch math engine (Buffer, Optimizer, Trainer, Networks, Observation Processor) is written.
* **Phase C (Integration):** Ready to proceed with the approved Team-Scalar baseline. Phase B dummy validation is treated as passed for integration readiness: KL and clip fractions are stable, and critic-loss behavior under random synthetic rewards is expected and non-blocking.

## Data Flow & Architecture
### 1. Observation Processor
* **Input:** Raw Webots dict (positions, velocities, 14-ray LiDAR, relative goals).
* **Output (per step):**
  * `local_fixed`: (N, 27) -> Own drone state.
  * `neighbor_states`: (N, K, 6) -> Relative states of K-nearest neighbors.
  * `neighbor_mask`: (N, K) -> Padding mask for empty neighbor slots.
  * `global_state`: (N, 17) -> Absolute state for centralized Critic.

### 2. Networks
* **Actor Network (Decentralized):** * Takes `local_fixed`, `neighbor_states`, `neighbor_mask`.
  * Internally runs attention to produce a 59-dim observation.
  * Outputs: `actions` (N, action_dim), `log_probs` (N,).
* **Critic Network (Centralized):** * Takes `global_state` (N, 17).
  * Uses a shared `Linear(17, 128)` encoder per drone -> Mean Pooling -> MLP. 
  * Outputs: team scalar `values` `(batch, 1)`. N-agnostic and permutation invariant.

### 3. Training Loop (The Engine)
* **RolloutBuffer:** Stores multi-agent tensors: `obs`, `neighbor_states`, `masks`, `global_state`, `actions`, `logprobs`, `rewards`, `values`. Computes GAE.
* **PPOOptimizer:** Fetches minibatches, evaluates actions, computes clipped surrogate loss + critic MSE + entropy, and backpropagates.
* **Trainer Adapter Policy:** For current buffer compatibility, the trainer broadcasts the team-scalar critic output to per-agent shape `(N,)` before storing transitions and computing GAE.



# Project Status: Multi-Agent Drone UTM System

### 1. Completed Foundation (Key Design Decisions)

_We built and validated the core math and architecture before starting the simulator to ensure no wasted GPU time._

- **The "Data Contract" (`default_config.yaml`):** Created a single source of truth for the entire system.
    
    - _Key Decision:_ **Decoupled Timesteps.** AI control runs at 32ms, physics runs at 16ms. This explicitly matches real-world flight controllers (like PX4) for future Sim-to-Real transfer.
        
- **Actor Network:** Built the decentralized policy network.
    
    - _Key Decision:_ **Custom Attention Module.** Instead of standard padding, we compress a variable number of neighbors into a fixed 32-dim vector. This allows the swarm to scale up dynamically without breaking the neural network.
        
- **Reward Engine Validation (See Graphs):** Implemented the 7-component reward structure.
    
    - _Crucial Note:_ The graphs show **synthetic/hardcoded trajectories, NOT a trained policy.** * _Why we did this:_ It mathematically proves that our logic correctly rewards goal progress and smoothly penalizes dangerous proximity/jerky motors _before_ we even touch the simulator.

        

### 2. Immediate Next Step (This Week)

- **Webots Proof of Concept:** Build the `WebotsAdapter` to read sensors and send motor commands.
    
    - _Deliverable:_ Get **one single drone** moving in the Webots environment using a hardcoded script to prove the bridge between Python and the physics engine is stable.
        

### 3. Implementation Roadmap (Phases)

- **Phase A: Webots Foundation:** Build and validate simulator control plumbing with a hardcoded single-drone proof of concept.

- **Phase B: MAPPO Engine Completion:** Finalize Buffer/Optimizer/Trainer/ObservationProcessor and validate math pipeline on dummy data.

- **Phase C: Integration:** Connect MAPPO to `SwarmEnv` and verify end-to-end rollout, reward flow, and tensor compatibility in Webots.

- **Phase D: Baseline Training (Team-Scalar):** Train and stabilize the approved Team-Scalar critic baseline as the reference system.

- **Phase E: Per-Agent Critic Ablation:** Compare per-agent critic variant against the Team-Scalar baseline under controlled conditions.

- **Phase F: Training + Curriculum:** Scale through Stage 1-4 curriculum once architecture decision is locked.

- **Phase G: Enhancements:** Add H3 indexing, dashboards, and safety/productization upgrades after training flow is stable.


## The Correct Build Order

```
Phase A — Webots Foundation
    ↓
Phase B — MAPPO Engine Completion  
    ↓
Phase C — Integration
    ↓
Phase D — Baseline Training (Team-Scalar)
    ↓
Phase E — Per-Agent Critic Ablation
    ↓
Phase F — Training + Curriculum
    ↓
Phase G — Enhancements (H3, WandB, LLM dashboard etc.)
```



## Phase A: Webots Foundation 

**Goal: One drone moving in Webots. Nothing more.**

```
Step 1 — Create basic Webots world file
  - Flat ground plane
  - One drone spawned
  - Boundary walls
  - GPS + IMU + distance sensors attached

Step 2 — WebotsAdapter core methods
  - get_raw_state()     → read drone position/velocity from Supervisor API
  - apply_actions()     → send hardcoded velocity command to motors
  - step_simulation()   → advance physics one step
  - reset_world()       → teleport drone back to start

Step 3 — Hardcoded test script
  - No policy, no MAPPO
  - Just: read position → compute direction to goal → send velocity
  - Episode ends when drone reaches goal
  - Proves the Webots plumbing works
```

This is your proof of concept. Once a drone moves toward a goal in Webots, even with hardcoded commands, everything else becomes plugging in smarter decision-making.

---

## Phase B: MAPPO Engine Completion (Parallel or After Phase A)

**Goal: Full training pipeline working on dummy data, no Webots.**

```
Files remaining:
  - rollout_buffer.py      ← stores transitions + GAE computation
  - ppo_optimizer.py       ← clipped surrogate loss + network updates
  - mappo_trainer.py       ← orchestrates full training loop
  - observation_processor.py ← builds local obs + global state tensors
```

These are all pure Python/PyTorch — no Webots needed. Can be done on your laptop. Can be tested with dummy tensors before ever touching the simulator.

---

## Phase C: Integration

**Goal: MAPPO training loop talking to Webots.**

```
Step 1 — SwarmEnv
  - Wraps WebotsAdapter + ObservationProcessor + RewardEngine + EpisodeManager
  - Exposes reset() and step() to MAPPOTrainer
  - This is the bridge between Phases A and B

Step 2 — End-to-end test
  - MAPPOTrainer calls SwarmEnv
  - Random policy, no training yet
  - Just verify shapes are correct, no crashes, rewards flow
```

---

## Phase D: Baseline Training (Team-Scalar)

**Goal: Establish a stable baseline with centralized team-scalar value learning.**

```
Step 1 — Baseline configuration lock
  - Freeze Team-Scalar critic setup and core PPO hyperparameters
  - Keep observation and reward contracts fixed for reproducibility

Step 2 — Baseline training runs
  - Train on the simplest Webots setup first
  - Track success rate, return trends, KL, entropy, and critic loss
  - Save best/final checkpoints for ablation comparison

Step 3 — Baseline acceptance criteria
  - Stable optimization behavior across repeated seeds
  - Reliable navigation improvement over random policy
  - Checkpoints ready as reference for Phase E
```

---

## Phase E: Per-Agent Critic Ablation

**Goal: Measure whether per-agent critic values improve over the Team-Scalar baseline.**

```
Step 1 — Implement ablation branch
  - Replace team-scalar value path with per-agent value prediction
  - Keep actor, reward engine, and rollout protocol unchanged

Step 2 — Controlled comparison
  - Run matched seeds and training budgets for both variants
  - Compare convergence speed, variance, and final success metrics

Step 3 — Decision gate
  - Keep Team-Scalar if gains are inconsistent
  - Promote per-agent critic only if improvements are clear and repeatable
```

---

## Phase F: Training + Curriculum

**Goal: Policy actually learning.**

```
Stage 1 training — empty sandbox
  → one drone, no obstacles, learns basic navigation
  → success rate > 80% → advance

Stage 2 training — sparse obstacles
  → builds on Stage 1 weights
  → learns obstacle avoidance

Stage 3 training — urban grid
  → yaw rate active
  → dense environment

Stage 4 validation — OSM map
  → real city map test
```

---

## Phase G: Enhancements

```
WandB dashboard        → add during Phase F
H3 integration         → replace KNN after Stage 1 works
LLM dashboard          → after basic training works
CBF safety layer       → final enhancement
Digital Sky            → final enhancement
```