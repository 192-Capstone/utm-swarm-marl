# Data Contract: Multi-Agent Drone Swarm MARL System

**Project:** Multi-Agent Drone Swarm UTM (Unmanned Traffic Management)  
**Algorithm:** MAPPO (Multi-Agent Proximal Policy Optimization)  
**Simulator:** Webots R2025a  
**Phase:** 2- Implementation  
**Last Updated:** February 2026

---

## 1. Coordinate System & Units

|Property|Value|
|---|---|
|Coordinate System|Webots Z-Up (ENU)|
|X|East|
|Y|North|
|Z|Altitude|
|Position Units|meters (m)|
|Velocity Units|meters per second (m/s)|
|Orientation Units|radians (rad)|
|Angular Velocity Units|radians per second (rad/s)|
|Simulation Timestep|32 ms|

---

## 2. Action Space

### Design Decision

Velocity-level control is used instead of raw motor torques. The `WebotsAdapter` internally handles conversion of velocity commands to motor RPM via a PID controller. This stabilizes the learning problem significantly. The RL agent only needs to learn high-level navigation behavior, not low-level motor physics.

### Curriculum-Based Action Space

| Curriculum Stage          | Shape | Components           |
| ------------------------- | ----- | -------------------- |
| Stage 1: Empty Sandbox    | (3,)  | Vx, Vy, Vz           |
| Stage 2: Sparse Obstacles | (3,)  | Vx, Vy, Vz           |
| Stage 3: Urban Grid       | (4,)  | Vx, Vy, Vz, Yaw Rate |
| Stage 4: Validation Map   | (4,)  | Vx, Vy, Vz, Yaw Rate |

### Action Vector Definition

|Index|Component|Range|Unit|Reason|
|---|---|---|---|---|
|0|Target Vx|[-v_max, v_max]|m/s|East-West velocity|
|1|Target Vy|[-v_max, v_max]|m/s|North-South velocity|
|2|Target Vz|[-v_max, v_max]|m/s|Altitude velocity|
|3|Target Yaw Rate|[-ω_max, ω_max]|rad/s|Rotation rate (Stage 3+)|

> **Note:** Yaw rate is added progressively in Stage 3 once basic navigation is stable. This prevents the agent from being overwhelmed by a 4D action space before it has learned to navigate.

> **Real-World Transfer:** Velocity commands map directly to real drone flight controllers (PX4, ArduPilot) via MAVLink. This is not a simulation-only abstraction, it is standard in real UAV systems.

---

## 3. Local Observation Space (Per Agent)

### Design Decision

Each drone observes only its own local state. No privileged global information is available at runtime. This enforces the Decentralized Execution part of CTDE (Centralized Training, Decentralized Execution).

Neighbor information is included via K-Nearest Neighbors using relative positions and velocities. K is not hardcoded, it is determined during implementation based on swarm size and computational budget. For small swarms (≤50 drones), brute force KNN is used. H3 hexagonal indexing is reserved for large-scale UTM deployment (100+ drones).

LiDAR rays use a 3D coverage pattern, not a flat horizontal disc, to provide full spatial awareness including threats above and below.

### Observation Vector Structure

|Component|Values|Reason|
|---|---|---|
|Own position (x, y, z)|3|Self-localization|
|Own velocity (vx, vy, vz)|3|Current motion state|
|Own orientation (roll, pitch, yaw)|3|Attitude awareness|
|Relative goal vector (gx-x, gy-y, gz-z)|3|Navigation direction toward goal|
|Goal distance scalar|1|Navigation progress signal|
|LiDAR rays (14 directional)|14|3D obstacle awareness|
|K neighbors × 6 (relative pos + vel)|K×6|Drone awareness for collision avoidance|

**Fixed component total: 27 + K×6**

### LiDAR Ray Configuration (14 Rays)

|Group|Count|Direction|Purpose|
|---|---|---|---|
|Cardinal horizontal|4|Front, Back, Left, Right (flat)|Primary horizontal obstacle detection|
|Diagonal upward|4|NE, NW, SE, SW at +45°|Upper spatial awareness|
|Diagonal downward|4|NE, NW, SE, SW at -45°|Lower spatial awareness|
|Vertical|2|Straight up, Straight down|Direct vertical threat detection|

> **Reason for 3D coverage:** In a UTM system drones operate at varying altitudes. Flat horizontal rays alone are insufficient, a drone directly above or below is an equal collision risk.

### Neighbor Encoding

- **Type:** Relative (not absolute)
- **Per neighbor:** relative position (3) + relative velocity (3) = **6 values**
- **K:** flexible, decided at implementation time
- **Neighbor goal not included:** sharing destination information raises privacy concerns in a real UTM context. Relative velocity already partially encodes heading intent.
- **Absolute positions excluded:** using relative positions ensures the policy is translation-invariant and generalizes across the entire airspace

---

## 4. Global State (Centralized Critic: Training Only)

### Design Decision

The centralized critic sees the full global picture during training. This is strictly a training construct, it does not exist at runtime. The global state enables the critic to give accurate value estimates that account for all agents simultaneously, which is the core advantage of MAPPO over IPPO.

LiDAR is summarized into a 4-quadrant format per drone rather than including all 14 raw rays. This preserves directional obstacle awareness while keeping the global state manageable as swarm size scales.

### Curriculum-Based Global State

|Stage|Formula|N=20|Reason|
|---|---|---|---|
|Stage 1|N×12|240|No obstacles exist, LiDAR provides no useful information|
|Stage 2+|N×16|320|Obstacles introduced, LiDAR summary added|

### Global State Vector Structure (Stage 2+)

|Component|Per Drone|Total|Reason|
|---|---|---|---|
|Absolute position (x, y, z)|3|N×3|Full swarm spatial picture|
|Absolute velocity (vx, vy, vz)|3|N×3|Full swarm motion state|
|Orientation (roll, pitch, yaw)|3|N×3|Full swarm attitude|
|Goal position (gx, gy, gz)|3|N×3|Critic understands each drone's intent|
|4-quadrant LiDAR summary|4|N×4|Directional obstacle context per drone|

**Total: N×16**

>Note: Global state uses absolute positions unlike local observation which uses relative positions. The critic requires absolute positions to reason about the full swarm layout.

### 4-Quadrant LiDAR Summary

The 14 per-agent LiDAR rays are compressed into 4 directional quadrant minimums for the global state:

| Quadrant | Rays Included                        | Value            |
| -------- | ------------------------------------ | ---------------- |
| Front    | Front cardinal + NE upper + NE lower | min ray distance |
| Back     | Back cardinal + SW upper + SW lower  | min ray distance |
| Left     | Left cardinal + NW upper + NW lower  | min ray distance |
| Right    | Right cardinal + SE upper + SE lower | min ray distance |

> **Reason for compression:** Full N×14 LiDAR in global state = N×26 total (520 values at N=20). The 4-quadrant summary reduces this to N×4, cutting LiDAR contribution by over 70% while retaining directional awareness. A single risk score was rejected because it loses directional information critical for the critic.

---

## 5. Reward Structure

### Design Decision

No explicit coordination reward is used. Coordination emerges implicitly from the collision penalties, agents naturally learn to avoid each other. Explicit coordination rewards are better suited for formation flying tasks. For UTM, emergent coordination is more realistic and avoids reward balancing complexity.

Two additional penalties specific to quadrotor physics are included, smoothness penalty and proximity danger zone, which are essential for both simulation stability and real-world transfer.

### Reward Components

| Component                     | Type                                 | Formula                                    | Reason                                                                                  |
| ----------------------------- | ------------------------------------ | ------------------------------------------ | --------------------------------------------------------------------------------------- |
| Goal progress reward          | Continuous positive                  | Δ(goal_distance) per step                  | Encourages consistent movement toward goal, provides dense signal                       |
| Goal reached bonus            | One-time large positive              | +R_goal if distance < threshold            | Strongly reinforces mission completion                                                  |
| Obstacle collision penalty    | Large negative                       | -R_obs on contact                          | Hard safety constraint, non-negotiable                                                  |
| Drone hard collision penalty  | Large negative                       | -R_drone on contact                        | Prevents physical conflicts in shared airspace                                          |
| Step penalty                  | Small negative per step              | -R_step per timestep                       | Encourages time efficiency, discourages hovering                                        |
| Smoothness penalty            | Small continuous negative            | -λ Σ(a_t - a_{t-1})²                       | Prevents violent action jerks that break PID and damage real rotors                     |
| Proximity danger zone penalty | Continuous negative, proximity-based | -λ × max(0, (r_danger - d_ij) / r_danger)² | Accounts for real-world downwash effect, enforces safe separation before hard collision |

### Reward Component Details

#### Smoothness Penalty

```
R_smoothness = -λ_smooth × Σ(a_t - a_{t-1})²
```

- Penalizes squared difference between consecutive action outputs
- Prevents the agent from outputting [1, -1, 1] then [-1, 1, -1] causing violent jerks
- Critical for PID stability in simulation and rotor integrity in real world

#### Proximity Danger Zone Penalty

```
R_proximity = -λ_prox × max(0, (r_danger - d_ij) / r_danger)²
```

- `d_ij` = distance between drone i and drone j
- `r_danger` = danger zone radius (starting value: 1.5m, subject to tuning)
- Penalty grows quadratically as drones approach each other within the danger zone
- Zero outside danger zone
- Accounts for aerodynamic downwash — propeller wash from an overhead drone can destabilize a drone below even without physical contact

### Reward Dictionary Schema

`RewardEngine` outputs a dictionary before summing, enabling per-component logging:

```python
{
    "goal_progress":      float,  # positive
    "goal_reached":       float,  # positive, one-time
    "obstacle_collision": float,  # negative
    "drone_collision":    float,  # negative
    "step_penalty":       float,  # negative
    "smoothness_penalty": float,  # negative
    "proximity_penalty":  float,  # negative
}

total_reward = weighted_sum(reward_dict)
```

> **Note:** Reward weights are hyperparameters to be tuned during implementation. Logging each component separately is essential for diagnosing training behavior.

---

## 6. Curriculum Learning Strategy

|Stage|Environment|Action Space|Global State|Domain Randomization|
|---|---|---|---|---|
|1|Empty sandbox|(3,)|N×12|None|
|2|Sparse obstacles|(3,)|N×16|Mild sensor noise|
|3|Urban grid|(4,)|N×16|Noise + wind forces|
|4|Validation map (OSM)|(4,)|N×16|Full randomization + latency|

**Progression Trigger:** Moving average success rate > threshold over last K episodes

---

## 7. Domain Randomization

Applied progressively across curriculum stages to bridge the sim-to-real gap:

|Type|Where Applied|Method|
|---|---|---|
|Sensor noise|`ObservationProcessor`|Gaussian noise added to position, IMU, distance sensor readings|
|Wind disturbances|`WebotsAdapter`|Random external forces applied to drone physics body|
|Actuation latency|`SwarmEnv.step()`|Actions held for 1-2 timesteps before execution|

---

## 8. Pending Decisions

| Item                               | Status                                                        |
| ---------------------------------- | ------------------------------------------------------------- |
| Exact reward weight values         | **Pending**,determined during training                        |
| K (number of neighbors)            | **Pending**, determined at implementation based on swarm size |
| v_max (max velocity)               | **Pending**, determined by drone physics in Webots            |
| ω_max (max yaw rate)               | **Pending**, determined by drone physics in Webots            |
| r_danger (danger zone radius)      | Starting value 1.5m, subject to tuning                        |
| λ_smooth, λ_prox (penalty weights) | **Pending**,determined during training                        |
| Final normalization strategy       | Pending                                                       |

---

_This document reflects finalized schema decisions as of Phase 2 Review 1. Pending items will be updated as implementation progresses._

---

## 9. Verified Runtime Interfaces (G3)

This section records interfaces verified in the current repository. It distinguishes implemented runtime contracts from design targets and components whose integration has not been confirmed. Where this section differs from earlier design tables, use the implementation-specific details here when describing current runtime behavior.

### 9.1 Simulator Adapter Contract

Both `envs/pybullet_adapter.py` and `envs/webots_adapter.py` expose the following common methods:

| Method | Input | Output / behavior |
|---|---|---|
| `reset_world(spawn_positions, goal_positions, obstacles=None)` | Spawn positions, goal positions, optional obstacles | Resets the simulator world |
| `apply_actions(actions)` | Per-drone action vectors `[vx, vy, vz, yaw_rate]`; yaw rate may be optional | Applies drone controls |
| `step_simulation()` | None | Advances the simulator |
| `get_raw_state()` | None | Returns a dictionary of drone states and goals |
| `get_positions()` | None | Returns an `(N, 3)` NumPy array of positions |
| `close()` | None | Closes the adapter/simulator |

The raw-state dictionary contains a `drone_i` entry for each drone, with `position`, `velocity`, `orientation`, and `lidar` fields. A `lidar_ring` field may also be present. The top-level `goals` entry maps drone identifiers to goal positions.

Position and velocity vectors use three components. The adapters expose a common method interface, but simulator-specific behavior is not guaranteed to be identical. In particular, Webots has its own process/controller constraints, and its adapter does not apply the configured dynamics lag in the same way as PyBullet.

### 9.2 Action Contract

The current adapter control interface accepts per-drone velocity commands in the form:

`[vx, vy, vz, yaw_rate]`

The yaw-rate component may be optional in adapter handling. This describes the adapter interface; it does not establish that every training stage currently uses a different action dimensionality.

The earlier curriculum table describes a staged 3D-to-4D action-space design. Treat that as a design target unless the active configuration and training code confirm the stage-specific behavior.

### 9.3 Observation Processor Contract

`modules/observation_processor.py` consumes the raw state dictionary returned by the simulator adapter and produces four tensors:

| Output | Shape | Meaning |
|---|---|---|
| `local_fixed` | `(N, D)` | Per-drone local observation; `D` depends on the configured observation schema |
| `neighbor_states` | `(N, K, 6)` | Relative position and relative velocity for up to `K` neighbors |
| `neighbor_mask` | `(N, K)` | Boolean mask; `True` indicates a padded/empty neighbor slot |
| `global_state` | `(N, C)` | Per-drone critic state; `C` depends on critic version |

The implemented observation schemas are:

| Schema | Local observation dimension | Additional content |
|---|---:|---|
| `v1` | 27 | Base local observation with running normalization; LiDAR proximity bypasses the normalizer |
| `v2` | 42 | Fixed-scale base features, previous LiDAR proximity, and log goal distance |
| `v3` | `42 + ring_rays` | Schema v2 features plus horizontal LiDAR-ring proximities |

The base 27-feature layout consists of position (3), velocity (3), orientation (3), relative goal vector (3), goal distance (1), and 14 LiDAR proximities.

Each neighbor contributes six values: relative position `(3)` and relative velocity `(3)`. Neighbors may be filtered by the configured sensing radius, sorted by distance, and padded to the configured maximum neighbor count.

The critic state dimensions are:

- Critic version 1: `(N, 17)`, comprising position, velocity, orientation, goal position, and five compressed LiDAR proximity values.
- Critic version 2: `(N, 32)`, adding 15 features for relative goal information, goal distance, goal-radius status, settle progress, reached status, and eight LiDAR-ring sectors.

These dimensions are implementation-dependent. The configured observation schema, ring-ray count, neighbor count, and critic version must be considered when validating a configuration or checkpoint.

### 9.4 Swarm Environment Contract

`envs/swarm_env.py` coordinates the simulator adapter, observation processor, reward engine, and episode manager.

The training-facing interface is:

- `reset()` returns an observation dictionary containing `local_fixed`, `neighbor_states`, `neighbor_mask`, and `global_state`.
- `step(actions_dict)` returns `(obs_dict, rewards_dict, done, info)`.
- `actions_dict` maps drone identifiers to action vectors.
- `rewards_dict` maps drone identifiers to scalar rewards.
- `done` is the episode termination flag.
- `info` contains episode outcome and diagnostic information, including success/reason and per-drone metrics.

`SwarmEnv` selects the simulator adapter through configuration or an injected adapter. The inspected code does not directly call the separate route-planner module.

### 9.5 Route Planner Contract

`envs/route_planner.py` provides a separate grid-based route-planning interface:

- `build_grid(layout, bounds, res=0.25, inflate=0.5, min_height=0.0)` builds a planning grid.
- `plan(grid, start_xy, goal_xy, snap_dist=1.0)` returns a path of world-coordinate points or `None` if planning fails.
- `simplify(grid, points)` simplifies a path using line-of-sight checks.
- `path_length(points)` calculates path length in metres.
- `RouteFollower(waypoints, lookahead=4.5).target(pos)` selects a lookahead target from 3D waypoints.

Grid planning uses the XY plane in metres; the route follower accepts 3D waypoints. The module is exercised by dedicated tests and scripts. Direct integration into `SwarmEnv` was not found during the inspected code search, so it should not be documented as part of the active training loop without further evidence.

`modules/grid_path_planner.py` is a separate grid-connectivity and path-diagnostic utility. It should not be conflated with `envs/route_planner.py`.

### 9.6 Deployment and Dashboard Status

The inspected files `deployment/central_policy_server.py`, `deployment/client_node.py`, and `deployment/analytics_engine.py` are empty (zero bytes) in the current working tree. They do not currently establish implemented runtime contracts.

The repository search performed for dashboard entry points did not identify a clear dashboard implementation. This is a search finding, not proof that no dashboard-related material exists anywhere in the repository.

Deployment message formats, client/server APIs, analytics outputs, and dashboard data contracts remain unspecified until implemented and verified.

### 9.7 Contract Validation Notes

Before treating a contract as finalized:

1. Confirm the active configuration values for observation schema, local observation dimension, neighbor count, LiDAR ring rays, and critic version.
2. Check adapter behavior in both PyBullet and Webots when making simulator-parity claims.
3. Distinguish interfaces implemented in code from curriculum targets and proposed deployment/dashboard interfaces.
4. Update this section when runtime interfaces change, and keep the design tables above consistent with the verified implementation.


_This section supplements the Phase 2 design contract with repository interfaces inspected during the G3 data-contract review._
