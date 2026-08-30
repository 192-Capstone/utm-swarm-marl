# Data Contract: Multi-Agent Drone Swarm MARL System

**Project:** Multi-Agent Drone Swarm UTM (Unmanned Traffic Management)  
**Algorithm:** MAPPO (Multi-Agent Proximal Policy Optimization)  
**Simulator:** Webots R2025a  
**Phase:** 2- Implementation  
**Last Updated:** February 2026

---

## Changelog (Since Phase 2 Review 1)

| Change                                                             | Reason                                                                                     |
| ------------------------------------------------------------------ | ------------------------------------------------------------------------------------------ |
| Action space fixed to always (4,) — yaw masked in Stage 1-2        | Network input layer cannot change shape between stages without re-initialization           |
| Global state Stage 2+ updated from N×16 to N×17                    | 2 vertical LiDAR rays were unaccounted in original 4-quadrant compression                  |
| 5th vertical quadrant added to LiDAR compression                   | min(up_ray, down_ray) now explicitly represented                                           |
| Neighbor encoding changed from padding to attention module         | Scales naturally with H3 indexing, handles variable K without fixed K_max                  |
| Control timestep decoupled from physics timestep via action repeat | MDP consistency maintained across training and evaluation                                  |
| World size locked at 50×50m (Stage 1-3), 100×100m (Stage 4)        | Balances drone density, navigation time, and LiDAR relevance                               |
| Object pooling mandatory for ScenarioManager                       | importMFNodeFromString repeated over 5000+ episodes causes memory leak and SPS degradation |
| Phase 3 enhancement noted — LiDAR classification flags             | Distinguishes cooperative drones from unknown objects in LiDAR hits                        |

----

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

| Component                               | Values | Reason                                  |
| --------------------------------------- | ------ | --------------------------------------- |
| Own position (x, y, z)                  | 3      | Self-localization                       |
| Own velocity (vx, vy, vz)               | 3      | Current motion state                    |
| Own orientation (roll, pitch, yaw)      | 3      | Attitude awareness                      |
| Relative goal vector (gx-x, gy-y, gz-z) | 3      | Navigation direction toward goal        |
| Goal distance scalar                    | 1      | Navigation progress signal              |
| LiDAR rays (14 directional)             | 14     | 3D obstacle awareness                   |
| K neighbors × 6 (relative pos + vel)    | K×6    | Drone awareness for collision avoidance |

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

| Component                      | Per Drone | Total | Reason                                 |
| ------------------------------ | --------- | ----- | -------------------------------------- |
| Absolute position (x, y, z)    | 3         | N×3   | Full swarm spatial picture             |
| Absolute velocity (vx, vy, vz) | 3         | N×3   | Full swarm motion state                |
| Orientation (roll, pitch, yaw) | 3         | N×3   | Full swarm attitude                    |
| Goal position (gx, gy, gz)     | 3         | N×3   | Critic understands each drone's intent |
| 4-quadrant LiDAR summary       | 5         | N×4   | Directional obstacle context per drone |

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