# Capstone Review Context Pack (Paste into Claude)

## 0) How to use this file
- Use this as source context to generate slides.
- Keep the main deck visual and high-level.
- Keep deep math and tensor derivations in appendix only.
- Prioritize story: problem -> proof -> architecture -> rigor -> roadmap.

---

## 1) Project Abstract (for opening slide)
**Title:** Autonomous Low-Airspace Drone Management using MAPPO in Webots

**One-line goal:** Build a proof-of-concept UTM intelligence layer for coordinated multi-drone navigation in low-altitude urban airspace.

**What is unique:**
- Multi-agent learning (MAPPO, CTDE) for swarm coordination.
- Webots supervisor-mode control using velocity commands (not direct motor-voltage tuning).
- Research-driven pipeline with baseline-first validation and planned ablation study.

**Current status snapshot:**
- Phase A (Webots single-drone control foundation): complete.
- Phase B (PyTorch MAPPO math engine and dummy validation): complete.
- Phase C (Webots integration bridge via SwarmEnv): next active bridge.

**Suggested 30-second opener script:**
"We are building a proof-of-concept traffic intelligence system for low-altitude drone swarms. Instead of hard-coding every behavior, we train coordinated policies in simulation using MAPPO. Today we will show visual proof from Webots, then explain the architecture flow, and finally present the research rigor behind our roadmap."

---

## 2) Suggestions from Review - 1 (What went wrong last time)
**Slide bullets:**
- Last review was too implementation-heavy.
- Too many config/tensor/terminal screenshots.
- Panel asked for visual proof and system-level clarity.
- Story was fragmented across modules, hard to follow end-to-end flow.

**Speaker script:**
"In the last review, we overloaded the panel with low-level details. The core feedback was clear: show what works visually, then explain how the system connects, and keep deep technical details for Q and A."

---

## 3) Suggestions from Review - 2 (What we changed now)
**Slide bullets:**
- Visual-first opening: simulator output shown early.
- Architecture-first narrative: data flow before equations.
- File-by-file responsibilities in simple analogies.
- Validation evidence in clear table form.
- Deep technical details moved to backup section.

**Speaker script:**
"This time, we redesigned the narrative for clarity. We lead with simulator evidence, then present one clean architecture flow, then module responsibilities, validation proof, and future research plan."

---

## 4) Analysis and Critique of Research/Relevance (Real world application)
**Problem framing:**
- Low-altitude airspace is getting crowded (delivery, inspection, emergency use).
- Manual rule-based conflict handling does not scale with density.
- Need distributed autonomy with centralized training oversight.

**Why MARL MAPPO for this problem:**
- Multiple drones are coupled decision-makers.
- Policy must handle non-stationary interactions.
- CTDE supports better training signal while preserving decentralized runtime control.

**Real-world relevance claim:**
- Velocity-level command abstraction aligns with practical flight stacks and future deployment interfaces.
- Safety shaping (collision/proximity/smoothness) maps to physical flight constraints, not only simulation score.

**Speaker script:**
"This is not just a simulator exercise. The core challenge is scalable coordination under uncertainty. MAPPO with centralized training and decentralized execution gives us a realistic path to train robust swarm policies while preserving deployable per-drone autonomy."

---

## 5) High Level Design (single pipeline view)
## End-to-end flow
1. Webots provides raw per-drone state and sensor readings.
2. Observation Processor transforms raw data into learning-ready tensors.
3. Actor (Captain) outputs per-drone velocity actions.
4. Critic (Coach, training only) evaluates team state quality.
5. Reward engine computes dense and sparse feedback.
6. Rollout buffer stores transitions and computes GAE.
7. PPO optimizer updates policy and value networks.
8. Trainer orchestrates episodes, updates, checkpoints, curriculum.

**Simple analogy for panel:**
- Actor = Captain of each drone.
- Critic = Coach watching whole field during training.
- Buffer = Black box recorder.
- Optimizer = Controlled policy surgery.
- Trainer = Orchestra conductor.

**Speaker script:**
"Think of our system as a closed learning loop. Sensors create state, the Captain picks actions, the Coach evaluates team-level quality, memory stores outcomes, and PPO updates behavior safely."

---

## 6) Design Methodology
## A) Control strategy decision
- We use supervisor-mode velocity control, not direct low-level motor PID tuning.
- Action command abstraction: per-drone velocity outputs mapped through simulator control bridge.
- Rationale: stable early-stage learning and cleaner sim-to-real control abstraction.

## B) CTDE decision
- Training: critic sees richer context.
- Execution: actor runs on local observation only.
- Rationale: practical decentralized runtime with stronger centralized learning signal.

## C) Curriculum decision
- Stage-wise complexity increase.
- Stage 1 starts simpler, then obstacle complexity increases.
- Rationale: avoids overwhelming policy early in training.

## D) Baseline-first research decision
- Start with Team-Scalar critic baseline for stability.
- Then run per-agent value ablation to address credit assignment.
- Rationale: controlled scientific comparison, one architecture change at a time.

**Speaker script:**
"Our methodology prioritizes stable learning and scientific rigor. We first prove the basic loop under a stable baseline, then run targeted ablations so any performance change is attributable, not accidental."

---

## 7) Project Progress (minimum 20 percent implementation)
## Phase status
- **Phase A complete:** Webots simulator control foundation validated.
- **Phase B complete:** MAPPO engine modules implemented and dummy-validated.
- **Phase C pending bridge:** integration from MAPPO loop to Webots SwarmEnv wrapper.

## Completed technical assets
- Observation processing pipeline.
- Actor and critic networks.
- Rollout buffer and GAE.
- PPO update engine.
- Trainer orchestration skeleton.
- Reward engine and core tests.

## Suggested progress message
"We are substantially beyond design stage. Core simulator foundation and learning engine are implemented independently; integration is the current bridge milestone."

---

## 8) Data Preprocessing and Base Model Implementation (tabular context)

## 8.1 Action Space Contract
| Space | Shape | Meaning | Bounds | Why this choice |
|---|---|---|---|---|
| Drone action | (N, 3) in presentation abstraction | Continuous velocity command [vx, vy, vz] per drone | normalized bounded range (policy output constrained) | Velocity-level control is more stable for early MARL and aligns with practical control interfaces |

Note for technical appendix:
- Internally, architecture may support an additional yaw-related channel in later curriculum stages, but main story should emphasize 3D velocity control for current review clarity.

## 8.2 Actor Local Observation Contract
| Component | Shape contribution | Why included |
|---|---|---|
| Own position, velocity, orientation | core self-state | Needed for local control and stabilization |
| Relative goal vector + distance | goal context | Drives progress-oriented behavior |
| 14-ray lidar | obstacle awareness | Safety and local collision avoidance |
| **Total** | **(N, 27)** | Fixed local representation for decentralized actor |

## 8.3 Neighbor Context Contract
| Space | Shape | Meaning | Why this design |
|---|---|---|---|
| Neighbor states | (N, K, 6) | Relative neighbor position and velocity for K nearest drones | Handles variable swarm interaction context |
| Neighbor mask | (N, K) | Marks valid/padded neighbor slots | Supports fixed tensor batching |

## 8.4 Critic Global State Contract
| Space | Shape | Meaning | Why this design |
|---|---|---|---|
| Global state | (N, 17) | Compressed centralized context for value estimation | Shared encoder + pooling gives permutation-invariant team valuation |

## 8.5 Why custom data, not Hugging Face/static datasets
- Our data is generated online from a closed-loop simulator.
- Each action changes future observations, so static i.i.d. data assumptions fail.
- State schema is task-specific (custom lidar layout, relative goals, multi-agent coupling).
- External pre-collected datasets do not carry this exact interaction contract.

**Panel-ready line:**
"We did not choose these values arbitrarily. They come from what the simulator can sense, what the controller must decide, and what the policy needs to optimize safely."

---

## 9) File-by-file architecture breakdown (core files asked by panel)

## 9.1 modules/observation_processor.py
**Role:** Translator from raw simulator dictionary to normalized learning tensors.

**Input:** raw per-drone dict (position, velocity, orientation, lidar, goals).

**Output:**
- local_fixed (N, 27)
- neighbor_states (N, K, 6)
- neighbor_mask (N, K)
- global_state (N, 17)

**Why important:** prevents training instability from heterogeneous sensor scales.

**Analogy:** Air-traffic translator converting noisy radio chatter into structured control boards.

## 9.2 training/networks/actor_network.py
**Role:** Decentralized policy (Captain) that chooses actions from local context.

**Input:** local_fixed + neighbor_states + neighbor_mask.

**Output:** action distribution parameters, sampled actions, log-probs.

**Design highlight:** attention over neighbors compresses variable K-neighbor context into fixed latent representation.

**Analogy:** Pilot prioritizing closest relevant traffic instead of reading every radar blip equally.

## 9.3 training/networks/critic_network.py
**Role:** Centralized value estimator (Coach, training only).

**Input:** global_state (N, 17).

**Output:** current baseline is team-scalar value estimate.

**Design highlight:** shared per-drone encoding + pooling for permutation invariance and scalable team context handling.

**Analogy:** Coach evaluating overall team position, not individual joystick commands.

## 9.4 training/rollout_buffer.py
**Role:** Transition memory for PPO.

**Input:** obs/action/logprob/reward/value/done across timesteps and agents.

**Output:** trajectories, advantages, returns via GAE.

**Analogy:** Flight data recorder used after mission to improve future behavior.

## 9.5 training/ppo_optimizer.py
**Role:** Safe policy update engine.

**Input:** minibatches from rollout buffer.

**Output:** updated actor and critic weights + health metrics.

**Key metrics tracked:** actor loss, critic loss, entropy, clip fraction, approx KL.

**Analogy:** Surgical tuning that avoids over-correcting policy behavior.

## 9.6 training/mappo_trainer.py
**Role:** Orchestrator for full loop.

**Input:** environment rollouts and optimizer pipeline.

**Output:** training progression, checkpoints, logs, phase/stage progression.

**Analogy:** Conductor ensuring all modules operate in the correct sequence.

---

## 10) Validation and Tensor Correctness (must-have defense section)

## 10.1 How tensor correctness is validated
- Unit-level shape and behavior tests for actor and reward pipeline.
- Dummy-data end-to-end rollout to verify module interfaces before simulator coupling.
- PPO health signals used as sanity checks:
  - KL divergence should remain controlled.
  - Clip fraction should remain in healthy band.
  - Losses should not explode.

## 10.2 Why this proves contract correctness
- If shapes are wrong: unit tests fail at module boundaries.
- If interfaces are inconsistent: dummy rollout fails in buffer/optimizer path.
- If policy update is unstable: KL/clip diagnostics reveal it early.

## 10.3 Validation message to panel
"We validate in layers: module tests, end-to-end dummy rollouts, and optimizer health diagnostics. This is why we are confident the tensor contracts are correct before full simulator integration."

---

## 11) Future Roadmap and Academic Rigor

## 11.1 Immediate roadmap
- Phase C: complete MAPPO <-> SwarmEnv <-> Webots integration bridge.
- Phase D: train Team-Scalar critic baseline and stabilize metrics.
- Phase E: run per-agent critic ablation study under controlled settings.

## 11.2 Ablation design (credit assignment research question)
**Baseline:** Team-scalar critic value.

**Upgrade variant:** Per-agent values to improve credit assignment granularity.

**Comparison protocol:**
- Matched seeds.
- Matched training budget.
- Same reward/config except critic output design.
- Compare convergence speed, variance, success rate, collision metrics.

**Scientific claim:**
- Keep baseline if gains are inconsistent.
- Promote per-agent critic only if improvements are clear and repeatable.

---

## 12) Known Gaps (state honestly in review)
- Integration wrappers are the active bridge milestone (Phase C focus).
- Some tests are stronger than others; test coverage expansion is part of ongoing rigor.
- Hyperparameter tuning and curriculum transition gates remain active optimization work.

Use this as strength, not weakness:
"We are transparent about what is complete, what is validated, and what is currently being integrated."

---

## 13) Individual Contribution (fill with your team names)
Use this matrix format:

| Member | Owned modules | Delivered artifact | Current responsibility |
|---|---|---|---|
| Member A | Webots adapter + env bridge | Stable state/action bridge prototype | Phase C integration |
| Member B | Actor/Critic networks | Core policy/value implementation | Training stabilization |
| Member C | Buffer/Optimizer/Trainer | PPO update loop and logging | Baseline experiment runs |
| Member D | Reward/Episode/Validation | Reward logic and test scaffolding | Metrics and ablation setup |

Narration tip:
- Show ownership by deliverable, not by effort statement.

---

## 14) References (suggested)
- MAPPO paper and PPO foundational paper.
- CTDE references for multi-agent learning.
- Webots official documentation.
- Project internal docs:
  - CONTEXT.md
  - docs/data_contract.md
  - core module source files and test files.

---

## 15) Slide-by-slide script map for your template

### Slide: Abstract
- Use Section 1.

### Slide: Suggestions from Review - 1
- Use Section 2.

### Slide: Suggestions from Review - 2
- Use Section 3.

### Slide: Analysis and Critique of Research/Relevance - Application in Real world
- Use Section 4.

### Slide: High Level Design
- Use Section 5.

### Slide: Design Methodology
- Use Section 6.

### Slide: Project Progress (Minimum 20 percent implementation)
- Use Section 7 + Section 10 (short proof lines).

### Slide: Data Preprocessing and Base Model Implementation
- Use Section 8 + Section 9.

### Slide: Individual Contribution
- Use Section 13.

### Slide: References
- Use Section 14.

### Backup/Q and A slide
- Use Section 10 + Section 11 + Section 12.

---

## 16) Panel Q and A rapid answers (one-liners)

**Q: Why custom tensors and not external data?**
A: Our state-action data is generated online in a closed-loop simulator; static datasets do not capture this interaction contract.

**Q: How do you know tensor dimensions are correct?**
A: We validate at three levels: unit shape tests, end-to-end dummy rollouts, and PPO stability signals like KL and clip fraction.

**Q: Why Team-Scalar first?**
A: It is the most stable baseline for integration; then we run controlled ablation to test whether per-agent values add measurable gains.

**Q: Is this just simulation theatre?**
A: No, we use control abstractions and safety terms aligned with real flight constraints, and we validate systematically before scaling complexity.

**Q: What is your strongest evidence today?**
A: Working simulator foundation, completed MAPPO engine modules, and layered validation strategy ready for full integration.

---

## 17) Final speaking advice for tomorrow
- Open with simulator visual proof before architecture.
- Keep one key message per slide.
- Do not say tensor dimensions unless asked; point to table if needed.
- Emphasize architecture connectivity, validation discipline, and ablation rigor.
- End with confidence: clear completed milestones + explicit next gate.
