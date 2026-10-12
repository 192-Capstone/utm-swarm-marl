# Stage-1 baseline evaluation

*October 3rd, 2026. Evaluation of the frozen `spring-vortex-58` checkpoint.*

Before adding obstacles, more agents or Webots, we wanted to know how good our current model actually is. During training it reported 100% success, but that was from 20 evaluation episodes under the exact conditions it was trained on. So we ran it through a much larger set of tests, without any retraining, to find out what it can and can't do.

Short version: the model is very reliable at flying to a goal and holding position, and it handles much longer distances and more drones than it was trained on. But it has **not** learned to avoid other drones. It crashes less than a simple controller in head-on and crossing situations, but that turns out to be a side effect of how it flies. When we set up situations where it has to react, it gets hit every single time.

---

## 1. Setup

**Model.** `best_eval_policy.pth` from W&B run `cje4yxrd` (`spring-vortex-58`), saved at episode 300. Trained with 3 drones in an empty PyBullet world, goals 1-2 m away, collision physics on. We downloaded it from W&B and checked it against W&B's checksum (MD5 `w+8tutcvQiWoFnepwRuc6w==`, 344,788 bytes).

**How we ran it.**
- Deterministic actions (the policy's mean action, no sampling noise).
- The observation normalizer restored from the checkpoint and frozen, so inputs are scaled exactly as in training.
- 5 seeds (0-4) for every test. Same seeds, same scenarios.
- 32 ms control step, success rule unchanged: a drone counts as done when it stays within 0.5 m of its goal, below 0.15 m/s, for 31 consecutive steps. An episode is a success when every drone has done this.
- Every test was run twice, on two separate days. All 78 result files came out byte-for-byte identical, so the numbers below are fully reproducible.

**Baseline to compare against.** A P-controller: `velocity = clip(3 × (goal − position), ±2 m/s)`. It flies straight at its goal and slows down as it gets close. It has no learning and no idea other drones exist. Every comparison uses exactly the same starting positions and goals for both controllers. We confirmed this with a per-episode scenario hash.

**Collision physics.** PyBullet treats our drones as 0.3 m boxes and pushes them apart when they touch. Our collision check only fires when centres come within 0.2 m, which can't happen while physics is pushing them apart. So with physics on, crashes are hidden. All safety numbers below come from runs with collision physics **off**, unless stated otherwise. Section 3 shows why this matters.

**Scenarios.**

| Scenario | What happens | Typical flight |
|---|---|---|
| Random (normal task) | Same as training. Random spawns at least 5 m apart, each goal 1-2 m from its own spawn | 1.6 m |
| Head-on | Drones on a circle, each flies to the exact opposite point. All paths meet at the centre at the same time | 7.9 m |
| Crossing | Like head-on but with random jitter in angle and radius | 8.3 m |
| Swap | Random spawns, each drone flies to another drone's spawn point | 11.6-12 m |

---

## 2. Test 1: the normal task

3 drones, random 1-2 m goals, 500 episodes.

| | Success | Collisions | Closest two drones got | Final error (mean / p95) | Final speed | Time to settle |
|---|---|---|---|---|---|---|
| MAPPO | **100%** [99.2-100] | 0 | 2.69 m | 5.4 cm / 8.8 cm | 0.004 m/s | 2.2 s |
| P-controller | 100% [99.2-100] | 0 | 2.71 m | 0.2 cm / 0.2 cm | 0.005 m/s | 2.2 s |

*Brackets are 95% confidence intervals.*

**Observations**
- 500 out of 500. The training-time result holds up with a proper sample size.
- The drones never got closer than 2.7 m to each other. In this task nobody is ever in anyone's way, so it tells us nothing about avoidance, and the P-controller ties.
- MAPPO enters the goal radius after about 0.5 s and latches as settled at about 2.2 s. It's held almost perfectly still at the end (0.004 m/s).
- The P-controller parks closer to the exact centre (2 mm vs 5-9 cm). Its command is proportional to the remaining error, so it keeps closing the gap until it's zero. MAPPO only has to get inside the 0.5 m radius, and the reward for closing the last few centimetres is tiny (about +0.45 out of a few hundred). So it brakes once it's inside and stays there. 9 cm is well within the radius and fine for our purposes.

---

## 3. Test 2: forced conflicts

3 drones, 100 episodes per scenario, physics off.

| Scenario | Controller | Success | Collisions | Episodes closer than 0.3 m | Closest approach |
|---|---|---|---|---|---|
| Head-on | MAPPO | **84%** [75.6-89.9] | **16%** [10.1-24.4] | 27% | 0.108 m |
| | P-controller | 51% [41.3-60.6] | 49% [39.4-58.7] | 78% | 0.089 m |
| Crossing | MAPPO | **94%** [87.5-97.2] | **6%** [2.8-12.5] | 15% | 0.143 m |
| | P-controller | 46% [36.6-55.7] | 54% [44.3-63.4] | 70% | 0.067 m |
| Swap | MAPPO | 100% | 0% | 1% | 0.212 m |
| | P-controller | 100% | 0% | 0% | 0.376 m |

The same scenarios with physics **on**:

| Scenario | Controller | Success | Logged collisions | Bodies actually touched |
|---|---|---|---|---|
| Head-on | MAPPO | 100% | 0 | 41% |
| | P-controller | 100% | 0 | **96%** |
| Crossing | MAPPO | 100% | 0 | 28% |
| | P-controller | 100% | 0 | **80%** |
| Swap | MAPPO | 100% | 0 | 2% |
| | P-controller | 100% | 0 | 0% |

**Observations**
- With physics off, MAPPO crashes about 3× less than the P-controller head-on and about 9× less when crossing. The confidence intervals don't overlap, so the difference is real.
- 16% head-on is still a lot of crashes for anything we'd call safe.
- Swap with 3 drones didn't produce real conflicts. With random positions, two drones rarely reach the same point at the same time, so both controllers score 100%.
- With physics on, the P-controller looks perfect (100% success, 0 collisions) while its drones physically crashed in 96% of head-on episodes. PyBullet's contact response was quietly doing the avoiding. Any safety number measured with physics on is meaningless for us.

---

## 4. Test 3: longer distances

MAPPO, 3 drones, random scenario with goals placed further away, physics off.

| Goal distance | Average flight | Success | Collisions | Time to settle | Final error (p95) | Clipped inputs |
|---|---|---|---|---|---|---|
| 1-2 m (trained) | 1.6 m | 100% | 0% | 2.2 s | 8.8 cm | 0% |
| 2.5-5 m | 3.9 m | 100% | 0% | 3.2 s | 8.5 cm | 0.6% |
| 5-10 m | 7.6 m | 100% | 0% | 5.0 s | 7.9 cm | 2.7% |
| 10-20 m | 13.9 m | 98% [93.0-99.4] | 2% | 7.6 s | 9.6 cm | 4.7% |

**Observations**
- The model was only ever trained on 1-2 m goals, yet it flies 20 m and settles just as precisely. It learned a general "go there and stop" behaviour, not something tied to short hops.
- The 2 failures at 10-20 m were collisions, not navigation problems. Longer flights cross each other more often, and the model doesn't avoid (see Test 5).
- We were worried the normalizer would break at long range. It was fitted on 1-2 m distances (spread about 0.3 m), so far goals get clipped. In practice under 5% of inputs were clipped, and it didn't hurt.
- **No need for the 2 → 20 m curriculum retrain** we were planning.

---

## 5. Test 4: more drones

The same model, no retraining. It only ever sees its 5 nearest neighbours, so with 8 and 10 drones some drones are invisible to each one.

Random task (1-2 m goals), 200 episodes each:

| Drones | MAPPO success | P-controller success | Closest two drones got |
|---|---|---|---|
| 1 | 100% | 100% | - |
| 3 | 100% | 100% | 2.69 m |
| 5 | 100% | 100% | 1.88 m |
| 8 | 100% | 100% | 2.12 m |
| 10 | 100% | 100% | 1.78 m |

Swap, 100 episodes each:

| Drones | MAPPO success | MAPPO collisions | P-controller success | P-controller collisions |
|---|---|---|---|---|
| 3 | 100% | 0% | 100% | 0% |
| 5 | 99% | 1% | 99% | 1% |
| 8 | 99% | 1% | 98% | 2% |
| 10 | 95% [88.8-97.8] | 5% | 97% [91.5-99.0] | 3% |

**Observations**
- Nothing breaks with more drones. One drone (no neighbours at all) works, and so do 8 and 10 drones where the model can only see 5 of them.
- The random task still doesn't bring drones close, even at 10 drones (closest 1.8 m), so it only shows the model doesn't get confused by extra neighbours.
- In crowded swaps MAPPO is no better than the P-controller. At 10 drones it's slightly worse, though the difference is within noise.

---

## 6. Test 5: is it actually avoiding other drones?

MAPPO beat the P-controller in head-on and crossing, so we checked why. We ran MAPPO again on the exact same scenarios, but wiped its view of other drones: every neighbour slot marked as empty, as if it were flying alone. If it was using its neighbours to avoid them, the blind version should crash a lot more.

| Scenario | MAPPO | MAPPO, blind | P-controller |
|---|---|---|---|
| Head-on collisions | 16% [10.1-24.4] | 11% [6.3-18.6] | 49% |
| Crossing collisions | 6% [2.8-12.5] | 7% [3.4-13.7] | 54% |
| Swap (10 drones) collisions | 5% | 5% | 3% |
| Normal task success | 100% | 100% | 100% |
| Path straightness (head-on) | 0.944 | 0.937 | 0.960 |

Which episodes crashed, on identical scenarios:

| Scenario | Both crashed | Only normal crashed | Only blind crashed |
|---|---|---|---|
| Head-on | 10 | 6 | 1 |
| Crossing | 6 | 0 | 1 |
| Swap (10) | 4 | 1 | 1 |

**Observations**
- Taking away its view of other drones changes nothing. If anything the blind version crashed slightly less head-on, though that's within noise.
- Both versions mostly crash in the same episodes. Those are situations where its normal flight path happens to run into someone, whether it can see them or not.
- So the advantage over the P-controller comes from how MAPPO flies, not from reacting to other drones. Its paths are slightly curved (about 94% as direct as a straight line) and have their own speed profile, so drones usually reach the middle at slightly different times or places. The P-controller flies dead-straight at identical speeds, so in a symmetric head-on everyone arrives at the centre at the same instant.
- The attention module that reads the neighbour states isn't contributing anything useful right now.

**Why it never learned this.** Training never gave it a reason to:
1. Spawns were at least 5 m apart and goals were only 1-2 m away, so drones essentially never met.
2. Training ran with collision physics on, so even when bodies touched, PyBullet pushed them apart before the -50 collision penalty could fire.

With no situations that needed avoiding, and no penalty when it failed to, there was nothing to learn from.

---

## 7. Test 6: two-drone interaction tests

Test 5 had one weakness. In head-on and crossing, MAPPO's own curved paths often miss each other anyway, so a drone that ignores its neighbours can still get lucky. We also had one observation pointing the other way: in the PyBullet GUI, when we dragged two drones close together, they seemed to move apart.

So we built three tests where a collision is guaranteed **unless the drone actively reacts**. Each uses two drones. Drone 0 is the one under test. Drone 1 is scripted and ignores everyone. 200 episodes each, physics off, same scenarios for all three controllers.

- **Intruder.** Drone 0 hovers on its goal. Drone 1 flies straight through that exact point at 0.5-2 m/s (random per episode) and carries on. It starts 5 m away, so even at top speed there are 2.5 s to react.
- **Blocker.** Drone 1 hovers exactly on the path drone 0 would fly if it ignored other drones, 40-60% of the way along a 3-6 m trip. We record that path first, with drone 0's neighbour view switched off, so its curved flying style can't miss the blocker by accident.
- **Neighbour.** Drone 0 hovers on its goal, with drone 1 frozen 0.3-0.5 m away. This is the controlled version of the GUI drag test.

| Test | Controller | Drone 0 collided | Drone 0 moved more than 10 cm | Other result |
|---|---|---|---|---|
| Intruder | MAPPO | **100%** [98.1-100] | 0% | max movement 5 cm (median) |
| | MAPPO, blind | 100% | 0% | max movement 4 cm |
| | P-controller | 100% | 0% | doesn't move |
| Blocker | MAPPO | **100%** [98.1-100] | - | reached goal 0%, strayed from its path 2.6 cm (median) |
| | MAPPO, blind | 100% | - | reached goal 0%, strayed 0 cm |
| | P-controller | 100% | - | reached goal 0%, strayed 0 cm |
| Neighbour | MAPPO | 0% | **0%** | separation changed by +1 mm on average |
| | MAPPO, blind | 0% | 0% | +2 mm |
| | P-controller | 0% | 0% | 0 mm |

Intruder collisions by speed, MAPPO:

| Intruder speed | Episodes | Collided |
|---|---|---|
| 0.5-1 m/s | 73 | 100% |
| 1-1.5 m/s | 58 | 100% |
| 1.5-2 m/s | 69 | 100% |

**Observations**
- The model doesn't react to another drone at all. It gets hit by every intruder, even slow ones that give it 10 seconds of warning. It flies straight into a drone parked on its path. It doesn't budge with a drone 0.3 m away.
- The few centimetres of movement in the MAPPO rows are just its normal hover. It stops about 5 cm from the exact goal centre (see Test 1), and the blind version shows the same thing.
- Normal and blind MAPPO are indistinguishable in all three tests, which confirms Test 5 much more sharply.
- So what we saw in the GUI wasn't learned avoidance. The GUI runs with collision physics on, so PyBullet pushes overlapping boxes apart on its own. A drone that's been dragged off its spot also flies back to its own goal, which can look like moving away.

---

## 8. Summary

| Question | Answer |
|---|---|
| Does it reliably fly to the goal and hold still? | Yes. 100% over 500 episodes, settles in about 2 s, stops within 9 cm |
| Does it work beyond the 1-2 m it was trained on? | Yes. 100% up to 10 m, 98% up to 20 m (the failures were collisions) |
| Does it work with more drones? | Yes, nothing breaks up to 10 drones |
| Does it avoid other drones? | **No.** It crashes 3-9× less than a P-controller in head-on and crossing, but that comes from its flying style. When a collision is guaranteed unless it reacts, it gets hit 100% of the time, exactly like the blind version |
| Was the "moving apart" in the GUI learned behaviour? | No. That was PyBullet's collision physics, or a dragged drone flying back to its own goal |
| Can we trust safety numbers measured with physics on? | No. PyBullet hides collisions (96% hidden contact for the P-controller head-on) |

For the report this means we can claim reliable goal-reaching and settling, generalization to longer distances and larger swarms, and a lower crash rate than a straight-line controller. We **can't** claim learned deconfliction, and the interaction tests show the current model has none at all. The blanking and interaction tests are worth including as they are, since they show we checked the claim properly instead of assuming it.

**Things to keep in mind**
- All of this is in an idealized PyBullet setup: the commanded velocity is applied instantly, there's no gravity, no motor lag and no wind. That's why the P-controller is so hard to beat on plain navigation.
- The collision check (0.2 m between centres) is tighter than the drone bodies (0.3 m), so "closer than 0.3 m" is a better measure of real contact than "collision".
- The P-controller clips each axis to ±2 m/s separately, so on long trips it flies at an angle first and then straightens out. Its paths aren't perfectly straight, which is part of why it "only" crashes about half the time head-on.
- Only the empty world (Stage 1) has been tested. There are no obstacles yet.

**Reproducing.** Results are in `eval_results/v1/`, one folder per test with `episodes.csv`, `summary.json` and (for the multi-drone tests) `agents.csv`. To regenerate everything (about 20 minutes on a laptop CPU):

```bash
bash scripts/run_eval_suite_v1.sh phase1   # normal task + conflicts
bash scripts/run_eval_suite_v1.sh phase2   # distances
bash scripts/run_eval_suite_v1.sh phase3   # more drones
bash scripts/run_eval_suite_v1.sh phase4   # blanking test
bash scripts/run_eval_suite_v1.sh phase5   # two-drone interaction tests
python eval_report.py --out eval_results/v1/report.md
```

---

## Immediate next step: a deconfliction training run

Continue training from `checkpoint_final.pth` (episode 500 of `spring-vortex-58`, which includes the critic and optimizer state) with two changes, and nothing else:

1. **Collision physics off during training**, so a crash is a real crash and the -50 penalty actually fires.
2. **Forced conflicts in the training mix.** For example, half the episodes from the normal random task and half from head-on, crossing and swap. That way drones regularly meet and the neighbour input finally matters.
3. **A sensing radius and scaled neighbour inputs.** Each drone only sees drones within a set distance (starting with 30 m), and neighbour positions and velocities are scaled to a fixed range instead of raw metres. Since the model ignores its neighbour inputs today, this is the cheapest moment to get them right. The reasoning is in `docs/neighbour_information_design.md`.

Reward, PPO settings and the 3-drone setup stay the same, so any change in behaviour can be traced to these changes. Train on 3 seeds, then rerun this exact evaluation, including the blanking and interaction tests.

What success looks like:
- head-on collisions well under 5% (from 16%), crossing close to 0%
- the normal task still at 100%
- in the intruder and blocker tests, drone 0 actually moves out of the way or goes around, instead of 100% collisions
- **blind MAPPO crashing clearly more than normal MAPPO.** That's the proof the model is actually using the other drones.

The code work for this is small: move the scenario generators from `record_rollout.py` into the environment, add config options (collision physics on/off, scenario mix, sensing radius), and apply the radius and scaling where the neighbour inputs are built. The trainer itself barely changes.
