# Birds: moving non-cooperative obstacles, shipped policy without retraining (stages 1-2)

Branch `feature/birds`. Everything is opt-in (`environment.birds`, scenario key `birds:`, `eval_suite.py --birds N`); with birds off every output is byte-identical to before.

## Main findings (read these first)

1. **The policy mostly does not avoid birds, because it hardly sees them.** In the runs, a bird within 6 m was detected (at least one LiDAR ray on it) in only about 3-11 % of the (drone, bird) steps: the 32-ray ring is horizontal and the 14-ray fan is sparse, against a 0.12 m sphere. Single drones survive largely because birds pass close but miss (every single-drone crash was a pass within 0.5 m; passes at 0.5-1 m never crashed), not because they dodge. This is the main finding, and the reason the first thing to try next is stage 3a: feeding tracked birds to the policy as non-cooperative neighbours.
2. **The crowd cells are a deliberate worst case, not typical traffic.** Birds are timed to reach a drone's path when that drone does, and in the 8-drone head-on test all drones meet at the centre, so a bird aimed there crosses the whole cluster (100 % encounter rate, median closest approach 0.26-0.35 m). The 58 % (2 birds) and 34 % (4 birds) success must not be read as what the policy would do in ordinary traffic with birds around.
3. **Stage 3a (an onboard detector feeds tracked birds to the policy as neighbours, no retraining): the tracking works, the crowd result does not move.** A realistic detector tracked 46-69 % of the nearby (drone, bird) pairs (the LiDAR alone: 3-13 %), yet crowd success stays at 54 % / 44 % (2 / 4 birds; 58 % / 34 % without), none of it significant, and even an oracle that knows every bird within 8 m only reaches 62 % / 50 %. The policy does react to a bird in a neighbour slot (probe in the Stage 3a section), so perception is not the only bottleneck: in the crowd the avoidance it has is not enough, and the next step is fine-tuning (3b), not more sensing. See "Stage 3a" at the end.
4. **Stage 3a+ (a classical bird guard on the tracked birds, like goal-hold): this works where the policy alone did not, and what limits it now is the tracking.** In the worst-case crowd the guard lifts 4-bird success from 17 to 30-31 of 50 (Holm 0.022 / 0.005) and 2-bird from 29 to 36 (not significant after correction), at 2-4 % extra path; on perfect tracks it reaches 45 and 42 of 50. In a fairer crowd (birds crossing at random times, not timed to the drones) the same guard helps less (37 to 41-45 of 50 with 4 birds) and none of it is significant after correction. The realistic detector tracks only about 55 % of nearby pairs in the worst-case crowd, so better tracks (3c: shared or radar tracks) look more valuable now than fine-tuning (3b). See "Stage 3a+" at the end.
5. **Which sensor the guard needs (sweep of detector range, field of view, detection probability and latency):** coverage matters most. A 360-degree field of view takes the track rate from 53 % / 39 % to 99 % and success from 30 to 33 (worst-case crowd, 4 birds) and from 41 to 46 (fair crowd); 4 m of range is clearly worse (24); detection probability (0.3-1.0) and latency (0-0.29 s) change little. The best realistic variants (360 degrees, 12 m) reach 34 and 47 of 50, against 42 and 48 for the oracle. Differences between variants are not significant at 50 episodes. See "Stage 3a+ sensor sensitivity" at the end.

## Setup

**Bird model** (`envs/birds.py`). A bird is a sphere (radius 0.12 m) on a pre-computed path; it never reacts to drones. Speed 1-3 m/s per bird, modes `cross` (straight line), `turn` (gentle random turns, heading rate <= 0.35 rad/s) and `circle` (radius 2-4 m), small vertical speed (|vz| <= 0.3 m/s), reflected at the map bounds and the altitude band (the paths' heights +- 0.5 m).

**Encounter timing.** Each bird is aimed at a random point 20-100 % along one drone's path (its straight start-goal line, or the route its follower flies: the follower's waypoints in routed/scenario runs) and passes that point at the time the drone is expected there: `departure + arc length / 2 m/s + U(-1, +1) s`. So the bird is "spawned" at aim point - velocity x time (it is generated backwards from the aim point), instead of arriving long after a 5 s flight is over.

**Simulation.** PyBullet kinematic spheres, moved once per control step (the LiDAR and the crash check only sample at control steps), seen by the 14-ray fan and the 32-ray ring like any body. A drone within `0.15 m (drone body) + 0.12 m (bird) = 0.27 m` of a bird has crashed, reason `'bird'` (`info['bird_collision']`, `bird_distance`, `bird_seen`). Birds draw from their own generator, seeded with `[birds.seed, hash(spawns, paths)]`: the same episode has the same drones, lag and sensor noise with and without birds, and bird i is the same bird whatever the count. PyBullet only (Webots adapters refuse the setting).

**Scale note.** Real birds fly 8-20 m/s. Our drones fly at 2 m/s with a 6 m LiDAR and 0.10-0.29 s lag, so a real-speed head-on bird (8 m/s closing speed + 2 m/s = 10 m/s: under 0.6 s from first sight at 6 m) cannot be dodged reactively by any controller. Birds are therefore scaled to the simulation (1-3 m/s).

**Flight log and scenarios.** `utm/scenario.py` accepts `birds: {count, speed, radius, modes, ...}` (validated; PyBullet only; `run_scenario.py` seeds it with the scenario seed and bounds it with the map). Recordings get an additive top-level `"birds": {"radius", "pos": [[x,y,z] per bird] per frame}` key, one entry per `frames.t` entry; the schema stays `utm-flight-log/2`, no existing field changes, and crash events of bird crashes carry `"cause": "bird"`. Checked end to end on a copy of `ecity_failures.yaml` (see "Scenario run" below).

**Evaluation.** Shipped policy `stage2_r1_crowd_rl_r7v3ft_s0/checkpoint_final.pth`, goal-hold on, `config_stage2_round7_eval_goalhold.yaml` (lag and sensor noise on), collision physics off, validation seeds 20-24 only (5 seeds x 10 episodes = 50 per cell). Driver `scripts/eval_birds.py`; statistics from `eval_stats.py` (95 % Wilson; paired exact McNemar against the same family's no-bird cell on identical episodes, checked by scenario id; Holm per family). Raw per-episode CSVs are in `eval_results/birds/` (not in git). Total run time on the laptop: 17 min.

* **single_noobs**: 1 drone, 7-10 m flight, no static obstacle. **single_obs**: the same with a building on the path. 0 / 1 / 2 / 4 birds.
* **speed**: single_noobs with 2 birds at fixed 1 / 2 / 3 m/s.
* **crowd**: 8 drones head-on, no static obstacles, 0 / 2 / 4 birds.

Encounter = a bird came within 1.5 m (centre to centre) of a drone. Detection = of the (drone, bird) pairs within 6 m in a step, the fraction where at least one ray hit that bird.

## Results

| cell | success [95 % CI] | bird crash | other crash | timeout | encounter < 1.5 m [95 % CI] | closest bird dist p10 / median / p90 (m) | lost / gained vs no birds, p (Holm) | LiDAR detection (0-2 / 2-4 / 4-6 m) |
|---|---|---|---|---|---|---|---|---|
| single_noobs, 0 birds | 50/50 = 100 % [93-100] | 0 | 0 | 0 | - | - | - | - |
| single_noobs, 1 bird | 48/50 = 96 % [87-99] | 2 | 0 | 0 | 28/50 [42-69 %] | 0.49 / 1.37 / 3.04 | 2 / 0, p 1.0 | 3.3 % (6 / 2 / 3 %) |
| single_noobs, 2 birds | 48/50 = 96 % [87-99] | 2 | 0 | 0 | 36/50 [58-83 %] | 0.47 / 1.01 / 1.85 | 2 / 0, p 1.0 | 3.0 % (6 / 3 / 2 %) |
| single_noobs, 4 birds | 48/50 = 96 % [87-99] | 2 | 0 | 0 | 44/50 [76-94 %] | 0.44 / 0.74 / 1.59 | 2 / 0, p 1.0 | 2.9 % (7 / 2 / 2 %) |
| single_obs, 0 birds | 46/50 = 92 % [81-97] | 0 | 0 | 4 | - | - | - | - |
| single_obs, 1 bird | 46/50 = 92 % [81-97] | 1 | 0 | 3 | 25/50 [37-63 %] | 0.41 / 1.40 / 3.29 | 0 / 0, p 1.0 | 7.6 % (27 / 6 / 2 %) |
| single_obs, 2 birds | 46/50 = 92 % [81-97] | 0 | 0 | 4 | 36/50 [58-83 %] | 0.39 / 1.06 / 2.46 | 0 / 0, p 1.0 | 6.1 % (22 / 5 / 2 %) |
| single_obs, 4 birds | 45/50 = 90 % [79-96] | 2 | 0 | 3 | 45/50 [79-96 %] | 0.34 / 0.75 / 1.50 | 1 / 0, p 1.0 | 6.8 % (24 / 5 / 2 %) |
| 2 birds at 1 m/s | 49/50 = 98 % [90-100] | 1 | 0 | 0 | 43/50 [74-93 %] | 0.49 / 0.92 / 1.74 | 1 / 0, p 1.0 | 4.0 % (8 / 3 / 3 %) |
| 2 birds at 2 m/s | 50/50 = 100 % [93-100] | 0 | 0 | 0 | 35/50 [56-81 %] | 0.59 / 1.07 / 2.00 | 0 / 0, p 1.0 | 3.5 % (8 / 3 / 2 %) |
| 2 birds at 3 m/s | 47/50 = 94 % [84-98] | 3 | 0 | 0 | 35/50 [56-81 %] | 0.46 / 1.13 / 1.86 | 3 / 0, p 0.75 | 3.1 % (8 / 2 / 2 %) |
| crowd (8 drones), 0 birds | 50/50 = 100 % [93-100] | 0 | 0 | 0 | - | - | - | - |
| crowd, 2 birds | 29/50 = 58 % [44-71] | 21 | 0 | 0 | 50/50 [93-100 %] | 0.24 / 0.35 / 0.58 | 21 / 0, p < 0.001 | 9.7 % (19 / 9 / 6 %) |
| crowd, 4 birds | 17/50 = 34 % [22-48] | 33 | 0 | 0 | 50/50 [93-100 %] | 0.22 / 0.26 / 0.41 | 33 / 0, p < 0.001 | 10.8 % (20 / 11 / 6 %) |

(Holm p for the crowd cells prints as 0.000: the exact McNemar p is about 1e-6 for 21 / 0 and 2e-10 for 33 / 0.) Every failure with birds is a bird crash; birds caused no other crashes and no timeouts. The "lost / gained" columns count episodes the no-bird run solved and the bird run did not, and the reverse (never a gain).

## What fails, and why

1. **Single drone: the policy is rarely hit, but not because it avoids.** 4-6 % of episodes end in a bird crash, flat across 1, 2 and 4 birds, and none of the single-cell differences is significant (n = 50, all Holm p >= 0.75; the intervals are about +-6-10 points wide, so a true effect of a few points is not excluded). Pooling all single-drone bird cells, every one of the 13 bird crashes happened when the bird came within 0.5 m (centre), and none of the 142 episodes whose closest approach was 0.5-1 m ended in a crash: of the 76 episodes with a pass closer than 0.5 m, 13 (17 %) were crashes (crash radius 0.27 m). The aim timing (+-1 s, i.e. +-2 m along the path) and the bird's own speed make a dead-centre encounter the exception: only about one episode in six (76 of 450) sees a bird closer than 0.5 m, and the encounter rate itself rises with the bird count (56 % to 88 %), which a flat crash rate with n = 50 cannot resolve.
2. **Crowd: birds are very damaging.** An 8-drone head-on swarm converges on the centre; birds aimed at the drones' paths pass through the cluster at a median 0.26-0.35 m from some drone (100 % encounter rate). Success falls from 100 % to 58 % (2 birds) and 34 % (4 birds), all of it bird crashes, with significance far beyond the Holm threshold. Part of the drop is geometry rather than weakness of the policy: each bird crosses a volume with eight drones, and eight chances per bird count as one failed episode.
3. **The LiDAR hardly sees birds.** Only 3-11 % of (drone, bird) pair-steps within 6 m have a ray on the bird (single drone: about 3 % without obstacles and 6-8 % with the building (I did not isolate why); crowd about 10 %). Detection grows when the bird is close (0-2 m: 6-27 %, versus 2-6 % at 2-6 m) but is still mostly a miss when it matters. Two reasons: the 32-ray ring is horizontal at the drone's height and the 14-ray fan is sparse (11.25 degrees apart is about 0.6 m at 3 m, and a 0.12 m radius sphere subtends only about 4 degrees at 3 m), and birds with vertical speed or an aim point at the path's height pass above or below the ring. I did not run a policy that sees ground-truth birds, so I cannot separate "never sees" from "sees but cannot react".
4. **Bird speed (2 birds, single drone):** 1 m/s: 98 %, 2 m/s: 100 %, 3 m/s: 94 %, no monotonic trend and no significant difference from the no-bird cell (best p 0.75). At n = 50 per cell this test cannot show a speed effect of this size; the closest-approach distributions are similar across speeds.
5. **Goal-hold with birds.** Verified by test (`tests/test_birds.py::test_goal_hold_hands_back_when_a_bird_is_inside_the_ring_distance`): a drone 0.2 m from its goal with a bird 0.45 m away is guarded, i.e. goal-hold does not drive it and the policy keeps control, because the ring sees the bird inside `obstacle_release = 0.5 m`; without the bird the same drone is driven to the goal. In the runs the guard was active in at most 0.11 % of drone-steps (crowd 4 birds; 0 in most single cells): birds seldom reach a drone that is holding at its goal. This handback relies on the ring, so a bird above or below the ring plane does not trigger it.

## Limits

* 50 episodes per cell: Wilson intervals are +-6-14 points; only the crowd effects are statistically clear. Single-drone "no effect" is "no effect detectable at this size".
* The shipped policy was trained on static obstacles only; this is a zero-shot test, not a verdict on what training could do.
* Bird speeds are scaled to 1-3 m/s (see the scale note); results do not transfer to real-speed birds.
* Birds are aimed at the drones, so encounters are far more frequent than in the wild. Only cells with a real encounter rate mean anything: all cells here have >= 50 % (single) or 100 % (crowd), so none are vacuous, but "success given encounter" (in the JSON) is the better single-drone figure.
* One crash is assessed by centre distance (0.27 m); real drones have propellers and a larger envelope.
* Single-drone no-obstacle flights use the existing random band test (goal heights 1-5 m), so heights vary a little; bird altitude follows the path height.
* Tests: the full suite passes with the project venv (758 passed, including `test_scenario.py` and `test_eval_layered_failures.py`), and the birds-off output of `eval_suite` is byte-identical to the code before this change (single-drone and 8-drone cells, seeds 20-21: `episodes.csv`, `agents.csv`, `summary.json` without wall time).
* Aim timing in routed scenario runs is looser than in the single-drone tests (see "Scenario run"): it ignores the climb, lateness and slow flights, and the policy flies below the route height.
* The kinematic bird moves once per control step (0.032 s), at most 0.1 m at 3 m/s; this cannot tunnel through the 0.27 m crash radius.

## Recommendation for stages 3-4 (written after stage 2; stage 3a has since been run, see its section at the end for the update)

The results point at **(a) first**: feed birds to the policy as non-cooperative neighbours, without retraining. The LiDAR detects too few birds (3-11 %), so the bird positions should come from tracking (the UTM surveillance track, or an ideal tracker as an upper bound) and use the neighbour slots, which is how the policy already avoids drones. Test it on the crowd cells, where the headroom is (34-58 % success). If the policy treats bird slots like drones and recovers most of the loss, no training is needed. If not, **(b)** fine-tune with birds, teacher treating them as non-cooperative objects (no expectation of a reciprocal sidestep, larger margin, birds in the crowd scenario mix), starting from the shipped checkpoint. **(c)** the UTM side (surveillance reports birds like intruders, the Responder books a protective volume and re-plans, reusing `utm/contingency.py _intruder`) matters for routed city flights; it needs (a) or a surveillance track anyway and cannot fix a bird that appears within seconds, so it is complementary. The single-drone cells would need more episodes (100-200 per cell, cheap: about 0.7 s per episode) before any of these can be judged there.

## Scenario run (`scripts/run_scenario.py`, routed city flights)

A temp copy of `scenarios/ecity_failures.yaml` with `birds: {count: 6, speed: [1, 3]}` and the shipped checkpoint (committed files unchanged), 29 s of wall time:

* **Recording:** the top-level `birds` key is written (`radius` 0.12, `pos` 469 frames x 6 birds x 3), one entry per `frames.t` entry; the scenario's `birds` block is stored in `scenario`; the schema is still `utm-flight-log/2` and `read()` accepts the file.
* **Aimed at routes:** every bird's trajectory passes within 0.00-0.04 m (x, y) of a planned route. Caveat: 11 of the 12 routes here are straight in x and y (climb, cruise, descend), so this run cannot tell route aiming from straight-line aiming; only f6's route bends (7.3 m). The bent-route case is covered by `test_routed_flights_aim_at_the_route` on a polyline.
* **Encounters are weaker than in the single-drone tests:** the closest approach of each bird to any flying drone was 0.75, 1.12, 1.58, 1.70, 4.46 and 5.92 m (3 of 6 under 1.5 m in 3D). The aim time assumes `t_dep + arc length / 2 m/s`, but real flights climb at 0.5 m/s first, some are late or slow, and the policy flies about 0.8 m under the route height. If routed runs need a tighter encounter rate, aim with E1's planned arrival time per flight and the policy's height sag.
* **`cause: 'bird'`:** with 6 birds the only crash was f6's motor failure (correctly without a cause). A second run with 60 birds (78 s) produced `{'type': 'crashed', 'flight': 'intruder_0', 'cause': 'bird'}` at 18.0 s, next to f6's motor-failure crash, which has no cause.

## Stage 3a: an onboard bird detector feeds tracked birds to the policy as neighbours

### What was built
`envs/bird_detector.py` (config `environment.birds.detector`, default off; scenario key `birds: {detector: {...}}`; `eval_suite.py --bird-detector realistic|oracle --detector-range --detector-fov`). Each drone detects birds itself, then tracks them; it is never given true bird positions.

* **Per drone and bird, every control step:** the bird must be within `range_m`, inside the horizontal field of view (centred on the drone's velocity direction, its last heading when hovering; 360 = a multi-camera rig), and not hidden behind a building (one ray per candidate, batched, ignoring drones and birds). It is then detected with a probability falling linearly from `p_near` at 0 m to `p_far` at `range_m`. The measurement is the bird's position plus Gaussian noise, delivered `latency_steps` late. One alpha-beta filter per (drone, bird) gives position and velocity; a track is reported after `confirm_hits` measurements, extrapolated to the current step, coasts `coast_s` after its last measurement, then drops.
* **Own random generator** per episode (detector seed + episode hash): drones, sensor noise and bird paths are identical with and without the detector (verified: with the detector off and birds on, every episode row equals the stage-2 row). Drones still come from shared states; only birds are detected here.
* **Feeding the policy:** tracked birds become neighbour candidates next to the drones a drone knows about: each drone's 5 slots hold the nearest of both within the 30 m sensing radius, in the policy's units (relative position / 3 m, relative velocity / 2 m/s). The slot does not say which is which (the input has no such flag), so the policy treats a bird like a drone. With the detector off the code path is untouched (byte-identical outputs, tested).
* **Goal-hold guard:** needed no change: it reads the nearest neighbour from the slots, so a tracked bird within `neighbour_release` (0.9 m) hands control back to the policy (tested with a bird directly above the drone, outside the horizontal ring).
* **Oracle mode** (an upper bound, never a result): 360 degrees, no occlusion, no noise, no misses, no latency, true velocity, same 8 m range.

**These values are assumptions, like the sensor-noise levels, not measurements of any camera or radar:** range 8 m, field of view 120 degrees, detection probability 0.9 near falling to 0.5 at max range, position noise 0.1 m, latency 3 control steps (0.096 s), coast 0.5 s, alpha 0.4, beta 0.05, 2 hits to confirm, occlusion on. Not modelled: false alarms, birds merging into one track, track identity swaps, other drones being detected as birds, occlusion by drones, weather and light.

### Results (shipped policy, validation seeds 20-24, 50 episodes per cell, paired with stage 2)
Same episodes, same noise, same bird paths in all three conditions; "none" is the stage-2 number. Each cell: successes out of 50 (bird crashes, other crashes, timeouts). p = exact McNemar against "none" on the paired episodes (raw; the Holm-corrected values in `eval_results/birds/tables_3a.md` are all >= 0.58). "Track rate" = share of the (drone, bird) pair-steps within 6 m with a live track.

| cell | none | realistic detector | oracle (upper bound) | realistic: p (lost / gained) | oracle: p (lost / gained) | closest bird, median m (none / realistic / oracle) | track rate (realistic / oracle) |
|---|---|---|---|---|---|---|---|
| 1 bird | 48 (2, 0, 0) | 50 (0, 0, 0) | - | 0.50 (0 / 2) | - | 1.37 / 1.46 / - | 62 % / - |
| 2 birds | 48 (2, 0, 0) | 49 (1, 0, 0) | 48 (2, 0, 0) | 1.00 (1 / 2) | 1.00 (2 / 2) | 1.01 / 0.94 / 0.95 | 65 % / 100 % |
| 4 birds | 48 (2, 0, 0) | 48 (2, 0, 0) | 46 (4, 0, 0) | 1.00 (2 / 2) | 0.69 (4 / 2) | 0.74 / 0.70 / 0.71 | 65 % / 100 % |
| 1 bird + building | 46 (1, 0, 3) | 49 (1, 0, 0) | - | 0.25 (0 / 3) | - | 1.40 / 1.53 / - | 46 % / - |
| 2 birds + building | 46 (0, 0, 4) | 49 (1, 0, 0) | - | 0.38 (1 / 4) | - | 1.06 / 0.97 / - | 50 % / - |
| 4 birds + building | 45 (2, 0, 3) | 47 (3, 0, 0) | 47 (3, 0, 0) | 0.69 (2 / 4) | 0.69 (2 / 4) | 0.75 / 0.71 / 0.76 | 51 % / 100 % |
| 2 birds at 1 m/s | 49 (1, 0, 0) | 49 (1, 0, 0) | - | 1.00 (1 / 1) | - | 0.92 / 0.90 / - | 61 % / - |
| 2 birds at 2 m/s | 50 (0, 0, 0) | 49 (1, 0, 0) | - | 1.00 (1 / 0) | - | 1.07 / 0.93 / - | 69 % / - |
| 2 birds at 3 m/s | 47 (3, 0, 0) | 47 (3, 0, 0) | - | 1.00 (1 / 1) | - | 1.13 / 0.85 / - | 62 % / - |
| crowd, 2 birds | 29 (21, 0, 0) | 27 (22, 1, 0) | 31 (16, 3, 0) | 0.75 (6 / 4) | 0.69 (2 / 4) | 0.35 / 0.32 / 0.39 | 55 % / 100 % |
| crowd, 4 birds | 17 (33, 0, 0) | 22 (26, 2, 0) | 25 (23, 2, 0) | 0.23 (3 / 8) | 0.10 (5 / 13) | 0.26 / 0.27 / 0.30 | 55 % / 100 % |

Pooled: the 9 single-drone cells 427 / 450 successes without a detector and 437 / 450 with it (13 bird crashes in both); the 2 crowd cells 46 / 100 without, 49 / 100 with the realistic detector and 56 / 100 with the oracle (bird crashes 54, 48 and 39). Encounter rates (a bird within 1.5 m) are the same by construction (bird paths are paired); the drones' own paths barely change them.

**Sensitivity** (2 birds; crowd and single drone without obstacle): range 4 m / FOV 120: track rate 37 % (crowd) and 47 % (single), 30 / 50 and 48 / 50 successes; range 8 m / FOV 360: track rate 98 % and 99 %, 30 / 50 and 48 / 50. The realistic default (8 m / 120 degrees) gave 27 / 50 and 49 / 50. Range and field of view change how often a bird is tracked a lot and the outcome not at all (all p = 1.0).

### What this says
1. **Tracking works, avoidance does not follow.** The detector multiplies the nearby tracking rate by roughly 5-10 compared with the LiDAR, and the oracle reaches 100 %, but the median closest bird distance stays where it was (crowd 0.26-0.39 m in every condition) and crowd bird crashes barely move (22 and 26 of 50 with the realistic detector against 21 and 33 without). No single-drone or crowd difference is significant (smallest raw p 0.10, oracle crowd with 4 birds, 13 gained against 5 lost). The best reading is "possibly a modest gain with perfect sensing in the densest cell", not a result.
2. **The single-drone cells have no room to improve:** 96 % without a detector already. Their apparent gains with a building (46 to 49 of 50) are timeouts, not bird crashes: the no-bird baseline of that cell has 4 timeouts, and with the detector's extra slot input the timeouts disappear. I do not know why; it is a side effect on the policy's behaviour at buildings, not an effect of birds, and I would not call it a gain.
3. **Why the crowd stays bad (what I checked and what I only suspect).** Checked: the policy does react to a bird in a neighbour slot. In a probe (one drone, one bird head-on, five lateral offsets; a throwaway script, not in the repo), the drone's sideways deviation doubles when the oracle puts a 2 m/s bird in a slot (0.72 m against 0.36 m) and the closest distance grows from 0.87 m to 1.22 m, against 1.59 m for a second cooperative drone flying the same policy (1 m/s bird: 1.21 m to 1.28 m, deviation 0.35 m to 0.59 m). So the slots are read, but the reaction is weaker than for a drone and, in the crowd, it competes with seven drones in a head-on swarm: the drones' median separation is 0.66-0.75 m while birds pass the cluster at 0.3 m, and a sidestep of 0.4-0.7 m has little room. Suspected, not shown: the policy learned that a neighbour in a slot dodges too (a bird does not), a bird's relative speed (up to 5 m/s head-on) is beyond what it saw in training, and 0.1-0.29 s of lag plus a 0.4 s response time leave about 1 s to react to a bird first tracked at 8 m. **New drone-drone crashes appear** with birds in the slots (3 / 100 crowd episodes with the realistic detector and 5 / 100 with the oracle, against 0 / 100 without): a drone dodging a bird into a neighbour. The numbers are small but point one way, so they are a safety metric to watch in 3b.
4. **Sensing limits are not the bottleneck.** Range 4 against 8 m and a 120 against 360 degree field of view give the same success within noise, and the oracle is not much better than the realistic detector.

### Is stage 3b or 3c worth doing next? (as judged after stage 3a; revised in Stage 3a+ below, where 3c looks more promising)
* **3b (fine-tune with birds): yes, this is the next step.** With perfect sensing the unmodified policy still loses roughly half of the crowd episodes, so what is missing is the behaviour, not the information. Fine-tune from the shipped checkpoint with tracked birds (the realistic detector, not the oracle) in the slots, birds as non-cooperative objects for the teacher (no reciprocal dodge, a wider margin, a bird-crash cost, and dodging away from neighbours too), birds in the crowd scenario mix, and drone-drone crashes watched as a safety metric. Decide the success criterion in advance (for example crowd bird crashes under 10 % at 4 birds). The oracle crowd 4-bird gain (13 gained, 5 lost, raw p 0.10) is worth a larger sample first, because it shows whether any gain exists to build on.
* **3c (UTM bird radar and a protective volume): not yet.** It would deliver the same kind of information the oracle already gives (track positions), which did not fix the crowd, and a volume booked ahead of a bird seconds from a drone cannot be re-planned in time. It becomes useful for routed city flights once drones can actually avoid a tracked bird, for slow or hovering birds, and as the source of tracks a fleet would share; it needs 3b's behaviour first.
* **Before any of this, tighten the single-drone numbers** (100-200 episodes per cell, about 0.7 s per episode) and re-aim birds in routed runs (the routed-scenario encounters were weaker than the single-drone ones).

### Limits of stage 3a
* Detector parameters are assumptions; no false alarms, merged tracks or track swaps; the slot hides whether the neighbour is a bird; occlusion by drones is not modelled.
* 50 episodes per cell: only differences of about 20 percentage points are significant; the crowd 4-bird oracle gain (+8 episodes net) is not.
* The head-on probe is a small hand-built check (5 offsets per condition), not part of the paired evaluation.
* Detector off (and birds off) outputs are byte-identical to before; a detector that sees nothing leaves the observations unchanged bit for bit (tested).

## Stage 3a+: a classical bird guard on the tracked birds, and a fairer crowd test

### The guard (`modules/bird_guard.py`, `environment.birds.guard`, default off)
The project already hands the precise final stop to a classical controller (goal-hold), so the same is done for birds: a small rule layer takes over the velocity command only while a bird is a threat. Order: policy, goal-hold, bird guard.

* **Inputs (honest):** only the detector's tracked birds (the realistic detector of stage 3a; the oracle only as an upper bound), the drone's own noisy state (what the policy sees), the LiDAR ring and the other drones' shared states. The guard never sees a bird's true position: a bird the detector has not tracked triggers nothing (tested with a detector that never detects: the guard never fires and the bird hits).
* **Threat:** the teacher's closest-point-of-approach (`baselines/repulsion_controller.cpa`, extracted from the teacher so there is one copy; the teacher's outputs are bit-for-bit unchanged: 9,600 comparisons with float32 and float64 inputs, with and without obstacles, 0 differences) between the drone and a tracked bird, both moved 0.4 s ahead (the lag look-ahead the teacher uses), both keeping their velocities. A threat if the predicted miss is under `miss_m` = 1.0 m within `horizon_s` = 3 s and the bird is closing.
* **Evasion:** eight candidate velocities: the policy's command plus `evade_speed` = 1.5 m/s sideways (perpendicular to the relative motion, either side), up, down, or a diagonal; each is scored by the smallest of the predicted miss to the birds, the predicted miss to the other drones (their shared states, never the neighbour slots, which may hold birds) + 0.3 m, and the LiDAR ring distance in the sideways direction. The best replaces the command (per axis capped at 2 m/s).
* **Altitude band:** centred on the drone's current target height (the carrot's for routed flights, the goal's for the test cells), half-width `min(1.5 m, 0.45 x layer spacing)` when a scenario has layers (0.9 m for layers 2 m apart, set by `run_scenario.py`), 1.5 m otherwise, never below 0.8 m.
* **Safety while active:** the teacher's push-away from drones inside `close_dist` and from LiDAR obstacle hits inside `obs_close` is added, so the guard cannot walk a drone into a neighbour or a wall.
* **Release:** hysteresis: it stays active until, for 5 steps, every tracked bird is opening or its predicted miss is above 1.4 m.
* **`birds_in_slots`** (`environment.birds.detector`, default true = the stage 3a behaviour): with false, the five neighbour slots stay for drones and the tracks are used only by the guard.
* **Parameters were set from the teacher's and the design's values before any run and not tuned on these results** (no sweep was run).

### The fairer crowd test (`environment.birds.spawn: random`)
The stage 2 and 3a crowd cells time every bird to reach a drone's path when that drone does, and in the head-on test all 8 drones meet at the centre: a deliberate worst case. The new mode sends birds on random straight lines through the arena (the drones' path box + 2 m) at random times during the flight, unrelated to the drones, at the same speeds. Even so, with 8 drones in a small arena a bird crosses a drone's neighbourhood in most episodes (47-50 of 50 episodes have a bird within 1.5 m), but much less often at the dense centre.

### Results (shipped policy, seeds 20-24, 50 episodes per cell, paired with stages 2 and 3a)
Same episodes, noise and bird paths in every condition. Each cell: successes of 50 (bird crashes, drone-drone crashes, obstacle crashes, timeouts). p = exact McNemar against "none", raw (the Holm-corrected values, over each group's tests, are in `eval_results/birds/tables_3ap.md` and quoted below). **The guard on the oracle is an upper bound, never a result.** "Guard, not in slots" means the realistic detector's tracks feed the guard only; the oracle guard is also run with birds not in the slots.

**Worst-case crowd (birds timed to the drones)**

| cell | none (stage 2) | detector + slots (3a) | guard, birds not in slots | guard, birds in slots | guard on the oracle (upper bound) |
|---|---|---|---|---|---|
| 2 birds | 29 (21, 0, 0, 0) | 27 (22, 1, 0, 0) | 36 (13, 1, 0, 0), p 0.07 (2 lost / 9 gained), Holm 0.13 | 31 (17, 2, 0, 0), p 0.77 | 45 (4, 1, 0, 0), p < 0.01 (2 / 18), Holm 0.002 |
| 4 birds | 17 (33, 0, 0, 0) | 22 (26, 2, 0, 0) | 30 (20, 0, 0, 0), p 0.01 (4 / 17), Holm 0.022 | 31 (16, 3, 0, 0), p < 0.01 (2 / 16), Holm 0.005 | 42 (7, 1, 0, 0), p < 0.01 (1 / 26), Holm < 0.001 |

**Fair crowd (random straight crossings)**

| cell | none | detector + slots (3a) | guard, birds not in slots | guard, birds in slots | guard on the oracle (upper bound) |
|---|---|---|---|---|---|
| 2 birds | 43 (7, 0, 0, 0) | 44 (6, 0, 0, 0) | 47 (3, 0, 0, 0), p 0.22 | 46 (4, 0, 0, 0), p 0.38 | 50 (0, 0, 0, 0), p 0.02 (0 / 7), Holm 0.094 |
| 4 birds | 37 (13, 0, 0, 0) | 35 (14, 1, 0, 0) | 41 (7, 2, 0, 0), p 0.34 | 45 (5, 0, 0, 0), p 0.01 (0 / 8), Holm 0.059 | 48 (0, 2, 0, 0), p 0.01 (2 / 13), Holm 0.059 |

**Guard activity and cost (crowd cells; cost = the same drones that settled in both runs, against the same episodes with no birds)**

| cell | condition | closest bird, median (m) | guard active (drone-steps / episodes) | extra path length / settle time |
|---|---|---|---|---|
| worst, 2 birds | none / guard not in slots / guard in slots / oracle | 0.35 / 0.52 / 0.47 / 0.82 | - / 6.0 % / 46 of 50 / 6.1 % / 47 of 50 / 8.3 % / 50 of 50 | x1.00 / x1.02 +0.1 s / x1.02 +0.1 s / x1.07 +0.5 s |
| worst, 4 birds | none / guard not in slots / guard in slots / oracle | 0.26 / 0.35 / 0.43 / 0.67 | - / 11.1 % / 47 of 50 / 10.3 % / 47 of 50 / 13.6 % / 50 of 50 | x1.00 / x1.04 +0.2 s / x1.04 +0.2 s / x1.14 +0.9 s |
| fair, 2 birds | none / guard not in slots / guard in slots / oracle | 0.48 / 0.63 / 0.66 / 1.00 | - / 2.9 % / 43 of 50 / 2.8 % / 43 of 50 / 4.2 % / 49 of 50 | x1.00 / x1.01 +0.1 s / x1.01 +0.1 s / x1.03 +0.2 s |
| fair, 4 birds | none / guard not in slots / guard in slots / oracle | 0.34 / 0.51 / 0.48 / 0.92 | - / 4.9 % / 48 of 50 / 4.6 % / 48 of 50 / 7.1 % / 49 of 50 | x1.00 / x1.02 +0.1 s / x1.02 +0.1 s / x1.05 +0.4 s |

**Single drone, pooled over 5 cells (2 and 4 birds without and with a building, 2 birds at 3 m/s; 250 episodes)**

| condition | successes | bird / drone-drone / obstacle crashes, timeouts | raw p vs none (lost / gained) | guard active |
|---|---|---|---|---|
| none (stage 2) | 234 | 9 / 0 / 0, 7 | - | - |
| detector + slots (3a) | 240 | 10 / 0 / 0, 0 | - | - |
| guard, birds not in slots | 237 | 7 / 0 / 0, 6 | 0.45 (2 / 5) | 5.3 % |
| guard, birds in slots | 242 | 8 / 0 / 0, 0 | 0.10 (5 / 13) | 6.8 % |

### What this says
1. **The guard works where the policy alone failed.** In the worst-case crowd the guard on the realistic detector's tracks lifts 4-bird success from 17 to 30-31 of 50 (Holm 0.022 and 0.005) and 2-bird from 29 to 36 (Holm 0.13, not significant on its own; the 31 of the in-slots variant is no better than none). Bird crashes in the two worst-case crowd cells fall from 54 to 33 (guard, not in slots) and 33 (in slots) of 100 episodes, against 48 for the detector in the slots (stage 3a). Median closest approach grows from 0.26-0.35 m to 0.35-0.52 m, i.e. the guard moves the passes away from the 0.27 m crash radius but they stay close. The path cost is small: 2-4 % longer, 0.1-0.2 s later.
2. **Fair crowd: smaller gains, not significant after the correction.** Even with random crossings 13 of 50 episodes with 4 birds (7 of 50 with 2) end in a bird crash without any avoidance, so birds are a real hazard to an 8-drone swarm, not only in the worst case. The guard brings these to 7 (not in slots) and 5 (in slots) of 50 with 4 birds, and to 3 and 4 with 2 birds; the largest gain (4 birds, in slots: 37 to 45, raw p 0.01) is Holm 0.059, and the others are not significant. With 50 episodes per cell the tests can only confirm differences of 15-20 points.
3. **Tracks are the limit.** The guard on the oracle (every bird within 8 m, no noise, no misses, 360 degrees) reaches 45 / 42 of 50 (worst case) and 50 / 48 (fair), with bird crashes 4 / 7 and 0 / 0 (the fair 4-bird cell's 2 failures are drone-drone crashes). The realistic detector tracked only about 55 % of the nearby (drone, bird) pairs in the crowd and 30-40 % in the fair crowd. In the realistic runs the episodes that ended in a bird crash had lower track rates than the others (45 % against 54 % in the worst case, 30 % against 40 % in the fair crowd), and in about 20 % of them the guard never fired at all: no track, no reaction. In the 11 oracle crashes the guard fired every time with 100 % tracking, so those are evasion limits: 8 m of range at up to 5 m/s of closing speed leaves about 1.6 s, the lag takes 0.1-0.3 s plus a 0.4 s response, and the swarm has little room. The gap between the realistic detector and the oracle (36 against 45 of 50 at 2 birds, 30 against 42 at 4 birds, worst case) is therefore mostly sensing.
4. **Slots or not.** With the guard on, putting tracked birds in the neighbour slots as well made no consistent difference (4 birds, worst case: 31 against 30; fair: 45 against 41; 2 birds: 31 against 36 and 46 against 47) and it adds drone-drone crashes in the worst-case crowd (5 against 1 in 100 episodes; fair: 0 against 2, so the count is not consistent either). The pooled single-drone gain of "in slots" (242 against 237) is again the timeout side effect at the building seen in stage 3a (the building cells' 7 timeouts disappear whenever a bird is in the slots), not a bird effect. So no reason to prefer slots over leaving them to drones: the simpler "guard only" design keeps the policy's inputs unchanged.
5. **Drone-drone and obstacle crashes.** There are no obstacle crashes anywhere. Drone-drone crashes remain rare but not zero: worst-case crowd, 0 with no detector, 3 with the detector in slots, 1 with the guard (not in slots), 5 with the guard in slots, 2 with the oracle (out of 100 episodes each); fair crowd 0 / 1 / 2 / 0 / 2. None of these is significant, and the guard's push-away from drones (inside 0.9 m) is there to prevent them, but they are a number to watch.
6. **Single drones** have nothing to gain (the stage 2 policy already survives about 94 % of these single-drone episodes) and nothing to lose: no significant difference, extra path 1-3 %.

### Is stage 3b or 3c still worth doing?
* **The cheap result stands: a classical guard on tracked birds recovers about a third to 40 % of the crowd episodes lost to birds in the worst case with the weak realistic detector (7 of 21 and 13 of 33), and about three quarters of them with perfect tracks (16 of 21 and 25 of 33).** The step that matters now is the quality of the tracks, not the policy.
* **3c (UTM bird tracks) becomes the more promising of the two.** In stage 3a it looked unhelpful because the information went through the neighbour slots, where the policy reacts weakly; with the guard the same information is used well (oracle: 42-50 of 50 against 17-43). Tracks from outside the drone (a surveillance radar, tracks shared across the fleet) would approach the oracle's coverage: 360 degrees, no misses from one drone's narrow field of view, and a longer warning than 8 m. What it is not is a protective volume booked in advance: a bird seconds from a drone can only be handled by a drone-side guard fed with fast tracks.
* **First a cheap, decisive experiment: sensitivity of the guard to the detector** (range 4 / 8 / 12 m, field of view 120 / 360 degrees, detection probability and latency). Done: see "Stage 3a+ sensor sensitivity" below (a 360-degree view is what matters).
* **3b (fine-tuning with birds) is the lowest priority now.** Its upper end is bounded by the oracle guard (42-50 of 50), which the classical layer already shows is reachable; fine-tuning only matters if the guard's evasion limits (lag, crowd room, 8 m of warning) turn out to be what blocks a good result with good tracks, and the remaining bird crashes with the oracle are few (4 and 7 of 100 worst-case episodes, 0 fair).
* **Before building on this:** repeat the fair-crowd cells with 100-200 episodes (about 1.4 s per crowd episode), because only the worst-case results are significant at 50, and tune nothing on the validation seeds' results alone.

### Limits of stage 3a+
* The detector values are assumptions (stage 3a); the guard's parameters are untuned defaults; no sweep over them was run.
* 50 episodes per cell; the fair-crowd differences are not significant after the Holm correction.
* The oracle guard is an upper bound for the sensing, with 360 degrees, no misses and no latency at 8 m; it is not an achievable sensor.
* The guard handles birds; it does not change how the policy handles drones, obstacles or buildings, and it needs a LiDAR ring for the obstacle check (no ceiling or floor sensing for vertical evasion).
* One bug was found by the full run's first attempt (a latched guard whose track had dropped computed an evasion against zero birds and crashed); it was fixed and covered by a test before any result here was produced (the run was restarted, and all results above come from the fixed code).
* Routed city flights were not evaluated with the guard (the plumbing exists: `birds.guard` in a scenario, layer spacing 2 m); that is an open item.

## Stage 3a+ sensor sensitivity: what detector does the guard need?

One detector setting changed at a time around the default (range 8 m, field of view 120 degrees, detection probability 0.9 at 0 m falling to 0.5 at the range, latency 3 control steps = 0.096 s), plus one combined best case. The guard runs on the realistic detector's tracks, birds not in the neighbour slots, 4 birds, the worst-case crowd and the fair crowd, seeds 20-24, 50 episodes per cell, every cell paired with the same episodes of the "none" cell (`eval_birds.py --stage sens`, flags `--detector-range/-fov/-p/-latency`). The default row and the oracle are the stage 3a+ cells. All values are assumptions, as before. (Run time: 16 runs took 40 minutes on the laptop, about twice my estimate, probably because another job shared the CPU.)

**Worst-case crowd, 4 birds** (successes of 50; bird / drone-drone crashes; Holm p over the cell's tests against "none"; "vs default" = episodes lost / gained against the default guard cell)

| detector | success | bird / d-d crashes | track rate (pairs < 6 m) | closest bird, median (m) | guard active | Holm p vs none | vs default (lost / gained) |
|---|---|---|---|---|---|---|---|
| none (no avoidance) | 17 | 33 / 0 | - | 0.26 | - | - | - |
| default: 8 m, 120 deg, 0.9 to 0.5, 3 steps | 30 | 20 / 0 | 53 % | 0.35 | 11.1 % | 0.022 | - |
| range 4 m | 24 | 26 / 0 | 38 % | 0.27 | 8.3 % | 0.065 | 11 / 5 |
| range 12 m | 34 | 16 / 0 | 53 % | 0.41 | 11.3 % | 0.001 | 1 / 5 |
| FOV 360 degrees | 33 | 15 / 2 | 99 % | 0.49 | 14.9 % | < 0.001 | 4 / 7 |
| detection 0.7 to 0.3 | 29 | 18 / 3 | 54 % | 0.35 | 10.4 % | 0.021 | 6 / 5 |
| detection 1.0 (no misses) | 32 | 15 / 3 | 54 % | 0.44 | 12.0 % | 0.009 | 4 / 6 |
| latency 0 | 30 | 19 / 1 | 55 % | 0.36 | 10.6 % | 0.021 | 3 / 3 |
| latency 9 steps (0.29 s) | 29 | 20 / 1 | 51 % | 0.34 | 10.5 % | 0.024 | 5 / 4 |
| range 12 m + FOV 360 degrees | 34 | 15 / 1 | 99 % | 0.46 | 15.5 % | 0.002 | 2 / 6 |
| oracle (upper bound) | 42 | 7 / 1 | 100 % | 0.67 | 13.6 % | < 0.001 | 2 / 14 |

**Fair crowd, 4 birds**

| detector | success | bird / d-d crashes | track rate | closest bird, median (m) | guard active | Holm p vs none | vs default (lost / gained) |
|---|---|---|---|---|---|---|---|
| none (no avoidance) | 37 | 13 / 0 | - | 0.34 | - | - | - |
| default | 41 | 7 / 2 | 39 % | 0.51 | 4.9 % | 1.000 | - |
| range 4 m | 39 | 10 / 1 | 19 % | 0.41 | 3.0 % | 1.000 | 5 / 3 |
| range 12 m | 41 | 9 / 0 | 39 % | 0.49 | 5.5 % | 1.000 | 3 / 3 |
| FOV 360 degrees | 46 | 3 / 1 | 99 % | 0.63 | 7.8 % | 0.180 | 1 / 6 |
| detection 0.7 to 0.3 | 40 | 10 / 0 | 39 % | 0.46 | 4.7 % | 1.000 | 3 / 2 |
| detection 1.0 (no misses) | 41 | 7 / 2 | 40 % | 0.50 | 5.4 % | 1.000 | 1 / 1 |
| latency 0 | 43 | 6 / 1 | 39 % | 0.50 | 5.0 % | 0.766 | 2 / 4 |
| latency 9 steps (0.29 s) | 41 | 9 / 0 | 37 % | 0.46 | 4.5 % | 1.000 | 3 / 3 |
| range 12 m + FOV 360 degrees | 47 | 3 / 0 | 99 % | 0.69 | 8.5 % | 0.063 | 1 / 7 |
| oracle (upper bound) | 48 | 0 / 2 | 100 % | 0.92 | 7.1 % | 0.066 | 2 / 9 |

### What this says
1. **Coverage is the setting that matters: a 360-degree field of view.** It raises the track rate from 53 % / 39 % to 99 %, success from 30 to 33 (worst case) and from 41 to 46 (fair), and the median closest approach from 0.35 to 0.49 m and from 0.51 to 0.63 m. Combined with 12 m of range it gives 34 and 47. Against the default guard these gains are 4 lost / 7 gained (worst case) and 1 lost / 6 gained (fair crowd) for 360 degrees alone, and 2 / 6 and 1 / 7 for the combined variant (exact McNemar p about 0.07 for the last): suggestive, not significant at 50 episodes. Against no avoidance, 360 degrees is significant in the worst case (Holm < 0.001) and close to it in the fair crowd (Holm 0.18 alone, 0.063 with 12 m of range).
2. **Range:** 4 m is clearly worse (24 against 30 in the worst case, 11 lost against 5 gained; the track rate falls to 38 % and 19 %), 12 m is slightly better (34, with the track rate unchanged because the tracked pairs are counted within 6 m). A bird that is first seen at 4 m, at up to 5 m/s of closing speed, leaves under a second, less than the 0.4 s response plus the evasion. Longer range helps less than coverage.
3. **Detection probability and latency barely matter in the tested ranges.** Raising the detection probability to 1.0 or lowering it to 0.7 to 0.3 changes success by -1 to +2 episodes in the worst case and by -1 to 0 in the fair crowd; latency 0 against 9 steps (0 to 0.29 s) by 30 against 29 and 43 against 41. The tracker coasts for 0.5 s and the guard re-evaluates every step, so missed frames and a short delay are covered. None of these differences is significant.
4. **What is still missing against the oracle** (42 / 48): the best realistic variants reach 33-34 (worst) and 46-47 (fair), with 99 % tracking, so coverage no longer explains the gap in the worst case (about 8 episodes). The oracle also has no position noise, no misses, no delay and the bird's true velocity; the tracker's velocity estimate (noise 0.1 m, alpha-beta) feeds the closest-point-of-approach directly. I did not test noise or the tracker, so this is a hypothesis: the next cheap experiment is position noise 0 and 0.05 m on the best realistic variant (360 degrees, 12 m) and a better velocity filter.
5. **A few drone-drone crashes appear** in some variants (2-3 of 50 in the worst case with 360 degrees, detection 0.7 and detection 1.0; none consistent with the setting).
6. **Cost:** the guard is active 8-15 % of drone-steps with 360 degrees (against 5-11 % with 120 degrees): more birds are seen, so it acts more; the earlier cost figures (2-4 % extra path) were measured with 120 degrees and have not been remeasured for these variants.

### What sensor a drone would need (from this test, with the limits below)
A field of view all around the drone (a multi-camera rig, or fleet-shared radar tracks as in 3c), a range of at least 8 m (12 m is a little better, 4 m is too short for a 2 m/s drone and 1-3 m/s birds), and nothing exceptional on detection probability (0.5 to 0.9 is enough here) or latency (up to 0.3 s was tolerated). The cheaper sensor options that were tested are not substitutes for coverage: raising the detection probability to 1.0 does not replace 360 degrees.

### Limits of this sweep
One setting at a time on two cells, 50 episodes each: only the largest differences can be seen, and interactions (for example short range with a narrow field of view) are not covered except for one combined case. Detector noise, the tracker's filter gains, false alarms and track identity errors were not varied. Detector values remain assumptions, not measurements of a real sensor.


## Stage 3a++ (shared tracks, Kalman tracker, guard brake; tuned on seeds 20-22, confirmed on 23-24, final run held out)

No retraining: the policy is the same checkpoint; only the classical layer and the onboard tracking changed.

### What was added (all opt-in; everything off = byte-identical to stage 3a+, checked on single-drone, 8-drone and guard cells)

* **Shared tracks** (`detector.share`): each drone also uses the tracks of drones within 30 m, 3 control steps old, merged by geometry (1.0 m gate; freshest, then most hits, own first). Tracks then have no bird ids.
* **Kalman tracker** (`detector.tracker: kalman`, `kalman_q`, `kalman_v0`): per-axis constant-velocity filter; much better velocity estimate for straight birds (0.10 vs 0.31 m/s error in a numpy check), about equal for turning birds with q=1, slightly better with q=2.
* **Guard brake** (`guard.brake`): extra evasion candidates that scale the command (hover, half speed).
* **`--episode-offset`** in `eval_suite.py`: skips the first N episodes of each seed (their scenarios are still generated, so the rest is exactly a longer run). Used to keep the final run held out from tuning.
* Driver: `scripts/eval_birds_pp.py` (phases `tune`, `confirm`, `final`); tests in `tests/test_bird_sharing.py`.

**Generosity note.** Without sharing, the detector associates each drone's measurements to its tracks perfectly (tracks are indexed by the true bird). A real tracker would not. Shared tracks use geometric association only. So the unshared tracker is the more generous of the two, and sharing is the more realistic.

### Tuning (seeds 20-22, episodes 0-14: 45 episodes per cell, 4 birds)

Selection rule, fixed before any run: highest combined successes over the worst-case and fair 4-bird cells; ties go to fewer drone-drone/obstacle crashes, then lower cost. One or two settings were changed at a time around the previous best (360 degrees, 12 m); sharing was also tried on the cheap 120 degrees / 8 m sensor.

| variant | worst-case | fair | combined (of 90) |
|---|---|---|---|
| no avoidance | 18 | 32 | 50 |
| previous best (base) | 30 | 42 | 72 |
| miss_m 0.8 / 1.2 / 1.5 | 30 / 33 / 32 | 39 / 41 / 41 | 69 / 74 / 73 |
| horizon 2 / 4 s | 29 / 30 | 42 / 41 | 71 / 71 |
| evade speed 1.0 / 2.0 | 29 / 30 | 41 / 39 | 70 / 69 |
| predict 0.2 / 0.6 s | 31 / 32 | 43 / 40 | 74 / 72 |
| release 1.2 / 1.8 m | 32 / 31 | 41 / 41 | 73 / 72 |
| brake [0, 0.5] | 31 | 39 | 70 |
| Kalman q=1 | 30 | 41 | 71 |
| **Kalman q=2** | **34** | **41** | **75** |
| sharing (12 m, 360) | 32 | 38 | 70 |
| cheap sensor (8 m, 120), alone / with sharing | 29 / 30 | 36 / 40 | 65 / 70 |
| combined: Kalman q=2 + miss 1.2 + predict 0.2 / plus sharing | 34 / 34 | 41 / 40 | 75 / 74 |

Chosen: **Kalman tracker with q=2, nothing else changed** (tie at 75 with the combined variant, which had 4 drone-drone/obstacle crashes against 2). Honest reading: no single setting is significantly better than the previous best at 45 episodes (best McNemar p = 0.29); the choice is the best of about 20 noisy candidates and should be expected to regress. The combined variants were added after seeing the single-setting results. Sharing did not help here (70 vs 72), and the brake did not either. Sharing did lift the cheap sensor (65 to 70; track rate 54 to 88 % worst-case), but not above the 360-degree sensor.

### Confirmation (seeds 23-24, episodes 0-19 per seed: 40 per condition)

| cell | no avoidance | previous best | new final | oracle |
|---|---|---|---|---|
| worst-case crowd, 4 birds | 35 % | 70 % | 65 % | 82 % |
| fair crowd, 4 birds | 72 % | 88 % | 98 % | 95 % |

New vs previous best: p = 0.69 (worst-case; 4 lost, 2 gained) and p = 0.22 (fair; 1 lost, 5 gained). The tuning gain did not show up in the worst-case cell; the fair cell looks better but is not significant.

### Final run (150 episodes per cell; seeds 20-22 use episodes 15-44, which tuning never flew, seeds 23-24 use episodes 0-29; the confirmation episodes are a subset of the latter)

Success rate with 95 % Wilson interval:

| cell | no avoidance | previous best | new final | oracle |
|---|---|---|---|---|
| worst-case crowd, 2 birds | 53 % [45-61] | 76 % [69-82] | 78 % [71-84] | 83 % [77-88] |
| worst-case crowd, 4 birds | 29 % [23-37] | 61 % [53-68] | 61 % [53-69] | 75 % [68-82] |
| fair crowd, 2 birds | 87 % [80-91] | 97 % [92-99] | 97 % [92-99] | 99 % [96-100] |
| fair crowd, 4 birds | 71 % [63-77] | 91 % [86-95] | 95 % [90-97] | 93 % [88-96] |

Single drone (no crowd): no avoidance / previous best / new final.

| cell | no avoidance | previous best | new final |
|---|---|---|---|
| no obstacles, 2 birds | 97 % | 97 % | 95 % |
| no obstacles, 4 birds | 96 % | 93 % | 94 % |
| buildings, 2 birds | 87 % | 88 % | 91 % |
| buildings, 4 birds | 77 % | 83 % | 83 % |
| speed 3 m/s, 2 birds | 95 % | 95 % | 95 % |

Paired tests (McNemar, Holm-adjusted over all comparisons): every guard condition beats no avoidance in all four crowd cells (Holm p <= 0.012). **New final vs previous best is not significant anywhere** (Holm p = 1.0 in all crowd cells; raw p = 0.65, 1.00, 1.00, 0.30). Oracle vs previous best: significant in the worst-case 4-bird cell (Holm p = 0.004, 75 % vs 61 %), not elsewhere. In single-drone cells nothing is significant against no avoidance (all Holm p >= 0.95); the guard is neither clearly helping nor hurting there. Costs are small: extra path x1.03-1.13, settle time +0.3 to +1.0 s in crowd cells; the new configuration is about the same as the previous best (x1.10 vs x1.11 worst-case, 4 birds).

Crash mix, fair crowd, 4 birds: bird crashes fall from 9 (previous best) to 3 (new), while drone-drone crashes rise from 4 to 5; the oracle has 2 bird and 8 drone-drone crashes. In the worst-case 4-bird cell: bird crashes 56 / 51 / 31 (previous / new / oracle), drone-drone 3 / 7 / 6.

### Result in one paragraph

The previous best configuration was already near what these changes can give. Kalman q=2 is at best a small gain (about +4 points in the fair 4-bird cell, +2 in the worst-case 2-bird cell, 0 in the worst-case 4-bird cell) and none of it is statistically significant at 150 episodes. Sharing and the brake did not help in tuning and were not used in the final configuration. The gap to the oracle (75 % vs 61 % in the worst-case 4-bird cell) is real and significant, so a better detector/tracker has headroom, but the tracker tweaks tried here do not close it.

### What still fails

* Worst-case crowd with 4 birds: 39 % of episodes still end in a crash (51 bird crashes + 7 drone-drone in 150), and the oracle also crashes in 25 %. The oracle result shows that part of this is not detection (a likely reason is lack of room and time to evade in a dense swarm; not tested).
* Evasion pushes drones into each other (drone-drone crashes go up when bird crashes go down; 7 in the new configuration, worst-case cells).
* The residual bird crashes with a perfect detector (oracle) show a limit of the evasion rule itself (single manoeuvre, 1.5 m/s, no coordination between drones).
* Single-drone cells: the guard gives no significant benefit.

### Limits

* Tuning used 45 episodes per cell: the selected variant is the best of many noisy candidates. The held-out confirmation (40 episodes) and final run (150) are the honest numbers; the final run's seeds 23-24 include the confirmation episodes.
* Sharing was tested with a fixed model (30 m range, 3 steps latency, 1 m gate). Other comms models were not tried.
* Without sharing the tracker uses true-id association (generous, see above).
* Part of the 38+4 tuning runs was done while the laptop was shared; this slows runs but does not change results. One confirmation run was killed by a low-memory event and was rerun from scratch.

### Files

`eval_results/birds/pp/` (`tune_*`, `confirm_*`, `final_*` run folders, `tables_tune.md`, `tables_confirm.md`, `tables_final.md`, `results_*.json`, `final_config.json`, `tuning_choice.md`).
