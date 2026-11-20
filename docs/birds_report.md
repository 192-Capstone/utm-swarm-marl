# Birds: moving non-cooperative obstacles, shipped policy without retraining (stages 1-2)

Branch `feature/birds`. Everything is opt-in (`environment.birds`, scenario key `birds:`, `eval_suite.py --birds N`); with birds off every output is byte-identical to before.

## Main findings (read these first)

1. **The policy mostly does not avoid birds, because it hardly sees them.** In the runs, a bird within 6 m was detected (at least one LiDAR ray on it) in only about 3-11 % of the (drone, bird) steps: the 32-ray ring is horizontal and the 14-ray fan is sparse, against a 0.12 m sphere. Single drones survive largely because birds pass close but miss (every single-drone crash was a pass within 0.5 m; passes at 0.5-1 m never crashed), not because they dodge. This is the main finding, and the reason the first thing to try next is stage 3a: feeding tracked birds to the policy as non-cooperative neighbours.
2. **The crowd cells are a deliberate worst case, not typical traffic.** Birds are timed to reach a drone's path when that drone does, and in the 8-drone head-on test all drones meet at the centre, so a bird aimed there crosses the whole cluster (100 % encounter rate, median closest approach 0.26-0.35 m). The 58 % (2 birds) and 34 % (4 birds) success must not be read as what the policy would do in ordinary traffic with birds around.

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

## Recommendation for stages 3-4 (not started)

The results point at **(a) first**: feed birds to the policy as non-cooperative neighbours, without retraining. The LiDAR detects too few birds (3-11 %), so the bird positions should come from tracking (the UTM surveillance track, or an ideal tracker as an upper bound) and use the neighbour slots, which is how the policy already avoids drones. Test it on the crowd cells, where the headroom is (34-58 % success). If the policy treats bird slots like drones and recovers most of the loss, no training is needed. If not, **(b)** fine-tune with birds, teacher treating them as non-cooperative objects (no expectation of a reciprocal sidestep, larger margin, birds in the crowd scenario mix), starting from the shipped checkpoint. **(c)** the UTM side (surveillance reports birds like intruders, the Responder books a protective volume and re-plans, reusing `utm/contingency.py _intruder`) matters for routed city flights; it needs (a) or a surveillance track anyway and cannot fix a bird that appears within seconds, so it is complementary. The single-drone cells would need more episodes (100-200 per cell, cheap: about 0.7 s per episode) before any of these can be judged there.

## Scenario run (`scripts/run_scenario.py`, routed city flights)

A temp copy of `scenarios/ecity_failures.yaml` with `birds: {count: 6, speed: [1, 3]}` and the shipped checkpoint (committed files unchanged), 29 s of wall time:

* **Recording:** the top-level `birds` key is written (`radius` 0.12, `pos` 469 frames x 6 birds x 3), one entry per `frames.t` entry; the scenario's `birds` block is stored in `scenario`; the schema is still `utm-flight-log/2` and `read()` accepts the file.
* **Aimed at routes:** every bird's trajectory passes within 0.00-0.04 m (x, y) of a planned route. Caveat: 11 of the 12 routes here are straight in x and y (climb, cruise, descend), so this run cannot tell route aiming from straight-line aiming; only f6's route bends (7.3 m). The bent-route case is covered by `test_routed_flights_aim_at_the_route` on a polyline.
* **Encounters are weaker than in the single-drone tests:** the closest approach of each bird to any flying drone was 0.75, 1.12, 1.58, 1.70, 4.46 and 5.92 m (3 of 6 under 1.5 m in 3D). The aim time assumes `t_dep + arc length / 2 m/s`, but real flights climb at 0.5 m/s first, some are late or slow, and the policy flies about 0.8 m under the route height. If routed runs need a tighter encounter rate, aim with E1's planned arrival time per flight and the policy's height sag.
* **`cause: 'bird'`:** with 6 birds the only crash was f6's motor failure (correctly without a cause). A second run with 60 birds (78 s) produced `{'type': 'crashed', 'flight': 'intruder_0', 'cause': 'bird'}` at 18.0 s, next to f6's motor-failure crash, which has no cause.
