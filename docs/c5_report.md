# C5 report: city worlds in Webots Level 2 vs PyBullet

All numbers come from `eval_results/c5/` and `logs/webots/c5/`. Each table names its source file.

## Summary

- C5 flies the city worlds (city easy / medium / hard and the OSM crop) in Webots Level 2 with
  several drones (`webots_l2_multi`), and compares them with PyBullet, paired on the same
  scenarios and seeds.
- City routes (4 drones, 400 episodes and 1600 flights per simulator): Webots and PyBullet
  agree. No McNemar test is significant. Webots had 0 lost-control flights in 1600
  (`eval_results/c5/c5_compare.md`, City routes, Per flight table).
- E1-flown traffic (358 flights per simulator): approvals and zone entries agree. Close
  passes < 2.0 m agree, but close passes < 1.4 m are more frequent in Webots on crop/no_e1
  (33.0% vs 24.2%, Holm p = 0.0234; pooled raw p = 0.0309). Webots lost control of 8 of 358
  flights; PyBullet of 0 (lost control, all 358 flights pooled, exact McNemar p = 0.0078, raw,
  not Holm-adjusted).
- The flips are not a park/place artefact (0 pad conflicts). 6 of 8 happen around the final
  descent. Two landing options (off by default) cut them from 8 to 4 (latch 2.4 m) and to 1
  (latch 2.4 m + vz limit 0.5 m/s). The last one is a climb flip.
- Webots flights settle later than PyBullet: median +1.8 to +1.9 s in part 4.
- C1 results are byte-identical after the C5 changes. Parity is 40/40.

## What C5 is

C5 checks whether the trained policy behaves the same in the city worlds when the drones are
flown by Bitcraze's onboard velocity PID in Webots (Level 2), instead of PyBullet.

- Checkpoint: `checkpoints/stage2_r1_crowd_rl_r7v3ft_s0_lab/checkpoint_final.pth`
  (md5 87a12b5c3799ac302e51ae1d10d18c6a).
- Config: `configs/config_stage2_round7_eval_goalhold.yaml` (md5 1d96843067df8db17cecf3e1d2d889a2).
- Validation seeds 20-24. Webots R2025a. Python 3.10.19.
- Two experiments:
  - City routes: `--modes route`, 4 drones, 20 episodes per seed, maps easy / medium / hard / crop.
  - E1-flown: maps medium and crop, `--flights 20 --window 60 --zones 0.2`, conditions
    `no_e1` and `e1`.

Source: `eval_results/c5/RUN_NOTES.txt`.

## How it works

**Park / place.** E1 schedules drones to take off and land. A drone that is not flying is
parked at its own spot (x = 1000 + 100·i m, y = 1000 m, z = 5 m), far beyond every sensor's
range. It is held in mode H (hover motor speed, no PID) and pinned there every step. To take
off, it is teleported onto its pad. It stays in H for one basic step, so the GPS reads the pad.
Then it gets F. The other drones are not affected. PyBullet uses the same park / place calls.

**Controller re-arm.** On F, the drone's onboard controller arms afresh and the drone departs
4 ms into that control step. The start and drop checks run for it as after a reset. At every
reset, a drone that tilted past 60° or ended too low is removed and imported again, so its
controller restarts with no leftover rotor speed.

**Layout reuse.** Buildings are rebuilt only when the layout changes
(`webots.reuse_layout`, default true). The crop and both conditions of an E1 trial keep their
nodes. Buildings are visual only, so this does not change the flight.

**Subprocess workers.** Webots allows one connection per Python process. Each route mode and
each E1 condition runs in a fresh Python worker. The worker writes its rows to a
`*.worker.json` file, and the parent merges them.

**Paired stats** (`scripts/c5_compare.py`). Pairs share seed, episode and flight id.
- Rates: 95% Wilson intervals.
- Yes/no metrics: exact McNemar per test, Holm over the tests of each metric. ALL rows are raw.
- Continuous metrics (cross-track p95, late_s): exact Wilcoxon signed-rank, Holm.
- Lost control (tilt > 60°, Webots only) counts as a failure: not settled / arrived, and
  counted in crash_or_lost.

## Validation

Gates passed before the runs (2026-10-28). Source: `eval_results/c5/RUN_NOTES.txt`.

1. **PyBullet before/after.** Seed 20, medium city, routes 3 episodes, E1 20 flights: results
   identical. Args differ only in the out path and the new simulator / modes keys.
2. **C1 regression.** determ / hover / delay / crash JSON byte-identical, old tree (2161648) vs
   HEAD. Determinism 0.0 m within and across processes. Delay new − old: +4 ms.
3. **Parity.** 40/40 (39 success + 1 collision in both, 0 lost control).
4. **C5 cases.** `c5_webots_city.py maps route place land schedule`: all pass.
5. **Speed.** See the table below.

After the rebase onto dev (E3 / E4), E1 medium seed 20 was re-run in both simulators. The
per-flight rows matched the stored part-4 rows, 36/36 flights each. The full test suite
passed (448 tests). Source: `eval_results/c5/RUN_NOTES.txt`, `eval_results/c5/rebase_check/`.

### C5 cases

Source: `logs/webots/c5/report_maps_route_place_place_ref_land_schedule.json`.

**Maps.** Every building of each map is in Webots, inside the 120 × 120 m floor.

| map | buildings | rotated | worst corner error (m) | worst height error (m) | worst area (rel) | reset, rebuild (s) | reset, reuse (s) |
|---|---|---|---|---|---|---|---|
| easy | 15 | 0 | 5.8e-05 | 1.4e-04 | 4.3e-05 | 0.18-0.32 | 0.041-0.042 |
| medium | 38 | 0 | 6.4e-05 | 1.4e-04 | 2.2e-04 | 0.24-0.38 | 0.042-0.045 |
| hard | 74 | 0 | 6.6e-05 | 1.5e-04 | 1.7e-04 | 0.36-0.63 | 0.047 |
| crop | 23 | 1 | 6.2e-05 | 0.0 | 3.4e-04 | 0.20-0.38 | 0.049-0.051 |

The layouts are still correct after reuse, with 0 rebuilds while reusing.

**Route.** One drone on the same 4-point route.

| sim | result | steps | cross-track median (m) | p95 (m) | max (m) | min wall gap (m) |
|---|---|---|---|---|---|---|
| Webots | success | 627 | 0.083 | 0.364 | 0.473 | 0.266 |
| PyBullet | success | 625 | 0.072 | 0.365 | 0.474 | 0.260 |

**Place.** After placing, and after a reset release: speed ≤ 2.2e-16 m/s in the first 6 steps.
Climb, drop and drift are 0.0 m. Max tilt is 0.0°. The drone's response track and the other
drones' tracks are unchanged (diff 0.0 m).

**Land.** A parked drone moves at most 6.0e-05 m per step (median 0.0). Climb and drop are
0.0 m.

**Schedule.** 3 drones, all succeed, none crash.

| sim | takeoff delay (s) | landed (s) | max airborne tilt (°) |
|---|---|---|---|
| Webots | 0.320, 0.336, 0.320 | 17.952, 19.968, 21.952 | 12.9 |
| PyBullet | 0.320, 0.272, 0.320 | 13.088, 15.040, 17.120 | - |

### Speed

Wall seconds per simulated second, with the policy. Source: `logs/webots/c5/report_speed.json`.

| drones | startup (s) | adapter only | env, chaser | env, policy |
|---|---|---|---|---|
| 4 | 1.16 | 0.019 | 0.121 | 0.13 |
| 8 | 1.22 | 0.038 | 0.211 | 0.22 |
| 12 | 1.29 | 0.065 | 0.301 | 0.34 |
| 16 | 1.38 | 0.096 | 0.134 | 0.46 |
| 20 | 1.39 | 0.130 | 0.204 | 0.57 |

The "env, chaser" values for 16 and 20 drones (0.134, 0.204) are what the file says. They are
lower than for 12 drones (0.301) because of how they are measured. In
`scripts/c5_webots_city.py` the chaser loop stops when the episode ends, but its wall time is
always divided by the full 10 s. An episode that ends early therefore gives a value that is
too low. The file does not record how many steps ran, so the chaser values for 16 and 20
drones should not be compared with the rest. The "env, policy" column divides by the steps
actually run, so it is not affected.

## Results

### City routes

Source: `eval_results/c5/c5_compare.md`. 100 episode pairs per map, 0 scenario-id mismatches.

Episode level:

| map | all settled W | all settled P | p | any crash or lost W | any crash or lost P | any stuck W / P |
|---|---|---|---|---|---|---|
| easy | 97.0% | 97.0% | 1.0000 | 0.0% | 0.0% | 0.0% / 0.0% |
| medium | 94.0% | 94.0% | 1.0000 | 1.0% | 1.0% | 4.0% / 4.0% |
| hard | 90.0% | 88.0% | 1.0000 | 2.0% | 2.0% | 5.0% / 5.0% |
| crop | 83.0% | 82.0% | 1.0000 | 4.0% | 5.0% | 5.0% / 5.0% |
| ALL | 91.0% | 90.2% | 0.5078 | 1.8% | 2.0% | 3.5% / 3.5% |

p is the exact McNemar test on all settled. Per-map p-values are Holm-adjusted over the 4 maps;
the ALL p-value is raw.

Per flight, settled (400 flights per map and sim): easy 99.2% / 99.2%, medium 97.8% / 97.8%,
hard 96.0% / 95.5%, crop 94.2% / 93.2% (Webots / PyBullet). Webots lost control in 0 of 1600
flights (`eval_results/c5/c5_compare.md`, City routes, Per flight table, lost control column:
0 on each map).

Cross-track error, p95 per flight (Wilcoxon, 400 pairs per map):

| map | median W (m) | median P (m) | median W−P (m) | p (Holm) |
|---|---|---|---|---|
| easy | 0.541 | 0.548 | -0.004 | 1.5e-07 |
| medium | 0.465 | 0.466 | -0.005 | 5.4e-08 |
| hard | 0.476 | 0.483 | -0.004 | 1.5e-05 |
| crop | 0.444 | 0.456 | -0.003 | 1.1e-05 |

Webots tracks the route 3-5 mm tighter. The difference is consistent but small.

### E1-flown

Source: `eval_results/c5/c5_compare.md`. E1 approvals are identical in both simulators.
Pairs: 88 flights per medium cell, 91 per crop cell, 358 in all.

| test | arrived W | arrived P | crash or lost W | crash or lost P | lost control W | zone entered W / P | close < 2.0 m W / P | close < 1.4 m W / P |
|---|---|---|---|---|---|---|---|---|
| medium/no_e1 | 97.7% | 98.9% | 1.1% | 0.0% | 1 | 36.4% / 35.2% | 46.6% / 45.5% | 37.5% / 37.5% |
| medium/e1 | 93.2% | 98.9% | 5.7% | 0.0% | 5 | 0.0% / 0.0% | 4.5% / 6.8% | 0.0% / 0.0% |
| crop/no_e1 | 100.0% | 97.8% | 0.0% | 2.2% | 0 | 23.1% / 22.0% | 39.6% / 33.0% | 33.0% / 24.2% |
| crop/e1 | 96.7% | 100.0% | 3.3% | 0.0% | 2 | 1.1% / 1.1% | 2.2% / 0.0% | 2.2% / 0.0% |
| ALL | 96.9% | 98.9% | 2.5% | 0.6% | 8 | 15.1% / 14.5% | 23.2% / 21.2% | 18.2% / 15.4% |

- Per cell (exact McNemar, Holm-adjusted over the 4 cells of each metric), only one test is
  below 0.05: close < 1.4 m on crop/no_e1 (33.0% vs 24.2%, 8 vs 0 discordant, Holm p = 0.0234).
- Pooled ALL rows (exact McNemar, raw, not Holm-adjusted): lost control p = 0.0078; arrived
  and crash_or_lost p = 0.0654; close < 1.4 m p = 0.0309.
- Episode level (5 pairs per cell, exact McNemar): no test is below 0.05. ALL
  any_crash_or_lost is 30.0% vs 5.0% (raw p = 0.1250).
- E1 works the same in both: zone entries fall from 36.4% / 23.1% to 0.0% / 1.1% in both simulators.

Lateness late_s, both settled (Wilcoxon):

| test | pairs | median W (s) | median P (s) | median W−P (s) | p (Holm) |
|---|---|---|---|---|---|
| medium/no_e1 | 86 | 3.075 | 1.630 | 1.890 | 2.7e-14 |
| medium/e1 | 82 | 2.710 | 0.990 | 1.815 | 8.8e-14 |
| crop/no_e1 | 89 | 2.540 | 0.810 | 1.890 | 7.6e-15 |
| crop/e1 | 88 | 2.125 | 0.405 | 1.875 | 1.0e-11 |

### The 8 flips

| map | seed | condition | flight | max tilt (°) | max true speed (m/s) | counted as crashed |
|---|---|---|---|---|---|---|
| medium | 20 | no_e1 | f7 | 180.0 | 6.2 | no |
| medium | 20 | e1 | f2 | 100.4 | 5.3 | yes |
| medium | 20 | e1 | f9 | 80.3 | 2.0 | yes |
| medium | 20 | e1 | f11 | 179.9 | 428.6 | no |
| medium | 22 | e1 | f10 | 179.1 | 6.6 | yes |
| medium | 23 | e1 | f4 | 179.9 | 373.9 | no |
| crop | 23 | e1 | f8 | 179.9 | 90.3 | yes |
| crop | 24 | e1 | f2 | 180.0 | 5.6 | no |

Source: `eval_results/c5/e1_wb_medium.rows.json`, `eval_results/c5/e1_wb_crop.rows.json`
(rows with max_tilt_deg > 60).

Three flips reach non-physical speeds (428, 374, 90 m/s). 4 of 8 are not counted as crashed.
Because of this, `c5_compare.py` counts lost control (tilt > 60°) as a failure.

### Flip analysis

Source: `eval_results/c5/RUN_NOTES.txt` (flip check, 14:30, scratchpad replays identical to
the part-4 rows).

- Pad conflicts at placement (an active drone < 0.5 m horizontally and < 1 m vertically from
  the pad): 0 in PyBullet (20 runs, 358 placements), and 0 in 6 replayed Webots runs
  (106 placements). The flips are not a park / place artefact.
- The flips happen in flight:
  - 6 of 8 around the final descent. 4 of these come just after the drone leaves
    LAND_RADIUS_M, where the carrot jumps back up and the vz command reverses.
  - 1 in the climb (medium s20 no_e1 f7).
  - 1 in low cruise.
- PyBullet tilt is 0 throughout.

### Sensitivity variants

Two landing options were added to `SwarmEnv` (off by default):
- `land_latch_m`: once landing, the carrot stays on the goal unless the drone drifts more
  than this from it.
- `land_vz_max`: while landing, the commanded vz is clipped to ± this value.

Variants, with the same E1 settings as part 4:
- A, `sens_latch24`: `--land-latch-m 2.4`.
- B, `sens_latch24_vz05`: `--land-latch-m 2.4 --land-vz-max 0.5`.

With the options off, E1 PyBullet output is byte-identical to HEAD.

Sources: `eval_results/c5/sens_latch24/{vs_webots_original,vs_pybullet_original,pybullet_vs_pybullet_original}/c5_compare.md`,
the same for `eval_results/c5/sens_latch24_vz05/`, and `eval_results/c5/c5_compare.md` for the original.

Webots, all 358 flights. p is the exact McNemar test on the pooled ALL row, raw (not
Holm-adjusted):

| variant | arrived | crash or lost | lost control | vs Webots original, crash or lost (only A / only B, p) | vs PyBullet original, crash or lost (p) |
|---|---|---|---|---|---|
| original | 96.9% | 2.5% | 8 | - | 2.5% vs 0.6% (0.0654) |
| A latch 2.4 | 98.0% | 1.4% | 4 | 0 / 4, 0.1250 | 1.4% vs 0.6% (0.4531) |
| B latch 2.4 + vz 0.5 | 99.2% | 0.3% | 1 | 0 / 8, 0.0078 | 0.3% vs 0.6% (1.0000) |

Webots lost control by cell:
- A: medium/no_e1 1, medium/e1 2, crop/e1 1.
- B: medium/no_e1 1 only. This is the climb flip, medium s20 no_e1 f7. Lost control, B vs
  Webots original, all 358 flights: 1 vs 8 (0 / 7 discordant), exact McNemar p = 0.0156,
  raw, not Holm-adjusted (`sens_latch24_vz05/vs_webots_original/c5_compare.md`).

Lateness, median late_s Webots variant − PyBullet original (s):

| variant | medium/no_e1 | medium/e1 | crop/no_e1 | crop/e1 |
|---|---|---|---|---|
| original | 1.890 | 1.815 | 1.890 | 1.875 |
| A | 1.890 | 1.800 | 1.890 | 1.890 |
| B | 1.540 | 1.470 | 0.040 | 1.340 |

Effect on PyBullet (variant vs PyBullet original):
- A: rates are unchanged. late_s changes in 2 pairs on crop/e1 (Wilcoxon, Holm p = 0.5000).
- B: 1 extra crash (medium s20 e1 f13; arrived 98.9% → 97.7%). The median late_s difference
  is +0.10 to +0.22 s (Wilcoxon, Holm p ≤ 8.1e-12).

## Limitations

- **Analytic LiDAR.** Webots Level 2 uses the analytic scan from `envs/webots_adapter.py`,
  the same as PyBullet. The buildings are visual only. A Webots-simulated LiDAR was not
  measured in C5. C5 tests flight dynamics and control, not sensing.
- **Display :1.** Every run used `DISPLAY=:1` (approved by Aman for C5),
  `OMP_NUM_THREADS=4`, `nice -n 19`.
- **5 seeds per E1 cell.** The episode-level E1 tests have only 5 pairs per cell. None is
  significant.
- **Per-flight p-values are optimistic.** Flights of one trial share the air, so they are not
  fully independent.
- **The remaining climb flip.** Variant B leaves 1 flip (medium s20 no_e1 f7). It happens in
  the climb, so the landing options cannot fix it.
- **Webots lands later.** With variant B, Webots still settles later than PyBullet by a
  median of 1.54 s (medium/no_e1), 1.47 s (medium/e1) and 1.34 s (crop/e1), and 0.04 s on
  crop/no_e1. In part 4 the gap is 1.8 to 1.9 s. The schedule check shows the same
  direction with a bigger gap, about 4.9 s per drone (landed 17.952 / 19.968 / 21.952 s in
  Webots vs 13.088 / 15.040 / 17.120 s in PyBullet: +4.864 / +4.928 / +4.832 s).
- **The sensitivity variants ran on uncommitted code.** They ran on HEAD 883d259 plus
  `eval_results/c5/sens_code.patch` (sha256 e0472f32…). The same landing options were later
  committed in PR #32 (c683b60 landing options, 290f1dd set_route latch fix). Compared with
  `git diff` (read only), the patch matches c683b60: the changed lines in its 4 files are
  identical, and c683b60's `tests/test_e1_landing_options.py` has the sha256 recorded in
  `RUN_NOTES.txt` (660e2925…). The only difference is 290f1dd's one line in
  `envs/swarm_env.py` (`set_route` resets `_landing[i]`), which is not in the patch.
  `set_route` comes from E4 and does not exist at 883d259, so the variants could not call it.

## How to reproduce

From the repo root, with the `utm-swarm-marl` conda env (the script calls its Python
explicitly). Each chunk writes a `.done` marker, so a restart only redoes the chunk that broke.

```bash
bash eval_results/c5/run_c5.sh pb          # all PyBullet chunks (routes 4 maps, E1 2 maps x 5 seeds) + E1 merges
bash eval_results/c5/run_c5.sh wb_crop     # Webots routes on the crop
bash eval_results/c5/run_c5.sh compare     # c5_compare.py on what exists (checked by hand)
bash eval_results/c5/run_c5.sh wb_routes   # Webots routes easy, medium, hard
bash eval_results/c5/run_c5.sh wb_e1       # Webots E1-flown (medium, crop x 5 seeds) + E1 merges
bash eval_results/c5/run_c5.sh compare     # final c5_compare.md / .csv

# sensitivity variants (NAME LATCH VZ; VZ '-' means no vz limit)
bash eval_results/c5/run_c5.sh sens sens_latch24 2.4 -
bash eval_results/c5/run_c5.sh sens sens_latch24_vz05 2.4 0.5

# check after a rebase: E1 medium seed 20, both simulators -> eval_results/c5/rebase_check/
bash eval_results/c5/run_c5.sh rebase_check
```

`eval_results/c5/RUN_NOTES.txt` has the exact per-chunk commands, the git commits and hashes,
the md5 sums, the timing (part 4: 12:44:47 to 13:31:51 IST; sensitivity: 15:03:52 to
15:48:36 IST), and the infrastructure fixes made during the runs. Per-chunk logs are in
`eval_results/c5/logs/`, and the chunk log is `eval_results/c5/driver.log`.
