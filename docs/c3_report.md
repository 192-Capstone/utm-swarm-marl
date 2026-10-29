# C3 report: a Webots Lidar on the drones vs the analytic LiDAR

All numbers come from `eval_results/c3/`. Each table names its source file. Run commands, code hashes and
timings are in `eval_results/c3/RUN_NOTES.txt`.

## Summary

- C3 puts an optional Webots `Lidar` on every drone of Webots Level 2 (`webots_l2_multi`), off by default
  (`webots.lidar.enabled`). It checks the analytic LiDAR the policy was trained on and was flown with in C5.
- With the Lidar off, nothing changes: C1's per-case outputs and a full C5 route chunk are byte-identical
  to before.
- With the Lidar on but the policy still on the analytic scan, all 1600 C5 city-route flights are
  byte-identical to C5's stored rows (4 maps × 400). The Lidar does not change the flight.
- Buildings: the Lidar and the analytic scan agree closely. Where both hit, the median range difference is
  3.5 mm (p95 8.5 cm), and 0.78% of the analytic building hits are missed. Almost all of the difference
  comes from sampling (tilt, ray rounding, the 4 ms age), not from the building meshes: against a box model
  cast along the Lidar's own rays, the error is 0.3 mm median, 2.5 mm p95.
- Drones: the Lidar misses 89.6% of the rays that hit a drone in the analytic scan. This is geometry: a
  real Crazyflie is about 10 cm across, the analytic drone a 0.3 m box.
- The policy flying on the Lidar scan, paired against the analytic scan: no test is significant.
  4 drones, 400 episodes: all settled 90.8% vs 91.0% (p = 1.0). 12 drones, crop and hard, 120 episodes,
  where a pair came within 2 m in 116 of 120 Lidar episodes: all settled 68.3% vs 67.5%, any crash or
  lost control 15.8% vs 15.8% (both p = 1.0), although the Lidar missed 88.6% of the drone rays there.
- Speed: with the Lidar, env + policy runs at 0.222 wall s per simulated s with 4 drones (0.133 without)
  and 1.044 with 20 (0.575). Rendering in Webots is 72-86% of the extra; receiving the scans about 1%.

## What C3 is

C1 and C5 fly the drones in Webots, but their LiDAR is computed analytically
(`envs/webots_adapter.py` `scan_lidar`): 14 fan rays and 32 ring rays from the drone, turned to the goal
heading, 6 m range, rays starting 0.3 m out, buildings as boxes / cylinders, other drones as 0.3 m boxes,
the ground ignored. C3 measures what a simulated sensor sees instead, and whether the policy still flies
with it.

- Checkpoint: `checkpoints/stage2_r1_crowd_rl_r7v3ft_s0_lab/checkpoint_final.pth`
  (md5 87a12b5c3799ac302e51ae1d10d18c6a, the C5 checkpoint).
- Config: `configs/config_stage2_round7_eval_goalhold.yaml` (md5 1d96843067df8db17cecf3e1d2d889a2).
- City routes as in C5: `--modes route`, 4 drones, validation seeds 20-24 × 20 episodes, maps easy /
  medium / hard / crop. Dense check: 12 drones, seeds 20-22 × 20, maps crop and hard.
- Webots R2025a, Python 3.10.19. Code: commit ac77c28 + the uncommitted C3 changes
  (`eval_results/c3/c3_code.patch` and the new files listed in `RUN_NOTES.txt`).

## How it works

**Config.** `webots.lidar: {enabled: false, use_for_policy: false, horizontal_res: 1440, layers: 5, log: null}`
(`envs/webots_lidar.py`). With `enabled`, every drone gets a Lidar and an Emitter in its `extensionSlot` and
the controller argument `--lidar`. `use_for_policy: false` ("compare") logs the Lidar scan next to the
analytic scan, which the policy still gets; `true` gives the policy the Lidar scan.
`scripts/eval_city_routes.py --lidar compare|policy` sets it; without the option the run arguments are
unchanged. `webots.lidar` together with `webots.draw_zones` (C7) is refused with an error, because the
Lidar sees the semi-transparent zone volumes as walls (step 0 below).

**The device.** Fixed (not rotating), cylindrical projection, `fieldOfView` 2π, `verticalFieldOfView`
3.14159, 5 layers, 1440 columns, `near` = `minRange` = 0.3 m, `maxRange` = the LiDAR range (6 m), no noise.
No physics, no bounding object, so it adds no mass or contacts.

- 5 layers over 3.14159 give layers at +89.875°, +45.06°, 0°, -45.06°, -89.875° (layer 0 up).
  3.141593 is rejected by Webots.
- Column c looks at 180° - (c + 0.5) · 0.25° in the body frame (clockwise).
- `near` 0.3 m matters. With only `minRange`, anything closer than 0.3 m reads "nothing" and also hides
  what is behind it, so a Lidar inside a Crazyflie would be blinded by its own body. With `near` 0.3 m that
  geometry is clipped and what is behind it is seen, the same rule as the analytic rays that start 0.3 m out.

**One scan per control step.** With the Lidar on, the supervisor appends a scan tag `L<k>/<n>` to every
drone command ("the scan is read n basic steps after this was written"; `envs/lidar_msg.py`). The drone
sees a new command one basic step later, switches the Lidar on at step n-2 and reads it at n-1. The image
shows the scene 4 ms before the supervisor reads the state, and the packet arrives exactly when the
supervisor's 32 ms step returns. The Lidar renders once per control step and is off otherwise. At a reset
the held drones get a tag too, so the first observation has a scan taken on the spawn.

**Transport.** The drone sends the range image as raw float32 bytes (5 × 1440, 28.8 KB) with a 76-byte
header: drone, episode, tag, time, GPS position, IMU roll / pitch / yaw, and the controller's copy time.
Channel 102, infinite range, to the supervisor's Receiver `lidar_rx` (in the world file, inert when off).
The supervisor checks episode, tag and time; it counts stale, missing and mistimed scans.

**Ray picking** (`pick_rays`). Each of the 46 analytic rays (turned to the goal heading in the world) is
turned into the body frame by the scan's own IMU yaw only, and the nearest layer and column are read.
Roll and pitch are not compensated: the Lidar tilts with the drone, and that error is part of what C3
measures. Inf or beyond range reads as the range.

**Ground mask.** A hit whose point (from the full scan pose) is below 0.05 m is the ground and reads as
"nothing", as in the analytic scan.

**Not covered.** `place()` with `use_for_policy` raises (a just-placed drone has no scan yet; only E1 uses
it). A drone parked by `crash_mode remove` can still appear in the other drones' scans of that step.

## Validation

Source: `eval_results/c3/RUN_NOTES.txt` unless named.

### Step 0: what the Webots Lidar sees

A probe world with known geometry, the Lidar 3 m above the floor (`eval_results/c3/step0/`).

| check | expected (m) | Lidar (m) | source |
|---|---|---|---|
| floor straight down | 3.000 | 3.000 | run4.json |
| floor at -45° | 4.243 | 4.2426 | run4.json |
| ceiling straight up | 2.000 | 2.000 | run4.json |
| ceiling at +45° | 2.828 | 2.8284 | run4.json |
| `SimpleBuilding`, no bounding object | 3.000 | 3.000 | run4.json |
| opaque box | 3.000 | 3.000 | run4.json |
| transparency 0.45 box (wall behind at 5 m) | 3 if seen, 5 if not | 3.000 (seen) | run4.json |
| transparency 0.6 box (wall behind at 5 m) | 3 if seen, 5 if not | 3.000 (seen) | run4.json |
| wall 2.25 m away behind a box closer than 0.3 m, `near` 0.3 | 2.250 | 2.250 | run4.json |
| Crazyflie, centre 2 m away | ~1.95-2.0 | 1.969-1.996 (12 of the 13 columns read, ±1.5°) | run4.json |

- With a Crazyflie around the Lidar and `near` 0.3, every reading is clean (the drone's own body is invisible).
- Moving box: the reading at a step equals the true distance at that step (1e-5 m). A Lidar switched on at
  one step gives a current image at the next.
- Cost in the probe (ms per 4 ms basic step): no Lidar 0.077, 1 Lidar every step 0.597, 1 Lidar on 1 step
  in 8 0.092, 4 Lidars on 1 step in 8 0.236 (`run4.json`).

### Flag off: byte-identical

1. Tests: `tests/test_webots_lidar.py` and `tests/test_radio.py` 71 passed; the full suite 534 passed.
   - Node strings and commands are unchanged with the flag off (radio on or off).
   - `parse_controller_args` stops at the next `--` flag; radio-only parsing is unchanged (tested with and
     without `--lidar`).
   - One existing test changed: the controller's import list now includes `ctypes`, `time` and
     `envs.lidar_msg` (all standard library).
2. C1: `webots_l2_multi_speed.py --cases hover determ delay crash` on a clean copy of ac77c28 and on the C3
   tree: the 6 per-case JSONs are byte-identical (`eval_results/c3/c1_check/`). Determinism 0.0 m.
3. C5 chunk: crop routes, seeds 20-24 × 20, Lidar off: 400 rows byte-identical to
   `~/projects/utm-c5/eval_results/c5/routes_wb_crop.route.rows.json`
   (`eval_results/c3/off_check/routes_wb_crop.route.rows.json`). So C5's stored rows are a valid reference.

### Lidar on, policy on the analytic scan: tracks unchanged

Pass rule: every flight's row byte-identical to C5's stored `routes_wb_<map>.route.rows.json`.
Source: `eval_results/c3/driver.log` (PASS-CHECK lines).

| map | flights | byte-identical | scans | missing / stale / mistimed |
|---|---|---|---|---|
| easy | 400 | yes | 225564 | 0 / 0 / 0 |
| medium | 400 | yes | 226700 | 0 / 0 / 0 |
| hard | 400 | yes | 251920 | 0 / 0 / 0 |
| crop | 400 | yes | 286716 | 0 / 0 / 0 |

Scans: `eval_results/c3/c3_lidar_compare.md` (scan stats per map). "Mistimed" counts scans whose image
time is not exactly one basic step (4 ms) before the state; it is 0 everywhere, so every scan is 4 ms old.

## Results

### Per ray: Lidar vs analytic

Source: `eval_results/c3/c3_lidar_compare.md` / `.json` (compare-mode logs
`routes_analytic_<map>.route.lidar.npz`). 46 rays per flying drone and control step. A hit is a reading
below 6 m - 1 mm. "Missed": the analytic ray hit something, the Lidar read nothing (% of analytic hits).
"Extra": the Lidar hit something, the analytic ray did not (% of analytic no-hits). The analysis re-casts
the analytic scan from the logged poses and matches the logged values to 2.4e-7 m (float32).

By map:

| map | drone-steps | building rays | building missed | building error median / p95 (m) | drone rays | drone missed | extra, nothing | extra, ground | mean proximity error |
|---|---|---|---|---|---|---|---|---|---|
| easy | 225564 | 2318678 | 0.62% | 0.0032 / 0.071 | 2815 | 93.5% | 0.107% | 0.189% | 0.0013 |
| medium | 226700 | 3157770 | 0.86% | 0.0038 / 0.093 | 3153 | 89.3% | 0.228% | 0.481% | 0.0025 |
| hard | 251920 | 5414173 | 0.91% | 0.0040 / 0.103 | 6819 | 83.3% | 0.491% | 1.113% | 0.0044 |
| crop | 286716 | 2280371 | 0.50% | 0.0028 / 0.047 | 18174 | 91.3% | 0.046% | 0.169% | 0.0014 |
| ALL | 990900 | 13170992 | 0.78% | 0.0035 / 0.085 | 30961 | 89.6% | 0.184% | 0.459% | 0.0024 |

Proximity is 1 - d / 6, what the policy reads.

**Sampling vs geometry** (ALL). The reference is the box model cast from the scan pose along the exact
direction of the Lidar ray that was read (its layer and column, turned by the full roll / pitch / yaw).

| | building missed | building error median / p95 (m) | drone missed | extra, nothing | extra, ground |
|---|---|---|---|---|---|
| sampling: reference vs analytic | 0.67% | 0.0035 / 0.085 | 14.5% | 0.198% | 0.614% |
| geometry: Lidar vs reference | 0.20% | 0.0003 / 0.0025 | 89.3% | 0.020% | 0.004% |

- Buildings: the `SimpleBuilding` meshes match the boxes to millimetres. The building differences come from
  sampling.
- Extra hits (nothing / ground): also sampling. A tilted ray, or one rounded to the nearest column, sometimes
  meets a building the analytic ray just misses.
- Drones: geometry. The box model along the Lidar's rays still sees 85% of the drones; the Lidar sees 10%.

By ray group (ALL):

| group | rays | analytic hit | Lidar hit | missed | extra | error median / p95 (m) |
|---|---|---|---|---|---|---|
| fan horizontal | 3963600 | 29.17% | 29.05% | 0.57% | 0.064% | 0.002 / 0.028 |
| fan up | 990900 | 0.01% | 0.39% | 86 of 95 | 0.394% | 0.190 / 5.455 |
| fan down | 990900 | 0.01% | 0.01% | 92 of 95 | 0.013% | 0.126 / 0.127 |
| fan up 45° | 3963600 | 26.27% | 26.28% | 2.78% | 1.005% | 0.052 / 0.435 |
| fan down 45° | 3963600 | 13.50% | 12.85% | 8.58% | 0.588% | 0.041 / 0.301 |
| ring | 31708800 | 33.01% | 32.91% | 0.46% | 0.081% | 0.003 / 0.028 |

The ±45° rays meet buildings at a slant, so a small tilt moves the hit point far; they carry most of the
misses. Up and down rays hardly ever hit anything in the analytic scan (95 rays each).

By tilt at the scan (max of |roll|, |pitch|; ALL):

| tilt | rays | missed | extra | error median / p95 (m) | mean proximity error |
|---|---|---|---|---|---|
| 0-2° | 33323228 | 0.63% | 0.081% | 0.003 / 0.038 | 0.0011 |
| 2-5° | 8181376 | 1.21% | 0.409% | 0.005 / 0.163 | 0.0040 |
| 5-10° | 3061760 | 2.22% | 0.975% | 0.013 / 0.294 | 0.0079 |
| > 10° | 1015036 | 4.67% | 1.738% | 0.039 / 0.480 | 0.0136 |

**Drones by distance** (ALL, rays whose analytic ray hit a drone):

| analytic distance | rays | Lidar hit | missed |
|---|---|---|---|
| 0.3-1 m | 2848 | 16.1% | 84.0% |
| 1-2 m | 3755 | 22.9% | 77.1% |
| 2-4 m | 14133 | 12.2% | 87.8% |
| 4-6 m | 10225 | 1.8% | 98.2% |

Where the Lidar does hit something on a drone ray (3232 rays), it reads a median 1.75 m further than the
box: it mostly sees what is behind the drone. Drone rays are 0.07% of all rays (30961 of 45581400). The
policy also gets the other drones' positions as a separate input (neighbour observations), not only
through the LiDAR.

### Policy on the Lidar scan (4 drones)

Source: `eval_results/c3/policy_vs_analytic/c5_compare.md`. A = the policy on the Lidar scan
(`routes_lidar_<map>`), B = on the analytic scan (`routes_analytic_<map>`, the compare-mode run,
byte-identical to C5's Webots rows). 100 episode pairs per map, 0 scenario-id mismatches. Exact McNemar;
per-map p-values Holm-adjusted over the 4 maps, ALL raw.

| metric | Lidar | analytic | only A / only B | p (ALL) |
|---|---|---|---|---|
| all settled | 90.8% | 91.0% | 3 / 4 | 1.0000 |
| any crash or lost control | 2.2% | 1.8% | 3 / 1 | 0.6250 |
| any stuck | 3.8% | 3.5% | 2 / 1 | 1.0000 |
| any lost control | 0.0% | 0.0% | 0 / 0 | - |

Every per-map p-value is 1.0000. Per flight, settled (Lidar / analytic): easy 99.0% / 99.2%, medium
96.8% / 97.8%, hard 95.5% / 96.0%, crop 95.0% / 94.2%. Lost control 0 in both.

Cross-track error, p95 per flight (Wilcoxon, 400 pairs per map):

| map | median A (m) | median B (m) | median A-B (m) | p (Holm) |
|---|---|---|---|---|
| easy | 0.541 | 0.541 | -0.001 | 2.0e-06 |
| medium | 0.460 | 0.465 | -0.002 | 4.5e-08 |
| hard | 0.458 | 0.476 | -0.003 | 2.4e-06 |
| crop | 0.445 | 0.444 | -0.000 | 0.0025 |

The Lidar runs track the route 0-3 mm tighter: consistent but negligible.

How much traffic there was (`eval_results/c3/encounters_lidar.json`, the Lidar policy run): the nearest
other drone was within 2 m in 0.92% (medium) to 2.69% (crop) of drone-steps, within 1 m in 0.03%
(medium) to 0.58% (crop); 22-45 of 100 episodes per map had a pair closer than 2 m.
So the 4-drone result mostly tests buildings.

### Dense traffic (12 drones)

With 4 drones, drone rays are 0.07% of all rays, so the 4-drone result mostly tests buildings. A denser
check: route mode, 12 drones, maps crop and hard, seeds 20-22 × 20 episodes. A = the policy on the Lidar
scan (`--lidar policy`), B = the Lidar off (analytic scan), on the same episodes.
Source files are in `eval_results/c3/dense12/`.

**How much traffic** (`encounters_lidar.json`, the Lidar run; flying drones only):

| map | drone-steps | nearest neighbour < 6 m | < 2 m | < 1 m | episodes with a pair < 2 m | < 1 m | closest pair (m) |
|---|---|---|---|---|---|---|---|
| crop | 546132 | 62.3% | 6.97% | 1.20% | 60 of 60 | 55 of 60 | 0.17 |
| hard | 521472 | 32.9% | 3.39% | 0.88% | 56 of 60 | 46 of 60 | 0.14 |

For comparison, with 4 drones the nearest neighbour was within 2 m in 0.92-2.69% of drone-steps
(`eval_results/c3/encounters_lidar.json`).

**Drone rays** (`c3_lidar_compare_dense12.md` / `.json`, from the Lidar run's logs; the analytic side ran
with the Lidar off and has no log): 119563 rays hit a drone in the analytic scan, 0.24% of all rays
(crop 81679, 0.33%; hard 37884, 0.16%), 3.4 times the share with 4 drones. The Lidar missed 88.6% of them
(crop 93.6%, hard 77.9%). Geometry explains it, as with 4 drones: the box model along the Lidar's rays
misses 14.9%, the Lidar 88.5% of what that reference sees as a drone.

| analytic distance | drone rays | Lidar hit |
|---|---|---|
| 0.3-1 m | 11302 | 28.4% |
| 1-2 m | 14511 | 23.3% |
| 2-4 m | 53085 | 11.8% |
| 4-6 m | 40665 | 1.8% |

Buildings in dense traffic: missed 0.75%, error median 3.4 mm, p95 7.9 cm (as with 4 drones). Scans: 0
missing / stale / mistimed on both maps.

**Outcomes** (`policy_vs_analytic/c5_compare.md`). 60 episode pairs per map, 0 scenario-id mismatches.
Exact McNemar; per-map p Holm-adjusted over the 2 maps, ALL raw.

| metric | test | Lidar | analytic | only A / only B | p |
|---|---|---|---|---|---|
| all settled | hard | 76.7% | 73.3% | 3 / 1 | 1.0000 |
| all settled | crop | 60.0% | 61.7% | 3 / 4 | 1.0000 |
| all settled | ALL | 68.3% | 67.5% | 6 / 5 | 1.0000 |
| any crash or lost control | hard | 8.3% | 11.7% | 1 / 3 | 1.0000 |
| any crash or lost control | crop | 23.3% | 20.0% | 4 / 2 | 1.0000 |
| any crash or lost control | ALL | 15.8% | 15.8% | 5 / 5 | 1.0000 |
| any stuck | ALL | 6.7% | 7.5% | 0 / 1 | 1.0000 |
| any lost control | ALL | 0.0% | 0.0% | 0 / 0 | - |

Per flight (descriptive, 720 per map; flights of one episode share the air, so they are not independent):

| map | settled, Lidar / analytic | crashed, Lidar / analytic | stuck, Lidar / analytic |
|---|---|---|---|
| hard | 91.0% / 87.4% | 1.2% / 1.5% | 0.6% / 0.8% |
| crop | 83.1% / 84.7% | 3.1% / 2.4% | 0.7% / 0.7% |

Cross-track error, p95 per flight (Wilcoxon, 720 pairs per map): hard 0.440 vs 0.444 m (median A-B
-0.005 m, Holm p = 1.5e-13), crop 0.419 vs 0.431 m (-0.002 m, Holm p = 6.7e-08).

So in dense traffic, where drone detection is exercised (every crop episode has a pair closer than
2 m), the policy flying on a scan that reads anything at all on only about 11% of the drone rays it was
trained on does as well as on the analytic scan. No test is significant. A likely reason is that the policy also gets the
neighbours' relative positions directly (`observation.neighbor_encoding: attention`, up to 5
neighbours), so it does not depend on the LiDAR to see drones. This was not tested separately.

### Speed

Source: `eval_results/c3/speed/report.json` (`scripts/c3_lidar_speed.py`: the hard city, 10 simulated
seconds, as C5's speed case). Wall seconds per simulated second. Policy mode; compare mode costs the same
(within 2%).

| drones | env + policy, off | env + policy, Lidar | adapter alone, off → Lidar | receive | ray picking | drone copy (CPU) | MB/s |
|---|---|---|---|---|---|---|---|
| 4 | 0.133 | 0.222 | 0.020 → 0.084 | 0.0011 | 0.0121 | 0.0013 | 4 |
| 8 | 0.219 | 0.411 | 0.039 → 0.184 | 0.0019 | 0.0229 | 0.0029 | 7 |
| 12 | 0.327 | 0.617 | 0.066 → 0.288 | 0.0029 | 0.0335 | 0.0056 | 11 |
| 16 | 0.458 | 0.834 | 0.097 → 0.405 | 0.0036 | 0.0436 | 0.0096 | 14 |
| 20 | 0.575 | 1.044 | 0.131 → 0.533 | 0.0045 | 0.0547 | 0.0124 | 18 |

- "Adapter alone" is Webots plus the drone controllers with zero commands, so its increase is rendering
  and sending: about 0.020 s per drone per simulated second, about 0.64 ms per scan. It is 72% (4 drones)
  to 86% (20 drones) of the added env + policy time.
- Receiving and unpacking the raw bytes is 1.0-1.3% of the added time; picking the rays (and logging)
  11.6-13.6%. Picking the rays on the drone instead was not needed.
- The "off" column matches C5's speed table within about 4% (12 drones: 0.327 here vs 0.34;
  `docs/c5_report.md`, Speed).

## Limitations

- **Drones are mostly invisible to the Lidar.** The policy was trained on other drones as 0.3 m boxes; a
  real Crazyflie is about 10 cm across and the Lidar misses about 90% of those rays (89.6% with 4 drones,
  88.6% with 12; `eval_results/c3/c3_lidar_compare.json`,
  `eval_results/c3/dense12/c3_lidar_compare_dense12.json`). Outcomes did not change, probably because
  avoiding other drones relies on the neighbours' positions given directly to the policy
  (`observation.neighbor_encoding: attention`, up to 5 neighbours,
  `configs/config_stage2_round7_eval_goalhold.yaml`), i.e. on communication, not on the LiDAR. A drone
  that does not share its position (comms loss, D5; an intruder) would only be seen by a Lidar that misses
  about 89% of drone rays. Not tested here. Follow-up: the policy on the Lidar with one drone's neighbour
  inputs removed (or the existing blind pilot). An invisible 0.3 m stand-in box on each drone (no bounding
  object) was not tried either; it would need its own run.
- **Dense check is smaller.** 12 drones on 2 maps × 60 episodes; episode-level tests have 60 pairs per map.
- **Tilt.** The Lidar tilts with the drone and is not compensated; misses rise from 0.63% below 2° to
  4.67% above 10°.
- **The 4 ms age.** The scan is one basic step older than the state the analytic scan uses (about 8 mm
  at 2 m/s). It is part of the sampling difference.
- **No noise model.** The Lidar has no range noise or dropouts; a real sensor would.
- **5 layers.** Only the elevations the analytic scan uses are rendered. Layers at ±45.06° and
  ±89.875° are a little off ±45° / ±90°.
- **Rendering-based.** The Lidar sees everything that is drawn, including semi-transparent shapes, so it
  cannot be used with C7's `draw_zones`.
- **E1-flown not tested.** `place()` with the Lidar driving the policy is not supported.
- **Uncommitted code.** All runs used ac77c28 + the working tree (`eval_results/c3/c3_code.patch`).
- **Display :1.** Every run used `DISPLAY=:1`, `OMP_NUM_THREADS=4`, `nice -n 19`.

## How to reproduce

From the repo root, with the `utm-swarm-marl` conda env (the scripts call its Python explicitly).

```bash
# step 0 probe (scratch world, ~25 s)
DISPLAY=:1 OMP_NUM_THREADS=4 nice -n 19 python eval_results/c3/step0/probe.py

# tests
python -m pytest -q tests/test_webots_lidar.py tests/test_radio.py

# compare mode (pass rule vs C5's stored rows), then the policy on the Lidar; 4 maps each
bash eval_results/c3/run_c3.sh compare
bash eval_results/c3/run_c3.sh policy

# per-ray tables, paired outcomes, traffic
python scripts/c3_lidar_compare.py --root eval_results/c3 --prefix analytic
python scripts/c5_compare.py --root eval_results/c3 --a eval_results/c3:lidar --b eval_results/c3:analytic \
    --label-a lidar --label-b analytic --out eval_results/c3/policy_vs_analytic
python scripts/c3_encounters.py --root eval_results/c3 --prefix lidar

# speed
DISPLAY=:1 OMP_NUM_THREADS=4 nice -n 19 python scripts/c3_lidar_speed.py \
    --checkpoint checkpoints/stage2_r1_crowd_rl_r7v3ft_s0_lab/checkpoint_final.pth

# dense traffic, 12 drones (~44 min), then its analysis
bash eval_results/c3/run_c3_dense.sh
python scripts/c5_compare.py --root eval_results/c3/dense12 --a eval_results/c3/dense12:lidar \
    --b eval_results/c3/dense12:analytic --label-a lidar --label-b analytic \
    --out eval_results/c3/dense12/policy_vs_analytic
python scripts/c3_encounters.py --root eval_results/c3/dense12 --prefix lidar --maps crop hard
python scripts/c3_lidar_compare.py --root eval_results/c3/dense12 --prefix lidar --maps crop hard \
    --name c3_lidar_compare_dense12
```

`eval_results/c3/RUN_NOTES.txt` has the exact commands, hashes and timings.
