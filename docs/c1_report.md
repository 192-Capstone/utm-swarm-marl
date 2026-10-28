# C1: Multi-drone Webots Level 2: report

*27 Oct 2026 · branch `c1-multi-l2` · validation seeds 20-24 only*

## Summary

The shipped policy (`r1_crowd_rl_r7v3ft_s0_lab`) transfers to realistic Crazyflie dynamics **without loss with 3 drones** and with a **small loss with 8 drones**:

| | PyBullet | Webots Level 2 | Paired McNemar |
|---|---|---|---|
| Standard, 4 paired tests (3 drones) | 96.0% | 96.0% | p = 1 |
| Standard, with goal-hold | 97.2% | 98.0% | p = 0.55 |
| Crowd, 4 tests (8 drones) | 95.5% | 92.8% | p = 0.061 |
| Crowd, with goal-hold | 95.0% | 91.8% | p = 0.035 |

- The crowd drop of about 3 points comes mostly from more drone-drone crashes in crowd crossing (91% vs 98%) and crowd flythrough. Loss of control (a flip) is rare: 0.2% of standard and 0.8% of crowd episodes.
- Goal-hold's benefit carries over to real quadrotor dynamics. In Webots, standard goes from 96.0% to 98.0% (8 flights saved, 0 lost, p = 0.008), and it is neutral in crowds, the same pattern as in PyBullet.
- Two problems were found and fixed on the way: a reset bug that let a crashed drone start the next episode with full motor speed (both Level-2 adapters, PR #17 and this branch), and a Webots start-up crash on the shared lab display (workaround: software OpenGL).

## 1. What C1 is

Level 2 so far flew one drone. Aman's single-drone adapter (`envs/webots_l2.py`) makes the drone itself the Webots supervisor, and Webots allows one robot per controller process, so it cannot fly several drones. C1 adds a multi-drone Level 2 so the 9 multi-drone suite tests can run on realistic quadrotor dynamics:

- **Standard 5 (3 drones, with obstacles):** random, head-on, crossing, swap, flythrough.
- **Crowd 4 (8 drones, no obstacles):** head-on, crossing, swap, flythrough.

## 2. How it works

- **One controller per drone.** Each official Crazyflie runs its own Webots controller (`webots/controllers/cf_velocity/cf_velocity.py`). Every 4 ms it reads its own IMU, gyro and GPS and runs Bitcraze's velocity PID. That PID is unchanged from single-drone Level 2, which now shares it via `envs/cf_velocity_pid.py`.
- **One supervisor.** A bodiless supervisor robot in the Python process (`envs/webots_l2_multi.py`) resets the scene, reads the true state for the policy and sends the commands.
- **Commands through `customData`.** Once per control step (32 ms) the supervisor writes `"<episode> <mode> vx vy vz"` into each drone's `customData` field. Mode `H` holds the drone at the spawn during the 2 s spin-up with the motors at hover speed; mode `F` flies. Empty means motors off. The yaw-rate command is always 0.
- **The rest is unchanged.** The analytic LiDAR is the same as in Levels 1 and 2. A new simulator option `webots_l2_multi` exists in `SwarmEnv`, `eval_suite.py --simulator` and `run_suite.py --simulator`. PyBullet output is byte-identical to before (checked on a smoke run against the old code).
- **Safety checks**, all of which raise an error naming the drone:
    - drop check in the first 0.5 s after release;
    - start check: faster than 1 m/s within 5 control steps of release;
    - controller-log check after every episode.
- **Lost control** per episode: max |roll| or |pitch| > 60°, recorded only for this simulator. True speed is not used, because diagonal commands reach about 2.8 m/s in controlled flight and speeds after a flip are not physical.

## 3. Validation of the setup

Every check ran before any policy results were taken.

| Check | Result |
|---|---|
| Parity with single-drone Level 2 (1 drone, 40 pillar episodes, shipped model) | Same outcome in **40/40** (39 successes, 1 collision in both). Track difference: median 0.013 m, max 0.47 m. |
| Cross-talk (3 drones, 10 m apart, different scripted commands vs each flown alone) | **0.0 m** difference |
| Determinism (same episode twice, and across processes) | **0.0 m** difference |
| Command delay vs single-drone Level 2 (step command) | Exactly **+4.0 ms** (one 4 ms step) at all thresholds; response to 0.05 m/s about 0.21 s in both |
| Hover: 0.5 m/s sideways for 1 s, 0.3 m/s up for 1 s, then hold | Moved 0.534 m / 0.297 m. Held to within 1.6 mm over the last 5 s. Max tilt 5.7°. |
| Malformed command to one drone | Caught at the next reset, naming the drone |
| Speed, 8 drones | Adapter alone 0.031 s wall per simulated second; full evaluation with policy about 0.27 s. One config of the 9 tests took about 25 min. |

After a climb, Bitcraze's altitude loop overshoots (to +0.42 m for a 0.30 m step) and settles in about 10 s. That is a property of the controller, the same in single-drone Level 2.

## 4. Problems found and fixed

### 4.1 Crashed drones restarted with full motor speed

The first full run showed 8% lost control in crowd crossing. Recording one seed showed that 3 of the 4 flips of one drone were not the policy's:

- After a flip, the drone lay on the ground for about 28 s while its controller kept receiving climb commands, so its motors saturated at 600 rad/s.
- Webots' `resetPhysics` does not reset rotor speed, and the official Crazyflie's motors ramp at only 30 rad/s² (measured exactly in Webots), so the 2 s spin-up hold left the rotors far too fast.
- At the next release the drone shot upward and flipped again. This also explains "speeds" of 97-426 m/s.

**Fixes:**

- **Multi-drone adapter:** any drone that lost control or ended below 0.1 m is removed and imported again before the next episode. The crash test went from 31-153 m/s at release to 0.0 m/s.
- **Single-drone adapter** (PR #17; the drone is the supervisor, so it cannot be re-imported): it tracks each rotor's speed from its own commands and holds the drone at the spawn until the rotors are back at hover. That is 2 s normally, and 17.5 s after a crash with the rotors at 582 rad/s. At release the drone then moved at 0.01-0.04 m/s and climbed 4 mm.
- **Both adapters:** a start check raises if a drone is faster than 1 m/s just after release.

Aman checked Part 4: its lost-control episodes all followed successful flights, so the single-drone results are not affected (its crash rule ends episodes before ground contact).

All Webots results in this report come from the rerun with the fix (`c1_wb2_*`). The first run (`c1_wb_*`) is kept for reference only.

### 4.2 Webots start-up crash on the shared lab machine

During the rerun, from 15:23, every Webots start failed (OpenGL out of memory, then a segfault) after the shared lab display (`:2`) changed. It was not the other team's GPU job, which started later. Running Webots with software OpenGL (`__GLX_VENDOR_LIBRARY_NAME=mesa LIBGL_ALWAYS_SOFTWARE=1`) fixed it. This only changes how Webots draws, not the physics, and the 10 remaining tests ran with it. The code was unchanged (checksums in `RUN_NOTES.txt`).

## 5. Results

Shipped model `checkpoints/stage2_r1_crowd_rl_r7v3ft_s0_lab/checkpoint_final.pth`, configs `config_stage2_round7_eval_paired.yaml` (without goal-hold) and `config_stage2_round7_eval_goalhold.yaml` (with goal-hold), validation seeds 20-24, 100 flights per test. Runs `c1_pb_*` (PyBullet) and `c1_wb2_*` (Webots) are paired on the same scenarios and the same sensor noise per episode. Statistics: exact McNemar per test with Holm correction; pooled rows uncorrected.

### 5.1 Strict success per test

| Test | PyBullet | Webots | PyBullet + GH | Webots + GH |
|---|---|---|---|---|
| Standard: random (unpaired, see 5.4) | 100% | 99% | 100% | 100% |
| Standard: head-on | 91% | 91% | 93% | 94% |
| Standard: crossing | 98% | 99% | 99% | 99% |
| Standard: swap | 96% | 96% | 98% | 99% |
| Standard: flythrough | 99% | 98% | 99% | 100% |
| Crowd: head-on | 100% | 99% | 100% | 99% |
| Crowd: crossing | 98% | 91% | 98% | 91% |
| Crowd: swap | 97% | 98% | 97% | 98% |
| Crowd: flythrough | 87% | 83% | 85% | 79% |

No single test differs significantly after Holm correction. The closest is crowd crossing, at 9 flights lost and 2 gained, p = 0.52.

### 5.2 Paired comparisons (pooled)

| Comparison | Group | A | B | Only A / only B | p |
|---|---|---|---|---|---|
| Webots vs PyBullet | standard, 4 paired tests | 96.0% | 96.0% | 7 / 7 | 1 |
| Webots vs PyBullet | crowd | 92.8% | 95.5% | 9 / 20 | 0.061 |
| Webots + GH vs PyBullet + GH | standard, 4 paired tests | 98.0% | 97.2% | 7 / 4 | 0.55 |
| Webots + GH vs PyBullet + GH | crowd | 91.8% | 95.0% | 10 / 23 | 0.035 |
| Goal-hold on vs off, Webots | standard, 4 paired tests | 98.0% | 96.0% | 8 / 0 | 0.008 |
| Goal-hold on vs off, Webots | crowd | 91.8% | 92.8% | 5 / 9 | 0.42 |

### 5.3 How the Webots flights failed

| Run | Group | Crash | Timeout | Lost control | Re-imported drones |
|---|---|---|---|---|---|
| Webots | standard | 1.8% | 1.6% | 0.2% | 1 |
| Webots | crowd | 6.2% | 1.0% | 0.8% | 3 |
| Webots + GH | standard | 1.6% | 0.0% | 0.2% | 1 |
| Webots + GH | crowd | 6.8% | 1.5% | 0.8% | 3 |
| PyBullet (for comparison) | crowd | 4.5% | 0.0% | n/a | n/a |

- **Crowd crossing in Webots:** 9 failures = 7 drone-drone crashes + 2 flips (both ended as "stuck" timeouts). So even preventing all flips would bring crossing only to about 93%. The gap to PyBullet is mainly closer encounters ending in crashes under real dynamics.
- **Each re-imported drone matches its own lost-control episode,** so there were no cascades after the fix.
- **For comparison, the first run with the reset bug** had crowd crossing at 88% with 8% lost control. The fix removed about 3 points of the gap, and the rest is genuine.

### 5.4 Standard random is compared unpaired

PyBullet's lag model (`envs/pybullet_adapter.py`, `reset_world`) draws per-episode delays from numpy's global random generator. The random test draws its scenarios from that same generator, and Webots ignores the lag model, so after episode 0 of each seed the two simulators fly different scenarios. Its rates are compared unpaired, and the pooled standard rows use the other 4 tests. All other tests paired with 0 scenario mismatches. The cause was left unchanged on purpose: fixing it would change the random-test scenarios of every earlier PyBullet result.

## 6. Limitations

- **No aerodynamic interaction:** Webots does not model downwash or propeller wake between drones, which is the biggest real-world gap for close 8-drone crowds.
- **True-state observations:** the policy sees the true position and velocity (no state estimator), and the LiDAR is the analytic scan, as in Levels 1 and 2.
- **Bitcraze's demo velocity controller:** its velocity-error derivative term reacts strongly to sudden command changes, a likely cause of the rare flips during sharp avoidance. The real Crazyflie firmware limits the requested lean angle (about 20°); adding that limit is the natural next experiment.
- **Command delay and yaw:** one extra 4 ms of command delay vs single-drone Level 2, and the yaw-rate command is always 0.
- **Validation seeds only:** these are validation-seed numbers. C1 is not part of the frozen test run unless it is merged before the freeze.

## 7. Reproduce

On the lab, from the repo root:

```
# checks
DISPLAY=:2 OMP_NUM_THREADS=4 nice -n 19 python scripts/webots_l2_multi_speed.py --cases hover crosstalk determ speed delay badcmd crash crash_l2
DISPLAY=:2 OMP_NUM_THREADS=4 nice -n 19 python scripts/webots_l2_multi_parity.py --checkpoint checkpoints/stage2_r1_crowd_rl_r7v3ft_s0_lab/checkpoint_final.pth

# suite runs (repeat with --name c1_wb2_goalhold --config configs/config_stage2_round7_eval_goalhold.yaml)
DISPLAY=:2 OMP_NUM_THREADS=4 nice -n 19 python scripts/run_suite.py --name c1_wb2_paired \
    --checkpoint checkpoints/stage2_r1_crowd_rl_r7v3ft_s0_lab/checkpoint_final.pth \
    --config configs/config_stage2_round7_eval_paired.yaml --groups standard crowd \
    --tests random headon crossing swap flythrough --seeds 20 21 22 23 24 --jobs 1 \
    --simulator webots_l2_multi --root eval_results/c1_val

# comparison
python scripts/c1_compare.py --webots c1_wb2_paired --webots-gh c1_wb2_goalhold
```

If Webots fails to start with an OpenGL error, prefix the Webots commands with `__GLX_VENDOR_LIBRARY_NAME=mesa LIBGL_ALWAYS_SOFTWARE=1`.

Results: `eval_results/c1_val/` on the lab (`c1_compare.md` / `.csv`, `RUN_NOTES.txt` with the git state of every run).
