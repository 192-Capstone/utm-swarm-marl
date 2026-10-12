#!/usr/bin/env bash
# Evaluation suite v1 for the frozen Stage-1 baseline (spring-vortex-58).
# Evaluation only: no training, no config/checkpoint/normalizer changes.
#
# Usage (from the repo root, conda env active):
#   CKPT=checkpoints/stage1_v1/best_eval_policy.pth bash scripts/run_eval_suite_v1.sh phase1
#   other model: CKPT=... CFG=... RESULTS=eval_results/<name> bash scripts/run_eval_suite_v1.sh phase1
#   ... phase2   distance sweep
#   ... phase3   zero-shot agent scaling
#   ... phase4   neighbour-blanking ablation
#   ... phase5   two-drone interaction tests (intruder, blocker, neighbour)
#   ... all
# Phases are independent, so they can run in separate terminals at once.
# W&B logging is OFF unless you set USE_WANDB=1.

set -euo pipefail

CKPT=${CKPT:-checkpoints/stage1_v1/best_eval_policy.pth}
CFG=${CFG:-configs/config_3agent_review.yaml}
SEEDS=${SEEDS:-"0 1 2 3 4"}
RESULTS=${RESULTS:-eval_results/v1}   # one folder per model, e.g. eval_results/imitation_v2
WANDB=""
if [ "${USE_WANDB:-0}" = "1" ]; then WANDB="--wandb"; fi
echo "W&B logging: ${WANDB:-off}"

# -u: unbuffered, so progress shows up in log files as it happens
run() { echo; echo ">>> python eval_suite.py $*"; python -u eval_suite.py --config "$CFG" --results-root "$RESULTS" $WANDB "$@"; }

phase1() {
  # 3 agents, training distribution (1-2 m), 5 seeds x 100 = 500 episodes
  for phys in off on; do
    run --checkpoint "$CKPT" --seeds $SEEDS --episodes 100 --scenario random --collision-physics $phys
    run --policy pcontroller  --seeds $SEEDS --episodes 100 --scenario random --collision-physics $phys
  done
  # Structured conflicts, 5 seeds x 20 = 100 episodes each, MAPPO vs P-controller paired
  for sc in headon crossing swap; do
    for phys in off on; do
      run --checkpoint "$CKPT" --seeds $SEEDS --episodes 20 --scenario $sc --collision-physics $phys
      run --policy pcontroller  --seeds $SEEDS --episodes 20 --scenario $sc --collision-physics $phys
    done
  done
}

phase2() {
  # Distance sweep. 2 m is phase1's random run. Goals land in [d/2, d].
  for d in 5 10 20; do
    run --checkpoint "$CKPT" --seeds $SEEDS --episodes 20 --scenario random --goal-dist $d --collision-physics off
  done
}

phase3() {
  # Zero-shot scaling, same Stage-1 world, 5 seeds x 40 = 200 episodes each
  for n in 1 3 5 8 10; do
    run --checkpoint "$CKPT" --seeds $SEEDS --episodes 40 --scenario random --n-agents $n --collision-physics off
    run --policy pcontroller  --seeds $SEEDS --episodes 40 --scenario random --n-agents $n --collision-physics off
  done
  # Same, but with real crossing traffic (random 2 m tasks rarely force conflicts)
  for n in 5 8 10; do
    run --checkpoint "$CKPT" --seeds $SEEDS --episodes 20 --scenario swap --n-agents $n --collision-physics off
    run --policy pcontroller  --seeds $SEEDS --episodes 20 --scenario swap --n-agents $n --collision-physics off
  done
}

phase4() {
  # Neighbour-blanking ablation: same scenarios and seeds as phases 1 and 3,
  # but the actor sees no other drones. If collisions don't rise, the policy
  # isn't using its neighbour input.
  for sc in headon crossing; do
    run --checkpoint "$CKPT" --blind-neighbors --seeds $SEEDS --episodes 20 --scenario $sc --collision-physics off
  done
  run --checkpoint "$CKPT" --blind-neighbors --seeds $SEEDS --episodes 20 --scenario swap --n-agents 10 --collision-physics off
  run --checkpoint "$CKPT" --blind-neighbors --seeds $SEEDS --episodes 100 --scenario random --collision-physics off
}

phase5() {
  # Two-drone interaction tests: drone_1 is scripted, so a collision happens
  # unless the drone under test reacts. 5 seeds x 40 = 200 episodes each.
  for sc in intruder blocker neighbour; do
    run --checkpoint "$CKPT" --seeds $SEEDS --episodes 40 --scenario $sc --collision-physics off
    run --checkpoint "$CKPT" --blind-neighbors --seeds $SEEDS --episodes 40 --scenario $sc --collision-physics off
    run --policy pcontroller  --seeds $SEEDS --episodes 40 --scenario $sc --collision-physics off
  done
}

case "${1:-}" in
  phase1) phase1 ;;
  phase2) phase2 ;;
  phase3) phase3 ;;
  phase4) phase4 ;;
  phase5) phase5 ;;
  all)    phase1; phase2; phase3; phase4; phase5 ;;
  *) echo "usage: $0 {phase1|phase2|phase3|phase4|phase5|all}"; exit 1 ;;
esac

echo; echo "done. summary table: python eval_report.py"
