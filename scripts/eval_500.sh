#!/bin/bash
# 9-test, 500-flight evaluation of one checkpoint (seeds 10-14 x 100 episodes per
# test, deterministic policy, drone-drone contact physics off) -- the same suite
# as every model in eval_results/stage2_final_500/.
#
# Usage (from the repo root):
#   bash scripts/eval_500.sh <name> <checkpoint> [parallel jobs, default 6] [config]
#   bash scripts/eval_500.sh critic2_long_s1_ep2100 checkpoints/stage2_critic2_long/seed1/checkpoint_ep2100.pth 9
#
# Results: eval_results/stage2_final_500/<name>/, one folder per test.
# Resumable: tests that already have a summary.json are skipped.
set -u
NAME=$1; CKPT=$2; PAR=${3:-6}; CFG=${4:-configs/config_stage2_ring.yaml}
OUT=eval_results/stage2_final_500/$NAME
LOGS=logs/eval_500/$NAME
mkdir -p "$LOGS"
export PYTHONIOENCODING=utf-8
[ -f "$CKPT" ] || { echo "no checkpoint: $CKPT"; exit 1; }
todo=""
for t in pillar building wide_building long_wall random headon crossing swap flythrough; do
  ls $OUT/*_${t}_*/summary.json >/dev/null 2>&1 && continue
  todo="$todo $t"
done
echo "$(date '+%F %T')  $NAME: tests to run:${todo:- none}"
for t in $todo; do echo $t; done | xargs -P "$PAR" -I{} bash -c \
  "python eval_suite.py --config $CFG --policy mappo --checkpoint $CKPT --scenario {} \
     --seeds 10 11 12 13 14 --episodes 100 --collision-physics off --results-root $OUT \
     > $LOGS/{}.log 2>&1 && echo \"\$(date '+%T')  done    $NAME {}\" || echo \"\$(date '+%T')  FAILED  $NAME {} (see $LOGS/{}.log)\""
python - "$OUT" <<'EOF'
import json, glob, sys
rows = [json.load(open(f)) for f in glob.glob(sys.argv[1] + '/*/summary.json')]
if rows:
    n = sum(r['episodes'] for r in rows)
    s = sum(r['success_rate'] * r['episodes'] for r in rows) / n
    c = sum(r['obstacle_collision_rate'] * r['episodes'] for r in rows) / n
    print(f"{sys.argv[1]}: {len(rows)}/9 tests, success {100 * s:.1f}%, obstacle crashes {100 * c:.1f}%")
EOF
