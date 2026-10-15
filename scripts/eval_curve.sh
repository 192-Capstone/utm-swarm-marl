#!/bin/bash
# Learning curve of a long run: the 9-test, 500-flight suite (scripts/eval_500.sh)
# on every 500th RL episode's checkpoint, then one table.
#
# Usage (from the repo root):
#   bash scripts/eval_curve.sh <checkpoint dir> <name prefix> [parallel jobs, default 6]
#   bash scripts/eval_curve.sh checkpoints/stage2_critic2_long/seed1 critic2_long_s1 9
#
# Evaluates episodes 1100, 1600, 2100, 2600, 3100 and the final checkpoint (3650);
# any that don't exist yet are skipped, so it can be rerun while training goes on.
set -u
DIR=$1; PREFIX=$2; PAR=${3:-6}
for ep in 1100 1600 2100 2600 3100 final; do
  f=$DIR/checkpoint_ep$ep.pth; [ "$ep" = final ] && f=$DIR/checkpoint_final.pth
  if [ -f "$f" ]; then bash scripts/eval_500.sh "${PREFIX}_ep$ep" "$f" "$PAR"
  else echo "skip ${PREFIX}_ep$ep (no $f yet)"; fi
done
python - "$PREFIX" <<'EOF'
import json, glob, sys
p = sys.argv[1]
print(f"\n{'checkpoint':28s} {'tests':>5s} {'success':>8s} {'obst crash':>10s}")
for ep in ['1100', '1600', '2100', '2600', '3100', 'final']:
    rows = [json.load(open(f)) for f in glob.glob(f'eval_results/stage2_final_500/{p}_ep{ep}/*/summary.json')]
    if rows:
        n = sum(r['episodes'] for r in rows)
        s = sum(r['success_rate'] * r['episodes'] for r in rows) / n
        c = sum(r['obstacle_collision_rate'] * r['episodes'] for r in rows) / n
        print(f"{p + '_ep' + ep:28s} {len(rows):5d} {100 * s:7.1f}% {100 * c:9.1f}%")
print("reference: BCA 74.0%, pure RL + new critic after 500 RL episodes 75.6%")
EOF
