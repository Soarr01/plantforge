#!/bin/bash
# Train seeds 1-4 (default: $SEEDS) for the 4 non-default held-out families
# (stribeck, saturate, boucwen, drivetrain -- corpus recipe) to 10000 steps
# each. backlash already has 5 seeds trained and is not retrained here.
# Resumable, stall-guarded, skips finished (seed, family) pairs without
# invoking evaluate.py.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_PARENT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PKG_PARENT"
export PLANTFORGE_DATA=${PLANTFORGE_DATA:-"$PKG_PARENT/plantforge_data"}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTHONUNBUFFERED=1
export PF_BUDGET=${PF_BUDGET:-500}
TOTAL=10000
SEEDS=${SEEDS:-"1 2 3 4"}
FAMILIES=${FAMILIES:-"stribeck saturate boucwen drivetrain"}

done_steps() {
    python - "$1" <<'EOF'
import sys, torch
try:
    print(torch.load(sys.argv[1], map_location="cpu")["step"])
except Exception:
    print(0)
EOF
}

for seed in $SEEDS; do
    for fam in $FAMILIES; do
        fam_upper=$(echo "$fam" | tr '[:lower:]' '[:upper:]')
        ck="$PLANTFORGE_DATA/eval_corpus_s${seed}_ho${fam_upper}.pt"
        export PF_SEED=$seed
        export PF_HOLD_FAMILY=$fam
        prev_steps=-1
        while true; do
            steps=$(done_steps "$ck")
            if [ "$steps" -ge "$TOTAL" ]; then
                echo "== seed $seed hold=$fam: done ($steps/$TOTAL)"
                break
            fi
            if [ "$steps" -le "$prev_steps" ]; then
                echo "== STALLED seed $seed hold=$fam: step stuck at $steps after a successful run"
                exit 1
            fi
            prev_steps=$steps
            echo "== seed $seed hold=$fam: at $steps/$TOTAL, training..."
            python -m plantforge.evaluate corpus \
                || { echo "== FAILED seed $seed hold=$fam"; exit 1; }
        done
    done
done
echo "== ALL SEEDS x FAMILIES DONE"
