#!/bin/bash
# Train eval_{wh_only,corpus}_s{SEED}.pt for every seed in $SEEDS to 10000
# steps, resuming per PF_BUDGET-bounded attempts. Skips finished checkpoints
# WITHOUT invoking evaluate.py (its pool build costs minutes even when
# there is nothing left to train).
set -uo pipefail
cd /data/nas07_new/PersonalData/phuocthien
export PLANTFORGE_DATA=${PLANTFORGE_DATA:-/data/nas07_new/PersonalData/phuocthien/plantforge_data}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-6}
export PYTHONUNBUFFERED=1
export PF_BUDGET=${PF_BUDGET:-500}
SEEDS=${SEEDS:-"0 1 2 3 4"}
TOTAL=10000

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
    for mode in headline corpus; do
        if [ "$mode" = headline ]; then name=wh_only; else name=corpus; fi
        ck="$PLANTFORGE_DATA/eval_${name}_s${seed}.pt"
        prev_steps=-1
        while true; do
            steps=$(done_steps "$ck")
            if [ "$steps" -ge "$TOTAL" ]; then
                echo "== seed $seed $mode: done ($steps/$TOTAL)"
                break
            fi
            if [ "$steps" -le "$prev_steps" ]; then
                echo "== STALLED seed $seed $mode: step stuck at $steps after a successful run"
                exit 1
            fi
            prev_steps=$steps
            echo "== seed $seed $mode: at $steps/$TOTAL, training..."
            PF_SEED=$seed python -m plantforge.evaluate "$mode" \
                || { echo "== FAILED seed $seed $mode"; exit 1; }
        done
    done
done
echo "== ALL SEEDS DONE"
