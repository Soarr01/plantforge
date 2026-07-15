#!/bin/bash
# Train seed 0 for each of the 4 alternative held-out families
# (stribeck, saturate, boucwen, drivetrain -- corpus recipe) to 10000
# steps. The existing default (backlash, already 5-seed trained) is not
# retrained. Resumable, stall-guarded, skips finished families without
# invoking evaluate.py, same pattern as scripts/train_ablation.sh.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_PARENT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PKG_PARENT"
export PLANTFORGE_DATA=${PLANTFORGE_DATA:-"$PKG_PARENT/plantforge_data"}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTHONUNBUFFERED=1
export PF_BUDGET=${PF_BUDGET:-500}
export PF_SEED=0
TOTAL=10000
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

for fam in $FAMILIES; do
    fam_upper=$(echo "$fam" | tr '[:lower:]' '[:upper:]')
    ck="$PLANTFORGE_DATA/eval_corpus_s0_ho${fam_upper}.pt"
    export PF_HOLD_FAMILY=$fam
    prev_steps=-1
    while true; do
        steps=$(done_steps "$ck")
        if [ "$steps" -ge "$TOTAL" ]; then
            echo "== hold=$fam: done ($steps/$TOTAL)"
            break
        fi
        if [ "$steps" -le "$prev_steps" ]; then
            echo "== STALLED hold=$fam: step stuck at $steps after a successful run"
            exit 1
        fi
        prev_steps=$steps
        echo "== hold=$fam: at $steps/$TOTAL, training..."
        python -m plantforge.evaluate corpus \
            || { echo "== FAILED hold=$fam"; exit 1; }
    done
done
echo "== ALL HELD-OUT FAMILIES DONE"
