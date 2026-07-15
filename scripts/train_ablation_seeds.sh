#!/bin/bash
# Train seeds 1-4 (default: $SEEDS) for the 4 non-default architecture
# variants (narrow/wide/shallow/deep, corpus recipe) to 10000 steps each.
# The default architecture already has 5 seeds trained (0-4) from an earlier
# plan and is not retrained here. Resumable, stall-guarded, skips finished
# (seed, variant) pairs without invoking evaluate.py.
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

done_steps() {
    python - "$1" <<'EOF'
import sys, torch
try:
    print(torch.load(sys.argv[1], map_location="cpu")["step"])
except Exception:
    print(0)
EOF
}

# name width layers
VARIANTS="narrow:80:5 wide:320:5 shallow:160:2 deep:160:8"

for seed in $SEEDS; do
    for spec in $VARIANTS; do
        name=${spec%%:*}
        rest=${spec#*:}
        width=${rest%%:*}
        layers=${rest#*:}
        suffix=""
        if [ "$width" != "160" ] || [ "$layers" != "5" ]; then
            suffix="_d${width}L${layers}"
        fi
        ck="$PLANTFORGE_DATA/eval_corpus_s${seed}${suffix}.pt"
        export PF_SEED=$seed
        export PF_WIDTH=$width
        export PF_LAYERS=$layers
        prev_steps=-1
        while true; do
            steps=$(done_steps "$ck")
            if [ "$steps" -ge "$TOTAL" ]; then
                echo "== seed $seed $name (d=$width L=$layers): done ($steps/$TOTAL)"
                break
            fi
            if [ "$steps" -le "$prev_steps" ]; then
                echo "== STALLED seed $seed $name: step stuck at $steps after a successful run"
                exit 1
            fi
            prev_steps=$steps
            echo "== seed $seed $name (d=$width L=$layers): at $steps/$TOTAL, training..."
            python -m plantforge.evaluate corpus \
                || { echo "== FAILED seed $seed $name"; exit 1; }
        done
    done
done
echo "== ALL SEEDS x VARIANTS DONE"
