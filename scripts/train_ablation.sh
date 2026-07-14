#!/bin/bash
# Train the 4 non-default architecture variants (corpus recipe, seed 0) to
# 10000 steps, resuming per PF_BUDGET-bounded attempts. Skips finished
# variants WITHOUT invoking evaluate.py, same pattern as train_seeds.sh.
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

for spec in $VARIANTS; do
    name=${spec%%:*}
    rest=${spec#*:}
    width=${rest%%:*}
    layers=${rest#*:}
    suffix=""
    if [ "$width" != "160" ] || [ "$layers" != "5" ]; then
        suffix="_d${width}L${layers}"
    fi
    ck="$PLANTFORGE_DATA/eval_corpus_s0${suffix}.pt"
    export PF_WIDTH=$width
    export PF_LAYERS=$layers
    prev_steps=-1
    while true; do
        steps=$(done_steps "$ck")
        if [ "$steps" -ge "$TOTAL" ]; then
            echo "== $name (d=$width L=$layers): done ($steps/$TOTAL)"
            break
        fi
        if [ "$steps" -le "$prev_steps" ]; then
            echo "== STALLED $name: step stuck at $steps after a successful run"
            exit 1
        fi
        prev_steps=$steps
        echo "== $name (d=$width L=$layers): at $steps/$TOTAL, training..."
        python -m plantforge.evaluate corpus \
            || { echo "== FAILED $name"; exit 1; }
    done
done
echo "== ALL VARIANTS DONE"
