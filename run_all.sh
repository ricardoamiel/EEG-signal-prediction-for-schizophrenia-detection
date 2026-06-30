#!/usr/bin/env bash
# =====================================================================
# run_all.sh -- sequential local execution of the full ASZED EEG pipeline
# =====================================================================
# Runs Modules 1..8 in order, stopping at the first failure. Logs each stage
# to logs/<NN_module>.log (and echoes to the console).
#
# Usage:
#   ./run_all.sh                 # run the whole pipeline
#   ./run_all.sh 5               # start from step 5 (time-frequency)
#   ./run_all.sh 3 6             # run steps 3..6 inclusive
#   PYTHON=/path/to/python ./run_all.sh   # override the interpreter
#
# The interpreter is resolved in this order:
#   $PYTHON  ->  ./.eeg/bin/python  ->  python3
# =====================================================================
set -euo pipefail

cd "$(dirname "$0")"

# ---- resolve the Python interpreter ----
if [[ -n "${PYTHON:-}" ]]; then
    PY="$PYTHON"
elif [[ -x "./.eeg/bin/python" ]]; then
    PY="./.eeg/bin/python"
else
    PY="$(command -v python3)"
fi
echo "[run_all] Using interpreter: $PY"
"$PY" --version

# ---- ordered pipeline steps (index = step number) ----
STEPS=(
    ""                                 # 0 (unused, keep 1-based indexing)
    "1_preprocessing.py"
    "2_eda_and_dim_reduction.py"
    "3_feature_extraction.py"
    "4_feature_selection.py"
    "5_time_frequency_transforms.py"
    "6_train_ml.py"
    "7_train_dl_sota.py"
    "8_evaluation.py"
)
LAST=$(( ${#STEPS[@]} - 1 ))

START="${1:-1}"
END="${2:-$LAST}"

if (( START < 1 || END > LAST || START > END )); then
    echo "[run_all] Invalid range: START=$START END=$END (valid: 1..$LAST)" >&2
    exit 1
fi

mkdir -p logs
echo "[run_all] Running steps $START..$END"
echo "[run_all] Tip: set DEV_MODE=True in config.py for a fast smoke test first."

for (( i=START; i<=END; i++ )); do
    script="${STEPS[$i]}"
    log="logs/$(printf '%02d' "$i")_${script%.py}.log"
    echo ""
    echo "====================================================================="
    echo "[run_all] STEP $i/$LAST -> $script   (log: $log)"
    echo "====================================================================="
    start_ts=$(date +%s)
    if ! "$PY" "$script" 2>&1 | tee "$log"; then
        echo "[run_all] STEP $i FAILED ($script). See $log" >&2
        exit 1
    fi
    elapsed=$(( $(date +%s) - start_ts ))
    echo "[run_all] STEP $i done in ${elapsed}s"
done

echo ""
echo "[run_all] Pipeline finished successfully (steps $START..$END)."
echo "[run_all] Final results: comparacion_final_modelos_SOTA.csv / matrices_confusion_finales_SOTA.pdf"
