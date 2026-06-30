#!/usr/bin/env bash
# =====================================================================
# pipeline.slurm -- run the ASZED EEG pipeline on a SLURM GPU cluster
# =====================================================================
# Submit the whole pipeline as one GPU job:
#     sbatch pipeline.slurm
#
# Run only a sub-range of steps (e.g. just the DL + eval stages 7..8):
#     sbatch pipeline.slurm 7 8
#
# The CPU-heavy classical stages (3 tsfresh, 4 selection, 6 ML) and the
# GPU-heavy DL stage (7) can also be split into two jobs with a dependency:
#     cid=$(sbatch --parsable pipeline.slurm 1 6)      # CPU work
#     sbatch --dependency=afterok:$cid pipeline.slurm 7 8   # GPU work
# (Adjust --partition/--gres below per job if your cluster separates them.)
# =====================================================================

#SBATCH --job-name=aszed_eeg
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#SBATCH --partition=gpu             # <-- set to your cluster's GPU partition
#SBATCH --gres=gpu:1               # request 1 GPU (Module 7)
#SBATCH --cpus-per-task=8          # tsfresh n_jobs / dataloader workers
#SBATCH --mem=32G                  # raise for full dataset + EfficientFCParameters
#SBATCH --time=08:00:00

set -euo pipefail

# Land in the directory the job was submitted from.
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs

echo "[slurm] job $SLURM_JOB_ID on $(hostname)"
echo "[slurm] submit dir: $SLURM_SUBMIT_DIR"
nvidia-smi || echo "[slurm] (no GPU visible at submit time)"

# ---- environment ----------------------------------------------------
# Uncomment / adapt the module loads your cluster needs:
# module purge
# module load python/3.12 cuda/12.1 cudnn

# Activate the project virtual environment (.eeg). Create + install once with:
#     python -m venv .eeg && ./.eeg/bin/pip install -r requirements.txt
if [[ -f "./.eeg/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source ./.eeg/bin/activate
    PY=python
else
    echo "[slurm] .eeg venv not found; falling back to system python3" >&2
    PY=python3
fi
echo "[slurm] interpreter: $($PY -c 'import sys; print(sys.executable)')"

# Let tsfresh / dataloaders use all allocated CPUs.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

# ---- run the pipeline ----------------------------------------------
# Delegate ordering + per-step logging to run_all.sh (forward any step range).
PYTHON="$PY" bash ./run_all.sh "$@"

echo "[slurm] job $SLURM_JOB_ID finished."
