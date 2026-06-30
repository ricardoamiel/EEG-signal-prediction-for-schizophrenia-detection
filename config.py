"""
config.py
=========
Single source of truth for every global parameter of the ASZED EEG
schizophrenia-classification pipeline (controls = 0, patients = 1).

Every numbered module (`1_preprocessing.py`, `3_feature_extraction.py`, ...)
imports this file and writes/reads its artifacts under :data:`ARTIFACTS_DIR`,
so each script can run standalone:

    python 1_preprocessing.py
    python 3_feature_extraction.py

Nothing here has side effects beyond creating the artifact directories, so it
is safe to `import config` from anywhere.
"""

from __future__ import annotations

import os
from typing import Dict, Final, List

# =====================================================================
# 0. PROJECT ROOT & ARTIFACT LAYOUT
# =====================================================================
PROJECT_ROOT: Final[str] = os.path.dirname(os.path.abspath(__file__))

# Raw dataset -------------------------------------------------------------
BASE_DATASET_DIR: Final[str] = os.path.join(PROJECT_ROOT, "ASZED-153")
CSV_PATH: Final[str] = os.path.join(BASE_DATASET_DIR, "ASZED_SpreadSheet.csv")

# All generated data lives under ARTIFACTS_DIR, one sub-folder per module.
ARTIFACTS_DIR: Final[str] = os.path.join(PROJECT_ROOT, "artifacts")

DIR_PREPROCESS: Final[str] = os.path.join(ARTIFACTS_DIR, "01_preprocessing")
DIR_EDA: Final[str] = os.path.join(ARTIFACTS_DIR, "02_eda")
DIR_FEATURES: Final[str] = os.path.join(ARTIFACTS_DIR, "03_features")
DIR_FEATURES_CACHE: Final[str] = os.path.join(DIR_FEATURES, "tsfresh_cache")
DIR_FEATSELECT: Final[str] = os.path.join(ARTIFACTS_DIR, "04_feature_selection")
DIR_TIMEFREQ: Final[str] = os.path.join(ARTIFACTS_DIR, "05_time_frequency")
DIR_ML: Final[str] = os.path.join(ARTIFACTS_DIR, "06_ml_models")
DIR_DL: Final[str] = os.path.join(ARTIFACTS_DIR, "07_dl_models")
DIR_EVAL: Final[str] = os.path.join(ARTIFACTS_DIR, "08_evaluation")
# Every training module (6, 7) drops its final TEST predictions here as
# preds_<model>.npz; Module 8 globs them to build the comparison table.
DIR_PREDICTIONS: Final[str] = os.path.join(DIR_EVAL, "predictions")

_ALL_DIRS: Final[List[str]] = [
    ARTIFACTS_DIR, DIR_PREPROCESS, DIR_EDA, DIR_FEATURES, DIR_FEATURES_CACHE,
    DIR_FEATSELECT, DIR_TIMEFREQ, DIR_ML, DIR_DL, DIR_EVAL, DIR_PREDICTIONS,
]


def ensure_dirs() -> None:
    """Create every artifact directory (idempotent). Call once at start-up."""
    for d in _ALL_DIRS:
        os.makedirs(d, exist_ok=True)


# =====================================================================
# 1. SIGNAL / PREPROCESSING PARAMETERS
# =====================================================================
SFREQ: Final[int] = 250                  # target sampling rate (Hz) after resample
WINDOW_SEC: Final[float] = 5.0           # fixed window length (seconds)
SAMPLES_PER_WINDOW: Final[int] = int(WINDOW_SEC * SFREQ)   # 1250 samples

FREQ_LOW: Final[float] = 0.5             # band-pass high-pass edge (Hz)
FREQ_HIGH: Final[float] = 45.0           # band-pass low-pass edge (Hz)
FREQ_NOTCH: Final[float] = 50.0          # mains notch (Hz); use 60.0 in NTSC regions

ZSCORE_EPS: Final[float] = 1e-8          # numerical floor for per-channel std

# Clinical frequency bands (Hz) used by EDA, FFT band-power and CWT modules.
FREQ_BANDS: Final[Dict[str, tuple[float, float]]] = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}

# =====================================================================
# 2. CHANNEL DEFINITIONS (10-20 montage, harmonised across both ASZED formats)
# =====================================================================
# ASZED ships two acquisition formats that share the same 19 scalp electrodes
# but name them differently ('EEG Fp1-LE' vs 'Fp1[1]'). normalize_channel_name()
# in 1_preprocessing.py harmonises both to this canonical 19-channel order; all
# saved tensors are stored in exactly this order.
CHANNELS_19: Final[List[str]] = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
    "T3", "C3", "Cz", "C4", "T4",
    "T5", "P3", "Pz", "P4", "T6", "O1", "O2",
]

# Top 18: the full montage minus Cz (geometric centre, p-value ~0.7 in the
# per-channel t-test -> not discriminative between classes).
CHANNELS_18: Final[List[str]] = [c for c in CHANNELS_19 if c != "Cz"]

# Top 6: the most discriminative electrodes found by the per-channel t-test in
# the EDA (all frontal/temporal: the regions altered in schizophrenia).
CHANNELS_6: Final[List[str]] = ["F4", "F3", "F7", "T3", "T4", "Fp1"]

# Map a logical name -> concrete channel list, so every module can switch the
# working montage from a single flag (CHANNEL_SET below).
CHANNEL_SETS: Final[Dict[str, List[str]]] = {
    "top6": CHANNELS_6,
    "top18": CHANNELS_18,
    "all19": CHANNELS_19,
}

# Active montage for feature extraction / image generation. Default "top6"
# keeps tsfresh + wavelet costs tractable; switch to "top18"/"all19" on a
# cluster with more RAM.
CHANNEL_SET: Final[str] = "top18"


def active_channels() -> List[str]:
    """Return the channel-name list selected by :data:`CHANNEL_SET`."""
    if CHANNEL_SET not in CHANNEL_SETS:
        raise ValueError(
            f"CHANNEL_SET={CHANNEL_SET!r} invalid; expected one of "
            f"{list(CHANNEL_SETS)}"
        )
    return CHANNEL_SETS[CHANNEL_SET]


def channel_indices(channels: List[str]) -> List[int]:
    """Map channel names to their column index in the canonical 19-ch order.

    Tensors are always saved in CHANNELS_19 order, so this is how downstream
    modules slice a sub-montage out of the stored arrays.
    """
    missing = [c for c in channels if c not in CHANNELS_19]
    if missing:
        raise ValueError(f"Channels not in canonical montage: {missing}")
    return [CHANNELS_19.index(c) for c in channels]


def pfx(name: str) -> str:
    """Tag an output filename with the active channel-set for ablation studies.

    e.g. with CHANNEL_SET='top6':  'X_stft_train_norm.npy' -> 'top6_X_stft_train_norm.npy'.
    Used by every CHANNEL_SET-dependent module (3-8) so a top6 run and a top18
    run leave their .npy/.parquet/.pt/.pdf/.csv side by side instead of
    overwriting each other.

    IMPORTANT: the per-channel tsfresh cache (feats_<split>_<channel>.parquet)
    is deliberately NOT passed through pfx(): it is keyed by channel name and is
    shared across runs, so switching top6 -> top18 reuses the already-extracted
    channels. Only the *fused* feature tables get the prefix.

    Module 1 (raw 19-channel tensors / splits) and Module 2 (EDA) are also left
    unprefixed: their outputs are identical regardless of CHANNEL_SET.
    """
    return f"{CHANNEL_SET}_{name}"


# =====================================================================
# 3. DATA-SPLIT PARAMETERS  (STRICT subject-level partition)
# =====================================================================
TRAIN_FRAC: Final[float] = 0.70
VAL_FRAC: Final[float] = 0.15
TEST_FRAC: Final[float] = 0.15
RANDOM_STATE: Final[int] = 42

# =====================================================================
# 4. FEATURE-EXTRACTION PARAMETERS (Module 3)
# =====================================================================
# tsfresh parameter set: "minimal" (~10 feats/ch, fast) | "efficient" | "comprehensive".
# Iterative channel-by-channel extraction + gc keeps even "efficient" within RAM.
TSFRESH_FC_PARAMETERS: Final[str] = "efficient"
# n_jobs for tsfresh. 0 = no multiprocessing (lowest RAM). Channel-by-channel
# extraction already bounds peak memory to a single channel, so we parallelise
# WITHIN each channel: ~3-4x faster on this 8-core box (13 -> ~4 min/channel).
# Drop to 2 if you hit memory pressure; raise toward $SLURM_CPUS_PER_TASK on a
# cluster node with more RAM.
TSFRESH_N_JOBS: Final[int] = 4

# ---- Shapelets (tslearn.LearningShapelets) ----
SHAPELETS_ENABLE: Final[bool] = True
# Shapelet learning is O(n_windows * length); subsample windows to stay viable.
SHAPELETS_MAX_TRAIN_WINDOWS: Final[int] = 600
SHAPELETS_MAX_ITER: Final[int] = 100         # hard cap on optimisation iters
SHAPELETS_TIME_LIMIT_SEC: Final[int] = 600   # wall-clock budget for the fit
SHAPELETS_N_PER_SIZE: Final[int] = 4         # shapelets per learned length

# =====================================================================
# 5. FEATURE-SELECTION FUNNEL (Module 4)
# =====================================================================
VARIANCE_THRESHOLD: Final[float] = 0.0       # drop (near-)constant features
CORRELATION_THRESHOLD: Final[float] = 0.85   # drop one of any pair with |r| > this
# Model-based selection: how to cut the SHAP-ranked features.
#   "threshold" -> keep features with normalised aggregate |SHAP| >= SHAP_THRESHOLD
#   "topk"      -> keep the TOP_K_FEATURES highest-ranked features
# NOTE: SHAP_aggregate is the LightGBM+RF mean|SHAP| normalised per-model to
# [0,1] then averaged, so SHAP_THRESHOLD is on that 0..1 scale (the top feature
# is always 1.0). 0.1 keeps roughly the strongest ~50 features on the top-6 set.
FEATURE_SELECTION_MODE: Final[str] = "threshold"
SHAP_THRESHOLD: Final[float] = 0.1
TOP_K_FEATURES: Final[int] = 100             # used only when MODE == "topk"
MIN_FEATURES: Final[int] = 10                # safety floor if a threshold is too strict
PCA_MAX_FEATURES: Final[int] = 100           # apply PCA if still above this count
PCA_VARIANCE_KEEP: Final[float] = 0.95       # variance retained by PCA

# =====================================================================
# 6. TIME-FREQUENCY IMAGE PARAMETERS (Module 5)
# =====================================================================
STFT_NPERSEG: Final[int] = 128
STFT_NOVERLAP: Final[int] = 64
CWT_N_FREQS: Final[int] = 40                  # vertical resolution of scalograms
CWT_WAVELET: Final[str] = "cmor1.5-1.0"       # complex Morlet (B=1.5, C=1.0)
# Wavelet/CWT is expensive: restrict to the Top-6 montage as required.
CWT_CHANNEL_SET: Final[str] = "top6"

# =====================================================================
# 7. DEEP-LEARNING PARAMETERS (Module 7)
# =====================================================================
USE_GPU: Final[bool] = True                   # fall back to CPU if unavailable
BATCH_SIZE: Final[int] = 32
DL_N_EPOCHS: Final[int] = 30
DL_LR: Final[float] = 1e-3
DL_FINETUNE_LR: Final[float] = 1e-4           # smaller LR for transfer learning
DL_PATIENCE: Final[int] = 6                   # early-stopping patience
DL_RESIZE: Final[tuple[int, int]] = (64, 64)  # spectrogram resize for ImageNet nets
CHRONOS_MODEL_NAME: Final[str] = "amazon/chronos-t5-small"
# Default time-frequency representation for the image models (CNN/ResNet/ViT/
# EfficientNet). Override per-run with `python 7_train_dl_sota.py --repr cwt`
# (or 'all' / 'stft,cwt,fft') for the wavelet-vs-STFT-vs-FFT ablation.
DL_REPR: Final[str] = "stft"
# CWT scalograms are stored at full time resolution (e.g. 40 x 1250), which is
# ~1000x larger than an STFT spectrogram and would OOM when loaded for DL. The
# image models resize their input anyway, so decimate the CWT time axis to at
# most this many columns on load (memory-mapped, so the full array never
# materialises). Lower this if you still hit memory pressure.
CWT_DL_TIME_BINS: Final[int] = 128

# =====================================================================
# 8. EXECUTION FLAGS
# =====================================================================
# DEV_MODE: run on a tiny subset (DEV_N_SUBJECTS_PER_CLASS per class) to smoke-
# test the whole pipeline end-to-end in minutes before the full run.
DEV_MODE: Final[bool] = False
DEV_N_SUBJECTS_PER_CLASS: Final[int] = 1      # -> 2 subjects total in dev mode

VERBOSE: Final[bool] = True


def device_str() -> str:
    """Resolve the torch device string honouring USE_GPU and actual CUDA avail."""
    if not USE_GPU:
        return "cpu"
    try:
        import torch  # local import: config must not hard-depend on torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


# =====================================================================
# 9. SHARED METRICS & PREDICTION IO  (used by Modules 6, 7, 8)
# =====================================================================
# Canonical class labels for the binary task (control vs patient).
CLASS_NAMES: Final[List[str]] = ["Control", "Patient"]


def compute_metrics(y_true, y_pred, y_proba=None) -> dict:
    """Compute the project's full binary-classification metric set.

    Returns accuracy, precision, recall/sensitivity, specificity, F1, balanced
    accuracy, ROC-AUC and the confusion matrix (as a nested list, JSON-safe).
    """
    import numpy as _np
    from sklearn.metrics import (
        accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
        precision_score, recall_score, roc_auc_score,
    )

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else _np.nan
    try:
        roc = roc_auc_score(y_true, y_proba) if y_proba is not None else _np.nan
    except ValueError:
        roc = _np.nan  # single-class y_true (e.g. degenerate DEV split)

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall_sensitivity": recall_score(y_true, y_pred, zero_division=0),
        "specificity": float(specificity),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "roc_auc": float(roc),
        "confusion_matrix": cm.tolist(),
    }


def print_metrics(name: str, m: dict) -> None:
    """One-line metric summary for console logs."""
    print(f"[{name}] Acc={m['accuracy']:.3f} | F1={m['f1']:.3f} | "
          f"ROC-AUC={m['roc_auc']:.3f} | Sens={m['recall_sensitivity']:.3f} | "
          f"Spec={m['specificity']:.3f}")


def save_predictions(name: str, y_true, y_pred, y_proba, groups=None) -> str:
    """Persist a model's TEST predictions to ``preds_<name>.npz`` for Module 8.

    ``groups`` (per-window subject ids, aligned with the predictions) enables
    Module 8 to also report subject-level metrics. Omit it and Module 8 falls
    back to window-level only for that model.
    """
    import numpy as _np
    safe = name.replace(" ", "_").replace("/", "_")
    path = os.path.join(DIR_PREDICTIONS, pfx(f"preds_{safe}.npz"))
    _np.savez(
        path,
        name=name,
        y_true=_np.asarray(y_true),
        y_pred=_np.asarray(y_pred),
        y_proba=_np.asarray(y_proba) if y_proba is not None else _np.array([]),
        groups=_np.asarray(groups) if groups is not None else _np.array([]),
    )
    return path


# Convenience: create the artifact tree as soon as config is imported.
ensure_dirs()
