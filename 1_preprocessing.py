"""
1_preprocessing.py
==================
Phase 1 of the pipeline: turn the raw ASZED `.edf` recordings into clean,
windowed, z-scored tensors with a STRICT subject-level train/val/test split.

Faithfully reproduces the notebook logic (channel harmonisation across the two
ASZED acquisition formats, 0.5-45 Hz band-pass, 50 Hz notch, resample to 250 Hz,
5 s windows, per-channel z-score) and adds:
  * DEV_MODE support (smoke-test on a couple of subjects),
  * per-channel discriminability ranking saved to CSV (PSD + variance t-tests),
  * all outputs persisted to disk so every later module runs standalone.

Run:
    python 1_preprocessing.py

Outputs (under artifacts/01_preprocessing/):
    X.npy, y.npy, groups.npy                 full dataset (windows, 19, 1250)
    X_train.npy / X_val.npy / X_test.npy     subject-level splits
    y_train.npy / y_val.npy / y_test.npy
    groups_train.npy / groups_val.npy / groups_test.npy
    channel_ranking.csv                      per-channel t-test (PSD & variance)
    split_info.json                          subject ids per split + balance
    preprocessing_meta.json                  run parameters & discard reasons
"""

from __future__ import annotations

import glob
import json
import os
import re
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import config

# MNE is heavy and noisy; import lazily-friendly and silence runtime warnings.
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
import mne  # noqa: E402

mne.set_log_level("ERROR")

# Type alias for a stack of windows for one recording: (n_windows, n_ch, n_samp)
WindowStack = np.ndarray


# =====================================================================
# 1. CHANNEL-NAME HARMONISATION
# =====================================================================
def normalize_channel_name(name: str) -> str:
    """Normalise a raw channel name from either ASZED format to 10-20 convention.

    Strips the ``EEG `` prefix, the ``-LE`` reference suffix and a trailing
    ``[n]`` index, e.g. ``'EEG Fp1-LE' -> 'Fp1'`` and ``'Fp1[1]' -> 'Fp1'``.
    """
    n = name.strip()
    n = re.sub(r"^EEG\s+", "", n, flags=re.IGNORECASE)
    n = re.sub(r"-LE$", "", n, flags=re.IGNORECASE)
    n = re.sub(r"\[\d+\]$", "", n)
    return n


# =====================================================================
# 2. METADATA (subject -> binary diagnosis)
# =====================================================================
def load_metadata(csv_path: str) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Load the ASZED spreadsheet and build a {subject_id -> 0/1} mapping.

    ``category`` containing 'patient' -> 1 (schizophrenia), else 0 (control).
    Returns the cleaned DataFrame and the mapping dict.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Master CSV not found at {csv_path}")

    df_meta = pd.read_csv(csv_path)
    df_meta.columns = df_meta.columns.str.strip()
    df_meta["sn_clean"] = df_meta["sn"].astype(str).str.strip().str.lower()
    df_meta["category_clean"] = df_meta["category"].astype(str).str.strip().str.lower()
    df_meta["Diagnosis"] = df_meta["category_clean"].apply(
        lambda x: 1 if "patient" in x else 0
    )
    mapping = dict(zip(df_meta["sn_clean"], df_meta["Diagnosis"]))
    if config.VERBOSE:
        print(f"[✓] Metadata mapped: {len(mapping)} subjects in master CSV.")
    return df_meta, mapping


# =====================================================================
# 3. SINGLE-FILE PREPROCESSING
# =====================================================================
def preprocess_edf_file(
    edf_path: str,
    freq_low: float = config.FREQ_LOW,
    freq_high: float = config.FREQ_HIGH,
    freq_notch: float = config.FREQ_NOTCH,
    window_sec: float = config.WINDOW_SEC,
    target_sfreq: float = float(config.SFREQ),
) -> Tuple[Optional[WindowStack], Optional[float], Optional[str]]:
    """Load, harmonise, filter, resample and window a single ``.edf`` file.

    Returns ``(windows, sfreq, reason)`` where ``windows`` has shape
    ``(n_windows, 19, samples_per_window)`` and ``reason`` is ``None`` on success
    or a short string explaining why the file was discarded.
    """
    try:
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)

        # --- harmonise channel names to the canonical 19-ch montage ---
        rename_map: Dict[str, str] = {}
        for ch in raw.ch_names:
            norm = normalize_channel_name(ch)
            for canon in config.CHANNELS_19:
                if norm.lower() == canon.lower():
                    rename_map[ch] = canon
                    break
        raw.rename_channels(rename_map)

        available = [c for c in config.CHANNELS_19 if c in raw.ch_names]
        if len(available) < len(config.CHANNELS_19):
            missing = sorted(set(config.CHANNELS_19) - set(available))
            return None, None, f"missing channels: {missing}"

        # Keep exactly the 19 standard electrodes, always in canonical order.
        raw.pick_channels(config.CHANNELS_19, ordered=True)

        if raw.info["sfreq"] != target_sfreq:
            raw.resample(sfreq=target_sfreq, verbose=False)

        raw.filter(l_freq=freq_low, h_freq=freq_high, fir_design="firwin", verbose=False)
        raw.notch_filter(freqs=freq_notch, fir_design="firwin", verbose=False)

        data = raw.get_data()
        sfreq = float(raw.info["sfreq"])

        samples_per_window = int(window_sec * sfreq)
        num_windows = data.shape[1] // samples_per_window
        if num_windows == 0:
            return None, None, f"recording too short ({data.shape[1]} samples)"

        windows: List[np.ndarray] = []
        for w in range(num_windows):
            seg = data[:, w * samples_per_window:(w + 1) * samples_per_window]
            mean = np.mean(seg, axis=1, keepdims=True)
            std = np.std(seg, axis=1, keepdims=True) + config.ZSCORE_EPS
            windows.append((seg - mean) / std)

        return np.asarray(windows), sfreq, None
    except Exception as e:  # noqa: BLE001 - report, never crash the whole run
        return None, None, f"exception: {e}"


# =====================================================================
# 4. RECURSIVE DATASET LOADING
# =====================================================================
def discover_subject_folders(base_dir: str) -> List[str]:
    """Recursively find every ``subject_*`` / ``Subject_*`` directory."""
    folders = (
        glob.glob(os.path.join(base_dir, "**", "subject_*"), recursive=True)
        + glob.glob(os.path.join(base_dir, "**", "Subject_*"), recursive=True)
    )
    return sorted(set(f for f in folders if os.path.isdir(f)))


def _select_dev_subset(
    subject_folders: List[str], mapping: Dict[str, int], n_per_class: int
) -> List[str]:
    """Pick ``n_per_class`` subjects from each class for a fast DEV_MODE run."""
    by_class: Dict[int, List[str]] = {0: [], 1: []}
    for folder in subject_folders:
        name = os.path.basename(folder.rstrip(os.sep)).strip().lower()
        label = mapping.get(name)
        if label in (0, 1) and len(by_class[label]) < n_per_class:
            by_class[label].append(folder)
        if len(by_class[0]) >= n_per_class and len(by_class[1]) >= n_per_class:
            break
    subset = by_class[0] + by_class[1]
    if config.VERBOSE:
        print(f"[DEV_MODE] Restricting to {len(subset)} subjects: "
              f"{[os.path.basename(s) for s in subset]}")
    return subset


def build_dataset(
    base_dir: str, mapping: Dict[str, int]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Counter]:
    """Process every subject's ``.edf`` files into X, y, groups arrays.

    Returns ``(X, y, groups, discard_reasons)`` with
    ``X.shape == (n_windows, 19, samples_per_window)``.
    """
    subject_folders = discover_subject_folders(base_dir)
    if not subject_folders:
        raise RuntimeError(f"No subject_* folders found under {base_dir}")

    if config.DEV_MODE:
        subject_folders = _select_dev_subset(
            subject_folders, mapping, config.DEV_N_SUBJECTS_PER_CLASS
        )

    all_windows: List[np.ndarray] = []
    all_labels: List[int] = []
    all_subjects: List[int] = []
    discard_reasons: Counter = Counter()
    processed_files = 0
    expected_shape = (len(config.CHANNELS_19), config.SAMPLES_PER_WINDOW)

    print("\n[*] Loading and preprocessing recordings...")
    for sub_folder in subject_folders:
        folder_name = os.path.basename(sub_folder.rstrip(os.sep)).strip().lower()
        label = mapping.get(folder_name)
        if label is None:
            continue

        edf_files = sorted(set(
            glob.glob(os.path.join(sub_folder, "**", "*.edf"), recursive=True)
            + glob.glob(os.path.join(sub_folder, "**", "*.EDF"), recursive=True)
        ))
        try:
            sub_id_int = int(folder_name.split("_")[-1])
        except ValueError:
            sub_id_int = 999

        for edf_path in edf_files:
            processed_files += 1
            windows, _sfreq, reason = preprocess_edf_file(edf_path)
            if windows is None:
                discard_reasons[reason] += 1
                continue
            for window in windows:
                if window.shape != expected_shape:
                    discard_reasons[f"unexpected shape {window.shape}"] += 1
                    continue
                all_windows.append(window)
                all_labels.append(label)
                all_subjects.append(sub_id_int)

    print(f"[INFO] .edf files processed: {processed_files}")
    if discard_reasons:
        print("[INFO] Discarded files by reason:")
        for reason, count in discard_reasons.most_common():
            print(f"   - {reason}: {count}")

    if not all_windows:
        raise RuntimeError("No windows produced; check dataset path and formats.")

    return (
        np.asarray(all_windows),
        np.asarray(all_labels),
        np.asarray(all_subjects),
        discard_reasons,
    )


# =====================================================================
# 5. SUBJECT-LEVEL SPLIT  (no information leakage)
# =====================================================================
def subject_level_split(
    y: np.ndarray, groups: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, List[int]]]:
    """Split at SUBJECT level, stratified by diagnosis; return window masks.

    All windows of a subject land in exactly one of train/val/test. Returns
    ``(train_mask, val_mask, test_mask, split_ids)``.
    """
    from sklearn.model_selection import train_test_split

    subject_df = pd.DataFrame({"subject": groups, "label": y}).drop_duplicates()
    if not subject_df["subject"].is_unique:
        raise ValueError("A subject has conflicting labels across windows.")

    # In DEV_MODE the population is tiny; stratification may be infeasible.
    stratify_full = subject_df["label"] if subject_df["label"].nunique() > 1 else None

    if len(subject_df) < 3:
        # Degenerate (e.g. DEV_MODE with 2 subjects): put everything in train,
        # mirror into val/test so downstream modules still have all splits.
        ids = subject_df["subject"].tolist()
        train_ids = val_ids = test_ids = set(ids)
        print("[WARN] <3 subjects: train/val/test all share the same subjects "
              "(DEV smoke-test only, metrics are not meaningful).")
    else:
        train_subj, temp_subj = train_test_split(
            subject_df,
            test_size=(config.VAL_FRAC + config.TEST_FRAC),
            stratify=stratify_full,
            random_state=config.RANDOM_STATE,
        )
        stratify_temp = (
            temp_subj["label"] if temp_subj["label"].nunique() > 1 else None
        )
        val_subj, test_subj = train_test_split(
            temp_subj,
            test_size=config.TEST_FRAC / (config.VAL_FRAC + config.TEST_FRAC),
            stratify=stratify_temp,
            random_state=config.RANDOM_STATE,
        )
        train_ids = set(train_subj["subject"])
        val_ids = set(val_subj["subject"])
        test_ids = set(test_subj["subject"])

    train_mask = np.isin(groups, list(train_ids))
    val_mask = np.isin(groups, list(val_ids))
    test_mask = np.isin(groups, list(test_ids))

    split_ids = {
        "train": sorted(int(i) for i in train_ids),
        "val": sorted(int(i) for i in val_ids),
        "test": sorted(int(i) for i in test_ids),
    }
    return train_mask, val_mask, test_mask, split_ids


# =====================================================================
# 6. PER-CHANNEL DISCRIMINABILITY RANKING (saved for Module 2/3)
# =====================================================================
def rank_channels(
    X_train: np.ndarray, y_train: np.ndarray, channels: List[str]
) -> pd.DataFrame:
    """Rank channels by class separability via Welch-PSD power and variance.

    Mirrors the two ranking heuristics from the notebook and stores both p-values
    so later modules can choose a montage data-drivenly if desired.
    """
    from scipy.signal import welch
    from scipy.stats import ttest_ind

    X0, X1 = X_train[y_train == 0], X_train[y_train == 1]
    rows = []
    for ch_idx, ch_name in enumerate(channels):
        # PSD-power based t-test
        try:
            _, psd0 = welch(X0[:, ch_idx, :], fs=config.SFREQ, nperseg=256)
            _, psd1 = welch(X1[:, ch_idx, :], fs=config.SFREQ, nperseg=256)
            _, p_psd = ttest_ind(
                psd0.mean(axis=1), psd1.mean(axis=1), equal_var=False
            )
        except Exception:
            p_psd = np.nan
        # Variance based t-test
        try:
            _, p_var = ttest_ind(
                np.var(X0[:, ch_idx, :], axis=1),
                np.var(X1[:, ch_idx, :], axis=1),
                equal_var=False,
            )
        except Exception:
            p_var = np.nan
        rows.append({"channel": ch_name, "p_psd": p_psd, "p_var": p_var})

    df = pd.DataFrame(rows).sort_values("p_psd").reset_index(drop=True)
    df["rank_psd"] = df["p_psd"].rank(method="first")
    return df


# =====================================================================
# 7. MAIN
# =====================================================================
def main() -> None:
    config.ensure_dirs()
    out = config.DIR_PREPROCESS

    print("=" * 64)
    print(" PHASE 1 - EEG PREPROCESSING" + ("  [DEV_MODE]" if config.DEV_MODE else ""))
    print("=" * 64)

    _df_meta, mapping = load_metadata(config.CSV_PATH)
    X, y, groups, discard_reasons = build_dataset(config.BASE_DATASET_DIR, mapping)

    print("\n" + "=" * 64)
    print(f"[✓] X (windows, channels, samples): {X.shape}")
    print(f"[✓] Controls (class 0): {int(np.sum(y == 0))} windows | "
          f"Patients (class 1): {int(np.sum(y == 1))} windows")
    print(f"[✓] Unique subjects: {len(np.unique(groups))}")

    train_mask, val_mask, test_mask, split_ids = subject_level_split(y, groups)
    if not (train_mask.astype(int) + val_mask.astype(int)
            + test_mask.astype(int) == 1).all() and len(np.unique(groups)) >= 3:
        raise AssertionError("Window assigned to !=1 split: leakage detected.")

    X_train, y_train, g_train = X[train_mask], y[train_mask], groups[train_mask]
    X_val, y_val, g_val = X[val_mask], y[val_mask], groups[val_mask]
    X_test, y_test, g_test = X[test_mask], y[test_mask], groups[test_mask]

    print("\n[✓] Subject-level split:")
    for name, mask in [("train", train_mask), ("val", val_mask), ("test", test_mask)]:
        print(f"    {name:5s}: windows={int(mask.sum()):6d} | "
              f"class1_ratio={y[mask].mean():.3f}")

    # Channel ranking (train only -> no leakage)
    ranking_df = rank_channels(X_train, y_train, config.CHANNELS_19)
    ranking_df.to_csv(os.path.join(out, "channel_ranking.csv"), index=False)
    print("\n[✓] Channel ranking (top by PSD t-test):")
    print(ranking_df.head(6).to_string(index=False))

    # ---- persist everything ----
    saves = {
        "X.npy": X, "y.npy": y, "groups.npy": groups,
        "X_train.npy": X_train, "y_train.npy": y_train, "groups_train.npy": g_train,
        "X_val.npy": X_val, "y_val.npy": y_val, "groups_val.npy": g_val,
        "X_test.npy": X_test, "y_test.npy": y_test, "groups_test.npy": g_test,
    }
    for fname, arr in saves.items():
        np.save(os.path.join(out, fname), arr)

    with open(os.path.join(out, "split_info.json"), "w") as f:
        json.dump(
            {
                "split_ids": split_ids,
                "n_windows": {
                    "train": int(train_mask.sum()),
                    "val": int(val_mask.sum()),
                    "test": int(test_mask.sum()),
                },
                "class1_ratio": {
                    "train": float(y_train.mean()) if len(y_train) else None,
                    "val": float(y_val.mean()) if len(y_val) else None,
                    "test": float(y_test.mean()) if len(y_test) else None,
                },
            },
            f,
            indent=2,
        )

    with open(os.path.join(out, "preprocessing_meta.json"), "w") as f:
        json.dump(
            {
                "sfreq": config.SFREQ,
                "window_sec": config.WINDOW_SEC,
                "samples_per_window": config.SAMPLES_PER_WINDOW,
                "band_pass": [config.FREQ_LOW, config.FREQ_HIGH],
                "notch": config.FREQ_NOTCH,
                "canonical_channels": config.CHANNELS_19,
                "dev_mode": config.DEV_MODE,
                "discard_reasons": dict(discard_reasons),
                "X_shape": list(X.shape),
            },
            f,
            indent=2,
        )

    print(f"\n[✓] All artifacts written to: {out}")
    print("=" * 64)


if __name__ == "__main__":
    main()
