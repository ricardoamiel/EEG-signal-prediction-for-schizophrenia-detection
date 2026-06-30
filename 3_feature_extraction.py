"""
3_feature_extraction.py
=======================
Phase 3 (classical branch): extract a massive feature table from the windowed
EEG signals, engineered to survive limited RAM.

Two feature families are produced:

  1. tsfresh features  --  extracted CHANNEL BY CHANNEL. Running tsfresh with
     EfficientFCParameters over all channels at once OOM-crashes, so each channel
     is processed independently, imputed, written to its own `.parquet`, and the
     in-memory frames are released with `gc.collect()` before moving on. The
     per-channel parquet files double as a resumable cache: re-running skips
     channels already on disk. They are concatenated only at the very end.

  2. Shapelets (tslearn.LearningShapelets) -- learned on a SUBSAMPLE of the
     training windows with a strict iteration/time budget so it stays viable,
     then used to transform every split into shapelet-distance features.

Run (requires `python 1_preprocessing.py` first):
    python 3_feature_extraction.py

Outputs (under artifacts/03_features/):
    tsfresh_cache/feats_<split>_<channel>.parquet   per-channel cache
    feats_train_raw.parquet / feats_val_raw.parquet / feats_test_raw.parquet
    shapelet_feats_<split>.parquet                  (if SHAPELETS_ENABLE)
    feature_extraction_meta.json
"""

from __future__ import annotations

import gc
import json
import os
import time
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import config


# =====================================================================
# 0. IO HELPERS
# =====================================================================
def _load_split(split: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load a preprocessed split tensor and labels, sliced to the active montage.

    Tensors are stored in canonical 19-channel order; we slice the columns of the
    montage selected by ``config.CHANNEL_SET`` (e.g. Top-6) for feature work.
    """
    src = config.DIR_PREPROCESS
    x_path = os.path.join(src, f"X_{split}.npy")
    y_path = os.path.join(src, f"y_{split}.npy")
    if not (os.path.exists(x_path) and os.path.exists(y_path)):
        raise FileNotFoundError(
            f"Missing {x_path} or {y_path}. Run `python 1_preprocessing.py` first."
        )
    X = np.load(x_path)
    y = np.load(y_path)
    idx = config.channel_indices(config.active_channels())
    return X[:, idx, :], y


def _resolve_fc_parameters():
    """Map config.TSFRESH_FC_PARAMETERS to a concrete tsfresh parameter object."""
    from tsfresh.feature_extraction import (
        ComprehensiveFCParameters,
        EfficientFCParameters,
        MinimalFCParameters,
    )

    table = {
        "minimal": MinimalFCParameters,
        "efficient": EfficientFCParameters,
        "comprehensive": ComprehensiveFCParameters,
    }
    key = config.TSFRESH_FC_PARAMETERS
    if key not in table:
        raise ValueError(f"TSFRESH_FC_PARAMETERS={key!r}; expected {list(table)}")
    return table[key]()


# =====================================================================
# 1. ITERATIVE (CHANNEL-BY-CHANNEL) TSFRESH EXTRACTION
# =====================================================================
def extract_tsfresh_iteratively(
    X: np.ndarray,
    channel_names: List[str],
    fc_parameters,
    n_jobs: int,
    split: str,
) -> pd.DataFrame:
    """Extract tsfresh features one channel at a time to bound peak RAM.

    Each channel's features are computed, imputed, and cached to parquet; memory
    is freed with ``gc.collect()`` after every channel. Channels already cached
    on disk are reloaded instead of recomputed (resumable). The per-channel
    frames are concatenated column-wise only at the end.
    """
    from tsfresh import extract_features
    from tsfresh.utilities.dataframe_functions import impute

    n_windows, n_channels, n_samples = X.shape
    os.makedirs(config.DIR_FEATURES_CACHE, exist_ok=True)

    per_channel_frames: List[pd.DataFrame] = []
    pbar = tqdm(channel_names, total=n_channels,
                desc=f"tsfresh ({split})", unit="ch", leave=True)

    for ch_idx, ch_name in enumerate(pbar):
        pbar.set_description_str(f"tsfresh ({split}) - {ch_name}")
        cache_path = os.path.join(
            config.DIR_FEATURES_CACHE, f"feats_{split}_{ch_name}.parquet"
        )

        if os.path.exists(cache_path):
            df_ch = pd.read_parquet(cache_path)
            per_channel_frames.append(df_ch)
            pbar.set_postfix_str(f"cache ({df_ch.shape[1]} feats)")
            continue

        t0 = time.perf_counter()
        try:
            # tsfresh long format: one row per (window, timestep) for this channel.
            df_long = pd.DataFrame({
                "id": np.repeat(np.arange(n_windows), n_samples),
                "time": np.tile(np.arange(n_samples), n_windows),
                ch_name: X[:, ch_idx, :].reshape(-1),
            })
            df_ch = extract_features(
                df_long,
                column_id="id",
                column_sort="time",
                default_fc_parameters=fc_parameters,
                n_jobs=n_jobs,
                disable_progressbar=True,
            )
            impute(df_ch)  # in-place: replaces NaN/inf with finite column stats
            df_ch.to_parquet(cache_path)
        except Exception as e:  # noqa: BLE001
            pbar.write(f"[ERROR] channel {ch_name} failed: {e}")
            raise
        finally:
            # Release the heavy long-format frame regardless of success.
            df_long = None  # noqa: F841
            gc.collect()

        per_channel_frames.append(df_ch)
        pbar.set_postfix_str(f"{df_ch.shape[1]} feats ({time.perf_counter()-t0:.1f}s)")

    pbar.close()

    print(f"[{split}] concatenating {n_channels} channels...")
    feats = pd.concat(per_channel_frames, axis=1)
    del per_channel_frames
    gc.collect()
    print(f"[{split}] -> {feats.shape[0]} windows x {feats.shape[1]} features")
    return feats


# =====================================================================
# 2. SHAPELETS (tslearn.LearningShapelets), subsampled + budgeted
# =====================================================================
def _subsample_indices(n: int, max_n: int, seed: int) -> np.ndarray:
    """Return up to ``max_n`` distinct row indices in [0, n) (sorted)."""
    if n <= max_n:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=max_n, replace=False))


def extract_shapelet_features(
    X_train: np.ndarray, y_train: np.ndarray,
    splits: dict[str, np.ndarray],
) -> Tuple[Optional[dict[str, pd.DataFrame]], Optional[dict]]:
    """Learn shapelets on a subsample of train, transform all splits.

    tslearn expects multivariate series as ``(n_ts, sz, n_dims)``; our windows are
    ``(channels, samples)`` so we transpose to ``(samples, channels)``. Learning
    is bounded by ``SHAPELETS_MAX_TRAIN_WINDOWS`` and ``SHAPELETS_MAX_ITER`` to
    keep it computationally viable. Returns (feature frames per split, meta) or
    ``(None, None)`` if disabled/unavailable.
    """
    if not config.SHAPELETS_ENABLE:
        print("[shapelets] disabled via config.SHAPELETS_ENABLE")
        return None, None

    try:
        from tslearn.shapelets import LearningShapelets
        from tslearn.shapelets import grabocka_params_to_shapelet_size_dict
    except Exception as e:  # noqa: BLE001
        print(f"[shapelets] tslearn unavailable ({e}); skipping.")
        return None, None

    def to_tslearn(x: np.ndarray) -> np.ndarray:
        # (n, channels, samples) -> (n, samples, channels)
        return np.transpose(x, (0, 2, 1)).astype(np.float64)

    sub_idx = _subsample_indices(
        len(X_train), config.SHAPELETS_MAX_TRAIN_WINDOWS, config.RANDOM_STATE
    )
    Xt_train = to_tslearn(X_train[sub_idx])
    yt_train = y_train[sub_idx]
    n_ts, sz, n_dims = Xt_train.shape
    print(f"[shapelets] training on {n_ts} subsampled windows "
          f"(sz={sz}, dims={n_dims}), max_iter={config.SHAPELETS_MAX_ITER}")

    shp_sizes = grabocka_params_to_shapelet_size_dict(
        n_ts=n_ts, ts_sz=sz, n_classes=2,
        l=0.1, r=1,  # shapelet base length = 10% of series, single length scale
    )
    # Cap shapelets-per-size to keep the model small.
    shp_sizes = {k: min(v, config.SHAPELETS_N_PER_SIZE) for k, v in shp_sizes.items()}

    model = LearningShapelets(
        n_shapelets_per_size=shp_sizes,
        max_iter=config.SHAPELETS_MAX_ITER,
        random_state=config.RANDOM_STATE,
        verbose=0,
    )

    t0 = time.perf_counter()
    try:
        model.fit(Xt_train, yt_train)
    except Exception as e:  # noqa: BLE001
        print(f"[shapelets] fit failed ({e}); skipping shapelet features.")
        return None, None
    elapsed = time.perf_counter() - t0
    if elapsed > config.SHAPELETS_TIME_LIMIT_SEC:
        print(f"[shapelets] WARNING: fit took {elapsed:.0f}s "
              f"(> budget {config.SHAPELETS_TIME_LIMIT_SEC}s). "
              f"Lower SHAPELETS_MAX_ITER / SHAPELETS_MAX_TRAIN_WINDOWS next run.")

    # Transform every split into shapelet-distance features.
    feat_frames: dict[str, pd.DataFrame] = {}
    n_shapelets = None
    for split, X_split in splits.items():
        dist = model.transform(to_tslearn(X_split))  # (n, n_shapelets)
        n_shapelets = dist.shape[1]
        cols = [f"shapelet_dist_{i}" for i in range(n_shapelets)]
        feat_frames[split] = pd.DataFrame(dist, columns=cols)

    meta = {
        "n_shapelets": int(n_shapelets) if n_shapelets else 0,
        "train_windows_used": int(n_ts),
        "fit_seconds": round(elapsed, 1),
        "shapelet_sizes": {int(k): int(v) for k, v in shp_sizes.items()},
    }
    print(f"[shapelets] produced {meta['n_shapelets']} features in {elapsed:.0f}s")
    return feat_frames, meta


# =====================================================================
# 3. MAIN
# =====================================================================
def main() -> None:
    config.ensure_dirs()
    out = config.DIR_FEATURES
    channels = config.active_channels()

    print("=" * 64)
    print(f" PHASE 3 - FEATURE EXTRACTION  (montage='{config.CHANNEL_SET}': "
          f"{channels})")
    print("=" * 64)

    fc_parameters = _resolve_fc_parameters()
    print(f"[INFO] tsfresh set={config.TSFRESH_FC_PARAMETERS}, "
          f"n_jobs={config.TSFRESH_N_JOBS}")

    # ---- tsfresh, per split, channel-by-channel ----
    X_by_split: dict[str, np.ndarray] = {}
    y_by_split: dict[str, np.ndarray] = {}
    for split in ("train", "val", "test"):
        X_split, y_split = _load_split(split)
        X_by_split[split] = X_split
        y_by_split[split] = y_split
        feats = extract_tsfresh_iteratively(
            X_split, channels, fc_parameters, config.TSFRESH_N_JOBS, split
        )
        # Fused table is channel-set-specific -> prefix it. (The per-channel
        # cache inside extract_tsfresh_iteratively stays unprefixed/shared.)
        feats.to_parquet(os.path.join(out, config.pfx(f"feats_{split}_raw.parquet")))
        del feats
        gc.collect()

    # ---- shapelets ----
    shp_frames, shp_meta = extract_shapelet_features(
        X_by_split["train"], y_by_split["train"], X_by_split
    )
    if shp_frames is not None:
        for split, frame in shp_frames.items():
            frame.to_parquet(os.path.join(out, config.pfx(f"shapelet_feats_{split}.parquet")))

    # ---- metadata ----
    with open(os.path.join(out, config.pfx("feature_extraction_meta.json")), "w") as f:
        json.dump(
            {
                "channel_set": config.CHANNEL_SET,
                "channels": channels,
                "tsfresh_parameters": config.TSFRESH_FC_PARAMETERS,
                "n_windows": {k: int(len(v)) for k, v in y_by_split.items()},
                "shapelets": shp_meta,
            },
            f,
            indent=2,
        )

    print(f"\n[✓] Feature artifacts written to: {out}")
    print("=" * 64)


if __name__ == "__main__":
    main()
