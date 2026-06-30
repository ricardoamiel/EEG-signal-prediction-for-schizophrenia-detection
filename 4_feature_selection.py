"""
4_feature_selection.py
======================
Phase 4: shrink the massive tsfresh (+ shapelet) feature table down to a small,
non-redundant, explainable set, then optionally compress with PCA.

Rigorous filtering funnel (each stage fit on TRAIN only, applied to val/test):
  1. tsfresh statistical relevance filter (select_features, p-value FDR).
  2. Variance filter        -- drop (near-)constant features.
  3. High-correlation filter -- drop one of any pair with |r| > CORRELATION_THRESHOLD.
  4. Model-based selection   -- LightGBM + RandomForest, aggregate SHAP |values|.
  5. Top-K retention         -- keep the TOP_K_FEATURES by mean |SHAP|.
  6. Optional PCA            -- if still above PCA_MAX_FEATURES, compress to
                               PCA_VARIANCE_KEEP of the variance.

Run (requires Modules 1 and 3):
    python 4_feature_selection.py

Outputs (under artifacts/04_feature_selection/):
    X_train_selected.parquet / X_val_selected.parquet / X_test_selected.parquet
    selected_features.json         final feature names + funnel counts
    shap_importance.csv            SHAP ranking of the post-correlation set
    shap_importance.pdf            bar plot of the Top-K SHAP features
    pca_model.joblib               (only if PCA was applied)
"""

from __future__ import annotations

import json
import os
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import config


# =====================================================================
# 0. LOAD RAW FEATURES + LABELS
# =====================================================================
def _load_features() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame,
                              np.ndarray, np.ndarray, np.ndarray]:
    """Load tsfresh feature tables, append shapelet features if present, +labels."""
    fsrc, psrc = config.DIR_FEATURES, config.DIR_PREPROCESS
    frames = {}
    for split in ("train", "val", "test"):
        fp = os.path.join(fsrc, config.pfx(f"feats_{split}_raw.parquet"))
        if not os.path.exists(fp):
            raise FileNotFoundError(
                f"Missing {fp}. Run `python 3_feature_extraction.py` first "
                f"(CHANNEL_SET={config.CHANNEL_SET})."
            )
        df = pd.read_parquet(fp).reset_index(drop=True)
        shp = os.path.join(fsrc, config.pfx(f"shapelet_feats_{split}.parquet"))
        if os.path.exists(shp):
            df = pd.concat([df, pd.read_parquet(shp).reset_index(drop=True)], axis=1)
        frames[split] = df

    ys = {s: np.load(os.path.join(psrc, f"y_{s}.npy")) for s in ("train", "val", "test")}
    for s in ("train", "val", "test"):
        if len(frames[s]) != len(ys[s]):
            raise ValueError(
                f"Row mismatch in {s}: feats={len(frames[s])} vs y={len(ys[s])}"
            )
    return (frames["train"], frames["val"], frames["test"],
            ys["train"], ys["val"], ys["test"])


# =====================================================================
# 1. TSFRESH STATISTICAL RELEVANCE FILTER
# =====================================================================
def tsfresh_relevance_filter(
    X_train: pd.DataFrame, y_train: np.ndarray
) -> List[str]:
    """Keep only features tsfresh deems statistically relevant for y (train)."""
    try:
        from tsfresh import select_features
        from tsfresh.utilities.dataframe_functions import impute
        impute(X_train)
        selected = select_features(X_train, pd.Series(y_train))
        cols = list(selected.columns)
        if cols:
            print(f"[1] tsfresh relevance: {X_train.shape[1]} -> {len(cols)}")
            return cols
        print("[1] tsfresh relevance kept 0 cols (tiny DEV set?); skipping stage.")
    except Exception as e:  # noqa: BLE001
        print(f"[1] tsfresh select_features failed ({e}); skipping stage.")
    return list(X_train.columns)


# =====================================================================
# 2. VARIANCE FILTER
# =====================================================================
def variance_filter(X_train: pd.DataFrame, threshold: float) -> List[str]:
    variances = X_train.var(axis=0)
    keep = variances[variances > threshold].index.tolist()
    print(f"[2] variance > {threshold}: {X_train.shape[1]} -> {len(keep)}")
    return keep


# =====================================================================
# 3. HIGH-CORRELATION FILTER
# =====================================================================
def correlation_filter(X_train: pd.DataFrame, threshold: float) -> List[str]:
    corr = X_train.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [c for c in upper.columns if any(upper[c] > threshold)]
    keep = [c for c in X_train.columns if c not in to_drop]
    print(f"[3] correlation < {threshold}: {X_train.shape[1]} -> {len(keep)} "
          f"(dropped {len(to_drop)})")
    return keep


# =====================================================================
# 4 & 5. MODEL-BASED SHAP SELECTION + TOP-K
# =====================================================================
def _mean_abs_shap(model, X: pd.DataFrame) -> np.ndarray:
    """Mean |SHAP| per feature for the positive class, robust to shap's many
    return shapes (list-per-class, 2D, or 3D (n, features, classes))."""
    import shap

    sv = shap.TreeExplainer(model).shap_values(X)
    if isinstance(sv, list):                 # [neg, pos]
        sv = sv[1]
    sv = np.asarray(sv)
    if sv.ndim == 3:                          # (n, features, classes)
        sv = sv[:, :, 1] if sv.shape[2] >= 2 else sv[:, :, 0]
    return np.abs(sv).mean(axis=0)


def shap_selection(
    X_train: pd.DataFrame, y_train: np.ndarray
) -> Tuple[List[str], pd.DataFrame]:
    """Aggregate SHAP |values| from LightGBM + RandomForest, then cut by the
    configured mode (``threshold`` on normalised aggregate, or ``topk``).

    Returns (selected feature names, full SHAP importance DataFrame).
    """
    from sklearn.ensemble import RandomForestClassifier

    importances = pd.DataFrame(index=X_train.columns)

    try:
        from lightgbm import LGBMClassifier
        lgbm = LGBMClassifier(n_estimators=300, class_weight="balanced",
                              random_state=config.RANDOM_STATE, verbose=-1)
        lgbm.fit(X_train, y_train)
        importances["lgbm"] = _mean_abs_shap(lgbm, X_train)
        print("[4] LightGBM SHAP: OK")
    except Exception as e:  # noqa: BLE001
        print(f"[4] LightGBM SHAP failed ({e}); skipping that model.")

    try:
        rf = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                    random_state=config.RANDOM_STATE, n_jobs=-1)
        rf.fit(X_train, y_train)
        importances["rf"] = _mean_abs_shap(rf, X_train)
        print("[4] RandomForest SHAP: OK")
    except Exception as e:  # noqa: BLE001
        print(f"[4] RandomForest SHAP failed ({e}); skipping that model.")

    if importances.shape[1] == 0:
        print("[4] No SHAP importances; falling back to variance ranking.")
        importances["fallback"] = X_train.var(axis=0)

    # Normalise each model's importance to [0,1] then average -> fair aggregation.
    norm = importances / (importances.max(axis=0) + 1e-12)
    importances["SHAP_aggregate"] = norm.mean(axis=1)
    importances = importances.sort_values("SHAP_aggregate", ascending=False)

    mode = config.FEATURE_SELECTION_MODE
    if mode == "threshold":
        mask = importances["SHAP_aggregate"] >= config.SHAP_THRESHOLD
        selected = importances.index[mask].tolist()
        if len(selected) < config.MIN_FEATURES:
            selected = importances.head(config.MIN_FEATURES).index.tolist()
            print(f"[5] threshold {config.SHAP_THRESHOLD} kept <{config.MIN_FEATURES}; "
                  f"floored to Top-{config.MIN_FEATURES}.")
        else:
            print(f"[5] SHAP threshold >= {config.SHAP_THRESHOLD}: "
                  f"{X_train.shape[1]} -> {len(selected)}")
    elif mode == "topk":
        k = min(config.TOP_K_FEATURES, len(importances))
        selected = importances.head(k).index.tolist()
        print(f"[5] Top-K by aggregated SHAP: {X_train.shape[1]} -> {len(selected)}")
    else:
        raise ValueError(f"FEATURE_SELECTION_MODE={mode!r}; expected 'threshold'|'topk'")

    return selected, importances.reset_index().rename(columns={"index": "feature"})


def plot_shap_importance(importance_df: pd.DataFrame, top_k: int) -> None:
    df = importance_df.head(min(top_k, 30))[::-1]
    fig, ax = plt.subplots(figsize=(9, max(4, 0.3 * len(df))))
    ax.barh(df["feature"], df["SHAP_aggregate"], color="tab:purple")
    ax.set(title="Top features by aggregated SHAP importance",
           xlabel="Mean |SHAP| (normalised, LightGBM+RF)")
    fig.tight_layout()
    path = os.path.join(config.DIR_FEATSELECT, config.pfx("shap_importance.pdf"))
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"   [✓] {path}")


# =====================================================================
# 6. OPTIONAL PCA
# =====================================================================
def maybe_pca(
    X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Optional[object]]:
    """Apply PCA (fit on train) if the feature count still exceeds the cap."""
    if X_train.shape[1] <= config.PCA_MAX_FEATURES:
        print(f"[6] {X_train.shape[1]} <= {config.PCA_MAX_FEATURES} features; "
              f"PCA not needed.")
        return X_train, X_val, X_test, None

    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(X_train)
    pca = PCA(n_components=config.PCA_VARIANCE_KEEP, random_state=config.RANDOM_STATE)
    Z_train = pca.fit_transform(scaler.transform(X_train))
    cols = [f"PC{i+1}" for i in range(Z_train.shape[1])]
    print(f"[6] PCA: {X_train.shape[1]} -> {len(cols)} comps "
          f"({config.PCA_VARIANCE_KEEP:.0%} variance)")

    def tf(df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(pca.transform(scaler.transform(df)), columns=cols,
                            index=df.index)

    import joblib
    joblib.dump({"scaler": scaler, "pca": pca},
                os.path.join(config.DIR_FEATSELECT, config.pfx("pca_model.joblib")))
    return tf(X_train), tf(X_val), tf(X_test), pca


# =====================================================================
# 7. MAIN
# =====================================================================
def main() -> None:
    config.ensure_dirs()
    out = config.DIR_FEATSELECT
    print("=" * 64)
    print(" PHASE 4 - FEATURE SELECTION FUNNEL")
    print("=" * 64)

    Xtr, Xva, Xte, ytr, _yva, _yte = _load_features()
    print(f"[INFO] raw features: train={Xtr.shape}")
    funnel = {"raw": Xtr.shape[1]}

    # Stage 1: tsfresh relevance
    cols = tsfresh_relevance_filter(Xtr.copy(), ytr)
    Xtr, Xva, Xte = Xtr[cols], Xva[cols], Xte[cols]
    funnel["tsfresh_relevance"] = Xtr.shape[1]

    # Stage 2: variance
    cols = variance_filter(Xtr, config.VARIANCE_THRESHOLD)
    Xtr, Xva, Xte = Xtr[cols], Xva[cols], Xte[cols]
    funnel["variance"] = Xtr.shape[1]

    # Stage 3: correlation
    cols = correlation_filter(Xtr, config.CORRELATION_THRESHOLD)
    Xtr, Xva, Xte = Xtr[cols], Xva[cols], Xte[cols]
    funnel["correlation"] = Xtr.shape[1]

    # Stages 4-5: SHAP-based model selection (threshold or top-k)
    selected, importance_df = shap_selection(Xtr, ytr)
    importance_df.to_csv(os.path.join(out, config.pfx("shap_importance.csv")), index=False)
    plot_shap_importance(importance_df, len(selected))
    Xtr, Xva, Xte = Xtr[selected], Xva[selected], Xte[selected]
    funnel["shap_selection"] = Xtr.shape[1]
    final_feature_names = list(Xtr.columns)

    # Stage 6: optional PCA
    Xtr, Xva, Xte, pca = maybe_pca(Xtr, Xva, Xte)
    funnel["after_pca"] = Xtr.shape[1]

    # Persist (channel-set-prefixed for ablation comparisons)
    Xtr.to_parquet(os.path.join(out, config.pfx("X_train_selected.parquet")))
    Xva.to_parquet(os.path.join(out, config.pfx("X_val_selected.parquet")))
    Xte.to_parquet(os.path.join(out, config.pfx("X_test_selected.parquet")))
    with open(os.path.join(out, config.pfx("selected_features.json")), "w") as f:
        json.dump({
            "funnel_counts": funnel,
            "selection_mode": config.FEATURE_SELECTION_MODE,
            "shap_threshold": config.SHAP_THRESHOLD,
            "top_k": config.TOP_K_FEATURES,
            "pca_applied": pca is not None,
            "feature_names_before_pca": final_feature_names,
            "final_columns": list(Xtr.columns),
        }, f, indent=2)

    print("\n[✓] Funnel:", " -> ".join(f"{k}:{v}" for k, v in funnel.items()))
    print(f"[✓] Selected-feature artifacts written to: {out}")
    print("=" * 64)


if __name__ == "__main__":
    main()
