"""
6_train_ml.py
============
Phase 4a: train the classical machine-learning models on the selected feature
set from Module 4 (Logistic Regression, SVM, Random Forest, XGBoost, LightGBM).

Models are selected/monitored on VALIDATION. The reserved TEST set is touched
exactly once at the end of this script to dump per-model predictions
(`preds_<model>.npz`) for Module 8 to assemble into the final comparison.

The best tree model (XGBoost, falling back to LightGBM) gets a SHAP summary
plot to explain the clinical decision.

Run (requires Modules 1 and 4):
    python 6_train_ml.py

Outputs:
    artifacts/06_ml_models/<model>.joblib          fitted estimators
    artifacts/06_ml_models/scaler.joblib           feature scaler (train-fit)
    artifacts/06_ml_models/val_metrics.json        validation metrics
    artifacts/06_ml_models/shap_summary.pdf        SHAP summary of best tree model
    artifacts/08_evaluation/predictions/preds_<model>.npz   TEST predictions
"""

from __future__ import annotations

import json
import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import config


# =====================================================================
# 0. LOAD SELECTED FEATURES + LABELS
# =====================================================================
def _load() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame,
                     np.ndarray, np.ndarray, np.ndarray]:
    fsel, psrc = config.DIR_FEATSELECT, config.DIR_PREPROCESS
    dfs = {}
    for s in ("train", "val", "test"):
        fp = os.path.join(fsel, config.pfx(f"X_{s}_selected.parquet"))
        if not os.path.exists(fp):
            raise FileNotFoundError(
                f"Missing {fp}. Run `python 4_feature_selection.py` first "
                f"(CHANNEL_SET={config.CHANNEL_SET})."
            )
        dfs[s] = pd.read_parquet(fp)
    ys = {s: np.load(os.path.join(psrc, f"y_{s}.npy")) for s in ("train", "val", "test")}
    return dfs["train"], dfs["val"], dfs["test"], ys["train"], ys["val"], ys["test"]


# =====================================================================
# 1. MODEL ZOO
# =====================================================================
def build_models() -> Dict[str, object]:
    """Instantiate the classical models, skipping any whose lib is missing."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC

    rs = config.RANDOM_STATE
    models: Dict[str, object] = {
        "LogisticRegression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=rs),
        "SVM": SVC(kernel="rbf", probability=True, class_weight="balanced",
                   random_state=rs),
        "RandomForest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=rs, n_jobs=-1),
    }
    try:
        from xgboost import XGBClassifier
        models["XGBoost"] = XGBClassifier(
            n_estimators=300, eval_metric="logloss", random_state=rs, verbosity=0)
    except Exception as e:  # noqa: BLE001
        print(f"[INFO] XGBoost unavailable ({e}); skipping.")
    try:
        from lightgbm import LGBMClassifier
        models["LightGBM"] = LGBMClassifier(
            n_estimators=300, class_weight="balanced", random_state=rs, verbose=-1)
    except Exception as e:  # noqa: BLE001
        print(f"[INFO] LightGBM unavailable ({e}); skipping.")
    return models


# =====================================================================
# 2. SHAP SUMMARY FOR THE BEST TREE MODEL
# =====================================================================
def shap_summary(models: Dict[str, object], X_val: pd.DataFrame) -> None:
    name = next((m for m in ("XGBoost", "LightGBM", "RandomForest") if m in models),
                None)
    if name is None:
        print("[SHAP] no tree model available; skipping summary plot.")
        return
    try:
        import shap
        explainer = shap.TreeExplainer(models[name])
        sv = explainer.shap_values(X_val)
        if isinstance(sv, list):
            sv = sv[1]
        elif getattr(sv, "ndim", 2) == 3:
            sv = sv[:, :, 1]
        plt.figure(figsize=(10, 6))
        shap.summary_plot(sv, X_val, max_display=15, show=False)
        plt.title(f"SHAP summary - {name} (schizophrenia decision)")
        plt.tight_layout()
        path = os.path.join(config.DIR_ML, config.pfx("shap_summary.pdf"))
        plt.savefig(path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"[SHAP] {name} summary -> {path}")
    except Exception as e:  # noqa: BLE001
        print(f"[SHAP] summary failed ({e}); continuing.")


# =====================================================================
# 3. MAIN
# =====================================================================
def main() -> None:
    config.ensure_dirs()
    import joblib
    from sklearn.preprocessing import StandardScaler

    print("=" * 64)
    print(" PHASE 6 - CLASSICAL ML TRAINING")
    print("=" * 64)

    X_train, X_val, X_test, y_train, y_val, y_test = _load()
    # Subject ids for the TEST windows (row-aligned with X_test) so Module 8 can
    # report subject-level metrics. select_features preserves window order.
    groups_test = np.load(os.path.join(config.DIR_PREPROCESS, "groups_test.npy"))
    print(f"[INFO] features: train={X_train.shape} | "
          f"classes train: {np.bincount(y_train)}")

    # Scale (fit on train only). Trees ignore it; SVM/LogReg need it.
    scaler = StandardScaler().fit(X_train)
    Xtr = scaler.transform(X_train)
    Xva = scaler.transform(X_val)
    Xte = scaler.transform(X_test)
    joblib.dump(scaler, os.path.join(config.DIR_ML, config.pfx("scaler.joblib")))

    models = build_models()
    val_metrics: Dict[str, dict] = {}
    fitted: Dict[str, object] = {}

    for name, model in models.items():
        try:
            model.fit(Xtr, y_train)
        except Exception as e:  # noqa: BLE001
            print(f"[{name}] training failed ({e}); skipping.")
            continue
        y_pred = model.predict(Xva)
        y_proba = model.predict_proba(Xva)[:, 1] if hasattr(model, "predict_proba") else None
        m = config.compute_metrics(y_val, y_pred, y_proba)
        val_metrics[name] = m
        fitted[name] = model
        config.print_metrics(name, m)
        joblib.dump(model, os.path.join(config.DIR_ML, config.pfx(f"{name}.joblib")))

    # SHAP explainability uses the UNSCALED frame so feature names/values read
    # clinically (tree models are scale-invariant, so refit on raw frame).
    tree_for_shap = {}
    for name in ("XGBoost", "LightGBM", "RandomForest"):
        if name in fitted:
            clone = build_models()[name]
            clone.fit(X_train, y_train)  # raw features for interpretable SHAP
            tree_for_shap[name] = clone
            break
    shap_summary(tree_for_shap, X_val)

    # ---- FINAL: TEST predictions (reserved set, touched once) ----
    print("\n[*] Generating TEST predictions for Module 8...")
    for name, model in fitted.items():
        y_pred = model.predict(Xte)
        y_proba = model.predict_proba(Xte)[:, 1] if hasattr(model, "predict_proba") else None
        config.save_predictions(name, y_test, y_pred, y_proba, groups=groups_test)

    with open(os.path.join(config.DIR_ML, config.pfx("val_metrics.json")), "w") as f:
        json.dump(val_metrics, f, indent=2)

    print(f"\n[✓] {len(fitted)} models trained. Artifacts -> {config.DIR_ML}")
    print(f"[✓] TEST predictions -> {config.DIR_PREDICTIONS}")
    print("=" * 64)


if __name__ == "__main__":
    main()
