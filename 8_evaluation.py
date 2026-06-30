"""
8_evaluation.py
==============
Phase 5: final comparison on the reserved TEST set.

Modules 6 (classical ML) and 7 (deep learning) each write their final TEST
predictions to `artifacts/08_evaluation/predictions/preds_<model>.npz`. This
script gathers every such file, recomputes the full project metric set, ranks
the models, and produces the two deliverables required by `proyecto.md`:

  * comparacion_final_modelos_SOTA.csv  -- Accuracy, Precision, Recall/Sensitivity,
    Specificity, F1-Score, ROC-AUC, Balanced Accuracy (one row per model).
  * matrices_confusion_finales_SOTA.pdf -- all confusion matrices in one figure.

Run (after Modules 6 and/or 7):
    python 8_evaluation.py

Outputs (under artifacts/08_evaluation/ and project root for convenience):
    comparacion_final_modelos_SOTA.csv
    matrices_confusion_finales_SOTA.pdf
"""

from __future__ import annotations

import glob
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import config

CSV_NAME = "comparacion_final_modelos_SOTA.csv"
PDF_NAME = "matrices_confusion_finales_SOTA.pdf"


# =====================================================================
# 1. GATHER PREDICTIONS
# =====================================================================
def load_all_predictions() -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Load every preds_<model>.npz into {name: (y_true, y_pred, y_proba, groups)}."""
    files = sorted(glob.glob(os.path.join(config.DIR_PREDICTIONS, config.pfx("preds_*.npz"))))
    if not files:
        raise FileNotFoundError(
            f"No predictions for CHANNEL_SET={config.CHANNEL_SET} in "
            f"{config.DIR_PREDICTIONS}. Run `python 6_train_ml.py` and/or "
            f"`python 7_train_dl_sota.py` first."
        )
    results = {}
    for fp in files:
        data = np.load(fp, allow_pickle=True)
        name = str(data["name"])
        proba = data["y_proba"]
        groups = data["groups"] if "groups" in data.files else np.array([])
        results[name] = (
            data["y_true"], data["y_pred"],
            proba if proba.size else None,
            groups if groups.size else None,
        )
    return results


def aggregate_to_subjects(
    y_true: np.ndarray, y_proba: np.ndarray, groups: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pool window predictions into one prediction per subject.

    A subject's score is the mean predicted probability over its windows; the
    label is the subject's (constant) ground truth. Returns (y_true_subj,
    y_pred_subj, y_proba_subj).
    """
    import pandas as pd
    df = pd.DataFrame({"subject": groups, "y_true": y_true, "y_proba": y_proba})
    agg = df.groupby("subject").agg(y_true=("y_true", "first"),
                                    y_proba=("y_proba", "mean")).reset_index()
    y_pred = (agg["y_proba"].to_numpy() >= 0.5).astype(int)
    return agg["y_true"].to_numpy(), y_pred, agg["y_proba"].to_numpy()


# =====================================================================
# 2. METRIC TABLE
# =====================================================================
def _metrics_row(m: dict) -> dict:
    return {
        "Accuracy": m["accuracy"],
        "Precision": m["precision"],
        "Recall (Sensitivity)": m["recall_sensitivity"],
        "Specificity": m["specificity"],
        "F1-Score": m["f1"],
        "ROC-AUC": m["roc_auc"],
        "Balanced Accuracy": m["balanced_accuracy"],
    }


def build_window_table(results: Dict) -> Tuple[pd.DataFrame, Dict[str, dict]]:
    """Window-level metrics (one prediction per 5 s window)."""
    rows, metrics_by_model = {}, {}
    for name, (y_true, y_pred, y_proba, _g) in results.items():
        m = config.compute_metrics(y_true, y_pred, y_proba)
        metrics_by_model[name] = m
        rows[name] = _metrics_row(m)
        config.print_metrics(name, m)
    df = pd.DataFrame(rows).T.sort_values("F1-Score", ascending=False)
    return df, metrics_by_model


def build_subject_table(results: Dict) -> Tuple[pd.DataFrame, Dict[str, dict]]:
    """Subject-level metrics (window probabilities pooled per subject).

    Only models whose predictions carry subject ids are included.
    """
    rows, metrics_by_model = {}, {}
    for name, (y_true, _y_pred, y_proba, groups) in results.items():
        if groups is None or y_proba is None:
            print(f"[subject] {name}: no subject ids/proba; skipped.")
            continue
        yt, yp, pr = aggregate_to_subjects(y_true, y_proba, groups)
        m = config.compute_metrics(yt, yp, pr)
        metrics_by_model[name] = m
        rows[name] = _metrics_row(m)
        config.print_metrics(f"{name} [subj n={len(yt)}]", m)
    if not rows:
        return pd.DataFrame(), {}
    df = pd.DataFrame(rows).T.sort_values("F1-Score", ascending=False)
    return df, metrics_by_model


# =====================================================================
# 3. CONFUSION-MATRIX GRID
# =====================================================================
def plot_confusions(metrics_by_model: Dict[str, dict], order: List[str],
                    path: str) -> None:
    n = len(order)
    ncols = min(4, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows),
                             squeeze=False)
    for ax in axes.flatten():
        ax.axis("off")
    for ax, name in zip(axes.flatten(), order):
        ax.axis("on")
        cm = np.array(metrics_by_model[name]["confusion_matrix"])
        ax.imshow(cm, cmap="Blues")
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_xticks([0, 1]); ax.set_xticklabels(config.CLASS_NAMES, fontsize=8)
        ax.set_yticks([0, 1]); ax.set_yticklabels(config.CLASS_NAMES, fontsize=8)
        thresh = cm.max() / 2 if cm.max() else 0
        for i in range(2):
            for j in range(2):
                ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
    fig.suptitle("Confusion matrices - TEST set", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"   [✓] {path}")


# =====================================================================
# 3b. GROUND-TRUTH vs PREDICTED (aggregate)
# =====================================================================
def plot_pred_vs_truth(results: Dict, level: str, path: str) -> None:
    """Per-model panel of ground truth vs predicted probability.

    Samples are sorted by predicted probability; marker colour encodes the TRUE
    class and the dashed line is the 0.5 decision threshold. A well-separated
    model shows controls (blue) bunched low and patients (red) bunched high.
    ``level`` is 'window' (raw predictions) or 'subject' (probabilities pooled
    per subject). Models lacking the needed data for the level are skipped.
    """
    panels = []
    for name, (y_true, _y_pred, y_proba, groups) in results.items():
        if y_proba is None:
            continue
        if level == "subject":
            if groups is None:
                continue
            yt, _yp, pr = aggregate_to_subjects(y_true, y_proba, groups)
        else:
            yt, pr = np.asarray(y_true), np.asarray(y_proba)
        panels.append((name, yt, pr))
    if not panels:
        print(f"[pred_vs_truth] no models eligible at {level} level; skipped.")
        return

    n = len(panels)
    ncols = min(4, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.6 * nrows),
                             squeeze=False)
    for ax in axes.flatten():
        ax.axis("off")
    for ax, (name, yt, pr) in zip(axes.flatten(), panels):
        ax.axis("on")
        order = np.argsort(pr)
        pr_s, yt_s = pr[order], yt[order]
        x = np.arange(len(pr_s))
        for cls, color in [(0, "tab:blue"), (1, "tab:red")]:
            m = yt_s == cls
            ax.scatter(x[m], pr_s[m], s=14, alpha=0.6, color=color,
                       label=config.CLASS_NAMES[cls])
        ax.axhline(0.5, ls="--", color="gray", lw=1)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel(f"{level} (sorted by prob)")
        ax.set_ylabel("P(patient)")
        ax.set_ylim(-0.02, 1.02)
        ax.legend(fontsize=7, loc="upper left")
    fig.suptitle(f"Ground truth vs predicted - {level}-level (TEST)", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"   [✓] {path}")


# =====================================================================
# 4. MAIN
# =====================================================================
def _emit(df: pd.DataFrame, metrics_by_model: Dict[str, dict],
          csv_name: str, pdf_name: str, level: str) -> None:
    """Write a metric table (CSV in eval dir + project root) and a confusion PDF."""
    print("\n" + "=" * 64)
    print(f" {level.upper()}-LEVEL RESULTS")
    print("=" * 64)
    print(df.round(3).to_string())
    for p in (os.path.join(config.DIR_EVAL, config.pfx(csv_name)),
              os.path.join(config.PROJECT_ROOT, config.pfx(csv_name))):
        df.to_csv(p)
        print(f"[✓] {level} CSV -> {p}")
    for p in (os.path.join(config.DIR_EVAL, config.pfx(pdf_name)),
              os.path.join(config.PROJECT_ROOT, config.pfx(pdf_name))):
        plot_confusions(metrics_by_model, list(df.index), p)
    best = df.index[0]
    print(f"[✓] Best ({level}) by F1-Score: {best} (F1={df.loc[best, 'F1-Score']:.3f})")


def main() -> None:
    config.ensure_dirs()
    print("=" * 64)
    print(" PHASE 8 - FINAL EVALUATION (TEST set)")
    print("=" * 64)

    results = load_all_predictions()
    print(f"[INFO] models found: {list(results)}\n")

    # ---- window-level (one prediction per 5 s window) ----
    win_df, win_metrics = build_window_table(results)
    _emit(win_df, win_metrics, CSV_NAME, PDF_NAME, "window")
    plot_pred_vs_truth(
        results, "window",
        os.path.join(config.DIR_EVAL, config.pfx("pred_vs_truth_window.pdf")))

    # ---- subject-level (window probabilities pooled per subject) ----
    print("\n[*] Aggregating to subject level...")
    subj_df, subj_metrics = build_subject_table(results)
    if not subj_df.empty:
        _emit(subj_df, subj_metrics,
              "comparacion_final_modelos_SOTA_subject.csv",
              "matrices_confusion_finales_SOTA_subject.pdf", "subject")
        # Aggregate ground-truth-vs-predicted figure (also copied to repo root
        # for direct inclusion in the manuscript).
        for p in (os.path.join(config.DIR_EVAL, config.pfx("pred_vs_truth_subject.pdf")),
                  os.path.join(config.PROJECT_ROOT, config.pfx("pred_vs_truth_subject.pdf"))):
            plot_pred_vs_truth(results, "subject", p)
    else:
        print("[subject] no models carried subject ids; re-run Modules 6/7 to "
              "enable subject-level metrics.")

    print("=" * 64)


if __name__ == "__main__":
    main()
