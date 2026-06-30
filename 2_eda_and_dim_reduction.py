"""
2_eda_and_dim_reduction.py
=========================
Phase 2: clinical exploratory data analysis and unsupervised structure probing.

Reproduces the notebook's EDA (PSD, temporal grand average, functional
connectivity, per-band relative-power boxplots) and adds two new analyses:

  * PCA -> UMAP 2D embedding of the windows, coloured by class, on the Top-6
    montage (does the raw signal already separate the two populations?);
  * Spectral Clustering on the same low-dim representation, to check whether
    natural clusters align with the diagnosis labels (ARI vs ground truth).

Every figure is saved to PDF (no interactive `plt.show`) for reproducibility.

Run (requires `python 1_preprocessing.py` first):
    python 2_eda_and_dim_reduction.py

Outputs (under artifacts/02_eda/):
    psd_<channel>.pdf, grand_average.pdf, connectivity.pdf,
    band_power_boxplots.pdf, pca_umap_top6.pdf, spectral_clustering_top6.pdf,
    eda_meta.json
"""

from __future__ import annotations

import json
import os
from typing import List, Optional, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")  # headless: write PDFs without a display server
import matplotlib.pyplot as plt  # noqa: E402

import config


# =====================================================================
# 0. DATA LOADING
# =====================================================================
def load_full() -> Tuple[np.ndarray, np.ndarray]:
    """Load the full window tensor and labels (19-channel canonical order)."""
    src = config.DIR_PREPROCESS
    x_path, y_path = os.path.join(src, "X.npy"), os.path.join(src, "y.npy")
    if not (os.path.exists(x_path) and os.path.exists(y_path)):
        raise FileNotFoundError(
            f"Missing {x_path}. Run `python 1_preprocessing.py` first."
        )
    return np.load(x_path), np.load(y_path)


def _savefig(fig: plt.Figure, fname: str) -> None:
    path = os.path.join(config.DIR_EDA, fname)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    if config.VERBOSE:
        print(f"   [✓] {path}")


# =====================================================================
# 1. POWER SPECTRAL DENSITY (per channel, per class)
# =====================================================================
def plot_psd(X: np.ndarray, y: np.ndarray, channels: List[str],
             target_channel: str = "Fz") -> None:
    from scipy.signal import welch

    if target_channel not in channels:
        target_channel = channels[0]
    ch = channels.index(target_channel)

    freqs, psd0 = welch(X[y == 0, ch, :], fs=config.SFREQ, nperseg=256)
    _, psd1 = welch(X[y == 1, ch, :], fs=config.SFREQ, nperseg=256)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(freqs, psd0.mean(axis=0), color="tab:blue", lw=2, label="Control (0)")
    ax.plot(freqs, psd1.mean(axis=0), color="tab:red", lw=2, label="Patient (1)")
    ax.axvspan(8, 13, color="gray", alpha=0.15, label="Alpha (8-13 Hz)")
    ax.set(title=f"Average PSD - channel {target_channel}",
           xlabel="Frequency (Hz)", ylabel=r"Power ($\mu V^2$/Hz)")
    ax.set_xlim(config.FREQ_LOW, config.FREQ_HIGH)
    ax.set_yscale("log")
    ax.grid(True, which="both", ls="--", alpha=0.5)
    ax.legend()
    _savefig(fig, f"psd_{target_channel}.pdf")


# =====================================================================
# 2. TEMPORAL GRAND AVERAGE
# =====================================================================
def plot_grand_average(X: np.ndarray, y: np.ndarray, channels: List[str],
                       targets: Optional[List[str]] = None) -> None:
    targets = targets or ["Cz", "Fz", "F4"]
    targets = [t for t in targets if t in channels] or [channels[0]]
    t_vec = np.arange(X.shape[2]) / config.SFREQ

    fig, axes = plt.subplots(len(targets), 1, figsize=(12, 3 * len(targets)),
                             squeeze=False)
    for ax, tgt in zip(axes[:, 0], targets):
        ch = channels.index(tgt)
        m0, m1 = X[y == 0, ch, :].mean(0), X[y == 1, ch, :].mean(0)
        s0 = X[y == 0, ch, :].std(0) * 0.1
        s1 = X[y == 1, ch, :].std(0) * 0.1
        ax.plot(t_vec, m0, color="tab:blue", label="Control")
        ax.fill_between(t_vec, m0 - s0, m0 + s0, color="tab:blue", alpha=0.15)
        ax.plot(t_vec, m1, color="tab:red", label="Patient")
        ax.fill_between(t_vec, m1 - s1, m1 + s1, color="tab:red", alpha=0.15)
        ax.set(title=f"Grand average - {tgt}", xlabel="Time (s)",
               ylabel="Z-scored amplitude")
        ax.grid(True, ls=":", alpha=0.5)
        ax.legend()
    fig.tight_layout()
    _savefig(fig, "grand_average.pdf")


# =====================================================================
# 3. FUNCTIONAL CONNECTIVITY (inter-channel correlation)
# =====================================================================
def plot_connectivity(X: np.ndarray, y: np.ndarray, channels: List[str],
                      max_windows: int = 200) -> None:
    import seaborn as sns

    idx = config.channel_indices(channels)
    X_sub = X[:, idx, :]

    def mean_corr(data: np.ndarray) -> np.ndarray:
        n = min(max_windows, len(data))
        return np.mean([np.corrcoef(data[i]) for i in range(n)], axis=0)

    corr0, corr1 = mean_corr(X_sub[y == 0]), mean_corr(X_sub[y == 1])
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, corr, title in [(axes[0], corr0, "Control"),
                            (axes[1], corr1, "Patient")]:
        sns.heatmap(corr, xticklabels=channels, yticklabels=channels,
                    cmap="RdBu_r", vmin=0.3, vmax=1.0, ax=ax, cbar=True)
        ax.set_title(f"Functional connectivity - {title}")
    fig.tight_layout()
    _savefig(fig, "connectivity.pdf")


# =====================================================================
# 4. RELATIVE BAND-POWER BOXPLOTS
# =====================================================================
def plot_band_boxplots(X: np.ndarray, y: np.ndarray, channels: List[str],
                       max_windows: int = 2000) -> None:
    import pandas as pd
    import seaborn as sns
    from scipy.signal import welch

    idx = config.channel_indices(channels)
    X_sub = X[:, idx, :]
    bands = {k: v for k, v in config.FREQ_BANDS.items() if k != "gamma"}

    n = min(max_windows, len(X_sub))
    sel = np.random.default_rng(config.RANDOM_STATE).choice(len(X_sub), n, replace=False)
    rows = []
    for i in sel:
        signal_mean = X_sub[i].mean(axis=0)
        freqs, psd = welch(signal_mean, fs=config.SFREQ, nperseg=256)
        total = np.sum(psd[(freqs >= config.FREQ_LOW) & (freqs <= config.FREQ_HIGH)])
        row = {"Class": config.CLASS_NAMES[y[i]]}
        for name, (lo, hi) in bands.items():
            mask = (freqs >= lo) & (freqs <= hi)
            row[name] = np.sum(psd[mask]) / (total + 1e-9)
        rows.append(row)
    df = pd.DataFrame(rows)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, name in zip(axes.flatten(), bands):
        sns.boxplot(x="Class", y=name, data=df, ax=ax, palette="Set2")
        ax.set(title=f"Relative power - {name}", ylabel="Energy proportion")
        ax.grid(True, ls=":", alpha=0.5)
    fig.tight_layout()
    _savefig(fig, "band_power_boxplots.pdf")


# =====================================================================
# 5. PCA -> UMAP 2D EMBEDDING (Top-6 montage)
# =====================================================================
def _flatten_windows(X: np.ndarray, channels: List[str]) -> np.ndarray:
    """Flatten (n, channels, samples) of the given montage to (n, features)."""
    idx = config.channel_indices(channels)
    return X[:, idx, :].reshape(len(X), -1)


def plot_pca_umap(X: np.ndarray, y: np.ndarray, channels: List[str],
                  max_windows: int = 3000) -> Optional[np.ndarray]:
    """PCA (50 comps) -> UMAP (2D); scatter coloured by class. Returns embedding."""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    try:
        import umap  # umap-learn
    except Exception as e:  # noqa: BLE001
        print(f"[pca_umap] umap-learn unavailable ({e}); skipping.")
        return None

    feats = _flatten_windows(X, channels)
    n = min(max_windows, len(feats))
    sel = np.random.default_rng(config.RANDOM_STATE).choice(len(feats), n, replace=False)
    feats, y_sel = feats[sel], y[sel]

    feats = StandardScaler().fit_transform(feats)
    n_pca = min(50, feats.shape[1], feats.shape[0] - 1)
    feats_pca = PCA(n_components=n_pca, random_state=config.RANDOM_STATE).fit_transform(feats)

    n_neighbors = min(15, max(2, n - 1))
    emb = umap.UMAP(n_components=2, n_neighbors=n_neighbors,
                    random_state=config.RANDOM_STATE).fit_transform(feats_pca)

    fig, ax = plt.subplots(figsize=(8, 7))
    for cls, color in [(0, "tab:blue"), (1, "tab:red")]:
        m = y_sel == cls
        ax.scatter(emb[m, 0], emb[m, 1], s=8, alpha=0.5, color=color,
                   label=config.CLASS_NAMES[cls])
    ax.set(title=f"PCA->UMAP 2D (Top-6 montage: {channels})",
           xlabel="UMAP-1", ylabel="UMAP-2")
    ax.legend()
    _savefig(fig, "pca_umap_top6.pdf")
    return np.column_stack([emb, y_sel])


# =====================================================================
# 6. SPECTRAL CLUSTERING (Top-6 montage)
# =====================================================================
def plot_spectral_clustering(embedding_with_label: Optional[np.ndarray]) -> Optional[float]:
    """Spectral Clustering on the 2D embedding; report ARI vs true labels."""
    if embedding_with_label is None:
        print("[spectral] no embedding available; skipping.")
        return None
    from sklearn.cluster import SpectralClustering
    from sklearn.metrics import adjusted_rand_score

    emb, y_true = embedding_with_label[:, :2], embedding_with_label[:, 2].astype(int)
    try:
        labels = SpectralClustering(
            n_clusters=2, affinity="nearest_neighbors",
            assign_labels="kmeans", random_state=config.RANDOM_STATE,
        ).fit_predict(emb)
    except Exception as e:  # noqa: BLE001
        print(f"[spectral] clustering failed ({e}); skipping.")
        return None

    ari = adjusted_rand_score(y_true, labels)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    axes[0].scatter(emb[:, 0], emb[:, 1], c=labels, cmap="coolwarm", s=8, alpha=0.6)
    axes[0].set_title("Spectral Clustering (unsupervised)")
    axes[1].scatter(emb[:, 0], emb[:, 1], c=y_true, cmap="coolwarm", s=8, alpha=0.6)
    axes[1].set_title("Ground-truth diagnosis")
    for ax in axes:
        ax.set(xlabel="UMAP-1", ylabel="UMAP-2")
    fig.suptitle(f"Spectral Clustering vs labels  |  Adjusted Rand Index = {ari:.3f}")
    fig.tight_layout()
    _savefig(fig, "spectral_clustering_top6.pdf")
    print(f"[spectral] Adjusted Rand Index (clusters vs diagnosis): {ari:.3f}")
    return ari


# =====================================================================
# 7. MAIN
# =====================================================================
def main() -> None:
    config.ensure_dirs()
    print("=" * 64)
    print(" PHASE 2 - EDA & DIMENSIONALITY REDUCTION")
    print("=" * 64)

    X, y = load_full()
    top6 = config.CHANNELS_6
    print(f"[INFO] X={X.shape} | classes: control={int((y==0).sum())}, "
          f"patient={int((y==1).sum())}")

    print("\n[*] Clinical EDA figures...")
    plot_psd(X, y, config.CHANNELS_19, target_channel="Fz")
    plot_grand_average(X, y, config.CHANNELS_19)
    plot_connectivity(X, y, top6)
    plot_band_boxplots(X, y, top6)

    print("\n[*] Dimensionality reduction (PCA->UMAP) on Top-6...")
    emb = plot_pca_umap(X, y, top6)

    print("\n[*] Spectral Clustering on Top-6 embedding...")
    ari = plot_spectral_clustering(emb)

    with open(os.path.join(config.DIR_EDA, "eda_meta.json"), "w") as f:
        json.dump({
            "top6_channels": top6,
            "spectral_clustering_ari": ari,
            "n_windows": int(len(y)),
        }, f, indent=2)

    print(f"\n[✓] EDA artifacts written to: {config.DIR_EDA}")
    print("=" * 64)


if __name__ == "__main__":
    main()
