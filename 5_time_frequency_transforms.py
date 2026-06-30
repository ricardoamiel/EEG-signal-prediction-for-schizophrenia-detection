"""
5_time_frequency_transforms.py
=============================
Phase 2 (DL branch): convert the 1-D EEG windows into 2-D/3-D time-frequency
images that the deep models in Module 7 consume.

Three representations:
  * FFT       -- per-channel magnitude spectrum (0.5-45 Hz), a compact 1-D-per-
                 channel feature image; also summarised as band powers.
  * STFT      -- spectrograms (magnitude of the short-time Fourier transform),
                 the primary CNN/Transformer input.
  * DWT/CWT   -- continuous-wavelet scalograms (complex Morlet). The CWT is the
                 most expensive transform, so it is computed ONLY on the Top-6
                 montage (config.CWT_CHANNEL_SET) to save computation time.

All transforms are generated per split (train/val/test) directly from the
matching preprocessed tensor, so sample counts always line up. Spectrograms are
log-compressed and z-scored using statistics fit on TRAIN ONLY (no leakage).

Run (requires `python 1_preprocessing.py`):
    python 5_time_frequency_transforms.py

Outputs (under artifacts/05_time_frequency/):
    X_stft_<split>_norm.npy      (N, C, F, T)   normalised spectrograms  [CNN input]
    X_cwt_<split>.npy            (N, 6, F, T)   Top-6 scalograms
    X_fft_<split>.npy            (N, C, F)      per-channel magnitude spectra
    X_fft_bandpower_<split>.npy  (N, C, 5)      delta..gamma band powers
    sample_timefrequency.pdf     QC figure (STFT + CWT of one window)
    timefreq_meta.json
"""

from __future__ import annotations

import json
import os
from typing import List, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import config


# =====================================================================
# 0. LOAD
# =====================================================================
def _load_split(split: str, channels: List[str]) -> np.ndarray:
    """Load preprocessed tensor for a split, sliced to the given montage."""
    x_path = os.path.join(config.DIR_PREPROCESS, f"X_{split}.npy")
    if not os.path.exists(x_path):
        raise FileNotFoundError(
            f"Missing {x_path}. Run `python 1_preprocessing.py` first."
        )
    X = np.load(x_path)
    return X[:, config.channel_indices(channels), :]


# =====================================================================
# 1. FFT
# =====================================================================
def generate_fft(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Per-channel single-sided magnitude spectrum + per-band power.

    Returns (spectra (N, C, F_band), band_power (N, C, n_bands)) where F_band is
    restricted to the [FREQ_LOW, FREQ_HIGH] band.
    """
    n, c, samples = X.shape
    freqs = np.fft.rfftfreq(samples, d=1.0 / config.SFREQ)
    band_mask = (freqs >= config.FREQ_LOW) & (freqs <= config.FREQ_HIGH)

    mag = np.abs(np.fft.rfft(X, axis=2))          # (N, C, F)
    spectra = mag[:, :, band_mask]                 # (N, C, F_band)

    bands = list(config.FREQ_BANDS.items())
    band_power = np.zeros((n, c, len(bands)), dtype=np.float32)
    for bi, (_name, (lo, hi)) in enumerate(bands):
        m = (freqs >= lo) & (freqs <= hi)
        band_power[:, :, bi] = mag[:, :, m].sum(axis=2)
    return spectra.astype(np.float32), band_power


# =====================================================================
# 2. STFT (spectrograms)
# =====================================================================
def generate_stft(X: np.ndarray) -> np.ndarray:
    """STFT magnitude per channel, cropped to [FREQ_LOW, FREQ_HIGH] -> (N,C,F,T)."""
    from scipy.signal import stft

    out = []
    for i in range(len(X)):
        per_channel = []
        for ch in range(X.shape[1]):
            f, _t, Zxx = stft(X[i, ch], fs=config.SFREQ,
                              nperseg=config.STFT_NPERSEG,
                              noverlap=config.STFT_NOVERLAP)
            mask = (f >= config.FREQ_LOW) & (f <= config.FREQ_HIGH)
            per_channel.append(np.abs(Zxx)[mask, :])
        out.append(np.asarray(per_channel))
    return np.asarray(out, dtype=np.float32)


def normalize_stft(
    train: np.ndarray, val: np.ndarray, test: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """log1p compression + per-channel z-score (stats from TRAIN only)."""
    tr, va, te = np.log1p(train), np.log1p(val), np.log1p(test)
    mean = tr.mean(axis=(0, 2, 3), keepdims=True)
    std = tr.std(axis=(0, 2, 3), keepdims=True) + 1e-8
    return ((tr - mean) / std).astype(np.float32), \
           ((va - mean) / std).astype(np.float32), \
           ((te - mean) / std).astype(np.float32)


# =====================================================================
# 3. CWT (scalograms) -- Top-6 only
# =====================================================================
def generate_cwt(X: np.ndarray) -> np.ndarray:
    """Complex-Morlet CWT magnitude per channel -> (N, C, n_freqs, samples)."""
    import pywt

    frequencies = np.linspace(config.FREQ_LOW, config.FREQ_HIGH, config.CWT_N_FREQS)
    scales = pywt.frequency2scale(config.CWT_WAVELET, frequencies / config.SFREQ)

    out = []
    for i in range(len(X)):
        per_channel = []
        for ch in range(X.shape[1]):
            coefs, _ = pywt.cwt(X[i, ch], scales, config.CWT_WAVELET,
                                sampling_period=1.0 / config.SFREQ)
            per_channel.append(np.abs(coefs).astype(np.float32))
        out.append(np.asarray(per_channel))
    return np.asarray(out, dtype=np.float32)


# =====================================================================
# 4. QC FIGURE
# =====================================================================
def plot_sample(stft_img: np.ndarray, cwt_img: np.ndarray) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    im0 = axes[0].imshow(stft_img, aspect="auto", cmap="jet", origin="lower")
    axes[0].set(title="STFT spectrogram (channel 0)",
                xlabel="Time bins", ylabel="Frequency (0.5-45 Hz)")
    fig.colorbar(im0, ax=axes[0], label="Magnitude")

    im1 = axes[1].imshow(cwt_img, aspect="auto", cmap="jet", origin="lower")
    axes[1].set(title="CWT scalogram (channel 0)",
                xlabel="Time samples", ylabel="Frequency (0.5-45 Hz)")
    fig.colorbar(im1, ax=axes[1], label="|Morlet coeff|")
    fig.tight_layout()
    path = os.path.join(config.DIR_TIMEFREQ, config.pfx("sample_timefrequency.pdf"))
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"   [✓] {path}")


# =====================================================================
# 5. MAIN
# =====================================================================
def main() -> None:
    config.ensure_dirs()
    out = config.DIR_TIMEFREQ
    stft_channels = config.active_channels()        # CNN montage (e.g. Top-6)
    cwt_channels = config.CHANNEL_SETS[config.CWT_CHANNEL_SET]

    print("=" * 64)
    print(" PHASE 5 - TIME-FREQUENCY TRANSFORMS")
    print(f"   STFT/FFT montage: {stft_channels}")
    print(f"   CWT montage (cost-limited): {cwt_channels}")
    print("=" * 64)

    # ---- FFT + STFT (active montage) ----
    stft_raw = {}
    for split in ("train", "val", "test"):
        X = _load_split(split, stft_channels)
        spectra, bandpow = generate_fft(X)
        np.save(os.path.join(out, config.pfx(f"X_fft_{split}.npy")), spectra)
        np.save(os.path.join(out, config.pfx(f"X_fft_bandpower_{split}.npy")), bandpow)
        print(f"[FFT]  {split}: spectra={spectra.shape}, bandpow={bandpow.shape}")

        stft_raw[split] = generate_stft(X)
        print(f"[STFT] {split}: {stft_raw[split].shape}")

    tr, va, te = normalize_stft(stft_raw["train"], stft_raw["val"], stft_raw["test"])
    np.save(os.path.join(out, config.pfx("X_stft_train_norm.npy")), tr)
    np.save(os.path.join(out, config.pfx("X_stft_val_norm.npy")), va)
    np.save(os.path.join(out, config.pfx("X_stft_test_norm.npy")), te)
    print(f"[STFT] normalised (log1p+zscore, train stats): train={tr.shape}")

    # ---- CWT (Top-6 only) ----
    cwt_shapes = {}
    cwt_first = None
    for split in ("train", "val", "test"):
        Xc = _load_split(split, cwt_channels)
        cwt = generate_cwt(Xc)
        np.save(os.path.join(out, config.pfx(f"X_cwt_{split}.npy")), cwt)
        cwt_shapes[split] = list(cwt.shape)
        if split == "train" and len(cwt):
            cwt_first = cwt[0, 0]
        print(f"[CWT]  {split}: {cwt.shape}")

    # ---- QC figure ----
    if len(tr) and cwt_first is not None:
        plot_sample(tr[0, 0], cwt_first)

    with open(os.path.join(out, config.pfx("timefreq_meta.json")), "w") as f:
        json.dump({
            "stft_channels": stft_channels,
            "cwt_channels": cwt_channels,
            "stft_norm_shape": list(tr.shape),
            "cwt_shapes": cwt_shapes,
            "stft_nperseg": config.STFT_NPERSEG,
            "cwt_n_freqs": config.CWT_N_FREQS,
        }, f, indent=2)

    print(f"\n[✓] Time-frequency artifacts written to: {out}")
    print("=" * 64)


if __name__ == "__main__":
    main()
