"""
7_train_dl_sota.py
=================
Phase 4b: deep-learning pipeline (PyTorch, GPU-aware).

Image-based models consume the normalised STFT spectrograms from Module 5
(N, C, F, T), treating the C EEG channels as image channels:
  * SimpleEEGCNN     -- custom CNN trained from scratch.
  * ResNet18         -- transfer learning. conv1 adapted to C channels; the
                        backbone is FROZEN except `layer4` + `fc`, which are
                        unfrozen so the model adapts to the EEG domain.
  * ViT              -- timm Vision Transformer (last block + head fine-tuned).
  * EfficientNet     -- timm EfficientNet-B0 (last block + head fine-tuned).

Signal-based models consume the raw time-domain windows:
  * EEG-Conformer    -- torcheeg conv+attention model for raw EEG.
  * Chronos          -- Amazon's time-series foundation model used as a frozen
                        zero/few-shot embedder; a LogisticRegression head is
                        trained on the pooled embeddings.

Every optional dependency (timm / torcheeg / chronos) is guarded: if it is not
installed, that model is skipped with a warning instead of crashing the run.
Each model writes its TEST predictions to preds_<model>.npz for Module 8.

Run (requires Modules 1 and 5):
    python 7_train_dl_sota.py

Outputs:
    artifacts/07_dl_models/<model>.pt         best weights
    artifacts/07_dl_models/val_metrics.json
    artifacts/08_evaluation/predictions/preds_<model>.npz
"""

from __future__ import annotations

import json
import os
from typing import Callable, Dict, Optional, Tuple

import numpy as np

import config


# =====================================================================
# 0. DATA LOADING
# =====================================================================
def _normalize_per_channel(tr: np.ndarray, va: np.ndarray, te: np.ndarray):
    """log1p compression + per-channel z-score, statistics fit on TRAIN only.

    Same scheme Module 5 applies to STFT; used here for CWT/FFT which are stored
    as raw magnitudes so all representations reach the models on a comparable scale.
    """
    tr, va, te = np.log1p(tr), np.log1p(va), np.log1p(te)
    mean = tr.mean(axis=(0, 2, 3), keepdims=True)
    std = tr.std(axis=(0, 2, 3), keepdims=True) + 1e-8
    return (((tr - mean) / std).astype(np.float32),
            ((va - mean) / std).astype(np.float32),
            ((te - mean) / std).astype(np.float32))


def _fft_to_image(x: np.ndarray) -> np.ndarray:
    """Fold a per-channel FFT spectrum (N, C, F) into a square 2D image
    (N, C, h, h) so it can feed the same 2D CNN/Transformer backbones. The F
    magnitude bins are zero-padded to h*h and reshaped row-major."""
    n, c, f = x.shape
    h = int(np.ceil(np.sqrt(f)))
    pad = h * h - f
    if pad:
        x = np.pad(x, ((0, 0), (0, 0), (0, pad)), mode="constant")
    return x.reshape(n, c, h, h).astype(np.float32)


def _load_images(repr_name: str) -> Optional[Tuple[np.ndarray, ...]]:
    """Load the chosen time-frequency representation for the image models.

    repr_name in {'stft','cwt','fft'}:
      * stft -> X_stft_*_norm.npy (already normalised by Module 5)   (N, C, F, T)
      * cwt  -> X_cwt_*.npy, normalised here                          (N, C, F, T)
      * fft  -> X_fft_*.npy, folded to a square image + normalised    (N, C, h, h)
    Note: CWT is generated only on config.CWT_CHANNEL_SET (Top-6), so a CWT run
    uses fewer input channels than STFT/FFT - reported as a separate ablation row.
    """
    tf, ps = config.DIR_TIMEFREQ, config.DIR_PREPROCESS
    y = {s: np.load(os.path.join(ps, f"y_{s}.npy")) for s in ("train", "val", "test")}

    if repr_name == "stft":
        paths = {s: os.path.join(tf, config.pfx(f"X_stft_{s}_norm.npy"))
                 for s in ("train", "val", "test")}
        if not all(os.path.exists(p) for p in paths.values()):
            print(f"[WARN] STFT tensors missing (CHANNEL_SET={config.CHANNEL_SET}); "
                  "run Module 5. Skipping STFT image models.")
            return None
        X = {s: np.load(p) for s, p in paths.items()}
    elif repr_name in ("cwt", "fft"):
        stem = "X_cwt" if repr_name == "cwt" else "X_fft"
        paths = {s: os.path.join(tf, config.pfx(f"{stem}_{s}.npy"))
                 for s in ("train", "val", "test")}
        if not all(os.path.exists(p) for p in paths.values()):
            print(f"[WARN] {repr_name.upper()} tensors missing "
                  f"(CHANNEL_SET={config.CHANNEL_SET}); run Module 5. Skipping.")
            return None
        if repr_name == "cwt":
            # Memory-map and decimate the (huge) time axis so the full ~GB
            # array never fully materialises (see config.CWT_DL_TIME_BINS).
            raw = {}
            for s, p in paths.items():
                mm = np.load(p, mmap_mode="r")
                step = max(1, mm.shape[-1] // config.CWT_DL_TIME_BINS)
                raw[s] = np.ascontiguousarray(mm[..., ::step], dtype=np.float32)
                del mm
        else:  # fft spectra are small; load fully and fold to a square image
            raw = {s: _fft_to_image(np.load(p)) for s, p in paths.items()}
        tr, va, te = _normalize_per_channel(raw["train"], raw["val"], raw["test"])
        X = {"train": tr, "val": va, "test": te}
    else:
        raise ValueError(f"Unknown repr {repr_name!r}; expected stft|cwt|fft")

    return X["train"], X["val"], X["test"], y["train"], y["val"], y["test"]


def _load_raw() -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray,
                                  np.ndarray, np.ndarray, np.ndarray]]:
    ps = config.DIR_PREPROCESS
    if not os.path.exists(os.path.join(ps, "X_train.npy")):
        print("[WARN] preprocessed tensors missing; run `python 1_preprocessing.py`.")
        return None
    idx = config.channel_indices(config.active_channels())
    X = {s: np.load(os.path.join(ps, f"X_{s}.npy"))[:, idx, :] for s in ("train", "val", "test")}
    y = {s: np.load(os.path.join(ps, f"y_{s}.npy")) for s in ("train", "val", "test")}
    return X["train"], X["val"], X["test"], y["train"], y["val"], y["test"]


# =====================================================================
# 1. TORCH TRAIN / EVAL HARNESS
# =====================================================================
def _make_loaders(X_tr, y_tr, X_va, y_va, X_te, y_te):
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    def ds(X, y):
        return TensorDataset(torch.tensor(X, dtype=torch.float32),
                             torch.tensor(y, dtype=torch.float32))
    bs = config.BATCH_SIZE
    return (DataLoader(ds(X_tr, y_tr), batch_size=bs, shuffle=True),
            DataLoader(ds(X_va, y_va), batch_size=bs),
            DataLoader(ds(X_te, y_te), batch_size=bs))


def train_model(model, train_loader, val_loader, device,
                n_epochs: int, lr: float, patience: int):
    """Train with BCEWithLogitsLoss + early stopping; restore best val weights."""
    import torch
    import torch.nn as nn

    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    best_loss, best_state, no_improve = float("inf"), None, 0
    for epoch in range(1, n_epochs + 1):
        model.train()
        tr_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb).reshape(-1)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            tr_loss += loss.item() * xb.size(0)
        tr_loss /= len(train_loader.dataset)

        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                va_loss += criterion(model(xb).reshape(-1), yb).item() * xb.size(0)
        va_loss /= len(val_loader.dataset)
        print(f"    epoch {epoch:02d} | train={tr_loss:.4f} | val={va_loss:.4f}")

        if va_loss < best_loss:
            best_loss = va_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"    early stop at epoch {epoch}")
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def evaluate_model(model, loader, device):
    import torch
    model.eval()
    true, proba = [], []
    with torch.no_grad():
        for xb, yb in loader:
            logits = model(xb.to(device)).reshape(-1)
            proba.extend(torch.sigmoid(logits).cpu().numpy())
            true.extend(yb.numpy())
    true, proba = np.array(true), np.array(proba)
    return true, (proba >= 0.5).astype(int), proba


# =====================================================================
# 2. MODEL BUILDERS
# =====================================================================
def build_simple_cnn(n_channels: int):
    import torch.nn as nn

    class SimpleEEGCNN(nn.Module):
        def __init__(self, c: int):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(c, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
            )
            self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.3), nn.Linear(64, 1))

        def forward(self, x):
            return self.head(self.conv(x))

    return SimpleEEGCNN(n_channels)


def build_resnet18(n_channels: int):
    """ResNet18 transfer learning: adapt conv1 to C channels, freeze all but
    layer4 + fc, resize spectrograms to DL_RESIZE."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torchvision.models as tvm

    base = tvm.resnet18(weights=tvm.ResNet18_Weights.DEFAULT)
    old = base.conv1
    new = nn.Conv2d(n_channels, old.out_channels, old.kernel_size,
                    old.stride, old.padding, bias=False)
    with torch.no_grad():
        new.weight[:] = old.weight.mean(dim=1, keepdim=True).repeat(1, n_channels, 1, 1)
    base.conv1 = new
    base.fc = nn.Linear(base.fc.in_features, 1)

    # Freeze everything, then UNFREEZE the last conv block + new conv1 + fc.
    for p in base.parameters():
        p.requires_grad = False
    for module in (base.conv1, base.layer4, base.fc):
        for p in module.parameters():
            p.requires_grad = True

    size = config.DL_RESIZE

    class ResizeWrap(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            x = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
            return self.m(x)

    return ResizeWrap(base)


def _build_timm(model_name: str, n_channels: int, input_size: int,
                unfreeze_substrings: Tuple[str, ...]):
    import timm
    import torch.nn as nn
    import torch.nn.functional as F

    base = timm.create_model(model_name, pretrained=True,
                             in_chans=n_channels, num_classes=1)
    for p in base.parameters():
        p.requires_grad = False
    # Unfreeze the classifier head + any param whose name matches a substring
    # (e.g. the last transformer block / last conv stage).
    for p in base.get_classifier().parameters():
        p.requires_grad = True
    for name, p in base.named_parameters():
        if any(s in name for s in unfreeze_substrings):
            p.requires_grad = True

    class ResizeWrap(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            x = F.interpolate(x, size=(input_size, input_size),
                              mode="bilinear", align_corners=False)
            return self.m(x)

    return ResizeWrap(base)


def build_vit(n_channels: int):
    return _build_timm("vit_tiny_patch16_224", n_channels, 224,
                       unfreeze_substrings=("blocks.11", "norm.", "head"))


def build_efficientnet(n_channels: int):
    return _build_timm("efficientnet_b0", n_channels, 224,
                       unfreeze_substrings=("blocks.6", "conv_head", "classifier"))


def build_conformer(n_channels: int, n_samples: int):
    """torcheeg EEG-Conformer for raw windows; input reshaped to (B,1,C,T).

    torcheeg sizes its classification head from `sampling_rate`, building a mock
    input of shape (1, 1, num_electrodes, sampling_rate). The conv kernels are
    fixed, so `sampling_rate` is effectively "expected time length". We therefore
    pass the actual window length (n_samples=1250 for 5 s @ 250 Hz), NOT 250 -
    otherwise the flattened features (e.g. 3080) don't match the head (440)."""
    import torch.nn as nn
    from torcheeg.models import Conformer

    base = Conformer(num_electrodes=n_channels, sampling_rate=n_samples,
                     num_classes=1)

    class ConformerWrap(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            # x: (B, C, T) -> (B, 1, C, T) as torcheeg expects
            if x.dim() == 3:
                x = x.unsqueeze(1)
            out = self.m(x)
            return out[0] if isinstance(out, tuple) else out

    return ConformerWrap(base)


# =====================================================================
# 3. CHRONOS (frozen embeddings + logistic head)
# =====================================================================
def run_chronos(X_tr, y_tr, X_te, y_te, groups_test=None) -> Optional[dict]:
    """Use Chronos as a frozen embedder; train a logistic head on pooled embeds.

    Embedding = mean over channels and time of the Chronos encoder states.
    Chronos subsamples the test windows, so the matching subset of
    ``groups_test`` is saved for subject-level metrics. Returns metrics or None.
    """
    try:
        import torch
    except Exception as e:  # noqa: BLE001
        print(f"[Chronos] torch unavailable ({e}); skipping.")
        return None

    # Self-heal a clobbered chronos/__init__.py. chronos-forecasting relies on
    # __init__.py to (a) trigger pipeline-class registration in PipelineRegistry
    # and (b) re-export names like `MeanScaleUniformBins` onto the top-level
    # `chronos` namespace (the tokenizer does getattr(chronos, ...)). If an
    # unrelated `chronos` package emptied __init__.py, both break. Importing the
    # submodules and copying their public names onto `chronos` reproduces what
    # __init__.py would have done.
    try:
        import chronos as _chronos
        import chronos.base as _cb, chronos.chronos as _cc, chronos.chronos_bolt as _cbolt
        for _sub in (_cb, _cc, _cbolt):
            for _n in dir(_sub):
                if not _n.startswith("_") and not hasattr(_chronos, _n):
                    setattr(_chronos, _n, getattr(_sub, _n))
    except Exception:  # noqa: BLE001
        pass  # handled by the resolver below

    # Resolve a Chronos pipeline class robustly. chronos-forecasting v2 exposes
    # BaseChronosPipeline; v1 exposes ChronosPipeline. If __init__.py is empty,
    # fall back to importing straight from the submodules.
    Pipe = None
    errs = []
    for mod, cls in (("chronos", "BaseChronosPipeline"),
                     ("chronos", "ChronosPipeline"),
                     ("chronos.base", "BaseChronosPipeline"),
                     ("chronos.chronos", "ChronosPipeline")):
        try:
            Pipe = getattr(__import__(mod, fromlist=[cls]), cls)
            break
        except Exception as e:  # noqa: BLE001
            errs.append(f"{mod}.{cls}: {e}")
    if Pipe is None:
        print("[Chronos] no pipeline class importable; skipping.\n   "
              + "\n   ".join(errs))
        return None

    device = config.device_str()
    try:
        pipe = Pipe.from_pretrained(
            config.CHRONOS_MODEL_NAME,
            device_map=device,
            torch_dtype=torch.float32,
        )
    except Exception as e:  # noqa: BLE001
        print(f"[Chronos] could not load {config.CHRONOS_MODEL_NAME} ({e}); skipping.")
        return None

    def embed(X: np.ndarray, max_n: int) -> Tuple[np.ndarray, np.ndarray]:
        n = min(max_n, len(X))
        sel = np.random.default_rng(config.RANDOM_STATE).choice(len(X), n, replace=False)
        feats = []
        for i in sel:
            # One context per channel; pool encoder states over time, avg channels.
            ctx = torch.tensor(X[i], dtype=torch.float32)  # (C, T)
            out = pipe.embed(ctx)                           # (emb, scale) or emb
            emb = out[0] if isinstance(out, tuple) else out  # (C, T+1, D)
            feats.append(emb.float().mean(dim=(0, 1)).cpu().numpy())
        return np.asarray(feats), sel

    # Few-shot: cap embeddings for tractability on CPU.
    max_train = min(len(X_tr), 800)
    max_test = min(len(X_te), 800)
    print(f"[Chronos] embedding {max_train} train / {max_test} test windows...")
    try:
        Etr, sel_tr = embed(X_tr, max_train)
        Ete, sel_te = embed(X_te, max_test)
    except Exception as e:  # noqa: BLE001
        print(f"[Chronos] embedding failed ({e}); skipping.")
        return None

    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(max_iter=2000, class_weight="balanced",
                             random_state=config.RANDOM_STATE)
    clf.fit(Etr, y_tr[sel_tr])
    y_pred = clf.predict(Ete)
    y_proba = clf.predict_proba(Ete)[:, 1]
    y_true = y_te[sel_te]
    groups = groups_test[sel_te] if groups_test is not None else None
    m = config.compute_metrics(y_true, y_pred, y_proba)
    config.print_metrics("Chronos", m)
    config.save_predictions("Chronos", y_true, y_pred, y_proba, groups=groups)
    return m


# =====================================================================
# 4. MAIN
# =====================================================================
def _train_one(name: str, builder: Callable, loaders, device,
               groups_test=None) -> Optional[dict]:
    """Build, train, evaluate one torch model; save weights + test predictions.

    Test loaders use shuffle=False, so ``groups_test`` (subject ids in the
    original test order) stays row-aligned with the predictions for Module 8.
    """
    import torch
    train_loader, val_loader, test_loader = loaders
    print(f"\n[*] {name}")
    try:
        model = builder()
    except Exception as e:  # noqa: BLE001
        print(f"[{name}] build failed ({e}); skipping.")
        return None
    lr = config.DL_FINETUNE_LR if name in ("ResNet18", "ViT", "EfficientNet") else config.DL_LR
    try:
        model = train_model(model, train_loader, val_loader, device,
                            config.DL_N_EPOCHS, lr, config.DL_PATIENCE)
        yv, pv, prv = evaluate_model(model, val_loader, device)
        m = config.compute_metrics(yv, pv, prv)
        config.print_metrics(name, m)
        torch.save(model.state_dict(), os.path.join(config.DIR_DL, config.pfx(f"{name}.pt")))
        yt, pt, prt = evaluate_model(model, test_loader, device)
        config.save_predictions(name, yt, pt, prt, groups=groups_test)
        return m
    except Exception as e:  # noqa: BLE001
        print(f"[{name}] train/eval failed ({e}); skipping.")
        return None


def _parse_reprs(arg: Optional[str]) -> list[str]:
    """Resolve the --repr argument into a list of representations to train.

    Accepts 'all', a single repr, or a comma-separated list. Defaults to
    config.DL_REPR so a bare `python 7_train_dl_sota.py` (e.g. via run_all.sh)
    keeps its prior single-representation behaviour.
    """
    valid = ("stft", "cwt", "fft")
    if not arg:
        return [config.DL_REPR]
    if arg.lower() == "all":
        return list(valid)
    reprs = [r.strip().lower() for r in arg.split(",") if r.strip()]
    bad = [r for r in reprs if r not in valid]
    if bad:
        raise SystemExit(f"--repr: unknown {bad}; choose from {valid} or 'all'")
    return reprs


def main(reprs: Optional[list[str]] = None, train_signal: bool = True) -> None:
    config.ensure_dirs()
    try:
        import torch  # noqa: F401
    except Exception as e:  # noqa: BLE001
        print(f"[FATAL] PyTorch not installed ({e}). Install requirements first.")
        return

    reprs = reprs or [config.DL_REPR]
    device = config.device_str()
    print("=" * 64)
    print(f" PHASE 7 - DEEP LEARNING (SOTA)  device={device} | reprs={reprs}")
    print("=" * 64)

    # Subject ids per TEST window (image & raw tensors share the same window
    # order as groups_test) -> lets Module 8 compute subject-level metrics.
    gt_path = os.path.join(config.DIR_PREPROCESS, "groups_test.npy")
    groups_test = np.load(gt_path) if os.path.exists(gt_path) else None

    val_metrics: Dict[str, dict] = {}

    # ---- image-based models, once per requested representation ----
    # Each image model is tagged with the representation (e.g. 'ViT-CWT') so the
    # STFT / CWT / FFT variants form distinct rows in the Module 8 ablation table.
    for repr_name in reprs:
        data = _load_images(repr_name)
        if data is None:
            continue
        Xtr, Xva, Xte, ytr, yva, yte = data
        n_ch = Xtr.shape[1]
        tag = repr_name.upper()
        print(f"\n[INFO] {tag} input: {Xtr.shape} (C={n_ch})")
        loaders = _make_loaders(Xtr, ytr, Xva, yva, Xte, yte)
        builders: Dict[str, Callable] = {
            f"CNN-{tag}": lambda nc=n_ch: build_simple_cnn(nc),
            f"ResNet18-{tag}": lambda nc=n_ch: build_resnet18(nc),
            f"ViT-{tag}": lambda nc=n_ch: build_vit(nc),
            f"EfficientNet-{tag}": lambda nc=n_ch: build_efficientnet(nc),
        }
        for name, b in builders.items():
            m = _train_one(name, b, loaders, device, groups_test=groups_test)
            if m is not None:
                val_metrics[name] = m

    # ---- signal-based models (raw windows) -- representation-independent,
    #      so they are trained exactly once regardless of --repr. Skip with
    #      --no-signal on a supplementary image-only ablation pass to avoid
    #      re-running the slow Chronos embedding / Conformer. ----
    raw = _load_raw() if train_signal else None
    if not train_signal:
        print("\n[INFO] --no-signal: skipping EEG-Conformer & Chronos.")
    if raw is not None:
        Xtr_r, Xva_r, Xte_r, ytr_r, yva_r, yte_r = raw
        n_ch_r, n_samp = Xtr_r.shape[1], Xtr_r.shape[2]
        loaders_r = _make_loaders(Xtr_r, ytr_r, Xva_r, yva_r, Xte_r, yte_r)
        m = _train_one("EEG-Conformer", lambda: build_conformer(n_ch_r, n_samp),
                       loaders_r, device, groups_test=groups_test)
        if m is not None:
            val_metrics["EEG-Conformer"] = m

        m = run_chronos(Xtr_r, ytr_r, Xte_r, yte_r, groups_test=groups_test)
        if m is not None:
            val_metrics["Chronos"] = m

    with open(os.path.join(config.DIR_DL, config.pfx("val_metrics.json")), "w") as f:
        json.dump(val_metrics, f, indent=2)

    print(f"\n[✓] DL models done: {list(val_metrics)}")
    print(f"[✓] TEST predictions -> {config.DIR_PREDICTIONS}")
    print("=" * 64)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Phase 7 - deep-learning models")
    parser.add_argument(
        "--repr", default=None,
        help="time-frequency representation for the image models: "
             "stft | cwt | fft | all | comma-list (default: config.DL_REPR='%s')"
             % config.DL_REPR)
    parser.add_argument(
        "--no-signal", dest="signal", action="store_false",
        help="skip the raw-signal models (EEG-Conformer, Chronos); useful for a "
             "supplementary image-only representation pass")
    args = parser.parse_args()
    main(reprs=_parse_reprs(args.repr), train_signal=args.signal)
