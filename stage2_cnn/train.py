"""
Training loop for Stage 2 CNN (EfficientNet-B4 on DICOM mammograms).

Usage:
    cd stage2_cnn
    python train.py

Two-phase strategy
──────────────────
Phase 1 (warmup):   Backbone frozen. Only the new classifier head trains.
                    Runs for config.EPOCHS_WARMUP epochs.
Phase 2 (finetune): All layers unfrozen at 1/10 the learning rate.
                    Runs for config.EPOCHS_FINETUNE epochs with early stopping.

Outputs saved to stage2_cnn/
  checkpoints/best_model.pth   — best val AUC checkpoint
  logs/train_log.csv           — per-epoch metrics
  results/training_curves.png  — loss + AUC plots
"""

import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

# allow imports from this folder regardless of cwd
sys.path.insert(0, str(Path(__file__).parent))
import config
from dataset import load_splits, MammogramDataset, make_balanced_sampler
from model import MammoNet

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── Reproducibility ────────────────────────────────────────────────────────
def set_seed(seed: int = config.SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


# ── Label-smoothed BCE loss ────────────────────────────────────────────────
class SmoothBCELoss(nn.Module):
    def __init__(self, smoothing: float = config.LABEL_SMOOTH):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets_smooth = targets * (1 - self.smoothing) + 0.5 * self.smoothing
        return nn.functional.binary_cross_entropy_with_logits(logits, targets_smooth)


# ── One epoch ─────────────────────────────────────────────────────────────
def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()
    total_loss, all_labels, all_probs = 0.0, [], []

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch_idx, (images, labels) in enumerate(loader):
            images, labels = images.to(device), labels.to(device)

            logits = model(images)
            loss   = criterion(logits, labels)

            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
                optimizer.step()

            total_loss += loss.item() * len(labels)
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(torch.sigmoid(logits).detach().cpu().numpy())

            if train and (batch_idx + 1) % 20 == 0:
                print(f"    batch {batch_idx+1}/{len(loader)}  loss={loss.item():.4f}",
                      flush=True)

    avg_loss = total_loss / len(all_labels)
    auc      = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.5
    return avg_loss, auc


# ── Save checkpoint ────────────────────────────────────────────────────────
def save_ckpt(model, epoch: int, val_auc: float, path: Path):
    torch.save({
        "epoch":       epoch,
        "model_state": model.state_dict(),
        "val_auc":     val_auc,
    }, path)
    print(f"  ✓ Checkpoint saved  →  {path.name}  (val AUC={val_auc:.4f})")


# ── Training curves ────────────────────────────────────────────────────────
def save_curves(log_df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(log_df["epoch"], log_df["train_loss"], label="Train Loss", color="#0f3460")
    axes[0].plot(log_df["epoch"], log_df["val_loss"],   label="Val Loss",   color="#e94560", ls="--")
    axes[0].set(title="Loss", xlabel="Epoch", ylabel="BCE Loss")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(log_df["epoch"], log_df["train_auc"], label="Train AUC", color="#0f3460")
    axes[1].plot(log_df["epoch"], log_df["val_auc"],   label="Val AUC",   color="#e94560", ls="--")
    axes[1].axhline(0.9, ls=":", color="gray", lw=1, label="0.90 target")
    axes[1].set(title="AUC-ROC", xlabel="Epoch", ylabel="AUC", ylim=(0.5, 1.02))
    axes[1].legend(); axes[1].grid(alpha=0.3)

    # mark phase boundary
    if "phase" in log_df.columns:
        boundary = log_df[log_df["phase"] == "finetune"]["epoch"].min()
        if pd.notna(boundary):
            for ax in axes:
                ax.axvline(boundary, ls=":", color="orange", lw=1.5,
                           label=f"Unfreeze (ep {int(boundary)})")

    plt.tight_layout()
    out = config.RESULTS_DIR / "training_curves.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Curves saved → {out}")


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    config.CKPT_DIR.mkdir(exist_ok=True)
    config.LOG_DIR.mkdir(exist_ok=True)
    config.RESULTS_DIR.mkdir(exist_ok=True)

    set_seed()
    device = config.get_device()
    print(f"\n{'='*55}")
    print(f"  Stage 2 CNN Training — MammoNet (EfficientNet-B4)")
    print(f"  Device: {device}")
    print(f"{'='*55}\n")

    # ── Data ──────────────────────────────────────────────────────
    print("[1/4] Loading DICOM paths from CSV...")
    train_df, test_df = load_splits()

    train_ds = MammogramDataset(train_df, train=True)
    val_ds   = MammogramDataset(test_df,  train=False)

    sampler    = make_balanced_sampler(train_ds)
    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE,
                              sampler=sampler, num_workers=config.NUM_WORKERS,
                              pin_memory=(str(device) != "mps"))
    val_loader   = DataLoader(val_ds, batch_size=config.BATCH_SIZE,
                              shuffle=False, num_workers=config.NUM_WORKERS,
                              pin_memory=(str(device) != "mps"))

    # ── Model ─────────────────────────────────────────────────────
    print("\n[2/4] Building model...")
    model     = MammoNet().to(device)
    criterion = SmoothBCELoss()

    # ── Phase 1: Warmup ───────────────────────────────────────────
    print(f"\n[3/4] Phase 1 — Warmup ({config.EPOCHS_WARMUP} epochs, backbone frozen)")
    model.freeze_backbone()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.LR_WARMUP, weight_decay=config.WEIGHT_DECAY,
    )

    log_rows = []
    best_auc, best_epoch, patience_ctr = 0.0, 0, 0

    for epoch in range(1, config.EPOCHS_WARMUP + 1):
        t0 = time.time()
        tr_loss, tr_auc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        va_loss, va_auc = run_epoch(model, val_loader,   criterion, None,      device, train=False)
        elapsed = time.time() - t0

        print(f"  Ep {epoch:02d}/{config.EPOCHS_WARMUP} | "
              f"train_loss={tr_loss:.4f} train_auc={tr_auc:.4f} | "
              f"val_loss={va_loss:.4f} val_auc={va_auc:.4f} | {elapsed:.0f}s")

        log_rows.append(dict(epoch=epoch, phase="warmup",
                             train_loss=tr_loss, train_auc=tr_auc,
                             val_loss=va_loss, val_auc=va_auc))

        if va_auc > best_auc:
            best_auc   = va_auc
            best_epoch = epoch
            save_ckpt(model, epoch, va_auc,
                      config.CKPT_DIR / "best_model.pth")

    # ── Phase 2: Fine-tune ────────────────────────────────────────
    print(f"\n[4/4] Phase 2 — Fine-tune ({config.EPOCHS_FINETUNE} epochs, all layers)")
    model.unfreeze_all()
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=config.LR_FINETUNE,
                                  weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.EPOCHS_FINETUNE, eta_min=1e-6
    )

    patience_ctr = 0
    for epoch in range(config.EPOCHS_WARMUP + 1,
                       config.EPOCHS_WARMUP + config.EPOCHS_FINETUNE + 1):
        t0 = time.time()
        tr_loss, tr_auc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        va_loss, va_auc = run_epoch(model, val_loader,   criterion, None,      device, train=False)
        scheduler.step()
        elapsed = time.time() - t0

        print(f"  Ep {epoch:02d}/{config.EPOCHS_WARMUP+config.EPOCHS_FINETUNE} | "
              f"train_loss={tr_loss:.4f} train_auc={tr_auc:.4f} | "
              f"val_loss={va_loss:.4f} val_auc={va_auc:.4f} | {elapsed:.0f}s")

        log_rows.append(dict(epoch=epoch, phase="finetune",
                             train_loss=tr_loss, train_auc=tr_auc,
                             val_loss=va_loss, val_auc=va_auc))

        if va_auc > best_auc:
            best_auc     = va_auc
            best_epoch   = epoch
            patience_ctr = 0
            save_ckpt(model, epoch, va_auc,
                      config.CKPT_DIR / "best_model.pth")
        else:
            patience_ctr += 1
            print(f"    No improvement ({patience_ctr}/{config.PATIENCE})")
            if patience_ctr >= config.PATIENCE:
                print(f"  Early stopping at epoch {epoch}")
                break

    # ── Save log & curves ─────────────────────────────────────────
    log_df = pd.DataFrame(log_rows)
    log_df.to_csv(config.LOG_DIR / "train_log.csv", index=False)
    save_curves(log_df)

    print(f"\n{'='*55}")
    print(f"  Training complete.")
    print(f"  Best Val AUC: {best_auc:.4f}  at epoch {best_epoch}")
    print(f"  Checkpoint  : {config.CKPT_DIR / 'best_model.pth'}")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
