"""
Training loop for Stage 2 CNN (EfficientNet-B4 on DICOM mammograms).

Usage:
    cd stage2_cnn
    python train.py               # fresh start
    python train.py --resume      # continue from last saved epoch

Two-phase strategy
──────────────────
Phase 1 (warmup):   Backbone frozen. Only the new classifier head trains.
Phase 2 (finetune): All layers unfrozen at lower LR with cosine schedule.

Checkpoints saved to stage2_cnn/checkpoints/
  best_model.pth      — best val-AUC ever seen
  last_checkpoint.pth — saved after EVERY epoch (used for --resume)
"""

import argparse
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

sys.path.insert(0, str(Path(__file__).parent))
import config
from dataset import load_splits, MammogramDataset, make_balanced_sampler
from model import MammoNet

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── Reproducibility ────────────────────────────────────────────────────────
def set_seed(seed: int = config.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


# ── Label-smoothed BCE ─────────────────────────────────────────────────────
class SmoothBCELoss(nn.Module):
    def __init__(self, smoothing: float = config.LABEL_SMOOTH):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, logits, targets):
        targets_s = targets * (1 - self.smoothing) + 0.5 * self.smoothing
        return nn.functional.binary_cross_entropy_with_logits(logits, targets_s)


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
                print(f"    batch {batch_idx+1}/{len(loader)}  "
                      f"loss={loss.item():.4f}", flush=True)

    avg_loss = total_loss / len(all_labels)
    auc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.5
    return avg_loss, auc


# ── Checkpointing ──────────────────────────────────────────────────────────
def save_best(model, epoch, val_auc):
    path = config.CKPT_DIR / "best_model.pth"
    torch.save({"epoch": epoch, "model_state": model.state_dict(),
                "val_auc": val_auc}, path)
    print(f"  ★ Best checkpoint saved (val AUC={val_auc:.4f})")


def save_resume(model, optimizer, scheduler, epoch, phase,
                best_auc, best_epoch, patience_ctr, log_rows):
    """Save everything needed to resume from this exact epoch."""
    path = config.CKPT_DIR / "last_checkpoint.pth"
    torch.save({
        "epoch":        epoch,
        "phase":        phase,
        "model_state":  model.state_dict(),
        "optim_state":  optimizer.state_dict(),
        "sched_state":  scheduler.state_dict() if scheduler else None,
        "best_auc":     best_auc,
        "best_epoch":   best_epoch,
        "patience_ctr": patience_ctr,
        "log_rows":     log_rows,
    }, path)
    print(f"  💾 Resume checkpoint saved → last_checkpoint.pth (epoch {epoch})")


def load_resume(device):
    path = config.CKPT_DIR / "last_checkpoint.pth"
    if not path.exists():
        print("  No resume checkpoint found — starting fresh.")
        return None
    ckpt = torch.load(path, map_location=device)
    print(f"\n  ✅ Resuming from epoch {ckpt['epoch']}  "
          f"phase={ckpt['phase']}  best_auc={ckpt['best_auc']:.4f}")
    return ckpt


# ── Training curves ────────────────────────────────────────────────────────
def save_curves(log_df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(log_df["epoch"], log_df["train_loss"], label="Train", color="#0f3460")
    axes[0].plot(log_df["epoch"], log_df["val_loss"],   label="Val",   color="#e94560", ls="--")
    axes[0].set(title="Loss", xlabel="Epoch", ylabel="BCE Loss")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(log_df["epoch"], log_df["train_auc"], label="Train", color="#0f3460")
    axes[1].plot(log_df["epoch"], log_df["val_auc"],   label="Val",   color="#e94560", ls="--")
    axes[1].axhline(0.9, ls=":", color="gray", lw=1, label="0.90 target")
    axes[1].set(title="AUC-ROC", xlabel="Epoch", ylabel="AUC", ylim=(0.5, 1.02))
    axes[1].legend(); axes[1].grid(alpha=0.3)

    if "phase" in log_df.columns:
        boundary = log_df[log_df["phase"] == "finetune"]["epoch"].min()
        if pd.notna(boundary):
            for ax in axes:
                ax.axvline(boundary, ls=":", color="orange", lw=1.5,
                           label=f"Unfreeze ep {int(boundary)}")

    plt.tight_layout()
    out = config.RESULTS_DIR / "training_curves.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Curves saved → {out}")


# ── Main ───────────────────────────────────────────────────────────────────
def main(resume: bool = False):
    config.CKPT_DIR.mkdir(exist_ok=True)
    config.LOG_DIR.mkdir(exist_ok=True)
    config.RESULTS_DIR.mkdir(exist_ok=True)

    set_seed()
    device = config.get_device()

    print(f"\n{'='*55}")
    print(f"  Stage 2 CNN — MammoNet (EfficientNet-B4)")
    print(f"  Device : {device}")
    print(f"  Mode   : {'RESUME' if resume else 'FRESH START'}")
    print(f"{'='*55}\n")

    # ── Data ──────────────────────────────────────────────────────
    print("[1/4] Loading DICOM paths...")
    train_df, test_df = load_splits()
    train_ds   = MammogramDataset(train_df, train=True)
    val_ds     = MammogramDataset(test_df,  train=False)
    sampler    = make_balanced_sampler(train_ds)
    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE,
                              sampler=sampler, num_workers=config.NUM_WORKERS,
                              pin_memory=False)
    val_loader   = DataLoader(val_ds, batch_size=config.BATCH_SIZE,
                              shuffle=False, num_workers=config.NUM_WORKERS,
                              pin_memory=False)

    # ── Model & loss ──────────────────────────────────────────────
    print("\n[2/4] Building model...")
    model     = MammoNet().to(device)
    criterion = SmoothBCELoss()

    # ── Restore or init state ─────────────────────────────────────
    ckpt         = load_resume(device) if resume else None
    log_rows     = ckpt["log_rows"]     if ckpt else []
    best_auc     = ckpt["best_auc"]     if ckpt else 0.0
    best_epoch   = ckpt["best_epoch"]   if ckpt else 0
    patience_ctr = ckpt["patience_ctr"] if ckpt else 0
    start_phase  = ckpt["phase"]        if ckpt else "warmup"
    start_epoch  = ckpt["epoch"] + 1    if ckpt else 1

    if ckpt:
        model.load_state_dict(ckpt["model_state"])

    # ─────────────────────────────────────────────────────────────
    # PHASE 1 — Warmup
    # ─────────────────────────────────────────────────────────────
    warmup_end   = config.EPOCHS_WARMUP
    finetune_end = config.EPOCHS_WARMUP + config.EPOCHS_FINETUNE

    if start_phase == "warmup":
        print(f"\n[3/4] Phase 1 — Warmup (epochs 1–{warmup_end}, backbone frozen)")
        model.freeze_backbone()
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config.LR_WARMUP, weight_decay=config.WEIGHT_DECAY,
        )
        if ckpt and ckpt["phase"] == "warmup":
            optimizer.load_state_dict(ckpt["optim_state"])

        for epoch in range(start_epoch, warmup_end + 1):
            t0 = time.time()
            tr_loss, tr_auc = run_epoch(model, train_loader, criterion,
                                        optimizer, device, train=True)
            va_loss, va_auc = run_epoch(model, val_loader, criterion,
                                        None, device, train=False)
            elapsed = time.time() - t0

            print(f"  Ep {epoch:02d}/{warmup_end} | "
                  f"train_loss={tr_loss:.4f} train_auc={tr_auc:.4f} | "
                  f"val_loss={va_loss:.4f} val_auc={va_auc:.4f} | {elapsed:.0f}s")

            log_rows.append(dict(epoch=epoch, phase="warmup",
                                 train_loss=tr_loss, train_auc=tr_auc,
                                 val_loss=va_loss, val_auc=va_auc))

            if va_auc > best_auc:
                best_auc, best_epoch = va_auc, epoch
                save_best(model, epoch, va_auc)

            save_resume(model, optimizer, None, epoch, "warmup",
                        best_auc, best_epoch, patience_ctr, log_rows)

        # reset for phase 2
        start_epoch  = warmup_end + 1
        patience_ctr = 0

    # ─────────────────────────────────────────────────────────────
    # PHASE 2 — Fine-tune
    # ─────────────────────────────────────────────────────────────
    print(f"\n[4/4] Phase 2 — Fine-tune (epochs {warmup_end+1}–{finetune_end}, all layers)")
    model.unfreeze_all()
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=config.LR_FINETUNE,
                                  weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.EPOCHS_FINETUNE, eta_min=1e-6,
    )

    if ckpt and ckpt["phase"] == "finetune":
        optimizer.load_state_dict(ckpt["optim_state"])
        if ckpt["sched_state"]:
            scheduler.load_state_dict(ckpt["sched_state"])
        patience_ctr = ckpt["patience_ctr"]
        start_epoch  = ckpt["epoch"] + 1

    for epoch in range(start_epoch, finetune_end + 1):
        t0 = time.time()
        tr_loss, tr_auc = run_epoch(model, train_loader, criterion,
                                    optimizer, device, train=True)
        va_loss, va_auc = run_epoch(model, val_loader, criterion,
                                    None, device, train=False)
        scheduler.step()
        elapsed = time.time() - t0

        print(f"  Ep {epoch:02d}/{finetune_end} | "
              f"train_loss={tr_loss:.4f} train_auc={tr_auc:.4f} | "
              f"val_loss={va_loss:.4f} val_auc={va_auc:.4f} | {elapsed:.0f}s")

        log_rows.append(dict(epoch=epoch, phase="finetune",
                             train_loss=tr_loss, train_auc=tr_auc,
                             val_loss=va_loss, val_auc=va_auc))

        if va_auc > best_auc:
            best_auc, best_epoch = va_auc, epoch
            patience_ctr = 0
            save_best(model, epoch, va_auc)
        else:
            patience_ctr += 1
            print(f"    No improvement ({patience_ctr}/{config.PATIENCE})")
            if patience_ctr >= config.PATIENCE:
                print(f"  Early stopping at epoch {epoch}")
                save_resume(model, optimizer, scheduler, epoch, "finetune",
                            best_auc, best_epoch, patience_ctr, log_rows)
                break

        save_resume(model, optimizer, scheduler, epoch, "finetune",
                    best_auc, best_epoch, patience_ctr, log_rows)

    # ── Save log & curves ─────────────────────────────────────────
    log_df = pd.DataFrame(log_rows)
    log_df.to_csv(config.LOG_DIR / "train_log.csv", index=False)
    save_curves(log_df)

    print(f"\n{'='*55}")
    print(f"  Training complete.")
    print(f"  Best Val AUC : {best_auc:.4f}  at epoch {best_epoch}")
    print(f"  Checkpoint   : {config.CKPT_DIR / 'best_model.pth'}")
    print(f"{'='*55}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true",
                        help="Resume from last_checkpoint.pth")
    args = parser.parse_args()
    main(resume=args.resume)
