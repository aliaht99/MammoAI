"""
Full evaluation of the trained CNN on the test set.

Outputs (all saved to stage2_cnn/results/):
  test_metrics.csv          — AUC, sensitivity, specificity, F1 etc.
  roc_curve.png             — ROC curve vs Stage 1 baseline
  confusion_matrix.png      — confusion matrix
  gradcam/                  — GradCAM saliency maps for 12 sample images

Usage:
    cd stage2_cnn
    python evaluate.py
    python evaluate.py --ckpt checkpoints/best_model.pth
"""

import sys, argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_auc_score, roc_curve, average_precision_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
import config
from dataset import load_splits, MammogramDataset
from model import load_checkpoint


# ── GradCAM implementation ─────────────────────────────────────────────────
class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model        = model
        self.gradients    = None
        self.activations  = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, _, __, output):
        self.activations = output.detach()

    def _save_gradient(self, _, __, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, image: torch.Tensor) -> np.ndarray:
        self.model.eval()
        image = image.unsqueeze(0)
        logit = self.model(image)
        self.model.zero_grad()
        logit.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam     = (weights * self.activations).sum(dim=1, keepdim=True)
        cam     = torch.relu(cam).squeeze().cpu().numpy()
        cam     = (cam - cam.min()) / (cam.ptp() + 1e-8)
        return cam


def save_gradcam_grid(model, dataset, device, n: int = 12):
    out_dir = config.RESULTS_DIR / "gradcam"
    out_dir.mkdir(exist_ok=True)

    # hook last conv block of EfficientNet-B4
    target_layer = model.backbone.features[-1]
    gcam = GradCAM(model, target_layer)

    indices = np.random.choice(len(dataset), size=min(n, len(dataset)), replace=False)
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes.flatten()

    for ax_idx, ds_idx in enumerate(indices):
        image, label = dataset[ds_idx]
        image_dev = image.to(device)
        cam = gcam(image_dev)

        # resize cam to image size
        from torchvision.transforms.functional import resize as tv_resize
        import torch.nn.functional as F
        cam_t = torch.tensor(cam).unsqueeze(0).unsqueeze(0)
        cam_r = F.interpolate(cam_t, size=image.shape[1:], mode="bilinear",
                              align_corners=False).squeeze().numpy()

        # display
        img_np = image.permute(1, 2, 0).numpy()
        img_np = (img_np - img_np.min()) / (img_np.ptp() + 1e-8)
        axes[ax_idx].imshow(img_np, cmap="gray")
        axes[ax_idx].imshow(cam_r, alpha=0.45, cmap="jet")
        lbl_str = "Malignant" if label.item() == 1 else "Benign"
        axes[ax_idx].set_title(lbl_str, fontsize=9, color="red" if label == 1 else "green")
        axes[ax_idx].axis("off")

    plt.suptitle("GradCAM — Suspicious Region Highlights", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_dir / "gradcam_grid.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  GradCAM grid saved → {out_dir / 'gradcam_grid.png'}")


# ── Main evaluation ────────────────────────────────────────────────────────
def evaluate(ckpt_path: str | None = None):
    ckpt_path = ckpt_path or str(config.CKPT_DIR / "best_model.pth")
    config.RESULTS_DIR.mkdir(exist_ok=True)

    device = config.get_device()
    print(f"\n{'='*55}")
    print(f"  Stage 2 CNN Evaluation")
    print(f"  Checkpoint: {Path(ckpt_path).name}")
    print(f"  Device    : {device}")
    print(f"{'='*55}\n")

    _, test_df = load_splits()
    test_ds    = MammogramDataset(test_df, train=False)
    test_loader= DataLoader(test_ds, batch_size=config.BATCH_SIZE,
                            shuffle=False, num_workers=config.NUM_WORKERS)

    model, ckpt_meta = load_checkpoint(ckpt_path, device)
    print(f"  Loaded checkpoint from epoch {ckpt_meta['epoch']}  "
          f"(val AUC={ckpt_meta['val_auc']:.4f})\n")

    model.eval()
    all_labels, all_probs = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            probs  = torch.sigmoid(model(images)).cpu().numpy()
            all_labels.extend(labels.numpy())
            all_probs.extend(probs)

    all_labels = np.array(all_labels)
    all_probs  = np.array(all_probs)
    all_preds  = (all_probs >= 0.5).astype(int)

    auc  = roc_auc_score(all_labels, all_probs)
    ap   = average_precision_score(all_labels, all_probs)
    report = classification_report(all_labels, all_preds,
                                   target_names=["Benign", "Malignant"],
                                   output_dict=True)
    sens = report["Malignant"]["recall"]
    spec = report["Benign"]["recall"]

    print(f"  AUC-ROC     : {auc:.4f}")
    print(f"  Avg Prec    : {ap:.4f}")
    print(f"  Sensitivity : {sens:.4f}")
    print(f"  Specificity : {spec:.4f}")
    print(classification_report(all_labels, all_preds,
                                target_names=["Benign", "Malignant"]))

    # save metrics CSV
    metrics = pd.DataFrame([{
        "Model": "EfficientNet-B4 (Stage 2 CNN)",
        "AUC-ROC": round(auc, 4), "Avg Precision": round(ap, 4),
        "Sensitivity": round(sens, 4), "Specificity": round(spec, 4),
    }])
    metrics.to_csv(config.RESULTS_DIR / "test_metrics.csv", index=False)

    # ── ROC curve ────────────────────────────────────────────────
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="#e94560", lw=2,
            label=f"EfficientNet-B4 CNN  (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    # Stage 1 baseline for comparison
    ax.plot([0, 0.168, 1], [0, 0.706, 1], color="#0f3460", lw=1.5, ls="--",
            label="Gradient Boosting Stage 1 (AUC=0.868)")
    ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate",
           title="ROC Curve — Stage 2 CNN vs Stage 1 Baseline")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(config.RESULTS_DIR / "roc_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  ROC curve saved → results/roc_curve.png")

    # ── Confusion matrix ─────────────────────────────────────────
    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay(cm, display_labels=["Benign", "Malignant"]).plot(
        ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"EfficientNet-B4 CNN\nAUC={auc:.3f}", fontsize=11)
    plt.tight_layout()
    plt.savefig(config.RESULTS_DIR / "confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Confusion matrix saved → results/confusion_matrix.png")

    # ── GradCAM ──────────────────────────────────────────────────
    print("\n  Generating GradCAM saliency maps...")
    save_gradcam_grid(model, test_ds, device, n=12)

    print(f"\n  All results saved to: {config.RESULTS_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default=None, help="Path to checkpoint .pth file")
    args = parser.parse_args()
    evaluate(args.ckpt)
