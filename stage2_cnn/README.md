# Stage 2 — CNN on Full DICOM Mammograms

EfficientNet-B4 fine-tuned on all **10,239 DICOM mammogram images** (152 GB)
from the CBIS-DDSM dataset. Optimised for Apple M3 (MPS backend).

---

## Files

| File | Purpose |
|---|---|
| `config.py` | All hyperparameters and paths — edit here first |
| `dataset.py` | DICOM loader, preprocessing, Dataset class, balanced sampler |
| `model.py` | EfficientNet-B4 with custom head, freeze/unfreeze helpers |
| `train.py` | Two-phase training loop (warmup → fine-tune) with early stopping |
| `evaluate.py` | Full test-set evaluation + GradCAM saliency maps |
| `predict.py` | Single-image inference (used by Streamlit app) |

---

## Training Strategy

```
Phase 1 — Warmup (5 epochs)
  • Backbone frozen (ImageNet weights preserved)
  • Only new classifier head (1792→512→1) trains
  • LR = 1e-3

Phase 2 — Fine-tune (up to 25 epochs, early stopping patience=7)
  • All layers unfrozen
  • LR = 1e-4 with cosine annealing schedule
  • Label smoothing (ε=0.05) to prevent overconfidence
  • Gradient clipping (max norm=1.0)
```

Balanced sampler ensures equal malignant/benign representation per batch.

---

## How to Run

```bash
cd /Users/alihamza/Desktop/AICD/stage2_cnn

# Step 1 — Train (takes several hours on M3 Air)
python train.py

# Step 2 — Evaluate on test set + generate GradCAM maps
python evaluate.py

# Step 3 — Predict a single image
python predict.py /path/to/mammogram.dcm
python predict.py /path/to/image.png
```

---

## Expected Outputs

```
stage2_cnn/
├── checkpoints/
│   └── best_model.pth          ← best val-AUC checkpoint
├── logs/
│   └── train_log.csv           ← per-epoch loss + AUC
└── results/
    ├── training_curves.png     ← loss + AUC over epochs
    ├── roc_curve.png           ← ROC vs Stage 1 baseline
    ├── confusion_matrix.png
    ├── test_metrics.csv
    └── gradcam/
        └── gradcam_grid.png    ← saliency overlays on 12 samples
```

---

## Hardware & Estimated Training Time

| Hardware | Estimated Time |
|---|---|
| MacBook Air M3 (8-core GPU, 16 GB) | ~6–10 hours |
| MacBook Pro M3 Max (40-core GPU) | ~2–3 hours |
| NVIDIA A100 | ~30 minutes |

M3 uses `torch.device("mps")` automatically — no code changes needed.

---

## Key Hyperparameters (config.py)

| Parameter | Value | Notes |
|---|---|---|
| IMAGE_SIZE | 512 | Higher = better but slower |
| BATCH_SIZE | 8 | Safe for 16 GB unified memory |
| EPOCHS_WARMUP | 5 | Freeze backbone |
| EPOCHS_FINETUNE | 25 | With early stopping |
| LR_WARMUP | 1e-3 | Head only |
| LR_FINETUNE | 1e-4 | All layers |
| PATIENCE | 7 | Early stopping epochs |

To train faster on M3 Air: set `IMAGE_SIZE = 256`, `BATCH_SIZE = 16`.
