"""
All hyperparameters and paths for Stage 2 CNN pipeline.
Edit this file to change settings — nothing else needs touching.
"""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).parent.parent
DICOM_ROOT  = ROOT / "manifest-ZkhPvrLo5216730872708713142" / "CBIS-DDSM"
CSV_DIR     = ROOT / "data" / "csv"
CKPT_DIR    = Path(__file__).parent / "checkpoints"
LOG_DIR     = Path(__file__).parent / "logs"
RESULTS_DIR = Path(__file__).parent / "results"

CSV_FILES = {
    "calc_train": CSV_DIR / "calc_case_description_train_set.csv",
    "calc_test":  CSV_DIR / "calc_case_description_test_set.csv",
    "mass_train": CSV_DIR / "mass_case_description_train_set.csv",
    "mass_test":  CSV_DIR / "mass_case_description_test_set.csv",
}

# ── Image settings ─────────────────────────────────────────────────────────
IMAGE_SIZE   = 512          # resize DICOM to 512×512
CHANNELS     = 3            # EfficientNet expects 3-channel input (we repeat grayscale)
CLAHE_CLIP   = 0.03         # adaptive histogram equalisation clip limit
CLAHE_GRID   = (8, 8)       # CLAHE tile grid size

# ── Model ──────────────────────────────────────────────────────────────────
MODEL_NAME   = "efficientnet_b4"
PRETRAINED   = True
DROP_RATE    = 0.3          # dropout before classifier head
NUM_CLASSES  = 1            # binary: sigmoid output

# ── Training ───────────────────────────────────────────────────────────────
BATCH_SIZE   = 8            # safe for 16 GB M3 unified memory at 512px
NUM_WORKERS  = 4
EPOCHS_WARMUP= 5            # freeze backbone, train head only
EPOCHS_FINETUNE = 25        # unfreeze all layers
LR_WARMUP    = 1e-3
LR_FINETUNE  = 1e-4
WEIGHT_DECAY = 1e-4
LABEL_SMOOTH = 0.05
GRAD_CLIP    = 1.0
PATIENCE     = 7            # early stopping patience (fine-tune phase)

# ── Augmentation (training only) ──────────────────────────────────────────
AUG_HFLIP    = 0.5          # horizontal flip probability
AUG_VFLIP    = 0.2          # vertical flip probability
AUG_ROTATE   = 15           # max rotation degrees
AUG_BRIGHTNESS = 0.15
AUG_CONTRAST   = 0.15

# ── Reproducibility ────────────────────────────────────────────────────────
SEED         = 42

# ── Device (auto-detect M3 MPS → CUDA → CPU) ──────────────────────────────
import torch

def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
