"""
Single-image inference — used by the Streamlit app for Stage 2 prediction.

Usage (CLI):
    python predict.py path/to/mammogram.dcm
    python predict.py path/to/image.png
"""

import sys
from pathlib import Path

import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from skimage import exposure
import pydicom

sys.path.insert(0, str(Path(__file__).parent))
import config
from model import load_checkpoint

_TRANSFORM = transforms.Compose([
    transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

_model_cache: dict = {}


def _load_model(ckpt_path: str | None = None) -> tuple:
    key = ckpt_path or "default"
    if key not in _model_cache:
        device = config.get_device()
        path   = ckpt_path or str(config.CKPT_DIR / "best_model.pth")
        model, meta = load_checkpoint(path, device)
        model.eval()
        _model_cache[key] = (model, device, meta)
    return _model_cache[key]


def load_image_array(file_path: str) -> np.ndarray:
    """Load DICOM or PNG/JPG → uint8 numpy array (H, W)."""
    path = str(file_path).lower()
    if path.endswith(".dcm"):
        ds  = pydicom.dcmread(file_path)
        arr = ds.pixel_array.astype(float)
        arr = (arr - arr.min()) / (arr.ptp() + 1e-8)
        arr = exposure.equalize_adapthist(arr,
                                          clip_limit=config.CLAHE_CLIP,
                                          kernel_size=config.CLAHE_GRID)
        return (arr * 255).astype(np.uint8)
    else:
        return np.array(Image.open(file_path).convert("L"))


def predict_image(file_path: str, ckpt_path: str | None = None) -> dict:
    """
    Returns:
        probability_malignant : float 0–1
        prediction            : "MALIGNANT" or "BENIGN"
        risk_level            : "LOW" | "MEDIUM" | "HIGH" | "VERY HIGH"
        model_epoch           : int (which epoch the checkpoint is from)
    """
    model, device, meta = _load_model(ckpt_path)

    arr = load_image_array(file_path)
    pil = Image.fromarray(arr, mode="L").convert("RGB")
    tensor = _TRANSFORM(pil).unsqueeze(0).to(device)

    with torch.no_grad():
        prob = torch.sigmoid(model(tensor)).item()

    label      = "MALIGNANT" if prob >= 0.5 else "BENIGN"
    risk_level = ("LOW"       if prob < 0.30 else
                  "MEDIUM"    if prob < 0.60 else
                  "HIGH"      if prob < 0.80 else
                  "VERY HIGH")

    return {
        "probability_malignant": round(prob, 4),
        "prediction":            label,
        "risk_level":            risk_level,
        "model_epoch":           meta.get("epoch", "?"),
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <path_to_image.dcm|png|jpg>")
        sys.exit(1)

    result = predict_image(sys.argv[1])
    print(f"\n  File            : {sys.argv[1]}")
    print(f"  Probability     : {result['probability_malignant']:.1%}")
    print(f"  Prediction      : {result['prediction']}")
    print(f"  Risk Level      : {result['risk_level']}")
    print(f"  Model Epoch     : {result['model_epoch']}")
