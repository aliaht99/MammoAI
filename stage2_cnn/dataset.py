"""
CBIS-DDSM DICOM Dataset Loader.

Discovers full-mammogram DICOM files on disk by matching
patient_id + breast side + view from the CSV annotations,
then applies preprocessing and augmentation.
"""

import random
from pathlib import Path
from functools import lru_cache

import numpy as np
import pandas as pd
import pydicom
import torch
from torch.utils.data import Dataset, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
from skimage import exposure

import config


# ── Label encoding ─────────────────────────────────────────────────────────
TARGET_MAP = {
    "MALIGNANT":               1,
    "BENIGN":                  0,
    "BENIGN_WITHOUT_CALLBACK": 0,
}


# ── CSV loading ────────────────────────────────────────────────────────────
def _load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, skipinitialspace=True)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    for col in df.select_dtypes("object"):
        df[col] = df[col].str.strip()
    return df


def load_splits() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (train_df, test_df) with label and dcm_path columns."""
    frames_train, frames_test = [], []

    for key, path in config.CSV_FILES.items():
        df = _load_csv(path)
        df["source"] = "calc" if "calc" in key else "mass"
        if "train" in key:
            frames_train.append(df)
        else:
            frames_test.append(df)

    train_df = pd.concat(frames_train, ignore_index=True)
    test_df  = pd.concat(frames_test,  ignore_index=True)

    for df in (train_df, test_df):
        df["label"] = df["pathology"].map(TARGET_MAP)
        df.dropna(subset=["label"], inplace=True)
        df["label"] = df["label"].astype(int)
        df["dcm_path"] = df.apply(_find_dcm, axis=1)

    # drop rows where we couldn't find the DICOM
    train_df = train_df.dropna(subset=["dcm_path"]).reset_index(drop=True)
    test_df  = test_df.dropna(subset=["dcm_path"]).reset_index(drop=True)

    print(f"Train: {len(train_df)} images  | "
          f"Malignant: {train_df['label'].sum()} | "
          f"Benign: {(train_df['label']==0).sum()}")
    print(f"Test:  {len(test_df)} images   | "
          f"Malignant: {test_df['label'].sum()} | "
          f"Benign: {(test_df['label']==0).sum()}")
    return train_df, test_df


def _find_dcm(row: pd.Series) -> str | None:
    """
    Locate the full-mammogram DICOM for a CSV row.

    The CSV image_file_path column starts with e.g.
      'Calc-Training_P_00005_RIGHT_CC/...'
    The actual folder on disk is:
      DICOM_ROOT / 'Calc-Training_P_00005_RIGHT_CC' / <date-folder> / <desc-folder> / 1-1.dcm

    We extract the top-level case folder name from the CSV path,
    then glob for the first full-mammogram DCM inside it.
    """
    img_col = "image_file_path"
    if img_col not in row.index or pd.isna(row[img_col]):
        return None

    # top-level case folder (first path component)
    case_folder = str(row[img_col]).split("/")[0].strip()
    case_dir = config.DICOM_ROOT / case_folder

    if not case_dir.exists():
        return None

    # find the full mammogram DICOM (not ROI mask)
    hits = list(case_dir.rglob("*full mammogram*/*.dcm"))
    if not hits:
        # fall back: any .dcm that isn't a mask
        hits = [p for p in case_dir.rglob("*.dcm")
                if "ROI mask" not in str(p)]
    if not hits:
        return None

    return str(hits[0])


# ── DICOM → numpy ──────────────────────────────────────────────────────────
@lru_cache(maxsize=256)
def _read_dicom(path: str) -> np.ndarray:
    """Read DICOM, return uint8 numpy array (H, W)."""
    ds  = pydicom.dcmread(path)
    arr = ds.pixel_array.astype(float)
    # window/level normalisation
    arr = (arr - arr.min()) / (arr.ptp() + 1e-8)
    # CLAHE contrast enhancement
    arr = exposure.equalize_adapthist(arr,
                                      clip_limit=config.CLAHE_CLIP,
                                      kernel_size=config.CLAHE_GRID)
    return (arr * 255).astype(np.uint8)


def _to_pil(arr: np.ndarray) -> Image.Image:
    """Convert grayscale uint8 (H,W) to 3-channel PIL image."""
    pil = Image.fromarray(arr, mode="L").convert("RGB")
    return pil


# ── Transforms ─────────────────────────────────────────────────────────────
def _build_transforms(train: bool) -> transforms.Compose:
    resize = transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE),
                                interpolation=transforms.InterpolationMode.BILINEAR)
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],   # ImageNet stats (model pre-trained on these)
        std=[0.229, 0.224, 0.225],
    )
    to_tensor = transforms.ToTensor()

    if train:
        return transforms.Compose([
            resize,
            transforms.RandomHorizontalFlip(p=config.AUG_HFLIP),
            transforms.RandomVerticalFlip(p=config.AUG_VFLIP),
            transforms.RandomRotation(degrees=config.AUG_ROTATE),
            transforms.ColorJitter(
                brightness=config.AUG_BRIGHTNESS,
                contrast=config.AUG_CONTRAST,
            ),
            to_tensor,
            normalize,
        ])
    else:
        return transforms.Compose([resize, to_tensor, normalize])


# ── Dataset ────────────────────────────────────────────────────────────────
class MammogramDataset(Dataset):
    def __init__(self, df: pd.DataFrame, train: bool = True):
        self.df        = df.reset_index(drop=True)
        self.transform = _build_transforms(train)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row   = self.df.iloc[idx]
        arr   = _read_dicom(row["dcm_path"])
        pil   = _to_pil(arr)
        image = self.transform(pil)
        label = torch.tensor(row["label"], dtype=torch.float32)
        return image, label

    def get_labels(self) -> list[int]:
        return self.df["label"].tolist()


def make_balanced_sampler(dataset: MammogramDataset) -> WeightedRandomSampler:
    """Over-sample the minority class so each batch is ~balanced."""
    labels  = dataset.get_labels()
    counts  = np.bincount(labels)
    weights = 1.0 / counts[labels]
    return WeightedRandomSampler(weights=weights,
                                 num_samples=len(weights),
                                 replacement=True)
