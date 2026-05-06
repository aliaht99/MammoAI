"""
CBIS-DDSM Dataset Loader — reads preprocessed PNGs from png_cache/.
Run preprocess.py once before training to build the cache.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
from pathlib import Path

import config

PNG_DIR = Path(__file__).parent / "png_cache"

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


def _find_png(row: pd.Series) -> str | None:
    """Match CSV row to its preprocessed PNG in png_cache/."""
    img_col = "image_file_path"
    if img_col not in row.index or pd.isna(row[img_col]):
        return None
    case_folder = str(row[img_col]).split("/")[0].strip()
    png_path = PNG_DIR / (case_folder + ".png")
    return str(png_path) if png_path.exists() else None


def load_splits() -> tuple[pd.DataFrame, pd.DataFrame]:
    frames_train, frames_test = [], []
    for key, path in config.CSV_FILES.items():
        df = _load_csv(path)
        df["source"] = "calc" if "calc" in key else "mass"
        (frames_train if "train" in key else frames_test).append(df)

    train_df = pd.concat(frames_train, ignore_index=True)
    test_df  = pd.concat(frames_test,  ignore_index=True)

    for df in (train_df, test_df):
        df["label"]    = df["pathology"].map(TARGET_MAP)
        df.dropna(subset=["label"], inplace=True)
        df["label"]    = df["label"].astype(int)
        df["png_path"] = df.apply(_find_png, axis=1)

    train_df = train_df.dropna(subset=["png_path"]).reset_index(drop=True)
    test_df  = test_df.dropna(subset=["png_path"]).reset_index(drop=True)

    print(f"Train: {len(train_df)} | Malignant: {train_df['label'].sum()} "
          f"| Benign: {(train_df['label']==0).sum()}")
    print(f"Test : {len(test_df)}  | Malignant: {test_df['label'].sum()} "
          f"| Benign: {(test_df['label']==0).sum()}")

    if len(train_df) == 0:
        raise RuntimeError(
            "No PNGs found in png_cache/. Run: python preprocess.py first."
        )
    return train_df, test_df


# ── Transforms ─────────────────────────────────────────────────────────────
def _build_transforms(train: bool) -> transforms.Compose:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    if train:
        return transforms.Compose([
            transforms.RandomHorizontalFlip(p=config.AUG_HFLIP),
            transforms.RandomVerticalFlip(p=config.AUG_VFLIP),
            transforms.RandomRotation(degrees=config.AUG_ROTATE),
            transforms.ColorJitter(brightness=config.AUG_BRIGHTNESS,
                                   contrast=config.AUG_CONTRAST),
            transforms.ToTensor(),
            normalize,
        ])
    return transforms.Compose([transforms.ToTensor(), normalize])


# ── Dataset ────────────────────────────────────────────────────────────────
class MammogramDataset(Dataset):
    def __init__(self, df: pd.DataFrame, train: bool = True):
        self.df        = df.reset_index(drop=True)
        self.transform = _build_transforms(train)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row   = self.df.iloc[idx]
        # PNGs are grayscale — convert to 3-channel for EfficientNet
        pil   = Image.open(row["png_path"]).convert("RGB")
        image = self.transform(pil)
        label = torch.tensor(row["label"], dtype=torch.float32)
        return image, label

    def get_labels(self) -> list[int]:
        return self.df["label"].tolist()


def make_balanced_sampler(dataset: MammogramDataset) -> WeightedRandomSampler:
    labels  = dataset.get_labels()
    counts  = np.bincount(labels)
    weights = 1.0 / counts[labels]
    return WeightedRandomSampler(weights=weights,
                                 num_samples=len(weights),
                                 replacement=True)
