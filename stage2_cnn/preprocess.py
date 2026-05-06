"""
One-time preprocessing: convert all DICOM mammograms → PNG at 512×512.
Run this ONCE before training. After that, training reads tiny PNGs
instead of 20-42 MB DICOM files — 50× faster per epoch.

Usage:
    cd stage2_cnn
    python preprocess.py

Output: stage2_cnn/png_cache/<case_id>.png  (~150-300 KB each vs 25 MB DICOM)
Estimated time: 15-30 min for all 3,103 mammograms on M3
"""

import sys
from pathlib import Path
from multiprocessing import Pool, cpu_count
import numpy as np
import pydicom
from PIL import Image
from skimage import exposure
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
import config

PNG_DIR = Path(__file__).parent / "png_cache"
PNG_DIR.mkdir(exist_ok=True)


def find_all_mammograms() -> list[Path]:
    """Find every full-mammogram DICOM on disk."""
    hits = list(config.DICOM_ROOT.rglob("*full mammogram*/*.dcm"))
    print(f"Found {len(hits)} full mammogram DICOMs")
    return hits


def convert_one(dcm_path: Path) -> str | None:
    """Read DICOM → CLAHE → resize → save PNG. Returns output path or None on error."""
    out_path = PNG_DIR / (dcm_path.parent.parent.parent.name + ".png")

    if out_path.exists():
        return str(out_path)   # already done — skip

    try:
        ds  = pydicom.dcmread(str(dcm_path))
        arr = ds.pixel_array.astype(float)
        arr = (arr - arr.min()) / (arr.ptp() + 1e-8)
        arr = exposure.equalize_adapthist(arr,
                                          clip_limit=config.CLAHE_CLIP,
                                          kernel_size=config.CLAHE_GRID)
        arr = (arr * 255).astype(np.uint8)
        pil = Image.fromarray(arr, mode="L").resize(
            (config.IMAGE_SIZE, config.IMAGE_SIZE),
            Image.LANCZOS,
        )
        pil.save(str(out_path), format="PNG", optimize=False)
        return str(out_path)
    except Exception as e:
        print(f"  ERROR {dcm_path.name}: {e}")
        return None


def main():
    dcm_paths = find_all_mammograms()

    print(f"\nConverting {len(dcm_paths)} DICOMs → {config.IMAGE_SIZE}×{config.IMAGE_SIZE} PNGs")
    print(f"Output dir : {PNG_DIR}")
    print(f"Workers    : {min(4, cpu_count())}")
    print("(Already-converted files are skipped automatically)\n")

    workers = min(4, cpu_count())
    with Pool(processes=workers) as pool:
        results = list(tqdm(
            pool.imap(convert_one, dcm_paths),
            total=len(dcm_paths),
            desc="Converting",
            unit="img",
        ))

    ok  = sum(1 for r in results if r is not None)
    err = sum(1 for r in results if r is None)
    print(f"\nDone: {ok} converted, {err} errors")
    print(f"PNG cache: {PNG_DIR}")

    # show disk usage
    import shutil
    total_mb = sum(p.stat().st_size for p in PNG_DIR.glob("*.png")) / 1e6
    print(f"Cache size: {total_mb:.0f} MB  (was {len(dcm_paths)*25:.0f} MB raw DICOM)")


if __name__ == "__main__":
    main()
