"""
MRNet Dataset Preprocessing Script

Converts raw MRNet data (Stanford ML Group) into the format expected by
the MRNetData dataloader:
  - Resized and normalized 3D volumes saved as Blosc2 (.b2nd) files
  - labelsTr.json: multilabel dict  { case_id: [abnormal, acl, meniscus] }
  - splits_final.json: cross-validation folds

Raw MRNet directory layout (see README_MRNet.md):
    MRNet-v1.0/
    ├── train/{axial,coronal,sagittal}/XXXX.npy
    ├── valid/{axial,coronal,sagittal}/XXXX.npy
    ├── train-abnormal.csv, train-acl.csv, train-meniscus.csv
    └── valid-abnormal.csv, valid-acl.csv, valid-meniscus.csv

Output layout (consumed by datasets/mrnet.py):
    <output_dir>/
    ├── nnUNetResEncUNetLPlans_3d_fullres/
    │   ├── <case_id>.b2nd
    │   └── ...
    ├── labelsTr.json
    └── splits_final.json

Usage:
    python -m datasets.preprocess_3D_data.datasets.mrnet_preprocessing \
        --raw_dir  /path/to/MRNet-v1.0 \
        --out_dir  /path/to/preprocessed \
        --view     sagittal \
        --patch_size 32 256 256 \
        --n_splits 5 \
        --num_workers 4
"""

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

import argparse
import json
from pathlib import Path
from typing import List, Tuple
from multiprocessing import Pool

import numpy as np
from skimage.transform import resize

from datasets.preprocess_3D_data.blosc_helper import save_case, comp_blosc2_params
from datasets.preprocess_3D_data.normalization import ZScoreNormalization
from datasets.preprocess_3D_data.cross_validation import generate_crossval_split


# ---------------------------------------------------------------------------
# Label helpers
# ---------------------------------------------------------------------------

def load_labels_csv(csv_path: str) -> dict:
    """
    Read an MRNet label CSV (no header, columns: case_id, label).
    Returns {case_id_str: int_label}.
    """
    labels = {}
    with open(csv_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            case_id = parts[0].strip()
            label = int(parts[1].strip())
            labels[case_id] = label
    return labels


def build_multilabel_dict(
    raw_dir: Path,
    splits: List[str] = ("train", "valid"),
    tasks: List[str] = ("abnormal", "acl", "meniscus"),
) -> dict:
    """
    Build { "XXXX": [abnormal, acl, meniscus], ... } for all cases across
    the given splits.
    """
    multilabel: dict = {}

    for split in splits:
        task_labels = []
        for task in tasks:
            csv_path = raw_dir / f"{split}-{task}.csv"
            task_labels.append(load_labels_csv(str(csv_path)))

        # Collect case IDs from the first task (all tasks share the same IDs)
        case_ids = sorted(task_labels[0].keys())
        for cid in case_ids:
            multilabel[cid] = [tl[cid] for tl in task_labels]

    return multilabel


# ---------------------------------------------------------------------------
# Per-case processing
# ---------------------------------------------------------------------------

def resize_volume(volume: np.ndarray, target_shape: Tuple[int, int, int]) -> np.ndarray:
    """
    Resize a 3-D volume to *target_shape* using 3rd-order interpolation.
    Input:  (D, H, W)   – original volume
    Output: (D', H', W') – resized to target_shape
    """
    resized = resize(
        volume.astype(np.float32),
        target_shape,
        order=3,
        mode='constant',
        cval=0,
        anti_aliasing=True,
        preserve_range=True,
    )
    return resized.astype(np.float32)


def process_single_case(args):
    """
    Worker function for parallel preprocessing.

    Parameters (packed as a tuple for Pool.map):
        npy_path   : Path to the raw .npy file.
        case_id    : String identifier (e.g. "0001").
        patch_size : Target (D, H, W).
        output_dir : Directory for .b2nd output.
    """
    npy_path, case_id, patch_size, output_dir = args

    # 1. Load raw volume  (slices, H, W) — dtype typically uint8/uint16/float
    volume = np.load(npy_path).astype(np.float32)  # (S, H, W)

    # 2. Resize to target patch_size
    volume = resize_volume(volume, tuple(patch_size))  # (D', H', W')

    # 3. Add channel dimension → (1, D, H, W)
    volume = volume[np.newaxis, ...]

    # 4. Z-score normalisation
    normalizer = ZScoreNormalization()
    volume = normalizer.run(volume)

    # 5. Compute Blosc2 chunking parameters
    block_size, chunk_size = comp_blosc2_params(
        image_size=volume.shape,
        patch_size=patch_size,
        bytes_per_pixel=volume.dtype.itemsize,
    )

    # 6. Save as .b2nd
    out_path = os.path.join(output_dir, case_id)
    save_case(volume, out_path, chunks=chunk_size, blocks=block_size)

    return case_id


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def preprocess_mrnet(
    raw_dir: str,
    out_dir: str,
    view: str = "sagittal",
    patch_size: Tuple[int, int, int] = (32, 256, 256),
    n_splits: int = 5,
    num_workers: int = 1,
) -> None:
    """
    End-to-end preprocessing of the MRNet dataset.

    Parameters:
        raw_dir    : Path to the raw MRNet-v1.0 directory.
        out_dir    : Path where preprocessed outputs will be saved.
        view       : MRI view to use — one of 'sagittal', 'coronal', 'axial'.
        patch_size : Target volume size (D, H, W).
        n_splits   : Number of cross-validation folds.
        num_workers: Parallel workers for image processing.
    """
    raw_dir = Path(raw_dir)
    out_dir = Path(out_dir)
    img_out_dir = out_dir / "nnUNetResEncUNetLPlans_3d_fullres"
    img_out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[MRNet preprocess] Raw dir  : {raw_dir}")
    print(f"[MRNet preprocess] Out dir  : {out_dir}")
    print(f"[MRNet preprocess] View     : {view}")
    print(f"[MRNet preprocess] Patch    : {patch_size}")
    print(f"[MRNet preprocess] Folds    : {n_splits}")
    print(f"[MRNet preprocess] Workers  : {num_workers}")
    print()

    # ------------------------------------------------------------------
    # 1.  Build multilabel dict and save labelsTr.json
    # ------------------------------------------------------------------
    print("[1/4] Building multilabel dictionary …")
    multilabel = build_multilabel_dict(raw_dir)
    label_path = out_dir / "labelsTr.json"
    with open(label_path, "w") as f:
        json.dump(multilabel, f, indent=2)
    print(f"  → Saved {len(multilabel)} labels to {label_path}")

    # ------------------------------------------------------------------
    # 2.  Collect .npy file paths for the chosen view
    # ------------------------------------------------------------------
    print("[2/4] Collecting .npy file paths …")
    work_items = []
    for split in ("train", "valid"):
        view_dir = raw_dir / split / view
        if not view_dir.exists():
            raise FileNotFoundError(f"View directory not found: {view_dir}")
        for npy_file in sorted(view_dir.glob("*.npy")):
            case_id = npy_file.stem  # e.g. "0001"
            work_items.append((str(npy_file), case_id, patch_size, str(img_out_dir)))

    print(f"  → Found {len(work_items)} volumes")

    # ------------------------------------------------------------------
    # 3.  Process all cases (resize → normalize → save as .b2nd)
    # ------------------------------------------------------------------
    print("[3/4] Processing volumes …")
    if num_workers <= 1:
        processed = []
        for i, item in enumerate(work_items):
            cid = process_single_case(item)
            processed.append(cid)
            if (i + 1) % 50 == 0 or (i + 1) == len(work_items):
                print(f"  → {i + 1}/{len(work_items)} done")
    else:
        with Pool(processes=num_workers) as pool:
            processed = pool.map(process_single_case, work_items)
        print(f"  → {len(processed)} volumes processed")

    # ------------------------------------------------------------------
    # 4.  Generate cross-validation splits and save splits_final.json
    #     Only training cases are used for the CV split; validation cases
    #     from the original dataset can optionally be included.
    # ------------------------------------------------------------------
    print("[4/4] Generating cross-validation splits …")

    # Separate train and valid case ids
    train_ids = sorted([
        npy.stem for npy in (raw_dir / "train" / view).glob("*.npy")
    ])
    valid_ids = sorted([
        npy.stem for npy in (raw_dir / "valid" / view).glob("*.npy")
    ])

    # Use ALL cases (train + valid) for cross-validation splits, so that
    # every case appears in some fold.  Alternatively, only train_ids can
    # be used if the original valid set should stay untouched.
    all_ids = train_ids + valid_ids
    splits = generate_crossval_split(all_ids, n_splits=n_splits)

    splits_path = out_dir / "splits_final.json"
    with open(splits_path, "w") as f:
        json.dump(splits, f, indent=2)
    print(f"  → Saved {n_splits} folds ({len(all_ids)} cases) to {splits_path}")

    print()
    print("=== Preprocessing complete ===")
    print(f"  Images : {img_out_dir}")
    print(f"  Labels : {label_path}")
    print(f"  Splits : {splits_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess raw MRNet dataset for the MRNetData dataloader."
    )
    parser.add_argument(
        "--raw_dir", type=str, required=True,
        help="Path to raw MRNet-v1.0 directory.",
    )
    parser.add_argument(
        "--out_dir", type=str, required=True,
        help="Output directory for preprocessed data.",
    )
    parser.add_argument(
        "--view", type=str, default="sagittal",
        choices=["sagittal", "coronal", "axial"],
        help="MRI view to preprocess (default: sagittal).",
    )
    parser.add_argument(
        "--patch_size", type=int, nargs=3, default=[32, 256, 256],
        help="Target volume size (D H W). Default: 32 256 256.",
    )
    parser.add_argument(
        "--n_splits", type=int, default=5,
        help="Number of cross-validation folds (default: 5).",
    )
    parser.add_argument(
        "--num_workers", type=int, default=4,
        help="Number of parallel workers (default: 4).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    preprocess_mrnet(
        raw_dir=args.raw_dir,
        out_dir=args.out_dir,
        view=args.view,
        patch_size=tuple(args.patch_size),
        n_splits=args.n_splits,
        num_workers=args.num_workers,
    )
