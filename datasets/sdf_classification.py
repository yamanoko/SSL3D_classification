import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .base_datamodule import BaseDataModule
from .blosc2io import Blosc2IO


class SDFClassificationData(Dataset):
    """
    FDSLxSDF4Seg の generate_sdf_dataset_classification.py で生成された
    SDF分類データセットを読み込むDataset。

    データ構造:
        <root>/
        ├── nnUNetResEncUNetLPlans_3d_fullres/
        │   ├── case_00000.b2nd
        │   ├── case_00001.b2nd
        │   └── ...
        ├── labelsTr.json
        └── splits_final.json
    """

    def __init__(self, root, split, fold, transform=None):
        super().__init__()

        self.img_dir = Path(root) / "nnUNetResEncUNetLPlans_3d_fullres"
        label_file = Path(root) / "labelsTr.json"
        split_file = Path(root) / "splits_final.json"

        with open(split_file) as f:
            splits = json.load(f)

        # fold は int または str で指定される
        fold_idx = int(fold) if fold is not None else 0
        self.img_files = splits[fold_idx]["train" if split == "train" else "val"]

        with open(label_file) as f:
            labels = json.load(f)
        # ラベルは直接整数値 (例: {"case_00000": 3, "case_00001": 0, ...})
        self.labels = [labels[i] for i in self.img_files]

        self.transform = transform

    def __getitem__(self, idx):
        img, _ = Blosc2IO.load(
            self.img_dir / (self.img_files[idx] + ".b2nd"), mode="r"
        )

        if self.transform:
            img = self.transform(**{"image": torch.from_numpy(img[...])})["image"]
        else:
            img = torch.from_numpy(img[...])

        return img, self.labels[idx]

    def __len__(self):
        return len(self.img_files)


class SDFClassificationDataModule(BaseDataModule):
    """
    SDF分類データセット用の LightningDataModule。
    BaseDataModule を継承し、train/val の Dataset を setup で構築する。
    """

    def __init__(self, **params):
        super(SDFClassificationDataModule, self).__init__(**params)

    def setup(self, stage: str):
        self.train_dataset = SDFClassificationData(
            self.data_path,
            split="train",
            transform=self.train_transforms,
            fold=self.fold,
        )
        self.val_dataset = SDFClassificationData(
            self.data_path,
            split="val",
            transform=self.test_transforms,
            fold=self.fold,
        )
