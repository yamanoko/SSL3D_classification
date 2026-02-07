import numpy as np
import torch
from torch.utils.data import Dataset

from medmnist import INFO
import medmnist

from .base_datamodule import BaseDataModule


# Supported 3D classification datasets (binary-class and multi-class only)
MEDMNIST_3D_CLASSIFICATION_FLAGS = [
    "organmnist3d",     # multi-class, 11 classes
    "nodulemnist3d",    # binary-class, 2 classes
    "adrenalmnist3d",   # binary-class, 2 classes
    "fracturemnist3d",  # multi-class, 3 classes
    "vesselmnist3d",    # binary-class, 2 classes
    "synapsemnist3d",   # binary-class, 2 classes
]


def get_num_classes(dataset_flag: str) -> int:
    """Get the number of classes for a given MedMNIST 3D dataset flag."""
    info = INFO[dataset_flag]
    task = info["task"]
    if task == "multi-class":
        return len(info["label"])
    elif task == "binary-class":
        return 2
    else:
        raise ValueError(
            f"Unsupported task type '{task}' for dataset '{dataset_flag}'. "
            f"Only 'multi-class' and 'binary-class' are supported."
        )


class MedMNIST3DData(Dataset):
    """
    A wrapper Dataset for MedMNIST 3D datasets that is compatible with the
    existing training pipeline using batchgeneratorsv2 transforms.

    MedMNIST 3D datasets provide volumetric data of shape (N, D, H, W) stored
    in .npz files with predefined train/val/test splits.

    The data is returned as (image, label) where:
        - image: torch.Tensor of shape (1, D, H, W), float32, normalized to [0, 1]
        - label: int class index (scalar)
    """

    def __init__(
        self,
        dataset_flag: str,
        split: str,
        data_root_dir: str,
        size: int = 64,
        download: bool = True,
        transform=None,
    ):
        """
        Args:
            dataset_flag: MedMNIST dataset flag (e.g. "organmnist3d").
            split: One of "train", "val", "test".
            data_root_dir: Root directory where the .npz files are stored.
            size: Resolution of the 3D volumes (28 or 64).
            download: Whether to download the dataset if not found.
            transform: Optional transform (batchgeneratorsv2 compatible).
        """
        super().__init__()

        if dataset_flag not in MEDMNIST_3D_CLASSIFICATION_FLAGS:
            raise ValueError(
                f"Unsupported dataset_flag '{dataset_flag}'. "
                f"Supported flags: {MEDMNIST_3D_CLASSIFICATION_FLAGS}"
            )

        if size not in (28, 64):
            raise ValueError(
                f"Unsupported size '{size}' for MedMNIST 3D. Must be 28 or 64."
            )

        self.dataset_flag = dataset_flag
        self.split = split
        self.size = size
        self.transform = transform
        self.info = INFO[dataset_flag]

        # Use medmnist's built-in dataset class to handle download & loading
        dataset_class = getattr(medmnist, self.info["python_class"])
        medmnist_dataset = dataset_class(
            split=split,
            root=data_root_dir,
            download=download,
            size=size,
            as_rgb=False,
        )

        # Extract raw arrays from the medmnist dataset
        # imgs: (N, D, H, W) uint8, labels: (N, 1) or (N, L)
        self.imgs = medmnist_dataset.imgs
        self.labels = medmnist_dataset.labels

        print(
            f"[MedMNIST3D] Loaded {dataset_flag} split={split} size={size}: "
            f"{len(self.imgs)} samples, task={self.info['task']}, "
            f"n_channels={self.info['n_channels']}"
        )

    def __getitem__(self, idx):
        # img: (D, H, W) uint8 -> (1, D, H, W) float32 in [0, 1]
        img = self.imgs[idx].astype(np.float32) / 255.0
        img = img[np.newaxis, ...]  # Add channel dim: (1, D, H, W)

        # label: (1,) or (L,) -> scalar int for classification
        label = self.labels[idx].astype(np.int64)
        if label.ndim > 0 and label.shape[0] == 1:
            label = label[0]

        if self.transform:
            img = self.transform(**{"image": torch.from_numpy(img)})["image"]
        else:
            img = torch.from_numpy(img)

        return img, label

    def __len__(self):
        return len(self.imgs)


class MedMNIST3DDataModule(BaseDataModule):
    """
    LightningDataModule for MedMNIST 3D classification datasets.

    Uses the predefined train/val/test splits provided by MedMNIST
    (no cross-validation). The `fold` parameter from BaseDataModule is
    accepted but ignored since MedMNIST provides fixed splits.

    Usage (via Hydra config):
        data=organmnist3d
        data.module.data_root_dir=/path/to/medmnist
    """

    def __init__(
        self,
        dataset_flag: str = "organmnist3d",
        size: int = 64,
        download: bool = True,
        **params,
    ):
        super().__init__(**params)
        self.dataset_flag = dataset_flag
        self.size = size
        self.download = download

    def setup(self, stage: str):
        self.train_dataset = MedMNIST3DData(
            dataset_flag=self.dataset_flag,
            split="train",
            data_root_dir=str(self.data_path),
            size=self.size,
            download=self.download,
            transform=self.train_transforms,
        )
        self.val_dataset = MedMNIST3DData(
            dataset_flag=self.dataset_flag,
            split="val",
            data_root_dir=str(self.data_path),
            size=self.size,
            download=self.download,
            transform=self.test_transforms,
        )
        self.test_dataset = MedMNIST3DData(
            dataset_flag=self.dataset_flag,
            split="test",
            data_root_dir=str(self.data_path),
            size=self.size,
            download=self.download,
            transform=self.test_transforms,
        )
