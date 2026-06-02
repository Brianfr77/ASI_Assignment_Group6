from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import datasets

from .utils import safe_torch_load


@dataclass(frozen=True)
class MNISTInfo:
    n_train: int
    n_test: int
    input_dim: int = 784
    image_shape: tuple[int, int, int] = (1, 28, 28)


def _load_raw_mnist(data_dir: Path, train: bool) -> tuple[torch.Tensor, torch.Tensor]:
    ds = datasets.MNIST(
        root=str(data_dir),
        train=train,
        download=True,
    )
    images = ds.data.unsqueeze(1).to(torch.float32) / 255.0
    labels = ds.targets.to(torch.long)
    return images, labels


def _cache_path(data_dir: Path, split: str, binarization: str, seed: int) -> Path:
    name = f"mnist_{split}_{binarization}_seed{seed}.pt"
    return data_dir / "processed" / name


def _save_static_binarized(
    data_dir: Path,
    split: str,
    images: torch.Tensor,
    labels: torch.Tensor,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    path = _cache_path(data_dir, split, "static_bernoulli", seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    gen = torch.Generator(device="cpu").manual_seed(seed + (0 if split == "train" else 10_000))
    binary = (torch.rand(images.shape, generator=gen) < images).to(torch.float32)
    torch.save({"images": binary, "labels": labels}, path)
    return binary, labels


def load_mnist_tensors(
    data_dir: str | Path,
    split: str,
    binarization: str = "static_bernoulli",
    binarization_seed: int = 2026,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load MNIST as flattened tensors with paper-oriented binarization."""
    data_dir = Path(data_dir)
    train = split == "train"
    if split not in {"train", "test"}:
        raise ValueError(f"split must be 'train' or 'test', got {split!r}")

    if binarization == "static_bernoulli":
        path = _cache_path(data_dir, split, binarization, binarization_seed)
        if path.exists():
            payload = safe_torch_load(path, map_location="cpu")
            images, labels = payload["images"], payload["labels"]
        else:
            raw_images, labels = _load_raw_mnist(data_dir, train=train)
            images, labels = _save_static_binarized(
                data_dir, split, raw_images, labels, binarization_seed
            )
    elif binarization == "threshold_0_5":
        raw_images, labels = _load_raw_mnist(data_dir, train=train)
        images = (raw_images >= 0.5).to(torch.float32)
    elif binarization == "grayscale_0_1":
        images, labels = _load_raw_mnist(data_dir, train=train)
    else:
        raise ValueError(f"Unsupported binarization: {binarization}")

    expected_n = 60_000 if train else 10_000
    if images.shape != (expected_n, 1, 28, 28):
        raise ValueError(f"Unexpected MNIST shape for {split}: {tuple(images.shape)}")
    if labels.shape != (expected_n,):
        raise ValueError(f"Unexpected label shape for {split}: {tuple(labels.shape)}")
    if float(images.min()) < 0.0 or float(images.max()) > 1.0:
        raise ValueError("MNIST pixels must be in [0, 1].")
    if binarization in {"static_bernoulli", "threshold_0_5"}:
        unique = torch.unique(images)
        if not torch.all((unique == 0.0) | (unique == 1.0)):
            raise ValueError(f"Binarized MNIST contains non-binary values: {unique[:8]}")

    return images.view(expected_n, -1).contiguous(), labels


def make_mnist_loaders(
    data_dir: str | Path,
    batch_size: int,
    binarization: str,
    binarization_seed: int,
    seed: int,
    train_subset_size: int | None = None,
    test_subset_size: int | None = None,
    num_workers: int = 2,
) -> tuple[DataLoader, DataLoader, MNISTInfo]:
    train_x, train_y = load_mnist_tensors(data_dir, "train", binarization, binarization_seed)
    test_x, test_y = load_mnist_tensors(data_dir, "test", binarization, binarization_seed)

    train_ds = TensorDataset(train_x, train_y)
    test_ds = TensorDataset(test_x, test_y)
    if train_subset_size is not None:
        train_ds = Subset(train_ds, list(range(int(train_subset_size))))
    if test_subset_size is not None:
        test_ds = Subset(test_ds, list(range(int(test_subset_size))))

    gen = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        generator=gen,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, test_loader, MNISTInfo(n_train=len(train_ds), n_test=len(test_ds))
