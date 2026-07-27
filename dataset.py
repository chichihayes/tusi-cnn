"""Dataset loading and preprocessing for the Tusi-Couple mammography experiment.

Provides a synthetic dataset (dense circular lumps with/without radiating
spicules) for fast pipeline iteration, and a real dataset built from the
CBIS-DDSM mass subset. Both return ``(image_tensor, label)`` pairs with
label 1 = malignant, 0 = benign, so ``train.py`` can consume either without
caring which is in use.

The real dataset expects the class-organized layout produced by
``organize_dataset.py`` (run once beforehand):

    <root_dir>/{train,test}/{benign,malignant}/<name>.jpg

Cropping each lesion around its mask centroid and sorting it by class is
done once by that script rather than on every load — see its module
docstring for why (CSV joins + full-mammogram loads are expensive to repeat
every epoch).
"""

import logging
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets, transforms

from tusi_filter import TusiRadialFilter

logger = logging.getLogger(__name__)

IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

Transform = Callable[[Image.Image], torch.Tensor]


def default_transform(image_size: int = IMAGE_SIZE) -> transforms.Compose:
    """Build the standard resize + normalize transform shared by all datasets.

    This is the baseline branch's transform: the model sees the raw cropped
    tissue, unmodified beyond resizing and intensity normalization.

    Args:
        image_size: side length (pixels) the image is resized to.

    Returns:
        A ``torchvision.transforms.Compose`` producing a normalized
        ``(3, image_size, image_size)`` tensor from a grayscale PIL image.
    """
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def tusi_transform(
    image_size: int = IMAGE_SIZE, n_angles: int = 48, n_radial: int = 112
) -> Transform:
    """Build the Tusi branch's transform: resize, then radial-sweep filter.

    Mirrors :func:`default_transform` in every way except the extra
    :class:`~tusi_filter.TusiRadialFilter` step — same resize target, same
    3-channel replication and normalization range, so the only difference
    between the baseline and Tusi branches is this one step, not incidental
    preprocessing differences.

    Args:
        image_size: side length (pixels) the image is resized to before
            filtering.
        n_angles: number of diameter angles the filter sweeps (output rows).
        n_radial: number of harmonic samples per diameter (output columns).

    Returns:
        A callable producing a normalized ``(3, n_angles, n_radial)`` tensor
        from a grayscale PIL image.
    """
    resize_to_gray_tensor = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
        ]
    )
    radial_filter = TusiRadialFilter(n_angles=n_angles, n_radial=n_radial)
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    def apply(image: Image.Image) -> torch.Tensor:
        x = resize_to_gray_tensor(image)  # (1, image_size, image_size)
        x = radial_filter(x)  # (1, n_angles, n_radial)
        x = x.repeat(3, 1, 1)  # match baseline's 3-channel input shape
        return normalize(x)

    return apply


class SyntheticLumpDataset(Dataset):
    """Synthetic mammography-patch generator for fast pipeline iteration.

    Generates grayscale image patches containing a dense circular "lump".
    Malignant samples additionally have thin radiating spicules extending
    from the lump edge outward — the exact radial structure the Tusi filter
    is designed to exploit — while benign samples are a smooth circle only.
    This gives ground-truth control over the signal, unlike real data where
    the presence/strength of spiculation varies case to case.
    """

    def __init__(
        self,
        num_samples: int = 500,
        image_size: int = IMAGE_SIZE,
        seed: int = 0,
        transform: Transform | None = None,
    ) -> None:
        """Create the synthetic dataset.

        Args:
            num_samples: total number of samples (split evenly malignant/benign).
            image_size: side length (pixels) of generated images.
            seed: RNG seed for reproducibility.
            transform: transform applied to the generated PIL image; defaults
                to :func:`default_transform`.
        """
        self.num_samples = num_samples
        self.image_size = image_size
        self.transform = transform or default_transform(image_size)
        self._rng = np.random.default_rng(seed)
        self._labels = [i % 2 for i in range(num_samples)]
        logger.info(
            "SyntheticLumpDataset: %d samples (%d malignant, %d benign)",
            num_samples,
            sum(self._labels),
            num_samples - sum(self._labels),
        )

    def __len__(self) -> int:
        """Return the number of samples."""
        return self.num_samples

    def _draw_lump(self, label: int) -> np.ndarray:
        """Render one synthetic lump patch as a float32 array in [0, 1].

        Args:
            label: 1 to draw a spiculated (malignant-pattern) lump, 0 for a
                smooth circular (benign-pattern) lump.

        Returns:
            ``(image_size, image_size)`` array.
        """
        size = self.image_size
        center = size / 2
        yy, xx = np.mgrid[0:size, 0:size]
        dist = np.sqrt((xx - center) ** 2 + (yy - center) ** 2)

        radius = self._rng.uniform(size * 0.12, size * 0.2)
        img = np.full((size, size), 0.15, dtype=np.float32)
        img += 0.05 * self._rng.standard_normal((size, size)).astype(np.float32)
        img[dist <= radius] = 0.85

        if label == 1:
            num_spicules = self._rng.integers(6, 14)
            angles = self._rng.uniform(0, 2 * np.pi, num_spicules)
            length = self._rng.uniform(size * 0.15, size * 0.3, num_spicules)
            theta = np.arctan2(yy - center, xx - center)
            for angle, spike_len in zip(angles, length):
                angular_dist = np.abs(np.angle(np.exp(1j * (theta - angle))))
                spike_mask = (
                    (angular_dist < 0.05)
                    & (dist > radius)
                    & (dist < radius + spike_len)
                )
                img[spike_mask] = 0.8

        img = np.clip(img, 0.0, 1.0)
        return img

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """Return the ``(image_tensor, label)`` pair at ``idx``."""
        label = self._labels[idx]
        array = self._draw_lump(label)
        image = Image.fromarray((array * 255).astype(np.uint8), mode="L")
        image_tensor = self.transform(image)
        return image_tensor, label


class CBISDDSMDataset(Dataset):
    """Real mammography dataset built from the CBIS-DDSM mass subset.

    Thin wrapper around a class-organized folder of pre-cropped lesion
    images (``<root_dir>/<split>/{benign,malignant}/*.jpg``), produced once
    by ``organize_dataset.py``. That script does the expensive part (CSV
    joins, loading full mammograms, cropping around each lesion's mask
    centroid) up front, so this class just needs to read images off disk —
    the same shape of work as any standard ``ImageFolder`` dataset.

    ``ImageFolder`` assigns class indices alphabetically, so ``benign`` = 0
    and ``malignant`` = 1 — matching this project's label convention without
    any extra remapping.
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        transform: Transform | None = None,
    ) -> None:
        """Load the class-organized CBIS-DDSM split.

        Args:
            root_dir: root of the organized tree produced by
                ``organize_dataset.py`` (contains ``train/`` and ``test/``).
            split: ``"train"`` or ``"test"`` — uses CBIS-DDSM's own split,
                not a re-shuffled one, so the held-out test set stays clean.
            transform: transform applied to each loaded image; defaults to
                :func:`default_transform`.
        """
        if split not in ("train", "test"):
            raise ValueError(f"split must be 'train' or 'test', got {split!r}")

        self.transform = transform or default_transform()
        split_dir = Path(root_dir) / split
        self._image_folder = datasets.ImageFolder(root=str(split_dir))

        assert self._image_folder.class_to_idx == {"benign": 0, "malignant": 1}, (
            f"unexpected class ordering: {self._image_folder.class_to_idx}"
        )

        labels = [label for _, label in self._image_folder.samples]
        logger.info(
            "CBISDDSMDataset[%s]: %d lesions (%d malignant, %d benign)",
            split,
            len(labels),
            sum(labels),
            len(labels) - sum(labels),
        )

    def __len__(self) -> int:
        """Return the number of lesions in this split."""
        return len(self._image_folder)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """Return the ``(image_tensor, label)`` pair for lesion ``idx``."""
        image, label = self._image_folder[idx]
        return self.transform(image), label
