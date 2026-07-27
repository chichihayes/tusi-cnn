"""Shared augmentation wrappers for the exploratory Tusi-only architecture
experiments (used by both ``mammography/explore_architectures.py`` and
``polyp/explore_architectures.py``).
"""

import random

import torch
from torch.utils.data import Dataset


class AngleRollAugment(Dataset):
    """Wrap a Tusi dataset with random circular shifts along the angle axis.

    A circular roll of the angle axis is equivalent to re-starting the
    Tusi sweep from a different angle — a label-preserving augmentation
    that only makes sense because that axis is circular (see
    :class:`core.models.TusiAwareCNN`'s docstring). This has no baseline
    equivalent, which is exactly why it's only used in the exploratory
    scripts, not the controlled comparison.
    """

    def __init__(self, base_dataset: Dataset, max_shift: int) -> None:
        """Wrap ``base_dataset``.

        Args:
            base_dataset: a dataset yielding ``(C, n_angles, n_radial)``
                Tusi tensors and labels.
            max_shift: rolls are drawn uniformly from
                ``[-max_shift, max_shift]`` angle rows.
        """
        self.base = base_dataset
        self.max_shift = max_shift

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        image, label = self.base[idx]
        shift = random.randint(-self.max_shift, self.max_shift)
        image = torch.roll(image, shifts=shift, dims=1)  # dim 1 = angle axis
        return image, label
