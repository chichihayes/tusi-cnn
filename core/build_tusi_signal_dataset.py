"""Materialize the Tusi-transformed version of a class-organized crop dataset.

Both ``mammography/train.py`` and ``polyp/train.py`` apply
:class:`core.tusi_filter.TusiRadialFilter` **live, in memory**, every time a
batch is loaded — the transformed (angle x radius) grid is never written to
disk as part of normal training. That's the right choice for training
(flexible, no wasted disk space, easy to change ``n_angles``/``n_radial``),
but it means there's nothing to literally open and look at as "the exact
signal data the model trained on."

This script closes that gap: it runs the same transform used during
training and saves every result as an image, in the same
``{train,test}/{class_a,class_b}/`` layout as the source crop dataset — so
``<out_dir>/train/polyp/<name>.png`` is pixel-for-pixel what the model saw
for that crop during training (modulo the resize+normalize steps, which are
just scaling and don't change the content).

Run from the repo root, e.g. for the polyp dataset:

    python core/build_tusi_signal_dataset.py --root-dir polyp/dataset --out-dir polyp/dataset_signal
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.tusi_filter import TusiRadialFilter

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

IMAGE_SIZE = 224


def build_split(
    root_dir: Path, out_dir: Path, split: str, radial_filter: TusiRadialFilter
) -> None:
    """Transform every crop in one split and save the result.

    Args:
        root_dir: root of the source crop dataset (has ``{split}/{class}/``).
        out_dir: root of the output tree (same layout, transformed images).
        split: ``"train"`` or ``"test"``.
        radial_filter: the (shared) transform to apply.
    """
    resize_to_gray_tensor = transforms.Compose(
        [transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.Grayscale(num_output_channels=1), transforms.ToTensor()]
    )

    split_dir = root_dir / split
    class_dirs = sorted(p for p in split_dir.iterdir() if p.is_dir())

    num_written = 0
    for class_dir in class_dirs:
        out_class_dir = out_dir / split / class_dir.name
        out_class_dir.mkdir(parents=True, exist_ok=True)

        for img_path in sorted(class_dir.glob("*.jpg")):
            image = Image.open(img_path).convert("L")
            tensor = resize_to_gray_tensor(image).unsqueeze(0)  # (1, 1, H, W)
            with torch.no_grad():
                transformed = radial_filter(tensor).squeeze(0).squeeze(0)  # (n_angles, n_radial)

            arr = transformed.numpy()
            arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
            out_img = Image.fromarray((arr * 255).astype(np.uint8), mode="L")
            out_img.save(out_class_dir / (img_path.stem + ".png"))
            num_written += 1

    logger.info("split=%s: wrote %d transformed images to %s", split, num_written, out_dir / split)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-dir", required=True, help="source crop dataset, e.g. polyp/dataset")
    parser.add_argument("--out-dir", required=True, help="output root for transformed images")
    parser.add_argument("--n-angles", type=int, default=48)
    parser.add_argument("--n-radial", type=int, default=112)
    args = parser.parse_args()

    radial_filter = TusiRadialFilter(n_angles=args.n_angles, n_radial=args.n_radial)
    root_dir, out_dir = Path(args.root_dir), Path(args.out_dir)
    for split in ("train", "test"):
        build_split(root_dir, out_dir, split, radial_filter)


if __name__ == "__main__":
    main()
