"""One-time preprocessing: build a polyp vs. normal crop dataset from
Kvasir-SEG (polyps, masked) and HyperKvasir (normal colon tissue, unmasked).

Mirrors ``organize_dataset.py``'s role for CBIS-DDSM, but for a
structurally cleaner comparison: polyp presence is confirmed by direct
visual annotation (an endoscopist marking what's in the image), not a
separate biopsy that can disagree with appearance — see the project
discussion for why that matters.

Adaptive crop size (fixes a real bug from the first version of this
script): a fixed 224px crop window, in original-image pixels, was too
small for most polyps — measured on the real data, 20% of crops were
*entirely* inside the polyp (no boundary visible at all) and another 46%
had the polyp cut off at the crop edge. Only 33% showed the full polyp
with a clean boundary. Fixed the same way as the mammography pipeline's
analogous bug: size the crop to the lesion's own bounding box (plus a
margin), *then* resize that down to the fixed model input size — so the
full boundary is always visible regardless of how large the polyp is in
the original image.

Two different centering (and now sizing) strategies, since only one class
has a lesion:

- **Polyp** images (Kvasir-SEG): crop box = the mask's bounding box x
  ``BBOX_MARGIN``, centered on the bounding box center, then resized down
  to ``FINAL_SIZE``.
- **Normal** images (HyperKvasir, ``anatomical-landmarks/cecum`` and
  ``retroflex-rectum`` — deliberately lower-GI/colon landmarks, matching
  where the polyps are, not upper-GI, to avoid the model learning "which
  organ" instead of "polyp or not"): crop box size is drawn **from the
  same distribution of box sizes used for the polyp crops** (not a fixed
  size), centered on a random point within visible tissue, then also
  resized down to ``FINAL_SIZE``. Matching the size distribution between
  classes matters: if polyp crops used variable original-image box sizes
  (then resized down by a variable factor) while normal crops always used
  one fixed box size, the resize factor itself — not tissue content —
  could become a shortcut the model learns to exploit. Random *position*
  (not the image center) similarly avoids a positional shortcut, for the
  same reason.

Output layout (standard ``ImageFolder``), all images ``FINAL_SIZE`` x
``FINAL_SIZE``:

    <out_dir>/{train,test}/{normal,polyp}/<name>.jpg

No official train/test split ships with either source here (unlike
CBIS-DDSM), so this script makes its own random split with a fixed seed.
"""

import argparse
import logging
import random
from pathlib import Path

import numpy as np
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

FINAL_SIZE = 224
BBOX_MARGIN = 1.6
MIN_CROP = 64
TEST_FRACTION = 0.2
SEED = 0


def mask_bbox_center_and_size(mask: Image.Image, margin: float) -> tuple[int, int, int]:
    """Compute the square crop box (center + side length) needed to
    contain the mask's foreground region plus a margin.

    Args:
        mask: the polyp mask (bright = polyp, dark = background).
        margin: multiplier applied to the mask's own bounding-box side
            length, so the crop shows some surrounding context, not just
            the lesion pixel-for-pixel.

    Returns:
        ``(cx, cy, box_size)`` — center point and side length of the
        square region to crop, in the mask's own (== source image's)
        pixel coordinates.
    """
    arr = np.array(mask.convert("L"))
    ys, xs = np.nonzero(arr > arr.max() / 2)
    if len(xs) == 0:
        height, width = arr.shape
        return width // 2, height // 2, min(width, height)

    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    box_size = int(max(x1 - x0, y1 - y0) * margin)
    return cx, cy, max(box_size, MIN_CROP)


def random_tissue_point(image: Image.Image, rng: random.Random) -> tuple[int, int]:
    """Pick a random point within the visible (non-black) tissue area.

    Endoscopy images typically have a black vignette border around the
    circular field of view; sampling uniformly over the whole rectangle
    would waste most draws on that border. This restricts sampling to
    pixels bright enough to be real tissue.

    Args:
        image: the normal-tissue image.
        rng: seeded random generator, for reproducibility.

    Returns:
        ``(x, y)`` pixel coordinates.
    """
    gray = np.array(image.convert("L"))
    ys, xs = np.nonzero(gray > 30)
    if len(xs) == 0:
        height, width = gray.shape
        return width // 2, height // 2
    idx = rng.randrange(len(xs))
    return int(xs[idx]), int(ys[idx])


def crop_and_resize(
    image: Image.Image, cx: int, cy: int, box_size: int, final_size: int
) -> Image.Image:
    """Crop a square region centered on ``(cx, cy)``, then resize it to
    ``final_size`` x ``final_size``.

    The crop box is clamped to the image's own dimensions (an oversized
    box, e.g. from a large polyp with margin applied, is shrunk to fit
    rather than going out of bounds), and centering is clamped so the box
    stays fully inside the image — same approach as
    ``organize_dataset.py``'s ``crop_around``, plus the resize step.

    Args:
        image: the source image.
        cx: crop center x coordinate.
        cy: crop center y coordinate.
        box_size: desired side length of the square crop, before resize.
        final_size: output side length after resizing.

    Returns:
        A ``(final_size, final_size)`` image.
    """
    width, height = image.size
    box_size = min(box_size, width, height)
    half = box_size // 2
    left = min(max(cx - half, 0), max(width - box_size, 0))
    top = min(max(cy - half, 0), max(height - box_size, 0))
    right, bottom = left + box_size, top + box_size
    crop = image.crop((left, top, right, bottom))
    return crop.resize((final_size, final_size), Image.BILINEAR)


def split_names(names: list[str], test_fraction: float, seed: int) -> tuple[set, set]:
    """Deterministically split a list of filenames into train/test sets."""
    rng = random.Random(seed)
    shuffled = sorted(names)  # sort first so shuffle is reproducible regardless of OS listing order
    rng.shuffle(shuffled)
    num_test = int(len(shuffled) * test_fraction)
    return set(shuffled[num_test:]), set(shuffled[:num_test])


def organize_polyps(kvasir_seg_dir: Path, out_dir: Path, final_size: int) -> list[int]:
    """Crop every Kvasir-SEG polyp image around its mask bounding box.

    Returns:
        The list of (pre-resize) box sizes used, so
        :func:`organize_normals` can draw from the same distribution.
    """
    img_dir = kvasir_seg_dir / "images"
    mask_dir = kvasir_seg_dir / "masks"
    image_paths = sorted(img_dir.glob("*.jpg"))

    train_names, test_names = split_names([p.name for p in image_paths], TEST_FRACTION, SEED)

    for split in ("train", "test"):
        (out_dir / split / "polyp").mkdir(parents=True, exist_ok=True)

    box_sizes = []
    for img_path in image_paths:
        mask_path = mask_dir / img_path.name
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)
        cx, cy, box_size = mask_bbox_center_and_size(mask, BBOX_MARGIN)
        crop = crop_and_resize(image, cx, cy, box_size, final_size)
        box_sizes.append(box_size)

        split = "train" if img_path.name in train_names else "test"
        crop.save(out_dir / split / "polyp" / img_path.name)

    logger.info(
        "polyp: wrote %d crops (%d train / %d test), box_size mean=%.0f min=%d max=%d",
        len(box_sizes), len(train_names), len(test_names),
        np.mean(box_sizes), min(box_sizes), max(box_sizes),
    )
    return box_sizes


def organize_normals(normal_dir: Path, out_dir: Path, final_size: int, box_sizes: list[int]) -> None:
    """Crop every HyperKvasir normal-tissue image around a random tissue
    point, using a box size drawn from ``box_sizes`` (the polyp crops'
    own size distribution — see module docstring for why that matters).
    """
    image_paths = sorted(normal_dir.glob("*.jpg"))
    train_names, test_names = split_names([p.name for p in image_paths], TEST_FRACTION, SEED)

    for split in ("train", "test"):
        (out_dir / split / "normal").mkdir(parents=True, exist_ok=True)

    rng = random.Random(SEED)
    num_written = 0
    for img_path in image_paths:
        image = Image.open(img_path).convert("RGB")
        cx, cy = random_tissue_point(image, rng)
        box_size = rng.choice(box_sizes)
        crop = crop_and_resize(image, cx, cy, box_size, final_size)

        split = "train" if img_path.name in train_names else "test"
        crop.save(out_dir / split / "normal" / img_path.name)
        num_written += 1

    logger.info("normal: wrote %d crops (%d train / %d test)", num_written, len(train_names), len(test_names))


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kvasir-seg-dir", default="kvasir_data/Kvasir-SEG")
    parser.add_argument("--normal-dir", default="kvasir_data/hyperkvasir_normal")
    parser.add_argument("--out-dir", default="organized_kvasir")
    parser.add_argument("--final-size", type=int, default=FINAL_SIZE)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    box_sizes = organize_polyps(Path(args.kvasir_seg_dir), out_dir, args.final_size)
    organize_normals(Path(args.normal_dir), out_dir, args.final_size, box_sizes)


if __name__ == "__main__":
    main()
