"""One-time preprocessing: crop CBIS-DDSM mass lesions and organize by class.

The raw CBIS-DDSM Kaggle mirror is CSV-driven: every image path has to be
resolved through ``dicom_info.csv`` (see the module docstring history in
``dataset.py``), and each lesion's crop location has to be recomputed from
its mask on every load. This script does that work once and writes plain,
predictable folders:

    <out_dir>/{train,test}/{malignant,benign}/<name>.jpg

Only the MLO (mediolateral oblique, side-view) images are used — CC is
skipped, so each lesion contributes at most one crop.

The mask's only job is to locate the crop center (its centroid). It is not
saved anywhere and is never touched again after that: the CNN downstream
only ever sees the cropped mammogram tissue itself, never the mask.

This is a standard ``torchvision.datasets.ImageFolder`` layout.
Run this once before using ``dataset.CBISDDSMDataset``:

    python organize_dataset.py --raw-dir data --out-dir organized
"""

import argparse
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CROP_SIZE = 224
BENIGN_LABELS = {"BENIGN", "BENIGN_WITHOUT_CALLBACK"}
_UNSAFE_CHARS = re.compile(r"[^A-Za-z0-9_-]")


def build_case_lookup(dicom_info: pd.DataFrame) -> dict[tuple[str, str], str]:
    """Map each (case name, image role) pair to its real jpeg path.

    ``dicom_info.csv`` is the reliable source for real image paths: the path
    columns in the case-description CSVs point at ``.dcm`` files that were
    never extracted (this Kaggle mirror only ships ``.jpg`` files), and the
    series-UID folder names embedded in those paths do not reliably match
    ``dicom_info.csv`` either (a known quirk of this mirror). What *does*
    match is ``dicom_info.csv``'s ``PatientID`` column, which holds the same
    human-readable case name used as the first path segment in the case
    CSVs — combined with ``SeriesDescription`` this pair uniquely keys every
    image in the dataset.

    Args:
        dicom_info: parsed ``dicom_info.csv``.

    Returns:
        Mapping ``(case_name, role) -> jpeg_path`` (path as stored in the
        CSV, prefixed ``"CBIS-DDSM/..."``).
    """
    lookup: dict[tuple[str, str], str] = {}
    for _, row in dicom_info.iterrows():
        key = (str(row["PatientID"]), str(row["SeriesDescription"]))
        lookup[key] = str(row["image_path"])
    return lookup


def case_name_from_case_path(case_path: str) -> str:
    """Extract the case name (first path segment) from a case CSV path."""
    return Path(case_path.strip()).parts[0]


def mask_centroid(mask: Image.Image) -> tuple[int, int]:
    """Compute the centroid of the lesion region in an ROI mask.

    Args:
        mask: the ROI mask image, grayscale, bright region marks the lesion.

    Returns:
        ``(x, y)`` pixel coordinates of the lesion centroid.
    """
    mask_array = np.array(mask)
    ys, xs = np.nonzero(mask_array > mask_array.max() / 2)
    if len(xs) == 0:
        height, width = mask_array.shape
        return width // 2, height // 2
    return int(xs.mean()), int(ys.mean())


def crop_around(image: Image.Image, cx: int, cy: int, crop_size: int) -> Image.Image:
    """Crop a fixed-size square out of ``image`` centered on ``(cx, cy)``.

    The crop box is clamped to stay inside the image bounds, so lesions near
    the edge of the mammogram still yield a full-size crop (just off-center
    rather than out-of-bounds).

    Args:
        image: the image to crop (full mammogram or its matching mask).
        cx: centroid x coordinate.
        cy: centroid y coordinate.
        crop_size: side length (pixels) of the square crop.

    Returns:
        A ``(crop_size, crop_size)`` crop.
    """
    half = crop_size // 2
    width, height = image.size
    left = min(max(cx - half, 0), max(width - crop_size, 0))
    top = min(max(cy - half, 0), max(height - crop_size, 0))
    right = min(left + crop_size, width)
    bottom = min(top + crop_size, height)
    return image.crop((left, top, right, bottom))


def resolve_jpeg(raw_dir: Path, case_lookup: dict, case_path: str, role: str) -> Path:
    """Resolve a case-description CSV path to its real jpeg file on disk.

    Args:
        raw_dir: root of the extracted CBIS-DDSM archive (has ``jpeg/``).
        case_lookup: table from :func:`build_case_lookup`.
        case_path: raw path from the case CSV's path column.
        role: ``SeriesDescription`` value to look up.

    Returns:
        Absolute path to the real ``.jpg`` file.
    """
    case_name = case_name_from_case_path(case_path)
    jpeg_path = case_lookup[(case_name, role)]
    # dicom_info.csv paths are prefixed "CBIS-DDSM/..." but this Kaggle
    # mirror extracts "jpeg/" and "csv/" directly under raw_dir.
    relative_path = Path(*Path(jpeg_path).parts[1:])
    return raw_dir / relative_path


def build_sample_name(row: pd.Series) -> str:
    """Build a readable, collision-resistant filename for one lesion row.

    Combines patient id, breast side, view, and abnormality id — the same
    fields CBIS-DDSM itself uses to identify a lesion — so the name stays
    traceable back to the source case.
    """
    raw = f"{row['patient_id']}_{row['left or right breast']}_{row['image view']}_{row['abnormality id']}"
    return _UNSAFE_CHARS.sub("_", raw)


def organize_split(raw_dir: Path, out_dir: Path, split: str, crop_size: int) -> None:
    """Crop and sort every MLO mass lesion in one split into class folders.

    Args:
        raw_dir: root of the extracted CBIS-DDSM archive.
        out_dir: root of the output tree.
        split: ``"train"`` or ``"test"``.
        crop_size: side length (pixels) of the square crop.
    """
    dicom_info = pd.read_csv(raw_dir / "csv" / "dicom_info.csv")
    case_lookup = build_case_lookup(dicom_info)

    cases = pd.read_csv(raw_dir / "csv" / f"mass_case_description_{split}_set.csv")
    cases = cases[cases["pathology"].notna()]
    cases = cases[cases["image view"] == "MLO"].reset_index(drop=True)

    for cls in ("malignant", "benign"):
        (out_dir / split / cls).mkdir(parents=True, exist_ok=True)

    num_written, num_skipped = 0, 0
    for _, row in cases.iterrows():
        cls = "malignant" if row["pathology"] == "MALIGNANT" else "benign"
        name = build_sample_name(row) + ".jpg"

        try:
            full_path = resolve_jpeg(
                raw_dir, case_lookup, row["image file path"], "full mammogram images"
            )
            mask_path = resolve_jpeg(
                raw_dir, case_lookup, row["ROI mask file path"], "ROI mask images"
            )
            full_image = Image.open(full_path).convert("L")
            mask_image = Image.open(mask_path).convert("L")
            if mask_image.size != full_image.size:
                mask_image = mask_image.resize(full_image.size)

            # The mask's only role: locate the crop center. It is discarded
            # immediately after — never saved, never seen again.
            cx, cy = mask_centroid(mask_image)
            image_crop = crop_around(full_image, cx, cy, crop_size)

            image_crop.save(out_dir / split / cls / name)
            num_written += 1
        except Exception:
            logger.exception("Skipping lesion %s (split=%s)", name, split)
            num_skipped += 1

    logger.info(
        "split=%s (MLO only): wrote %d lesions, skipped %d",
        split,
        num_written,
        num_skipped,
    )


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default="data", help="extracted CBIS-DDSM root")
    parser.add_argument("--out-dir", default="organized", help="output root")
    parser.add_argument("--crop-size", type=int, default=CROP_SIZE)
    args = parser.parse_args()

    raw_dir, out_dir = Path(args.raw_dir), Path(args.out_dir)
    for split in ("train", "test"):
        organize_split(raw_dir, out_dir, split, args.crop_size)


if __name__ == "__main__":
    main()
