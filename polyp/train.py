"""Polyp training loop — baseline vs. Tusi-filtered (Kvasir-SEG + HyperKvasir).

Structurally identical to ``mammography/train.py``'s baseline-vs-Tusi
comparison — same model factory, same ``run_epoch``, same optimizer/loss —
just pointed at ``polyp/dataset/`` instead. Kept as a separate script
rather than sharing one train.py because the dataset loading differs
(plain ``ImageFolder`` here, no CSV-driven centroid lookup needed since
``organize_dataset.py`` already did that once).

``ImageFolder`` assigns classes alphabetically: ``normal`` = 0, ``polyp``
= 1, matching the "1 = the finding of interest" convention used for
malignant=1 in the mammography experiment.

Run from the repo root:

    python polyp/train.py --epochs 20
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.dataset import default_transform, tusi_transform
from core.models import create_model
from core.training import BRANCHES, get_device, run_epoch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

THIS_DIR = Path(__file__).resolve().parent
DEFAULT_ROOT_DIR = THIS_DIR / "dataset"
DEFAULT_OUT_DIR = THIS_DIR / "results"


def build_datasets(branch: str, root_dir: str) -> tuple[datasets.ImageFolder, datasets.ImageFolder]:
    """Build the train/test ``ImageFolder`` datasets for one branch.

    Args:
        branch: ``"baseline"`` or ``"tusi"`` — selects the input transform.
        root_dir: root of the organized dataset (see
            ``organize_dataset.py``).

    Returns:
        ``(train_dataset, test_dataset)``.
    """
    transform = default_transform() if branch == "baseline" else tusi_transform()
    train_ds = datasets.ImageFolder(root=str(Path(root_dir) / "train"), transform=transform)
    test_ds = datasets.ImageFolder(root=str(Path(root_dir) / "test"), transform=transform)
    assert train_ds.class_to_idx == {"normal": 0, "polyp": 1}, train_ds.class_to_idx
    return train_ds, test_ds


def run_training(
    branch: str,
    root_dir: str = str(DEFAULT_ROOT_DIR),
    epochs: int = 20,
    batch_size: int = 16,
    lr: float = 1e-3,
    seed: int = 0,
) -> tuple[nn.Module, list[dict]]:
    """Train one branch (baseline or Tusi) end to end on the polyp data.

    Mirrors ``mammography.train.run_training`` exactly, just with this
    module's ``build_datasets``. See that function for parameter details.
    """
    if branch not in BRANCHES:
        raise ValueError(f"branch must be one of {BRANCHES}, got {branch!r}")

    device = get_device()
    logger.info("branch=%s device=%s", branch, device)

    train_ds, test_ds = build_datasets(branch, root_dir)
    logger.info("train=%d test=%d", len(train_ds), len(test_ds))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    torch.manual_seed(seed)
    model = create_model("cnn").to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    history = []
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, loss_fn, device, optimizer)
        test_loss, test_acc = run_epoch(model, test_loader, loss_fn, device, optimizer=None)

        logger.info(
            "[%s] epoch %d/%d train_loss=%.4f train_acc=%.4f test_loss=%.4f test_acc=%.4f",
            branch, epoch, epochs, train_loss, train_acc, test_loss, test_acc,
        )
        history.append({
            "epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
            "test_loss": test_loss, "test_acc": test_acc,
        })

    return model, history


def main() -> None:
    """CLI entry point: train one or both branches with identical hyperparameters."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-dir", default=str(DEFAULT_ROOT_DIR))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--branch", choices=BRANCHES, default=None,
        help="train only this branch (default: both, sequentially)",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    branches = [args.branch] if args.branch else list(BRANCHES)
    for branch in branches:
        model, history = run_training(
            branch, root_dir=args.root_dir, epochs=args.epochs,
            batch_size=args.batch_size, lr=args.lr, seed=args.seed,
        )
        torch.save(model.state_dict(), out_dir / f"{branch}_model.pt")
        with open(out_dir / f"{branch}_history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
        logger.info("saved %s model + history to %s", branch, out_dir)


if __name__ == "__main__":
    main()
