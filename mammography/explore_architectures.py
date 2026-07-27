"""Exploratory: how well can a Tusi-tailored architecture do, on its own?

This is deliberately outside the controlled baseline-vs-Tusi comparison in
``mammography/train.py`` / ``mammography/evaluate.py`` — it only runs the
Tusi branch, with a model (``tusi_aware_cnn`` or ``tusi_signal_cnn``, see
``core/models.py``) and training recipe (circular angle-axis augmentation,
weight decay, more epochs) that only make sense for Tusi input. It answers
a different question than the main experiment: not "does Tusi beat raw
pixels under identical conditions," but "what's the upside if we lean into
Tusi's structure specifically." Results here are not directly comparable
to ``mammography/results/metrics.json`` as a fair A/B test — they're a
ceiling estimate for the Tusi representation. Results are written to
``mammography/results/exploratory/`` to keep them clearly separate from
the fair comparison's results.

Run from the repo root:

    python mammography/explore_architectures.py --architecture tusi_signal_cnn
"""

import argparse
import json
import logging
import random
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.dataset import CBISDDSMDataset, tusi_transform
from core.evaluation import collect_predictions, compute_metrics
from core.models import create_model
from core.training import get_device, run_epoch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

THIS_DIR = Path(__file__).resolve().parent
DEFAULT_ROOT_DIR = THIS_DIR / "dataset"
DEFAULT_OUT_DIR = THIS_DIR / "results" / "exploratory"


class AngleRollAugment(Dataset):
    """Wrap a Tusi dataset with random circular shifts along the angle axis.

    A circular roll of the angle axis is equivalent to re-starting the
    Tusi sweep from a different angle — a label-preserving augmentation
    that only makes sense because that axis is circular (see
    :class:`core.models.TusiAwareCNN`'s docstring). This has no baseline
    equivalent, which is exactly why this script is kept separate from the
    controlled comparison.
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


def main() -> None:
    """CLI entry point: train and evaluate the Tusi-aware exploratory model."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--architecture",
        default="tusi_aware_cnn",
        choices=["tusi_aware_cnn", "tusi_signal_cnn"],
    )
    parser.add_argument("--root-dir", default=str(DEFAULT_ROOT_DIR))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-angle-shift", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    device = get_device()
    logger.info("device=%s", device)

    train_ds = CBISDDSMDataset(root_dir=args.root_dir, split="train", transform=tusi_transform())
    train_ds = AngleRollAugment(train_ds, max_shift=args.max_angle_shift)
    test_ds = CBISDDSMDataset(root_dir=args.root_dir, split="test", transform=tusi_transform())

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    model = create_model(args.architecture).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    history = []
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, loss_fn, device, optimizer)
        test_loss, test_acc = run_epoch(model, test_loader, loss_fn, device, optimizer=None)
        logger.info(
            "epoch %d/%d train_loss=%.4f train_acc=%.4f test_loss=%.4f test_acc=%.4f",
            epoch, args.epochs, train_loss, train_acc, test_loss, test_acc,
        )
        history.append({
            "epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
            "test_loss": test_loss, "test_acc": test_acc,
        })

    probs, labels = collect_predictions(model, test_loader, device)
    metrics = compute_metrics(probs, labels)
    logger.info("%s test metrics: %s", args.architecture, metrics)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / f"{args.architecture}_model.pt")
    with open(out_dir / f"{args.architecture}_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    with open(out_dir / f"{args.architecture}_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n=== {args.architecture} test metrics ===")
    print(f"accuracy={metrics['accuracy']:.3f} roc_auc={metrics['roc_auc']:.3f} "
          f"sensitivity={metrics['sensitivity']:.3f} f1={metrics['f1']:.3f}")
    print("\n(compare against mammography/results/metrics.json's 'tusi' entry — "
          "same data, plain shared-backbone CNN, no augmentation)")


if __name__ == "__main__":
    main()
