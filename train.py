"""Training loop, shared between the baseline and Tusi-filtered branches.

Both branches run through the exact same :func:`run_training` function —
only the ``branch`` argument differs, which selects the input transform
(:func:`dataset.default_transform` vs. :func:`dataset.tusi_transform`) and,
correspondingly, the expected input shape. Same optimizer, learning rate,
epoch count, loss function, batch size, and (via seeding) the same initial
model weights for both runs, so any difference in outcome traces back to
the input representation, not incidental training differences.
"""

import argparse
import json
import logging
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset import CBISDDSMDataset, default_transform, tusi_transform
from models import create_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BRANCHES = ("baseline", "tusi")


def get_device() -> torch.device:
    """Return the CUDA device if available, else CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_datasets(branch: str, root_dir: str) -> tuple[CBISDDSMDataset, CBISDDSMDataset]:
    """Build the train/test datasets for one branch.

    Args:
        branch: ``"baseline"`` or ``"tusi"`` — selects the input transform.
        root_dir: root of the class-organized dataset (see
            ``organize_dataset.py``).

    Returns:
        ``(train_dataset, test_dataset)``, sharing CBIS-DDSM's own split.
    """
    transform = default_transform() if branch == "baseline" else tusi_transform()
    train_ds = CBISDDSMDataset(root_dir=root_dir, split="train", transform=transform)
    test_ds = CBISDDSMDataset(root_dir=root_dir, split="test", transform=transform)
    return train_ds, test_ds


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float]:
    """Run one pass over ``loader``, training if ``optimizer`` is given.

    Args:
        model: the model to run.
        loader: batches of ``(image, label)``.
        loss_fn: loss function (expects raw logits and float labels).
        device: device to run on.
        optimizer: if given, the model is put in train mode and weights are
            updated; if ``None``, the model is evaluated in eval mode with
            no gradient updates.

    Returns:
        ``(average_loss, accuracy)`` over the whole loader.
    """
    model.train(mode=optimizer is not None)
    total_loss, num_correct, num_samples = 0.0, 0, 0

    context = torch.enable_grad() if optimizer is not None else torch.no_grad()
    with context:
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device).float()

            logits = model(images)
            loss = loss_fn(logits, labels)

            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            preds = (torch.sigmoid(logits) > 0.5).float()
            num_correct += (preds == labels).sum().item()
            num_samples += batch_size

    return total_loss / num_samples, num_correct / num_samples


def run_training(
    branch: str,
    root_dir: str = "organized",
    epochs: int = 20,
    batch_size: int = 16,
    lr: float = 1e-3,
    seed: int = 0,
) -> tuple[nn.Module, list[dict]]:
    """Train one branch (baseline or Tusi) end to end.

    Args:
        branch: ``"baseline"`` or ``"tusi"``.
        root_dir: root of the class-organized dataset.
        epochs: number of training epochs.
        batch_size: batch size for both train and test loaders.
        lr: Adam learning rate.
        seed: RNG seed, applied before model construction so both branches'
            models start from identical initial weights (same architecture,
            same shapes — only the input differs downstream).

    Returns:
        ``(trained_model, history)`` where ``history`` is a list of one
        dict per epoch: ``epoch, train_loss, train_acc, test_loss, test_acc``.
    """
    if branch not in BRANCHES:
        raise ValueError(f"branch must be one of {BRANCHES}, got {branch!r}")

    device = get_device()
    logger.info("branch=%s device=%s", branch, device)

    train_ds, test_ds = build_datasets(branch, root_dir)
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
            branch,
            epoch,
            epochs,
            train_loss,
            train_acc,
            test_loss,
            test_acc,
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "test_loss": test_loss,
                "test_acc": test_acc,
            }
        )

    return model, history


def main() -> None:
    """CLI entry point: train both branches with identical hyperparameters."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-dir", default="organized")
    parser.add_argument("--out-dir", default="runs")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    for branch in BRANCHES:
        model, history = run_training(
            branch,
            root_dir=args.root_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=args.seed,
        )
        torch.save(model.state_dict(), out_dir / f"{branch}_model.pt")
        with open(out_dir / f"{branch}_history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
        logger.info("saved %s model + history to %s", branch, out_dir)


if __name__ == "__main__":
    main()
