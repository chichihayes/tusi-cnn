"""Metrics and plotting primitives shared by the mammography and polyp
evaluation scripts.

Both ``mammography/evaluate.py`` and ``polyp/evaluate.py`` load a trained
model + its per-epoch history, compute the same four metrics on the
held-out test set, and produce the same two comparison plots — this module
holds that shared logic so it isn't duplicated between the two pipelines.
"""

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import f1_score, recall_score, roc_auc_score, roc_curve
from torch.utils.data import DataLoader

from core.models import create_model

logger = logging.getLogger(__name__)


def load_history(run_dir: Path, branch: str) -> list[dict]:
    """Load a branch's per-epoch training history saved by a train script."""
    with open(run_dir / f"{branch}_history.json", encoding="utf-8") as f:
        return json.load(f)


def load_model(run_dir: Path, branch: str, device: torch.device) -> torch.nn.Module:
    """Load a branch's trained model weights saved by a train script."""
    model = create_model("cnn").to(device)
    state_dict = torch.load(run_dir / f"{branch}_model.pt", map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def collect_predictions(
    model: torch.nn.Module, loader: DataLoader, device: torch.device
) -> tuple[list[float], list[int]]:
    """Run inference over a loader and collect predicted probabilities.

    Args:
        model: a trained model in eval mode.
        loader: batches of ``(image, label)``.
        device: device to run inference on.

    Returns:
        ``(probs, labels)`` — predicted positive-class probability and true
        label for every sample in the loader, in loader order.
    """
    probs, labels = [], []
    with torch.no_grad():
        for images, batch_labels in loader:
            logits = model(images.to(device))
            batch_probs = torch.sigmoid(logits).cpu().tolist()
            probs.extend(batch_probs)
            labels.extend(batch_labels.tolist())
    return probs, labels


def compute_metrics(probs: list[float], labels: list[int]) -> dict:
    """Compute Accuracy, ROC-AUC, Sensitivity (Recall), and F1 for one branch.

    Args:
        probs: predicted positive-class probability per sample.
        labels: true binary label per sample (1 = positive class).

    Returns:
        Dict with keys ``accuracy``, ``roc_auc``, ``sensitivity``, ``f1``.
    """
    preds = [1 if p > 0.5 else 0 for p in probs]
    accuracy = sum(p == l for p, l in zip(preds, labels)) / len(labels)
    return {
        "accuracy": accuracy,
        "roc_auc": roc_auc_score(labels, probs),
        "sensitivity": recall_score(labels, preds),
        "f1": f1_score(labels, preds),
    }


def plot_training_curves(histories: dict[str, list[dict]], out_path: Path) -> None:
    """Plot side-by-side loss and accuracy curves for both branches.

    Args:
        histories: mapping ``branch -> history`` (as returned by
            :func:`load_history`).
        out_path: file to save the figure to.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = {"baseline": "tab:blue", "tusi": "tab:orange"}

    for branch, history in histories.items():
        epochs = [h["epoch"] for h in history]
        color = colors.get(branch, None)
        axes[0].plot(epochs, [h["train_loss"] for h in history], "--", color=color, alpha=0.6)
        axes[0].plot(epochs, [h["test_loss"] for h in history], "-", color=color, label=f"{branch} test")
        axes[1].plot(epochs, [h["train_acc"] for h in history], "--", color=color, alpha=0.6)
        axes[1].plot(epochs, [h["test_acc"] for h in history], "-", color=color, label=f"{branch} test")

    axes[0].set_title("Loss (dashed=train, solid=test)")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("BCE loss")
    axes[0].legend()

    axes[1].set_title("Accuracy (dashed=train, solid=test)")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("accuracy")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    logger.info("saved training curves to %s", out_path)


def plot_roc_curves(predictions: dict[str, tuple[list[float], list[int]]], out_path: Path) -> None:
    """Plot overlaid ROC curves for both branches.

    Args:
        predictions: mapping ``branch -> (probs, labels)`` (as returned by
            :func:`collect_predictions`).
        out_path: file to save the figure to.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    for branch, (probs, labels) in predictions.items():
        fpr, tpr, _ = roc_curve(labels, probs)
        auc = roc_auc_score(labels, probs)
        ax.plot(fpr, tpr, label=f"{branch} (AUC={auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC: baseline vs. Tusi")
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    logger.info("saved ROC curves to %s", out_path)
