"""Training-loop primitives shared by every pipeline (mammography, polyp,
and the exploratory architecture experiments).

Both the mammography and polyp pipelines run through the same
``run_epoch`` for one training/eval pass — only the model, data, and which
transform produced the input tensor differ between a "baseline" and a
"tusi" run. Keeping this shared is what keeps the two branches' training
process identical apart from the input representation.
"""

import logging

import torch
from torch import nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

BRANCHES = ("baseline", "tusi")


def get_device() -> torch.device:
    """Return the CUDA device if available, else CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
