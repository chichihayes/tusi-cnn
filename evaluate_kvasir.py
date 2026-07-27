"""Metrics and plots for the polyp vs. normal comparison.

Mirrors ``evaluate.py`` exactly, pointed at ``runs_kvasir/`` and
``organized_kvasir/`` instead of the mammography artifacts — reuses all
the same metric/plot functions rather than redefining them.
"""

import argparse
import json
import logging
from pathlib import Path

from torch.utils.data import DataLoader

from evaluate import (
    collect_predictions,
    compute_metrics,
    load_history,
    load_model,
    plot_roc_curves,
    plot_training_curves,
)
from train import BRANCHES, get_device
from train_kvasir import build_datasets

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    """CLI entry point: evaluate both branches and produce comparison plots."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-dir", default="organized_kvasir")
    parser.add_argument("--run-dir", default="runs_kvasir")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    device = get_device()

    histories, predictions, metrics = {}, {}, {}
    for branch in BRANCHES:
        histories[branch] = load_history(run_dir, branch)

        _, test_ds = build_datasets(branch, args.root_dir)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

        model = load_model(run_dir, branch, device)
        probs, labels = collect_predictions(model, test_loader, device)
        predictions[branch] = (probs, labels)
        metrics[branch] = compute_metrics(probs, labels)
        logger.info("[%s] test metrics: %s", branch, metrics[branch])

    with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    plot_training_curves(histories, run_dir / "training_curves.png")
    plot_roc_curves(predictions, run_dir / "roc_curves.png")

    print("\n=== Test set metrics (polyp vs. normal) ===")
    for branch, m in metrics.items():
        print(f"{branch}: accuracy={m['accuracy']:.3f} roc_auc={m['roc_auc']:.3f} "
              f"sensitivity={m['sensitivity']:.3f} f1={m['f1']:.3f}")


if __name__ == "__main__":
    main()
