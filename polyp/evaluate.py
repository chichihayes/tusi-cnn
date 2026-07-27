"""Polyp evaluation — metrics and plots, baseline vs. Tusi.

Loads the trained model weights and per-epoch history that
``polyp/train.py`` saves to ``polyp/results/``, computes held-out
test-set metrics for each branch, and produces side-by-side
training-curve and ROC-curve plots.

Run from the repo root:

    python polyp/evaluate.py
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.evaluation import (
    collect_predictions,
    compute_metrics,
    load_history,
    load_model,
    plot_roc_curves,
    plot_training_curves,
)
from core.training import BRANCHES, get_device
from polyp.train import DEFAULT_OUT_DIR, DEFAULT_ROOT_DIR, build_datasets

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    """CLI entry point: evaluate both branches and produce comparison plots."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-dir", default=str(DEFAULT_ROOT_DIR))
    parser.add_argument("--run-dir", default=str(DEFAULT_OUT_DIR))
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

    print("\n=== Polyp detection test set metrics ===")
    for branch, m in metrics.items():
        print(f"{branch}: accuracy={m['accuracy']:.3f} roc_auc={m['roc_auc']:.3f} "
              f"sensitivity={m['sensitivity']:.3f} f1={m['f1']:.3f}")


if __name__ == "__main__":
    main()
