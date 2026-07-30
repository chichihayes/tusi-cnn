"""5-fold stratified cross-validation for the polyp pipeline, baseline vs. Tusi.

``polyp/train.py`` + ``polyp/evaluate.py`` report metrics on one fixed
train/test split — useful for headline numbers, but a single split gives no
sense of variance, and there's no way to say whether baseline vs. Tusi's
metric gap is real or noise. This script instead:

1. Pools every image from ``polyp/dataset/{train,test}`` (2400 total) into
   one set, and re-splits it into 5 stratified folds (``StratifiedKFold``,
   so each fold keeps the ~58/42 normal/polyp ratio).
2. For each fold, trains both branches from scratch on the other 4 folds and
   evaluates on the held-out fold — identical training procedure
   (``polyp.train.train_on_datasets``) and hyperparameters as the main
   experiment, just re-split.
3. Because both branches are evaluated on the *same* held-out samples in
   every fold, every sample ends up with exactly one out-of-fold prediction
   per branch. Pooling those across all 5 folds gives a full-dataset
   out-of-fold prediction set per branch, which is what
   :func:`core.delong.delong_roc_test` compares — the proper paired
   significance test for two AUCs measured on shared samples (see that
   module's docstring for why a naive test would be invalid here).

Run from the repo root:

    python polyp/cross_validate.py --epochs 20
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
from torchvision import datasets

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.dataset import PathListDataset, default_transform, tusi_transform
from core.delong import delong_roc_test
from core.evaluation import collect_predictions, compute_metrics, plot_cv_fold_aucs
from core.training import BRANCHES
from polyp.train import DEFAULT_ROOT_DIR, train_on_datasets

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "results" / "cross_validation"


def pool_samples(root_dir: str) -> list[tuple[str, int]]:
    """Pool ``dataset/train`` and ``dataset/test`` into one sample list.

    Args:
        root_dir: root of the organized dataset (contains ``train/`` and
            ``test/``, each with ``normal/`` and ``polyp/`` subfolders).

    Returns:
        List of ``(image_path, label)`` pairs from both splits combined.
    """
    samples = []
    for split in ("train", "test"):
        folder = datasets.ImageFolder(root=str(Path(root_dir) / split))
        assert folder.class_to_idx == {"normal": 0, "polyp": 1}, folder.class_to_idx
        samples.extend(folder.samples)
    return samples


def run_cross_validation(
    root_dir: str = str(DEFAULT_ROOT_DIR),
    n_folds: int = 5,
    epochs: int = 20,
    batch_size: int = 16,
    lr: float = 1e-3,
    seed: int = 0,
) -> dict:
    """Run stratified k-fold CV for both branches and assemble the report.

    Args:
        root_dir: root of the organized polyp dataset.
        n_folds: number of stratified folds.
        epochs: training epochs per fold (same for both branches).
        batch_size: batch size for both loaders.
        lr: Adam learning rate.
        seed: seed for fold assignment and each fold's model init.

    Returns:
        Dict with ``fold_metrics`` (per branch, per fold), ``pooled_metrics``
        (per branch, computed on the full out-of-fold prediction set), and
        ``delong`` (paired AUC significance test on the pooled predictions).
    """
    samples = pool_samples(root_dir)
    labels = np.array([label for _, label in samples])
    logger.info("pooled dataset: %d samples (%d polyp, %d normal)", len(samples), labels.sum(), (labels == 0).sum())

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    fold_metrics: dict[str, list[dict]] = {branch: [] for branch in BRANCHES}
    oof_probs: dict[str, np.ndarray] = {branch: np.full(len(samples), np.nan) for branch in BRANCHES}

    for fold, (train_idx, test_idx) in enumerate(skf.split(np.zeros(len(samples)), labels), start=1):
        logger.info("=== fold %d/%d: train=%d test=%d ===", fold, n_folds, len(train_idx), len(test_idx))

        for branch in BRANCHES:
            transform = default_transform() if branch == "baseline" else tusi_transform()
            train_ds = PathListDataset([samples[i] for i in train_idx], transform)
            test_ds = PathListDataset([samples[i] for i in test_idx], transform)

            model, _ = train_on_datasets(
                branch, train_ds, test_ds, epochs=epochs, batch_size=batch_size, lr=lr, seed=seed,
            )

            device = next(model.parameters()).device
            test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
            probs, fold_labels = collect_predictions(model, test_loader, device)
            oof_probs[branch][test_idx] = probs

            metrics = compute_metrics(probs, fold_labels)
            metrics["fold"] = fold
            fold_metrics[branch].append(metrics)
            logger.info("[%s] fold %d test metrics: %s", branch, fold, metrics)

    assert not any(np.isnan(oof_probs[branch]).any() for branch in BRANCHES), "every sample must get exactly one OOF prediction"

    pooled_metrics = {
        branch: compute_metrics(oof_probs[branch].tolist(), labels.tolist()) for branch in BRANCHES
    }
    delong = delong_roc_test(labels.tolist(), oof_probs["tusi"].tolist(), oof_probs["baseline"].tolist())
    delong = {"auc_tusi": delong["auc_a"], "auc_baseline": delong["auc_b"], "z": delong["z"], "p_value": delong["p_value"]}

    return {
        "n_folds": n_folds,
        "epochs": epochs,
        "fold_metrics": fold_metrics,
        "pooled_metrics": pooled_metrics,
        "delong": delong,
    }


def main() -> None:
    """CLI entry point: run CV, save the report, and plot per-fold AUCs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-dir", default=str(DEFAULT_ROOT_DIR))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    report = run_cross_validation(
        root_dir=args.root_dir, n_folds=args.n_folds, epochs=args.epochs,
        batch_size=args.batch_size, lr=args.lr, seed=args.seed,
    )

    with open(out_dir / "cv_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info("saved CV report to %s", out_dir / "cv_report.json")

    fold_aucs = {
        branch: [m["roc_auc"] for m in report["fold_metrics"][branch]] for branch in BRANCHES
    }
    plot_cv_fold_aucs(fold_aucs, out_dir / "cv_fold_aucs.png")

    print(f"\n=== Polyp {args.n_folds}-fold CV (pooled out-of-fold metrics) ===")
    for branch, m in report["pooled_metrics"].items():
        print(f"{branch}: accuracy={m['accuracy']:.3f} roc_auc={m['roc_auc']:.3f} "
              f"sensitivity={m['sensitivity']:.3f} f1={m['f1']:.3f}")
    d = report["delong"]
    print(f"\nDeLong's test (tusi vs. baseline AUC, pooled OOF predictions): "
          f"z={d['z']:.3f} p={d['p_value']:.4f}")


if __name__ == "__main__":
    main()
