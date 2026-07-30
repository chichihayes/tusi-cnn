"""DeLong's test for comparing two correlated ROC-AUC estimates.

Used by ``polyp/cross_validate.py`` to test whether the Tusi branch's AUC
differs significantly from the baseline's. A naive two-sample test (e.g.
comparing AUCs as if independent) is invalid here: in k-fold cross-
validation, both branches are evaluated on the *same* held-out samples in
each fold, so their prediction scores are correlated — a model that finds
a sample easy tends to do so regardless of input representation. DeLong's
test (DeLong, DeLong & Clarke-Pearson, 1988) accounts for that correlation
directly, via the covariance between each model's placement (Mann-Whitney
U) statistics, rather than treating the two AUCs as independent.

This is the standard O(n log n) implementation from Sun & Xu, 2014,
"Fast Implementation of DeLong's Algorithm for Comparing the Areas Under
Correlated Receiver Operating Characteristic Curves."
"""

import numpy as np
from scipy import stats


def _compute_midrank(x: np.ndarray) -> np.ndarray:
    """Compute midranks of ``x``, averaging ranks across tied values.

    Args:
        x: 1D array of scores.

    Returns:
        Midrank of each element of ``x``, in ``x``'s original order.
    """
    sorted_idx = np.argsort(x)
    sorted_x = x[sorted_idx]
    n = len(x)
    ranks = np.zeros(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n and sorted_x[j] == sorted_x[i]:
            j += 1
        ranks[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    midranks = np.empty(n, dtype=float)
    midranks[sorted_idx] = ranks
    return midranks


def _fast_delong(predictions_sorted: np.ndarray, num_positive: int) -> tuple[np.ndarray, np.ndarray]:
    """Compute AUCs and their covariance matrix (Sun & Xu, 2014, Algorithm 2).

    Args:
        predictions_sorted: ``(n_classifiers, n_samples)`` array of scores,
            with all positive-class samples ordered before all negatives.
        num_positive: number of positive-class samples.

    Returns:
        ``(aucs, covariance)`` — one AUC per classifier, and their
        ``(n_classifiers, n_classifiers)`` covariance matrix.
    """
    m = num_positive
    n = predictions_sorted.shape[1] - m
    k = predictions_sorted.shape[0]

    positive_examples = predictions_sorted[:, :m]
    negative_examples = predictions_sorted[:, m:]

    tx = np.empty([k, m], dtype=float)
    ty = np.empty([k, n], dtype=float)
    tz = np.empty([k, m + n], dtype=float)
    for r in range(k):
        tx[r, :] = _compute_midrank(positive_examples[r, :])
        ty[r, :] = _compute_midrank(negative_examples[r, :])
        tz[r, :] = _compute_midrank(predictions_sorted[r, :])

    aucs = tz[:, :m].sum(axis=1) / (m * n) - (m + 1.0) / (2.0 * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    covariance = sx / m + sy / n
    return aucs, np.atleast_2d(covariance)


def delong_roc_test(labels: list[int], probs_a: list[float], probs_b: list[float]) -> dict:
    """Paired DeLong test comparing two models' AUC on the same samples.

    Args:
        labels: true binary labels (0/1), shared by both models.
        probs_a: model A's predicted positive-class probabilities.
        probs_b: model B's predicted positive-class probabilities.

    Returns:
        Dict with ``auc_a``, ``auc_b``, ``z`` (z-statistic for the AUC
        difference), and ``p_value`` (two-sided).
    """
    labels = np.asarray(labels)
    order = np.argsort(-labels, kind="mergesort")  # positives (label=1) first
    num_positive = int(labels.sum())

    predictions_sorted = np.vstack([np.asarray(probs_a), np.asarray(probs_b)])[:, order]
    aucs, covariance = _fast_delong(predictions_sorted, num_positive)

    auc_diff = aucs[0] - aucs[1]
    variance = covariance[0, 0] + covariance[1, 1] - 2 * covariance[0, 1]
    z = auc_diff / np.sqrt(variance)
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))

    return {
        "auc_a": float(aucs[0]),
        "auc_b": float(aucs[1]),
        "z": float(z),
        "p_value": float(p_value),
    }
