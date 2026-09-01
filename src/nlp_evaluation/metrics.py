"""Evaluation metrics and uncertainty estimates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _binary_array(values: ArrayLike, name: str) -> NDArray[np.int64]:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if len(array) == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.isin(array, [0, 1]).all():
        raise ValueError(f"{name} must contain only 0 and 1")
    return array.astype(np.int64)


def binary_metrics(labels: ArrayLike, predictions: ArrayLike) -> dict[str, float | int]:
    """Compute binary classification metrics with zero-safe division."""
    truth = _binary_array(labels, "labels")
    predicted = _binary_array(predictions, "predictions")
    if truth.shape != predicted.shape:
        raise ValueError("labels and predictions must have the same shape")

    true_positive = int(np.sum((truth == 1) & (predicted == 1)))
    true_negative = int(np.sum((truth == 0) & (predicted == 0)))
    false_positive = int(np.sum((truth == 0) & (predicted == 1)))
    false_negative = int(np.sum((truth == 1) & (predicted == 0)))

    def divide(numerator: int, denominator: int) -> float:
        return float(numerator / denominator) if denominator else 0.0

    precision = divide(true_positive, true_positive + false_positive)
    recall = divide(true_positive, true_positive + false_negative)
    return {
        "accuracy": divide(true_positive + true_negative, len(truth)),
        "precision": precision,
        "recall": recall,
        "f1": divide(2 * precision * recall, precision + recall),
        "true_positive": true_positive,
        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative,
    }


@dataclass(frozen=True)
class BootstrapComparison:
    """Accuracy difference and percentile confidence interval for model A minus B."""

    observed_difference: float
    lower_bound: float
    upper_bound: float
    confidence: float
    resamples: int


def paired_bootstrap_accuracy(
    labels: ArrayLike,
    predictions_a: ArrayLike,
    predictions_b: ArrayLike,
    *,
    resamples: int = 2_000,
    confidence: float = 0.95,
    seed: int = 42,
) -> BootstrapComparison:
    """Compare classifier accuracies using paired bootstrap resampling."""
    truth = _binary_array(labels, "labels")
    first = _binary_array(predictions_a, "predictions_a")
    second = _binary_array(predictions_b, "predictions_b")
    if truth.shape != first.shape or truth.shape != second.shape:
        raise ValueError("labels and both prediction arrays must have the same shape")
    if resamples <= 0:
        raise ValueError("resamples must be positive")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")

    first_correct = (first == truth).astype(float)
    second_correct = (second == truth).astype(float)
    observed = float(first_correct.mean() - second_correct.mean())
    generator = np.random.default_rng(seed)
    differences = np.empty(resamples, dtype=float)
    for index in range(resamples):
        sample = generator.integers(0, len(truth), size=len(truth))
        differences[index] = first_correct[sample].mean() - second_correct[sample].mean()

    tail = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(differences, [tail, 1.0 - tail])
    return BootstrapComparison(observed, float(lower), float(upper), confidence, resamples)

