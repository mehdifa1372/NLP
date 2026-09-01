"""Metrics and error-analysis utilities for NLP classifiers."""

from .errors import ClassificationError, rank_errors
from .metrics import BootstrapComparison, binary_metrics, paired_bootstrap_accuracy

__all__ = [
    "BootstrapComparison",
    "ClassificationError",
    "binary_metrics",
    "paired_bootstrap_accuracy",
    "rank_errors",
]

