import numpy as np
import pytest

from nlp_evaluation import binary_metrics, paired_bootstrap_accuracy


def test_binary_metrics_reports_confusion_counts_and_scores():
    result = binary_metrics([1, 1, 0, 0], [1, 0, 1, 0])
    assert result["accuracy"] == 0.5
    assert result["precision"] == 0.5
    assert result["recall"] == 0.5
    assert result["f1"] == 0.5
    assert result["false_positive"] == 1
    assert result["false_negative"] == 1


def test_paired_bootstrap_is_deterministic():
    labels = np.array([0, 1, 0, 1, 1, 0])
    first = paired_bootstrap_accuracy(labels, labels, np.zeros_like(labels), resamples=100, seed=7)
    second = paired_bootstrap_accuracy(labels, labels, np.zeros_like(labels), resamples=100, seed=7)
    assert first == second
    assert first.observed_difference == 0.5


def test_metrics_reject_non_binary_values():
    with pytest.raises(ValueError, match="only 0 and 1"):
        binary_metrics([0, 2], [0, 1])

