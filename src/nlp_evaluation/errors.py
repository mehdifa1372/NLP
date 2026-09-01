"""Qualitative error-analysis helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike


@dataclass(frozen=True)
class ClassificationError:
    index: int
    kind: str
    label: int
    prediction: int
    score: float
    text: str


def rank_errors(
    texts: Sequence[str],
    labels: ArrayLike,
    positive_scores: ArrayLike,
    *,
    threshold: float = 0.5,
    top_k: int | None = 20,
) -> list[ClassificationError]:
    """Return the most confident binary-classification errors first."""
    truth = np.asarray(labels)
    scores = np.asarray(positive_scores, dtype=float)
    if truth.ndim != 1 or scores.ndim != 1:
        raise ValueError("labels and positive_scores must be one-dimensional")
    if len(texts) != len(truth) or truth.shape != scores.shape:
        raise ValueError("texts, labels, and scores must have equal length")
    if not np.isin(truth, [0, 1]).all():
        raise ValueError("labels must contain only 0 and 1")
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("positive_scores must be finite probabilities between 0 and 1")
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")
    if top_k is not None and top_k < 0:
        raise ValueError("top_k must be non-negative or None")

    predictions = (scores >= threshold).astype(int)
    errors: list[ClassificationError] = []
    for index in np.flatnonzero(predictions != truth):
        prediction = int(predictions[index])
        errors.append(
            ClassificationError(
                index=int(index),
                kind="false_positive" if prediction == 1 else "false_negative",
                label=int(truth[index]),
                prediction=prediction,
                score=float(scores[index]),
                text=str(texts[index]),
            )
        )

    errors.sort(key=lambda item: abs(item.score - threshold), reverse=True)
    return errors if top_k is None else errors[:top_k]

