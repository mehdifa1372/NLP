# NLP Evaluation Toolkit

A small, dependency-light toolkit for evaluating binary text classifiers and performing transparent error analysis. It is designed to support reproducible NLP experiments without coupling evaluation code to a particular transformer or training framework.

## Features

- Accuracy, precision, recall, F1, and confusion counts with explicit zero-division behavior.
- Paired bootstrap confidence intervals for comparing two classifiers on the same examples.
- Ranked false-positive and false-negative examples for qualitative error analysis.
- Deterministic random sampling through an explicit seed.
- Unit tests and continuous integration.

## Install and test

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
pytest -q
```

## Example

```python
from nlp_evaluation import binary_metrics, rank_errors

labels = [1, 0, 1, 0]
predictions = [1, 1, 0, 0]
scores = [0.9, 0.8, 0.4, 0.1]
texts = ["excellent", "not for me", "surprisingly good", "poor"]

print(binary_metrics(labels, predictions))
for error in rank_errors(texts, labels, scores):
    print(error.kind, error.score, error.text)
```

## Research guidance

A single score rarely explains model quality. Report uncertainty, compare against a meaningful baseline, inspect errors, document the decision threshold, and preserve a held-out test set. For imbalanced or high-cost decisions, add metrics appropriate to the application rather than relying on accuracy alone.

The paired bootstrap routine measures uncertainty in the accuracy difference between two systems. It is intended as an accessible research utility, not as a substitute for a preregistered statistical analysis.

## Roadmap

- Multiclass and multilabel metrics.
- Threshold sweeps and calibration diagnostics.
- Group-wise performance and confidence intervals.
- HTML/Markdown experiment reports.
- Integration example using the IMDb transformer project.

## Responsible use

Error reports may contain sensitive, offensive, or personally identifying text from the evaluated dataset. Redact or aggregate examples before sharing reports publicly.

## Author

Mehdi Faraz — computer vision, machine learning, data science, and applied AI.

