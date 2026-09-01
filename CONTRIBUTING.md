# Contributing

Evaluation behavior must be explicit, deterministic, and tested against small examples with known answers. New statistical routines should document assumptions and should not overstate what an interval or test establishes.

```bash
pip install -e ".[dev]"
ruff check src tests
pytest -q
```

Do not commit private evaluation text, credentials, downloaded datasets, model weights, or generated reports containing sensitive examples.

