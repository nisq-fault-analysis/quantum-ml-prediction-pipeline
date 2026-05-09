# Semester Work Submission Notes

## Primary Review Artifacts

- `README.md`: project overview, dataset roles, leakage policy, benchmark summary, and run instructions.
- `dataprocessing.ipynb` / `dataprocessing.html`: data framing, split audit, missingness, and leakage boundary.
- `benchmark.ipynb` / `benchmark.html`: validation-only ML benchmark with train-only preprocessing.
- `data/hw2/`: committed grouped train/validation/test sample plus split and feature-policy sidecars.
- `src/` and `tests/`: reusable implementation and automated checks supporting the notebooks.

## Machine Learning Framing

- Task: supervised regression.
- Target: `reliability`, a continuous value in `[0, 1]`.
- Inputs: pre-run circuit, compiler, and hardware-calibration features only.
- Main selection metric: validation MAE.
- Supporting metrics: validation RMSE and validation R2.
- Held-out test policy: saved but not used for HW2 model comparison.

## Leakage And Split Controls

- Splits are grouped by `base_circuit_id`, so raw/transpiled variants of the same base
  circuit do not cross train, validation, and test.
- `data/hw2/split_summary.json` reports zero group overlap across all split pairs.
- Outcome-like columns such as `reliability`, `fidelity`, `error_rate`, output counts,
  and mitigation-derived fields are excluded from model inputs.
- Learned preprocessing is inside sklearn pipelines fitted on training rows only.

## Current Executed Benchmark

| Model | Validation MAE | Validation RMSE | Validation R2 |
| --- | ---: | ---: | ---: |
| Ridge regression | 0.072702 | 0.109432 | 0.664128 |
| Dummy mean | 0.145326 | 0.188831 | -0.000074 |

The Ridge baseline roughly halves validation MAE versus the train-mean dummy baseline,
showing that the pre-run features carry useful signal without using post-run leakage.

## Verification

Run from the repository root:

```powershell
.venv\Scripts\python.exe -m pytest
.venv\Scripts\python.exe -m ruff check .
```

At the time of preparation, the test suite passed with 74 tests.
