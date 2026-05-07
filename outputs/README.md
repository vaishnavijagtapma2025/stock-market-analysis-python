# Indian Equity Predictor — ECO6810

## Setup
Install [uv](https://docs.astral.sh/uv/getting-started/installation/) then:

## Run
```bash
uv run main.py
```

## Expected outputs
- `outputs/primary_metric.json` — best model test MSE vs baseline MSE (pass/fail)
- `outputs/baseline_metric.json` — naive persistence baseline MSE
- `outputs/milestone_manifest.json` — data source status and run metadata
- `outputs/model_comparison.json` — all 4 models compared
- `outputs/full_predictions.csv` — per-firm predictions
- `outputs/chart*.png` — 7 analysis charts
