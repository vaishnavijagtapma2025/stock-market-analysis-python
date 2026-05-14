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
- `outputs/source_probes/yfinance_probe.md` — data-source probe
- `data/probe_output.txt` — probe log written by the pipeline
- `outputs/chart1_data_overview.png` — data overview
- `outputs/chart2_eda_deepdive.png` — sector heatmap + return correlations
- `outputs/chart3_price_distribution.png` — price distribution + sector boxplots
- `outputs/chart4_correlation_matrix.png` — correlation matrix and fundamentals panels
- `outputs/chart5_model_comparison.png` — baseline vs models comparison
- `outputs/chart6_actual_vs_predicted.png` — actual vs predicted scatter
- `outputs/chart7_residuals.png` — residual diagnostics
- `outputs/chart8_shap_importance.png` — SHAP feature importance
- `outputs/chart9_portfolio.png` — portfolio performance
- `outputs/eda_chart4_sector_performance.png` — sector performance supplemental
- `outputs/eda_chart6_technical.png` — technical signal supplemental
