# Indian Equity Price Predictor — ECO 6810 Final Project

**A Sector-Wise Analysis and Prediction of Indian Equity Prices**

**Team:** Anushmitaa Ghosh · Vaishnavi Jagtap · Anushka Bid
 **Date:** May 2026

---

## What This Project Does

This project forecasts 1-year-ahead closing prices for ~91 NSE large-cap equities using 21 publicly available features — financial fundamentals, sector-relative signals and technical momentum indicators and evaluates whether the best ML model beats a naive persistence baseline (predict price unchanged) on a held-out test set.

**Research question:** Does a 21-feature model outperform naive persistence on out-of-sample test MSE, and does it achieve ≥ 60% directional accuracy?

---

## Run the Notebook

Open in Google Colab and run all cells top-to-bottom:

```
notebooks/Indian_Equity_Predictor_ECO6810_CLEAN(1).ipynb
```

No configuration needed. The notebook installs its own dependencies in Cell 1 and auto-activates a synthetic fallback if fewer than 20 live tickers are fetched.

Expected runtime: **~3–5 minutes** in Colab (data fetch is the bottleneck).

---

## Run the Script (Reproducible Pipeline)

```bash
uv sync
uv run main.py
```

This is the single reproducible entry point. It writes all required output files and exits. No Jupyter kernel or manual steps required.

---

## Outputs Written

| File | Contents |
|---|---|
| `outputs/primary_metric.json` | Best model test MSE, baseline MSE threshold, pass/fail |
| `outputs/baseline_metric.json` | Naive persistence baseline MSE |
| `outputs/milestone_manifest.json` | Data source status, run metadata |
| `data/probe_output.txt` | Yahoo Finance access confirmation (or fallback notice) |

**Pass condition:** `best_model_mse < baseline_mse` → `"passed": true` in `primary_metric.json`.

---

## Models Trained

Four models are trained and compared on the same 80/20 stratified split (~72 train / ~18 test firms):

- Ridge Regression
- Random Forest
- Gradient Boosting
- XGBoost (+ SHAP feature importance)

The best test-set MSE is reported as the primary result.

---

## Features Used (21 total)

| Group | Features |
|---|---|
| Firm fundamentals (17) | `pe_ratio`, `roe`, `roa`, `profit_margin`, `revenue_growth`, `earnings_growth`, `debt_to_equity`, `current_ratio`, `beta`, `book_value`, `price_to_book`, `dividend_yield`, `eps`, `ebitda_margin`, `log_market_cap`, `earnings_yield`, `peg_proxy` |
| Sector-relative signals (4) | `sector_median_pe`, `relative_pe`, `sector_avg_margin`, `sector_avg_growth` |
| Technical momentum (5) | `mom_1q`, `mom_4q`, `rsi`, `price_vs_sma4`, `price_vs_sma8` |

All technical signals are derived strictly from quarterly price history prior to `t−1` (no look-ahead bias).

---

## Data Source

Live data is fetched from Yahoo Finance via `yfinance >= 0.2.36` — no API key required.

- `current_price`: closing price ~365 days ago (≈ May 2025)
- `target_price`: closing price today (≈ May 2026)
- Quarterly price history: 2-year window for technical signal construction

**Fallback:** Synthetic data (90 firms, sector-calibrated parameters) activates automatically if fewer than 20 tickers succeed.

---

## Repo Map

| Path | Purpose |
|---|---|
| `CHARTER.md` | Full project plan and methodology |
| `main.py` | Reproducible run entry point |
| `notebooks/` | Colab notebook for exploration and full pipeline |
| `project_code/` | Reusable helper functions |
| `data/` | Data files and probe outputs |
| `outputs/` | Required JSON metric files, tables, figures |
| `report.md` | Final written report |
| `AI_USAGE_LOG.md` | AI tool usage disclosure |

---

## Scope

**In scope:** ~90 NSE large-cap firms, cross-sectional, 1-year price prediction (May 2025 → May 2026), 12 sectors, supplementary Top-15 portfolio analysis (Sharpe ratio, IR, Max Drawdown — reported as exploratory only, not part of pass/fail grading).

**Out of scope:** Causal inference, trading strategies, intraday or multi-year modelling, generalisation beyond the ~90 sample firms.

---

*Signed:* Anushmitaa Ghosh, Vaishnavi Jagtap, Anushka Bid
