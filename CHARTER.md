# Project Charter — ECO 6810 Final Project

# A Sector-Wise Analysis and Prediction of Indian Equity Prices

This project presents a data-driven analysis and prediction of stock prices for large-cap Indian equities, integrating firm-level financial fundamentals, technical price signals, and sector classification. The study first examines cross-sectional patterns in stock performance across firms and industries, and then develops machine learning models to forecast 1-year-ahead stock prices. By explicitly comparing model performance against a naive persistence benchmark, the project evaluates whether publicly available information contains meaningful predictive signals beyond simple price continuation. The results provide insights into the extent of predictability in equity markets and the practical value of data-driven approaches for investment analysis.

---

| Field | Value |
|---|---|
| Team members | Anushmitaa Ghosh, Vaishnavi Jagtap, Anushka Bid |
| Project type | Predictive |
| Estimated hours per person | 50 |
| Charter version | v4 |
| Date | 2026-05-09 |

---


## Team Roles

| Team Member | Primary Role | Responsibilities |
|---|---|---|
| **Anushmitaa Ghosh** | Charter & Pipeline Lead & Code Lead| Project charter authoring and all revisions (`CHARTER.md`); defining the research question, hypothesis, and scope limits; `pyproject.toml` dependency management; `main.py` pipeline integration and end-to-end testing; synthetic fallback design; outputs folder structure and reproducibility checks, report.md|
| **Vaishnavi Jagtap** | Modelling & Repository Lead & Code Lead| Repository setup, branching, and version control; `main.py` model implementation (Ridge Regression, Random Forest, Gradient Boosting, XGBoost); hyperparameter tuning; SHAP feature importance analysis; `pyproject.toml` updates; `README.md` documentation; overall code quality and review, AI_USAGE_LOG.md|
| **Anushka Bid** | Evaluation & Data Lead and Engineering & Code Lead | Python development and implementation of `main.py`; management and structuring of `outputs/`, `data/`, `notebooks/`, and `artifacts/probes/` repository workflows; handling of `report.md` and project documentation; generation of `primary_metric.json`, `baseline_metric.json`, `milestone_manifest.json`, `full_predictions_.csv`, and model output visualizations; implementation of directional accuracy evaluation and baseline persistence benchmarks; portfolio analytics including Sharpe Ratio, Information Ratio, and Max Drawdown; management of `probe_output.txt` |

All three members jointly contribute to writing, review, and final submission.


## 1. Problem, Stakeholder, and Decision-Maker

**Who this project is for:**
The primary stakeholder is an **equity research analyst** (or junior portfolio manager) at a domestic Indian asset management firm or brokerage who monitors NSE large-cap stocks. Specifically, this analyst needs to triage which stocks warrant deeper fundamental review each quarter. They currently rely on a manual, judgment-based process using a handful of ratios and recent price momentum. The decision they face is: *which firms among approximately 91 NSE large-cap names are likely to deliver above-average price appreciation over the next 12 months, given publicly available fundamentals and technical signals as of today?*

**The analytical task:**
We build a model that ingests 21 features — comprising firm-level financial fundamentals (e.g., PE ratio, ROE, ROA, profit margin, revenue growth, earnings growth, debt-to-equity, current ratio, price-to-book, EBITDA margin, EPS, dividend yield, log market cap, earnings yield, PEG proxy), sector-relative signals (sector-median PE, relative PE, sector average margin and growth), and technical momentum indicators (1-quarter and 4-quarter momentum, RSI, price vs. 4-quarter and 8-quarter SMA) — and outputs a predicted 1-year-forward closing price per firm. The model is trained on approximately 72 firms (~80% of the sample) and evaluated on approximately 18 held-out test firms (~20%).

**Why this matters beyond curiosity:**
Indian large-cap equity markets are heavily covered yet analysts remain uncertain how much systematic predictive information is embedded in public fundamentals versus simple price continuation. This project quantifies that gap with a structured, reproducible pipeline.

---

## 2. Main Outcome Variable

| Field | Detail |
|---|---|
| **Name** | 1-year-ahead stock closing price (`target_price`) |
| **Unit** | Indian Rupees (₹ per share) |
| **Source** | Yahoo Finance via `yfinance` Python library |
| **Field** | `Close` price from `yf.download()` |
| **Population / Panel** | Cross-sectional dataset of approximately 91 NSE-listed large-cap firms. For each firm, `target_price` is the most recent available closing price at time *t* (today, May 2026), and `current_price` is the closing price approximately 365 days earlier at time *t−1* (May 2025). The dataset is split 80/20: ~72 training firms and ~18 test firms. |

The primary metric is the **out-of-sample MSE on the 20% test set**, computed on predicted vs. actual closing prices in ₹. As a secondary metric we report **directional accuracy** — the fraction of test firms for which the model correctly predicts whether the price went up or down over the year.

`outputs/primary_metric.json` is the sole grading artifact. It contains `test_mse` as `metric_name`, the baseline MSE as `threshold`, and a `passed` boolean. `outputs/baseline_metric.json` records the persistence baseline separately.

---

## 3. Exact Prediction Task

Given the following inputs, observed at time *t−1* (approximately May 2025):

- Firm-level fundamentals: `pe_ratio`, `roe`, `roa`, `profit_margin`, `revenue_growth`, `earnings_growth`, `debt_to_equity`, `current_ratio`, `beta`, `book_value`, `price_to_book`, `dividend_yield`, `eps`, `ebitda_margin`, `log_market_cap`, `earnings_yield`, `peg_proxy`
- Sector-relative signals: `sector_median_pe`, `relative_pe`, `sector_avg_margin`, `sector_avg_growth`
- Technical signals derived from quarterly price history prior to *t−1*: `mom_1q`, `mom_4q`, `rsi`, `price_vs_sma4`, `price_vs_sma8`
- `current_price` (closing price at *t−1*)

**Predict:** the closing price of each firm at time *t* (approximately May 2026), i.e., `target_price`.

Four models are trained and compared: Ridge Regression, Random Forest, Gradient Boosting, and XGBoost (with SHAP for feature importance). The best-performing model on the held-out test set is reported as the primary result. All technical signals are constructed strictly from data prior to `current_price` to avoid look-ahead bias.

---

## 4. Numeric Success Threshold

The project is considered successful if:

1. The best-performing ML model achieves a lower out-of-sample test MSE than the naive persistence baseline on the same held-out test set.
2. The model achieves directional accuracy of at least 60% on the held-out test set.

**The single primary metric for grading is `test_mse`, recorded in `outputs/primary_metric.json`.** The project passes if `best_model_mse < baseline_mse` (the `passed` boolean in that file).

Portfolio analysis (Sharpe ratio, benchmark comparison, and Top-15 portfolio evaluation) is reported as supplementary exploratory output only and plays **no role** in the pass/fail grading decision. It is not used as a primary or secondary grading metric under any circumstance.

Results are saved to:

- `outputs/primary_metric.json` — Model test MSE, baseline MSE threshold, and pass/fail indicator
- `outputs/baseline_metric.json` — Baseline persistence MSE
- `outputs/milestone_manifest.json` — Data source status and run metadata

---

## 5. Baseline to Beat

**Naive Persistence:** For every firm in the test set, predict that the 1-year-forward price equals the closing price observed exactly 365 days earlier:

predicted_price = current_price

This is the standard no-information benchmark in equity price forecasting. Based on the cross-section of ~91 NSE large-caps over May 2025–May 2026, the baseline MSE will be non-trivial given the wide price range across the sample (roughly ₹150 to ₹35,000), and substantial variation in 1-year returns across sectors (Energy, Technology, Finance, Consumer, etc.).


The baseline MSE is computed on the **same 20% held-out test set** as the model MSE, and is always reported first for transparency.

---

## 6. Falsifiable Hypothesis

Among NSE large-cap equities, a model trained on 21 features (financial fundamentals, sector-relative signals, and technical momentum indicators) will:

1. Forecast 1-year-forward closing prices with a **lower out-of-sample MSE** than the naive persistence benchmark on the held-out test set of ~18 firms; **and**
2. Correctly predict the **direction of price movement** (up or down) for at least **60%** of firms in the held-out test set.

Directional accuracy is reported as a secondary evaluation metric alongside test MSE. Pass/fail is determined solely by condition 1 via `outputs/primary_metric.json`.

---

## 7. Data Sources and Access Plan

**Source: Yahoo Finance via `yfinance` Python library**

- URL / API endpoint: `https://finance.yahoo.com` (accessed programmatically via `yfinance >= 0.2.36`)
- Licence: Yahoo Finance data is publicly available for personal and academic non-commercial use. No login required. No scraping — `yfinance` uses the official Yahoo Finance query API.
- Access method: Direct API call via Python; no authentication needed; no rate-limit issues at 91-ticker scale.

**Data fetched per ticker:**

- `current_price`: closing price from 365 days ago via `yf.download()` (used as the main predictor)
- `target_price`: closing price today via `yf.download()` (ground truth outcome)
- Quarterly price history (2-year window, `interval="3mo"`) for technical signal construction
- Fundamentals from `yf.Ticker(ticker).info`: `trailingPE`, `returnOnEquity`, `returnOnAssets`, `profitMargins`, `revenueGrowth`, `earningsGrowth`, `debtToEquity`, `currentRatio`, `marketCap`, `beta`, `bookValue`, `priceToBook`, `dividendYield`, `trailingEps`, `ebitdaMargins`

**10-line fetch probe** (verifies data access):

```python
import yfinance as yf
from datetime import datetime, timedelta

ticker = yf.Ticker("RELIANCE.NS")
hist = yf.download("RELIANCE.NS",
                   start=(datetime.today()-timedelta(days=5)).strftime('%Y-%m-%d'),
                   end=datetime.today().strftime('%Y-%m-%d'),
                   progress=False)
print("Current price:", float(hist['Close'].iloc[-1]))
print("PE Ratio:", ticker.info.get('trailingPE'))
```

**Probe path:** `data/probe_output.txt` — generated automatically when running `uv run main.py`.

If Yahoo Finance rate-limits a session, the notebook activates a **synthetic fallback** automatically (< 20 tickers fetched triggers synthetic data generation for all 91 firms using sector-level parameters). No manual intervention required.

The main dataset is available in the data folder.

---

## 8. Scope Limits

**In scope:**
- ~91 NSE-listed large-cap firms, cross-sectional dataset
- Prediction horizon: exactly 1 year (365 days), fixed
- Period: closing price at May 2025 → closing price at May 2026
- Sectors covered: Energy, Technology, Finance, Consumer, Automobile, Healthcare, Chemicals, Metals, Real Estate, Textiles, Retail, Defense
- Models: Ridge Regression, Random Forest, Gradient Boosting, XGBoost (with SHAP feature importance)
- Portfolio construction and backtest: Top-15 composite-score portfolio vs. equal-weight benchmark (Sharpe ratio, Information Ratio, Max Drawdown reported as supplementary only)

**Out of scope:**
- No causal inference; analysis is purely predictive and associational
- No trading strategies, portfolio optimisation beyond the illustrative Top-15 backtest, or live trading
- No adjustment for corporate actions beyond adjusted closing prices from `yfinance`
- No intraday, weekly, or multi-year modelling
- No generalisation beyond the ~91 NSE large-cap firms in the sample
- No harmonisation of accounting/reporting differences; data used as reported by Yahoo Finance
- No production system, web app, or real-time API

---

## 9. Risks and Fallbacks

**Risk 1:** The dataset is small (~91 firms, cross-sectional) and may limit model generalisation.
**Fallback:** If models fail to beat the baseline MSE, the analysis pivots to directional prediction (up/down classification) and reports directional accuracy as the primary metric with a 60% threshold.

**Risk 2:** Stock prices are non-stationary and scale-dependent; direct price level prediction may yield large errors.
**Fallback:** If large prediction errors persist, the outcome variable is transformed to 1-year returns and model performance is re-evaluated using return-based MSE and directional accuracy.

**Risk 3:** Yahoo Finance rate-limits or returns incomplete data for some tickers.
**Fallback:** Synthetic data fallback activates automatically when fewer than 20 tickers are successfully fetched, using sector-calibrated distributional parameters. No manual intervention required.

---

## 10. Reproducibility Checklist

- [x] `uv run main.py` runs end-to-end in under 10 minutes on a clean machine with no manual intervention.
- [x] All dependencies are declared in `pyproject.toml`; `main.py` contains no shell or notebook commands (no `!pip install` or any `!` prefixed lines).
- [x] `main.py` is plain Python and serves as the single entry point — no Jupyter/Colab-specific syntax anywhere in the file.
- [x] It writes `outputs/primary_metric.json` containing `{"metric_name": "test_mse", "value": <number>, "threshold": <baseline_mse>, "passed": <bool>}`.
- [x] It writes `outputs/baseline_metric.json` in the same shape with `"metric_name": "baseline_mse"`.
- [x] It writes `outputs/milestone_manifest.json` confirming data source status and run metadata.
- [x] A `README.md` documents the commands and expected outputs in ≤ 20 lines.
- [x] All data is fetched in-script via `yfinance`; synthetic fallback is triggered automatically if live data is unavailable.
- [x] A `data/probe_output.txt` is written on every run confirming Yahoo Finance access (or fallback activation).

---

## Sign-off

*Signed:* Anushmitaa Ghosh, Vaishnavi Jagtap, Anushka Bid
