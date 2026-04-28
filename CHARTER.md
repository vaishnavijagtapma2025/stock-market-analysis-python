# Project Charter — ECO 6810 Final Project

# A Sector-Wise Analysis and Prediction of Indian Equity Prices

This project presents a data-driven analysis and prediction of stock prices for large-cap Indian equities, integrating firm-level financial fundamentals, historical price data, and sector classification. The study first examines cross-sectional patterns in stock performance across firms and industries, and then develops machine learning models to forecast 1-year-ahead stock prices. By explicitly comparing model performance against a naive persistence benchmark, the project evaluates whether publicly available information contains meaningful predictive signals beyond simple price continuation. The results provide insights into the extent of predictability in equity markets and the practical value of data-driven approaches for investment analysis.

---

## Header

| Field | Value |
|---|---|
| Team members | Anushmitaa Ghosh, Vaishnavi Jagtap, Anushka Bid |
| Project type | predictive|
| Estimated hours per person |50|
| Charter version | v1 |
| Date |2026-04-28|


## 1. Problem and stakeholder


Equity analysts in the Indian stock market evaluate firms using financial fundamentals, historical price trends, and sector performance to guide investment decisions. In large-cap equities, however, stock prices often exhibit strong persistence, making it unclear whether publicly available information provides meaningful predictive power beyond simple benchmarks.

Over the past decade, large-cap Indian stocks have shown substantial variation in returns across firms and sectors such as Energy, Information Technology, and Financials. While analysts frequently rely on indicators such as return on equity, earnings growth, and valuation ratios, it remains an open question whether these variables can systematically explain or predict future stock performance.

We aim to quantify, at the firm level, whether financial fundamentals, historical prices, and sector characteristics contain statistically meaningful predictive information for 1-year-ahead stock performance across approximately 90 NSE-listed large-cap firms. This is framed as an evaluation of whether machine learning models can improve prediction accuracy relative to a naive persistence benchmark, which assumes that stock prices follow a random walk.


Prior empirical work in finance often emphasizes market efficiency, suggesting limited predictability using public information. Our contribution is to test this proposition in the context of the Indian equity market using a structured, sector-aware dataset and flexible machine learning methods, while explicitly benchmarking performance against a naive baseline.


## 2. Main outcome variable

Main outcome variable
Name: 1-year-ahead stock price (future closing price)
Unit: Indian Rupees (₹ per share)
Source table/column/field:
Yahoo Finance (via yfinance API),
field: Close price from yf.download()
Population / panel:
Cross-sectional dataset of approximately 90 NSE-listed large-cap firms.
For each firm, the outcome variable is the most recent available closing price (t), paired with a lagged price approximately 1 year prior (t−1).
The dataset is split into 80% training firms (~72) and 20% test firms (~18)

---

## 3. Main quantitative success threshold

The success of the predictive model is evaluated using out-of-sample Mean Squared Error (MSE), computed on a held-out test set comprising the most recent 20% of firms (N ≈ 18). MSE measures the average squared deviation between predicted and actual stock prices, providing a scale-sensitive measure of prediction accuracy. The model is considered successful if its MSE is strictly less than that of a naive persistence baseline, which predicts the future price as equal to the past price (Pt=Pt−1) and is computed on the same test set and reported first.

## 4. Baseline to beat

**Naive Persistence:** For every firm in the test set, predict that the 1-year-forward price equals the price observed exactly 365 days earlier (i.e., `predicted = current_price`). This is the standard no-information benchmark in equity price forecasting.

Based on the cross-section of 90 NSE large-caps over April 2025–April 2026, the baseline MSE reflects the average squared rupee deviation of prices from their year-ago levels — a non-trivial number given the wide price range (₹150 to ₹35,000) across the sample.

---

## 5. Falsifiable hypothesis

Among NSE large-cap equities, a model trained on fundamental indicators (PE ratio, ROE, ROA, debt-to-equity, profit margin, revenue growth, earnings growth, EBITDA margin, price-to-book, earnings yield, sector-relative PE, and log market capitalisation) will forecast 1-year-forward closing prices with a lower out-of-sample MSE than the naive persistence benchmark, and will correctly predict the direction of price movement (up or down) for at least 60% of firms in the held-out test set

---

## 6. Data sources and access plan

**Source: Yahoo Finance via `yfinance` Python library**

- **URL / API endpoint:** `https://finance.yahoo.com` (accessed programmatically via `yfinance>=0.2.36`)
- **Licence:** Yahoo Finance data is publicly available for personal and academic non-commercial use. No login required. No scraping — `yfinance` uses the official Yahoo Finance query API.
- **Access method:** Direct API call via Python; no authentication needed; no rate-limit issues at 90-ticker scale.

**Data fetched per ticker (Cell 3 of notebook):**
- Historical closing price 365 days ago → `current_price`
- Historical closing price today → `target_price` (ground truth)
- Fundamentals from `yf.Ticker(ticker).info`: `trailingPE`, `returnOnEquity`, `returnOnAssets`, `profitMargins`, `revenueGrowth`, `earningsGrowth`, `debtToEquity`, `currentRatio`, `marketCap`, `beta`, `bookValue`, `priceToBook`, `dividendYield`, `trailingEps`, `ebitdaMargins`

**10-line fetch probe (paste in any notebook cell to verify access):**
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

No manual scraping, no login, no permissions required. If Yahoo Finance rate-limits a session, the fallback is to add `time.sleep(0.5)` between ticker calls.

---


## 7. Scope limits

-We do not estimate any causal effect of firm fundamentals on stock prices; the analysis is purely predictive and associational.
*We do not construct or evaluate trading strategies, portfolio optimisation, or backtesting; the output is limited to price forecasts.
*We do not adjust for corporate actions beyond the adjusted closing prices provided by yfinance.
*We do not model intraday, weekly, or multi-year price dynamics; the prediction horizon is fixed at 1 year (365 days).
*We do not generalise findings beyond the sample of ~90 NSE large-cap firms used in the analysis.
*We do not harmonise accounting standards or reporting differences; all financial variables are used as reported by Yahoo Finance.
*We do not develop a production system, web application, or API; deliverables are limited to a reproducible notebook and output files.

## 8. Risks and fallback

*Risk: The dataset is small (≈ 90 firms) and cross-sectional, which may limit the ability of machine learning models to generalize and outperform the naive persistence baseline.
*Fallback: If models fail to beat the baseline MSE, we will reframe the analysis to focus on directional prediction (up/down classification) and report directional accuracy alongside error metrics.
*Risk: Stock prices are non-stationary and scale-dependent, which may lead to poor performance when predicting price levels directly.
*Fallback: If large prediction errors persist, we will transform the outcome variable to returns and evaluate model performance using return-based metrics.

## 9. Reproducibility checklist

Your final repo must satisfy all of these:

- [x] `uv run main.py` runs end-to-end in under 10 minutes on a clean machine with no manual intervention.
- [x] It writes `outputs/primary_metric.json` containing a single JSON object with at least `{"metric_name": "...", "value": <number>, "threshold": <number>, "passed": <bool>}`.
- [x] It writes `outputs/baseline_metric.json` in the same shape.
- [x] A `README.md` documents the commands and expected outputs in ≤ 20 lines.
- [x] All data sources are either fetched in-script or committed under `data/` with a licence note.

---

## Sign-off

By submitting this charter, the team agrees that this is the plan the project will be graded against. The instructor will not penalize you just because the topic turns out to be difficult, as long as the project stays honest and within the approved scope.

*Signed:*  Anushmitaa Ghosh, Vaishnavi Jagtap, Anushka Bid
