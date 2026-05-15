# Data Folder — Indian Equity Price Predictor 

This folder contains the processed datasets and generated outputs used in the Indian Equity Price Predictor project.

---

## Sources

### 1. Yahoo Finance — NSE Stock Prices & Fundamentals

| Field | Detail |
|-------|--------|
| **Source** | Yahoo Finance via the `yfinance` Python library |
| **URL** | https://finance.yahoo.com |
| **Licence / access rule** | Public, free, no API key required. Yahoo Finance data is used strictly for academic and educational purposes. |
| **How it is fetched** | `Cell 3` of the notebook using `yf.download()` for price data and `yf.Ticker().info` for firm fundamentals |
| **Tickers covered** | ~91 NSE large-cap stocks listed on the National Stock Exchange of India (`.NS` suffix) |
| **What is included in the repository** | Cleaned and feature-engineered datasets generated from the live fetch pipeline |
| **What is excluded** | Raw bulk downloads, temporary cache files, and intermediate API outputs are intentionally excluded |

---

## Data Pull Structure

Three separate pulls are made per ticker:

| Pull | Variable | Window | Purpose |
|------|----------|--------|---------|
| Past price | `current_price` | `PUBLISH_DATE → PUBLISH_END` | Price observed at prediction time (`t−1`) |
| Current price | `target_price` | `TARGET_START → TARGET_END` | Realised price approximately one year later (`t`) |
| Historical closes | Technical indicators | `HIST_START → PUBLISH_END` | Used to compute RSI, momentum, and SMA signals without look-ahead bias |

---

## Features Generated

### Financial Variables
- PE Ratio
- ROE
- ROA
- Profit Margin
- Revenue Growth
- Earnings Growth
- Debt-to-Equity Ratio
- Current Ratio
- Beta
- Book Value
- Price-to-Book Ratio
- Dividend Yield
- EPS
- EBITDA Margin
- Market Capitalization

### Technical Indicators
- RSI
- Quarterly Momentum (`mom_1q`)
- Annual Momentum (`mom_4q`)
- Price vs SMA4
- Price vs SMA8

All technical indicators are generated strictly using historical information available prior to the prediction date to prevent look-ahead bias.

---

## Synthetic Fallback

If fewer than 20 tickers are successfully fetched from Yahoo Finance, the pipeline automatically activates a synthetic fallback generator.

The fallback:
- Generates 91 synthetic firms
- Uses sector-specific parameter distributions
- Preserves reproducibility using a fixed random seed
- Ensures the project remains executable even if live API fetches fail

The pipeline records this information in:

```text
data/probe_output.txt
```

---

## Date Windows (Computed at Runtime)

```python
TODAY        = datetime.today()
PUBLISH_DATE = TODAY - 365 days   # current_price start (t−1)
PUBLISH_END  = TODAY - 358 days   # current_price end (7-day window)

TARGET_START = TODAY - 5 days     # target_price start (t)
TARGET_END   = TODAY + 1 day      # target_price end

HIST_START   = TODAY - 730 days   # 2-year history start (technical signals only)
```

Because these windows are generated dynamically using `datetime.today()`, rerunning the notebook on a different date will fetch updated market prices. While the structure of the dataset remains stable, exact model outputs and MSE values may vary with changing market conditions.

---

## Historical Price Fetch

```python
hist_past = yf.download(
    ticker,
    start=PUBLISH_DATE,
    end=PUBLISH_END,
    progress=False
)
```

---

## Data Fetching Method

Data is fetched programmatically using:

- `yf.download()` for historical stock prices
- `yf.Ticker().info` for firm fundamentals

---

## How to Re-fetch the Data

No manual steps are required.

### Google Colab

```text
Runtime → Run all
```

### Local Execution

```bash
uv run main.py
```

Cells 3 and 4 automatically:

- Fetch live Yahoo Finance data
- Generate financial and technical indicators
- Create processed datasets
- Write probe logs for verification

If Yahoo Finance is unavailable or fewer than 20 valid tickers are returned, the synthetic fallback generator activates automatically so the pipeline still completes successfully.
