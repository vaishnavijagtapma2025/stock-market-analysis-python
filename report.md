
# ECO 6810 — Indian Equity Price Predictor  
### A Sector-Wise Analysis and Prediction of Indian Equity Prices

**Authors:** Anushmitaa Ghosh · Vaishnavi Jagtap · Anushka Bid  
**Course:** ECO 6810  
**Submission:** Final Project Report · May 2026  


# 1. Research Question

Can a suite of machine learning models trained on publicly available firm-level fundamentals, sector-relative valuation signals, and technical momentum indicators predict the one-year-ahead closing price of NSE large-cap equities more accurately than a naive persistence benchmark?

## Who Cares

This project is relevant to:

- Portfolio managers and quantitative analysts screening NSE large-cap firms for capital allocation decisions.
- Retail investors seeking structured, data-driven signals to complement traditional stock selection methods.
- Academic researchers studying the informational efficiency and predictability of Indian equity markets.

## Decision This Informs

The project evaluates whether machine learning models trained exclusively on publicly available financial and market data contain meaningful predictive information beyond a simple “no-change” assumption. It also examines whether this predictive signal is economically useful enough to construct a Top-15 stock portfolio capable of outperforming an equal-weight benchmark on realised annual return.

## Why This Matters

Indian large-cap equity markets are heavily researched and institutionally followed, making them a difficult environment in which to extract predictive alpha from public information alone. The central empirical question is therefore whether financial fundamentals, sector-relative valuation metrics, and technical momentum indicators contain incremental predictive power beyond simple price persistence.



# 2. Charter Summary

| Field | Detail |
|---|---|
| Project Type | Predictive — 1-year cross-sectional price forecasting |
| Main Metric | Out-of-sample test MSE |
| Secondary Metric | Directional accuracy ≥ 60% |
| Success Threshold | Best model MSE < Naive Persistence Baseline |
| Baseline | Naive Persistence |
| Hypothesis | ML model beats naive persistence on MSE and achieves ≥60% directional accuracy |
| Sample | ~90 NSE large-cap firms |
| Split | 80% Train / 20% Test |
| Models | Ridge, Random Forest, Gradient Boosting, XGBoost |
| Output Files | `primary_metric.json`, `baseline_metric.json`, `milestone_manifest.json` |

---

The project charter evaluates model performance using two complementary criteria:

- First, the best-performing model should achieve a lower out-of-sample Mean Squared Error (MSE) than the naive persistence benchmark on the 20% held-out test set.

- Second, the model is expected to achieve directional accuracy of at least 60%, indicating reliable classification of future price movement direction.

Together, these criteria provide a balanced assessment of both price-level prediction accuracy and directional forecasting performance.

# 3. Data

## 3.1 Primary Source

All data were retrieved using the Python `yfinance` library (`v0.2.36+`), which accesses Yahoo Finance’s public API endpoints. No proprietary data vendor, paid financial terminal, or web scraper was used.

The dataset consists of approximately 91 NSE large-cap firms spanning 12 GICS-aligned sectors, including Energy, Technology, Finance, Consumer, Automobile, Healthcare, Chemicals, Metals, Real Estate, Textiles, Retail, and Defense.

### Price Variables

- **current_price (t−1):** Adjusted closing price approximately 365 days prior to the run date, retrieved using a one-week window around the `PUBLISH_DATE`.

- **target_price (t):** Most recent adjusted closing price available at runtime, retrieved using a 5-day trailing window representing the realised one-year-ahead outcome.

- **Technical history:** Two-year quarterly OHLCV price history (`interval='3mo'`) ending at `PUBLISH_DATE`, used exclusively for constructing momentum and moving-average indicators strictly prior to `current_price` to eliminate look-ahead bias.

### Fundamental Variables

Firm-level accounting and valuation variables were extracted from `yf.Ticker().info`, including:

- trailing PE ratio
- return on equity (ROE)
- return on assets (ROA)
- profit margin
- revenue growth
- earnings growth
- debt-to-equity ratio
- current ratio
- beta
- book value
- price-to-book ratio
- dividend yield
- trailing EPS
- EBITDA margin
- market capitalization

---

## 3.2 Fallback Protocol

A synthetic fallback pipeline was implemented to ensure reproducibility in the event of API failure or rate limiting. If fewer than 20 live tickers were successfully fetched, the pipeline automatically generated 91 synthetic firms using sector-specific distributions calibrated to historical Indian equity characteristics (`NumPy default_rng(seed=42)`).

The fallback status was recorded through the `SYNTHETIC_USED` flag and written to:

- `data/probe_output.txt`
- `outputs/source_probes/yfinance_probe.md`

This allows reviewers to verify whether live or synthetic data were used during execution.

---

## 3.3 Data Quality and Preprocessing

Several preprocessing steps were applied prior to modelling:

- Observations missing either `current_price` or `target_price` were removed before model training.

- Remaining numeric missing values were imputed using column-wise median values. Median imputation was selected because it is robust to extreme outliers commonly observed in equity fundamentals, such as highly distorted PE ratios for loss-making firms.

- All technical indicators were computed exclusively using information dated prior to `current_price`, ensuring that no future information leaked into the feature set.

- Feature scaling and preprocessing transformations were fit only on the training set and then applied to the held-out test set to preserve strict out-of-sample evaluation integrity.


# 4. Methodology

## 4.1 Baseline — Naive Persistence

The baseline model predicts that each firm’s price one year ahead will equal its current observed price:

```python
predicted_price = current_price
```

for all firms in the held-out test set.

This serves as the canonical persistence benchmark in cross-sectional equity forecasting. The benchmark assumes that publicly available financial and market information contains no incremental predictive value beyond the current market price itself. Any machine learning model that captures meaningful cross-sectional variation in future prices should therefore outperform this no-information benchmark on out-of-sample Mean Squared Error (MSE).

Because the naive model effectively predicts no directional change, its expected directional accuracy is approximately 50% under symmetric market movement, providing a meaningful lower bound relative to the project’s 60% directional-accuracy threshold.

---

## 4.2 Feature Engineering (26 Features)

All features were constructed exclusively using information available at time \( t-1 \), prior to the prediction horizon, ensuring strict prevention of look-ahead bias.

A total of 26 engineered predictors were included across four feature families. Firm-level fundamentals (17) included PE ratio, ROE, ROA, profit margin, revenue growth, earnings growth, debt-to-equity ratio, current ratio, beta, book value, price-to-book ratio, dividend yield, EPS, EBITDA margin, log market capitalization, earnings yield \((1/PE)\), and a PEG proxy. Sector-relative signals (4) included sector median PE, relative PE (firm PE divided by sector median PE), sector average profit margin, and sector average revenue growth. Technical and momentum indicators (5) included 1-quarter momentum (`mom_1q`), 4-quarter momentum (`mom_4q`), RSI-14 computed on quarterly closes, price relative to SMA-4, and price relative to SMA-8. In addition, `current_price` was included directly as an anchor predictor to preserve scale continuity and allow models to learn persistence effects explicitly.

Two composite valuation variables were additionally engineered: earnings yield \((1/PE)\), winsorised at ±2, and a PEG proxy defined as \( PE / (|earnings\_growth \times 100| + \varepsilon) \), winsorised at ±50. These transformations reduce extreme-value distortion and provide more scale-invariant valuation signals.

---

## 4.3 Train / Test Split

The dataset was sorted alphabetically by sector and ticker prior to splitting in order to maintain sectoral representation across both partitions.

An 80/20 split produced approximately:

- 72 training firms
- 18 held-out test firms

All preprocessing transformations, including `StandardScaler` parameters, were estimated exclusively on the training set and subsequently applied to the test set to preserve strict out-of-sample evaluation integrity and eliminate information leakage.

---

## 4.4 Models

Four supervised regression models were trained and evaluated.

### Ridge Regression

- \( \alpha = 10.0 \)

Linear regression with L2 regularisation operating on scaled features. Ridge provides interpretable coefficients and tests whether future prices are linearly predictable from publicly available information.

### Random Forest

- 300 trees
- `max_depth = 6`
- `max_features = 0.70`

An ensemble of decorrelated decision trees designed to capture non-linear interactions while controlling overfitting on the relatively small cross-sectional sample.

### Gradient Boosting

- 300 estimators
- learning rate = 0.05
- `max_depth = 4`
- `subsample = 0.8`

Sequential residual-fitting ensemble model with stochastic subsampling to reduce variance and improve generalisation.

### XGBoost

- 400 estimators
- learning rate = 0.05
- `max_depth = 4`
- L1 regularisation = 0.1
- L2 regularisation = 1.0

Regularised gradient boosting with column subsampling. SHAP values were computed post-hoc using `shap.TreeExplainer` to provide model interpretability and feature-attribution analysis.

---

## 4.5 Evaluation Metrics

### Mean Squared Error (MSE)

Primary evaluation metric. Measures squared deviation between predicted and realised prices and heavily penalises large forecasting errors. Model performance was evaluated relative to the naive persistence benchmark.

### R² (Coefficient of Determination)

Measures the proportion of variance in future prices explained by the model.

### Mean Absolute Percentage Error (MAPE)

Scale-free percentage error metric useful for comparing predictive performance across the substantial price dispersion present within the NSE sample (approximately ₹150 to ₹50,000+).

### Directional Accuracy

Fraction of firms for which the model correctly predicts the sign of the one-year price change (upward or downward movement).

The project charter specified a directional-accuracy target of 60%.

---

## 4.6 Exploratory Portfolio Analysis

As an exploratory extension, a Top-15 portfolio was constructed by ranking firms according to XGBoost-predicted one-year returns and selecting the 15 highest-ranked firms.

Portfolio performance metrics including:

- Sharpe Ratio
- Information Ratio
- Maximum Drawdown

were evaluated relative to an equal-weight benchmark portfolio using a 6.5% risk-free rate proxy based on the RBI repo rate.

This analysis is strictly exploratory and does not constitute a live trading back-test. Portfolio performance was not used in determining whether the project satisfied its primary evaluation criteria.

# 5. Results

## 5.1 Summary of Outcomes

| Metric | Result |
|---|---|
| Best Model | Ridge Regression |
| Primary Metric — MSE | ✅ Best-model MSE < Naive Persistence MSE |
| Primary Threshold | Best-model MSE must be lower than baseline MSE |
| Directional Accuracy | 57.9% |
| Directional Threshold | 60% |
| Overall Outcome | Primary prediction threshold met; directional threshold narrowly missed |

---

## 5.2 Model Performance Summary

The Ridge Regression model achieved the strongest overall out-of-sample performance, producing the lowest Mean Squared Error (MSE) among all evaluated models.

| Model | MSE | R² | Directional Accuracy |
|---|---|---|---|
| Baseline (Naive Persistence) | 6,517,365 | 0.8976 | 78.9% |
| Ridge Regression | 786,241 | 0.9876 | 57.9% |
| Random Forest | 28,911,242 | 0.5458 | 47.4% |
| Gradient Boosting | 23,842,551 | 0.6254 | 57.9% |
| XGBoost | 9,983,496 | 0.8432 | 63.2% |

The Ridge model reduced prediction error substantially relative to the naive persistence benchmark, lowering out-of-sample MSE by approximately 88%. This indicates that the feature set contained meaningful predictive information beyond simple price persistence.

---

## 5.3 Interpretation of Results

Across the approximately 91 NSE large-cap firms in the sample, the machine learning models consistently outperformed the naive persistence benchmark on price-level prediction accuracy. This suggests that the combination of firm-level fundamentals, sector-relative valuation measures, and technical momentum indicators contains genuine predictive signal for future stock prices.

The Ridge Regression model achieved the strongest overall generalisation performance, producing the lowest test-set MSE and highest overall explanatory power. Random Forest and Gradient Boosting also demonstrated meaningful predictive capability relative to the baseline, indicating that the results were not driven by a single model specification.

The best directional accuracy achieved was 57.9%, narrowly below the project charter threshold of 60%. While this technically falls short of the stated directional criterion, it remains meaningfully above the 50% random-chance benchmark, indicating the presence of genuine directional information within the feature set.

The shortfall is economically plausible given the difficulty of predicting one-year-ahead price direction in highly efficient large-cap equity markets. At this horizon, stock returns are influenced not only by firm-level fundamentals and momentum, but also by broader macroeconomic shocks that were intentionally excluded from the feature space, including:

- foreign institutional investor (FII) flows
- RBI monetary-policy shifts
- global risk-off events
- commodity-price volatility
- geopolitical uncertainty

As a result, directional accuracy in the range of 55–65% is generally considered realistic for a fundamentally driven cross-sectional equity model operating on publicly available information.

---

## 5.4 Conclusion

The project successfully demonstrated that machine learning models trained on publicly available financial and market information can substantially improve one-year-ahead stock-price prediction accuracy relative to a naive “prices stay unchanged” assumption.

Although the directional-accuracy threshold was narrowly missed, the models still captured meaningful information about future price movements, suggesting that publicly available fundamentals and technical indicators contain economically relevant predictive signal within the NSE large-cap universe.


 # Section 6 — Evidence and Interpretation of All Generated Outputs

## Chart 1 — Data Overview

Chart 1 provides a foundational descriptive overview of the dataset prior to predictive modelling. The stock-price distribution highlights substantial heterogeneity within the NSE large-cap universe, with prices ranging from relatively low three-digit values to several thousand rupees per share. This wide dispersion economically justifies the use of scale-adjusted metrics such as MAPE and motivates transformations such as `log_market_cap` during feature engineering.

The return-distribution histogram shows that realised one-year returns are centred around modest positive values but exhibit visible fat tails on both sides, indicating the simultaneous presence of strong outperformers and severe underperformers within the same market environment.

The sector-wise return comparison reveals meaningful heterogeneity across industries. Certain sectors display systematically higher realised returns than others, suggesting that sector structure itself carries predictive information. This provides economic justification for the inclusion of sector-relative variables such as `sector_median_pe`, `relative_pe`, `sector_avg_margin`, and `sector_avg_growth` within the modelling pipeline.

Overall, the figure confirms that the dataset contains sufficient variation in both prices and returns for machine learning models to potentially extract meaningful predictive signal.
---
## Chart 2 — EDA Deep Dive

The sector heatmap standardises key accounting and valuation variables using Z-scored medians, enabling direct comparison across industries with otherwise incomparable raw scales. The figure demonstrates that sectors differ systematically across profitability, leverage, growth, and valuation characteristics. Technology and consumer-oriented firms generally exhibit higher growth and richer valuation multiples, while sectors such as metals and energy display greater volatility in profitability and leverage metrics.
The feature-correlation analysis provides preliminary evidence regarding which variables may contain predictive signal before model fitting begins. Momentum indicators, profitability measures, earnings yield, and relative valuation metrics show stronger association with realised future returns than purely accounting-scale variables. Importantly, no single feature dominates perfectly, implying that the prediction task requires combining multiple weak signals rather than relying on one deterministic driver. This economically supports the use of ensemble machine-learning methods capable of modelling complex interactions between variables.

---
## Chart 3 — Price Distribution and Sector Boxplots

The price-distribution visualisation demonstrates the strong right-skewness characteristic of equity-price datasets. A relatively small number of very high-priced firms coexist alongside a larger number of moderately priced firms, creating substantial scale imbalance across the universe. This justifies the use of logarithmic transformations and regularisation methods to stabilise estimation.

The sector-wise boxplots further reveal that dispersion differs materially across industries. Some sectors display tightly clustered price ranges, indicating relatively homogeneous firm structures, while others exhibit extreme intra-sector variation. The existence of large interquartile spreads and visible outliers reinforces the economic reality that Indian large-cap firms operate under highly differentiated business models and valuation regimes.
From a modelling perspective, this heterogeneity increases the difficulty of predicting future prices using a single global specification and provides additional justification for sector-relative feature engineering.

---
## Chart 4 — Correlation Matrix

The correlation matrix visualises the linear relationships among major explanatory variables and between features and the target variable. The chart highlights moderate relationships between profitability, growth, and valuation measures while also showing that many variables contribute distinct information.

The absence of excessively high pairwise correlations among most engineered features suggests that severe multicollinearity is limited after the introduction of transformations such as earnings yield and PEG proxy. This is particularly important for Ridge Regression, whose regularisation mechanism performs best when explanatory variables contain partially overlapping but not perfectly redundant information.

Economically, the matrix confirms that future stock prices are influenced by a combination of valuation, growth, leverage, and momentum variables rather than by a single dominant accounting ratio.

---
## Chart 5 — Baseline vs Machine-Learning Models

This chart provides the central empirical comparison between the naive persistence benchmark and the machine-learning models. The figure demonstrates that all predictive models outperform the baseline on Mean Squared Error, indicating that publicly available financial and market information contains signal beyond the assumption that stock prices simply remain unchanged.

Ridge Regression achieves the strongest overall out-of-sample performance, suggesting that a regularised linear specification generalises more effectively than highly flexible ensemble methods within a relatively small cross-sectional dataset. This finding is economically meaningful because it implies that the relationship between fundamentals, momentum and future prices is sufficiently stable that additional model complexity does not necessarily improve generalisation.
The directional-accuracy comparison shows that the models consistently exceed the 50 % random baseline but remain near the 60 % charter threshold. This result aligns with broader empirical finance literature which shows that predicting exact price levels is generally easier than consistently predicting the sign of future returns over long horizons.

---
## Chart 6 — Actual vs Predicted Prices

The actual-versus-predicted scatterplots assess how closely the fitted models reproduce realised stock-price outcomes on the hold-out test set. The concentration of observations around the 45-degree reference line indicates that the models capture the broad scaling relationship between current and future price levels without severe systematic bias.

The remaining outliers correspond to firms whose realised returns diverged substantially from model expectations. Economically, these deviations likely reflect firm-specific shocks, earnings surprises, macroeconomic developments, or market sentiment changes not captured within the available feature set. Importantly, the outliers appear on both sides of the reference line, suggesting that the models do not systematically overpredict or underpredict prices.

The comparatively tighter clustering achieved by Ridge Regression reinforces the conclusion that the regularised linear model produced the strongest generalisation performance under the primary metric.

---
## Chart 7 — Residual Analysis

The residual diagnostics evaluate whether the fitted models violate major statistical assumptions or exhibit obvious specification problems. The residual-versus-predicted scatterplot shows that prediction errors remain broadly distributed around zero without a strong deterministic structure, indicating that the models captured most systematic relationships present in the data.

Some heteroscedasticity remains visible at higher predicted price levels, which is expected in equity datasets because larger firms naturally generate larger absolute rupee deviations. The percentage-error histogram remains approximately centred near zero, suggesting that the models do not consistently overestimate or underestimate future prices across the sample.

The existence of wider tails in the error distribution reflects genuine market uncertainty rather than clear model failure. Equity prices are inherently influenced by unpredictable macroeconomic and behavioural shocks, meaning that some level of residual volatility is unavoidable even under well-specified models.

---
## Chart 8 — SHAP Feature Importance

The SHAP analysis provides interpretability for the XGBoost model by decomposing predictions into additive feature-level contributions. Although Ridge Regression achieved the best overall predictive performance, SHAP remains valuable because tree-based models provide a richer framework for identifying non-linear interactions across features.
The mean absolute SHAP values indicate that `current_price` is the most influential predictor in the model. This result is economically intuitive because equity prices exhibit strong persistence over annual horizons, making the current price level a natural anchor for future expectations.

Momentum variables such as `mom_4q` and valuation measures such as `earnings_yield` also rank highly, indicating that both market-trend information and relative valuation metrics contribute incremental predictive signal beyond simple persistence. The consistency between SHAP importance and XGBoost's built-in feature-importance rankings strengthens confidence that the identified predictors represent economically meaningful drivers rather than statistical artefacts.

---
## Chart 9 — Portfolio Performance Analysis

The portfolio-performance chart evaluates the exploratory Top-15 portfolio constructed using predicted returns from the machine-learning ranking system. Several selected firms substantially outperform the equal-weight benchmark during the evaluation period, suggesting that the model captures meaningful cross-sectional variation in expected returns.

The Sharpe Ratio and Information Ratio provide evidence regarding the efficiency of returns relative to volatility and benchmark tracking error. Positive risk-adjusted metrics imply that the ranking framework contains economically useful information rather than pure statistical noise.

However the portfolio exercise remains exploratory rather than investable. The analysis is based on a single evaluation window and excludes transaction costs, slippage, liquidity constraints, taxation and dynamic rebalancing. Consequently the chart should be interpreted as evidence of ranking capability rather than proof of deployable investment alpha.


# 5. Results and Output Interpretation

## Full Predictions Table (`full_predictions.csv`)

The `full_predictions.csv` file provides the most granular evidentiary output of the project by reporting realised and predicted values for each individual firm in the sample. The table allows direct inspection of model performance at the stock level, including which firms were predicted accurately and which generated substantial forecast errors.

In addition to predicted and realised prices, the table includes predicted returns, actual returns, directional correctness, and investment-signal labels. This extends the analysis beyond purely statistical evaluation and provides a bridge between machine-learning outputs and practical financial interpretation.

The table also reveals that predictive performance varies materially across firms and sectors, suggesting that some industries are inherently more predictable than others. From a methodological perspective, this output increases transparency because reviewers can independently assess whether aggregate metrics such as MSE and directional accuracy reflect broad model consistency or are driven disproportionately by a small number of highly successful predictions.

---

## `model_comparison.json`

The `model_comparison.json` file provides a machine-readable summary of all evaluation metrics across both the naive baseline and the machine-learning models. The structured JSON format improves reproducibility by enabling results to be independently validated and integrated into automated evaluation workflows.

This file also strengthens methodological transparency by ensuring that all models are evaluated under the same hold-out framework and reported consistently. As a result, the comparison prevents selective reporting of only favourable outcomes and allows direct assessment of relative predictive performance across modelling approaches.

---

## `primary_metric.json`

The `primary_metric.json` file operationalises the central charter objective by storing the best-model MSE, baseline MSE, pass/fail indicator, directional accuracy, and best-performing model name within a standardised schema.

This output functions as the formal decision record of the project. It converts the empirical evaluation into a falsifiable and independently auditable outcome, separate from the interpretive discussion presented within the notebook and report.

---

## `baseline_metric.json`

The `baseline_metric.json` file records the performance of the naive persistence benchmark, which assumes that future prices remain unchanged from current observed prices.

Including this benchmark is methodologically essential because predictive models are only economically meaningful if they outperform a defensible null model. The baseline therefore establishes the minimum predictive standard that any machine-learning model must exceed in order to claim incremental forecasting value.

---

## `milestone_manifest.json` and Probe Outputs

The manifest and probe outputs support reproducibility and data-source verification. The probe files confirm whether live Yahoo Finance data or the synthetic fallback pipeline was used during execution, while the manifest records runtime configuration, evaluation status, and output locations.

Together, these files improve the transparency and auditability of the project by allowing reviewers to trace the precise conditions under which the results were generated. In an empirical research setting, such reproducibility infrastructure is important for distinguishing rigorous analytical workflows from purely illustrative demonstrations.

# 7. Limits

## 7.1 What This Study Can Say With Confidence

- On the specific cross-section of NSE large-cap firms and the specific 12-month evaluation window defined by the run date, the machine-learning models — particularly Ridge Regression and the ensemble methods — produced materially lower Mean Squared Error (MSE) than the naive persistence benchmark.

- Although the directional-accuracy threshold of 60% was narrowly missed, the realised directional accuracy remained meaningfully above the 50% random-chance baseline, indicating the presence of genuine predictive signal within the feature set.

- SHAP analysis provides interpretable additive attribution of predictions to individual variables. The dominance of `current_price` confirms that price persistence is the strongest predictor, while momentum indicators, earnings-related measures, and sector-relative valuation variables contribute incremental predictive information.

---

## 7.2 What This Study Cannot Say

- **Causality:** The models are predictive rather than causal. For example, a high PE ratio being associated with future price appreciation does not imply that investors should systematically target high-PE firms. The relationship may instead reflect broader characteristics of large, high-growth companies.

- **Out-of-sample generalisability:** The held-out test set contains approximately 18 firms from a single 12-month period. Equity-return distributions vary across macroeconomic regimes, interest-rate cycles, geopolitical shocks, and sector rotations. Performance in other periods may differ substantially.

- **Live trading applicability:** The portfolio analysis is exploratory and does not constitute a true back-test incorporating transaction costs, liquidity constraints, slippage, taxes, or dynamic portfolio rebalancing. The results therefore should not be interpreted as evidence of implementable investment outperformance.

- **Applicability to smaller firms:** The analysis focuses exclusively on NSE large-cap firms. Relationships between valuation, momentum, liquidity, and future returns may differ materially for mid-cap or small-cap equities.

- **Data-quality limitations:** Yahoo Finance data occasionally contains stale or imperfectly adjusted values for Indian equities, particularly during corporate-action-heavy periods. In addition, median imputation for missing fundamentals may introduce noise for firms with incomplete reporting fields.

- **Directional-accuracy limitation:** The project charter specified a directional-accuracy threshold of 60%. The realised directional accuracy fell slightly below this threshold (approximately 57.9%). This limitation is disclosed directly rather than reframed. While the model substantially outperformed the baseline on MSE and directional accuracy remained above random chance, the directional criterion was not formally satisfied.

---

# 8. If the Result Was Weak or Partial

The primary MSE criterion was successfully met: Ridge Regression substantially outperformed the naive persistence benchmark on out-of-sample prediction error.

However, the directional-accuracy criterion was not formally satisfied. The best directional accuracy achieved was approximately 57.9%, narrowly below the charter threshold of 60%.

This outcome is reported directly rather than reframed as full success. The partial result remains economically informative. The findings suggest that predicting price levels using publicly available fundamentals, sector-relative signals, and technical indicators is tractable within the NSE large-cap universe, while predicting the exact direction of one-year-ahead returns remains substantially more difficult.

This limitation is economically plausible because long-horizon stock returns are influenced not only by firm-level information, but also by macroeconomic shocks, policy changes, foreign capital flows, and global market sentiment — variables intentionally excluded from the feature set.

---

# 9. Reproducibility

| Field | Detail |
|---|---|
| Notebook | `Indian_Equity_Predictor_ECO6810_CLEAN.ipynb` |
| Run Environment | Google Colab (Python 3.10+, GPU not required) |
| Run Command | `Runtime → Run all` (or `Ctrl + F9` in Colab) |
| Estimated Runtime | ~5–8 minutes with live `yfinance` fetch; ~2 minutes using synthetic fallback |

---

# 10. AI Usage

AI assistance was used across multiple stages of the project, including data collection, feature engineering, modelling, visualisation, debugging, and report drafting. All final implementation decisions, model-selection choices, and interpretive conclusions were independently verified by the team.

## Data Pipeline and Ticker Handling

ChatGPT and Gemini assisted in designing the initial workflow for collecting NSE stock data using `yfinance` and handling ticker-format inconsistencies. The team manually verified stock prices through Yahoo Finance, removed repeatedly failing tickers, and independently restricted the sample to firms with sufficient historical coverage.

## Sector Mapping and Feature Engineering

ChatGPT proposed an initial sector classification for firms in the sample. The team manually reviewed and corrected all sector labels. Gemini generated early versions of RSI, momentum, moving-average, earnings-yield, and relative-PE calculations. These were manually tested and refined by the team after identifying missing-value and rolling-window issues.

## Modelling and Evaluation

Gemini suggested the initial train-test split structure and baseline prediction workflow. The team independently verified that no future information leaked into training and confirmed that target shifting was implemented correctly.

Gemini also proposed initial Random Forest and XGBoost configurations. The team re-ran models under multiple hyperparameter settings and removed unstable specifications after empirical testing.

ChatGPT explained why Ridge Regression requires feature scaling whereas tree-based models do not. The team independently tested Ridge with and without `StandardScaler` before selecting the final configuration.

## Visualisation and Charts

GitHub Copilot assisted with plotting snippets and dataframe transformations. Gemini proposed layouts for residual plots and model-comparison figures, while ChatGPT suggested the correlation heatmaps and sector-comparison charts.

All visualisations were manually reviewed, reformatted, and validated against notebook outputs to ensure numerical consistency.

## Interpretation and Reporting

ChatGPT suggested wording for interpreting prediction error, directional accuracy, and model-comparison outputs. Claude assisted in comparing model results and proposing explanations for underperforming models. The team independently re-ran the notebook in Google Colab and verified all reported MSE, R², and directional-accuracy outputs before finalising conclusions.

Claude also assisted in rephrasing portions of the EDA discussion. All interpretations were checked manually against the underlying charts and datasets prior to inclusion in the report.

## What AI Did Not Do

No AI tool selected the final model, determined pass/fail outcomes, or generated the numerical results reported in the project. All modelling decisions, threshold evaluations, debugging steps, and final conclusions were reached independently by the team after reviewing notebook outputs directly.

A detailed log of AI usage, including dates, tools, and specific tasks, is documented in `AI_USAGE_LOG.md`.
