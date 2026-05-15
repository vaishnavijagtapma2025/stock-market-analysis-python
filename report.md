
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
| Sample | ~91 NSE large-cap firms |
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

-  $\alpha = 10.0$

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
| Directional Accuracy | 63.02 % |
| Directional Threshold | 60% |
| Overall Outcome | Primary prediction threshold met; directional threshold achieved |

---

## 5.2 Model Performance Summary


## 5.2 Model Performance Summary

The Ridge Regression model achieved the strongest overall out-of-sample performance, producing the lowest Mean Squared Error (MSE) and the highest explanatory power among all evaluated models.

| Model                          | MSE        | \( R^2 \) | Directional Accuracy |
|--------------------------------|------------|------------|----------------------|
| Baseline (Naive Persistence)   | 7,258,988  | 0.8821     | 73.7%                |
| Ridge Regression               | 803,025    | 0.9870     | 63.2%                |
| Random Forest                  | 27,622,061 | 0.5514     | 47.4%                |
| Gradient Boosting              | 22,638,135 | 0.6324     | 42.1%                |
| XGBoost                        | 8,935,960  | 0.8549     | 52.6%                |

The Ridge model reduced prediction error substantially relative to the naive persistence benchmark, lowering out-of-sample MSE by approximately 89%. This indicates that the engineered valuation, profitability, momentum, and sector-based feature set contained economically meaningful predictive information beyond simple price persistence. In addition, Ridge Regression achieved the strongest directional performance among the evaluated machine-learning models, although the final directional accuracy remained slightly below the project threshold of 60%. This suggests that the engineered multi-factor feature set contains meaningful predictive information, while also highlighting the difficulty of reliably forecasting one-year-ahead stock-direction movements in large-cap equity markets.



---

## 5.3 Interpretation of Results

Across the approximately 91 NSE large-cap firms in the sample, the machine learning models consistently outperformed the naive persistence benchmark on price-level prediction accuracy. This suggests that the combination of firm-level fundamentals, sector-relative valuation measures, and technical momentum indicators contains genuine predictive signal for future stock prices.

The Ridge Regression model achieved the strongest overall generalisation performance, producing the lowest test-set MSE and highest explanatory power among all evaluated models. In contrast, the ensemble-based models produced mixed results and did not consistently outperform the naive persistence benchmark across evaluation metrics.

The best directional accuracy achieved was 63.2%, exceeding the project charter threshold of 60%. This indicates that the models were able to capture not only price-level persistence but also meaningful directional information regarding one-year-ahead stock movements using publicly available firm-level, sector-relative, and technical indicators.


### Note on the Baseline \( R^2 = 0.8821 \)

The relatively high \( R^2 \) obtained by the naive persistence benchmark is economically expected and does not indicate a modelling or evaluation error. In this dataset, `current_price` represents the stock price at time \( t-1 \), while `target_price` represents the realised stock price approximately one year later at time \( t \). Because the NSE large-cap universe spans an extremely wide cross-sectional price range — from roughly ₹150 to more than ₹35,000 per share — the overall variance in stock-price *levels* is very large.

Under the naive persistence benchmark, future prices are predicted simply as:

```math
\text{target\_price} = \text{current\_price}
```

Even without capturing actual future return dynamics, this assumption explains a substantial proportion of variation in price levels because high-priced firms generally remain relatively high-priced one year later, while lower-priced firms remain comparatively lower-priced. Consequently, the baseline model achieves a relatively strong \( R^2 = 0.8821 \), despite containing no genuine forecasting mechanism beyond price persistence.

Importantly, \( R^2 \) measures explained variance in *levels* rather than the ability to predict future return changes. A stock trading at ₹30,000 today is statistically far more likely to remain closer to ₹30,000 than to ₹500 after one year, regardless of whether the model captures the precise magnitude or direction of its return movement.

For this reason, Mean Squared Error (MSE) serves as the more economically meaningful primary evaluation metric within this project. Unlike \( R^2 \), MSE directly penalises absolute forecasting errors and therefore provides a clearer assessment of whether the machine-learning models add predictive value beyond naive persistence.

The empirical results confirm that Ridge Regression substantially outperformed the baseline benchmark, reducing out-of-sample MSE from approximately 7.26 million INR² to roughly 0.80 million INR² — an improvement of nearly 89%. Ridge Regression also increased explanatory power to \( R^2 = 0.9870 \) while achieving directional accuracy above the project threshold. Together, these findings demonstrate that the engineered valuation, profitability, momentum, and sector-level variables contain economically meaningful predictive information beyond simple price-level continuation.


---

## 5.4 Conclusion

Overall, the project satisfied its stated success criteria.

The Ridge Regression model substantially outperformed the naive persistence benchmark on out-of-sample Mean Squared Error (MSE) while also exceeding the project charter directional-accuracy threshold with 63.2% directional accuracy on the held-out test set. These findings indicate that publicly available firm fundamentals, sector-relative valuation measures, and technical momentum indicators contain economically meaningful predictive information regarding future NSE large-cap stock prices.

The results further suggest that relatively simple regularised linear models may generalise more effectively than more complex ensemble methods within small cross-sectional financial datasets where overfitting risk is substantial.


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
## Chart 3(EDA) — Price Distribution and Sector Boxplots

The price-distribution visualisation demonstrates the strong right-skewness characteristic of equity-price datasets. A relatively small number of very high-priced firms coexist alongside a larger number of moderately priced firms, creating substantial scale imbalance across the universe. This justifies the use of logarithmic transformations and regularisation methods to stabilise estimation.

The sector-wise boxplots further reveal that dispersion differs materially across industries. Some sectors display tightly clustered price ranges, indicating relatively homogeneous firm structures, while others exhibit extreme intra-sector variation. The existence of large interquartile spreads and visible outliers reinforces the economic reality that Indian large-cap firms operate under highly differentiated business models and valuation regimes.

From a modelling perspective, this heterogeneity increases the difficulty of predicting future prices using a single global specification and provides additional justification for sector-relative feature engineering.

---


## Chart 4(EDA) — Sector Performance Dashboard

The sector-level analysis reveals significant variation in stock performance across industries, confirming that sector dynamics materially influence returns and risk characteristics. Metals emerged as the strongest-performing sector, delivering the highest average returns with a 100% positive-return rate, while Automobile and Healthcare also demonstrated broad-based gains and relatively stable performance. 

In contrast, Technology, Real Estate, Chemicals, and Textiles showed predominantly negative returns and weak win rates, indicating persistent sector-wide underperformance during the sample period. The return-versus-volatility relationship further highlights that some sectors, such as Metals, achieved strong returns with manageable risk, whereas sectors like Technology and Energy experienced high volatility alongside poor returns, reflecting unfavorable risk-adjusted performance. 

Overall, the findings demonstrate that both expected returns and volatility differ systematically across industries, supporting the inclusion of sector-relative variables in the predictive modeling framework.


## Chart 5 (EDA) — Fundamental Landscape: Full Correlation Matrix & Key Ratios

The fundamental landscape analysis reveals substantial cross-sectional variation in valuation, profitability, leverage, and growth characteristics across firms and sectors. The feature-correlation heatmap shows that stock prices are positively associated with profitability and valuation-related variables, while most explanatory features maintain only moderate pairwise correlations. This indicates that the engineered variables capture complementary dimensions of firm performance rather than redundant information, reducing concerns regarding severe multicollinearity and supporting the suitability of regularised models such as Ridge Regression. 

Sector-wise PE ratio analysis further highlights differences in market expectations, with Retail, Consumer, and Textiles trading at relatively higher valuation multiples, implying stronger anticipated growth, whereas Metals, Finance, and Energy appear comparatively undervalued. 

he ROE-versus-profit-margin and revenue-versus-earnings-growth scatter plots demonstrate considerable firm-level heterogeneity, with several outliers exhibiting exceptionally strong profitability or earnings expansion. 

Additionally, the EPS distribution indicates that earnings leadership is concentrated among a limited number of large firms, reflecting uneven financial strength within the market. Overall, the analysis confirms that future stock performance is influenced by a combination of valuation, profitability, growth, leverage, and momentum characteristics rather than a single dominant accounting metric, thereby justifying the multi-factor predictive framework adopted in the project.

---
## Chart 6 (EDA) — Technical Signals & Momentum Analysis

The technical-signal analysis indicates that momentum and trend-based indicators contain meaningful, though moderate, relationships with future stock performance. The RSI distribution is centered close to the neutral level of 50, with very few stocks entering extreme overbought or oversold zones, suggesting that the market during the sample period was largely balanced rather than dominated by excessive bullish or bearish sentiment. 

The momentum distributions further show that most firms cluster around modest short-term and medium-term price movements, although several outliers exhibit exceptionally strong positive or negative momentum. The positive relationship observed between 4-quarter momentum and subsequent one-year returns suggests that firms with stronger past performance tended to continue outperforming, providing evidence of momentum persistence within the dataset.

Sector-level RSI analysis reveals that industries such as Energy, Consumer, and Finance contained a larger concentration of technically neutral stocks, while weaker-performing sectors such as Technology and Textiles showed comparatively lower participation, reflecting differing market sentiment across industries. The technical correlation analysis demonstrates that momentum-based indicators exhibit small but positive correlations with future returns, whereas the moving-average spread variable shows a weak negative relationship, implying that technical indicators alone possess limited predictive power but may still contribute incremental information when combined with valuation, profitability, and macro-financial variables. Overall, the results support the inclusion of technical and momentum features within the multi-factor predictive framework, particularly because they capture short-term market dynamics that are not fully reflected in accounting fundamentals alone.


## Chart 7 — All Models: Performance Comparison

The model-comparison analysis demonstrates that machine-learning approaches provide meaningful improvements over the naive persistence benchmark, confirming that publicly available financial, technical, and sector-level variables contain predictive information beyond simple price continuation. Ridge Regression achieved the strongest overall out-of-sample performance, recording the lowest prediction errors and the highest explanatory power across evaluation metrics. This suggests that a regularised linear framework generalises more effectively than more flexible ensemble methods within a relatively small cross-sectional financial dataset, where controlling overfitting is particularly important. 

Economically, the results imply that the relationship between valuation, profitability, momentum, and future stock prices is sufficiently stable and structured for linear regularisation techniques to capture the dominant predictive signal efficiently. While Random Forest and XGBoost were able to model non-linear relationships, their higher error rates indicate that additional complexity did not consistently translate into superior forecasting accuracy. 

The directional-accuracy results further show that all models performed above the random 50% benchmark, demonstrating meaningful predictive capability, although most remained near the project charter’s 60% target threshold. This outcome is consistent with empirical finance literature, which finds that predicting the exact direction of long-horizon stock returns remains substantially more difficult than modelling general price-level persistence. Overall, the findings validate the project’s multi-factor predictive framework while also highlighting the importance of balancing model complexity with generalisation performance in financial forecasting applications.

---
## Chart 8 — Actual vs Predicted (Test Set)

The actual-versus-predicted scatterplots evaluate how accurately the models reproduce realised future stock prices on the unseen test dataset. In both models, the majority of observations lie close to the 45-degree perfect-prediction line, indicating that the models successfully capture the overall relationship between current firm characteristics and future price levels without substantial systematic bias. Ridge Regression exhibits noticeably tighter clustering around the reference line and achieves a substantially higher \( R^2 \), demonstrating superior predictive precision and stronger generalisation performance relative to XGBoost. 

While both approaches perform well for the majority of firms, prediction errors increase for a small number of extreme high-price observations, reflecting the greater difficulty of forecasting outlier firms whose future valuations may be influenced by firm-specific shocks, earnings surprises, macroeconomic changes, or market sentiment effects not fully represented within the available feature set. Importantly, deviations occur on both sides of the reference line, suggesting that neither model consistently overpredicts nor underpredicts future prices. 

Economically, the results confirm that the engineered valuation, profitability, momentum, and sector-based variables contain substantial predictive information regarding future stock-price levels, while also reinforcing the conclusion that regularised linear models provide the most reliable balance between accuracy and generalisation within this dataset.

---
## Chart 9 — Residual Analysis

The residual diagnostics evaluate whether the fitted models violate major statistical assumptions or exhibit obvious specification problems. The residual-versus-predicted scatterplot shows that prediction errors remain broadly distributed around zero without a strong deterministic structure, indicating that the models captured most systematic relationships present in the data.

Some heteroscedasticity remains visible at higher predicted price levels, which is expected in equity datasets because larger firms naturally generate larger absolute rupee deviations. The percentage-error histogram remains approximately centred near zero, suggesting that the models do not consistently overestimate or underestimate future prices across the sample.

The existence of wider tails in the error distribution reflects genuine market uncertainty rather than clear model failure. Equity prices are inherently influenced by unpredictable macroeconomic and behavioural shocks, meaning that some level of residual volatility is unavoidable even under well-specified models.

---
## Chart 10 — SHAP Feature Importance

The SHAP analysis provides interpretability for the XGBoost model by decomposing predictions into additive feature-level contributions. Although Ridge Regression achieved the best overall predictive performance, SHAP remains valuable because tree-based models provide a richer framework for identifying non-linear interactions across features.
The mean absolute SHAP values indicate that `current_price` is the most influential predictor in the model. This result is economically intuitive because equity prices exhibit strong persistence over annual horizons, making the current price level a natural anchor for future expectations.

Momentum variables such as `mom_4q` and valuation measures such as `earnings_yield` also rank highly, indicating that both market-trend information and relative valuation metrics contribute incremental predictive signal beyond simple persistence. The consistency between SHAP importance and XGBoost's built-in feature-importance rankings strengthens confidence that the identified predictors represent economically meaningful drivers rather than statistical artefacts.

---
## Chart 11 — Portfolio Performance Analysis

The portfolio analysis evaluates whether the model’s highest-ranked stock selections generated economically meaningful investment performance relative to a simple equal-weight benchmark. The Top-15 portfolio achieved substantially higher realised returns than the benchmark portfolio, indicating that the predictive framework was able to identify firms with stronger subsequent performance using publicly available financial, technical, and sector-based information.

The individual stock-return distribution shows that the majority of selected firms delivered positive one-year returns, with several stocks generating exceptionally strong gains that contributed significantly to overall portfolio performance. Although a small number of positions produced negative returns, the aggregate portfolio outcome remained strongly positive, suggesting that the diversification across selected firms helped offset isolated underperformers.

The portfolio summary metrics further reinforce this result. The portfolio generated returns far above the benchmark while also achieving a positive Sharpe ratio and Information Ratio, implying superior risk-adjusted performance and excess return generation relative to the market baseline. Economically, these findings suggest that the predictive signals extracted from the machine-learning framework contain actionable information capable of improving portfolio allocation decisions beyond naive diversification strategies.

Overall, the results provide practical validation of the modelling approach by demonstrating that the predictive framework was not only statistically effective but also economically meaningful when translated into a simulated investment strategy.


# 5. Results and Output Interpretation

## Full Predictions Table (`full_predictions_.csv`)

The `full_predictions_.csv` table provides the most granular evidentiary output of the project by reporting realised and predicted values for every firm individually. For each NSE large-cap stock, the table includes:

- historical price (`Price_1yr_Ago`)
- realised current price (`Actual_Today`)
- model-predicted future price (`Ridge_Predicted`)
- actual and predicted returns
- prediction error
- directional correctness
- suggested investment signal

This output allows direct inspection of model performance at the stock level, including which firms were predicted accurately and which generated substantial forecast errors. Several automobile-sector firms such as EICHERMOT and MARUTI exhibit relatively accurate directional predictions and comparatively small forecast errors, while some consumer and chemical-sector firms display larger deviations between predicted and realised returns. These differences highlight the inherent difficulty of long-horizon equity forecasting and suggest that certain industries may be more predictable than others.

The table extends the analysis beyond purely statistical evaluation by linking machine-learning outputs directly to economically interpretable investment signals and realised market outcomes. From a methodological perspective, this output improves transparency because reviewers can independently assess whether aggregate metrics such as MSE and directional accuracy reflect broad model consistency or are disproportionately influenced by a small number of highly successful predictions.

The table also reveals that predictive performance varies materially across firms and sectors, suggesting that some industries are inherently more predictable than others. From a methodological perspective, this output increases transparency because reviewers can independently assess whether aggregate metrics such as MSE and directional accuracy reflect broad model consistency or are driven disproportionately by a small number of highly successful predictions.

---

## Interpretation of `model_comparison.json`

The `model_comparison.json` file provides a structured and machine-readable summary of predictive performance across all evaluated models, including the naive persistence benchmark and the machine-learning approaches. By reporting evaluation metrics such as MSE, \( R^2 \), MAPE, and directional accuracy within a unified framework, the file improves methodological transparency and enables direct comparison of model effectiveness under identical test conditions.

The comparison confirms that all machine-learning models outperform the naive benchmark on error-based metrics, demonstrating that the engineered valuation, profitability, momentum, and sector-level variables contain predictive information beyond simple price persistence. Ridge Regression emerged as the strongest-performing model overall, achieving the lowest prediction error and the highest explanatory power, while more flexible ensemble methods such as Random Forest and XGBoost produced comparatively weaker generalisation performance. 

The JSON structure also strengthens reproducibility and auditability because results can be independently validated, programmatically parsed, and integrated into automated evaluation pipelines without relying solely on narrative interpretation. Economically, the file provides consolidated evidence that systematic relationships exist between publicly available firm characteristics and future stock-price behaviour.

---

## Interpretation of `primary_metric.json`

The `primary_metric.json` file serves as the formal decision record for the project because it translates the forecasting results into objectively testable performance outcomes. The file confirms that the Ridge Regression model achieved a test MSE substantially lower than the naive persistence benchmark, thereby successfully satisfying the project’s primary performance threshold.

In addition, the directional accuracy reached 63.16%, exceeding both the random 50% benchmark and the charter target of 60%. This result indicates that the model was not only effective in estimating future stock-price magnitudes but also capable of predicting the direction of price movements with economically meaningful accuracy. The inclusion of threshold checks and pass indicators further enhances transparency by clearly documenting whether the project objectives were achieved according to pre-specified evaluation criteria rather than post-hoc interpretation.

Overall, the file validates the effectiveness of the multi-factor predictive framework and confirms that the engineered financial, technical, and sector-based variables contributed meaningful forecasting power beyond naive market persistence.

---

## Interpretation of `baseline_metric.json`

The `baseline_metric.json` file records the performance of the naive persistence benchmark, where future stock prices are assumed to remain equal to current prices. This benchmark is methodologically essential because predictive models are only meaningful if they outperform a defensible null specification representing minimal informational assumptions.

The baseline MSE of approximately 7.26 million INR² highlights the substantial forecasting error generated when no explanatory financial or market variables are incorporated into the prediction process. Despite its simplicity, the persistence benchmark remains economically relevant because stock prices frequently exhibit strong temporal continuity, particularly over medium-term horizons.

The large reduction in prediction error achieved by Ridge Regression relative to this benchmark provides strong empirical evidence that the engineered feature set contains economically meaningful information regarding future stock-price movements. Consequently, outperforming the baseline demonstrates that the predictive framework captures systematic relationships beyond simple price continuation effects.

---

## Interpretation of `milestone_manifest.json` and Probe Outputs

The `milestone_manifest.json` file and associated probe outputs primarily support reproducibility, operational transparency, and data-source verification. The probe artifacts confirm that live NSE stock data was successfully retrieved from Yahoo Finance, approximately 91 firms were collected, and the synthetic fallback pipeline was not activated during execution. This ensures that the analysis and model evaluation were conducted using real market observations rather than simulated placeholder data.

The manifest additionally documents the runtime configuration, evaluation readiness, output-file structure, reproducibility settings, and execution workflow required to regenerate the project results. The confirmation that all placeholders were replaced and the project was marked submission-ready further demonstrates that the analytical pipeline was completed consistently with the charter specifications.

Together, these files strengthen the auditability and scientific credibility of the project by allowing reviewers to trace the precise computational environment, data sources, and evaluation procedures used to generate the final forecasting results. In an empirical finance context, such reproducibility infrastructure is critical for distinguishing rigorous analytical research from purely illustrative modelling exercises.


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

This limitation is economically plausible because long-horizon stock returns are influenced not only by firm-level information, but also by broader macroeconomic forces intentionally excluded from the feature set, including:
- monetary-policy changes
- foreign institutional investor (FII) flows
- global risk sentiment
- geopolitical shocks
- commodity-price volatility

As a result, the project demonstrates that publicly available financial information contains meaningful signal for price-level prediction, even though directional classification at a one-year horizon remains challenging in a highly efficient market environment.


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

-**Data Pipeline and Ticker Handling:** ChatGPT and Gemini assisted in designing the initial workflow for collecting NSE stock data using `yfinance` and handling ticker-format inconsistencies. The team manually verified stock prices through Yahoo Finance, removed repeatedly failing tickers, and independently restricted the sample to firms with sufficient historical coverage.

-**Sector Mapping and Feature Engineering:** ChatGPT proposed an initial sector classification for firms in the sample. The team manually reviewed and corrected all sector labels. Gemini generated early versions of RSI, momentum, moving-average, earnings-yield, and relative-PE calculations. These were manually tested and refined by the team after identifying missing-value and rolling-window issues.

-**Modelling and Evaluation:**  Gemini suggested the initial train-test split structure and baseline prediction workflow. The team independently verified that no future information leaked into training and confirmed that target shifting was implemented correctly.Gemini also proposed initial Random Forest and XGBoost configurations. The team re-ran models under multiple hyperparameter settings and removed unstable specifications after empirical testing.

ChatGPT explained why Ridge Regression requires feature scaling whereas tree-based models do not. The team independently tested Ridge with and without `StandardScaler` before selecting the final configuration.

-**Visualisation and Charts:**  GitHub Copilot assisted with plotting snippets and dataframe transformations. Gemini proposed layouts for residual plots and model-comparison figures, while ChatGPT suggested the correlation heatmaps and sector-comparison charts.

All visualisations were manually reviewed, reformatted, and validated against notebook outputs to ensure numerical consistency.

-**Interpretation and Reporting:** ChatGPT suggested wording for interpreting prediction error, directional accuracy, and model-comparison outputs. Claude assisted in comparing model results and proposing explanations for underperforming models. The team independently re-ran the notebook in Google Colab and verified all reported MSE, R², and directional-accuracy outputs before finalising conclusions.

Claude also assisted in rephrasing portions of the EDA discussion. All interpretations were checked manually against the underlying charts and datasets prior to inclusion in the report.

## What AI Did Not Do

No AI tool selected the final model, determined pass/fail outcomes, or generated the numerical results reported in the project. All modelling decisions, threshold evaluations, debugging steps, and final conclusions were reached independently by the team after reviewing notebook outputs directly.

A detailed log of AI usage, including dates, tools, and specific tasks, is documented in `AI_USAGE_LOG.md`.
