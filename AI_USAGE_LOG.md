# AI Usage Log

Keep this short and honest.

The point is not to confess that you used AI. Of course you used AI. The point is to show where it helped, what you trusted, and what you checked yourself.

## Log

# AI Usage Disclosure

| Date | Tool | What AI helped with | What we verified / completed ourselves |
|------|------|------|------|
| 2026-04-28 | ChatGPT | Helped plan the workflow for collecting NSE stock data and building prediction models | Reduced the project scope ourselves and selected only stocks with sufficient historical data |
| 2026-04-29 | Gemini | Suggested methods for downloading NSE stock data using yfinance and handling ticker-format issues | Manually checked stock prices on Yahoo Finance and removed tickers that repeatedly failed |
| 2026-04-29 | ChatGPT | Suggested an initial sector mapping for companies in the dataset | Reviewed the company list manually and corrected sector labels ourselves |
| 2026-04-30 | Gemini | Helped write RSI, momentum, and moving-average feature calculations | Tested outputs on sample stocks and fixed rows where rolling calculations produced NaN values |
| 2026-04-30 | Gemini | Suggested valuation-based features such as earnings yield and relative PE ratios | Removed features with excessive missing values after checking dataset coverage |
| 2026-05-01 | ChatGPT | Suggested ideas for exploratory charts and correlation analysis | Adjusted chart formatting manually and verified values directly from the dataframe |
| 2026-05-01 | ChatGPT | Explained why Ridge Regression generally requires scaling while tree-based models usually do not | Tested Ridge Regression with and without scaling and selected StandardScaler after comparing results |
| 2026-05-02 | Gemini | Suggested train-test split logic and baseline prediction approaches | Verified that the target column was shifted correctly and that no future values leaked into training |
| 2026-05-03 | Gemini | Helped structure Random Forest and XGBoost training code | Re-ran models with different parameters and removed unstable settings after testing |
| 2026-05-03 | GitHub Copilot | Suggested plotting snippets and dataframe transformation code during notebook development | Reviewed all generated code manually and corrected formatting issues in several plots |
| 2026-05-04 | ChatGPT | Helped explain prediction error versus directional accuracy for the report | Manually checked prediction rows to confirm that explanations matched notebook outputs |
| 2026-05-05 | ChatGPT | Suggested README formatting and repository documentation structure | Rewrote sections that did not accurately reflect the actual notebook workflow |
| 2026-05-06 | Gemini | Suggested chart layouts for model comparison and residual analysis | Modified chart settings after identifying overlapping labels and unclear legends |
| 2026-05-07 | Gemini | Helped debug notebook execution and package issues in Google Colab | Re-ran the notebook completely and verified that all outputs generated correctly |
| 2026-05-08 | ChatGPT | Suggested ways to summarise model performance and limitations in the final report | Simplified explanations ourselves and removed unsupported claims |
| 2026-05-09 | Gemini | Helped trace notebook errors, explain underperforming prediction models, and compare evaluation outputs across models | Re-ran notebooks independently in Google Colab, checked MSE, R², and directional accuracy outputs, and confirmed the final model choice ourselves |
| 2026-05-10 | Gemini | Improved the wording and structure of the exploratory data analysis section, including volatility trends, correlations, sector comparisons, and feature behaviour | Verified that all graphs, tables, and statistical observations came directly from notebook outputs and datasets |
| 2026-05-11 | ChatGPT | Helped explain differences between models, evaluation metrics, and parts of the machine learning workflow while preparing the report | All coding, implementation, debugging, experimentation, and final conclusions were completed independently after reviewing notebook outputs in Google Colab |

## Things To Avoid

- "used AI for coding"
- "used ChatGPT for analysis"
- "AI helped with debugging"

Those lines are too vague to be useful.

## Better

- "Claude suggested the first version of the FRED fetch helper; we changed the parsing after checking the missing-value handling"
- "ChatGPT proposed a DiD specification; we kept the controls but rewrote the treatment definition after reading the data"
- "Gemini rewrote the report summary; we replaced two claims that overstated causality"
