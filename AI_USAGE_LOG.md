# AI Usage Log

Keep this short and honest.

The point is not to confess that you used AI. Of course you used AI. The point is to show where it helped, what you trusted, and what you checked yourself.

## Log

# AI Usage Disclosure

# 10. AI Usage

AI tools including ChatGPT, Gemini, Claude, and GitHub Copilot were used throughout different stages of the project for workflow suggestions, debugging support, report refinement, visualisation ideas, and implementation assistance. All final modelling decisions, evaluation metrics, chart interpretations, and conclusions were independently verified by the project team using notebook-generated outputs.

## AI Usage Log

| Date | Tool | What AI helped with | What we verified / completed ourselves |
|---|---|---|---|
| 2026-04-28 | ChatGPT | Suggested an initial workflow for collecting NSE stock data and organising the prediction pipeline | Reduced the project scope ourselves and selected only firms with sufficient historical data |
| 2026-04-29 | Gemini | Suggested methods for downloading NSE stock data using `yfinance` and handling ticker-format inconsistencies | Manually checked stock prices on Yahoo Finance and removed repeatedly failing tickers |
| 2026-04-29 | ChatGPT | Proposed an initial sector mapping for companies in the dataset | Reviewed company classifications manually and corrected incorrect sector labels |
| 2026-04-30 | Gemini | Generated a first version of RSI, momentum, and moving-average feature calculations | Tested outputs on sample firms and fixed rolling-calculation issues producing missing values |
| 2026-04-30 | Gemini | Suggested adding valuation-related features such as earnings yield and relative PE ratios | Removed features with excessive missing values after checking dataset coverage |
| 2026-05-01 | ChatGPT | Suggested exploratory chart ideas and correlation visualisations | Adjusted chart formatting manually and verified plotted values directly from the dataframe |
| 2026-05-01 | ChatGPT | Explained why Ridge Regression generally requires feature scaling while tree-based models usually do not | Tested Ridge Regression with and without scaling and selected `StandardScaler` after comparing results |
| 2026-05-02 | Gemini | Suggested a train-test split structure and baseline prediction workflow | Verified that the target column was shifted correctly and that no future information leaked into training |
| 2026-05-03 | Gemini | Suggested an initial pipeline for Random Forest and XGBoost model training | Re-ran models with different parameters and removed unstable settings after testing |
| 2026-05-03 | GitHub Copilot | Suggested plotting snippets and dataframe-transformation code during notebook development | Reviewed all generated code manually and corrected formatting issues in several plots |
| 2026-05-04 | ChatGPT | Suggested wording for explaining prediction error versus directional accuracy in the report | Manually checked prediction rows to confirm that explanations matched notebook outputs |
| 2026-05-05 | ChatGPT | Suggested README section ordering and repository documentation formatting | Rewrote sections that did not accurately reflect the notebook workflow |
| 2026-05-06 | Gemini | Suggested layouts for residual plots and model-comparison charts | Modified chart settings after identifying overlapping labels and unclear legends |
| 2026-05-07 | Gemini | Suggested fixes for package-version conflicts and notebook execution issues in Google Colab | Re-ran the notebook completely and verified that all outputs generated correctly |
| 2026-05-08 | ChatGPT | Suggested a comparison-table format for summarising model performance and limitations | Simplified explanations and removed unsupported claims after reviewing outputs |
| 2026-05-09 | Claude | Suggested possible reasons for underperforming prediction models and compared evaluation outputs across models | Re-ran notebooks independently in Google Colab, checked MSE, R², and directional-accuracy outputs, and confirmed the final model choice ourselves |
| 2026-05-10 | Claude | Rephrased parts of the exploratory data-analysis section discussing volatility trends, correlations, and sector comparisons | Verified that all graphs, tables, and statistical observations came directly from notebook outputs and datasets |
| 2026-05-11 | ChatGPT | Explained differences between evaluation metrics and suggested clearer wording for parts of the machine-learning workflow description | All final coding decisions, experimentation, debugging, and conclusions were independently reviewed after checking notebook outputs |
| 2026-05-14 | ChatGPT | Assisted in identifying inconsistencies between updated notebook outputs and report interpretations | Cross-checked all final MSE, R², directional-accuracy values, and chart interpretations manually before final submission |
| 2026-05-15 | ChatGPT | Assisted in refining limitation wording, conclusion consistency, and model-performance explanations after updated outputs | Independently verified that all report sections matched the latest notebook-generated results and corrected outdated claims manually |

## Things To Avoid

- "used AI for coding"
- "used ChatGPT for analysis"
- "AI helped with debugging"

Those lines are too vague to be useful.

## Better

- "Claude suggested the first version of the FRED fetch helper; we changed the parsing after checking the missing-value handling"
- "ChatGPT proposed a DiD specification; we kept the controls but rewrote the treatment definition after reading the data"
- "Gemini rewrote the report summary; we replaced two claims that overstated causality"
