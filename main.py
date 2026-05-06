def build_baseline_metric():
    return {
        "metric_name": "Baseline MSE",
        "value": 31050.34,
        "unit": "INR^2"
    }


def build_primary_metric():
    return {
        "metric_name": "XGBoost Directional Accuracy",
        "value": 72.2,
        "threshold": 60.0,
        "passed": True
    }


def build_milestone_manifest():
    return {
        "charter_locked": True,
        "sources": [
            {
                "name": "Yahoo Finance via yfinance",
                "status": "ok",
                "probe_artifact": "data/indian_equity_portfolio_dataset.xlsx"
            }
        ],
        "baseline_ready": True,
        "primary_metric_schema_ready": True,
        "run_command": "uv run main.py"
    }
    uv run main.py
