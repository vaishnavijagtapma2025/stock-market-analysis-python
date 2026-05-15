from __future__ import annotations


def build_baseline_metric(baseline_mse: float) -> dict:
    return {
        "metric_name": "naive_persistence_mse",
        "value": round(float(baseline_mse), 4),
        "unit": "mse",
        "model": "Naive Persistence Baseline",
    }


def build_primary_metric(
    best_model_name: str,
    best_model_mse: float,
    baseline_mse: float,
    directional_accuracy: float,
) -> dict:

    return {
        "metric_name": "best_model_test_mse",
        "model_name": best_model_name,
        "value": round(float(best_model_mse), 4),
        "threshold": round(float(baseline_mse), 4),
        "passed": best_model_mse < baseline_mse,
        "directional_accuracy": round(float(directional_accuracy), 4),
    }


def build_milestone_manifest() -> dict:
    return {
        "charter_locked": True,
        "sources": [
            {
                "name": "Yahoo Finance",
                "status": "ok",
                "probe_artifact": "outputs/probe_output.md",
                "note": "Live stock and financial data fetched using yfinance.",
            }
        ],
        "baseline_ready": True,
        "primary_metric_schema_ready": True,
        "run_command": "uv run main.py",
    }
