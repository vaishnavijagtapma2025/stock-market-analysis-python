# main.py  —  Indian Equity Predictor ECO6810
# Run with: uv run main.py

import os, json, warnings
import numpy as np
import pandas as pd
import yfinance as yf
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from datetime import datetime, timedelta
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

warnings.filterwarnings("ignore")

os.makedirs("outputs", exist_ok=True)
os.makedirs("outputs/source_probes", exist_ok=True)
os.makedirs("data", exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 120, "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3
})
