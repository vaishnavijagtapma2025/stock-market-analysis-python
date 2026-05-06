!pip install yfinance xgboost shap --quiet
print('✅ All packages installed.')

import pandas as pd
import numpy as np
import yfinance as yf
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
import tensorflow as tf

from datetime import timedelta, datetime
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from IPython.display import display

warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

# Matplotlib settings for clean, readable charts
plt.rcParams.update({
    'figure.dpi': 120,
    'font.size': 11,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3
})

print('✅ Imports done.')

# ── 90 Large-Cap Indian Tickers ──────────────────────────────────────────────
TICKERS = [
    'RELIANCE.NS','ONGC.NS','TCS.NS','INFY.NS','WIPRO.NS','HCLTECH.NS','TECHM.NS',
    'HDFCBANK.NS','ICICIBANK.NS','SBIN.NS','KOTAKBANK.NS','AXISBANK.NS','BAJFINANCE.NS',
    'HINDUNILVR.NS','ITC.NS','NESTLEIND.NS','BRITANNIA.NS','TATAMOTORS.NS',
    'MARUTI.NS','M&M.NS','EICHERMOT.NS','HEROMOTOCO.NS',
    'SUNPHARMA.NS','DRREDDY.NS','CIPLA.NS','APOLLOHOSP.NS',
    'BHARTIARTL.NS','LT.NS','TITAN.NS','ASIANPAINT.NS','ULTRACEMCO.NS',
    'PIDILITIND.NS','SRF.NS','UPL.NS','DLF.NS','GODREJPROP.NS',
    'GRASIM.NS','TRENT.NS','DMART.NS','ADANIPORTS.NS','GAIL.NS','IOC.NS',
    'BPCL.NS','ZOMATO.NS','PAYTM.NS','BEL.NS','HAL.NS','VBL.NS',
    'SHREECEM.NS','HINDZINC.NS','VEDL.NS','AMBUJACEM.NS','ACC.NS',
    'TATACOMM.NS','INDHOTEL.NS','PAGEIND.NS','COLPAL.NS','DABUR.NS',
    'MARICO.NS','BERGEPAINT.NS','MUTHOOTFIN.NS','CHOLAFIN.NS',
    'BANDHANBNK.NS','BIOCON.NS','LUPIN.NS','AUROPHARMA.NS',
    'MPHASIS.NS','COFORGE.NS','PERSISTENT.NS','LTTS.NS','DIXON.NS',
    'POLYCAB.NS','KEI.NS','HAVELLS.NS','VOLTAS.NS','CONCOR.NS',
    'PETRONET.NS','IGL.NS','MGL.NS','GUJGASLTD.NS','RECLTD.NS',
    'PFC.NS','IRFC.NS','RVNL.NS','IRCON.NS','BAJAJFINSV.NS',
    'JSWSTEEL.NS','TATASTEEL.NS','HINDALCO.NS','COALINDIA.NS','NTPC.NS'
]

SECTOR_MAP = {
    'RELIANCE.NS':'Energy','ONGC.NS':'Energy','LT.NS':'Energy','CONCOR.NS':'Energy',
    'PETRONET.NS':'Energy','IGL.NS':'Energy','MGL.NS':'Energy','GUJGASLTD.NS':'Energy',
    'GAIL.NS':'Energy','IOC.NS':'Energy','BPCL.NS':'Energy','ADANIPORTS.NS':'Energy',
    'ULTRACEMCO.NS':'Energy','SHREECEM.NS':'Energy','AMBUJACEM.NS':'Energy',
    'ACC.NS':'Energy','HINDZINC.NS':'Energy','VEDL.NS':'Energy','RVNL.NS':'Energy',
    'IRCON.NS':'Energy','NTPC.NS':'Energy','COALINDIA.NS':'Energy',
    'TCS.NS':'Technology','INFY.NS':'Technology','WIPRO.NS':'Technology',
    'HCLTECH.NS':'Technology','TECHM.NS':'Technology','BHARTIARTL.NS':'Technology',
    'ZOMATO.NS':'Technology','PAYTM.NS':'Technology','TATACOMM.NS':'Technology',
    'MPHASIS.NS':'Technology','COFORGE.NS':'Technology','PERSISTENT.NS':'Technology',
    'LTTS.NS':'Technology','DIXON.NS':'Technology',
    'HDFCBANK.NS':'Finance','ICICIBANK.NS':'Finance','SBIN.NS':'Finance',
    'KOTAKBANK.NS':'Finance','AXISBANK.NS':'Finance','BAJFINANCE.NS':'Finance',
    'MUTHOOTFIN.NS':'Finance','CHOLAFIN.NS':'Finance','BANDHANBNK.NS':'Finance',
    'RECLTD.NS':'Finance','PFC.NS':'Finance','IRFC.NS':'Finance','BAJAJFINSV.NS':'Finance',
    'HINDUNILVR.NS':'Consumer','ITC.NS':'Consumer','NESTLEIND.NS':'Consumer',
    'BRITANNIA.NS':'Consumer','TATAMOTORS.NS':'Consumer','TITAN.NS':'Consumer',
    'ASIANPAINT.NS':'Consumer','INDHOTEL.NS':'Consumer','COLPAL.NS':'Consumer',
    'DABUR.NS':'Consumer','MARICO.NS':'Consumer','BERGEPAINT.NS':'Consumer',
    'VBL.NS':'Consumer','POLYCAB.NS':'Consumer','KEI.NS':'Consumer',
    'HAVELLS.NS':'Consumer','VOLTAS.NS':'Consumer',
    'MARUTI.NS':'Automobile','M&M.NS':'Automobile','EICHERMOT.NS':'Automobile','HEROMOTOCO.NS':'Automobile',
    'SUNPHARMA.NS':'Healthcare','DRREDDY.NS':'Healthcare','CIPLA.NS':'Healthcare',
    'APOLLOHOSP.NS':'Healthcare','BIOCON.NS':'Healthcare','LUPIN.NS':'Healthcare','AUROPHARMA.NS':'Healthcare',
    'PIDILITIND.NS':'Chemicals','SRF.NS':'Chemicals','UPL.NS':'Chemicals',
    'DLF.NS':'RealEstate','GODREJPROP.NS':'RealEstate',
    'GRASIM.NS':'Textiles','PAGEIND.NS':'Textiles',
    'TRENT.NS':'Retail','DMART.NS':'Retail',
    'BEL.NS':'Defense','HAL.NS':'Defense',
    'JSWSTEEL.NS':'Metals','TATASTEEL.NS':'Metals','HINDALCO.NS':'Metals'
}

# ── Date windows ─────────────────────────────────────────────────────────────
# Publish date = 1 year ago. Target = price today.
PUBLISH_DATE = (datetime.today() - timedelta(days=365)).strftime('%Y-%m-%d')
TODAY        = datetime.today().strftime('%Y-%m-%d')
TOMORROW     = (datetime.today() + timedelta(days=1)).strftime('%Y-%m-%d')

print(f'Fetching data for {len(TICKERS)} tickers...')
print(f'Current price window : {PUBLISH_DATE}')
print(f'Target price window  : {TODAY}')
print('This takes ~2 minutes...')

records = []
for ticker in TICKERS:
    try:
        t = yf.Ticker(ticker)

        # Current price (1 year ago)
        hist_past = yf.download(ticker,
                                start=PUBLISH_DATE,
                                end=(datetime.strptime(PUBLISH_DATE,'%Y-%m-%d') + timedelta(days=7)).strftime('%Y-%m-%d'),
                                progress=False)
        if hist_past.empty: continue
        current_price = float(hist_past['Close'].iloc[0])

        # Target price (today = actual 1-year-forward price)
        hist_now = yf.download(ticker,
                               start=(datetime.today() - timedelta(days=5)).strftime('%Y-%m-%d'),
                               end=TOMORROW,
                               progress=False)
        if hist_now.empty: continue
        target_price = float(hist_now['Close'].iloc[-1])

        # Fundamentals from info dict
        info = t.info

        records.append({
            'Ticker':        ticker,
            'Sector':        SECTOR_MAP.get(ticker, 'Other'),
            'current_price': current_price,
            'target_price':  target_price,
            'pe_ratio':      info.get('trailingPE',        np.nan),
            'roe':           info.get('returnOnEquity',    np.nan),
            'roa':           info.get('returnOnAssets',    np.nan),
            'profit_margin': info.get('profitMargins',     np.nan),
            'revenue_growth':info.get('revenueGrowth',     np.nan),
            'earnings_growth':info.get('earningsGrowth',   np.nan),
            'debt_to_equity':info.get('debtToEquity',      np.nan),
            'current_ratio': info.get('currentRatio',      np.nan),
            'market_cap':    info.get('marketCap',         np.nan),
            'beta':          info.get('beta',              np.nan),
            'book_value':    info.get('bookValue',         np.nan),
            'price_to_book': info.get('priceToBook',       np.nan),
            'dividend_yield':info.get('dividendYield',     np.nan),
            'eps':           info.get('trailingEps',       np.nan),
            'ebitda_margin': info.get('ebitdaMargins',     np.nan),
        })
        print(f'  ✅ {ticker:25s}  CP={current_price:8.2f}  TP={target_price:8.2f}')

    except Exception as e:
        print(f'  ⚠️  {ticker} skipped: {e}')

df_raw = pd.DataFrame(records)
print(f'\n✅ Fetched {len(df_raw)} tickers successfully.')

df = df_raw.copy()

# ── 1. Log-scale market cap (handles scale from ₹150 Cr to ₹20 Lakh Cr) ────
df['log_market_cap'] = np.log1p(df['market_cap'].clip(lower=0))

# ── 2. Earnings yield (1/PE) — more stable than raw PE for regression ───────
df['earnings_yield'] = (1 / df['pe_ratio'].replace(0, np.nan)).clip(-2, 2)

# ── 3. Sector-relative PE — how expensive vs sector peers ───────────────────
sector_median_pe = df.groupby('Sector')['pe_ratio'].transform('median')
df['relative_pe'] = df['pe_ratio'] / (sector_median_pe + 1e-9)

# ── 4. 1-year actual return (our target variable, for reference) ─────────────
df['actual_1yr_return'] = (df['target_price'] - df['current_price']) / df['current_price']

# ── 5. Price-to-earnings-growth proxy ───────────────────────────────────────
df['peg_proxy'] = df['pe_ratio'] / (df['earnings_growth'].abs() * 100 + 1e-9)
df['peg_proxy'] = df['peg_proxy'].clip(-50, 50)

# ── Drop rows missing target ─────────────────────────────────────────────────
df = df.dropna(subset=['target_price', 'current_price'])

# ── Fill remaining NaNs with column median ───────────────────────────────────
numeric_cols = df.select_dtypes(include=np.number).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

print(f'Dataset shape: {df.shape}')
print(f'Features available: {list(numeric_cols)}')
display(df[['Ticker','Sector','current_price','target_price','actual_1yr_return']].head(10))

df.to_excel("indian_equity_portfolio_dataset.xlsx", index=False)

print("✅ Excel file exported successfully!")

import matplotlib.pyplot as plt
import numpy as np

# -----------------------------------
# CLEAN STYLE SETTINGS
# -----------------------------------

plt.style.use('default')

fig, axes = plt.subplots(
    1, 3,
    figsize=(20, 6),
    facecolor='white'
)

# ===================================
# 1. PRICE DISTRIBUTION
# ===================================

axes[0].hist(
    df['current_price'],
    bins=20,
    edgecolor='white',
    linewidth=1.2,
    alpha=0.9
)

axes[0].set_title(
    'Distribution of Stock Prices',
    fontsize=14,
    fontweight='bold'
)

axes[0].set_xlabel('Price (₹)', fontsize=11)
axes[0].set_ylabel('Number of Stocks', fontsize=11)

axes[0].grid(alpha=0.2)

# Mean line
mean_price = df['current_price'].mean()

axes[0].axvline(
    mean_price,
    linestyle='--',
    linewidth=2,
    label=f'Mean = ₹{mean_price:.0f}'
)

axes[0].legend()

# ===================================
# 2. RETURN DISTRIBUTION
# ===================================

returns_pct = df['actual_1yr_return'] * 100

axes[1].hist(
    returns_pct,
    bins=20,
    edgecolor='white',
    linewidth=1.2,
    alpha=0.9
)

axes[1].axvline(
    0,
    linestyle='--',
    linewidth=2,
    label='Break-even'
)

axes[1].set_title(
    'Distribution of 1-Year Returns',
    fontsize=14,
    fontweight='bold'
)

axes[1].set_xlabel('Return (%)', fontsize=11)
axes[1].set_ylabel('Number of Stocks', fontsize=11)

axes[1].grid(alpha=0.2)

# Mean return line
mean_ret = returns_pct.mean()

axes[1].axvline(
    mean_ret,
    linestyle=':',
    linewidth=2,
    label=f'Mean = {mean_ret:.1f}%'
)

axes[1].legend()

# ===================================
# 3. SECTOR PERFORMANCE
# ===================================

sector_ret = (
    df.groupby('Sector')['actual_1yr_return']
    .mean()
    .sort_values()
    * 100
)

bars = axes[2].barh(
    sector_ret.index,
    sector_ret.values,
    edgecolor='white',
    linewidth=1.2
)

# Add labels
for i, v in enumerate(sector_ret.values):
    axes[2].text(
        v,
        i,
        f' {v:.1f}%',
        va='center',
        fontsize=10
    )

axes[2].axvline(
    0,
    color='black',
    linewidth=1
)

axes[2].set_title(
    'Average 1-Year Return by Sector',
    fontsize=14,
    fontweight='bold'
)

axes[2].set_xlabel('Average Return (%)', fontsize=11)

axes[2].grid(axis='x', alpha=0.2)

# ===================================
# MAIN TITLE
# ===================================

plt.suptitle(
    'Chart 1 — Indian Equity Market Overview',
    fontsize=18,
    fontweight='bold',
    y=1.03
)

plt.tight_layout()

plt.show()

# ===================================
# SUMMARY STATS
# ===================================

print("\n" + "="*60)
print("DATASET SUMMARY")
print("="*60)

print(f"Number of Stocks       : {len(df)}")
print(f"Average 1Y Return      : {returns_pct.mean():.2f}%")
print(f"Median 1Y Return       : {returns_pct.median():.2f}%")
print(f"Best Sector            : {sector_ret.idxmax()}")
print(f"Worst Sector           : {sector_ret.idxmin()}")
print("="*60)

# Features used — current_price EXCLUDED to prevent leakage
FEATURE_COLS = [
    'pe_ratio', 'roe', 'roa', 'profit_margin', 'revenue_growth',
    'earnings_growth', 'debt_to_equity', 'current_ratio', 'log_market_cap',
    'beta', 'price_to_book', 'dividend_yield', 'eps', 'ebitda_margin',
    'earnings_yield', 'relative_pe', 'peg_proxy'
]
# Only keep columns that exist in df
FEATURE_COLS = [c for c in FEATURE_COLS if c in df.columns]

TARGET_COL = 'target_price'

split = int(len(df) * 0.80)
train_df = df.iloc[:split].copy()
test_df  = df.iloc[split:].copy()

X_train = train_df[FEATURE_COLS].values
y_train = train_df[TARGET_COL].values
X_test  = test_df[FEATURE_COLS].values
y_test  = test_df[TARGET_COL].values

# Scale features
scaler_x = StandardScaler()
X_train_s = scaler_x.fit_transform(X_train)
X_test_s  = scaler_x.transform(X_test)

# Scale target
scaler_y = StandardScaler()
y_train_s = scaler_y.fit_transform(y_train.reshape(-1,1)).flatten()

print(f'Train size : {len(train_df)} firms')
print(f'Test size  : {len(test_df)} firms')
print(f'Features   : {len(FEATURE_COLS)}')
print(f'Features   : {FEATURE_COLS}')


# Baseline: predict target = current price (no change assumed)
baseline_preds = test_df['current_price'].values
baseline_mse   = mean_squared_error(y_test, baseline_preds)
baseline_r2    = r2_score(y_test, baseline_preds)
baseline_mape  = np.mean(np.abs((y_test - baseline_preds) / np.where(y_test==0,1,y_test)))
baseline_acc   = max(0, (1 - baseline_mape) * 100)

# Directional accuracy
actual_dir = (y_test > test_df['current_price'].values).astype(int)
pred_dir   = (baseline_preds > test_df['current_price'].values).astype(int)
baseline_dir = np.mean(actual_dir == pred_dir) * 100

print('━'*45)
print('  BASELINE (Naive Persistence)')
print('━'*45)
print(f'  MSE                : {baseline_mse:>15,.2f}')
print(f'  R²                 : {baseline_r2:>15.4f}')
print(f'  Accuracy (1-MAPE)  : {baseline_acc:>14.2f}%')
print(f'  Directional Acc    : {baseline_dir:>14.1f}%')
print('━'*45)


xgb_model = xgb.XGBRegressor(
    n_estimators=400,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    verbosity=0
)
xgb_model.fit(X_train, y_train)
xgb_preds = xgb_model.predict(X_test)

xgb_mse  = mean_squared_error(y_test, xgb_preds)
xgb_r2   = r2_score(y_test, xgb_preds)
xgb_mape = np.mean(np.abs((y_test - xgb_preds) / np.where(y_test==0,1,y_test)))
xgb_acc  = max(0, (1 - xgb_mape) * 100)
xgb_dir  = np.mean((xgb_preds > test_df['current_price'].values).astype(int) == actual_dir) * 100

print('━'*45)
print('  XGBOOST')
print('━'*45)
print(f'  MSE                : {xgb_mse:>15,.2f}')
print(f'  R²                 : {xgb_r2:>15.4f}')
print(f'  Accuracy (1-MAPE)  : {xgb_acc:>14.2f}%')
print(f'  Directional Acc    : {xgb_dir:>14.1f}%')
print(f'  Beat Baseline MSE  : {"✅ YES" if xgb_mse < baseline_mse else "❌ NO"}')
print('━'*45)


# Build improved neural network
inputs = tf.keras.Input(shape=(X_train_s.shape[1],))
x = tf.keras.layers.Dense(128, activation='relu')(inputs)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dropout(0.3)(x)
x = tf.keras.layers.Dense(64, activation='relu')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dropout(0.2)(x)
x = tf.keras.layers.Dense(32, activation='relu')(x)
x = tf.keras.layers.Dropout(0.1)(x)
output = tf.keras.layers.Dense(1)(x)
nn_model = tf.keras.Model(inputs, output)

nn_model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss='huber'
)

history = nn_model.fit(
    X_train_s, y_train_s,
    epochs=300,
    batch_size=16,
    validation_split=0.15,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=30,
                                          restore_best_weights=True, verbose=0),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                              patience=15, min_lr=1e-6, verbose=0)
    ],
    verbose=0
)

nn_preds_s = nn_model.predict(X_test_s, verbose=0)
nn_preds   = scaler_y.inverse_transform(nn_preds_s).flatten()
nn_preds   = np.maximum(nn_preds, 0)   # no negative prices

nn_mse  = mean_squared_error(y_test, nn_preds)
nn_r2   = r2_score(y_test, nn_preds)
nn_mape = np.mean(np.abs((y_test - nn_preds) / np.where(y_test==0,1,y_test)))
nn_acc  = max(0, (1 - nn_mape) * 100)
nn_dir  = np.mean((nn_preds > test_df['current_price'].values).astype(int) == actual_dir) * 100

print(f'Stopped at epoch: {len(history.history["loss"])}')
print('━'*45)
print('  NEURAL NETWORK')
print('━'*45)
print(f'  MSE                : {nn_mse:>15,.2f}')
print(f'  R²                 : {nn_r2:>15.4f}')
print(f'  Accuracy (1-MAPE)  : {nn_acc:>14.2f}%')
print(f'  Directional Acc    : {nn_dir:>14.1f}%')
print(f'  Beat Baseline MSE  : {"✅ YES" if nn_mse < baseline_mse else "❌ NO"}')
print('━'*45)

models     = ['Baseline\n(Naive)', 'XGBoost', 'Neural\nNetwork']
mse_vals   = [baseline_mse, xgb_mse, nn_mse]
r2_vals    = [baseline_r2,  xgb_r2,  nn_r2]
acc_vals   = [baseline_acc, xgb_acc, nn_acc]
dir_vals   = [baseline_dir, xgb_dir, nn_dir]

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
colors = ['#d9534f', '#5bc0de', '#5cb85c']

def bar_chart(ax, vals, title, ylabel, fmt='{:.0f}', highlight_min=True):
    best = min(vals) if highlight_min else max(vals)
    bar_colors = ['gold' if v == best else c for v, c in zip(vals, colors)]
    bars = ax.bar(models, vals, color=bar_colors, edgecolor='white', linewidth=1.5)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.02,
                fmt.format(val), ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.set_title(title, fontweight='bold')
    ax.set_ylabel(ylabel)

bar_chart(axes[0], mse_vals,  'MSE (lower=better)', 'MSE (INR²)', '{:,.0f}', highlight_min=True)
bar_chart(axes[1], r2_vals,   'R² Score (higher=better)', 'R²', '{:.3f}', highlight_min=False)
bar_chart(axes[2], acc_vals,  'Accuracy % (1−MAPE)', '%', '{:.1f}%', highlight_min=False)
bar_chart(axes[3], dir_vals,  'Directional Accuracy', '%', '{:.1f}%', highlight_min=False)

plt.suptitle('Chart 2 — Model Performance Comparison  (🥇 = Best)', fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

def scatter_plot(ax, y_true, y_pred, title, r2):
    mn = min(y_true.min(), y_pred.min())
    mx = max(y_true.max(), y_pred.max())
    ax.plot([mn, mx], [mn, mx], 'r--', linewidth=1.5, label='Perfect prediction')
    ax.scatter(y_true, y_pred, alpha=0.7, edgecolors='white', linewidth=0.5, s=60, color='steelblue')
    # Label a few tickers
    for i, row in test_df.iterrows():
        idx = test_df.index.get_loc(i)
        if abs(y_pred[idx] - y_true[idx]) / (y_true[idx] + 1) > 0.3:
            ax.annotate(row['Ticker'].replace('.NS',''),
                        (y_true[idx], y_pred[idx]),
                        fontsize=7, ha='left', color='gray')
    ax.set_xlabel('Actual Future Price (₹)')
    ax.set_ylabel('Predicted Future Price (₹)')
    ax.set_title(f'{title}\nR² = {r2:.4f}', fontweight='bold')
    ax.legend()

scatter_plot(axes[0], y_test, xgb_preds, 'XGBoost — Actual vs Predicted', xgb_r2)
scatter_plot(axes[1], y_test, nn_preds,  'Neural Network — Actual vs Predicted', nn_r2)

plt.suptitle('Chart 3 — Actual vs Predicted Price (Test Set)', fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()

explainer   = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(pd.DataFrame(X_test, columns=FEATURE_COLS))

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Bar plot — mean absolute SHAP
mean_shap = np.abs(shap_values).mean(axis=0)
sorted_idx = np.argsort(mean_shap)
axes[0].barh([FEATURE_COLS[i] for i in sorted_idx],
              mean_shap[sorted_idx], color='steelblue', edgecolor='white')
axes[0].set_xlabel('Mean |SHAP Value|')
axes[0].set_title('Feature Importance (XGBoost SHAP)', fontweight='bold')

# Built-in XGBoost importance
xgb_imp = pd.Series(xgb_model.feature_importances_, index=FEATURE_COLS).sort_values()
xgb_imp.plot(kind='barh', ax=axes[1], color='seagreen', edgecolor='white')
axes[1].set_xlabel('Feature Importance Score')
axes[1].set_title('XGBoost Built-in Feature Importance', fontweight='bold')

plt.suptitle('Chart 5 — Feature Importance & SHAP Analysis', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()

print('\nTop 5 most important features (SHAP):')
top5 = sorted(zip(FEATURE_COLS, mean_shap), key=lambda x: -x[1])[:5]
for rank, (feat, val) in enumerate(top5, 1):
    print(f'  {rank}. {feat:20s}  SHAP={val:.4f}')


fig, ax = plt.subplots(figsize=(9, 4))
epochs = range(1, len(history.history['loss']) + 1)
ax.plot(epochs, history.history['loss'],     label='Train Loss',      color='steelblue', linewidth=2)
ax.plot(epochs, history.history['val_loss'], label='Validation Loss', color='tomato',    linewidth=2, linestyle='--')
ax.set_xlabel('Epoch')
ax.set_ylabel('Huber Loss')
ax.set_title('Chart 6 — Neural Network Training Curve', fontweight='bold')
ax.legend()
plt.tight_layout()
plt.show()
print(f'Early stopped at epoch {len(epochs)}')

# Generate predictions for ALL tickers using XGBoost
X_all   = scaler_x.transform(df[FEATURE_COLS].values)
all_pred = xgb_model.predict(X_all)

results = pd.DataFrame({
    'Ticker':              df['Ticker'].str.replace('.NS','', regex=False),
    'Sector':              df['Sector'],
    'Price_1yr_Ago (₹)':  df['current_price'].round(2),
    'Actual_Today (₹)':   df['target_price'].round(2),
    'XGB_Predicted (₹)':  all_pred.round(2),
    'Actual_Return (%)':  (df['actual_1yr_return'] * 100).round(1),
    'Pred_Error (%)':     (((all_pred - df['target_price']) / df['target_price']) * 100).round(1),
    'Direction_Correct':  [
        '✅' if (p > c) == (t > c) else '❌'
        for p, c, t in zip(all_pred, df['current_price'], df['target_price'])
    ]
})

# Style the table
def color_return(val):
    if isinstance(val, (int, float)):
        color = 'green' if val > 0 else 'red' if val < 0 else 'black'
        return f'color: {color}'
    return ''

styled = results.style\
    .applymap(color_return, subset=['Actual_Return (%)','Pred_Error (%)'])\
    .set_caption('Full Predictions — All Tickers (XGBoost Model)')\
    .format({'Price_1yr_Ago (₹)': '₹{:.2f}', 'Actual_Today (₹)': '₹{:.2f}',
              'XGB_Predicted (₹)': '₹{:.2f}', 'Actual_Return (%)': '{:.1f}%',
              'Pred_Error (%)': '{:.1f}%'})

display(styled)
print(f'\nDirectional accuracy: {(results["Direction_Correct"]=="✅").mean()*100:.1f}%')

def build_baseline_metric():
    return {
        "model": "Naive Persistence Baseline",
        "mse": 31050.34,
        "r2": 0.8816,
        "accuracy_1_minus_mape": 83.46,
        "directional_accuracy": 61.1,
        "status": "real notebook output"
    }


def build_primary_metric():
    return {
        "model": "XGBoost",
        "mse": 852286.61,
        "r2": -2.2491,
        "directional_accuracy": 72.2,
        "best_metric": "directional_accuracy",
        "status": "preliminary but real"
    }


def build_milestone_manifest():
    return {
        "project": "Indian Equity Price Predictor",
        "status": "working",
        "data_source": "Yahoo Finance via yfinance",
        "models": [
            "Naive Baseline",
            "XGBoost",
            "Neural Network"
        ],
        "notebook": "notebooks/Indian_Equity_Predictor_Masters.ipynb",
        "outputs_generated": True
    }

















