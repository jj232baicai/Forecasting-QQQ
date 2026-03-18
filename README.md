# QQQ Return Forecasting — Time Series, Statistical & Deep Learning Models

**Authors:** Andy Jiang & Collin McDevitt  
**Date:** February 2026

---

## Project Overview

This project investigates the predictability of daily **QQQ (Invesco QQQ Trust) log returns** using a combination of macroeconomic indicators, market-derived features, and a hierarchy of forecasting models — from classical statistical baselines up to deep learning architectures.

The central research question: *Can macroeconomic and market signals meaningfully improve out-of-sample forecasting of QQQ daily returns beyond a random walk baseline?*

---

## Repository Structure

```
.
├── dataframe.csv                        # Base dataset — raw FRED + Yahoo Finance features
├── Time_Series_Final_Data_Setup.ipynb   # Data collection & preprocessing pipeline
├── Stats.ipynb                          # Statistical EDA, SARIMAX, and GARCH modeling
├── market_base_eda.ipynb                # Market-based EDA (return distribution, volatility)
├── market_base_eda.html                 # Rendered HTML output of market EDA
├── final_temp1.ipynb                    # Preliminary N-HiTS prototype (v1, log-price target)
├── ML_Dl.ipynb                          # Final ML & DL pipeline (XGBoost, LSTM, N-HiTS)
├── ML_DL.html                           # Rendered HTML output of ML/DL notebook
└── Time_Series_Final.pptx               # Final project presentation slides
```

---

## Dataset

### `dataframe.csv` — Base Features Only

**Coverage:** 2003-01-02 to 2025-12-31 (5,787 trading days)

> ⚠️ **Note:** `dataframe.csv` contains only the 9 raw base columns downloaded from Yahoo Finance and FRED. The additional engineered features used in model training (rolling volatilities, lagged returns, constituent returns, event flags, etc.) are **not** stored in this file — they are computed on-the-fly inside `ML_Dl.ipynb`.

| Column | Description | Source |
|---|---|---|
| `ds` | Date (daily, trading days) | — |
| `y` | QQQ closing price | Yahoo Finance |
| `vix` | CBOE Volatility Index | Yahoo Finance (`^VIX`) |
| `EPU` | Economic Policy Uncertainty Index (daily) | FRED (`USEPUINDXD`) |
| `UMCSENT` | University of Michigan Consumer Sentiment | FRED (`UMCSENT`) |
| `JOBLESS` | Initial Jobless Claims | FRED (`ICSA`) |
| `FEDFUNDS` | Federal Funds Rate | FRED (`FEDFUNDS`) |
| `FLOWS` | Global Liquidity / Equity Market Uncertainty Index | FRED (`WLEMUINDXD`) |
| `CPI` | Consumer Price Index (All Urban) | FRED (`CPIAUCSL`) |
| `inflation_yoy` | Year-over-year CPI inflation (%) | Derived from CPI |

---

### Additional Engineered Features — Built in `ML_Dl.ipynb`

The full model feature set extends the base CSV with the following columns, all constructed with careful **data leakage prevention** (outer `shift(1)` ensures only t-1 information is used when predicting day t).

**Return & Moving Averages** (stationary — computed on log returns, not log price):

| Feature | Description |
|---|---|
| `RET_MA5` | 5-day rolling mean of log returns |
| `RET_MA10` | 10-day rolling mean of log returns |
| `RET_MA21` | 21-day rolling mean of log returns |
| `RET_LAG1` | Log return lagged 1 day (t-1) |
| `RET_LAG5` | Log return lagged 5 days (t-5) |

**Volatility Features:**

| Feature | Description |
|---|---|
| `ROLL_VOL5` | 5-day rolling std of log returns (short-term) |
| `ROLL_VOL10` | 10-day rolling std of log returns (medium-term) |
| `ROLL_VOL21` | 21-day rolling std of log returns (1-month baseline) |
| `GARCH_VOL` | GARCH(1,1) conditional volatility (fitted via `arch`) |

**Market Signals:**

| Feature | Description |
|---|---|
| `VIX_DIFF` | Daily change in VIX |
| `FED_RATE_DIFF` | Daily change in the Fed Funds Rate |
| `SPY_RET` | S&P 500 (SPY) daily log return |
| `AAPL_RET` | Apple (AAPL) daily log return |
| `NVDA_RET` | NVIDIA (NVDA) daily log return |
| `MSFT_RET` | Microsoft (MSFT) daily log return |

**Publication-lag corrected macro features** (monthly series shifted ~21 trading days, weekly series shifted ~5 trading days to reflect real-world data availability):

| Feature | Derived From | Lag Applied |
|---|---|---|
| `INFLATION` | CPI | 21 trading days (~1 month) |
| `SENTIMENT` | UMCSENT | 21 trading days (~1 month) |
| `FED_RATE` | FEDFUNDS | 21 trading days (~1 month) |
| `JOBLESS` | ICSA | 5 trading days (~1 week) |
| `GLOBAL_LIQ` | WLEMUINDXD | None (daily series) |

**Event Flags (future exogenous):**

| Feature | Description |
|---|---|
| `EARN_FLAG_W` | Weighted earnings flag for top-5 QQQ constituents (AAPL, MSFT, NVDA, AMZN, GOOGL) — non-zero within ±3 trading days of each announcement, weighted by index weight |
| `FOMC_FLAG` | Binary flag marking FOMC meeting dates (2020–2024) |

---

## Workflow

### 1. Data Collection — `Time_Series_Final_Data_Setup.ipynb`
- Downloads QQQ and VIX from **Yahoo Finance**
- Pulls macro indicators from the **FRED API**
- Aligns all series to the trading-day calendar via forward-fill
- Computes `inflation_yoy` as 12-month CPI percentage change
- Saves the merged base dataset as `dataframe.csv`

### 2. Statistical EDA & Baseline Models — `Stats.ipynb`
- **Stationarity testing** via ADF test on all features; UMCSENT, FEDFUNDS, and CPI found to be non-stationary and first-differenced
- **Cross-correlation analysis** of lagged macro indicators vs. QQQ log returns — correlations remain near zero after removing spurious trends
- **Optimal lag selection** per indicator (VIX: 1, EPU: 2, JOBLESS: 3, FLOWS: 26, etc.)
- **SARIMAX / AR(1) baseline**: AR(1) achieves the lowest test RMSE, marginally beating the naive forecast; adding lagged macro regressors worsens out-of-sample performance — consistent with weak-form market efficiency
- **GARCH(1,1)**: Confirms strong volatility clustering (ARCH effects) in QQQ returns

### 3. Market-Based EDA — `market_base_eda.ipynb`
Focuses on market-derived signals for QQQ (2000–2025):
- Return distribution vs. Normal (fat tails, excess kurtosis, Jarque-Bera / KS tests)
- Volatility clustering and ARCH-effect testing (Ljung-Box, ARCH-LM)
- ACF / PACF of returns and squared returns
- Constituent return correlations: SPY, AAPL, NVDA, MSFT
- VIX regime analysis and lagged return predictive power
- Rolling feature stability over time

### 4. Preliminary Prototype — `final_temp1.ipynb`
An earlier v1 pipeline using **log price (non-stationary)** as the model target and a reduced feature set (no rolling volatility, no GARCH vol, no constituent returns, no publication lags). Kept for reference; superseded by `ML_Dl.ipynb`.

### 5. Final ML & Deep Learning Models — `ML_Dl.ipynb`

| Step | Description |
|---|---|
| 1 | Download market & macro data (Yahoo Finance + FRED) |
| 2 | Build event flags — earnings windows (top-5 constituents) and FOMC meeting dates |
| 3 | Feature engineering — log returns, multi-scale rolling volatility, GARCH vol, lagged returns, publication-lag corrected macro, constituent log returns |
| 4 | Build NeuralForecast-compatible DataFrame |
| 5 | Time-series split — Train ≤ 2021-12-31 · Val ≤ 2023-12-31 · Test ≤ 2024-12-31 |
| 6 | Model config — **N-HiTS** and **LSTM** (forecast horizon H = 21 days, input size = 63) |
| 7 | Training via `NeuralForecast.fit()` |
| 8 | Rolling cross-validation on test set (step_size = H) |
| 9 | **XGBoost** baseline with RandomizedSearchCV + expanding-window CV |
| 10 | Evaluation — RMSE, MAE, MAPE, Directional Accuracy, Ljung-Box on residuals |
| 11 | Diagnostic plots — return forecasts, reconstructed QQQ price, residuals, Q-Q plots, ACF of residuals, XGBoost feature importance |

---

## Models Summary

| Model | Type | Library | Target |
|---|---|---|---|
| Naive (zero-return) | Baseline | — | Log return |
| AR(1) / SARIMAX | Statistical | `statsmodels` | Log return |
| GARCH(1,1) | Volatility model | `arch` | Conditional variance |
| XGBoost | Gradient Boosting | `xgboost` | Log return |
| LSTM | Recurrent Neural Network | `neuralforecast` | Log return |
| N-HiTS | Neural Hierarchical Interpolation | `neuralforecast` | Log return |

---

## Installation & Dependencies

```bash
pip install neuralforecast xgboost fredapi yfinance statsmodels arch \
            scikit-learn scipy matplotlib pandas numpy darts pmdarima \
            setuptools pandas-datareader
```

> **Note:** A valid **FRED API key** is required for data collection in `Time_Series_Final_Data_Setup.ipynb` and `ML_Dl.ipynb`.

---

## Key Findings

- Daily QQQ log returns closely follow a **random walk**; lagged macro indicators show near-zero cross-correlation with future returns once spurious trends are removed.
- AR(1) marginally outperforms the naive forecast; adding lagged macro regressors leads to in-sample overfitting.
- GARCH(1,1) confirms significant **volatility clustering** in QQQ returns.
- Deep learning models (N-HiTS, LSTM) and XGBoost leverage the richer engineered feature set — including event flags, multi-scale volatility, GARCH vol, and constituent returns — with performance benchmarked via RMSE, directional accuracy, and full residual diagnostics.

---

## Presentation

See `Time_Series_Final.pptx` for a full summary of methodology, results, and conclusions.
