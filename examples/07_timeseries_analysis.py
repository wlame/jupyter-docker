#!/usr/bin/env python3
"""
Time Series Analysis
====================
Demonstrates time series analysis with tsfresh, sktime, statsmodels, and pmdarima.

tsfresh: https://tsfresh.readthedocs.io/
sktime: https://www.sktime.net/
statsmodels: https://www.statsmodels.org/
pmdarima: https://alkaline-ml.com/pmdarima/

Note: Some operations may take time due to feature extraction.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Time series libraries
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.arima.model import ARIMA
import pmdarima as pm
from tsfresh import extract_features
from tsfresh.feature_extraction import MinimalFCParameters
from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.performance_metrics.forecasting import mean_absolute_percentage_error

# =============================================================================
# Generate Sample Time Series Data
# =============================================================================
print("=" * 60)
print("Generating Sample Time Series Data")
print("=" * 60)

np.random.seed(42)

# Create time series with trend, seasonality, and noise
n_periods = 365 * 2  # 2 years of daily data
dates = pd.date_range("2023-01-01", periods=n_periods, freq="D")

# Components
trend = np.linspace(100, 150, n_periods)  # Linear trend
seasonal = 20 * np.sin(2 * np.pi * np.arange(n_periods) / 365)  # Yearly seasonality
weekly = 5 * np.sin(2 * np.pi * np.arange(n_periods) / 7)  # Weekly seasonality
noise = np.random.randn(n_periods) * 5

# Combined series
ts = pd.Series(trend + seasonal + weekly + noise, index=dates, name="value")

print(f"Time series shape: {ts.shape}")
print(f"Date range: {ts.index.min()} to {ts.index.max()}")
print(f"Value range: {ts.min():.2f} to {ts.max():.2f}")
print(ts.head(10))

# =============================================================================
# Statsmodels - Time Series Decomposition
# =============================================================================
print("\n" + "=" * 60)
print("Statsmodels Time Series Decomposition")
print("=" * 60)

# Decompose the time series
decomposition = seasonal_decompose(ts, model="additive", period=365)

fig, axes = plt.subplots(4, 1, figsize=(12, 12))

decomposition.observed.plot(ax=axes[0], title="Observed")
decomposition.trend.plot(ax=axes[1], title="Trend")
decomposition.seasonal.plot(ax=axes[2], title="Seasonal")
decomposition.resid.plot(ax=axes[3], title="Residual")

plt.tight_layout()
plt.savefig("/home/jupyter/examples/output/ts_decomposition.png", dpi=150)
print("Saved: ts_decomposition.png")
plt.close()

# =============================================================================
# Statistical Tests
# =============================================================================
print("\n" + "=" * 60)
print("Statistical Tests for Stationarity")
print("=" * 60)

# Augmented Dickey-Fuller test
result = adfuller(ts.dropna())
print("Augmented Dickey-Fuller Test:")
print(f"  Test Statistic: {result[0]:.4f}")
print(f"  p-value: {result[1]:.4f}")
print(f"  Lags Used: {result[2]}")
print(f"  Observations: {result[3]}")
print("  Critical Values:")
for key, value in result[4].items():
    print(f"    {key}: {value:.4f}")

if result[1] < 0.05:
    print("\n  -> Series is stationary (reject null hypothesis)")
else:
    print("\n  -> Series is non-stationary (fail to reject null hypothesis)")

# ACF and PACF
fig, axes = plt.subplots(1, 2, figsize=(14, 4))

# ACF
acf_values = acf(ts, nlags=40)
axes[0].bar(range(len(acf_values)), acf_values)
axes[0].axhline(y=0, linestyle="-", color="black")
axes[0].axhline(y=1.96 / np.sqrt(len(ts)), linestyle="--", color="gray")
axes[0].axhline(y=-1.96 / np.sqrt(len(ts)), linestyle="--", color="gray")
axes[0].set_title("Autocorrelation Function (ACF)")
axes[0].set_xlabel("Lag")

# PACF
pacf_values = pacf(ts, nlags=40)
axes[1].bar(range(len(pacf_values)), pacf_values)
axes[1].axhline(y=0, linestyle="-", color="black")
axes[1].axhline(y=1.96 / np.sqrt(len(ts)), linestyle="--", color="gray")
axes[1].axhline(y=-1.96 / np.sqrt(len(ts)), linestyle="--", color="gray")
axes[1].set_title("Partial Autocorrelation Function (PACF)")
axes[1].set_xlabel("Lag")

plt.tight_layout()
plt.savefig("/home/jupyter/examples/output/ts_acf_pacf.png", dpi=150)
print("Saved: ts_acf_pacf.png")
plt.close()

# =============================================================================
# ARIMA Modeling with Statsmodels
# =============================================================================
print("\n" + "=" * 60)
print("ARIMA Modeling (Statsmodels)")
print("=" * 60)

# Use monthly data for faster computation
monthly_ts = ts.resample("ME").mean()

# Split data
train_size = int(len(monthly_ts) * 0.8)
train, test = monthly_ts[:train_size], monthly_ts[train_size:]

# Fit ARIMA model
model = ARIMA(train, order=(1, 1, 1))
fitted = model.fit()

print("ARIMA(1,1,1) Model Summary:")
print(f"  AIC: {fitted.aic:.2f}")
print(f"  BIC: {fitted.bic:.2f}")

# Forecast
forecast = fitted.forecast(steps=len(test))
print(f"\nForecast MAPE: {np.mean(np.abs((test - forecast) / test)) * 100:.2f}%")

# Plot
fig, ax = plt.subplots(figsize=(12, 5))
train.plot(ax=ax, label="Training Data")
test.plot(ax=ax, label="Test Data")
forecast.plot(ax=ax, label="Forecast", linestyle="--")
ax.set_title("ARIMA Forecast")
ax.legend()
plt.tight_layout()
plt.savefig("/home/jupyter/examples/output/ts_arima.png", dpi=150)
print("Saved: ts_arima.png")
plt.close()

# =============================================================================
# Auto-ARIMA with pmdarima
# =============================================================================
print("\n" + "=" * 60)
print("Auto-ARIMA (pmdarima)")
print("=" * 60)

# Use pmdarima's auto_arima for automatic parameter selection
auto_model = pm.auto_arima(
    train,
    start_p=0,
    start_q=0,
    max_p=3,
    max_q=3,
    d=None,
    seasonal=False,
    trace=True,
    error_action="ignore",
    suppress_warnings=True,
    stepwise=True,
)

print(f"\nBest model: ARIMA{auto_model.order}")
print(f"AIC: {auto_model.aic():.2f}")

# Forecast with auto_arima
auto_forecast = auto_model.predict(n_periods=len(test))
auto_forecast = pd.Series(auto_forecast, index=test.index)

print(f"Auto-ARIMA MAPE: {np.mean(np.abs((test - auto_forecast) / test)) * 100:.2f}%")

# =============================================================================
# tsfresh - Feature Extraction
# =============================================================================
print("\n" + "=" * 60)
print("tsfresh Feature Extraction")
print("=" * 60)

# Prepare data for tsfresh (requires id column)
# Create multiple time series for feature extraction
tsfresh_data = []
for series_id in range(5):
    for t, val in enumerate(ts.values[:100]):  # Use subset for speed
        tsfresh_data.append({"id": series_id, "time": t, "value": val + np.random.randn() * 10})

tsfresh_df = pd.DataFrame(tsfresh_data)
print(f"tsfresh input shape: {tsfresh_df.shape}")

# Extract features (using minimal parameters for speed)
features = extract_features(
    tsfresh_df,
    column_id="id",
    column_sort="time",
    column_value="value",
    default_fc_parameters=MinimalFCParameters(),
    n_jobs=1,  # Use 1 job for container compatibility
)

print(f"\nExtracted {features.shape[1]} features for {features.shape[0]} time series")
print("\nSample features:")
print(features.iloc[0, :10])

# =============================================================================
# sktime - Forecasting
# =============================================================================
print("\n" + "=" * 60)
print("sktime Forecasting")
print("=" * 60)

# Prepare data for sktime
y = monthly_ts.copy()
y.index = pd.PeriodIndex(y.index, freq="M")

# Split
y_train, y_test = temporal_train_test_split(y, test_size=0.2)

# Naive forecaster (baseline)
forecaster = NaiveForecaster(strategy="last")
forecaster.fit(y_train)
y_pred = forecaster.predict(fh=list(range(1, len(y_test) + 1)))

# Calculate MAPE
mape = mean_absolute_percentage_error(y_test, y_pred)
print(f"Naive Forecaster MAPE: {mape * 100:.2f}%")

# Plot sktime forecast
fig, ax = plt.subplots(figsize=(12, 5))
y_train.plot(ax=ax, label="Training")
y_test.plot(ax=ax, label="Test")
y_pred.plot(ax=ax, label="Naive Forecast", linestyle="--")
ax.set_title("sktime Naive Forecaster")
ax.legend()
plt.tight_layout()
plt.savefig("/home/jupyter/examples/output/ts_sktime.png", dpi=150)
print("Saved: ts_sktime.png")
plt.close()

# =============================================================================
# Rolling Statistics and Anomaly Detection
# =============================================================================
print("\n" + "=" * 60)
print("Rolling Statistics and Anomaly Detection")
print("=" * 60)

# Calculate rolling statistics
window = 30
rolling_mean = ts.rolling(window=window).mean()
rolling_std = ts.rolling(window=window).std()

# Simple anomaly detection using z-score
z_scores = (ts - rolling_mean) / rolling_std
anomalies = ts[np.abs(z_scores) > 3]

print(f"Detected {len(anomalies)} anomalies (|z-score| > 3)")

# Plot
fig, axes = plt.subplots(2, 1, figsize=(14, 8))

# Time series with rolling stats
axes[0].plot(ts.index, ts.values, alpha=0.5, label="Original")
axes[0].plot(rolling_mean.index, rolling_mean.values, label="30-day MA")
axes[0].fill_between(
    rolling_mean.index,
    rolling_mean - 2 * rolling_std,
    rolling_mean + 2 * rolling_std,
    alpha=0.2,
    label="2 Std Dev",
)
axes[0].scatter(anomalies.index, anomalies.values, color="red", s=50, label="Anomalies")
axes[0].set_title("Time Series with Rolling Statistics and Anomalies")
axes[0].legend()

# Z-scores
axes[1].plot(z_scores.index, z_scores.values)
axes[1].axhline(y=3, color="red", linestyle="--", label="Threshold (3)")
axes[1].axhline(y=-3, color="red", linestyle="--")
axes[1].set_title("Z-Scores")
axes[1].legend()

plt.tight_layout()
plt.savefig("/home/jupyter/examples/output/ts_anomalies.png", dpi=150)
print("Saved: ts_anomalies.png")
plt.close()

print("\n" + "=" * 60)
print("Time series analysis examples complete!")
print("=" * 60)
