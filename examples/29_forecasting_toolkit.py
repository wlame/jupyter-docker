#!/usr/bin/env python3
"""
Forecasting Toolkit: statsforecast, mlforecast, skforecast, arch, tslearn
=========================================================================
Forecasts a seasonal monthly series three ways — fast statistical models
(statsforecast AutoETS/AutoARIMA), gradient boosting on lag features
(mlforecast + LightGBM), and a recursive scikit-learn forecaster
(skforecast) — then models return volatility with a GARCH(1,1) from arch and
clusters series shapes with dynamic time warping in tslearn.

statsforecast: https://nixtlaverse.nixtla.io/statsforecast/
mlforecast:    https://nixtlaverse.nixtla.io/mlforecast/
skforecast:    https://skforecast.org/
arch:          https://arch.readthedocs.io/
tslearn:       https://tslearn.readthedocs.io/
"""

import os

import lightgbm as lgb
import matplotlib
import numpy as np
import pandas as pd
from arch import arch_model
from mlforecast import MLForecast
from skforecast.recursive import ForecasterRecursive
from sklearn.linear_model import Ridge
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA, AutoETS
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance

matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

rng = np.random.default_rng(seed=0)

# =============================================================================
# A seasonal monthly series with trend
# =============================================================================
print("=" * 60)
print("Monthly Series")
print("=" * 60)

months = pd.date_range('2014-01-01', periods=144, freq='MS')
values = (200 + 1.5 * np.arange(144) + 25 * np.sin(2 * np.pi * months.month / 12)
          + rng.normal(scale=8, size=144))
series = pd.DataFrame({'unique_id': 'sales', 'ds': months, 'y': values})
horizon = 12
train, test = series.iloc[:-horizon], series.iloc[-horizon:]
print(f"{len(train)} training months, {horizon}-month holdout")


def mape(actual: np.ndarray, forecast: np.ndarray) -> float:
    """Mean absolute percentage error, in percent."""
    return float(np.mean(np.abs((actual - forecast) / actual)) * 100)


# =============================================================================
# statsforecast — AutoETS and AutoARIMA
# =============================================================================
print("\n" + "=" * 60)
print("statsforecast: AutoETS and AutoARIMA")
print("=" * 60)

stats = StatsForecast(models=[AutoETS(season_length=12), AutoARIMA(season_length=12)], freq='MS')
stats_fc = stats.forecast(df=train, h=horizon)
for model in ('AutoETS', 'AutoARIMA'):
    print(f"{model:10} MAPE {mape(test['y'].to_numpy(), stats_fc[model].to_numpy()):.2f}%")

# =============================================================================
# mlforecast — LightGBM on lag and calendar features
# =============================================================================
print("\n" + "=" * 60)
print("mlforecast: LightGBM with Lags")
print("=" * 60)

ml = MLForecast(
    models={'lightgbm': lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, verbose=-1)},
    freq='MS',
    lags=[1, 2, 12],
    date_features=['month'],
)
ml.fit(train)
ml_fc = ml.predict(h=horizon)
print(f"lightgbm   MAPE {mape(test['y'].to_numpy(), ml_fc['lightgbm'].to_numpy()):.2f}%")

# =============================================================================
# skforecast — recursive multi-step forecaster
# =============================================================================
print("\n" + "=" * 60)
print("skforecast: Recursive Ridge Forecaster")
print("=" * 60)

y_train = train.set_index('ds')['y'].asfreq('MS')
forecaster = ForecasterRecursive(Ridge(alpha=1.0), lags=12)
forecaster.fit(y=y_train)
sk_fc = forecaster.predict(steps=horizon)
print(f"ridge      MAPE {mape(test['y'].to_numpy(), sk_fc.to_numpy()):.2f}%")

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(series['ds'], series['y'], color='black', lw=1, label='actual')
ax.plot(test['ds'], stats_fc['AutoETS'], label='AutoETS')
ax.plot(test['ds'], stats_fc['AutoARIMA'], label='AutoARIMA')
ax.plot(test['ds'], ml_fc['lightgbm'], label='mlforecast LightGBM')
ax.plot(test['ds'], sk_fc.to_numpy(), label='skforecast Ridge')
ax.axvline(test['ds'].iloc[0], color='grey', ls='--', lw=0.8)
ax.legend(ncol=3, fontsize=8)
ax.set_title(f'{horizon}-month forecasts')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'forecast_toolkit.png'), dpi=120)
plt.close()
print("Saved: forecast_toolkit.png")

# =============================================================================
# arch — GARCH(1,1) volatility
# =============================================================================
print("\n" + "=" * 60)
print("arch: GARCH(1,1) Volatility")
print("=" * 60)

n_days = 1_000
volatility = np.empty(n_days)
returns = np.empty(n_days)
volatility[0] = 1.0
for i in range(n_days):
    if i:
        volatility[i] = np.sqrt(0.05 + 0.1 * returns[i - 1] ** 2 + 0.85 * volatility[i - 1] ** 2)
    returns[i] = volatility[i] * rng.standard_normal()
fitted = arch_model(returns, vol='GARCH', p=1, q=1, mean='Zero').fit(disp='off')
params = fitted.params
print(f"Estimated omega {params['omega']:.3f}, alpha {params['alpha[1]']:.3f}, beta {params['beta[1]']:.3f} "
      "(simulated with 0.05, 0.10, 0.85)")
next_var = fitted.forecast(horizon=5).variance.iloc[-1].to_numpy()
print(f"5-day variance forecast: {np.round(next_var, 3).tolist()}")

# =============================================================================
# tslearn — DTW k-means on series shapes
# =============================================================================
print("\n" + "=" * 60)
print("tslearn: DTW Clustering")
print("=" * 60)

grid = np.linspace(0, 2 * np.pi, 50)
shapes = [np.sin(grid), np.abs(np.sin(grid)), np.linspace(-1, 1, 50)]
dataset = np.array([
    np.roll(shapes[k % 3], rng.integers(-5, 6)) + rng.normal(scale=0.1, size=50)
    for k in range(30)
])[:, :, None]
dataset = TimeSeriesScalerMeanVariance().fit_transform(dataset)
km = TimeSeriesKMeans(n_clusters=3, metric='dtw', max_iter=10, random_state=0).fit(dataset)
truth = np.arange(30) % 3
agreement = max(np.mean(km.labels_ == np.array([perm[t] for t in truth]))
                for perm in ([0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]))
print(f"Cluster sizes {np.bincount(km.labels_).tolist()}, agreement with true shapes {agreement:.0%}")

print("\nDone.")
