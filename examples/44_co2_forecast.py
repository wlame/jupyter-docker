#!/usr/bin/env python3
# # Forecasting Atmospheric CO₂
#
# Weekly CO₂ measurements from the Mauna Loa observatory (1958–2001, bundled
# with statsmodels) have a rising trend and a yearly cycle. The notebook turns
# them into a monthly series, compares a seasonal-naive baseline with ETS and
# ARIMA models using rolling-origin cross-validation (the honest way to score a
# forecast), and then forecasts five years ahead with prediction intervals.
#
# To forecast your own series, put a CSV with `date` and `value` columns in
# `/home/jupyter/data/` and set `DATA_FILE`.
#
# - [statsforecast](https://nixtlaverse.nixtla.io/statsforecast/)
# - [statsmodels datasets](https://www.statsmodels.org/stable/datasets/)

import json
import os

import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA, AutoETS, SeasonalNaive

# Outputs go to ./output next to the notebook (the kernel's working directory).
OUTPUT_DIR = os.path.abspath('output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATA_FILE = None  # e.g. '/home/jupyter/data/sales.csv' with columns date, value
SEASON = 12  # months per cycle
HORIZON = 24  # months scored in each cross-validation window
FORECAST_MONTHS = 60

# ## Load and tidy the series
#
# statsforecast expects a long table with `unique_id`, `ds` (date), and `y`.
# Gaps in the weekly data disappear when averaging to months.

if DATA_FILE:
    raw = pd.read_csv(DATA_FILE, parse_dates=['date']).set_index('date')['value']
else:
    raw = sm.datasets.co2.load_pandas().data['co2']
monthly = raw.resample('MS').mean().interpolate()
series = pd.DataFrame({'unique_id': 'co2', 'ds': monthly.index, 'y': monthly.to_numpy()})
print(f"{len(series)} months from {series['ds'].min():%Y-%m} to {series['ds'].max():%Y-%m}")

# ## Cross-validate three models
#
# Each window trains on everything before a cut-off and forecasts the next
# 24 months; four windows, one year apart, give a stable error estimate.

models = [SeasonalNaive(season_length=SEASON), AutoETS(season_length=SEASON), AutoARIMA(season_length=SEASON)]
forecaster = StatsForecast(models=models, freq='MS', n_jobs=1)
cv = forecaster.cross_validation(df=series, h=HORIZON, n_windows=4, step_size=12)
model_names = [str(model) for model in models]
errors_by_model = cv[model_names].sub(cv['y'], axis=0)
scores = pd.DataFrame({'MAE': errors_by_model.abs().mean(), 'RMSE': (errors_by_model ** 2).mean() ** 0.5})
print(scores.round(3).to_string())
best = scores['MAE'].idxmin()
print(f"Best model by MAE: {best}")

# ## Forecast five years with prediction intervals

forecaster.fit(series)
future = forecaster.predict(h=FORECAST_MONTHS, level=[80, 95])
print(future[['ds', best, f'{best}-lo-95', f'{best}-hi-95']].tail(3).round(2).to_string(index=False))

fig, ax = plt.subplots(figsize=(11, 4.5))
recent = series[series['ds'] >= series['ds'].max() - pd.DateOffset(years=12)]
ax.plot(recent['ds'], recent['y'], color='black', lw=1, label='observed')
ax.plot(future['ds'], future[best], color='#c44e52', label=f'{best} forecast')
ax.fill_between(future['ds'], future[f'{best}-lo-95'], future[f'{best}-hi-95'], color='#c44e52', alpha=0.15,
                label='95% interval')
ax.fill_between(future['ds'], future[f'{best}-lo-80'], future[f'{best}-hi-80'], color='#c44e52', alpha=0.3,
                label='80% interval')
ax.set(ylabel='CO₂ (ppm)', title='Mauna Loa CO₂: last 12 years and a 5-year forecast')
ax.legend(loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'co2_forecast.png'), dpi=120)
plt.show()

# ## Where did the models miss?
#
# Cross-validation errors by forecast step show how quickly accuracy decays with
# the distance from the cut-off.

cv['step'] = cv.groupby('cutoff').cumcount() + 1
errors = cv.assign(**{name: (cv[name] - cv['y']).abs() for name in model_names}).groupby('step')[model_names].mean()
fig, ax = plt.subplots(figsize=(8, 3.5))
errors.plot(ax=ax)
ax.set(xlabel='months after the cut-off', ylabel='mean absolute error (ppm)', title='Error by forecast horizon')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'co2_error_by_horizon.png'), dpi=120)
plt.show()

future.to_csv(os.path.join(OUTPUT_DIR, 'co2_forecast.csv'), index=False)
with open(os.path.join(OUTPUT_DIR, 'co2_summary.json'), 'w') as f:
    json.dump({'best_model': best, 'mae': scores['MAE'].round(4).to_dict()}, f, indent=2)
print("Saved: co2_forecast.png, co2_error_by_horizon.png, co2_forecast.csv, co2_summary.json")
