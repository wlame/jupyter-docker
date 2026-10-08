#!/usr/bin/env python3
"""
Explaining, Bounding, and Shipping Models: CatBoost, SHAP, MAPIE, UMAP, skrub, skops
=====================================================================================
Fits a CatBoost model on mixed categorical/numeric data, explains it with SHAP,
wraps a regressor in conformal prediction intervals with MAPIE, embeds the
digits dataset with UMAP, builds features from a messy table with skrub, and
saves a model with skops (a safer format than pickle).

CatBoost: https://catboost.ai/docs/
SHAP:     https://shap.readthedocs.io/
MAPIE:    https://mapie.readthedocs.io/
UMAP:     https://umap-learn.readthedocs.io/
skrub:    https://skrub-data.org/
skops:    https://skops.readthedocs.io/
"""

import os

import matplotlib
import numpy as np
import pandas as pd
import shap
import skops.io as sio
import umap
from catboost import CatBoostClassifier
from mapie.regression import SplitConformalRegressor
from sklearn.datasets import load_digits, make_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from skrub import TableVectorizer

matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

rng = np.random.default_rng(seed=0)

# =============================================================================
# CatBoost on mixed categorical and numeric features
# =============================================================================
print("=" * 60)
print("CatBoost: Native Categorical Features")
print("=" * 60)

n = 2_000
customers = pd.DataFrame({
    'plan': rng.choice(['basic', 'plus', 'pro'], size=n),
    'country': rng.choice(['DE', 'FR', 'US', 'JP', 'BR'], size=n),
    'tenure_months': rng.integers(1, 72, size=n),
    'support_tickets': rng.poisson(1.5, size=n),
    'monthly_spend': rng.gamma(2.0, 20.0, size=n).round(2),
})
logit = (-1.0 + 0.6 * customers['support_tickets'] - 0.04 * customers['tenure_months']
         + customers['plan'].map({'basic': 0.8, 'plus': 0.0, 'pro': -0.7}))
churned = (rng.uniform(size=n) < 1 / (1 + np.exp(-logit))).astype(int)

X_train, X_test, y_train, y_test = train_test_split(customers, churned, test_size=0.25, random_state=0)
model = CatBoostClassifier(iterations=200, depth=4, verbose=False, random_seed=0, allow_writing_files=False)
model.fit(X_train, y_train, cat_features=['plan', 'country'])
print(f"Test accuracy: {model.score(X_test, y_test):.3f}")

# =============================================================================
# SHAP — which features drive the predictions
# =============================================================================
print("\n" + "=" * 60)
print("SHAP: Feature Attributions")
print("=" * 60)

explainer = shap.TreeExplainer(model)
shap_values = explainer(X_test)
mean_abs = pd.Series(np.abs(shap_values.values).mean(axis=0), index=X_test.columns).sort_values(ascending=False)
print(mean_abs.round(3).to_string())

shap.plots.beeswarm(shap_values, show=False)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'shap_beeswarm.png'), dpi=120)
plt.close()
print("Saved: shap_beeswarm.png")

# =============================================================================
# MAPIE — conformal prediction intervals
# =============================================================================
print("\n" + "=" * 60)
print("MAPIE: 90% Prediction Intervals")
print("=" * 60)

X_reg, y_reg = make_regression(n_samples=1_500, n_features=6, noise=15.0, random_state=0)
X_fit, X_rest, y_fit, y_rest = train_test_split(X_reg, y_reg, test_size=0.5, random_state=0)
X_conf, X_eval, y_conf, y_eval = train_test_split(X_rest, y_rest, test_size=0.5, random_state=0)

conformal = SplitConformalRegressor(
    RandomForestRegressor(n_estimators=100, random_state=0), confidence_level=0.9, prefit=False
)
conformal.fit(X_fit, y_fit)
conformal.conformalize(X_conf, y_conf)
_, intervals = conformal.predict_interval(X_eval)
lower, upper = intervals[:, 0, 0], intervals[:, 1, 0]
coverage = np.mean((y_eval >= lower) & (y_eval <= upper))
print(f"Empirical coverage: {coverage:.3f} (target 0.90), mean width {np.mean(upper - lower):.1f}")

# =============================================================================
# UMAP — non-linear embedding of handwritten digits
# =============================================================================
print("\n" + "=" * 60)
print("UMAP: Digits Embedding")
print("=" * 60)

digits = load_digits()
embedding = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=0).fit_transform(digits.data)
fig, ax = plt.subplots(figsize=(6, 5))
scatter = ax.scatter(embedding[:, 0], embedding[:, 1], c=digits.target, cmap='tab10', s=5)
ax.legend(*scatter.legend_elements(), title='digit', fontsize=7, loc='best')
ax.set_title('UMAP projection of the digits dataset')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'umap_digits.png'), dpi=120)
plt.close()
print(f"Embedded {digits.data.shape} -> {embedding.shape}; saved umap_digits.png")

# =============================================================================
# skrub — features from a messy table
# =============================================================================
print("\n" + "=" * 60)
print("skrub: TableVectorizer")
print("=" * 60)

messy = pd.DataFrame({
    'signup': pd.date_range('2025-01-01', periods=6, freq='17D').astype(str),
    'job_title': ['Data Scientist', 'data scientist ', 'ML Engineer', 'Analyst', 'Senior Analyst', 'ml engineer'],
    'salary': ['85000', '87000', '99000', '61000', '72000', '101000'],
    'remote': ['yes', 'no', 'yes', 'no', 'yes', 'yes'],
})
features = TableVectorizer().fit_transform(messy)
print(f"{messy.shape[1]} raw columns -> {features.shape[1]} numeric features")
print(list(features.columns)[:8])

# =============================================================================
# skops — safe model persistence
# =============================================================================
print("\n" + "=" * 60)
print("skops: Save and Load Without Pickle")
print("=" * 60)

ridge = Ridge(alpha=1.0).fit(X_fit, y_fit)
model_path = os.path.join(OUTPUT_DIR, 'ridge.skops')
sio.dump(ridge, model_path)
untrusted = sio.get_untrusted_types(file=model_path)
restored = sio.load(model_path, trusted=untrusted)
print(f"Untrusted types to review: {untrusted or 'none'}")
print(f"Predictions identical after reload: {np.allclose(restored.predict(X_eval), ridge.predict(X_eval))}")

print("\nDone.")
