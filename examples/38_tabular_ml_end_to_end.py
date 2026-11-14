#!/usr/bin/env python3
# # Tabular ML End to End: Breast Cancer Diagnosis
#
# A complete, honest workflow on a real dataset that ships with scikit-learn:
# hold out a test set first, compare models with cross-validation inside a
# pipeline, tune the best one with Optuna, check whether its probabilities can
# be trusted (calibration), explain it with SHAP, and save it with skops.
#
# To use your own table, put a CSV in `/home/jupyter/data/` and set `DATA_FILE`
# and `TARGET_COLUMN` below; the rest of the notebook stays the same.
#
# - [scikit-learn](https://scikit-learn.org/stable/)
# - [Optuna](https://optuna.readthedocs.io/)
# - [SHAP](https://shap.readthedocs.io/)
# - [skops](https://skops.readthedocs.io/)

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import shap
import skops.io as sio
from sklearn.calibration import CalibrationDisplay, CalibratedClassifierCV
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, classification_report, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Outputs go to ./output next to the notebook (the kernel's working directory).
OUTPUT_DIR = os.path.abspath('output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATA_FILE = None  # e.g. '/home/jupyter/data/patients.csv'
TARGET_COLUMN = 'target'
SEED = 0
TUNING_TRIALS = 15
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ## Load the data
#
# The bundled dataset has 569 tumours described by 30 measurements; the label is
# malignant (0) or benign (1).

if DATA_FILE:
    frame = pd.read_csv(DATA_FILE)
else:
    frame = load_breast_cancer(as_frame=True).frame
X = frame.drop(columns=[TARGET_COLUMN])
y = frame[TARGET_COLUMN]
print(f"{X.shape[0]} rows, {X.shape[1]} features; class balance: {y.value_counts().to_dict()}")

# ## Hold out a test set before looking at anything else
#
# Every choice below (model, hyperparameters) is made on the training part only,
# so the test score at the end is an unbiased estimate.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=SEED)
folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

# ## Compare three model families with cross-validation
#
# Scaling lives inside the pipeline, so each fold scales with its own training
# statistics and nothing leaks from the validation fold.

candidates = {
    'logistic regression': make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)),
    'random forest': RandomForestClassifier(n_estimators=300, random_state=SEED),
    'gradient boosting': HistGradientBoostingClassifier(random_state=SEED),
}
cv_auc = {}
for name, model in candidates.items():
    scores = cross_val_score(model, X_train, y_train, cv=folds, scoring='roc_auc')
    cv_auc[name] = scores.mean()
    print(f"  {name:20} ROC AUC {scores.mean():.4f} ± {scores.std():.4f}")

# ## Tune gradient boosting with Optuna
#
# Optuna proposes hyperparameters, scores each set with the same cross-validation,
# and concentrates the search where scores are high.


def objective(trial: optuna.Trial) -> float:
    """Mean cross-validated ROC AUC of one hyperparameter set."""
    model = HistGradientBoostingClassifier(
        learning_rate=trial.suggest_float('learning_rate', 0.02, 0.3, log=True),
        max_leaf_nodes=trial.suggest_int('max_leaf_nodes', 8, 64, log=True),
        min_samples_leaf=trial.suggest_int('min_samples_leaf', 5, 40),
        l2_regularization=trial.suggest_float('l2_regularization', 1e-6, 1.0, log=True),
        random_state=SEED,
    )
    return cross_val_score(model, X_train, y_train, cv=folds, scoring='roc_auc').mean()


study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=SEED))
study.optimize(objective, n_trials=TUNING_TRIALS)
print(f"Best CV ROC AUC {study.best_value:.4f} after {TUNING_TRIALS} trials with {study.best_params}")

# ## Calibrate the probabilities
#
# A model can rank cases well and still be over- or under-confident. Isotonic
# calibration (fitted with internal cross-validation) fixes the probabilities
# without changing the ranking much. Points on the diagonal mean "70% predicted,
# 70% observed".

tuned = HistGradientBoostingClassifier(**study.best_params, random_state=SEED).fit(X_train, y_train)
calibrated = CalibratedClassifierCV(
    HistGradientBoostingClassifier(**study.best_params, random_state=SEED), method='isotonic', cv=5
).fit(X_train, y_train)

fig, ax = plt.subplots(figsize=(6, 5))
CalibrationDisplay.from_estimator(tuned, X_test, y_test, n_bins=8, name='tuned', ax=ax)
CalibrationDisplay.from_estimator(calibrated, X_test, y_test, n_bins=8, name='tuned + isotonic', ax=ax)
ax.set_title('Calibration on the held-out test set')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'tabular_calibration.png'), dpi=120)
plt.show()

# ## Final test-set evaluation

probabilities = calibrated.predict_proba(X_test)[:, 1]
test_auc = roc_auc_score(y_test, probabilities)
test_brier = brier_score_loss(y_test, probabilities)
print(f"Test ROC AUC {test_auc:.4f}, Brier score {test_brier:.4f} (lower is better)")
print(classification_report(y_test, calibrated.predict(X_test), target_names=['malignant', 'benign']))

# ## Explain the tuned model with SHAP
#
# Each dot is one test case; its position shows how much that feature pushed the
# prediction towards benign (right) or malignant (left).

explainer = shap.TreeExplainer(tuned)
shap_values = explainer(X_test)
shap.plots.beeswarm(shap_values, max_display=10, show=False)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'tabular_shap.png'), dpi=120)
plt.show()

mean_impact = pd.Series(np.abs(shap_values.values).mean(axis=0), index=X.columns).sort_values(ascending=False)
print("Most influential features:", ", ".join(mean_impact.index[:5]))

# ## Save the model without pickle
#
# skops stores the model in a format that lists the types it contains, so you can
# review them before loading a file from someone else.

model_path = os.path.join(OUTPUT_DIR, 'tabular_model.skops')
sio.dump(calibrated, model_path)
restored = sio.load(model_path, trusted=sio.get_untrusted_types(file=model_path))
assert np.allclose(restored.predict_proba(X_test), calibrated.predict_proba(X_test))

summary = {'cv_roc_auc': cv_auc, 'tuned_params': study.best_params, 'test_roc_auc': test_auc, 'test_brier': test_brier}
with open(os.path.join(OUTPUT_DIR, 'tabular_summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)
print(f"Saved: {os.path.basename(model_path)}, tabular_summary.json")
