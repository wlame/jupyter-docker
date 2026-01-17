#!/usr/bin/env python3
"""
Machine Learning Examples
=========================
Demonstrates machine learning with scikit-learn, XGBoost, and LightGBM.

Scikit-learn: https://scikit-learn.org/
XGBoost: https://xgboost.readthedocs.io/
LightGBM: https://lightgbm.readthedocs.io/
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Scikit-learn
from sklearn.datasets import make_classification, make_regression, load_iris
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_squared_error,
    r2_score,
    roc_curve,
    auc,
)
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Gradient boosting libraries
import xgboost as xgb
import lightgbm as lgb

# =============================================================================
# Classification Example
# =============================================================================
print("=" * 60)
print("Classification with Multiple Models")
print("=" * 60)

# Generate classification dataset
X, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=15,
    n_redundant=5,
    n_classes=2,
    random_state=42,
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set: {X_train.shape}")
print(f"Test set: {X_test.shape}")

# Define models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
    "XGBoost": xgb.XGBClassifier(n_estimators=100, random_state=42, eval_metric="logloss"),
    "LightGBM": lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1),
}

# Train and evaluate
results = []
for name, model in models.items():
    # Create pipeline with scaling
    pipeline = Pipeline([("scaler", StandardScaler()), ("model", model)])

    # Fit
    pipeline.fit(X_train, y_train)

    # Predict
    y_pred = pipeline.predict(X_test)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    cv_scores = cross_val_score(pipeline, X, y, cv=5)

    results.append(
        {
            "Model": name,
            "Accuracy": accuracy,
            "CV Mean": cv_scores.mean(),
            "CV Std": cv_scores.std(),
        }
    )

    print(f"\n{name}:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# Results DataFrame
results_df = pd.DataFrame(results)
print("\n" + "=" * 60)
print("Model Comparison:")
print(results_df.to_string(index=False))

# =============================================================================
# Detailed Classification Report
# =============================================================================
print("\n" + "=" * 60)
print("Detailed Classification Report (XGBoost)")
print("=" * 60)

# Best model detailed analysis
xgb_model = xgb.XGBClassifier(n_estimators=100, random_state=42, eval_metric="logloss")
xgb_model.fit(X_train, y_train)
y_pred = xgb_model.predict(X_test)
y_proba = xgb_model.predict_proba(X_test)[:, 1]

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# ROC Curve
axes[0].plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
axes[0].plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
axes[0].set_xlabel("False Positive Rate")
axes[0].set_ylabel("True Positive Rate")
axes[0].set_title("ROC Curve")
axes[0].legend(loc="lower right")

# Feature Importance
importance = pd.Series(xgb_model.feature_importances_).sort_values(ascending=True)
importance.tail(10).plot(kind="barh", ax=axes[1])
axes[1].set_title("Top 10 Feature Importances")
axes[1].set_xlabel("Importance")

plt.tight_layout()
plt.savefig("/home/jupyter/examples/output/ml_classification.png", dpi=150)
print("\nSaved: ml_classification.png")
plt.close()

# =============================================================================
# Regression Example
# =============================================================================
print("\n" + "=" * 60)
print("Regression Example")
print("=" * 60)

# Generate regression dataset
X_reg, y_reg = make_regression(
    n_samples=1000, n_features=10, n_informative=8, noise=20, random_state=42
)

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

# Train models
reg_models = {
    "Ridge": Ridge(alpha=1.0),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "XGBoost": xgb.XGBRegressor(n_estimators=100, random_state=42),
    "LightGBM": lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1),
}

reg_results = []
for name, model in reg_models.items():
    model.fit(X_train_r, y_train_r)
    y_pred_r = model.predict(X_test_r)

    mse = mean_squared_error(y_test_r, y_pred_r)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_r, y_pred_r)

    reg_results.append({"Model": name, "RMSE": rmse, "R2": r2})

    print(f"{name}: RMSE={rmse:.2f}, R2={r2:.4f}")

# =============================================================================
# Hyperparameter Tuning
# =============================================================================
print("\n" + "=" * 60)
print("Hyperparameter Tuning with GridSearchCV")
print("=" * 60)

# Define parameter grid for Random Forest
param_grid = {
    "n_estimators": [50, 100],
    "max_depth": [5, 10, None],
    "min_samples_split": [2, 5],
}

rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(rf, param_grid, cv=3, scoring="accuracy", n_jobs=1)
grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best CV score: {grid_search.best_score_:.4f}")
print(f"Test accuracy: {grid_search.score(X_test, y_test):.4f}")

# =============================================================================
# Clustering Example
# =============================================================================
print("\n" + "=" * 60)
print("Clustering with K-Means")
print("=" * 60)

# Load Iris dataset for clustering
iris = load_iris()
X_iris = iris.data

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_iris)

# K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)

# PCA for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
print(f"Cluster sizes: {np.bincount(clusters)}")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# True labels
scatter1 = axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=iris.target, cmap="viridis")
axes[0].set_title("True Labels")
axes[0].set_xlabel("PC1")
axes[0].set_ylabel("PC2")

# Predicted clusters
scatter2 = axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap="viridis")
centers = pca.transform(kmeans.cluster_centers_)
axes[1].scatter(centers[:, 0], centers[:, 1], c="red", marker="x", s=200, linewidths=3)
axes[1].set_title("K-Means Clusters")
axes[1].set_xlabel("PC1")
axes[1].set_ylabel("PC2")

plt.tight_layout()
plt.savefig("/home/jupyter/examples/output/ml_clustering.png", dpi=150)
print("Saved: ml_clustering.png")
plt.close()

# =============================================================================
# Dimensionality Reduction
# =============================================================================
print("\n" + "=" * 60)
print("Dimensionality Reduction")
print("=" * 60)

# PCA analysis
pca_full = PCA()
pca_full.fit(X_scaled)

# Cumulative explained variance
cumvar = np.cumsum(pca_full.explained_variance_ratio_)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Explained variance
axes[0].bar(range(1, len(pca_full.explained_variance_ratio_) + 1), pca_full.explained_variance_ratio_)
axes[0].set_xlabel("Principal Component")
axes[0].set_ylabel("Explained Variance Ratio")
axes[0].set_title("Variance Explained by Each Component")

# Cumulative variance
axes[1].plot(range(1, len(cumvar) + 1), cumvar, "bo-")
axes[1].axhline(y=0.95, color="r", linestyle="--", label="95% variance")
axes[1].set_xlabel("Number of Components")
axes[1].set_ylabel("Cumulative Explained Variance")
axes[1].set_title("Cumulative Variance Explained")
axes[1].legend()

plt.tight_layout()
plt.savefig("/home/jupyter/examples/output/ml_pca.png", dpi=150)
print("Saved: ml_pca.png")
plt.close()

# Find number of components for 95% variance
n_components_95 = np.argmax(cumvar >= 0.95) + 1
print(f"Components needed for 95% variance: {n_components_95}")

print("\n" + "=" * 60)
print("Machine Learning examples complete!")
print("=" * 60)
