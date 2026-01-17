#!/usr/bin/env python3
"""
Matplotlib and Seaborn Visualization
====================================
Demonstrates static visualizations with Matplotlib and Seaborn.

Matplotlib: https://matplotlib.org/
Seaborn: https://seaborn.pydata.org/

Note: Run this in Jupyter for interactive plots, or use plt.savefig() to save.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")

# =============================================================================
# Matplotlib Basics
# =============================================================================
print("=" * 60)
print("Matplotlib Basic Plots")
print("=" * 60)

# Line plot
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: Simple line plot
x = np.linspace(0, 10, 100)
axes[0, 0].plot(x, np.sin(x), label="sin(x)", linewidth=2)
axes[0, 0].plot(x, np.cos(x), label="cos(x)", linewidth=2)
axes[0, 0].set_xlabel("x")
axes[0, 0].set_ylabel("y")
axes[0, 0].set_title("Trigonometric Functions")
axes[0, 0].legend()

# Plot 2: Scatter plot
np.random.seed(42)
x_scatter = np.random.randn(100)
y_scatter = 2 * x_scatter + np.random.randn(100) * 0.5
colors = np.random.rand(100)
sizes = np.abs(np.random.randn(100)) * 100

scatter = axes[0, 1].scatter(x_scatter, y_scatter, c=colors, s=sizes, alpha=0.6)
axes[0, 1].set_xlabel("X")
axes[0, 1].set_ylabel("Y")
axes[0, 1].set_title("Scatter Plot with Color and Size")
plt.colorbar(scatter, ax=axes[0, 1])

# Plot 3: Bar plot
categories = ["A", "B", "C", "D", "E"]
values = [23, 45, 56, 78, 32]
bars = axes[1, 0].bar(categories, values, color=sns.color_palette("husl", 5))
axes[1, 0].set_xlabel("Category")
axes[1, 0].set_ylabel("Value")
axes[1, 0].set_title("Bar Chart")

# Add value labels on bars
for bar, val in zip(bars, values):
    axes[1, 0].text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 1,
        str(val),
        ha="center",
    )

# Plot 4: Histogram
data = np.random.normal(100, 15, 1000)
axes[1, 1].hist(data, bins=30, edgecolor="black", alpha=0.7)
axes[1, 1].axvline(np.mean(data), color="red", linestyle="--", label=f"Mean: {np.mean(data):.1f}")
axes[1, 1].set_xlabel("Value")
axes[1, 1].set_ylabel("Frequency")
axes[1, 1].set_title("Histogram with Mean Line")
axes[1, 1].legend()

plt.tight_layout()
plt.savefig("/home/jupyter/examples/output/matplotlib_basics.png", dpi=150, bbox_inches="tight")
print("Saved: matplotlib_basics.png")
plt.close()

# =============================================================================
# Seaborn Statistical Visualizations
# =============================================================================
print("\n" + "=" * 60)
print("Seaborn Statistical Visualizations")
print("=" * 60)

# Create sample dataset
np.random.seed(42)
n = 200
df = pd.DataFrame(
    {
        "group": np.random.choice(["A", "B", "C"], n),
        "x": np.random.randn(n),
        "y": np.random.randn(n),
        "size": np.random.randint(10, 100, n),
        "category": np.random.choice(["Low", "Medium", "High"], n),
    }
)
df["y"] = df["y"] + df["group"].map({"A": 0, "B": 2, "C": 4})

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Plot 1: Distribution plot
sns.histplot(data=df, x="y", hue="group", kde=True, ax=axes[0, 0])
axes[0, 0].set_title("Distribution by Group")

# Plot 2: Box plot
sns.boxplot(data=df, x="group", y="y", hue="category", ax=axes[0, 1])
axes[0, 1].set_title("Box Plot by Group and Category")

# Plot 3: Violin plot
sns.violinplot(data=df, x="group", y="y", ax=axes[0, 2])
axes[0, 2].set_title("Violin Plot")

# Plot 4: Regression plot
sns.regplot(data=df, x="x", y="y", ax=axes[1, 0], scatter_kws={"alpha": 0.5})
axes[1, 0].set_title("Regression Plot")

# Plot 5: Pair relationships (simplified)
sns.scatterplot(data=df, x="x", y="y", hue="group", style="category", ax=axes[1, 1])
axes[1, 1].set_title("Scatter with Multiple Dimensions")

# Plot 6: Count plot
sns.countplot(data=df, x="group", hue="category", ax=axes[1, 2])
axes[1, 2].set_title("Count Plot")

plt.tight_layout()
plt.savefig("/home/jupyter/examples/output/seaborn_stats.png", dpi=150, bbox_inches="tight")
print("Saved: seaborn_stats.png")
plt.close()

# =============================================================================
# Heatmaps and Correlation
# =============================================================================
print("\n" + "=" * 60)
print("Heatmaps and Correlation Matrices")
print("=" * 60)

# Create correlation data
np.random.seed(42)
corr_data = pd.DataFrame(np.random.randn(100, 6), columns=["A", "B", "C", "D", "E", "F"])
corr_data["B"] = corr_data["A"] * 0.7 + np.random.randn(100) * 0.3
corr_data["C"] = corr_data["A"] * -0.5 + np.random.randn(100) * 0.5

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Correlation heatmap
corr_matrix = corr_data.corr()
sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    ax=axes[0],
    square=True,
)
axes[0].set_title("Correlation Matrix Heatmap")

# Clustered heatmap data
cluster_data = np.random.rand(10, 10)
sns.heatmap(cluster_data, cmap="YlGnBu", ax=axes[1], cbar_kws={"label": "Value"})
axes[1].set_title("General Heatmap")

plt.tight_layout()
plt.savefig("/home/jupyter/examples/output/heatmaps.png", dpi=150, bbox_inches="tight")
print("Saved: heatmaps.png")
plt.close()

# =============================================================================
# Time Series Visualization
# =============================================================================
print("\n" + "=" * 60)
print("Time Series Visualization")
print("=" * 60)

# Create time series data
dates = pd.date_range("2024-01-01", periods=365, freq="D")
ts_data = pd.DataFrame(
    {
        "date": dates,
        "value": np.cumsum(np.random.randn(365)) + 100,
        "category": np.tile(["Product A", "Product B"], 183)[:365],
    }
)

fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# Line plot with confidence interval
ts_grouped = ts_data.groupby("date")["value"].mean()
rolling_mean = ts_grouped.rolling(window=7).mean()
rolling_std = ts_grouped.rolling(window=7).std()

axes[0].plot(ts_grouped.index, ts_grouped.values, alpha=0.3, label="Daily")
axes[0].plot(rolling_mean.index, rolling_mean.values, linewidth=2, label="7-day MA")
axes[0].fill_between(
    rolling_mean.index,
    rolling_mean - 2 * rolling_std,
    rolling_mean + 2 * rolling_std,
    alpha=0.2,
    label="95% CI",
)
axes[0].set_xlabel("Date")
axes[0].set_ylabel("Value")
axes[0].set_title("Time Series with Moving Average and Confidence Interval")
axes[0].legend()

# Monthly aggregation
monthly = ts_data.set_index("date").resample("ME")["value"].agg(["mean", "std"])
axes[1].bar(monthly.index, monthly["mean"], width=20, yerr=monthly["std"], capsize=3)
axes[1].set_xlabel("Month")
axes[1].set_ylabel("Average Value")
axes[1].set_title("Monthly Aggregation with Error Bars")

plt.tight_layout()
plt.savefig("/home/jupyter/examples/output/timeseries_viz.png", dpi=150, bbox_inches="tight")
print("Saved: timeseries_viz.png")
plt.close()

print("\n" + "=" * 60)
print("Matplotlib/Seaborn examples complete!")
print("Check the 'output' directory for saved plots.")
print("=" * 60)
