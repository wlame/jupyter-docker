#!/usr/bin/env python3
"""
Plotly Interactive Visualizations
=================================
Demonstrates interactive visualizations with Plotly.

Plotly: https://plotly.com/python/

Note: Best viewed in Jupyter notebooks for full interactivity.
HTML files can be opened in any web browser.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# =============================================================================
# Plotly Express Quick Plots
# =============================================================================
print("=" * 60)
print("Plotly Express Quick Plots")
print("=" * 60)

# Create sample data
np.random.seed(42)
df = pd.DataFrame(
    {
        "date": pd.date_range("2024-01-01", periods=100, freq="D"),
        "sales": np.random.randint(100, 500, 100),
        "profit": np.random.randint(10, 100, 100),
        "region": np.random.choice(["North", "South", "East", "West"], 100),
        "category": np.random.choice(["Electronics", "Clothing", "Food"], 100),
    }
)
df["profit_margin"] = df["profit"] / df["sales"]

# Line chart with time series
fig = px.line(
    df,
    x="date",
    y="sales",
    color="region",
    title="Daily Sales by Region",
    labels={"sales": "Sales ($)", "date": "Date"},
)
fig.write_html("/home/jupyter/examples/output/plotly_line.html")
print("Saved: plotly_line.html")

# Scatter plot with multiple dimensions
fig = px.scatter(
    df,
    x="sales",
    y="profit",
    color="region",
    size="profit_margin",
    hover_data=["category", "date"],
    title="Sales vs Profit by Region",
    labels={"sales": "Sales ($)", "profit": "Profit ($)"},
)
fig.write_html("/home/jupyter/examples/output/plotly_scatter.html")
print("Saved: plotly_scatter.html")

# Bar chart
agg_df = df.groupby(["region", "category"]).agg({"sales": "sum", "profit": "sum"}).reset_index()
fig = px.bar(
    agg_df,
    x="region",
    y="sales",
    color="category",
    barmode="group",
    title="Total Sales by Region and Category",
)
fig.write_html("/home/jupyter/examples/output/plotly_bar.html")
print("Saved: plotly_bar.html")

# =============================================================================
# Advanced Plotly Graph Objects
# =============================================================================
print("\n" + "=" * 60)
print("Advanced Plotly Graph Objects")
print("=" * 60)

# Subplots
fig = make_subplots(
    rows=2,
    cols=2,
    subplot_titles=("Line Chart", "Histogram", "Box Plot", "Pie Chart"),
    specs=[
        [{"type": "scatter"}, {"type": "histogram"}],
        [{"type": "box"}, {"type": "pie"}],
    ],
)

# Line chart
x = np.linspace(0, 10, 100)
fig.add_trace(go.Scatter(x=x, y=np.sin(x), name="sin(x)", mode="lines"), row=1, col=1)
fig.add_trace(go.Scatter(x=x, y=np.cos(x), name="cos(x)", mode="lines"), row=1, col=1)

# Histogram
data = np.random.normal(0, 1, 1000)
fig.add_trace(go.Histogram(x=data, name="Normal Distribution", nbinsx=30), row=1, col=2)

# Box plot
for region in df["region"].unique():
    fig.add_trace(
        go.Box(y=df[df["region"] == region]["sales"], name=region), row=2, col=1
    )

# Pie chart
pie_data = df.groupby("category")["sales"].sum()
fig.add_trace(
    go.Pie(labels=pie_data.index, values=pie_data.values, name="Sales"), row=2, col=2
)

fig.update_layout(height=800, title_text="Dashboard with Multiple Chart Types")
fig.write_html("/home/jupyter/examples/output/plotly_subplots.html")
print("Saved: plotly_subplots.html")

# =============================================================================
# 3D Visualizations
# =============================================================================
print("\n" + "=" * 60)
print("3D Visualizations")
print("=" * 60)

# 3D Scatter
fig = px.scatter_3d(
    df,
    x="sales",
    y="profit",
    z="profit_margin",
    color="region",
    size="sales",
    title="3D Sales Analysis",
)
fig.write_html("/home/jupyter/examples/output/plotly_3d_scatter.html")
print("Saved: plotly_3d_scatter.html")

# 3D Surface
x = np.linspace(-5, 5, 50)
y = np.linspace(-5, 5, 50)
X, Y = np.meshgrid(x, y)
Z = np.sin(np.sqrt(X**2 + Y**2))

fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, colorscale="Viridis")])
fig.update_layout(
    title="3D Surface Plot",
    scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
)
fig.write_html("/home/jupyter/examples/output/plotly_3d_surface.html")
print("Saved: plotly_3d_surface.html")

# =============================================================================
# Animated Visualizations
# =============================================================================
print("\n" + "=" * 60)
print("Animated Visualizations")
print("=" * 60)

# Create animation data
animation_df = []
for month in range(1, 13):
    for region in ["North", "South", "East", "West"]:
        animation_df.append(
            {
                "month": month,
                "region": region,
                "sales": np.random.randint(1000, 5000) + month * 100,
                "profit": np.random.randint(100, 500) + month * 10,
            }
        )
animation_df = pd.DataFrame(animation_df)

fig = px.scatter(
    animation_df,
    x="sales",
    y="profit",
    color="region",
    size="sales",
    animation_frame="month",
    animation_group="region",
    range_x=[800, 6500],
    range_y=[50, 700],
    title="Sales Evolution Over Months (Animated)",
)
fig.write_html("/home/jupyter/examples/output/plotly_animated.html")
print("Saved: plotly_animated.html")

# =============================================================================
# Financial Charts
# =============================================================================
print("\n" + "=" * 60)
print("Financial Charts")
print("=" * 60)

# Candlestick chart
dates = pd.date_range("2024-01-01", periods=60, freq="D")
np.random.seed(42)
price = 100
ohlc_data = []
for date in dates:
    open_price = price
    close_price = price + np.random.randn() * 2
    high_price = max(open_price, close_price) + abs(np.random.randn())
    low_price = min(open_price, close_price) - abs(np.random.randn())
    ohlc_data.append(
        {
            "date": date,
            "open": open_price,
            "high": high_price,
            "low": low_price,
            "close": close_price,
        }
    )
    price = close_price

ohlc_df = pd.DataFrame(ohlc_data)

fig = go.Figure(
    data=[
        go.Candlestick(
            x=ohlc_df["date"],
            open=ohlc_df["open"],
            high=ohlc_df["high"],
            low=ohlc_df["low"],
            close=ohlc_df["close"],
        )
    ]
)
fig.update_layout(title="Stock Price Candlestick Chart", xaxis_rangeslider_visible=False)
fig.write_html("/home/jupyter/examples/output/plotly_candlestick.html")
print("Saved: plotly_candlestick.html")

print("\n" + "=" * 60)
print("Plotly examples complete!")
print("Open the HTML files in a browser for interactivity.")
print("=" * 60)
