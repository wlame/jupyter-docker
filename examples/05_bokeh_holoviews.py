#!/usr/bin/env python3
"""
Bokeh and HoloViews Visualizations
==================================
Demonstrates interactive visualizations with Bokeh and HoloViews.

Bokeh: https://bokeh.org/
HoloViews: https://holoviews.org/
hvPlot: https://hvplot.holoviz.org/
Panel: https://panel.holoviz.org/

Note: Best viewed in Jupyter notebooks for full interactivity.
"""

import numpy as np
import pandas as pd

# Bokeh imports
from bokeh.plotting import figure, output_file, save
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.palettes import Category10

# HoloViews imports
import holoviews as hv
from holoviews import opts

hv.extension("bokeh")

# =============================================================================
# Bokeh Basics
# =============================================================================
print("=" * 60)
print("Bokeh Basic Plots")
print("=" * 60)

# Create sample data
np.random.seed(42)
n = 100
x = np.random.randn(n)
y = 2 * x + np.random.randn(n) * 0.5
colors = ["blue" if val > 0 else "red" for val in y]
sizes = np.abs(y) * 5 + 5

# Create data source
source = ColumnDataSource(data=dict(x=x, y=y, colors=colors, sizes=sizes))

# Scatter plot with hover
output_file("/home/jupyter/examples/output/bokeh_scatter.html")
p = figure(
    title="Interactive Scatter Plot",
    x_axis_label="X",
    y_axis_label="Y",
    tools="pan,wheel_zoom,box_zoom,reset,hover,save",
    width=800,
    height=400,
)

p.scatter("x", "y", color="colors", size="sizes", alpha=0.6, source=source)

hover = p.select_one(HoverTool)
hover.tooltips = [("X", "@x{0.2f}"), ("Y", "@y{0.2f}")]

save(p)
print("Saved: bokeh_scatter.html")

# Multiple line plot
output_file("/home/jupyter/examples/output/bokeh_lines.html")
x_line = np.linspace(0, 4 * np.pi, 100)
p = figure(
    title="Multiple Time Series",
    x_axis_label="Time",
    y_axis_label="Value",
    width=800,
    height=400,
)

for i, (label, func) in enumerate(
    [("sin", np.sin), ("cos", np.cos), ("tan (clipped)", lambda x: np.clip(np.tan(x), -3, 3))]
):
    p.line(x_line, func(x_line), legend_label=label, line_color=Category10[3][i], line_width=2)

p.legend.location = "top_left"
p.legend.click_policy = "hide"
save(p)
print("Saved: bokeh_lines.html")

# Bar chart with interaction
output_file("/home/jupyter/examples/output/bokeh_bar.html")
categories = ["Product A", "Product B", "Product C", "Product D", "Product E"]
values = [28, 55, 43, 91, 67]
colors = Category10[5]

source = ColumnDataSource(data=dict(categories=categories, values=values, colors=colors))

p = figure(
    x_range=categories,
    title="Sales by Product",
    tools="hover,save",
    width=800,
    height=400,
)
p.vbar(x="categories", top="values", width=0.8, color="colors", source=source)

p.hover.tooltips = [("Product", "@categories"), ("Sales", "@values")]
p.xaxis.major_label_orientation = 0.5
save(p)
print("Saved: bokeh_bar.html")

# =============================================================================
# Bokeh Dashboard Layout
# =============================================================================
print("\n" + "=" * 60)
print("Bokeh Dashboard Layout")
print("=" * 60)

output_file("/home/jupyter/examples/output/bokeh_dashboard.html")

# Create multiple plots
dates = pd.date_range("2024-01-01", periods=50, freq="D")
values = np.cumsum(np.random.randn(50)) + 100

# Time series
p1 = figure(title="Time Series", x_axis_type="datetime", width=400, height=300)
p1.line(dates, values, line_width=2)
p1.circle(dates, values, size=4)

# Histogram
hist, edges = np.histogram(np.random.randn(1000), bins=30)
p2 = figure(title="Distribution", width=400, height=300)
p2.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:], alpha=0.7)

# Scatter
p3 = figure(title="Correlation", width=400, height=300)
x = np.random.randn(100)
y = x * 0.7 + np.random.randn(100) * 0.3
p3.scatter(x, y, alpha=0.6)

# Box plot approximation
p4 = figure(title="Comparison", width=400, height=300, x_range=["A", "B", "C"])
for i, (cat, vals) in enumerate([("A", [20, 30, 25]), ("B", [35, 40, 38]), ("C", [15, 25, 20])]):
    p4.vbar(x=[cat], top=[np.mean(vals)], width=0.5, color=Category10[3][i])

# Layout
layout = column(row(p1, p2), row(p3, p4))
save(layout)
print("Saved: bokeh_dashboard.html")

# =============================================================================
# HoloViews Declarative Visualization
# =============================================================================
print("\n" + "=" * 60)
print("HoloViews Declarative Visualization")
print("=" * 60)

# Create sample DataFrame
df = pd.DataFrame(
    {
        "x": np.random.randn(200),
        "y": np.random.randn(200),
        "category": np.random.choice(["A", "B", "C"], 200),
        "value": np.random.rand(200) * 100,
    }
)
df["y"] = df["y"] + df["category"].map({"A": 0, "B": 2, "C": 4})

# HoloViews Scatter
scatter = hv.Scatter(df, kdims=["x"], vdims=["y", "category", "value"])
scatter = scatter.opts(
    opts.Scatter(color="category", cmap="Category10", size=8, tools=["hover"], width=600, height=400)
)
hv.save(scatter, "/home/jupyter/examples/output/holoviews_scatter.html")
print("Saved: holoviews_scatter.html")

# HoloViews with multiple elements
curve_data = [(x, np.sin(x)) for x in np.linspace(0, 2 * np.pi, 100)]
curve = hv.Curve(curve_data, label="sin(x)")
curve2 = hv.Curve([(x, np.cos(x)) for x in np.linspace(0, 2 * np.pi, 100)], label="cos(x)")

overlay = (curve * curve2).opts(opts.Curve(width=600, height=400, tools=["hover"]))
hv.save(overlay, "/home/jupyter/examples/output/holoviews_overlay.html")
print("Saved: holoviews_overlay.html")

# HoloViews Heatmap
heatmap_data = [(i, j, np.sin(i) * np.cos(j)) for i in range(10) for j in range(10)]
heatmap = hv.HeatMap(heatmap_data).opts(
    opts.HeatMap(colorbar=True, width=500, height=400, cmap="viridis", tools=["hover"])
)
hv.save(heatmap, "/home/jupyter/examples/output/holoviews_heatmap.html")
print("Saved: holoviews_heatmap.html")

# HoloViews Layout
bars = hv.Bars([("A", 10), ("B", 20), ("C", 15)]).opts(opts.Bars(width=300, height=300))
hist = hv.Histogram(np.histogram(np.random.randn(1000), bins=30)).opts(
    opts.Histogram(width=300, height=300)
)
layout = (bars + hist).opts(shared_axes=False)
hv.save(layout, "/home/jupyter/examples/output/holoviews_layout.html")
print("Saved: holoviews_layout.html")

# =============================================================================
# hvPlot - Pandas Integration
# =============================================================================
print("\n" + "=" * 60)
print("hvPlot - Pandas Integration")
print("=" * 60)

import hvplot.pandas  # noqa

# Time series with hvplot
dates = pd.date_range("2024-01-01", periods=100, freq="D")
ts_df = pd.DataFrame(
    {
        "date": dates,
        "value1": np.cumsum(np.random.randn(100)),
        "value2": np.cumsum(np.random.randn(100)),
    }
)
ts_df = ts_df.set_index("date")

plot = ts_df.hvplot.line(title="Time Series with hvPlot", width=700, height=400)
hv.save(plot, "/home/jupyter/examples/output/hvplot_timeseries.html")
print("Saved: hvplot_timeseries.html")

# Scatter with hvplot
scatter_plot = df.hvplot.scatter(
    x="x",
    y="y",
    c="category",
    s="value",
    title="Scatter Plot with hvPlot",
    width=700,
    height=400,
)
hv.save(scatter_plot, "/home/jupyter/examples/output/hvplot_scatter.html")
print("Saved: hvplot_scatter.html")

print("\n" + "=" * 60)
print("Bokeh/HoloViews examples complete!")
print("Open the HTML files in a browser for interactivity.")
print("=" * 60)
