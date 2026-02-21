#!/usr/bin/env python3
"""
Altair and Panel: Declarative and Dashboard Visualization
=========================================================
Demonstrates declarative charts with Altair and composable dashboards with Panel.

Altair: https://altair-viz.github.io/
Panel:  https://panel.holoviz.org/
hvplot: https://hvplot.holoviz.org/

Note: Interactive Panel dashboards require a running server or Jupyter.
      This script saves static exports of the charts.
"""

import os
import json

import numpy as np
import pandas as pd

import altair as alt
import panel as pn
import hvplot.pandas  # registers .hvplot accessor
import holoviews as hv

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# Generate Sample Datasets
# =============================================================================
rng = np.random.default_rng(seed=42)

# Stock-like time series for 4 companies
dates = pd.date_range('2023-01-01', periods=252, freq='B')
companies = ['TechCorp', 'EnergyBiz', 'HealthInc', 'RetailCo']
prices = pd.DataFrame(
    {
        company: 100 * np.cumprod(1 + rng.normal(loc=0.0003, scale=0.015, size=len(dates)))
        for company in companies
    },
    index=dates,
)
prices.index.name = 'date'

# Tidy (long) format for Altair
prices_long = prices.reset_index().melt(id_vars='date', var_name='company', value_name='price')
prices_long['return'] = prices_long.groupby('company')['price'].pct_change()

# Scatter dataset: performance vs. market cap
n_stocks = 80
scatter_df = pd.DataFrame(
    {
        'revenue_growth': rng.normal(0.12, 0.08, n_stocks),
        'profit_margin': rng.normal(0.15, 0.07, n_stocks),
        'market_cap_B': rng.lognormal(mean=3, sigma=1.2, size=n_stocks),
        'sector': rng.choice(['Tech', 'Finance', 'Health', 'Energy', 'Consumer'], size=n_stocks),
        'ticker': [f"S{i:03d}" for i in range(n_stocks)],
    }
)

# Monthly aggregations
monthly = prices.resample('ME').last()
monthly_returns = monthly.pct_change().dropna()

# =============================================================================
# Altair: Layered Line Chart with Selection
# =============================================================================
print("=" * 60)
print("Altair: Stock Price Time Series")
print("=" * 60)

# Brush selection for zooming/filtering
brush = alt.selection_interval(encodings=['x'])

# Detail view (upper chart)
detail = (
    alt.Chart(prices_long, title="Stock Prices — Click and drag to select a time window")
    .mark_line(strokeWidth=1.8)
    .encode(
        x=alt.X('date:T', title='Date'),
        y=alt.Y('price:Q', title='Price (USD)', scale=alt.Scale(zero=False)),
        color=alt.Color('company:N', legend=alt.Legend(title='Company')),
        tooltip=['date:T', 'company:N', alt.Tooltip('price:Q', format='.2f')],
    )
    .properties(width=700, height=280)
    .transform_filter(brush)
)

# Overview / navigator (lower chart)
overview = (
    alt.Chart(prices_long)
    .mark_line(strokeWidth=1.2, opacity=0.7)
    .encode(
        x=alt.X('date:T', title=''),
        y=alt.Y('price:Q', title='Price', axis=alt.Axis(labels=False)),
        color=alt.Color('company:N', legend=None),
    )
    .properties(width=700, height=80, title='Drag to zoom')
    .add_params(brush)
)

stock_chart = detail & overview
stock_chart.save(os.path.join(OUTPUT_DIR, 'altair_stock_prices.html'))
print("Saved: altair_stock_prices.html  (interactive)")

# =============================================================================
# Altair: Scatter with Brush-Linked Histogram
# =============================================================================
print("\nAltair: Linked Scatter and Histogram")

color_scale = alt.Scale(
    domain=['Tech', 'Finance', 'Health', 'Energy', 'Consumer'],
    range=['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B2'],
)

point_select = alt.selection_point(fields=['sector'], bind='legend')
brush_scatter = alt.selection_interval()

scatter = (
    alt.Chart(scatter_df, title="Profit Margin vs Revenue Growth")
    .mark_circle(opacity=0.75, stroke='white', strokeWidth=0.5)
    .encode(
        x=alt.X('revenue_growth:Q', title='Revenue Growth', axis=alt.Axis(format='%')),
        y=alt.Y('profit_margin:Q', title='Profit Margin', axis=alt.Axis(format='%')),
        size=alt.Size('market_cap_B:Q', scale=alt.Scale(range=[40, 600]), legend=alt.Legend(title='Market Cap ($B)')),
        color=alt.Color('sector:N', scale=color_scale, legend=alt.Legend(title='Sector')),
        tooltip=[
            'ticker:N',
            'sector:N',
            alt.Tooltip('revenue_growth:Q', format='.1%'),
            alt.Tooltip('profit_margin:Q', format='.1%'),
            alt.Tooltip('market_cap_B:Q', format='.1f', title='Market Cap ($B)'),
        ],
        opacity=alt.condition(point_select, alt.value(0.85), alt.value(0.15)),
    )
    .properties(width=420, height=320)
    .add_params(point_select, brush_scatter)
)

histogram = (
    alt.Chart(scatter_df)
    .mark_bar(opacity=0.8)
    .encode(
        x=alt.X('revenue_growth:Q', bin=alt.Bin(maxbins=20), title='Revenue Growth'),
        y=alt.Y('count()', title='Count'),
        color=alt.Color('sector:N', scale=color_scale),
        tooltip=['count()'],
    )
    .transform_filter(brush_scatter)
    .properties(width=420, height=160, title='Distribution of Selected Points')
)

linked_chart = scatter & histogram
linked_chart.save(os.path.join(OUTPUT_DIR, 'altair_scatter_linked.html'))
print("Saved: altair_scatter_linked.html  (interactive)")

# =============================================================================
# Altair: Heatmap (Correlation Matrix)
# =============================================================================
print("\nAltair: Correlation Heatmap")

corr = monthly_returns.corr().reset_index().melt(id_vars='index', var_name='variable', value_name='correlation')
corr.columns = ['company_x', 'company_y', 'correlation']

heatmap = (
    alt.Chart(corr, title="Monthly Return Correlations")
    .mark_rect()
    .encode(
        x=alt.X('company_x:N', title=''),
        y=alt.Y('company_y:N', title=''),
        color=alt.Color(
            'correlation:Q',
            scale=alt.Scale(scheme='redblue', domain=[-1, 1]),
            legend=alt.Legend(title='Correlation'),
        ),
        tooltip=['company_x:N', 'company_y:N', alt.Tooltip('correlation:Q', format='.3f')],
    )
    .properties(width=320, height=300)
)

text_layer = heatmap.mark_text(fontSize=13).encode(
    text=alt.Text('correlation:Q', format='.2f'),
    color=alt.condition(
        alt.datum.correlation > 0.5,
        alt.value('white'),
        alt.value('black'),
    ),
)

(heatmap + text_layer).save(os.path.join(OUTPUT_DIR, 'altair_correlation.html'))
print("Saved: altair_correlation.html  (interactive)")

# =============================================================================
# hvplot: Quick Exploratory Charts
# =============================================================================
print("\n" + "=" * 60)
print("hvplot: Quick Exploratory Visualization")
print("=" * 60)

# hvplot returns HoloViews objects — export as HTML via Panel
price_hvplot = prices.hvplot.line(
    title="Stock Prices (hvplot)",
    ylabel="Price (USD)",
    xlabel="Date",
    width=750,
    height=350,
    legend='top_left',
)

return_hist = monthly_returns.hvplot.hist(
    bins=20,
    title="Monthly Return Distribution (hvplot)",
    ylabel="Count",
    xlabel="Monthly Return",
    width=750,
    height=300,
    alpha=0.7,
)

pn.panel(price_hvplot).save(os.path.join(OUTPUT_DIR, 'hvplot_prices.html'))
pn.panel(return_hist).save(os.path.join(OUTPUT_DIR, 'hvplot_returns_hist.html'))
print("Saved: hvplot_prices.html")
print("Saved: hvplot_returns_hist.html")

# =============================================================================
# Panel: Composable Dashboard Layout
# =============================================================================
print("\n" + "=" * 60)
print("Panel: Dashboard Composition")
print("=" * 60)

# KPI cards using Panel panes
def kpi_card(title: str, value: str, delta: str, positive: bool) -> pn.Column:
    color = '#27ae60' if positive else '#e74c3c'
    return pn.pane.HTML(
        f"""
        <div style="background:#f8f9fa;border-radius:8px;padding:16px 20px;
                    border-left:4px solid {color};font-family:sans-serif;">
            <div style="color:#666;font-size:12px;text-transform:uppercase;letter-spacing:1px;">{title}</div>
            <div style="font-size:26px;font-weight:700;color:#2c3e50;margin:4px 0;">{value}</div>
            <div style="color:{color};font-size:13px;">{delta}</div>
        </div>
        """,
        width=200,
        height=100,
    )

# Compute summary KPIs from the data
final_prices = prices.iloc[-1]
initial_prices = prices.iloc[0]
ytd_returns = (final_prices / initial_prices - 1) * 100

kpis = pn.Row(
    *[
        kpi_card(
            title=company,
            value=f"${final_prices[company]:.1f}",
            delta=f"YTD: {ytd_returns[company]:+.1f}%",
            positive=ytd_returns[company] >= 0,
        )
        for company in companies
    ]
)

# Altair chart as Panel pane
stock_pane = pn.pane.Vega(stock_chart.to_dict(), height=400)

dashboard = pn.Column(
    pn.pane.HTML(
        "<h2 style='font-family:sans-serif;color:#2c3e50;margin-bottom:4px;'>"
        "Market Dashboard</h2>"
        "<p style='font-family:sans-serif;color:#7f8c8d;'>Simulated data — Panel + Altair</p>"
    ),
    pn.layout.Divider(),
    pn.pane.HTML("<h4 style='font-family:sans-serif;'>Key Performance Indicators</h4>"),
    kpis,
    pn.layout.Divider(),
    pn.pane.HTML("<h4 style='font-family:sans-serif;'>Price History</h4>"),
    stock_pane,
    sizing_mode='stretch_width',
)

dashboard.save(os.path.join(OUTPUT_DIR, 'panel_dashboard.html'))
print("Saved: panel_dashboard.html  (interactive)")

# =============================================================================
# Altair: Summary Stats — Grouped Bar Chart
# =============================================================================
print("\nAltair: Grouped Statistics Chart")

sector_stats = (
    scatter_df.groupby('sector')[['revenue_growth', 'profit_margin']]
    .mean()
    .reset_index()
    .melt(id_vars='sector', var_name='metric', value_name='value')
)
sector_stats['metric'] = sector_stats['metric'].map(
    {'revenue_growth': 'Revenue Growth', 'profit_margin': 'Profit Margin'}
)

grouped_bar = (
    alt.Chart(sector_stats, title="Average Metrics by Sector")
    .mark_bar(opacity=0.85)
    .encode(
        x=alt.X('sector:N', title='Sector', axis=alt.Axis(labelAngle=0)),
        y=alt.Y('value:Q', title='Value', axis=alt.Axis(format='%')),
        color=alt.Color('metric:N', legend=alt.Legend(title='Metric')),
        xOffset='metric:N',
        tooltip=['sector:N', 'metric:N', alt.Tooltip('value:Q', format='.2%')],
    )
    .properties(width=550, height=320)
)

grouped_bar.save(os.path.join(OUTPUT_DIR, 'altair_grouped_bar.html'))
print("Saved: altair_grouped_bar.html  (interactive)")

# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 60)
print("Altair + Panel visualization complete!")
print("All outputs are interactive HTML files — open in any browser:")
print("  altair_stock_prices.html   — line chart with brush zoom navigator")
print("  altair_scatter_linked.html — scatter with linked histogram")
print("  altair_correlation.html    — annotated correlation heatmap")
print("  hvplot_prices.html         — quick price chart via hvplot")
print("  hvplot_returns_hist.html   — return distribution via hvplot")
print("  panel_dashboard.html       — KPI + chart dashboard via Panel")
print("  altair_grouped_bar.html    — grouped bar chart")
print("=" * 60)
