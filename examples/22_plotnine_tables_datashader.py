#!/usr/bin/env python3
"""
Grammar of Graphics, Tables, and Big-Data Rendering
===================================================
Demonstrates plotnine (ggplot2 grammar), great-tables (publication tables),
datashader (rasterizing millions of points), and vl-convert (static export of
Altair charts without a browser). ipympl and jupyter-bokeh are JupyterLab
widget renderers: use `%matplotlib widget` for interactive matplotlib, and
`jupyter_bokeh.BokehModel` to embed Bokeh/Panel objects as live widgets.

plotnine:      https://plotnine.org/
great-tables:  https://posit-dev.github.io/great-tables/
datashader:    https://datashader.org/
vl-convert:    https://github.com/vega/vl-convert
"""

import os

import altair as alt
import datashader as ds
import datashader.transfer_functions as tf
import numpy as np
import pandas as pd
import vl_convert as vlc
from great_tables import GT, loc, style
from plotnine import aes, facet_wrap, geom_point, geom_smooth, ggplot, labs, theme_minimal

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

rng = np.random.default_rng(seed=0)

# =============================================================================
# plotnine — layered grammar of graphics
# =============================================================================
print("=" * 60)
print("plotnine: Faceted Scatter with Trend Lines")
print("=" * 60)

n = 300
cars = pd.DataFrame({
    'weight': rng.uniform(1.0, 2.5, size=n),
    'fuel': rng.choice(['petrol', 'diesel', 'hybrid'], size=n),
})
efficiency = {'petrol': 18.0, 'diesel': 22.0, 'hybrid': 28.0}
cars['km_per_l'] = cars['fuel'].map(efficiency) - 5 * cars['weight'] + rng.normal(scale=1.2, size=n)

plot = (
    ggplot(cars, aes('weight', 'km_per_l', color='fuel'))
    + geom_point(alpha=0.6)
    + geom_smooth(method='lm')
    + facet_wrap('~fuel')
    + labs(x='Weight (t)', y='km per litre', title='Fuel efficiency vs. weight')
    + theme_minimal()
)
plot.save(os.path.join(OUTPUT_DIR, 'plotnine_efficiency.png'), width=9, height=3.5, dpi=120, verbose=False)
print("Saved: plotnine_efficiency.png")

# =============================================================================
# great-tables — presentation-quality summary table
# =============================================================================
print("\n" + "=" * 60)
print("great-tables: Summary Table")
print("=" * 60)

summary = (
    cars.groupby('fuel')
    .agg(cars=('weight', 'size'), mean_weight=('weight', 'mean'), mean_km_per_l=('km_per_l', 'mean'))
    .reset_index()
)
table = (
    GT(summary, rowname_col='fuel')
    .tab_header(title='Fleet summary', subtitle='Synthetic data')
    .fmt_number(columns=['mean_weight', 'mean_km_per_l'], decimals=2)
    .cols_label(cars='Cars', mean_weight='Mean weight (t)', mean_km_per_l='Mean km/l')
    .tab_style(style=style.text(weight='bold'), locations=loc.body(columns='mean_km_per_l'))
)
with open(os.path.join(OUTPUT_DIR, 'great_tables_summary.html'), 'w') as f:
    f.write(table.as_raw_html())
print(summary.to_string(index=False))
print("Saved: great_tables_summary.html")

# =============================================================================
# datashader — two million points rendered to a raster
# =============================================================================
print("\n" + "=" * 60)
print("datashader: 2 Million Points")
print("=" * 60)

n_points = 2_000_000
centers = np.array([[0, 0], [3, 3], [-3, 2]])
which = rng.integers(0, len(centers), size=n_points)
points = pd.DataFrame(centers[which] + rng.normal(size=(n_points, 2)), columns=['x', 'y'])

canvas = ds.Canvas(plot_width=600, plot_height=450)
aggregate = canvas.points(points, 'x', 'y')
image = tf.shade(aggregate, cmap=['#e0f3f8', '#08306b'], how='log')
image.to_pil().save(os.path.join(OUTPUT_DIR, 'datashader_points.png'))
print(f"Aggregated {n_points:,} points into a {aggregate.shape} grid")
print("Saved: datashader_points.png")

# =============================================================================
# vl-convert — static PNG/SVG export of an Altair chart
# =============================================================================
print("\n" + "=" * 60)
print("vl-convert: Static Altair Export")
print("=" * 60)

chart = (
    alt.Chart(summary)
    .mark_bar()
    .encode(x=alt.X('fuel:N', title='Fuel'), y=alt.Y('mean_km_per_l:Q', title='Mean km/l'), color='fuel:N')
    .properties(width=300, height=200, title='Mean efficiency by fuel')
)
png = vlc.vegalite_to_png(chart.to_json(), scale=2)
with open(os.path.join(OUTPUT_DIR, 'altair_vlconvert.png'), 'wb') as f:
    f.write(png)
print(f"Saved: altair_vlconvert.png ({len(png):,} bytes)")

print("\nDone.")
