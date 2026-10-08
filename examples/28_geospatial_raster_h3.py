#!/usr/bin/env python3
"""
Rasters, Hexagons, and Street Networks: rasterio, rioxarray, H3, mapclassify, OSMnx
===================================================================================
Writes a georeferenced elevation raster with rasterio, reprojects it with
rioxarray, indexes points into Uber H3 hexagons, classifies values for a
choropleth with mapclassify, builds a street-network graph with OSMnx, and
prepares a GPU-rendered lonboard map layer.

contextily (basemap tiles), geodatasets (sample data), and OSMnx's download
functions need network access, so they are only named here:
`contextily.add_basemap(ax)`, `geodatasets.get_path('nybb')`,
`osmnx.graph_from_place('Piedmont, California')`.

rasterio:    https://rasterio.readthedocs.io/
rioxarray:   https://corteva.github.io/rioxarray/
H3:          https://h3geo.org/
mapclassify: https://pysal.org/mapclassify/
OSMnx:       https://osmnx.readthedocs.io/
lonboard:    https://developmentseed.org/lonboard/
"""

import os

import geopandas as gpd
import h3
import lonboard
import mapclassify
import matplotlib
import numpy as np
import osmnx as ox
import pandas as pd
import rasterio
import rioxarray
from rasterio.transform import from_origin
from shapely.geometry import LineString, Point, Polygon

matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

rng = np.random.default_rng(seed=0)

# =============================================================================
# rasterio — write and read a georeferenced GeoTIFF
# =============================================================================
print("=" * 60)
print("rasterio: Elevation GeoTIFF")
print("=" * 60)

rows, cols = 200, 300
x, y = np.meshgrid(np.linspace(-3, 3, cols), np.linspace(-2, 2, rows))
elevation = (1200 * np.exp(-((x - 1) ** 2 + y**2)) + 800 * np.exp(-((x + 1.5) ** 2 + (y - 0.5) ** 2) / 0.5)
             + rng.normal(scale=15, size=(rows, cols))).astype('float32')
transform = from_origin(west=8.0, north=47.5, xsize=0.005, ysize=0.005)  # degrees, near the Alps
tif_path = os.path.join(OUTPUT_DIR, 'elevation.tif')
with rasterio.open(tif_path, 'w', driver='GTiff', height=rows, width=cols, count=1, dtype='float32',
                   crs='EPSG:4326', transform=transform, compress='deflate') as dst:
    dst.write(elevation, 1)
with rasterio.open(tif_path) as src:
    print(f"CRS {src.crs}, size {src.width}x{src.height}, bounds {tuple(round(b, 3) for b in src.bounds)}")
    summit_row, summit_col = np.unravel_index(np.argmax(src.read(1)), (rows, cols))
    lon, lat = src.xy(summit_row, summit_col)
    print(f"Highest cell at lon {lon:.3f}, lat {lat:.3f}")

# =============================================================================
# rioxarray — reproject the raster to a metric CRS
# =============================================================================
print("\n" + "=" * 60)
print("rioxarray: Reproject to UTM")
print("=" * 60)

raster = rioxarray.open_rasterio(tif_path).squeeze('band', drop=True)
utm = raster.rio.reproject('EPSG:32632')
resolution = utm.rio.resolution()
print(f"Reprojected to {utm.rio.crs}: {utm.shape}, pixel size {abs(resolution[0]):.0f} m")

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
raster.plot(ax=axes[0], cmap='terrain')
axes[0].set_title('EPSG:4326 (degrees)')
utm.where(utm != utm.rio.nodata).plot(ax=axes[1], cmap='terrain')
axes[1].set_title('EPSG:32632 (metres)')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'raster_reprojected.png'), dpi=110)
plt.close()
print("Saved: raster_reprojected.png")

# =============================================================================
# H3 — hexagonal indexing
# =============================================================================
print("\n" + "=" * 60)
print("H3: Hexagon Aggregation")
print("=" * 60)

stations = pd.DataFrame({
    'lat': 47.5 - rng.uniform(0, 1.0, size=500),
    'lon': 8.0 + rng.uniform(0, 1.5, size=500),
    'reading': rng.gamma(3.0, 10.0, size=500),
})
stations['cell'] = [h3.latlng_to_cell(la, lo, 6) for la, lo in zip(stations['lat'], stations['lon'])]
per_cell = stations.groupby('cell')['reading'].agg(['count', 'mean']).reset_index()
print(f"{len(stations)} stations fall into {len(per_cell)} resolution-6 cells "
      f"(~{h3.average_hexagon_area(6, unit='km^2'):.0f} km² each)")
neighbours = h3.grid_disk(per_cell['cell'].iloc[0], 1)
print(f"A cell and its ring-1 neighbours: {len(neighbours)} cells")

# =============================================================================
# mapclassify — choropleth classes
# =============================================================================
print("\n" + "=" * 60)
print("mapclassify: Classification Schemes")
print("=" * 60)

for scheme in (mapclassify.Quantiles, mapclassify.NaturalBreaks, mapclassify.EqualInterval):
    classes = scheme(per_cell['mean'], k=5)
    print(f"{scheme.__name__:14} bins {np.round(classes.bins, 1).tolist()}")

hexagons = gpd.GeoDataFrame(
    per_cell,
    geometry=[Polygon([(lng, lat) for lat, lng in h3.cell_to_boundary(c)]) for c in per_cell['cell']],
    crs='EPSG:4326',
)
ax = hexagons.plot(column='mean', scheme='NaturalBreaks', k=5, cmap='viridis', legend=True,
                   edgecolor='white', linewidth=0.3, figsize=(7, 5))
ax.set_title('Mean reading per H3 cell (natural breaks)')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'h3_choropleth.png'), dpi=110)
plt.close()
print("Saved: h3_choropleth.png")

# =============================================================================
# OSMnx — a street network graph from local geometries
# =============================================================================
print("\n" + "=" * 60)
print("OSMnx: Graph from GeoDataFrames")
print("=" * 60)

coords = {1: (8.540, 47.370), 2: (8.545, 47.372), 3: (8.548, 47.368), 4: (8.542, 47.366)}
nodes = gpd.GeoDataFrame(
    {'x': [c[0] for c in coords.values()], 'y': [c[1] for c in coords.values()]},
    geometry=[Point(c) for c in coords.values()],
    index=pd.Index(list(coords), name='osmid'),
    crs='EPSG:4326',
)
edge_pairs = [(1, 2), (2, 3), (3, 4), (4, 1), (1, 3)]
edges = gpd.GeoDataFrame(
    {'length': [Point(coords[u]).distance(Point(coords[v])) * 111_000 for u, v in edge_pairs]},
    geometry=[LineString([coords[u], coords[v]]) for u, v in edge_pairs],
    index=pd.MultiIndex.from_tuples([(u, v, 0) for u, v in edge_pairs], names=['u', 'v', 'key']),
    crs='EPSG:4326',
)
graph = ox.convert.graph_from_gdfs(nodes, edges)
route = ox.routing.shortest_path(graph, 2, 4, weight='length')
print(f"Graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges; "
      f"shortest path 2 -> 4: {route}")

# =============================================================================
# lonboard — GPU-rendered map layer (displays as a widget in JupyterLab)
# =============================================================================
print("\n" + "=" * 60)
print("lonboard: Scatterplot Layer")
print("=" * 60)

points = gpd.GeoDataFrame(stations, geometry=gpd.points_from_xy(stations['lon'], stations['lat']), crs='EPSG:4326')
layer = lonboard.ScatterplotLayer.from_geopandas(points, get_radius=200, radius_units='meters')
print(f"Layer holds {len(points)} points; in a notebook, display it with lonboard.Map(layer)")

print("\nDone.")
