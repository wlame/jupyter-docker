#!/usr/bin/env python3
"""
Geospatial Data Analysis
========================
Demonstrates geospatial analysis and mapping with Cartopy, GeoPandas, and Folium.

Cartopy: https://scitools.org.uk/cartopy/
GeoPandas: https://geopandas.org/
Folium: https://python-visualization.github.io/folium/
Shapely: https://shapely.readthedocs.io/

Note: Some examples require matplotlib backend for static maps.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Geospatial imports
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
from shapely.geometry import Point, Polygon, LineString
import folium
from folium.plugins import HeatMap, MarkerCluster

# =============================================================================
# Cartopy - Map Projections and Features
# =============================================================================
print("=" * 60)
print("Cartopy Map Projections")
print("=" * 60)

# Create figure with different projections
fig = plt.figure(figsize=(15, 10))

# Plate Carree (standard lat/lon)
ax1 = fig.add_subplot(2, 2, 1, projection=ccrs.PlateCarree())
ax1.set_global()
ax1.add_feature(cfeature.LAND, facecolor="lightgray")
ax1.add_feature(cfeature.OCEAN, facecolor="lightblue")
ax1.add_feature(cfeature.COASTLINE, linewidth=0.5)
ax1.add_feature(cfeature.BORDERS, linestyle=":", linewidth=0.5)
ax1.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5)
ax1.set_title("Plate Carree Projection")

# Orthographic (globe view)
ax2 = fig.add_subplot(2, 2, 2, projection=ccrs.Orthographic(-80, 35))
ax2.set_global()
ax2.add_feature(cfeature.LAND, facecolor="lightgray")
ax2.add_feature(cfeature.OCEAN, facecolor="lightblue")
ax2.add_feature(cfeature.COASTLINE, linewidth=0.5)
ax2.gridlines(linewidth=0.5, alpha=0.5)
ax2.set_title("Orthographic Projection (Americas)")

# Mollweide (equal-area)
ax3 = fig.add_subplot(2, 2, 3, projection=ccrs.Mollweide())
ax3.set_global()
ax3.add_feature(cfeature.LAND, facecolor="lightgray")
ax3.add_feature(cfeature.OCEAN, facecolor="lightblue")
ax3.add_feature(cfeature.COASTLINE, linewidth=0.5)
ax3.gridlines(linewidth=0.5, alpha=0.5)
ax3.set_title("Mollweide Projection")

# Robinson
ax4 = fig.add_subplot(2, 2, 4, projection=ccrs.Robinson())
ax4.set_global()
ax4.add_feature(cfeature.LAND, facecolor="lightgray")
ax4.add_feature(cfeature.OCEAN, facecolor="lightblue")
ax4.add_feature(cfeature.COASTLINE, linewidth=0.5)
ax4.gridlines(linewidth=0.5, alpha=0.5)
ax4.set_title("Robinson Projection")

plt.tight_layout()
plt.savefig("/home/jupyter/examples/output/cartopy_projections.png", dpi=150, bbox_inches="tight")
print("Saved: cartopy_projections.png")
plt.close()

# =============================================================================
# Cartopy - Plotting Data on Maps
# =============================================================================
print("\n" + "=" * 60)
print("Cartopy Data Visualization")
print("=" * 60)

# Sample city data
cities = pd.DataFrame(
    {
        "city": [
            "New York",
            "Los Angeles",
            "Chicago",
            "Houston",
            "Phoenix",
            "London",
            "Paris",
            "Tokyo",
            "Sydney",
            "Mumbai",
        ],
        "lat": [40.7128, 34.0522, 41.8781, 29.7604, 33.4484, 51.5074, 48.8566, 35.6762, -33.8688, 19.0760],
        "lon": [-74.0060, -118.2437, -87.6298, -95.3698, -112.0740, -0.1278, 2.3522, 139.6503, 151.2093, 72.8777],
        "population": [8.3, 3.9, 2.7, 2.3, 1.6, 8.9, 2.1, 13.9, 5.3, 20.4],
    }
)

fig = plt.figure(figsize=(14, 8))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())
ax.set_global()

# Add map features
ax.add_feature(cfeature.LAND, facecolor="#E0E0E0")
ax.add_feature(cfeature.OCEAN, facecolor="#B0D0E8")
ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
ax.add_feature(cfeature.BORDERS, linestyle=":", linewidth=0.3, alpha=0.5)

# Plot cities
scatter = ax.scatter(
    cities["lon"],
    cities["lat"],
    c=cities["population"],
    s=cities["population"] * 20,
    cmap="YlOrRd",
    alpha=0.8,
    edgecolors="black",
    linewidth=0.5,
    transform=ccrs.PlateCarree(),
    zorder=5,
)

# Add city labels
for idx, row in cities.iterrows():
    ax.annotate(
        row["city"],
        xy=(row["lon"], row["lat"]),
        xytext=(5, 5),
        textcoords="offset points",
        fontsize=8,
        transform=ccrs.PlateCarree(),
    )

plt.colorbar(scatter, label="Population (millions)", shrink=0.5)
ax.set_title("Major World Cities by Population", fontsize=14)
plt.savefig("/home/jupyter/examples/output/cartopy_cities.png", dpi=150, bbox_inches="tight")
print("Saved: cartopy_cities.png")
plt.close()

# =============================================================================
# GeoPandas - Vector Data Operations
# =============================================================================
print("\n" + "=" * 60)
print("GeoPandas Vector Operations")
print("=" * 60)

# Create GeoDataFrame from points
geometry = [Point(lon, lat) for lon, lat in zip(cities["lon"], cities["lat"])]
gdf_cities = gpd.GeoDataFrame(cities, geometry=geometry, crs="EPSG:4326")

print("GeoDataFrame info:")
print(gdf_cities.head())
print(f"\nCRS: {gdf_cities.crs}")
print(f"Geometry type: {gdf_cities.geometry.geom_type.unique()}")

# Create polygons (example: bounding boxes around cities)
def create_bbox(point, size=2):
    """Create a bounding box around a point."""
    return Polygon(
        [
            (point.x - size, point.y - size),
            (point.x + size, point.y - size),
            (point.x + size, point.y + size),
            (point.x - size, point.y + size),
        ]
    )


gdf_cities["bbox"] = gdf_cities.geometry.apply(create_bbox)

# Create LineString (example: connecting cities)
coords = list(zip(cities["lon"], cities["lat"]))
route = LineString(coords[:5])  # Connect first 5 cities
print(f"\nRoute length (degrees): {route.length:.2f}")

# Spatial operations
# Buffer around points
gdf_cities["buffer"] = gdf_cities.geometry.buffer(5)  # 5 degree buffer

# Centroid calculation
all_cities_union = gdf_cities.geometry.union_all()
centroid = all_cities_union.centroid
print(f"Centroid of all cities: ({centroid.x:.2f}, {centroid.y:.2f})")

# Distance calculation (in degrees, approximate)
ny = gdf_cities[gdf_cities["city"] == "New York"].geometry.iloc[0]
la = gdf_cities[gdf_cities["city"] == "Los Angeles"].geometry.iloc[0]
print(f"Distance NY to LA (degrees): {ny.distance(la):.2f}")

# Plot with GeoPandas
fig, ax = plt.subplots(figsize=(12, 8))
world = gpd.GeoDataFrame(
    geometry=[Polygon([(-180, -90), (180, -90), (180, 90), (-180, 90)])], crs="EPSG:4326"
)
world.plot(ax=ax, color="lightgray", edgecolor="white")
gdf_cities.plot(
    ax=ax, column="population", cmap="YlOrRd", markersize=gdf_cities["population"] * 10, legend=True
)
ax.set_title("Cities GeoDataFrame Visualization")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
plt.savefig("/home/jupyter/examples/output/geopandas_cities.png", dpi=150, bbox_inches="tight")
print("Saved: geopandas_cities.png")
plt.close()

# =============================================================================
# Folium - Interactive Web Maps
# =============================================================================
print("\n" + "=" * 60)
print("Folium Interactive Maps")
print("=" * 60)

# Basic map centered on a location
m = folium.Map(location=[40, -95], zoom_start=4, tiles="OpenStreetMap")

# Add markers for cities
for idx, row in cities.iterrows():
    popup_text = f"{row['city']}<br>Population: {row['population']}M"
    folium.CircleMarker(
        location=[row["lat"], row["lon"]],
        radius=row["population"] * 2,
        popup=popup_text,
        color="red",
        fill=True,
        fill_color="red",
        fill_opacity=0.6,
    ).add_to(m)

# Add a polygon (continental US approximate bounds)
us_bounds = [
    [49, -125],
    [49, -66],
    [24, -66],
    [24, -125],
]
folium.Polygon(
    locations=us_bounds,
    color="blue",
    weight=2,
    fill=True,
    fill_opacity=0.1,
    popup="Continental US (approximate)",
).add_to(m)

m.save("/home/jupyter/examples/output/folium_basic.html")
print("Saved: folium_basic.html")

# Folium with marker clusters
m_cluster = folium.Map(location=[0, 0], zoom_start=2)

# Generate random points for clustering demo
np.random.seed(42)
random_points = [
    [np.random.uniform(-60, 60), np.random.uniform(-150, 150)] for _ in range(100)
]

marker_cluster = MarkerCluster().add_to(m_cluster)
for point in random_points:
    folium.Marker(location=point, popup=f"Point: {point[0]:.2f}, {point[1]:.2f}").add_to(
        marker_cluster
    )

m_cluster.save("/home/jupyter/examples/output/folium_cluster.html")
print("Saved: folium_cluster.html")

# Folium heatmap
m_heat = folium.Map(location=[40, -95], zoom_start=4)

# Generate heat data (lat, lon, weight)
heat_data = [
    [row["lat"], row["lon"], row["population"]] for idx, row in cities.iterrows()
]

HeatMap(heat_data, radius=50).add_to(m_heat)
m_heat.save("/home/jupyter/examples/output/folium_heatmap.html")
print("Saved: folium_heatmap.html")

# Folium with different tile layers
m_tiles = folium.Map(location=[40.7128, -74.0060], zoom_start=12)

# Add different tile layers
folium.TileLayer("OpenStreetMap").add_to(m_tiles)
folium.TileLayer("cartodbpositron", name="CartoDB Positron").add_to(m_tiles)
folium.TileLayer("cartodbdark_matter", name="CartoDB Dark").add_to(m_tiles)

# Add layer control
folium.LayerControl().add_to(m_tiles)

# Add marker for NYC
folium.Marker(
    location=[40.7128, -74.0060],
    popup="New York City",
    icon=folium.Icon(color="red", icon="info-sign"),
).add_to(m_tiles)

m_tiles.save("/home/jupyter/examples/output/folium_layers.html")
print("Saved: folium_layers.html")

print("\n" + "=" * 60)
print("Geospatial examples complete!")
print("Open HTML files in browser for interactive maps.")
print("=" * 60)
