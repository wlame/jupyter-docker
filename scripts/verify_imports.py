#!/usr/bin/env python3
"""
Verify all required packages can be imported successfully.
Run this script inside the container to validate the installation.
"""
import sys

packages = [
    # Core Scientific Computing
    ("numpy", "NumPy"),
    ("scipy", "SciPy"),
    ("pandas", "Pandas"),
    ("IPython", "IPython"),
    ("jupyter", "Jupyter"),
    ("jupyterlab", "JupyterLab"),

    # Visualization
    ("matplotlib", "Matplotlib"),
    ("matplotlib.pyplot", "Matplotlib.pyplot"),
    ("seaborn", "Seaborn"),
    ("plotly", "Plotly"),
    ("plotly.express", "Plotly Express"),
    ("plotly.graph_objects", "Plotly Graph Objects"),
    ("bokeh", "Bokeh"),
    ("bokeh.plotting", "Bokeh Plotting"),
    ("holoviews", "HoloViews"),
    ("hvplot", "hvPlot"),
    ("hvplot.pandas", "hvPlot Pandas"),
    ("altair", "Altair"),
    ("panel", "Panel"),

    # Geospatial
    ("cartopy", "Cartopy"),
    ("cartopy.crs", "Cartopy CRS"),
    ("geopandas", "GeoPandas"),
    ("shapely", "Shapely"),
    ("shapely.geometry", "Shapely Geometry"),
    ("pyproj", "PyProj"),
    ("folium", "Folium"),
    ("geoviews", "GeoViews"),

    # Time Series
    ("tsfresh", "tsfresh"),
    ("tsfresh.feature_extraction", "tsfresh Feature Extraction"),
    ("sktime", "sktime"),
    ("sktime.forecasting", "sktime Forecasting"),
    ("statsmodels", "Statsmodels"),
    ("statsmodels.tsa", "Statsmodels TSA"),
    ("pmdarima", "pmdarima"),
    ("prophet", "Prophet"),

    # Machine Learning
    ("sklearn", "Scikit-learn"),
    ("sklearn.ensemble", "Scikit-learn Ensemble"),
    ("sklearn.model_selection", "Scikit-learn Model Selection"),
    ("xgboost", "XGBoost"),
    ("lightgbm", "LightGBM"),

    # Data I/O - JSON
    ("orjson", "orjson"),
    ("ujson", "ujson"),
    ("simplejson", "simplejson"),
    ("json", "json (stdlib)"),

    # Data I/O - XML
    ("lxml", "lxml"),
    ("lxml.etree", "lxml etree"),
    ("xml.etree.ElementTree", "ElementTree (stdlib)"),
    ("xmltodict", "xmltodict"),
    ("bs4", "BeautifulSoup4"),

    # Data I/O - Other formats
    ("yaml", "PyYAML"),
    ("openpyxl", "openpyxl"),
    ("xlrd", "xlrd"),
    ("pyarrow", "PyArrow"),
    ("pyarrow.parquet", "PyArrow Parquet"),
    ("fastparquet", "fastparquet"),
    ("h5py", "h5py"),
    ("tables", "PyTables"),

    # HTTP Clients
    ("requests", "Requests"),
    ("httpx", "HTTPX"),
    ("aiohttp", "aiohttp"),
    ("urllib3", "urllib3"),

    # Utilities
    ("pydantic", "Pydantic"),
    ("tqdm", "tqdm"),
    ("tqdm.auto", "tqdm.auto"),
    ("loguru", "Loguru"),
    ("dotenv", "python-dotenv"),
    ("joblib", "joblib"),
    ("toolz", "toolz"),
    ("more_itertools", "more-itertools"),
    ("dateutil", "python-dateutil"),
    ("pytz", "pytz"),
    ("pendulum", "pendulum"),
]


def main():
    print("=" * 70)
    print("Package Import Verification")
    print("=" * 70)
    print(f"Python version: {sys.version}")
    print("=" * 70)
    print()

    failed = []
    success = []

    for module, name in packages:
        try:
            __import__(module)
            print(f"  [OK]   {name}")
            success.append(name)
        except ImportError as e:
            print(f"  [FAIL] {name}: {e}")
            failed.append((name, str(e)))

    print()
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"  Total packages: {len(packages)}")
    print(f"  Successful:     {len(success)}")
    print(f"  Failed:         {len(failed)}")
    print()

    if failed:
        print("Failed packages:")
        for name, error in failed:
            print(f"  - {name}: {error}")
        print()
        print("VERIFICATION FAILED")
        sys.exit(1)
    else:
        print("All packages imported successfully!")
        print("VERIFICATION PASSED")
        sys.exit(0)


if __name__ == "__main__":
    main()
