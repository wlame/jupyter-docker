"""
Smoke tests for all data science example scripts.

Each test executes a .py example file as a subprocess and asserts:
  - The script exits with code 0
  - Expected output files (PNGs, HTMLs, data files) are created

Marks correspond to Docker target names; run only the relevant tests
for a given image with: pytest -m <target>

Examples:
    pytest -m scientific       # run scientific target tests
    pytest -m "not slow"       # skip slow / network-dependent tests
    pytest -v                  # verbose output
"""

import pytest
from conftest import run_example


# =============================================================================
# SCIENTIFIC target — numpy, scipy, pandas
# =============================================================================

@pytest.mark.scientific
def test_example_01_numpy_scipy_basics():
    """NumPy arrays, SciPy integration, linear algebra, FFT basics."""
    run_example('01_numpy_scipy_basics.py')


@pytest.mark.scientific
def test_example_02_pandas_data_analysis():
    """Pandas DataFrames, groupby, merge, time-series indexing."""
    run_example('02_pandas_data_analysis.py')


@pytest.mark.scientific
def test_example_17_scipy_signal_processing():
    """FFT, digital filters, spectrogram, optimization, statistics, interpolation."""
    run_example(
        '17_scipy_signal_processing.py',
        expected_outputs=[
            'scipy_fft.png',
            'scipy_filters.png',
            'scipy_spectrogram.png',
            'scipy_optimization.png',
            'scipy_statistics.png',
            'scipy_interpolation.png',
        ],
    )


# =============================================================================
# VISUALIZATION target — matplotlib, seaborn, plotly, bokeh, altair, panel
# =============================================================================

@pytest.mark.visualization
def test_example_03_matplotlib_seaborn():
    """Static charts with matplotlib and seaborn."""
    run_example(
        '03_matplotlib_seaborn_viz.py',
        expected_outputs=[
            'matplotlib_basics.png',
            'seaborn_stats.png',
            'heatmaps.png',
            'timeseries_viz.png',
        ],
    )


@pytest.mark.visualization
def test_example_04_plotly_interactive():
    """Interactive Plotly charts exported to HTML."""
    run_example(
        '04_plotly_interactive.py',
        expected_outputs=[
            'plotly_line.html',
            'plotly_scatter.html',
            'plotly_bar.html',
            'plotly_subplots.html',
        ],
    )


@pytest.mark.visualization
def test_example_05_bokeh_holoviews():
    """Bokeh server-ready charts and HoloViews compositions."""
    run_example(
        '05_bokeh_holoviews.py',
        expected_outputs=[
            'bokeh_scatter.html',
            'bokeh_lines.html',
            'bokeh_bar.html',
            'bokeh_dashboard.html',
        ],
    )


@pytest.mark.visualization
def test_example_16_altair_panel_viz():
    """Altair declarative charts, hvplot, and Panel dashboard."""
    run_example(
        '16_altair_panel_viz.py',
        expected_outputs=[
            'altair_stock_prices.html',
            'altair_scatter_linked.html',
            'altair_correlation.html',
            'hvplot_prices.html',
            'hvplot_returns_hist.html',
            'panel_dashboard.html',
            'altair_grouped_bar.html',
        ],
    )


# =============================================================================
# DATAIO target — pyarrow, parquet, HDF5, SQLAlchemy
# =============================================================================

@pytest.mark.dataio
def test_example_08_data_io_serialization():
    """JSON, CSV, Parquet, HDF5, Excel read/write operations."""
    run_example('08_data_io_serialization.py')


@pytest.mark.dataio
def test_example_18_sqlalchemy_database():
    """SQLAlchemy 2.0 ORM with relationships, window functions, Parquet and HDF5."""
    run_example(
        '18_sqlalchemy_database.py',
        expected_outputs=[
            'transactions.parquet',
            'simulation_data.h5',
        ],
    )


# =============================================================================
# ML target — scikit-learn, XGBoost, LightGBM
# =============================================================================

@pytest.mark.ml
def test_example_09_machine_learning():
    """Classification, regression, clustering, hyperparameter tuning."""
    run_example(
        '09_machine_learning.py',
        expected_outputs=[
            'ml_classification.png',
            'ml_clustering.png',
            'ml_pca.png',
        ],
    )


# =============================================================================
# DEEPLEARN target — PyTorch, TensorFlow/Keras
# =============================================================================

@pytest.mark.deeplearn
@pytest.mark.slow
def test_example_10_deep_learning_pytorch():
    """PyTorch: custom Dataset, DataLoader, training loop, autograd."""
    run_example('10_deep_learning_pytorch.py', timeout=300)


@pytest.mark.deeplearn
@pytest.mark.slow
def test_example_11_deep_learning_tensorflow():
    """TensorFlow/Keras: model definition, training, evaluation."""
    run_example('11_deep_learning_tensorflow.py', timeout=300)


# =============================================================================
# VISION target — PIL, OpenCV, scikit-image
# =============================================================================

@pytest.mark.vision
def test_example_12_image_processing():
    """Pillow, OpenCV, scikit-image, imageio transformations."""
    run_example(
        '12_image_processing.py',
        expected_outputs=[
            'sample_image.png',
            'pil_resized.png',
            'cv_canny_edges.png',
            'ski_sobel.png',
        ],
    )


@pytest.mark.vision
@pytest.mark.slow
def test_example_13_object_detection_yolo():
    """YOLO object detection (downloads model on first run, ~6 MB)."""
    run_example(
        '13_object_detection_yolo.py',
        expected_outputs=[
            'yolo_sample_scene.png',
            'yolo_annotated.png',
        ],
        timeout=300,
    )


# =============================================================================
# AUDIO target — librosa, torchaudio, soundfile
# =============================================================================

@pytest.mark.audio
def test_example_15_audio_analysis():
    """Waveform, HPSS, beat tracking, Mel spectrogram, MFCC, torchaudio."""
    run_example(
        '15_audio_analysis.py',
        expected_outputs=[
            'audio_clip.wav',
            'audio_waveform.png',
            'audio_hpss.png',
            'audio_beats.png',
            'audio_spectral_features.png',
            'audio_beat_features.png',
            'audio_torchaudio.png',
        ],
    )


# =============================================================================
# GEOSPATIAL target — cartopy, geopandas, folium
# =============================================================================

@pytest.mark.geospatial
def test_example_06_geospatial():
    """Cartopy projections, GeoPandas spatial joins, Folium interactive maps."""
    run_example(
        '06_geospatial.py',
        expected_outputs=[
            'cartopy_projections.png',
            'cartopy_cities.png',
            'geopandas_cities.png',
            'folium_basic.html',
        ],
    )


# =============================================================================
# TIMESERIES target — tsfresh, sktime, statsmodels, pmdarima, prophet
# =============================================================================

@pytest.mark.timeseries
def test_example_07_timeseries_analysis():
    """Decomposition, ACF/PACF, ARIMA, sktime forecasting, anomaly detection."""
    run_example(
        '07_timeseries_analysis.py',
        expected_outputs=[
            'ts_decomposition.png',
            'ts_acf_pacf.png',
            'ts_arima.png',
        ],
        timeout=240,
    )


# =============================================================================
# NLP target — spaCy, NLTK, transformers, sentence-transformers
# =============================================================================

@pytest.mark.nlp
@pytest.mark.slow
def test_example_14_nlp_text_analysis():
    """NER, dependency parsing, VADER sentiment, WordNet, semantic search."""
    run_example(
        '14_nlp_text_analysis.py',
        expected_outputs=[
            'nlp_sentiment.png',
            'nlp_similarity_heatmap.png',
        ],
        timeout=300,
    )
