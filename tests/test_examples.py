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
# SCIENTIFIC target — numpy, scipy, pandas, polars, duckdb, xarray, numba, dask
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


@pytest.mark.scientific
def test_example_21_polars_duckdb_xarray():
    """polars, DuckDB, Arrow, xarray, pint, numexpr, bottleneck."""
    run_example(
        '21_polars_duckdb_xarray.py',
        expected_outputs=[
            'xarray_zonal_mean.png',
            'duckdb_sales.parquet',
        ],
    )


@pytest.mark.scientific
def test_example_33_lorenz_numba_dask():
    """SymPy fixed points, Numba RK4, FFT, Dask ensemble on an in-process cluster."""
    run_example(
        '33_lorenz_numba_dask.py',
        expected_outputs=[
            'lorenz_attractor.png',
            'lorenz_spectrum.png',
            'lorenz_divergence.png',
        ],
        timeout=300,
    )


# =============================================================================
# VISUALIZATION target — matplotlib, seaborn, plotly, bokeh, altair, panel, plotnine, datashader, itables
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


@pytest.mark.visualization
def test_example_22_plotnine_tables_datashader():
    """plotnine, great-tables, itables, datashader, vl-convert."""
    run_example(
        '22_plotnine_tables_datashader.py',
        expected_outputs=[
            'plotnine_efficiency.png',
            'great_tables_summary.html',
            'itables_cars.html',
            'datashader_points.png',
            'altair_vlconvert.png',
        ],
    )


# =============================================================================
# DATAIO target — pyarrow, parquet, HDF5, SQLAlchemy, Delta Lake, zarr, netCDF, Ibis, ADBC
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


@pytest.mark.dataio
def test_example_23_modern_data_formats():
    """Delta Lake, Excel via calamine, SPSS, zarr, netCDF, connectorx."""
    run_example(
        '23_modern_data_formats.py',
        expected_outputs=[
            'survey_report.xlsx',
            'survey.sav',
            'temperature.nc',
            'survey.sqlite',
        ],
    )


@pytest.mark.dataio
def test_example_32_dataframe_engines():
    """One query in pandas, Polars, DuckDB, and Ibis over one Parquet file, plus ADBC."""
    run_example(
        '32_dataframe_engines.py',
        expected_outputs=[
            'orders.parquet',
            'engines_answer.csv',
            'engines_timings.json',
            'engines.sqlite',
        ],
        timeout=300,
    )


# =============================================================================
# ML target — scikit-learn, XGBoost, LightGBM, CatBoost, SHAP, MAPIE, UMAP, ONNX
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


@pytest.mark.ml
def test_example_24_ml_explain_and_uncertainty():
    """CatBoost, SHAP, MAPIE, UMAP, skrub, skops, skl2onnx + onnxruntime."""
    run_example(
        '24_ml_explain_and_uncertainty.py',
        expected_outputs=[
            'shap_beeswarm.png',
            'umap_digits.png',
            'ridge.skops',
            'digits_pipeline.onnx',
        ],
        timeout=300,
    )


# =============================================================================
# DEEPLEARN target — PyTorch, TensorFlow/Keras, Lightning, ONNX
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


@pytest.mark.deeplearn
def test_example_25_lightning_onnx():
    """Lightning, torchmetrics, Accelerate, einops, ONNX + onnxruntime."""
    run_example(
        '25_lightning_onnx.py',
        expected_outputs=[
            'lightning_training.png',
            'tiny_mlp.onnx',
        ],
        timeout=300,
    )


# =============================================================================
# VISION target — PIL, OpenCV, scikit-image, timm, kornia, supervision
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
def test_example_13_object_detection_yolo():
    """YOLO object detection (weights pre-baked into the image; runs offline)."""
    run_example(
        '13_object_detection_yolo.py',
        expected_outputs=[
            'yolo_sample_scene.png',
            'yolo_annotated.png',
        ],
        timeout=300,
    )


@pytest.mark.vision
def test_example_26_vision_backbones_kornia():
    """timm, kornia, supervision, OpenCLIP (architectures only)."""
    run_example(
        '26_vision_backbones_kornia.py',
        expected_outputs=[
            'kornia_ops.png',
            'supervision_annotated.png',
        ],
    )


# =============================================================================
# AUDIO target — librosa, torchaudio, soundfile, pedalboard, parselmouth
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


@pytest.mark.audio
def test_example_27_audio_effects_loudness():
    """pedalboard effects and augmentation, pyloudnorm, noisereduce, parselmouth."""
    run_example(
        '27_audio_effects_loudness.py',
        expected_outputs=[
            'pedalboard_fx.wav',
            'audio_effects.png',
        ],
    )


# =============================================================================
# GEOSPATIAL target — cartopy, geopandas, folium, rasterio, H3, OSMnx
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


@pytest.mark.geospatial
def test_example_28_geospatial_raster_h3():
    """rasterio, rioxarray, H3, mapclassify, OSMnx, lonboard."""
    run_example(
        '28_geospatial_raster_h3.py',
        expected_outputs=[
            'elevation.tif',
            'raster_reprojected.png',
            'h3_choropleth.png',
        ],
    )


# =============================================================================
# TIMESERIES target — tsfresh, sktime, statsmodels, prophet, statsforecast, skforecast
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


@pytest.mark.timeseries
def test_example_29_forecasting_toolkit():
    """statsforecast, mlforecast, skforecast, arch, tslearn."""
    run_example(
        '29_forecasting_toolkit.py',
        expected_outputs=[
            'forecast_toolkit.png',
        ],
        timeout=300,
    )


# =============================================================================
# NLP target — spaCy, NLTK, transformers, sentence-transformers, datasets, PEFT
# =============================================================================

@pytest.mark.nlp
def test_example_14_nlp_text_analysis():
    """NER, dependency parsing, VADER sentiment, WordNet, semantic search.

    NLTK data and the sentence-transformers model are pre-baked; runs offline.
    """
    run_example(
        '14_nlp_text_analysis.py',
        expected_outputs=[
            'nlp_sentiment.png',
            'nlp_similarity_heatmap.png',
        ],
        timeout=300,
    )


@pytest.mark.nlp
def test_example_30_nlp_toolkit():
    """rapidfuzz, lingua, datasets, PEFT LoRA, KeyBERT on the baked MiniLM, SentencePiece."""
    run_example(
        '30_nlp_toolkit.py',
        expected_outputs=[
            'nlp_toolkit.json',
        ],
        timeout=300,
    )


# =============================================================================
# SPEECH target — whisper, gTTS, torchaudio, SpeechRecognition, jiwer, silero-vad
# =============================================================================

@pytest.mark.speech
def test_example_19_speech_processing():
    """Whisper ASR, gTTS synthesis, torchaudio spectrogram, waveform visualization.

    The Whisper model is pre-baked; gTTS (external service) skips without network.
    """
    run_example(
        '19_speech_processing.py',
        # speech_gtts_output.mp3 is intentionally NOT asserted: it depends on
        # an external Google service and the example skips it without network.
        expected_outputs=[
            'speech_synthetic_audio.wav',
            'speech_waveforms.png',
        ],
        timeout=300,
    )


@pytest.mark.speech
def test_example_31_speech_metrics_vad():
    """jiwer error rates, silero-vad segments, parselmouth voice quality."""
    run_example(
        '31_speech_metrics_vad.py',
        expected_outputs=[
            'speech_quality.json',
        ],
    )


# =============================================================================
# FACE target — dlib, deepface, face-alignment, mtcnn
# =============================================================================

@pytest.mark.face
@pytest.mark.slow
def test_example_20_face_analysis():
    """dlib HOG detection, DeepFace attributes, face-alignment landmarks.

    face-alignment weights are pre-baked; the asserted outputs run offline.
    Still marked slow because the DeepFace attribute models (~1.5 GB) download
    on first use and its section is skipped without network.
    """
    run_example(
        '20_face_analysis.py',
        expected_outputs=[
            'face_synthetic_input.png',
            'face_detection.png',
            'face_landmarks.png',
        ],
        timeout=300,
    )
