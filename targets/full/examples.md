<!-- GENERATED FILE — do not edit by hand. Source: targets/matrix.toml and the example titles. -->
# Examples in `ghcr.io/wlame/jupyter-docker:full`

Complete data science environment with all libraries.

Open a notebook and run it top to bottom (**Run > Run All Cells**). Files an
example writes go to `output/` in this folder. Your own data belongs in
`/home/jupyter/data`: mount a host folder there with
`-v "$(pwd)/data:/home/jupyter/data"`.

| Notebook | What it covers | Image it comes from |
|---|---|---|
| [01_numpy_scipy_basics](01_numpy_scipy_basics.ipynb) | NumPy and SciPy Basics | `scientific` |
| [02_pandas_data_analysis](02_pandas_data_analysis.ipynb) | Pandas Data Analysis | `scientific` |
| [03_matplotlib_seaborn_viz](03_matplotlib_seaborn_viz.ipynb) | Matplotlib and Seaborn Visualization | `visualization` |
| [04_plotly_interactive](04_plotly_interactive.ipynb) | Plotly Interactive Visualizations | `visualization` |
| [05_bokeh_holoviews](05_bokeh_holoviews.ipynb) | Bokeh and HoloViews Visualizations | `visualization` |
| [06_geospatial](06_geospatial.ipynb) | Geospatial Data Analysis | `geospatial` |
| [07_timeseries_analysis](07_timeseries_analysis.ipynb) | Time Series Analysis | `timeseries` |
| [08_data_io_serialization](08_data_io_serialization.ipynb) | Data I/O and Serialization | `dataio` |
| [09_machine_learning](09_machine_learning.ipynb) | Machine Learning Examples | `ml` |
| [10_deep_learning_pytorch](10_deep_learning_pytorch.ipynb) | PyTorch Deep Learning Basics | `deeplearn` |
| [11_deep_learning_tensorflow](11_deep_learning_tensorflow.ipynb) | TensorFlow and Keras Deep Learning Basics | `deeplearn` |
| [12_image_processing](12_image_processing.ipynb) | Image Processing Basics | `vision` |
| [13_object_detection_yolo](13_object_detection_yolo.ipynb) | Object Detection with Ultralytics YOLO | `vision` |
| [14_nlp_text_analysis](14_nlp_text_analysis.ipynb) | NLP Text Analysis | `nlp` |
| [15_audio_analysis](15_audio_analysis.ipynb) | Audio Analysis | `audio` |
| [16_altair_panel_viz](16_altair_panel_viz.ipynb) | Altair and Panel: Declarative and Dashboard Visualization | `visualization` |
| [17_scipy_signal_processing](17_scipy_signal_processing.ipynb) | SciPy: Signal Processing and Optimization | `scientific` |
| [18_sqlalchemy_database](18_sqlalchemy_database.ipynb) | SQLAlchemy ORM and Columnar Data Formats | `dataio` |
| [19_speech_processing](19_speech_processing.ipynb) | Speech Processing | `speech` |
| [20_face_analysis](20_face_analysis.ipynb) | Face Analysis | `face` |
| [21_polars_duckdb_xarray](21_polars_duckdb_xarray.ipynb) | Modern Tabular and Array Tools: polars, DuckDB, Arrow, xarray, pint | `scientific` |
| [22_plotnine_tables_datashader](22_plotnine_tables_datashader.ipynb) | Grammar of Graphics, Tables, and Big-Data Rendering | `visualization` |
| [23_modern_data_formats](23_modern_data_formats.ipynb) | Modern Data Formats: Delta Lake, Excel, SPSS, Zarr, netCDF, Fast SQL Loading | `dataio` |
| [24_ml_explain_and_uncertainty](24_ml_explain_and_uncertainty.ipynb) | Explaining, Bounding, and Shipping Models: CatBoost, SHAP, MAPIE, UMAP, skrub, skops, ONNX | `ml` |
| [25_lightning_onnx](25_lightning_onnx.ipynb) | Training Loops and Deployment: Lightning, torchmetrics, Accelerate, einops, ONNX | `deeplearn` |
| [26_vision_backbones_kornia](26_vision_backbones_kornia.ipynb) | Vision Building Blocks: timm, kornia, supervision, OpenCLIP, onnxruntime | `vision` |
| [27_audio_effects_loudness](27_audio_effects_loudness.ipynb) | Audio Effects, Loudness, Augmentation, and Phonetics | `audio` |
| [28_geospatial_raster_h3](28_geospatial_raster_h3.ipynb) | Rasters, Hexagons, and Street Networks: rasterio, rioxarray, H3, mapclassify, OSMnx | `geospatial` |
| [29_forecasting_toolkit](29_forecasting_toolkit.ipynb) | Forecasting Toolkit: statsforecast, mlforecast, skforecast, arch, tslearn | `timeseries` |
| [30_nlp_toolkit](30_nlp_toolkit.ipynb) | NLP Toolkit: rapidfuzz, lingua, Hugging Face datasets, PEFT, KeyBERT, SentencePiece | `nlp` |
| [31_speech_metrics_vad](31_speech_metrics_vad.ipynb) | Speech Evaluation and Segmentation: jiwer, silero-vad, parselmouth | `speech` |
| [32_dataframe_engines](32_dataframe_engines.ipynb) | One Query, Four Engines: pandas, Polars, DuckDB, Ibis, and ADBC | `dataio` |
| [33_lorenz_numba_dask](33_lorenz_numba_dask.ipynb) | Lorenz Attractor: SymPy, Numba, FFT, and Dask | `scientific` |
| [34_optimization](34_optimization.ipynb) | Optimization: Assignment, Vehicle Routing, Portfolios, and Production Planning | `optimization` |
| [35_jax_jit_autodiff](35_jax_jit_autodiff.ipynb) | JAX: JIT, Vectorization, Autodiff, Optax, Flax, and Equinox | `jax` |
| [36_bayesian_modeling](36_bayesian_modeling.ipynb) | Bayesian Modeling: PreliZ, PyMC, nutpie, NumPyro, ArviZ, and Bambi | `probabilistic` |
| [37_retrieval_and_generation](37_retrieval_and_generation.ipynb) | Retrieval and Generation: BM25, Embeddings, FAISS, a Small LM, and TRL | `genai` |
| [38_tabular_ml_end_to_end](38_tabular_ml_end_to_end.ipynb) | Tabular ML End to End: Breast Cancer Diagnosis | `ml` |
| [39_bayesian_ab_test](39_bayesian_ab_test.ipynb) | Bayesian A/B Test: Is the New Checkout Page Better? | `probabilistic` |
| [40_ask_your_documents](40_ask_your_documents.ipynb) | Ask Questions About Your Own Documents | `genai` |
| [41_shift_scheduling](41_shift_scheduling.ipynb) | Shift Scheduling with CP-SAT | `optimization` |
| [42_same_network_torch_keras](42_same_network_torch_keras.ipynb) | The Same Network in PyTorch and in Keras | `deeplearn` |
| [43_coins_and_superpixels](43_coins_and_superpixels.ipynb) | Counting Coins and Segmenting a Photo | `vision` |
| [44_co2_forecast](44_co2_forecast.ipynb) | Forecasting Atmospheric CO₂ | `timeseries` |
| [45_chord_synthesis](45_chord_synthesis.ipynb) | Synthesize a Chord, Find Its Notes, and Rebuild It from a Spectrogram | `audio` |
| [46_review_analysis](46_review_analysis.ipynb) | What Are Customers Saying? Sentiment, Entities, and Topics in Reviews | `nlp` |
| [47_face_anonymization](47_face_anonymization.ipynb) | Anonymize Faces in a Photo | `face` |
| [48_speech_roundtrip](48_speech_roundtrip.ipynb) | Transcribe a Recording and Measure the Error Rate | `speech` |
| [49_messy_csv_to_parquet](49_messy_csv_to_parquet.ipynb) | From a Messy CSV to Clean, Validated Parquet | `dataio` |

Each notebook has a matching `.py` file with the same code, which you can run
with `python <name>.py` or open as a notebook through Jupytext.
