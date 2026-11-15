<!-- GENERATED FILE — do not edit by hand. Source: targets/matrix.toml and the example titles. -->
# Examples in `ghcr.io/wlame/jupyter-docker:deeplearn`

Deep learning with PyTorch and TensorFlow.

Open a notebook and run it top to bottom (**Run > Run All Cells**). Files an
example writes go to `output/` in this folder. Your own data belongs in
`/home/jupyter/data`: mount a host folder there with
`-v "$(pwd)/data:/home/jupyter/data"`.

| Notebook | What it covers | Image it comes from |
|---|---|---|
| [10_deep_learning_pytorch](10_deep_learning_pytorch.ipynb) | PyTorch Deep Learning Basics | `deeplearn` |
| [11_deep_learning_tensorflow](11_deep_learning_tensorflow.ipynb) | TensorFlow and Keras Deep Learning Basics | `deeplearn` |
| [25_lightning_onnx](25_lightning_onnx.ipynb) | Training Loops and Deployment: Lightning, torchmetrics, Accelerate, einops, ONNX | `deeplearn` |
| [42_same_network_torch_keras](42_same_network_torch_keras.ipynb) | The Same Network in PyTorch and in Keras | `deeplearn` |
| [01_numpy_scipy_basics](01_numpy_scipy_basics.ipynb) | NumPy and SciPy Basics | `scientific` |
| [02_pandas_data_analysis](02_pandas_data_analysis.ipynb) | Pandas Data Analysis | `scientific` |
| [09_machine_learning](09_machine_learning.ipynb) | Machine Learning Examples | `ml` |
| [17_scipy_signal_processing](17_scipy_signal_processing.ipynb) | SciPy: Signal Processing and Optimization | `scientific` |
| [21_polars_duckdb_xarray](21_polars_duckdb_xarray.ipynb) | Modern Tabular and Array Tools: polars, DuckDB, Arrow, xarray, pint | `scientific` |
| [24_ml_explain_and_uncertainty](24_ml_explain_and_uncertainty.ipynb) | Explaining, Bounding, and Shipping Models: CatBoost, SHAP, MAPIE, UMAP, skrub, skops, ONNX | `ml` |
| [33_lorenz_numba_dask](33_lorenz_numba_dask.ipynb) | Lorenz Attractor: SymPy, Numba, FFT, and Dask | `scientific` |
| [38_tabular_ml_end_to_end](38_tabular_ml_end_to_end.ipynb) | Tabular ML End to End: Breast Cancer Diagnosis | `ml` |
| [50_reliable_api_client](50_reliable_api_client.ipynb) | Calling an API Reliably: Retries, Validation, and Logging | `base` |
| [51_parse_web_and_config](51_parse_web_and_config.ipynb) | Parsing the Formats Data Arrives In: HTML, XML, YAML, .env, and Dates | `base` |

Each notebook has a matching `.py` file with the same code, which you can run
with `python <name>.py` or open as a notebook through Jupytext.
