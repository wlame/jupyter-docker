---
tags:
  - reference
  - targets
---

# Targets

`jupyter-docker` ships as a family of 14 image targets. Each is a separate build
stage in one `Dockerfile`, published to the GitHub Container Registry at:

```
ghcr.io/wlame/jupyter-docker:<target>
```

Every target inherits all of its parent's packages and layers its own on top, so
you pull only the libraries you need — from a ~330 MB `base` to a ~5.3 GB `full`.

```bash
docker pull ghcr.io/wlame/jupyter-docker:scientific
```

Each target is published for every Python version it supports. The plain tag is
the target's default Python; a `-py<X.Y>` suffix picks one explicitly:

```bash
docker pull ghcr.io/wlame/jupyter-docker:scientific          # default (3.14)
docker pull ghcr.io/wlame/jupyter-docker:scientific-py3.13   # Python 3.13
```

Images are rebuilt on every push to `main` (and weekly for OS patches). Alongside
each moving tag, an immutable `-<short-sha>` tag is published for pinning and
rollback:

```bash
docker pull ghcr.io/wlame/jupyter-docker:scientific-py3.14-a1b2c3d
```

!!! note "Version overrides per target"
    A few packages are pinned to older versions in specific targets to satisfy real
    dependency constraints — for example `face` and `full` use OpenCV 4.13, and `speech` and
    `full` use transformers 4.57.6. See [Configuration](configuration.md) for the
    full constraint web.

## Summary

| Target | Inherits from | Python (default first) | Approx. compressed size | Focus |
|---|---|---|---|---|
| `base` | — (root) | 3.14, 3.13 | ~330 MB | Common Python utilities for data science |
| `scientific` | `base` | 3.14, 3.13 | ~530 MB | NumPy, SciPy, Pandas numerical computing |
| `visualization` | `base` | 3.14, 3.13 | ~470 MB | Matplotlib, Seaborn, Plotly, Bokeh charts |
| `dataio` | `base` | 3.14, 3.13 | ~465 MB | Parquet, HDF5, Excel, and database I/O |
| `ml` | `scientific` | 3.14, 3.13 | ~655 MB | scikit-learn, XGBoost, LightGBM |
| `deeplearn` | `ml` | 3.13 | ~3900 MB | PyTorch and TensorFlow |
| `vision` | `base` | 3.14, 3.13 | ~4680 MB | Computer vision and image processing |
| `audio` | `base` | 3.14, 3.13 | ~3360 MB | Audio processing and analysis |
| `geospatial` | `scientific` | 3.14, 3.13 | ~750 MB | Geospatial analysis and mapping |
| `timeseries` | `scientific` | 3.14, 3.13 | ~810 MB | Time series analysis and forecasting |
| `optimization` | `scientific` | 3.14, 3.13 | not yet measured | Linear, integer, convex, routing, scheduling |
| `nlp` | `base` | 3.14, 3.13 | ~3390 MB | Natural language processing |
| `speech` | `base` | 3.14, 3.13 | ~3680 MB | Speech recognition and text-to-speech |
| `face` | `base` | 3.13 | ~4130 MB | Face detection, recognition, and analysis |
| `full` | — (root, union) | 3.13 | ~5260 MB | Everything combined |

Sizes are approximate compressed pull sizes from GHCR (measured before the
October 2026 refresh) and vary between rebuilds. `deeplearn`, `face`, and `full`
are 3.13-only because TensorFlow 2.21 ships no Python 3.14 wheels.
For help choosing, see [Pick a target](../use-cases/pick-a-target.md); for how the
inheritance tree is built, see [Architecture](../concepts/architecture.md).

## base

**Inherits from:** — (root stage on Ubuntu 24.04 with the selected Python)

Base Python environment with common utilities for data science. Every specialized
target builds on this stage, so JupyterLab and the shared utility belt below are
present everywhere.

**Notable libraries:** JupyterLab, IPython, and Jupyter, with the JupyterLab
extensions every image gets — Jupytext (pair notebooks with `.py` files),
jupyterlab-git + nbdime (Git UI and notebook-aware diffs), jupyter-resource-usage
(memory/CPU indicator), and jupyterlab-lsp + python-lsp-server (completion,
hover, diagnostics); `requests` / `httpx` / `aiohttp` HTTP clients; Pydantic;
loguru, rich (terminal formatting), tenacity (retries), and tqdm.

??? note "Full package list (base)"
    aiohttp, beautifulsoup4, httpx, ipython, joblib, jupyter, jupyter-resource-usage,
    jupyterlab, jupyterlab-git, jupyterlab-lsp, jupytext, loguru, lxml,
    more-itertools, nbdime, orjson, pendulum, pip, pydantic, pytest, pytest-timeout,
    python-dateutil, python-dotenv, python-lsp-server, pytz, pyyaml, requests, rich,
    simplejson, tenacity, toolz, tqdm, ujson, xmltodict.

## scientific

**Inherits from:** `base`

Scientific computing with NumPy, SciPy, and Pandas.

**Adds:** NumPy, SciPy, Pandas, Statsmodels, SymPy, and Matplotlib, plus the
modern tabular and array stack — polars, DuckDB, PyArrow (which also backs pandas 3's
default string type), xarray, pint (units), and pandas' numexpr/bottleneck
accelerators. Numba compiles numeric Python functions to machine code, and Dask
(with distributed) runs pandas, NumPy, and xarray work in parallel or out of core.
`Client()` in a notebook links to the Dask dashboard, which opens through
JupyterLab at `/proxy/8787/status` (jupyter-server-proxy; Bokeh renders it), so no
extra port has to be published. System libraries: OpenBLAS, LAPACK, and libgfortran.

## visualization

**Inherits from:** `base`

Data visualization with Matplotlib, Seaborn, Plotly, and Bokeh.

**Adds:** Matplotlib, Seaborn, Plotly, Bokeh, Altair, HoloViews, hvPlot, and Panel,
plus plotnine (ggplot2 grammar), great-tables, datashader (millions of points),
vl-convert (static Altair export), itables (sortable, searchable DataFrame
tables), and the JupyterLab renderers ipympl
(`%matplotlib widget`) and jupyter-bokeh. Because its parent is `base` (not
`scientific`), it re-declares NumPy and Pandas for its own examples. System
libraries: FreeType, libpng, libjpeg-turbo.

## dataio

**Inherits from:** `base`

Data I/O for Parquet, HDF5, Excel, and databases.

**Adds:** PyArrow, fastparquet, h5py, PyTables, openpyxl, xlrd, and SQLAlchemy, plus
polars and DuckDB, Delta Lake tables (deltalake), fast Excel reading (python-calamine)
and formatted writing (xlsxwriter), SPSS/SAS/Stata files (pyreadstat), zarr and
netCDF4, cloud filesystems for fsspec (s3fs, gcsfs, adlfs), database drivers
(psycopg with its binary build, pymysql), connectorx for fast SQL-to-DataFrame
loading, Ibis (one dataframe API compiled to SQL, with its DuckDB backend), and
ADBC drivers for SQLite and PostgreSQL (Arrow tables in and out of a database
without row conversion) (plus NumPy and Pandas). System libraries: HDF5.

## ml

**Inherits from:** `scientific`

Classical machine learning with scikit-learn, XGBoost, and LightGBM.

**Adds:** scikit-learn, XGBoost, LightGBM, imbalanced-learn, and Optuna, plus
CatBoost, SHAP (explanations), MAPIE (conformal prediction intervals), UMAP, skrub
(DataFrame preparation), skops (safe model persistence), and ONNX + onnxruntime
with skl2onnx (export pipelines and serve them without scikit-learn). XGBoost
pulls the ~200 MB `nvidia-nccl-cu13` wheel (distributed GPU only); it stays, because
`deeplearn` and `full` inherit this target's exclusions and torch needs it.

## deeplearn

**Inherits from:** `ml`

Deep learning with PyTorch and TensorFlow.

**Adds:** PyTorch, TorchVision, TorchAudio, TensorFlow, and Keras, plus Lightning,
torchmetrics, Accelerate, einops, and TensorBoard; ONNX + onnxruntime for portable
inference come from `ml`. This is the only target that stacks the full classical-ML and
deep-learning frameworks together. It does not install triton, which segfaults
when TensorFlow is already loaded, so `torch.compile` is unavailable; eager PyTorch,
including on a GPU, is unaffected.

## vision

**Inherits from:** `base`

Computer vision and image processing.

**Adds:** OpenCV (headless), Pillow, scikit-image, imageio, and Ultralytics (YOLO26),
plus timm (backbones), kornia (differentiable image ops), supervision (detection
annotation), OpenCLIP, onnxruntime, and NumPy. The GUI `opencv-python` wheel is excluded so it cannot double-install
`cv2` over the pinned headless build. System libraries: libGL and glib for OpenCV.

!!! note "Pre-baked weights"
    YOLO26n weights are baked into the image so the object-detection example runs
    offline with no download on first use.

## audio

**Inherits from:** `base`

Audio processing and analysis.

**Adds:** PyTorch, TorchAudio, torchcodec, librosa, soundfile, pydub, audioread, and
numba, plus pedalboard (studio effects and fast audio I/O), pyloudnorm (loudness),
noisereduce, and praat-parselmouth (Praat phonetics) (plus NumPy and Matplotlib). System libraries: libsndfile and FFmpeg — the
latter provides the shared libraries torchcodec needs for audio decode.

??? note "Full package list (audio)"
    audioread, librosa, matplotlib, noisereduce, numba, numpy, pedalboard,
    praat-parselmouth, pydub, pyloudnorm, soundfile, torch, torchaudio, torchcodec.

## geospatial

**Inherits from:** `scientific`

Geospatial analysis and mapping.

**Adds:** Cartopy, GeoPandas, Shapely, PyProj, Folium, and GeoViews, plus the raster
stack (rasterio, rioxarray on the inherited xarray), Uber H3, mapclassify,
OSMnx, contextily (basemaps), geodatasets, and lonboard (GPU-rendered maps). System
libraries: GEOS, PROJ (with data), and GDAL.

## timeseries

**Inherits from:** `scientific`

Time series analysis and forecasting.

**Adds:** sktime, tsfresh, pmdarima, and Prophet, plus statsforecast and mlforecast
(Nixtla), skforecast, arch (volatility models), tslearn, the classical-ML family
(scikit-learn, XGBoost, LightGBM, imbalanced-learn, Optuna), and numba.

!!! note "pandas 2.x"
    statsforecast, mlforecast, and skforecast require pandas < 3, so this target
    (and `full`) holds pandas 2.3.3 while the other targets run pandas 3.

??? note "Full package list (timeseries)"
    arch, imbalanced-learn, lightgbm, mlforecast, numba, optuna, pmdarima, prophet,
    scikit-learn, skforecast, sktime, statsforecast, tsfresh, tslearn, xgboost.

## optimization

**Inherits from:** `scientific`

Mathematical optimization: linear, integer, convex, routing, and scheduling.

**Adds:** OR-Tools (CP-SAT, vehicle routing, and embedded SCIP, CBC, HiGHS, GLOP,
and PDLP solvers), CVXPY (convex problems, with Clarabel, OSQP, and SCS), Pyomo,
and PuLP. System package: coinor-cbc, the `cbc` command that PuLP and Pyomo call.

!!! note "No highspy"
    highspy ships a `libhighs.so.1` that clashes with the one OR-Tools bundles:
    whichever loads first breaks the other's import. The image leaves highspy out
    (OR-Tools already embeds HiGHS, and CVXPY solves integer problems through
    SciPy's HiGHS), so PuLP and Pyomo use CBC. Don't `pip install highspy` here.

## nlp

**Inherits from:** `base`

Natural language processing.

**Adds:** spaCy (with the `en_core_web_sm` model), Transformers, sentence-transformers,
NLTK, and tokenizers, plus the Hugging Face training stack (datasets, evaluate,
Accelerate, PEFT), rapidfuzz, lingua (language detection), tiktoken, SentencePiece,
BERTopic, KeyBERT, and PyTorch. Because its parent is `base` (not `scientific`), it
re-declares NumPy and Matplotlib for its own examples. System libraries: OpenBLAS, LAPACK,
and libgfortran.

!!! note "Pre-baked weights"
    NLTK corpora and the MiniLM sentence-embedding model are baked in so the NLP
    example runs offline.

## speech

**Inherits from:** `base`

Speech recognition and text-to-speech synthesis.

**Adds:** openai-whisper, faster-whisper, SpeechRecognition, coqui-tts, gTTS,
piper-tts, pyannote-audio, and SpeechBrain, plus jiwer (WER/CER), silero-vad (voice
activity detection), praat-parselmouth, PyTorch/TorchAudio/torchcodec, and
Transformers. System libraries: libsndfile, FFmpeg, and espeak-ng (for TTS).

!!! note "Version overrides and pre-baked weights"
    coqui-tts requires transformers < 5, so this target holds transformers 4.57.6.
    Whisper tiny weights are baked in so the ASR example runs offline. See
    [Configuration](configuration.md).

??? note "Full package list (speech)"
    coqui-tts, faster-whisper, gtts, jiwer, matplotlib, numba, numpy, openai-whisper,
    piper-tts, praat-parselmouth, pyannote-audio, silero-vad, soundfile, speechbrain,
    speechrecognition, torch, torchaudio, torchcodec, transformers.

## face

**Inherits from:** `base`

Face detection, recognition, analysis, and generation.

**Adds:** DeepFace, dlib, MTCNN, RetinaFace, face-alignment, and diffusers (for face
generation), plus InsightFace and MediaPipe with onnxruntime,
TensorFlow/Keras/tf-keras, PyTorch/TorchVision, OpenCV, Pillow, and scikit-image.
InsightFace and MediaPipe download their models on first use. dlib has no wheel, so it is compiled from source in a throwaway
builder stage and only the finished virtual environment ships in the runtime image.

!!! note "Version overrides and pre-baked weights"
    DeepFace needs the Haar-cascade files removed in OpenCV 5, so this target uses
    OpenCV 4.13. face-alignment weights are baked in; DeepFace's ~1.5 GB attribute
    models are intentionally not baked. See [Configuration](configuration.md).

??? note "Full package list (face)"
    deepface, diffusers, dlib, face-alignment, insightface, keras, matplotlib,
    mediapipe, mtcnn, numpy, onnxruntime, opencv-python-headless, pillow,
    retina-face, scikit-image, tensorflow, tf-keras, torch, torchvision.

## Licenses and large wheels

Most packages are permissively licensed. A few are copyleft, which matters if you
redistribute an image or code linked against them: Ultralytics (`vision`, `full`)
is AGPL-3.0; pedalboard and praat-parselmouth (`audio`, `speech`, `full`) are
GPL-3.0; psycopg (`dataio`, `full`) is LGPL-3.0. Some additions are large on their
own: lingua's language models (~160 MB, `nlp`), CatBoost (~90 MB, `ml`),
connectorx (~55 MB, `dataio`), and XGBoost's `nvidia-nccl-cu13` (~200 MB, `ml`,
`timeseries`).

## full

**Inherits from:** — (root, union of every target)

Complete data science environment with all libraries. `full` is a standalone stage
built directly on `base`: it installs the union of every specialized target's system
libraries and the union of every package in the matrix, rather than inheriting from
a single parent.

Because it merges every stack in one environment, `full` carries the same
constraint-driven pins as the individual targets — OpenCV 4.13 (from `face`) and
transformers 4.57.6 (from `speech`) — plus the holds transformers 4 forces on its
neighbours here: tokenizers 0.22, diffusers 0.39, sentence-transformers 5.2, and
spaCy 3.8.14. Every target's model weights (YOLO26n, NLTK + MiniLM,
Whisper tiny, face-alignment) are pre-baked. See [Configuration](configuration.md)
for the complete override table.

## Related pages

- [Configuration](configuration.md) — package versions, per-target overrides, and the constraint web
- [Pick a target](../use-cases/pick-a-target.md) — choose the right image for your work
- [Architecture](../concepts/architecture.md) — how the inheritance tree and build stages fit together
- [CLI reference](cli.md) — building, running, and verifying target images
