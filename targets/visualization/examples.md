<!-- GENERATED FILE — do not edit by hand. Source: targets/matrix.toml and the example titles. -->
# Examples in `ghcr.io/wlame/jupyter-docker:visualization`

Data visualization with Matplotlib, Seaborn, Plotly, and Bokeh.

Open a notebook and run it top to bottom (**Run > Run All Cells**). Files an
example writes go to `output/` in this folder. Your own data belongs in
`/home/jupyter/data`: mount a host folder there with
`-v "$(pwd)/data:/home/jupyter/data"`.

| Notebook | What it covers | Image it comes from |
|---|---|---|
| [03_matplotlib_seaborn_viz](03_matplotlib_seaborn_viz.ipynb) | Matplotlib and Seaborn Visualization | `visualization` |
| [04_plotly_interactive](04_plotly_interactive.ipynb) | Plotly Interactive Visualizations | `visualization` |
| [05_bokeh_holoviews](05_bokeh_holoviews.ipynb) | Bokeh and HoloViews Visualizations | `visualization` |
| [16_altair_panel_viz](16_altair_panel_viz.ipynb) | Altair and Panel: Declarative and Dashboard Visualization | `visualization` |
| [22_plotnine_tables_datashader](22_plotnine_tables_datashader.ipynb) | Grammar of Graphics, Tables, and Big-Data Rendering | `visualization` |
| [50_reliable_api_client](50_reliable_api_client.ipynb) | Calling an API Reliably: Retries, Validation, and Logging | `base` |
| [51_parse_web_and_config](51_parse_web_and_config.ipynb) | Parsing the Formats Data Arrives In: HTML, XML, YAML, .env, and Dates | `base` |

Each notebook has a matching `.py` file with the same code, which you can run
with `python <name>.py` or open as a notebook through Jupytext.
