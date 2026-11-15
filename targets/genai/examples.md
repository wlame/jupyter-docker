<!-- GENERATED FILE — do not edit by hand. Source: targets/matrix.toml and the example titles. -->
# Examples in `ghcr.io/wlame/jupyter-docker:genai`

Generative AI: retrieval, fine-tuning, diffusion, and LLM API clients.

Open a notebook and run it top to bottom (**Run > Run All Cells**). Files an
example writes go to `output/` in this folder. Your own data belongs in
`/home/jupyter/data`: mount a host folder there with
`-v "$(pwd)/data:/home/jupyter/data"`.

| Notebook | What it covers | Image it comes from |
|---|---|---|
| [37_retrieval_and_generation](37_retrieval_and_generation.ipynb) | Retrieval and Generation: BM25, Embeddings, FAISS, a Small LM, and TRL | `genai` |
| [40_ask_your_documents](40_ask_your_documents.ipynb) | Ask Questions About Your Own Documents | `genai` |
| [14_nlp_text_analysis](14_nlp_text_analysis.ipynb) | NLP Text Analysis | `nlp` |
| [30_nlp_toolkit](30_nlp_toolkit.ipynb) | NLP Toolkit: rapidfuzz, lingua, Hugging Face datasets, PEFT, KeyBERT, SentencePiece | `nlp` |
| [46_review_analysis](46_review_analysis.ipynb) | What Are Customers Saying? Sentiment, Entities, and Topics in Reviews | `nlp` |
| [50_reliable_api_client](50_reliable_api_client.ipynb) | Calling an API Reliably: Retries, Validation, and Logging | `base` |
| [51_parse_web_and_config](51_parse_web_and_config.ipynb) | Parsing the Formats Data Arrives In: HTML, XML, YAML, .env, and Dates | `base` |

Each notebook has a matching `.py` file with the same code, which you can run
with `python <name>.py` or open as a notebook through Jupytext.
