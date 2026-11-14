#!/usr/bin/env python3
# # Ask Questions About Your Own Documents
#
# Retrieval-augmented generation on files you choose: put Markdown or text files
# in `/home/jupyter/data/docs/` and ask questions about them. With no files there,
# the notebook uses the example notebooks shipped in this image as its library,
# so it always has something to search.
#
# The steps: split documents into overlapping chunks, find the chunks that match
# a question with both keyword search (BM25) and meaning (MiniLM embeddings),
# then let a small local model (SmolLM2-135M, baked into the image) answer from
# those chunks only, citing its sources. An optional last section sends the same
# prompt to a larger model on any OpenAI-compatible server (Ollama, vLLM, LM
# Studio) when one is running.
#
# - [bm25s](https://github.com/xhluca/bm25s)
# - [sentence-transformers](https://sbert.net/)
# - [SmolLM2](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct)
# - [OpenAI Python client](https://github.com/openai/openai-python)

import glob
import json
import os
import re

import bm25s
import numpy as np
import torch
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

# Outputs go to ./output next to the notebook (the kernel's working directory).
OUTPUT_DIR = os.path.abspath('output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

DOCS_DIR = '/home/jupyter/data/docs'
QUESTIONS = [
    'How do I count coins in a photo?',
    'Which notebook shows Bayesian modeling, and which samplers does it compare?',
    'How can I save a scikit-learn model without pickle?',
]
CHUNK_WORDS, OVERLAP_WORDS = 120, 30
TOP_K = 3
# Same repository and revision as scripts/bake_models.sh, so the image loads the baked copy.
LM_REPO = 'HuggingFaceTB/SmolLM2-135M-Instruct'
LM_REVISION = '12fd25f77366fa6b3b4b768ec3050bf629380bac'
# Optional larger model: any OpenAI-compatible server. Ollama's default is shown.
LLM_BASE_URL = os.environ.get('LLM_BASE_URL', 'http://localhost:11434/v1')
LLM_MODEL = os.environ.get('LLM_MODEL', 'llama3.2')
torch.manual_seed(0)

# ## Load the documents
#
# Your files win; otherwise the markdown text of the example notebooks next to
# this one becomes the library.


def notebook_prose(path: str) -> str:
    """The markdown comments (`# text`) of a jupytext .py example, as plain text."""
    lines = [line[2:] for line in open(path, encoding='utf-8') if line.startswith('# ') or line == '#\n']
    return '\n'.join(lines)


paths = sorted(glob.glob(os.path.join(DOCS_DIR, '**', '*.md'), recursive=True)
               + glob.glob(os.path.join(DOCS_DIR, '**', '*.txt'), recursive=True))
if paths:
    documents = {os.path.relpath(p, DOCS_DIR): open(p, encoding='utf-8').read() for p in paths}
    print(f"Using {len(documents)} file(s) from {DOCS_DIR}")
else:
    examples = sorted(glob.glob(os.path.join(os.getcwd(), '[0-9][0-9]_*.py')))
    documents = {os.path.basename(p): notebook_prose(p) for p in examples}
    print(f"No files in {DOCS_DIR}; using the prose of {len(documents)} example notebooks instead")

# ## Split into overlapping chunks
#
# Chunks of about 120 words keep each passage focused; a 30-word overlap stops an
# answer from being cut in half at a chunk boundary.

chunks, sources = [], []
for name, text in documents.items():
    words = re.sub(r'\s+', ' ', text).split()
    for start in range(0, max(len(words) - OVERLAP_WORDS, 1), CHUNK_WORDS - OVERLAP_WORDS):
        chunks.append(' '.join(words[start:start + CHUNK_WORDS]))
        sources.append(name)
print(f"{len(chunks)} chunks")

# ## Hybrid retrieval
#
# BM25 finds exact terms ("skops", "Haar"); embeddings find the same meaning in
# other words ("save a model" vs "persist"). Reciprocal rank fusion combines both
# rankings without tuning weights.

bm25 = bm25s.BM25()
bm25.index(bm25s.tokenize(chunks, stopwords='en', show_progress=False), show_progress=False)
embedder = SentenceTransformer('all-MiniLM-L6-v2')  # baked into the image
chunk_vectors = embedder.encode(chunks, normalize_embeddings=True, batch_size=64)


def retrieve(question: str, k: int = TOP_K, depth: int = 20, smoothing: int = 60) -> list[int]:
    """Indices of the k best chunks by reciprocal rank fusion of BM25 and embeddings."""
    keyword_hits, _ = bm25.retrieve(bm25s.tokenize(question, stopwords='en', show_progress=False),
                                    k=min(depth, len(chunks)), show_progress=False)
    semantic_hits = np.argsort(chunk_vectors @ embedder.encode(question, normalize_embeddings=True))[::-1][:depth]
    scores: dict[int, float] = {}
    for ranking in (keyword_hits[0].tolist(), semantic_hits.tolist()):
        for rank, index in enumerate(ranking):
            scores[index] = scores.get(index, 0.0) + 1.0 / (smoothing + rank)
    return sorted(scores, key=scores.get, reverse=True)[:k]


# ## Answer with a small local model
#
# The prompt carries the retrieved chunks with numbered sources, and the model is
# told to answer from them only. A 135-million-parameter model is fast on a CPU
# but simple; the sources it was given are the reliable part of the output.

tokenizer = AutoTokenizer.from_pretrained(LM_REPO, revision=LM_REVISION)
model = AutoModelForCausalLM.from_pretrained(LM_REPO, revision=LM_REVISION, dtype=torch.float32)


def build_messages(question: str, hits: list[int]) -> list[dict]:
    """Chat messages: instructions, numbered context chunks, and the question."""
    context = '\n\n'.join(f'[{n}] ({sources[i]}) {chunks[i]}' for n, i in enumerate(hits, start=1))
    return [
        {'role': 'system', 'content': 'Answer in two sentences using only the context. Cite sources like [1].'},
        {'role': 'user', 'content': f'Context:\n{context}\n\nQuestion: {question}'},
    ]


answers = []
for question in QUESTIONS:
    hits = retrieve(question)
    prompt = tokenizer.apply_chat_template(build_messages(question, hits), add_generation_prompt=True,
                                           return_tensors='pt', return_dict=True)
    with torch.no_grad():
        output = model.generate(**prompt, max_new_tokens=80, do_sample=False)
    answer = tokenizer.decode(output[0, prompt['input_ids'].shape[1]:], skip_special_tokens=True).strip()
    cited = [sources[i] for i in hits]
    answers.append({'question': question, 'answer': answer, 'sources': cited})
    print(f"\nQ: {question}\nSources: {', '.join(dict.fromkeys(cited))}\nA: {answer}")

# ## Optional: the same prompt to a larger model
#
# Start a server such as Ollama (`ollama serve`, then `ollama pull llama3.2`) on
# the host and run the container with `--add-host=host.docker.internal:host-gateway`
# and `-e LLM_BASE_URL=http://host.docker.internal:11434/v1`. Hosted APIs work the
# same way with their base URL and an API key in `OPENAI_API_KEY`. Without a
# server this section says so and moves on.

client = OpenAI(base_url=LLM_BASE_URL, api_key=os.environ.get('OPENAI_API_KEY', 'local'), timeout=5, max_retries=0)
try:
    reply = client.chat.completions.create(model=LLM_MODEL, messages=build_messages(QUESTIONS[0], retrieve(QUESTIONS[0])))
    print(f"{LLM_MODEL} at {LLM_BASE_URL}:\n{reply.choices[0].message.content}")
except Exception as error:  # no server is the normal case in a fresh container
    print(f"No model server at {LLM_BASE_URL} ({type(error).__name__}); skipped. See the notes above to connect one.")

with open(os.path.join(OUTPUT_DIR, 'ask_documents.json'), 'w') as f:
    json.dump(answers, f, indent=2)
print("\nSaved: ask_documents.json")
