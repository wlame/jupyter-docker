#!/usr/bin/env python3
"""
Retrieval and Generation: BM25, Embeddings, FAISS, a Small LM, and TRL
======================================================================
Builds a search index over the docstrings of Python's standard library, then
answers plain-English questions with three retrievers: BM25 keyword search
(bm25s), dense embeddings (MiniLM) in a FAISS index, and a hybrid of the two
(reciprocal rank fusion). A small instruction-tuned model (SmolLM2-135M) then
answers one question from the retrieved passages, and TRL fine-tunes a LoRA
adapter on it for a few steps.

Everything runs offline on the CPU with weights baked into the image. To use a
larger model, point the `openai` client at any OpenAI-compatible server (vLLM,
Ollama, LM Studio) or a hosted API; keys come from the environment
(OPENAI_API_KEY, ANTHROPIC_API_KEY), never from code.

bm25s:     https://github.com/xhluca/bm25s
FAISS:     https://faiss.ai/
sentence-transformers: https://sbert.net/
SmolLM2:   https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct
TRL:       https://huggingface.co/docs/trl/
"""

import importlib
import inspect
import json
import os

import bm25s
import faiss
import numpy as np
import torch
from datasets import Dataset
from openai import OpenAI
from peft import LoraConfig
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Same repository and revision as scripts/bake_models.sh, so the image loads the baked copy.
LM_REPO = 'HuggingFaceTB/SmolLM2-135M-Instruct'
LM_REVISION = '12fd25f77366fa6b3b4b768ec3050bf629380bac'
MODULES = [
    'bisect', 'collections', 'csv', 'datetime', 'difflib', 'functools', 'heapq', 'itertools', 'json', 'math',
    'os.path', 'pathlib', 'random', 're', 'shutil', 'statistics', 'string', 'textwrap', 'urllib.parse', 'zipfile',
]
# Each question names a task in everyday words; the expected answer is a function whose docstring
# rarely repeats those words, which is where keyword search and embeddings differ.
QUESTIONS = {
    'find the middle value of a list of numbers': 'statistics.median',
    'copy a folder with everything inside it': 'shutil.copytree',
    'merge several sorted inputs into one sorted output': 'heapq.merge',
    'break a long paragraph into lines of a fixed width': 'textwrap.wrap',
    'split a web address into scheme, host, and path': 'urllib.parse.urlsplit',
    'show the differences between two lists of lines': 'difflib.unified_diff',
}
TOP_K = 3
torch.manual_seed(0)
results: dict[str, object] = {}


def stdlib_corpus() -> tuple[list[str], list[str]]:
    """(ids, texts): the first paragraph of each public function's or class's docstring."""
    ids, texts = [], []
    for module_name in MODULES:
        module = importlib.import_module(module_name)
        for attr in sorted(dir(module)):
            obj = getattr(module, attr)
            if attr.startswith('_') or not (inspect.isroutine(obj) or inspect.isclass(obj)):
                continue
            paragraph = (inspect.getdoc(obj) or '').split('\n\n')[0].replace('\n', ' ').strip()
            if len(paragraph) >= 30:
                ids.append(f'{module_name}.{attr}')
                texts.append(paragraph)
    return ids, texts


# =============================================================================
# The corpus: standard-library docstrings
# =============================================================================
print("=" * 60)
print("Corpus: Python Standard Library Docstrings")
print("=" * 60)

doc_ids, doc_texts = stdlib_corpus()
print(f"{len(doc_texts)} passages from {len(MODULES)} modules, for example:")
print(f"  {doc_ids[0]}: {doc_texts[0][:90]}...")

# =============================================================================
# Three retrievers
# =============================================================================
print("\n" + "=" * 60)
print("BM25 vs Dense (MiniLM + FAISS) vs Hybrid")
print("=" * 60)

bm25 = bm25s.BM25()
bm25.index(bm25s.tokenize(doc_texts, stopwords='en', show_progress=False), show_progress=False)


def bm25_ranking(question: str, k: int) -> list[int]:
    """Indices of the k best passages by BM25 keyword score."""
    hits, _scores = bm25.retrieve(bm25s.tokenize(question, stopwords='en', show_progress=False), k=k, show_progress=False)
    return hits[0].tolist()


# Pre-baked in the nlp image (and inherited here); downloads (~80 MB) on first use elsewhere.
embedder = SentenceTransformer('all-MiniLM-L6-v2')
doc_vectors = embedder.encode(doc_texts, normalize_embeddings=True, batch_size=64).astype(np.float32)
index = faiss.IndexFlatIP(doc_vectors.shape[1])  # inner product of unit vectors = cosine similarity
index.add(doc_vectors)


def dense_ranking(question: str, k: int) -> list[int]:
    """Indices of the k passages whose embeddings are closest to the question's."""
    query = embedder.encode([question], normalize_embeddings=True).astype(np.float32)
    _scores, hits = index.search(query, k)
    return hits[0].tolist()


def hybrid_ranking(question: str, k: int, depth: int = 20, smoothing: int = 60) -> list[int]:
    """Reciprocal rank fusion: score 1/(smoothing + rank) in each list, summed."""
    scores: dict[int, float] = {}
    for ranking in (bm25_ranking(question, depth), dense_ranking(question, depth)):
        for rank, doc in enumerate(ranking):
            scores[doc] = scores.get(doc, 0.0) + 1.0 / (smoothing + rank)
    return sorted(scores, key=scores.get, reverse=True)[:k]


RETRIEVERS = {'bm25': bm25_ranking, 'dense': dense_ranking, 'hybrid': hybrid_ranking}
hits_at_k = dict.fromkeys(RETRIEVERS, 0)
for question, expected in QUESTIONS.items():
    print(f"\nQ: {question}   (expected {expected})")
    for name, ranking in RETRIEVERS.items():
        found = [doc_ids[i] for i in ranking(question, TOP_K)]
        hit = expected in found
        hits_at_k[name] += hit
        print(f"  {name:7} {'hit ' if hit else 'miss'}  {', '.join(found)}")

print(f"\nTop-{TOP_K} hits out of {len(QUESTIONS)}: " + ", ".join(f"{k} {v}" for k, v in hits_at_k.items()))
results['top_k_hits'] = hits_at_k
if max(hits_at_k['dense'], hits_at_k['hybrid']) < len(QUESTIONS) // 2:
    raise SystemExit("semantic retrieval found fewer than half of the expected passages")

# =============================================================================
# Generation: answer from retrieved passages with a small local model
# =============================================================================
print("\n" + "=" * 60)
print("Retrieval-Augmented Generation with SmolLM2-135M")
print("=" * 60)

tokenizer = AutoTokenizer.from_pretrained(LM_REPO, revision=LM_REVISION)
model = AutoModelForCausalLM.from_pretrained(LM_REPO, revision=LM_REVISION, dtype=torch.float32)
question = 'How do I copy a folder with everything inside it?'
context = '\n'.join(f'- {doc_ids[i]}: {doc_texts[i]}' for i in hybrid_ranking(question, TOP_K))
messages = [
    {'role': 'system', 'content': 'Answer in one sentence, naming the Python function to use.'},
    {'role': 'user', 'content': f'Documentation:\n{context}\n\nQuestion: {question}'},
]
prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors='pt', return_dict=True)
with torch.no_grad():
    output = model.generate(**prompt, max_new_tokens=48, do_sample=False)
answer = tokenizer.decode(output[0, prompt['input_ids'].shape[1]:], skip_special_tokens=True).strip()
print(f"Context passages:\n{context}\n")
print(f"Answer: {answer}")
with open(os.path.join(OUTPUT_DIR, 'genai_answer.txt'), 'w') as f:
    f.write(f"Q: {question}\n\nContext:\n{context}\n\nA: {answer}\n")
print("Saved: genai_answer.txt")

# =============================================================================
# TRL: a few steps of supervised fine-tuning with a LoRA adapter
# =============================================================================
print("\n" + "=" * 60)
print("TRL: LoRA Supervised Fine-Tuning (a few steps)")
print("=" * 60)

pairs = [(q, f'Use {expected}.') for q, expected in QUESTIONS.items()]
train = Dataset.from_list([
    {'messages': [{'role': 'user', 'content': q}, {'role': 'assistant', 'content': a}]} for q, a in pairs
])
trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=train,
    peft_config=LoraConfig(r=8, lora_alpha=16, target_modules=['q_proj', 'v_proj'], task_type='CAUSAL_LM'),
    args=SFTConfig(
        output_dir=os.path.join(OUTPUT_DIR, 'trl_sft'),
        max_steps=6,
        per_device_train_batch_size=2,
        learning_rate=2e-4,
        logging_steps=1,
        save_strategy='no',
        report_to='none',
        use_cpu=True,
        max_length=128,
    ),
)
trained = trainer.train()
losses = [entry['loss'] for entry in trainer.state.log_history if 'loss' in entry]
trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
total = sum(p.numel() for p in trainer.model.parameters())
print(f"Trained {trainable:,} of {total:,} parameters ({trainable / total:.2%}) for {trained.global_step} steps")
print("Loss per step: " + ", ".join(f"{loss:.3f}" for loss in losses))
results['sft_losses'] = losses

# =============================================================================
# The same chat API against a bigger model elsewhere
# =============================================================================
print("\n" + "=" * 60)
print("OpenAI-Compatible Clients")
print("=" * 60)

# Constructing a client makes no request. With a local server running, call
# client.chat.completions.create(model=..., messages=messages).
client = OpenAI(base_url='http://localhost:11434/v1', api_key=os.environ.get('OPENAI_API_KEY', 'local-server'))
print(f"Client ready for {client.base_url} (Ollama's default port); no request sent.")

with open(os.path.join(OUTPUT_DIR, 'genai_results.json'), 'w') as f:
    json.dump(results, f, indent=2)
print("\nSaved: genai_results.json")
print("Done.")
