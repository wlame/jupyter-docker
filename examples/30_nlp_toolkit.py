#!/usr/bin/env python3
"""
NLP Toolkit: rapidfuzz, lingua, Hugging Face datasets, PEFT, KeyBERT
====================================================================
Matches messy strings with rapidfuzz, detects languages with lingua (models
ship inside the package), builds a dataset pipeline with Hugging Face
`datasets`, attaches a LoRA adapter to a small transformer with PEFT, and
extracts keywords with KeyBERT on the MiniLM sentence-embedding model.

tiktoken (OpenAI tokenizers), evaluate (metrics), and BERTopic (topic models)
are installed too; they download encodings/metrics on first use or need a
larger corpus, so this example only names them:
`tiktoken.get_encoding('cl100k_base')`, `evaluate.load('accuracy')`,
`BERTopic().fit_transform(documents)`.

rapidfuzz: https://rapidfuzz.github.io/RapidFuzz/
lingua:    https://github.com/pemistahl/lingua-py
datasets:  https://huggingface.co/docs/datasets/
PEFT:      https://huggingface.co/docs/peft/
KeyBERT:   https://maartengr.github.io/KeyBERT/
"""

import json
import os

import torch
from datasets import Dataset
from keybert import KeyBERT
from lingua import Language, LanguageDetectorBuilder
from peft import LoraConfig, TaskType, get_peft_model
from rapidfuzz import fuzz, process
from sentence_transformers import SentenceTransformer
from transformers import BertConfig, BertForSequenceClassification

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

torch.manual_seed(0)
results: dict[str, object] = {}

# =============================================================================
# rapidfuzz — fuzzy string matching
# =============================================================================
print("=" * 60)
print("rapidfuzz: Fuzzy Matching")
print("=" * 60)

companies = ['Acme Corporation', 'Globex Inc.', 'Initech LLC', 'Umbrella Corp', 'Hooli', 'Vandelay Industries']
queries = ['acme corp', 'Globex Incorporated', 'initech', 'vandelay ind.']
matches = {}
for query in queries:
    best, score, _ = process.extractOne(query, companies, scorer=fuzz.WRatio)
    matches[query] = best
    print(f"{query!r:24} -> {best!r:22} (score {score:.0f})")
print(f"token_sort_ratio('Corp Umbrella', 'Umbrella Corp') = {fuzz.token_sort_ratio('Corp Umbrella', 'Umbrella Corp'):.0f}")
results['fuzzy_matches'] = matches

# =============================================================================
# lingua — language detection
# =============================================================================
print("\n" + "=" * 60)
print("lingua: Language Detection")
print("=" * 60)

detector = LanguageDetectorBuilder.from_languages(
    Language.ENGLISH, Language.GERMAN, Language.FRENCH, Language.SPANISH, Language.ITALIAN
).build()
samples = [
    'The quick brown fox jumps over the lazy dog.',
    'Der schnelle braune Fuchs springt über den faulen Hund.',
    'Le renard brun rapide saute par-dessus le chien paresseux.',
    'El rápido zorro marrón salta sobre el perro perezoso.',
]
languages = {}
for text in samples:
    language = detector.detect_language_of(text)
    confidence = detector.compute_language_confidence(text, language)
    languages[text[:24]] = language.name
    print(f"{language.name:8} ({confidence:.2f})  {text}")
results['languages'] = languages

# =============================================================================
# datasets — a processing pipeline
# =============================================================================
print("\n" + "=" * 60)
print("datasets: Map, Filter, Split")
print("=" * 60)

reviews = Dataset.from_dict({
    'text': [
        'Absolutely loved it, would buy again', 'Terrible quality, broke in a day',
        'Decent value for the price', 'Not what I expected at all', 'Fantastic support team',
        'The packaging was damaged', 'Works exactly as described', 'Waste of money',
    ],
    'label': [1, 0, 1, 0, 1, 0, 1, 0],
})
processed = (
    reviews.map(lambda batch: {'n_words': [len(t.split()) for t in batch['text']]}, batched=True)
    .filter(lambda row: row['n_words'] >= 4)
    .train_test_split(test_size=0.25, seed=0)
)
print(processed)
processed.save_to_disk(os.path.join(OUTPUT_DIR, 'reviews_dataset'))
print("Saved: reviews_dataset/ (Arrow files)")

# =============================================================================
# PEFT — LoRA adapter on a small transformer
# =============================================================================
print("\n" + "=" * 60)
print("PEFT: LoRA Adapter")
print("=" * 60)

config = BertConfig(vocab_size=2000, hidden_size=64, num_hidden_layers=2, num_attention_heads=2,
                    intermediate_size=128, num_labels=2)
base_model = BertForSequenceClassification(config)  # architecture only, random weights
lora = LoraConfig(task_type=TaskType.SEQ_CLS, r=4, lora_alpha=8, lora_dropout=0.0, target_modules=['query', 'value'])
peft_model = get_peft_model(base_model, lora)
trainable, total = peft_model.get_nb_trainable_parameters()
print(f"Trainable parameters: {trainable:,} of {total:,} ({100 * trainable / total:.2f}%)")
with torch.no_grad():
    logits = peft_model(input_ids=torch.randint(0, 2000, (2, 16))).logits
print(f"Forward pass logits: {tuple(logits.shape)}")
results['lora_trainable_fraction'] = round(trainable / total, 4)

# =============================================================================
# KeyBERT — keywords from sentence embeddings
# =============================================================================
print("\n" + "=" * 60)
print("KeyBERT: Keyword Extraction")
print("=" * 60)

# Pre-baked in the nlp image; downloads (~80 MB) on first use elsewhere.
embedder = SentenceTransformer('all-MiniLM-L6-v2')
document = (
    'Supervised learning trains models on labeled examples, while unsupervised learning finds '
    'structure in unlabeled data such as clusters or latent topics. Gradient boosting and neural '
    'networks are popular supervised methods.'
)
keywords = KeyBERT(model=embedder).extract_keywords(document, keyphrase_ngram_range=(1, 2), stop_words='english', top_n=5)
for phrase, score in keywords:
    print(f"{score:.3f}  {phrase}")
results['keywords'] = [phrase for phrase, _ in keywords]

with open(os.path.join(OUTPUT_DIR, 'nlp_toolkit.json'), 'w') as f:
    json.dump(results, f, indent=2)
print("\nSaved: nlp_toolkit.json")
print("Done.")
