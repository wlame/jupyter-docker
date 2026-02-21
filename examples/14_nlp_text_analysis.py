#!/usr/bin/env python3
"""
NLP Text Analysis
=================
Demonstrates natural language processing with spaCy, NLTK, and sentence-transformers.

spaCy: https://spacy.io/
NLTK: https://www.nltk.org/
Sentence Transformers: https://www.sbert.net/

Note: spaCy model required — run once inside the container:
    python -m spacy download en_core_web_sm
Sentence transformer models download automatically on first use (~80 MB).
"""

import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nltk

# Download NLTK data (silent if already present)
for resource in ('punkt', 'punkt_tab', 'stopwords', 'vader_lexicon', 'wordnet', 'averaged_perceptron_tagger', 'averaged_perceptron_tagger_eng'):
    nltk.download(resource, quiet=True)

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.sentiment.vader import SentimentIntensityAnalyzer

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# Sample Texts
# =============================================================================
TEXTS = [
    "Apple Inc. was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in Cupertino, California in 1976.",
    "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France.",
    "NASA's Mars rover Perseverance successfully landed in Jezero Crater on February 18, 2021.",
    "Albert Einstein was a German-born theoretical physicist who developed the theory of relativity.",
    "The Amazon River flows through nine nations in South America and discharges into the Atlantic Ocean.",
]

REVIEWS = [
    "This product is absolutely fantastic! I love everything about it. Highly recommended!",
    "Terrible experience. The item broke after one day. Complete waste of money.",
    "It's okay, nothing special. Does what it's supposed to do, I guess.",
    "Outstanding quality and fast shipping. Will definitely buy again!",
    "Not what I expected. The description was misleading and customer service was unhelpful.",
    "Pretty good overall. A few minor issues but nothing dealbreaking.",
    "Worst purchase ever. Avoid at all costs. Save your money.",
    "Exceeded all expectations. Remarkable craftsmanship and attention to detail.",
]

QUERIES = [
    "Who founded Apple?",
    "Where is the Eiffel Tower located?",
    "What did Einstein develop?",
]

# =============================================================================
# spaCy: Named Entity Recognition & Dependency Parsing
# =============================================================================
print("=" * 60)
print("spaCy: NER and Dependency Parsing")
print("=" * 60)

try:
    import spacy

    nlp = spacy.load('en_core_web_sm')

    print("\nNamed Entity Recognition:")
    all_entities = []
    for text in TEXTS:
        doc = nlp(text)
        entities = [(ent.text, ent.label_, spacy.explain(ent.label_)) for ent in doc.ents]
        all_entities.extend(entities)
        print(f"\n  Text: {text[:70]}...")
        for ent_text, label, explanation in entities:
            print(f"    [{label:8s}] {ent_text!r:35s} — {explanation}")

    # Entity type distribution
    entity_types = [e[1] for e in all_entities]
    type_counts = {}
    for t in entity_types:
        type_counts[t] = type_counts.get(t, 0) + 1

    print("\nEntity type distribution:")
    for etype, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"  {etype:12s}: {count}")

    # Dependency parsing on one sentence
    print("\nDependency Parsing (first sentence):")
    doc = nlp(TEXTS[0])
    for token in doc:
        if token.dep_ != 'punct':
            print(
                f"  {token.text:20s} dep={token.dep_:12s} "
                f"pos={token.pos_:8s} head={token.head.text!r}"
            )

    # Token analysis: lemma, POS, shape
    print("\nLinguistic Features Sample (first sentence tokens):")
    doc = nlp(TEXTS[0])
    header = f"  {'Token':20s} {'Lemma':20s} {'POS':8s} {'Is Stop':8s}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for token in doc:
        if not token.is_space and not token.is_punct:
            print(
                f"  {token.text:20s} {token.lemma_:20s} "
                f"{token.pos_:8s} {str(token.is_stop):8s}"
            )

except ImportError:
    print("spaCy not installed — skipping NER/dependency examples.")

# =============================================================================
# NLTK: Tokenization, Lemmatization, and Stopword Removal
# =============================================================================
print("\n" + "=" * 60)
print("NLTK: Tokenization and Text Preprocessing")
print("=" * 60)

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

sample_text = "The quick brown foxes were jumping over the lazily sleeping dogs in the park."

print(f"\nOriginal: {sample_text}")

sentences = sent_tokenize(sample_text)
print(f"Sentences: {sentences}")

words = word_tokenize(sample_text)
print(f"\nAll tokens ({len(words)}): {words}")

filtered = [w for w in words if w.isalpha() and w.lower() not in stop_words]
print(f"After stopword removal ({len(filtered)}): {filtered}")

lemmatized = [lemmatizer.lemmatize(w.lower()) for w in filtered]
stemmed = [stemmer.stem(w.lower()) for w in filtered]
print(f"Lemmatized: {lemmatized}")
print(f"Stemmed:    {stemmed}")

# =============================================================================
# NLTK: VADER Sentiment Analysis
# =============================================================================
print("\n" + "=" * 60)
print("NLTK: VADER Sentiment Analysis")
print("=" * 60)

sia = SentimentIntensityAnalyzer()

sentiments = []
print(f"\n{'Review':<55} {'Compound':>9} {'Label':>10}")
print("-" * 77)
for review in REVIEWS:
    scores = sia.polarity_scores(review)
    compound = scores['compound']
    label = 'positive' if compound >= 0.05 else ('negative' if compound <= -0.05 else 'neutral')
    sentiments.append({'text': review, 'compound': compound, 'label': label, **scores})
    short = review[:53] + '..' if len(review) > 55 else review
    print(f"{short:<55} {compound:>9.3f} {label:>10}")

# Plot sentiment distribution
labels = ['positive', 'neutral', 'negative']
counts = [sum(1 for s in sentiments if s['label'] == l) for l in labels]
colors = ['#2ecc71', '#95a5a6', '#e74c3c']

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].bar(labels, counts, color=colors, edgecolor='black', alpha=0.85)
axes[0].set_title("Sentiment Distribution (VADER)")
axes[0].set_ylabel("Count")
axes[0].set_xlabel("Sentiment")

compound_scores = [s['compound'] for s in sentiments]
bar_colors = [colors[0] if c >= 0.05 else (colors[2] if c <= -0.05 else colors[1]) for c in compound_scores]
axes[1].barh(
    range(len(compound_scores)),
    compound_scores,
    color=bar_colors,
    edgecolor='black',
    alpha=0.85,
)
axes[1].axvline(x=0, color='black', linewidth=0.8, linestyle='--')
axes[1].axvline(x=0.05, color='green', linewidth=0.8, linestyle=':')
axes[1].axvline(x=-0.05, color='red', linewidth=0.8, linestyle=':')
axes[1].set_yticks(range(len(REVIEWS)))
axes[1].set_yticklabels([f"Review {i+1}" for i in range(len(REVIEWS))], fontsize=9)
axes[1].set_xlabel("Compound Score")
axes[1].set_title("Per-Review Compound Scores")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'nlp_sentiment.png'), dpi=150)
print("\nSaved: nlp_sentiment.png")
plt.close()

# =============================================================================
# NLTK: WordNet — Synonyms, Antonyms, Definitions
# =============================================================================
print("\n" + "=" * 60)
print("NLTK: WordNet Lexical Database")
print("=" * 60)

words_to_explore = ['happy', 'fast', 'beautiful']

for word in words_to_explore:
    synsets = wordnet.synsets(word)
    print(f"\n'{word}' — {len(synsets)} synset(s):")

    synonyms = set()
    antonyms = set()
    for syn in synsets[:3]:
        print(f"  [{syn.name()}] {syn.definition()}")
        for lemma in syn.lemmas():
            synonyms.add(lemma.name().replace('_', ' '))
            for ant in lemma.antonyms():
                antonyms.add(ant.name().replace('_', ' '))

    print(f"  Synonyms: {', '.join(sorted(synonyms)[:8])}")
    if antonyms:
        print(f"  Antonyms: {', '.join(sorted(antonyms))}")

# =============================================================================
# Sentence Transformers: Semantic Similarity and Search
# =============================================================================
print("\n" + "=" * 60)
print("Sentence Transformers: Semantic Similarity & Search")
print("=" * 60)

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity

    print("\nLoading sentence transformer model (downloads ~80 MB on first use)...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Encode corpus
    corpus = TEXTS
    corpus_embeddings = model.encode(corpus)
    print(f"Encoded {len(corpus)} sentences → shape {corpus_embeddings.shape}")

    # Semantic search: find most relevant sentence for each query
    print("\nSemantic Search Results:")
    query_embeddings = model.encode(QUERIES)
    for query, q_emb in zip(QUERIES, query_embeddings):
        scores = cosine_similarity([q_emb], corpus_embeddings)[0]
        best_idx = int(np.argmax(scores))
        print(f"\n  Query: {query!r}")
        print(f"  Best match (score={scores[best_idx]:.3f}):")
        print(f"    {corpus[best_idx]}")

    # Pairwise similarity heatmap across all sentences
    sim_matrix = cosine_similarity(corpus_embeddings)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(sim_matrix, cmap='Blues', vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label='Cosine Similarity')

    short_labels = [t[:35] + '...' for t in corpus]
    ax.set_xticks(range(len(corpus)))
    ax.set_yticks(range(len(corpus)))
    ax.set_xticklabels(short_labels, rotation=30, ha='right', fontsize=8)
    ax.set_yticklabels(short_labels, fontsize=8)
    ax.set_title("Sentence Similarity Heatmap (all-MiniLM-L6-v2)")

    for i in range(len(corpus)):
        for j in range(len(corpus)):
            ax.text(j, i, f"{sim_matrix[i, j]:.2f}", ha='center', va='center', fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'nlp_similarity_heatmap.png'), dpi=150)
    print("\nSaved: nlp_similarity_heatmap.png")
    plt.close()

    # Semantic clustering of reviews
    print("\nSemantic Clustering of Reviews:")
    review_embeddings = model.encode(REVIEWS)
    review_sim = cosine_similarity(review_embeddings)

    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(review_embeddings)

    cluster_names = {0: 'Cluster A', 1: 'Cluster B', 2: 'Cluster C'}
    for cluster_id in sorted(set(cluster_labels)):
        print(f"\n  {cluster_names[cluster_id]}:")
        for idx, label in enumerate(cluster_labels):
            if label == cluster_id:
                compound = sentiments[idx]['compound']
                print(f"    [{compound:+.3f}] {REVIEWS[idx][:65]}")

except ImportError:
    print("sentence-transformers not installed — skipping semantic search examples.")

# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 60)
print("NLP text analysis complete!")
print("Outputs: nlp_sentiment.png, nlp_similarity_heatmap.png")
print("=" * 60)
