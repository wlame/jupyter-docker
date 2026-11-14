#!/usr/bin/env python3
# # What Are Customers Saying? Sentiment, Entities, and Topics in Reviews
#
# A typical first pass over customer feedback: score each review's sentiment
# with VADER (a rule-based model made for short informal text), pull out the
# products, places, and companies mentioned with spaCy, and group the reviews
# into topics by embedding them with a sentence-transformer and clustering. All
# models are baked into the image, so it runs offline.
#
# To analyse your own feedback, put a CSV with a `text` column in
# `/home/jupyter/data/` and set `DATA_FILE`.
#
# - [NLTK VADER](https://www.nltk.org/howto/sentiment.html)
# - [spaCy](https://spacy.io/usage/linguistic-features#named-entities)
# - [sentence-transformers](https://sbert.net/)

import collections
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import spacy
from nltk.sentiment import SentimentIntensityAnalyzer
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

# Outputs go to ./output next to the notebook (the kernel's working directory).
OUTPUT_DIR = os.path.abspath('output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATA_FILE = None  # e.g. '/home/jupyter/data/feedback.csv' with a text column
TOPICS = 4
REVIEWS = [
    "The Acme blender is fantastic, crushes ice in seconds and cleaning it is easy.",
    "Delivery to Berlin took two weeks and the box arrived crushed. Very disappointed.",
    "Customer support at Acme answered within an hour and replaced my broken lid for free!",
    "Battery life on the X200 headphones is great, easily 30 hours on a single charge.",
    "The X200 headphones hurt my ears after an hour, and the sound is muddy.",
    "Shipping was fast, it reached Madrid in two days.",
    "I waited 40 minutes on the phone with support and nobody solved my problem.",
    "Love the new blender, smoothies every morning now. Worth every euro.",
    "Noise cancelling on the X200 is decent but not as good as the Sony ones.",
    "The courier left the parcel in the rain in Lyon, everything inside was wet.",
    "Support was friendly but the refund still has not arrived after a month.",
    "Blender motor died after three weeks. Not buying from Acme again.",
    "Package tracking never updated, but it showed up on time in Vienna.",
    "Pairing the headphones with my laptop was painless and the app is clear.",
    "The chat assistant understood my issue right away and sent a new charger.",
]

if DATA_FILE:
    import pandas as pd

    REVIEWS = pd.read_csv(DATA_FILE)['text'].dropna().astype(str).tolist()
print(f"{len(REVIEWS)} reviews")

# ## Sentiment with VADER
#
# The compound score runs from -1 (very negative) to +1 (very positive); the
# usual cut-offs are ±0.05.

vader = SentimentIntensityAnalyzer()
compound = np.array([vader.polarity_scores(text)['compound'] for text in REVIEWS])
label = np.where(compound >= 0.05, 'positive', np.where(compound <= -0.05, 'negative', 'neutral'))
for text, score in sorted(zip(REVIEWS, compound), key=lambda pair: pair[1])[:3]:
    print(f"  {score:+.2f}  {text}")
print("  ...")
print(f"Share: {dict(collections.Counter(label.tolist()))}")

# ## Entities with spaCy
#
# Named-entity recognition finds organisations (ORG), places (GPE), products,
# durations, and amounts without any training on this data.

nlp = spacy.load('en_core_web_sm')
entities = collections.Counter()
for doc in nlp.pipe(REVIEWS):
    entities.update((ent.text, ent.label_) for ent in doc.ents)
print("Most mentioned entities:")
for (text, kind), count in entities.most_common(8):
    print(f"  {count}x {text} ({kind})")

# ## Topics with sentence embeddings
#
# MiniLM maps each review to a 384-dimensional vector where similar meanings are
# close together. K-means groups the vectors, and the most distinctive words of
# each group (TF-IDF) name the topic.

embedder = SentenceTransformer('all-MiniLM-L6-v2')  # baked into the image
vectors = embedder.encode(REVIEWS, normalize_embeddings=True)
clusters = KMeans(n_clusters=TOPICS, n_init=10, random_state=0).fit_predict(vectors)

tfidf = TfidfVectorizer(stop_words='english')
weights = tfidf.fit_transform(REVIEWS)
vocabulary = np.array(tfidf.get_feature_names_out())
topics = {}
for topic in range(TOPICS):
    members = clusters == topic
    top_words = vocabulary[np.asarray(weights[members].mean(axis=0)).ravel().argsort()[::-1][:4]]
    topics[topic] = {
        'keywords': top_words.tolist(),
        'reviews': int(members.sum()),
        'mean_sentiment': float(compound[members].mean()),
    }
    print(f"Topic {topic}: {', '.join(top_words)} — {members.sum()} reviews, mean sentiment {compound[members].mean():+.2f}")

fig, ax = plt.subplots(figsize=(8, 4))
names = [' / '.join(topics[t]['keywords'][:2]) for t in range(TOPICS)]
ax.barh(names, [topics[t]['mean_sentiment'] for t in range(TOPICS)],
        color=['#55a868' if topics[t]['mean_sentiment'] > 0 else '#c44e52' for t in range(TOPICS)])
ax.axvline(0, color='black', lw=0.8)
ax.set(xlabel='mean VADER compound score', title='Sentiment by topic')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'reviews_by_topic.png'), dpi=120)
plt.show()

with open(os.path.join(OUTPUT_DIR, 'reviews_summary.json'), 'w') as f:
    json.dump({'sentiment': dict(collections.Counter(label.tolist())), 'topics': topics,
               'entities': [f'{t} ({k})' for (t, k), _ in entities.most_common(10)]}, f, indent=2)
print("Saved: reviews_by_topic.png, reviews_summary.json")
