#!/usr/bin/env python3
# # Face Recognition by Similarity
#
# Face recognition systems do not memorize people. They map each face to an
# embedding (here 512 numbers from InsightFace's ArcFace model) so that photos of
# the same person land close together, then decide by a similarity threshold.
# This notebook embeds the Olivetti faces (40 people, 10 photos each), measures
# how well similarity separates "same person" from "different people", picks a
# threshold, and finds each face's closest match.
#
# The model and the dataset are baked into the image, so it runs offline. Only use
# face recognition on people who have agreed to it.
#
# - [InsightFace](https://github.com/deepinsight/insightface)
# - [Olivetti faces (AT&T)](https://cam-orl.co.uk/facedatabase.html)

import glob
import json
import os

import cv2
import insightface
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

# Outputs go to ./output next to the notebook (the kernel's working directory).
OUTPUT_DIR = os.path.abspath('output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Both baked by scripts/bake_models.sh face.
DATASET = os.path.expanduser('~/.cache/olivetti/olivettifaces.mat')
MODEL_DIR = os.path.expanduser('~/.insightface/models/buffalo_sc')
PEOPLE, PHOTOS = 40, 10

# ## Load the faces
#
# The file stores 400 greyscale 64×64 images as columns, ordered person by person.

faces = loadmat(DATASET)['faces'].T.reshape(-1, 64, 64).transpose(0, 2, 1).astype(np.float32)
faces = (255 * (faces - faces.min()) / (faces.max() - faces.min())).astype(np.uint8)
person = np.repeat(np.arange(PEOPLE), PHOTOS)
print(f"{len(faces)} photos of {PEOPLE} people, {faces.shape[1]}x{faces.shape[2]} pixels")

fig, axes = plt.subplots(3, 10, figsize=(12, 4))
for row, who in enumerate([0, 1, 2]):
    for col in range(10):
        axes[row, col].imshow(faces[who * PHOTOS + col], cmap='gray')
        axes[row, col].axis('off')
fig.suptitle('Three of the 40 people, ten photos each')
plt.tight_layout()
plt.show()

# ## Embed every face
#
# The recognition model expects a 112×112 colour crop of a centred face. These
# photos are already centred, so they are only resized; for your own photos, run
# a detector first (example 47) and crop each face.

model_path = next(p for p in glob.glob(os.path.join(MODEL_DIR, '*.onnx')) if 'w600k' in p)
recognizer = insightface.model_zoo.get_model(model_path, providers=['CPUExecutionProvider'])
recognizer.prepare(ctx_id=-1)
crops = [cv2.cvtColor(cv2.resize(img, recognizer.input_size, interpolation=cv2.INTER_CUBIC), cv2.COLOR_GRAY2BGR)
         for img in faces]
embeddings = np.vstack([recognizer.get_feat(crop) for crop in crops])
embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
print(f"Embeddings: {embeddings.shape} from {os.path.basename(model_path)}")

# ## Same person or not?
#
# Cosine similarity of every pair: pairs of the same person should score higher
# than pairs of different people. The overlap of the two histograms is where
# mistakes happen.

similarity = embeddings @ embeddings.T
upper = np.triu_indices(len(faces), k=1)
pair_scores = similarity[upper]
same = person[upper[0]] == person[upper[1]]
print(f"{same.sum()} same-person pairs, {(~same).sum()} different-person pairs")
print(f"Mean similarity: same person {pair_scores[same].mean():.3f}, different people {pair_scores[~same].mean():.3f}")


def roc_auc(positive: np.ndarray, negative: np.ndarray) -> float:
    """Probability that a random same-person pair outscores a random different-person pair."""
    ranks = np.argsort(np.argsort(np.concatenate([positive, negative]))) + 1
    return float((ranks[:len(positive)].sum() - len(positive) * (len(positive) + 1) / 2) / (len(positive) * len(negative)))


auc = roc_auc(pair_scores[same], pair_scores[~same])
thresholds = np.linspace(pair_scores.min(), pair_scores.max(), 400)
# Balanced accuracy: the threshold should not simply say "different" to every pair.
balanced = [((pair_scores[same] >= t).mean() + (pair_scores[~same] < t).mean()) / 2 for t in thresholds]
best = thresholds[int(np.argmax(balanced))]
true_accept = (pair_scores[same] >= best).mean()
false_accept = (pair_scores[~same] >= best).mean()
print(f"ROC AUC {auc:.3f}; threshold {best:.3f} accepts {true_accept:.1%} of same-person pairs "
      f"and {false_accept:.2%} of different-person pairs")

fig, ax = plt.subplots(figsize=(8, 4))
bins = np.linspace(pair_scores.min(), pair_scores.max(), 60)
ax.hist(pair_scores[~same], bins=bins, density=True, alpha=0.6, label='different people')
ax.hist(pair_scores[same], bins=bins, density=True, alpha=0.6, label='same person')
ax.axvline(best, color='black', ls='--', label=f'threshold {best:.2f}')
ax.set(xlabel='cosine similarity', ylabel='density', title=f'Pair similarity (ROC AUC {auc:.3f})')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'face_similarity.png'), dpi=120)
plt.show()

# ## Identify: who is the closest match?
#
# Treat each photo as unknown and look up its most similar other photo. When that
# photo shows the same person, identification succeeded.

np.fill_diagonal(similarity, -1)
nearest = similarity.argmax(axis=1)
rank1 = float((person[nearest] == person).mean())
print(f"Closest-match identification: {rank1:.1%} of {len(faces)} photos")

misses = np.flatnonzero(person[nearest] != person)[:4]
if len(misses):
    fig, axes = plt.subplots(2, len(misses), figsize=(2.4 * len(misses), 5), squeeze=False)
    for col, index in enumerate(misses):
        for row, (img, label) in enumerate([(faces[index], f'person {person[index]}'),
                                            (faces[nearest[index]], f'matched {person[nearest[index]]}')]):
            axes[row, col].imshow(img, cmap='gray')
            axes[row, col].set_title(label, fontsize=9)
            axes[row, col].axis('off')
    fig.suptitle('Some mistakes: the photo (top) and its closest match (bottom)')
    plt.tight_layout()
    plt.show()

with open(os.path.join(OUTPUT_DIR, 'face_recognition.json'), 'w') as f:
    json.dump({'roc_auc': auc, 'threshold': float(best), 'true_accept': float(true_accept),
               'false_accept': float(false_accept), 'rank1_identification': rank1}, f, indent=2)
print("Saved: face_similarity.png, face_recognition.json")
