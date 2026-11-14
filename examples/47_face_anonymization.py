#!/usr/bin/env python3
# # Anonymize Faces in a Photo
#
# Before sharing photos of people, faces often have to be hidden. This notebook
# finds faces with OpenCV's bundled Haar-cascade detector (fast, no download) and
# with MTCNN (a small neural detector), then hides each face three ways:
# Gaussian blur, pixelation, and a solid box. The sample is the astronaut photo
# that ships with scikit-image, so it runs offline.
#
# To anonymize your own photo, put it in `/home/jupyter/data/` and set
# `PHOTO_FILE`. Check the result by eye: no detector finds every face.
#
# - [OpenCV cascade classifiers](https://docs.opencv.org/4.x/db/d28/tutorial_cascade_classifier.html)
# - [MTCNN](https://github.com/ipazc/mtcnn)

import json
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage import data, io

# Outputs go to ./output next to the notebook (the kernel's working directory).
OUTPUT_DIR = os.path.abspath('output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

PHOTO_FILE = None  # e.g. '/home/jupyter/data/team.jpg'
MARGIN = 0.15  # grow each box by 15% so hair and chin are covered too
PIXEL_BLOCKS = 10  # pixelation: the face becomes a 10 x 10 grid of blocks

photo = io.imread(PHOTO_FILE)[..., :3] if PHOTO_FILE else data.astronaut()
print(f"Photo {photo.shape[1]}x{photo.shape[0]}")

# ## Detect faces with a Haar cascade
#
# The cascade scans the image at several scales with simple light-dark patterns.
# It is fast and needs no network, but it only finds roughly frontal faces.

cascade = cv2.CascadeClassifier(os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml'))
grey = cv2.cvtColor(photo, cv2.COLOR_RGB2GRAY)
haar_boxes = [tuple(int(v) for v in box) for box in cascade.detectMultiScale(grey, scaleFactor=1.1, minNeighbors=5,
                                                                              minSize=(40, 40))]
print(f"Haar cascade: {len(haar_boxes)} face(s) {haar_boxes}")

# ## Detect faces with MTCNN
#
# MTCNN is a three-stage neural network that also handles tilted faces and
# reports a confidence per face. Its weights ship inside the package.

try:
    from mtcnn import MTCNN

    mtcnn_faces = [face for face in MTCNN().detect_faces(photo) if face['confidence'] > 0.9]
    mtcnn_boxes = [tuple(int(v) for v in face['box']) for face in mtcnn_faces]
    print(f"MTCNN: {len(mtcnn_boxes)} face(s), confidence {[round(float(f['confidence']), 3) for f in mtcnn_faces]}")
except ImportError:
    mtcnn_boxes = []
    print("MTCNN is not installed in this environment; using the Haar cascade only.")

boxes = mtcnn_boxes or haar_boxes
if not boxes:
    raise SystemExit("no faces found; try a lower minNeighbors or a larger photo")


# ## Three ways to hide a face


def grown(box: tuple[int, int, int, int], shape: tuple[int, ...]) -> tuple[int, int, int, int]:
    """Box (x, y, w, h) enlarged by MARGIN on every side and clipped to the image."""
    x, y, w, h = box
    dx, dy = int(w * MARGIN), int(h * MARGIN)
    x0, y0 = max(x - dx, 0), max(y - dy, 0)
    x1, y1 = min(x + w + dx, shape[1]), min(y + h + dy, shape[0])
    return x0, y0, x1, y1


def blur(region: np.ndarray) -> np.ndarray:
    """Heavy Gaussian blur: the kernel is half the face size, so small faces are hidden too.

    A light blur still looks like a face to a detector (and sometimes to a person);
    the check below confirms the face is gone.
    """
    k = max(min(region.shape[:2]) // 2, 3) | 1
    return cv2.GaussianBlur(region, (k, k), sigmaX=k / 2)


def pixelate(region: np.ndarray) -> np.ndarray:
    """Shrink to PIXEL_BLOCKS x PIXEL_BLOCKS, then scale back up without smoothing."""
    small = cv2.resize(region, (PIXEL_BLOCKS, PIXEL_BLOCKS), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(small, (region.shape[1], region.shape[0]), interpolation=cv2.INTER_NEAREST)


def solid(region: np.ndarray) -> np.ndarray:
    """A plain dark box: the only method that leaves nothing to reverse."""
    return np.full_like(region, 30)


METHODS = {'blur': blur, 'pixelate': pixelate, 'solid box': solid}
results = {}
for name, hide in METHODS.items():
    anonymized = photo.copy()
    for box in boxes:
        x0, y0, x1, y1 = grown(box, photo.shape)
        anonymized[y0:y1, x0:x1] = hide(anonymized[y0:y1, x0:x1])
    results[name] = anonymized
    io.imsave(os.path.join(OUTPUT_DIR, f"face_{name.replace(' ', '_')}.png"), anonymized, check_contrast=False)

# ## Check that the faces are really gone
#
# Running the detector again on each result is a cheap sanity check: a hidden
# face should no longer be found.

for name, image in results.items():
    found = cascade.detectMultiScale(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), scaleFactor=1.1, minNeighbors=5,
                                     minSize=(40, 40))
    print(f"  {name:10} faces still detected: {len(found)}")

fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
marked = photo.copy()
for box in boxes:
    x0, y0, x1, y1 = grown(box, photo.shape)
    cv2.rectangle(marked, (x0, y0), (x1, y1), (255, 0, 0), 3)
for ax, (title, image) in zip(axes, [('detected', marked), *results.items()]):
    ax.imshow(image)
    ax.set_title(title)
    ax.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'face_anonymization.png'), dpi=100)
plt.show()

with open(os.path.join(OUTPUT_DIR, 'face_anonymization.json'), 'w') as f:
    json.dump({'haar_boxes': haar_boxes, 'mtcnn_boxes': mtcnn_boxes}, f)
print("Saved: face_anonymization.png, face_blur.png, face_pixelate.png, face_solid_box.png")
