#!/usr/bin/env python3
"""
Face Analysis
=============
Demonstrates face detection, recognition, and landmark extraction.

deepface: https://github.com/serengil/deepface
dlib: http://dlib.net/
face-alignment: https://github.com/1adrianb/face-alignment
mtcnn: https://github.com/ipazc/mtcnn

Uses synthetically generated images — no external files required.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from PIL import Image, ImageDraw

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# Generate Synthetic Test Images
# =============================================================================
print("=" * 60)
print("Generating Synthetic Test Images")
print("=" * 60)

# Create a simple face-like image using geometric shapes
img_size = 256
img = Image.new('RGB', (img_size, img_size), color=(200, 180, 160))
draw = ImageDraw.Draw(img)

# Face outline (oval)
face_bbox = (48, 38, 208, 228)
draw.ellipse(face_bbox, fill=(220, 195, 170), outline=(180, 150, 130), width=2)

# Eyes
draw.ellipse((88, 95, 118, 115), fill='white', outline=(100, 80, 60))
draw.ellipse((138, 95, 168, 115), fill='white', outline=(100, 80, 60))
draw.ellipse((98, 100, 112, 112), fill=(60, 40, 20))
draw.ellipse((148, 100, 162, 112), fill=(60, 40, 20))

# Nose
draw.polygon([(128, 120), (118, 155), (138, 155)], outline=(180, 150, 130))

# Mouth
draw.arc((100, 150, 156, 185), start=0, end=180, fill=(180, 100, 100), width=2)

# Eyebrows
draw.arc((83, 78, 123, 98), start=180, end=360, fill=(100, 70, 40), width=2)
draw.arc((133, 78, 173, 98), start=180, end=360, fill=(100, 70, 40), width=2)

input_path = os.path.join(OUTPUT_DIR, 'face_synthetic_input.png')
img.save(input_path)
img_array = np.array(img)

print(f"Image size   : {img_size}x{img_size}")
print(f"Image shape  : {img_array.shape}")
print(f"Saved        : {input_path}")

# =============================================================================
# dlib — HOG Face Detection
# =============================================================================
print("\n" + "=" * 60)
print("dlib — HOG Face Detection")
print("=" * 60)

try:
    import dlib

    detector = dlib.get_frontal_face_detector()
    detections = detector(img_array, 1)

    print(f"dlib version  : {dlib.__version__}")
    print(f"Detector type : HOG + SVM")
    print(f"Faces detected: {len(detections)}")

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img_array)

    for i, det in enumerate(detections):
        x1, y1, x2, y2 = det.left(), det.top(), det.right(), det.bottom()
        rect = plt.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor='lime', facecolor='none'
        )
        ax.add_patch(rect)
        ax.text(x1, y1 - 5, f"Face {i+1}", color='lime', fontsize=10, fontweight='bold')
        print(f"  Face {i+1}: ({x1}, {y1}) - ({x2}, {y2})")

    if not detections:
        ax.text(
            img_size // 2, img_size // 2,
            "No faces detected\n(synthetic image)",
            ha='center', va='center', fontsize=12, color='yellow',
            bbox={'boxstyle': 'round', 'facecolor': 'black', 'alpha': 0.7},
        )
        print("  Note: HOG detector may not detect stylized/synthetic faces")

    ax.set_title("dlib HOG Face Detection")
    ax.axis('off')
    plt.tight_layout()

    detection_path = os.path.join(OUTPUT_DIR, 'face_detection.png')
    plt.savefig(detection_path, dpi=150)
    print(f"Saved: {detection_path}")
    plt.close()

except Exception as e:
    print(f"dlib detection skipped: {e}")
    # Create placeholder
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img_array)
    ax.set_title("Face Detection (dlib unavailable)")
    ax.axis('off')
    detection_path = os.path.join(OUTPUT_DIR, 'face_detection.png')
    plt.savefig(detection_path, dpi=150)
    plt.close()

# =============================================================================
# DeepFace — Attribute Analysis
# =============================================================================
print("\n" + "=" * 60)
print("DeepFace — Attribute Analysis")
print("=" * 60)

try:
    from deepface import DeepFace

    print("Running DeepFace analysis (may download models on first use)...")
    results = DeepFace.analyze(
        img_path=input_path,
        actions=['age', 'gender', 'race', 'emotion'],
        enforce_detection=False,
        silent=True,
    )

    if isinstance(results, list):
        results = results[0] if results else {}

    if results:
        print(f"  Age              : {results.get('age', 'N/A')}")
        print(f"  Dominant gender  : {results.get('dominant_gender', 'N/A')}")
        print(f"  Dominant race    : {results.get('dominant_race', 'N/A')}")
        print(f"  Dominant emotion : {results.get('dominant_emotion', 'N/A')}")

        if 'emotion' in results:
            print("\n  Emotion scores:")
            for emotion, score in sorted(results['emotion'].items(), key=lambda x: -x[1]):
                bar = '█' * int(score / 2)
                print(f"    {emotion:12s}: {score:6.2f}% {bar}")
    else:
        print("  No analysis results (synthetic image may not be recognized as face)")

except Exception as e:
    print(f"DeepFace analysis skipped: {e}")

# =============================================================================
# face-alignment — Landmark Extraction
# =============================================================================
print("\n" + "=" * 60)
print("face-alignment — Landmark Extraction")
print("=" * 60)

try:
    import face_alignment

    fa = face_alignment.FaceAlignment(
        face_alignment.LandmarksType.TWO_D,
        device='cpu',
        flip_input=False,
    )

    landmarks = fa.get_landmarks(img_array)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img_array)

    if landmarks is not None and len(landmarks) > 0:
        print(f"Faces with landmarks: {len(landmarks)}")
        for face_idx, face_landmarks in enumerate(landmarks):
            print(f"  Face {face_idx + 1}: {face_landmarks.shape[0]} landmarks")
            ax.scatter(
                face_landmarks[:, 0], face_landmarks[:, 1],
                c='lime', s=8, zorder=5, edgecolors='black', linewidths=0.5,
            )
            # Connect landmark groups for visualization
            # Jaw: 0-16, Eyebrow L: 17-21, Eyebrow R: 22-26
            # Nose: 27-35, Eye L: 36-41, Eye R: 42-47, Mouth: 48-67
            groups = [
                (range(0, 17), 'cyan'),     # Jaw
                (range(17, 22), 'yellow'),   # Left eyebrow
                (range(22, 27), 'yellow'),   # Right eyebrow
                (range(27, 36), 'orange'),   # Nose
                (range(36, 42), 'lime'),     # Left eye
                (range(42, 48), 'lime'),     # Right eye
                (range(48, 60), 'red'),      # Outer mouth
                (range(60, 68), 'pink'),     # Inner mouth
            ]
            for indices, color in groups:
                idx_list = list(indices)
                if max(idx_list) < len(face_landmarks):
                    pts = face_landmarks[idx_list]
                    ax.plot(pts[:, 0], pts[:, 1], color=color, linewidth=1, alpha=0.7)
    else:
        print("  No landmarks detected (synthetic image)")
        ax.text(
            img_size // 2, img_size // 2,
            "No landmarks detected\n(synthetic image)",
            ha='center', va='center', fontsize=12, color='yellow',
            bbox={'boxstyle': 'round', 'facecolor': 'black', 'alpha': 0.7},
        )

    ax.set_title("Face Landmarks (face-alignment)")
    ax.axis('off')
    plt.tight_layout()

    landmarks_path = os.path.join(OUTPUT_DIR, 'face_landmarks.png')
    plt.savefig(landmarks_path, dpi=150)
    print(f"Saved: {landmarks_path}")
    plt.close()

except Exception as e:
    print(f"face-alignment skipped: {e}")
    # Create placeholder
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img_array)
    ax.set_title("Face Landmarks (face-alignment unavailable)")
    ax.axis('off')
    landmarks_path = os.path.join(OUTPUT_DIR, 'face_landmarks.png')
    plt.savefig(landmarks_path, dpi=150)
    plt.close()

# =============================================================================
# Library Version Summary
# =============================================================================
print("\n" + "=" * 60)
print("Library Versions")
print("=" * 60)

libraries = [
    ('numpy', 'numpy'),
    ('torch', 'torch'),
    ('torchvision', 'torchvision'),
    ('tensorflow', 'tensorflow'),
    ('cv2', 'opencv'),
    ('PIL', 'Pillow'),
    ('skimage', 'scikit-image'),
    ('dlib', 'dlib'),
    ('deepface', 'deepface'),
    ('face_alignment', 'face-alignment'),
    ('diffusers', 'diffusers'),
    ('mtcnn', 'mtcnn'),
]

for module_name, display_name in libraries:
    try:
        mod = __import__(module_name)
        version = getattr(mod, '__version__', 'unknown')
        print(f"  {display_name:25s}: {version}")
    except ImportError:
        print(f"  {display_name:25s}: not installed")

# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 60)
print("Face analysis complete!")
print("Outputs: face_synthetic_input.png, face_detection.png,")
print("         face_landmarks.png")
print("=" * 60)
