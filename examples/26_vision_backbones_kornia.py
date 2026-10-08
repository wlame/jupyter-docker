#!/usr/bin/env python3
"""
Vision Building Blocks: timm, kornia, supervision, OpenCLIP, onnxruntime
========================================================================
Builds image backbones with timm and OpenCLIP (architectures only: pass
`pretrained=True` / a `pretrained` tag to download weights), runs
differentiable image ops and augmentations with kornia, and draws detections
with supervision — the annotation toolkit that pairs with Ultralytics YOLO.

timm:        https://huggingface.co/docs/timm/
kornia:      https://kornia.readthedocs.io/
supervision: https://supervision.roboflow.com/
OpenCLIP:    https://github.com/mlfoundations/open_clip
onnxruntime: https://onnxruntime.ai/docs/
"""

import os

import cv2
import kornia
import kornia.augmentation as K
import numpy as np
import onnxruntime as ort
import open_clip
import supervision as sv
import timm
import torch

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

torch.manual_seed(0)
rng = np.random.default_rng(seed=0)

# A synthetic scene: two filled shapes on a gradient background.
height, width = 240, 320
scene = np.zeros((height, width, 3), dtype=np.uint8)
scene[:] = np.linspace(40, 200, width, dtype=np.uint8)[None, :, None]
cv2.rectangle(scene, (30, 40), (130, 160), (220, 60, 60), thickness=-1)
cv2.circle(scene, (230, 140), 55, (60, 180, 90), thickness=-1)

# =============================================================================
# timm — backbone architectures and feature maps
# =============================================================================
print("=" * 60)
print("timm: Backbones")
print("=" * 60)

print(f"timm offers {len(timm.list_models())} architectures, e.g. {timm.list_models('convnext*')[:3]}")
backbone = timm.create_model('resnet18', pretrained=False, features_only=True, out_indices=(2, 3, 4))
backbone.eval()
batch = torch.from_numpy(scene).permute(2, 0, 1).float().unsqueeze(0) / 255.0
with torch.no_grad():
    feature_maps = backbone(batch)
print("Feature map shapes:", [tuple(f.shape) for f in feature_maps])

# =============================================================================
# kornia — differentiable image processing and augmentation
# =============================================================================
print("\n" + "=" * 60)
print("kornia: Edges and Augmentations")
print("=" * 60)

gray = kornia.color.rgb_to_grayscale(batch)
edges = kornia.filters.sobel(gray)
blurred = kornia.filters.gaussian_blur2d(batch, kernel_size=(9, 9), sigma=(2.0, 2.0))
augment = K.AugmentationSequential(
    K.RandomHorizontalFlip(p=1.0),
    K.ColorJitter(0.3, 0.3, 0.3, 0.05, p=1.0),
    K.RandomRotation(degrees=15.0, p=1.0),
)
augmented = augment(batch)
print(f"Edge magnitude range: {edges.min().item():.3f}..{edges.max().item():.3f}; "
      f"augmented batch {tuple(augmented.shape)}")


def to_uint8(tensor: torch.Tensor) -> np.ndarray:
    """(1, C, H, W) float tensor in [0, 1] -> (H, W, 3) uint8 image."""
    image = tensor[0].clamp(0, 1).permute(1, 2, 0).numpy()
    if image.shape[2] == 1:
        image = np.repeat(image, 3, axis=2)
    return (image * 255).astype(np.uint8)


edge_view = to_uint8(edges / edges.max())
panel = np.concatenate([scene, to_uint8(blurred), edge_view, to_uint8(augmented)], axis=1)
cv2.imwrite(os.path.join(OUTPUT_DIR, 'kornia_ops.png'), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
print("Saved: kornia_ops.png (original | blur | edges | augmented)")

# =============================================================================
# supervision — annotate detections
# =============================================================================
print("\n" + "=" * 60)
print("supervision: Draw Detections")
print("=" * 60)

detections = sv.Detections(
    xyxy=np.array([[30, 40, 130, 160], [175, 85, 285, 195]], dtype=float),
    confidence=np.array([0.94, 0.88]),
    class_id=np.array([0, 1]),
)
labels = [f"{name} {conf:.2f}" for name, conf in zip(['box', 'disc'], detections.confidence)]
annotated = sv.BoxAnnotator(thickness=2).annotate(scene.copy(), detections)
annotated = sv.LabelAnnotator(text_scale=0.5).annotate(annotated, detections, labels=labels)
cv2.imwrite(os.path.join(OUTPUT_DIR, 'supervision_annotated.png'), cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
print(f"{len(detections)} detections, areas {detections.area.astype(int).tolist()}")
print("Saved: supervision_annotated.png")

# =============================================================================
# OpenCLIP — image/text embedding model (architecture only)
# =============================================================================
print("\n" + "=" * 60)
print("OpenCLIP: Image and Text Towers")
print("=" * 60)

clip_model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained=None)
tokenizer = open_clip.get_tokenizer('ViT-B-32')
clip_model.eval()
with torch.no_grad():
    image_features = clip_model.encode_image(torch.randn(1, 3, 224, 224))
    text_features = clip_model.encode_text(tokenizer(['a red box', 'a green disc']))
print(f"Image embedding {tuple(image_features.shape)}, text embeddings {tuple(text_features.shape)}")
print(f"{len(open_clip.list_pretrained())} pretrained checkpoints are available by name")

# =============================================================================
# onnxruntime — inference engine for exported models
# =============================================================================
print("\n" + "=" * 60)
print("onnxruntime")
print("=" * 60)
print(f"onnxruntime {ort.__version__}, providers: {ort.get_available_providers()}")
print("Export a YOLO model to ONNX and run it here with ort.InferenceSession(path).")

print("\nDone.")
