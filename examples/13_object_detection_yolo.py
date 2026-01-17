#!/usr/bin/env python3
"""
Object Detection with Ultralytics YOLO
======================================
Demonstrates object detection using Ultralytics YOLOv8.

Ultralytics: https://ultralytics.com/
Documentation: https://docs.ultralytics.com/
"""

import os
import numpy as np
from PIL import Image, ImageDraw
from ultralytics import YOLO

# Create output directory
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# YOLO Model Overview
# =============================================================================
print("=" * 60)
print("Ultralytics YOLO Overview")
print("=" * 60)

print("""
YOLO (You Only Look Once) is a state-of-the-art object detection model.
Ultralytics provides easy-to-use implementations of YOLOv8 and beyond.

Available model sizes (speed vs accuracy trade-off):
  - yolov8n.pt : Nano     - fastest, lowest accuracy
  - yolov8s.pt : Small    - fast, good accuracy
  - yolov8m.pt : Medium   - balanced
  - yolov8l.pt : Large    - slower, better accuracy
  - yolov8x.pt : XLarge   - slowest, best accuracy

Task-specific models:
  - Detection:    yolov8n.pt (default)
  - Segmentation: yolov8n-seg.pt
  - Classification: yolov8n-cls.pt
  - Pose:         yolov8n-pose.pt
""")

# =============================================================================
# Create Sample Image with Objects
# =============================================================================
print("=" * 60)
print("Creating Sample Image")
print("=" * 60)

# Create a sample scene image
width, height = 640, 480
sample_image = Image.new("RGB", (width, height), color=(135, 206, 235))  # Sky blue
draw = ImageDraw.Draw(sample_image)

# Draw ground
draw.rectangle([0, height * 0.7, width, height], fill=(34, 139, 34))  # Green grass

# Draw a simple "car" (rectangle with wheels)
car_x, car_y = 100, 300
draw.rectangle([car_x, car_y, car_x + 150, car_y + 60], fill=(255, 0, 0))  # Car body
draw.rectangle([car_x + 20, car_y - 30, car_x + 130, car_y], fill=(255, 0, 0))  # Car top
draw.ellipse([car_x + 20, car_y + 40, car_x + 50, car_y + 70], fill=(30, 30, 30))  # Wheel 1
draw.ellipse([car_x + 100, car_y + 40, car_x + 130, car_y + 70], fill=(30, 30, 30))  # Wheel 2

# Draw a simple "person" (stick figure representation)
person_x, person_y = 400, 280
draw.ellipse([person_x - 15, person_y - 50, person_x + 15, person_y - 20], fill=(255, 218, 185))  # Head
draw.rectangle([person_x - 20, person_y - 20, person_x + 20, person_y + 50], fill=(0, 0, 255))  # Body
draw.rectangle([person_x - 10, person_y + 50, person_x + 10, person_y + 100], fill=(0, 0, 139))  # Legs

# Draw a "dog" shape
dog_x, dog_y = 500, 340
draw.ellipse([dog_x, dog_y, dog_x + 60, dog_y + 40], fill=(139, 69, 19))  # Body
draw.ellipse([dog_x + 45, dog_y - 15, dog_x + 70, dog_y + 10], fill=(139, 69, 19))  # Head
draw.rectangle([dog_x + 5, dog_y + 35, dog_x + 15, dog_y + 55], fill=(139, 69, 19))  # Leg 1
draw.rectangle([dog_x + 45, dog_y + 35, dog_x + 55, dog_y + 55], fill=(139, 69, 19))  # Leg 2

# Draw sun
draw.ellipse([width - 100, 30, width - 30, 100], fill=(255, 255, 0))

# Draw cloud
for cx, cy in [(150, 60), (180, 50), (210, 60), (180, 70)]:
    draw.ellipse([cx - 25, cy - 20, cx + 25, cy + 20], fill=(255, 255, 255))

sample_path = os.path.join(OUTPUT_DIR, "yolo_sample_scene.png")
sample_image.save(sample_path)
print(f"Sample scene created: {sample_path}")
print(f"Size: {sample_image.size}")

# =============================================================================
# Load YOLO Model
# =============================================================================
print("\n" + "=" * 60)
print("Loading YOLO Model")
print("=" * 60)

# Load pretrained YOLOv8 nano model (smallest, fastest)
print("Loading YOLOv8n model (this may download the model on first run)...")
model = YOLO("yolov8n.pt")

print(f"Model loaded successfully!")
print(f"Model type: {type(model).__name__}")
print(f"Task: {model.task}")

# =============================================================================
# Model Information
# =============================================================================
print("\n" + "=" * 60)
print("Model Information")
print("=" * 60)

# Get class names
class_names = model.names
print(f"Number of classes: {len(class_names)}")
print(f"Sample classes: {list(class_names.values())[:10]}...")

# =============================================================================
# Run Inference
# =============================================================================
print("\n" + "=" * 60)
print("Running Inference on Sample Image")
print("=" * 60)

# Run inference
results = model(sample_path, verbose=False)

# Process results
result = results[0]
print(f"Image shape: {result.orig_shape}")
print(f"Inference speed: {result.speed}")

# Get detections
boxes = result.boxes
print(f"\nDetections found: {len(boxes)}")

if len(boxes) > 0:
    print("\nDetected objects:")
    for i, box in enumerate(boxes):
        cls_id = int(box.cls[0])
        cls_name = class_names[cls_id]
        conf = float(box.conf[0])
        xyxy = box.xyxy[0].tolist()
        print(f"  {i+1}. {cls_name}: confidence={conf:.2f}, box={[int(x) for x in xyxy]}")
else:
    print("No objects detected in the simple drawn image.")
    print("(YOLO is trained on real photos, not simple drawings)")

# =============================================================================
# Save Annotated Image
# =============================================================================
print("\n" + "=" * 60)
print("Saving Results")
print("=" * 60)

# Save annotated image
annotated_path = os.path.join(OUTPUT_DIR, "yolo_annotated.png")
annotated_img = result.plot()  # Returns numpy array with annotations
Image.fromarray(annotated_img).save(annotated_path)
print(f"Annotated image saved: {annotated_path}")

# =============================================================================
# Using YOLO with NumPy Arrays
# =============================================================================
print("\n" + "=" * 60)
print("Using YOLO with NumPy Arrays")
print("=" * 60)

# Convert PIL image to numpy array
img_array = np.array(sample_image)
print(f"Input array shape: {img_array.shape}")

# Run inference on numpy array
results_np = model(img_array, verbose=False)
print(f"Inference on numpy array: {len(results_np[0].boxes)} detections")

# =============================================================================
# Batch Processing Example
# =============================================================================
print("\n" + "=" * 60)
print("Batch Processing Example")
print("=" * 60)

# Create multiple images for batch processing
batch_images = []
for i in range(3):
    # Create slightly different images
    img = sample_image.copy()
    draw = ImageDraw.Draw(img)
    # Add a unique marker to each
    draw.rectangle([10, 10, 50, 50], fill=(i * 80, 100, 200))
    batch_images.append(np.array(img))

# Run batch inference
batch_results = model(batch_images, verbose=False)
print(f"Processed {len(batch_results)} images in batch")
for i, res in enumerate(batch_results):
    print(f"  Image {i+1}: {len(res.boxes)} detections")

# =============================================================================
# Export Model (Information)
# =============================================================================
print("\n" + "=" * 60)
print("Model Export Options")
print("=" * 60)

print("""
YOLO models can be exported to various formats for deployment:

  model.export(format='onnx')      # ONNX format
  model.export(format='torchscript')  # TorchScript
  model.export(format='tflite')    # TensorFlow Lite
  model.export(format='coreml')    # CoreML (iOS)
  model.export(format='engine')    # TensorRT

Supported export formats:
  - PyTorch, TorchScript, ONNX, OpenVINO
  - TensorRT, CoreML, TF SavedModel
  - TF GraphDef, TF Lite, TF Edge TPU
  - TF.js, PaddlePaddle, ncnn
""")

# =============================================================================
# Training Information
# =============================================================================
print("\n" + "=" * 60)
print("Training Custom Models")
print("=" * 60)

print("""
To train YOLO on custom data:

1. Prepare dataset in YOLO format:
   dataset/
   ├── train/
   │   ├── images/
   │   └── labels/
   └── val/
       ├── images/
       └── labels/

2. Create data.yaml:
   train: path/to/train/images
   val: path/to/val/images
   nc: 2  # number of classes
   names: ['class1', 'class2']

3. Train:
   model = YOLO('yolov8n.pt')
   model.train(data='data.yaml', epochs=100, imgsz=640)

4. Validate:
   metrics = model.val()

5. Predict:
   results = model('image.jpg')
""")

# =============================================================================
# Available Tasks
# =============================================================================
print("\n" + "=" * 60)
print("YOLO Task Examples")
print("=" * 60)

print("""
Detection (default):
  model = YOLO('yolov8n.pt')
  results = model('image.jpg')

Segmentation:
  model = YOLO('yolov8n-seg.pt')
  results = model('image.jpg')
  masks = results[0].masks  # Instance segmentation masks

Classification:
  model = YOLO('yolov8n-cls.pt')
  results = model('image.jpg')
  probs = results[0].probs  # Classification probabilities

Pose Estimation:
  model = YOLO('yolov8n-pose.pt')
  results = model('image.jpg')
  keypoints = results[0].keypoints  # Body keypoints
""")

print("\n" + "=" * 60)
print("YOLO object detection example complete!")
print("=" * 60)
