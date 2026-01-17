#!/usr/bin/env python3
"""
Image Processing Basics
=======================
Demonstrates image processing with PIL, OpenCV, scikit-image, and imageio.

Pillow: https://pillow.readthedocs.io/
OpenCV: https://docs.opencv.org/
scikit-image: https://scikit-image.org/
imageio: https://imageio.readthedocs.io/
"""

import os
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
import cv2
from skimage import io, filters, feature, color, transform, exposure
from skimage.draw import disk, rectangle
import imageio.v3 as iio

# Create output directory
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# Create Sample Image
# =============================================================================
print("=" * 60)
print("Creating Sample Image")
print("=" * 60)

# Create a sample image using PIL
width, height = 400, 300
sample_image = Image.new("RGB", (width, height), color=(255, 255, 255))
draw = ImageDraw.Draw(sample_image)

# Draw shapes
draw.rectangle([50, 50, 150, 150], fill=(255, 0, 0), outline=(0, 0, 0))
draw.ellipse([200, 50, 350, 200], fill=(0, 255, 0), outline=(0, 0, 0))
draw.polygon([(100, 200), (50, 280), (150, 280)], fill=(0, 0, 255), outline=(0, 0, 0))

# Add gradient background
for i in range(width):
    for j in range(height):
        pixel = sample_image.getpixel((i, j))
        if pixel == (255, 255, 255):
            gray = int(255 * (1 - j / height * 0.3))
            sample_image.putpixel((i, j), (gray, gray, int(gray * 0.9)))

sample_path = os.path.join(OUTPUT_DIR, "sample_image.png")
sample_image.save(sample_path)
print(f"Sample image created: {sample_path}")
print(f"Size: {sample_image.size}")
print(f"Mode: {sample_image.mode}")

# =============================================================================
# PIL (Pillow) Operations
# =============================================================================
print("\n" + "=" * 60)
print("PIL (Pillow) Operations")
print("=" * 60)

# Load and basic info
img = Image.open(sample_path)
print(f"Image size: {img.size}")
print(f"Image mode: {img.mode}")
print(f"Image format: {img.format}")

# Resize
resized = img.resize((200, 150), Image.Resampling.LANCZOS)
resized.save(os.path.join(OUTPUT_DIR, "pil_resized.png"))
print(f"Resized to: {resized.size}")

# Rotate
rotated = img.rotate(45, expand=True, fillcolor=(128, 128, 128))
rotated.save(os.path.join(OUTPUT_DIR, "pil_rotated.png"))
print("Rotated 45 degrees")

# Apply filters
blurred = img.filter(ImageFilter.GaussianBlur(radius=5))
blurred.save(os.path.join(OUTPUT_DIR, "pil_blurred.png"))
print("Applied Gaussian blur")

edges = img.filter(ImageFilter.FIND_EDGES)
edges.save(os.path.join(OUTPUT_DIR, "pil_edges.png"))
print("Applied edge detection")

sharpened = img.filter(ImageFilter.SHARPEN)
sharpened.save(os.path.join(OUTPUT_DIR, "pil_sharpened.png"))
print("Applied sharpening")

# Enhance
enhancer = ImageEnhance.Contrast(img)
contrast = enhancer.enhance(1.5)
contrast.save(os.path.join(OUTPUT_DIR, "pil_contrast.png"))
print("Enhanced contrast")

# Convert to grayscale
grayscale = img.convert("L")
grayscale.save(os.path.join(OUTPUT_DIR, "pil_grayscale.png"))
print("Converted to grayscale")

# Thumbnail
thumbnail = img.copy()
thumbnail.thumbnail((100, 100))
thumbnail.save(os.path.join(OUTPUT_DIR, "pil_thumbnail.png"))
print(f"Created thumbnail: {thumbnail.size}")

# =============================================================================
# OpenCV Operations
# =============================================================================
print("\n" + "=" * 60)
print("OpenCV Operations")
print("=" * 60)

print(f"OpenCV version: {cv2.__version__}")

# Load image (OpenCV uses BGR by default)
cv_img = cv2.imread(sample_path)
print(f"Image shape: {cv_img.shape}")
print(f"Image dtype: {cv_img.dtype}")

# Convert BGR to RGB
cv_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)

# Resize
cv_resized = cv2.resize(cv_img, (200, 150), interpolation=cv2.INTER_LANCZOS4)
cv2.imwrite(os.path.join(OUTPUT_DIR, "cv_resized.png"), cv_resized)
print(f"Resized to: {cv_resized.shape[:2]}")

# Gaussian blur
cv_blurred = cv2.GaussianBlur(cv_img, (15, 15), 0)
cv2.imwrite(os.path.join(OUTPUT_DIR, "cv_blurred.png"), cv_blurred)
print("Applied Gaussian blur")

# Canny edge detection
cv_gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
cv_edges = cv2.Canny(cv_gray, 100, 200)
cv2.imwrite(os.path.join(OUTPUT_DIR, "cv_canny_edges.png"), cv_edges)
print("Applied Canny edge detection")

# Thresholding
_, cv_thresh = cv2.threshold(cv_gray, 127, 255, cv2.THRESH_BINARY)
cv2.imwrite(os.path.join(OUTPUT_DIR, "cv_threshold.png"), cv_thresh)
print("Applied binary thresholding")

# Adaptive thresholding
cv_adaptive = cv2.adaptiveThreshold(
    cv_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
)
cv2.imwrite(os.path.join(OUTPUT_DIR, "cv_adaptive_threshold.png"), cv_adaptive)
print("Applied adaptive thresholding")

# Morphological operations
kernel = np.ones((5, 5), np.uint8)
cv_dilated = cv2.dilate(cv_edges, kernel, iterations=1)
cv_eroded = cv2.erode(cv_edges, kernel, iterations=1)
cv2.imwrite(os.path.join(OUTPUT_DIR, "cv_dilated.png"), cv_dilated)
cv2.imwrite(os.path.join(OUTPUT_DIR, "cv_eroded.png"), cv_eroded)
print("Applied morphological operations (dilate, erode)")

# Draw on image
cv_draw = cv_img.copy()
cv2.rectangle(cv_draw, (10, 10), (100, 50), (0, 255, 255), 2)
cv2.circle(cv_draw, (300, 150), 40, (255, 0, 255), 3)
cv2.line(cv_draw, (0, 0), (400, 300), (255, 255, 0), 2)
cv2.putText(cv_draw, "OpenCV", (150, 280), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
cv2.imwrite(os.path.join(OUTPUT_DIR, "cv_drawing.png"), cv_draw)
print("Drew shapes and text")

# =============================================================================
# scikit-image Operations
# =============================================================================
print("\n" + "=" * 60)
print("scikit-image Operations")
print("=" * 60)

# Load image
ski_img = io.imread(sample_path)
print(f"Image shape: {ski_img.shape}")
print(f"Image dtype: {ski_img.dtype}")

# Convert to grayscale
ski_gray = color.rgb2gray(ski_img)
io.imsave(os.path.join(OUTPUT_DIR, "ski_grayscale.png"), (ski_gray * 255).astype(np.uint8))
print("Converted to grayscale")

# Edge detection - Sobel
ski_sobel = filters.sobel(ski_gray)
io.imsave(os.path.join(OUTPUT_DIR, "ski_sobel.png"), (ski_sobel * 255).astype(np.uint8))
print("Applied Sobel edge detection")

# Edge detection - Canny
ski_canny = feature.canny(ski_gray, sigma=2)
io.imsave(os.path.join(OUTPUT_DIR, "ski_canny.png"), (ski_canny * 255).astype(np.uint8))
print("Applied Canny edge detection")

# Gaussian filter
ski_gaussian = filters.gaussian(ski_img, sigma=3, channel_axis=-1)
io.imsave(os.path.join(OUTPUT_DIR, "ski_gaussian.png"), (ski_gaussian * 255).astype(np.uint8))
print("Applied Gaussian filter")

# Histogram equalization
ski_eq = exposure.equalize_hist(ski_gray)
io.imsave(os.path.join(OUTPUT_DIR, "ski_equalized.png"), (ski_eq * 255).astype(np.uint8))
print("Applied histogram equalization")

# Contrast stretching
p2, p98 = np.percentile(ski_gray, (2, 98))
ski_rescale = exposure.rescale_intensity(ski_gray, in_range=(p2, p98))
io.imsave(os.path.join(OUTPUT_DIR, "ski_rescaled.png"), (ski_rescale * 255).astype(np.uint8))
print("Applied contrast stretching")

# Resize
ski_resized = transform.resize(ski_img, (150, 200), anti_aliasing=True)
io.imsave(os.path.join(OUTPUT_DIR, "ski_resized.png"), (ski_resized * 255).astype(np.uint8))
print(f"Resized to: {ski_resized.shape[:2]}")

# Rotate
ski_rotated = transform.rotate(ski_img, 30, resize=True)
io.imsave(os.path.join(OUTPUT_DIR, "ski_rotated.png"), (ski_rotated * 255).astype(np.uint8))
print("Rotated 30 degrees")

# Corner detection (Harris)
ski_corners = feature.corner_harris(ski_gray)
ski_corners_peaks = feature.corner_peaks(ski_corners, min_distance=10)
print(f"Detected {len(ski_corners_peaks)} corners using Harris detector")

# =============================================================================
# imageio Operations
# =============================================================================
print("\n" + "=" * 60)
print("imageio Operations")
print("=" * 60)

# Read image
iio_img = iio.imread(sample_path)
print(f"Image shape: {iio_img.shape}")
print(f"Image dtype: {iio_img.dtype}")

# Get image metadata
props = iio.improps(sample_path)
print(f"Image properties: shape={props.shape}, dtype={props.dtype}")

# Write image
iio.imwrite(os.path.join(OUTPUT_DIR, "iio_copy.png"), iio_img)
print("Saved copy of image")

# Create an animated GIF
print("\nCreating animated GIF...")
frames = []
for angle in range(0, 360, 30):
    rotated_frame = transform.rotate(ski_img, angle, resize=False)
    frames.append((rotated_frame * 255).astype(np.uint8))

gif_path = os.path.join(OUTPUT_DIR, "iio_animation.gif")
iio.imwrite(gif_path, frames, duration=100, loop=0)
print(f"Created animated GIF with {len(frames)} frames: {gif_path}")

# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 60)
print("Summary - Output Files Created")
print("=" * 60)

output_files = sorted(os.listdir(OUTPUT_DIR))
for f in output_files:
    filepath = os.path.join(OUTPUT_DIR, f)
    size = os.path.getsize(filepath)
    print(f"  {f}: {size:,} bytes")

print(f"\nTotal files: {len(output_files)}")

print("\n" + "=" * 60)
print("Image processing example complete!")
print("=" * 60)
