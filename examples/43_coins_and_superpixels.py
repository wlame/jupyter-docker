#!/usr/bin/env python3
# # Counting Coins and Segmenting a Photo
#
# Two classic image-analysis tasks on photos that ship with scikit-image, so the
# notebook runs offline: count and measure the coins in a greyscale photo
# (threshold, clean up, label connected regions, measure each one), then split a
# colour photo into superpixels (SLIC) and merge similar neighbours into larger
# segments.
#
# To try your own picture, put it in `/home/jupyter/data/` and set `PHOTO_FILE`.
#
# - [scikit-image](https://scikit-image.org/docs/stable/)

import csv
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from skimage import color, data, filters, graph, io, measure, morphology, segmentation

# Outputs go to ./output next to the notebook (the kernel's working directory).
OUTPUT_DIR = os.path.abspath('output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

PHOTO_FILE = None  # e.g. '/home/jupyter/data/street.jpg'

# ## Coins: separate the coins from the background
#
# Otsu's method picks the threshold that best splits the brightness histogram in
# two. Lighting varies across this photo, so a local (block-wise) threshold works
# better than one global value.

coins = data.coins()
local = filters.threshold_local(coins, block_size=101, offset=-10)
mask = coins > local
mask = morphology.remove_small_objects(morphology.closing(mask, morphology.disk(3)), 300)
mask = morphology.remove_small_holes(mask, 500)
print(f"Photo {coins.shape}, global Otsu threshold would be {filters.threshold_otsu(coins):.0f}")

# ## Coins: label and measure each region
#
# Connected foreground pixels form one region. Regions touching the border are
# dropped because those coins are cut off.

labels = segmentation.clear_border(measure.label(mask))
regions = sorted(
    (region for region in measure.regionprops(labels, intensity_image=coins) if region.area > 400),
    key=lambda region: region.equivalent_diameter_area,
    reverse=True,
)
print(f"Coins found: {len(regions)}")
print(f"{'diameter':>9} {'area':>7} {'eccentricity':>13} {'brightness':>11}")
for region in regions[:8]:
    print(f"{region.equivalent_diameter_area:9.1f} {region.area:7.0f} {region.eccentricity:13.2f} "
          f"{region.intensity_mean:11.1f}")

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
axes[0].imshow(coins, cmap='gray')
axes[0].set_title('Original')
axes[1].imshow(color.label2rgb(labels, image=coins, bg_label=0, alpha=0.4))
for region in regions:
    y, x = region.centroid
    axes[1].text(x, y, f"{region.equivalent_diameter_area:.0f}", color='white', ha='center', fontsize=8)
axes[1].set_title(f'{len(regions)} coins, diameter in pixels')
for ax in axes:
    ax.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'coins_measured.png'), dpi=110)
plt.show()

# ## Superpixels: SLIC
#
# SLIC groups pixels that are close in both position and colour into compact
# regions of similar size, a common first step before classifying image parts.

photo = io.imread(PHOTO_FILE)[..., :3] if PHOTO_FILE else data.astronaut()
superpixels = segmentation.slic(photo, n_segments=400, compactness=10, start_label=1)
print(f"Photo {photo.shape}: {superpixels.max()} superpixels")

# ## Merge similar neighbours
#
# A region adjacency graph links touching superpixels, weighted by how different
# their mean colours are; cutting weak links merges similar neighbours.

rag = graph.rag_mean_color(photo, superpixels)
merged = graph.cut_threshold(superpixels, rag, thresh=29)
print(f"After merging: {len(np.unique(merged))} segments")

fig, axes = plt.subplots(1, 3, figsize=(14, 5))
axes[0].imshow(photo)
axes[0].set_title('Original')
axes[1].imshow(segmentation.mark_boundaries(photo, superpixels))
axes[1].set_title(f'{superpixels.max()} SLIC superpixels')
axes[2].imshow(color.label2rgb(merged, photo, kind='avg', bg_label=0))
axes[2].set_title(f'{len(np.unique(merged))} merged segments (mean colour)')
for ax in axes:
    ax.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'superpixels.png'), dpi=110)
plt.show()

with open(os.path.join(OUTPUT_DIR, 'coins.csv'), 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['label', 'diameter', 'area', 'eccentricity', 'brightness'])
    writer.writerows([r.label, round(r.equivalent_diameter_area, 1), r.area, round(r.eccentricity, 3),
                      round(r.intensity_mean, 1)] for r in regions)
with open(os.path.join(OUTPUT_DIR, 'segmentation_summary.json'), 'w') as f:
    json.dump({'coins': len(regions), 'superpixels': int(superpixels.max()),
               'merged_segments': int(len(np.unique(merged)))}, f)
print("Saved: coins_measured.png, superpixels.png, coins.csv, segmentation_summary.json")
