import os
import cv2
import numpy as np
from utils import readlines

SPLIT_FILE = "splits/c3vd/test_files.txt"
OUTPUT_PATH = "splits/c3vd/gt_depths.npz"

MIN_DEPTH = 0.01
MAX_DEPTH = 10.0

filenames = readlines(SPLIT_FILE)

gt_depths = []

for line in filenames:

    color_path = line.strip()

    # color ? depth? ??
    depth_path = color_path.replace("_color.png", "_depth.tiff")

    if not os.path.exists(depth_path):
        print("Missing:", depth_path)
        continue

    depth = cv2.imread(depth_path, -1)

    if depth is None:
        print("Failed:", depth_path)
        continue

    depth = depth.astype(np.float32)

    # mm ? meter
    depth /= 1000.0

    depth[(depth < MIN_DEPTH) | (depth > MAX_DEPTH)] = 0

    gt_depths.append(depth)

gt_depths = np.array(gt_depths)

print("GT shape:", gt_depths.shape)

np.savez_compressed(OUTPUT_PATH, data=gt_depths)

print("Saved to:", OUTPUT_PATH)
