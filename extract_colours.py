#! /bin/env python
from matplotlib.gridspec import GridSpec
from skimage.color import lab2rgb, rgb2lab
from sklearn.cluster import DBSCAN
import numpy as np
import argparse
import cv2
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("image", help="Path to input image")

args = parser.parse_args()

img = cv2.imread(args.image)
if img is None:
    print("Could not read image")
    exit()
img = cv2.resize(img, (64, 64))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
rgb = img.reshape(-1, 3) / 255
rgb_sample = rgb[:: len(rgb) // 512]
lab_img = rgb2lab(img)
lab = lab_img.reshape(-1, 3)
lab_sample = lab[:: len(lab) // 512]

model = DBSCAN(eps=5, min_samples=10)
labels = model.fit_predict(lab)
unique_labels = set(labels)
unique_labels.discard(-1)

centers = []
for lbl in unique_labels:
    centers.append(lab[labels == lbl].mean(axis=0))

centers = np.array(centers)
centers_rgb = lab2rgb(centers)

fig = plt.figure()
gs = GridSpec(2, 1, height_ratios=[8, 1])
ax = fig.add_subplot(gs[0], projection="3d")
ax.scatter(lab_sample[:, 0], lab_sample[:, 1], lab_sample[:, 2], c=rgb_sample, s=10)  # type: ignore
ax.scatter(
    centers[:, 0],  # type: ignore
    centers[:, 1],  # type: ignore
    centers[:, 2],  # type: ignore
    c=centers_rgb,
    edgecolors="black",
    linewidths=1.5,
    s=100,
)

ax_colors = fig.add_subplot(gs[1])
for i, color in enumerate(centers_rgb):
    ax_colors.add_patch(plt.Rectangle((i, 0), 1, 1, color=color))
ax_colors.set_xlim(0, len(centers_rgb))
ax_colors.set_ylim(0, 1)
ax_colors.axis("off")

plt.tight_layout()

plt.show()

