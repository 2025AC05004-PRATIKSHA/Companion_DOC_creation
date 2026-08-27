#!/usr/bin/env python3
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../scripts")))
from figstyle import use_house_style, PALETTE

use_house_style()
here = os.path.dirname(os.path.abspath(__file__))

fig, ax = plt.subplots(figsize=(6.5, 3.2))

# Draw 6x6 grid
input_img = np.array([
    [1, 0, 0, 0, 0, 1],
    [0, 1, 0, 0, 1, 0],
    [0, 0, 1, 1, 0, 0],
    [1, 0, 0, 0, 1, 0],
    [0, 1, 0, 0, 1, 0],
    [0, 0, 1, 0, 1, 0]
])

feature_map = np.array([
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 0, 3]
])

maxpool = np.array([
    [0, 0],
    [0, 3]
])

ax.imshow(input_img, cmap="Blues", alpha=0.6)
for i in range(6):
    for j in range(6):
        ax.text(j, i, str(input_img[i, j]), ha="center", va="center", fontweight="bold", fontsize=10)

ax.set_xticks(np.arange(-0.5, 6, 1))
ax.set_yticks(np.arange(-0.5, 6, 1))
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.grid(True, color=PALETTE["muted"], lw=1)
ax.set_title("Convolution Step: 6x6 Input Image -> 3x3 Filter -> 4x4 Output -> 2x2 Pool")

out = os.path.join(here, "fig_conv_pool.png")
fig.tight_layout()
fig.savefig(out, dpi=220, bbox_inches="tight")
plt.close(fig)
print("Saved fig_conv_pool.png")
