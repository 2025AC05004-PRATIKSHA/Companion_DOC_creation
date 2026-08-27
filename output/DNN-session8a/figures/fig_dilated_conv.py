#!/usr/bin/env python3
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../scripts")))
from figstyle import use_house_style, PALETTE

use_house_style()
here = os.path.dirname(os.path.abspath(__file__))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.2))

# Standard Convolution (r=1)
grid1 = np.zeros((7, 7))
grid1[2:5, 2:5] = 1.0 # 3x3 kernel
ax1.imshow(grid1, cmap="Blues", alpha=0.7)
ax1.set_title("Standard Conv (Rate r=1, 3x3)")
ax1.set_xticks([])
ax1.set_yticks([])
ax1.grid(True, color=PALETTE["muted"])
for i in range(7):
    for j in range(7):
        if grid1[i, j] == 1.0:
            ax1.text(j, i, "w", ha="center", va="center", color=PALETTE["ink"], fontweight="bold")

# Dilated Convolution (r=2)
grid2 = np.zeros((7, 7))
grid2[1, 1] = 1.0; grid2[1, 3] = 1.0; grid2[1, 5] = 1.0
grid2[3, 1] = 1.0; grid2[3, 3] = 1.0; grid2[3, 5] = 1.0
grid2[5, 1] = 1.0; grid2[5, 3] = 1.0; grid2[5, 5] = 1.0
ax2.imshow(grid2, cmap="Oranges", alpha=0.7)
ax2.set_title("Dilated Conv (Rate r=2, Field 5x5)")
ax2.set_xticks([])
ax2.set_yticks([])
ax2.grid(True, color=PALETTE["muted"])
for i in range(7):
    for j in range(7):
        if grid2[i, j] == 1.0:
            ax2.text(j, i, "w", ha="center", va="center", color=PALETTE["ink"], fontweight="bold")

out = os.path.join(here, "fig_dilated_conv.png")
fig.tight_layout()
fig.savefig(out, dpi=220, bbox_inches="tight")
plt.close(fig)
print("Saved fig_dilated_conv.png")
