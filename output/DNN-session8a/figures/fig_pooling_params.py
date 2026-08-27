#!/usr/bin/env python3
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../scripts")))
from figstyle import use_house_style, PALETTE

use_house_style()
here = os.path.dirname(os.path.abspath(__file__))

layers = ["Input", "Conv1", "Pool1", "Conv2", "Pool2", "FC Output"]
params = [0, 4864, 0, 73856, 0, 2570]

fig, ax = plt.subplots(figsize=(6.5, 3.2))
bars = ax.bar(layers, [p/1000 for p in params], color=PALETTE["blue"], width=0.5)

ax.set_ylabel("Parameters (in Thousands)")
ax.set_title("CNN Layer Parameters: Pooling Layers Have ZERO Parameters")
for bar, val in zip(bars, params):
    ax.text(bar.get_x() + bar.get_width()/2, (val/1000) + 1.5, f"{val:,}", ha="center", fontweight="bold", fontsize=9)

ax.set_ylim(0, 90)

out = os.path.join(here, "fig_pooling_params.png")
fig.tight_layout()
fig.savefig(out, dpi=220, bbox_inches="tight")
plt.close(fig)
print("Saved fig_pooling_params.png")
