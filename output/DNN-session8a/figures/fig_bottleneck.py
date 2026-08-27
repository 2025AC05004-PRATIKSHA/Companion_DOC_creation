#!/usr/bin/env python3
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../scripts")))
from figstyle import use_house_style, PALETTE

use_house_style()
here = os.path.dirname(os.path.abspath(__file__))

labels = ["Standard 3x3 Conv\n(256->256 channels)", "Bottleneck ResNet Block\n(256->64->64->256)"]
params = [589824, 69632] # 3*3*256*256 vs (1*1*256*64 + 3*3*64*64 + 1*1*64*256)

fig, ax = plt.subplots(figsize=(6.5, 3.2))
bars = ax.bar(labels, [p/1000 for p in params], color=[PALETTE["red"], PALETTE["green"]], width=0.45)

ax.set_ylabel("Parameters (in Thousands)")
ax.set_title("8.5x Parameter Reduction with Bottleneck Design (1x1 Convs)")
for bar, val in zip(bars, params):
    ax.text(bar.get_x() + bar.get_width()/2, (val/1000) + 15, f"{val:,} params", ha="center", fontweight="bold", fontsize=9.5)

ax.set_ylim(0, 700)

out = os.path.join(here, "fig_bottleneck.png")
fig.tight_layout()
fig.savefig(out, dpi=220, bbox_inches="tight")
plt.close(fig)
print("Saved fig_bottleneck.png")
