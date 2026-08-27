#!/usr/bin/env python3
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Ensure scripts directory is in path for figstyle
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../scripts")))
from figstyle import use_house_style, PALETTE, _finish

use_house_style()
here = os.path.dirname(os.path.abspath(__file__))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.2))

# AND Gate - Linearly Separable
ax1.scatter([0, 0, 1], [0, 1, 0], color=PALETTE["red"], s=80, label="Class 0", zorder=3)
ax1.scatter([1], [1], color=PALETTE["green"], s=80, label="Class 1", zorder=3)
ax1.plot([-0.2, 1.4], [1.3, -0.3], color=PALETTE["blue"], lw=2, linestyle="--", label="Decision Line")
ax1.set_xlim(-0.3, 1.3)
ax1.set_ylim(-0.3, 1.3)
ax1.set_title("AND Gate (Linearly Separable)")
ax1.set_xlabel("x1")
ax1.set_ylabel("x2")
ax1.legend(loc="upper left", fontsize=8)

# XOR Gate - Non-Linearly Separable
ax2.scatter([0, 1], [0, 1], color=PALETTE["red"], s=80, label="Class 0 (0)", zorder=3)
ax2.scatter([0, 1], [1, 0], color=PALETTE["green"], s=80, label="Class 1 (1)", zorder=3)
ax2.set_xlim(-0.3, 1.3)
ax2.set_ylim(-0.3, 1.3)
ax2.set_title("XOR Gate (Fails Single Perceptron)")
ax2.set_xlabel("x1")
ax2.set_ylabel("x2")
ax2.annotate("No single straight line\ncan separate green & red!", xy=(0.5, 0.5), xytext=(0.15, 0.55),
             bbox=dict(boxstyle="round,pad=0.3", fc="#FFF2E6", ec=PALETTE["amber"], lw=1),
             fontsize=8.5, fontweight="bold", color=PALETTE["ink"])

out = os.path.join(here, "fig_perceptron_xor.png")
_finish(fig, ax1, "", None) # tight layout and save via finish or direct
fig.tight_layout()
fig.savefig(out, dpi=220, bbox_inches="tight")
plt.close(fig)
print("Saved fig_perceptron_xor.png")
