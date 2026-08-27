#!/usr/bin/env python3
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../scripts")))
from figstyle import use_house_style, PALETTE, _finish

use_house_style()
here = os.path.dirname(os.path.abspath(__file__))

z = np.linspace(-5, 5, 200)
sig = 1.0 / (1.0 + np.exp(-z))

fig, ax = plt.subplots(figsize=(6.5, 3.2))
ax.plot(z, sig, color=PALETTE["blue"], lw=2.5, label="Sigmoid σ(z)")
ax.axhline(0.5, color=PALETTE["muted"], linestyle=":", lw=1)
ax.axvline(0, color=PALETTE["muted"], linestyle=":", lw=1)
ax.scatter([1.0], [1.0 / (1.0 + np.exp(-1.0))], color=PALETTE["red"], s=60, zorder=5)
ax.annotate("z = 1.0 -> σ(z) = 0.731\nP(y=1) > 0.5 -> Class 1", xy=(1.0, 0.731), xytext=(1.5, 0.4),
            arrowprops=dict(arrowstyle="->", color=PALETTE["ink"], lw=1.2),
            bbox=dict(boxstyle="round,pad=0.3", fc=PALETTE["fill"], ec=PALETTE["blue"], lw=1),
            fontsize=9, fontweight="bold")

ax.set_xlabel("z = w1*x1 + w2*x2 + b")
ax.set_ylabel("Probability σ(z)")
ax.set_title("Logistic Regression: S-Curve Mapping Score z to Probability")
ax.set_ylim(-0.05, 1.05)

out = os.path.join(here, "fig_logistic_sigmoid.png")
fig.tight_layout()
fig.savefig(out, dpi=220, bbox_inches="tight")
plt.close(fig)
print("Saved fig_logistic_sigmoid.png")
