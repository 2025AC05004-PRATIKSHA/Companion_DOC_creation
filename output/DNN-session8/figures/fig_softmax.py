#!/usr/bin/env python3
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../scripts")))
from figstyle import use_house_style, PALETTE

use_house_style()
here = os.path.dirname(os.path.abspath(__file__))

logits = np.array([2.0, 1.0, 0.1])
exp_logits = np.exp(logits)
softmax_probs = exp_logits / np.sum(exp_logits)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.2))

classes = ["Cat", "Dog", "Bird"]
bars1 = ax1.bar(classes, logits, color=PALETTE["amber"], width=0.5, edgecolor="white")
ax1.set_title("Raw Logits (z)")
ax1.set_ylabel("Score")
for bar, val in zip(bars1, logits):
    ax1.text(bar.get_x() + bar.get_width()/2, val + 0.05, f"{val:.1f}", ha="center", fontweight="bold", fontsize=9)

bars2 = ax2.bar(classes, softmax_probs, color=PALETTE["green"], width=0.5, edgecolor="white")
ax2.set_title("Softmax Probabilities (exp(z) / ∑exp(z))")
ax2.set_ylabel("Probability")
ax2.set_ylim(0, 1.0)
for bar, val in zip(bars2, softmax_probs):
    ax2.text(bar.get_x() + bar.get_width()/2, val + 0.03, f"{val:.3f} ({val*100:.1f}%)", ha="center", fontweight="bold", fontsize=9)

out = os.path.join(here, "fig_softmax.png")
fig.tight_layout()
fig.savefig(out, dpi=220, bbox_inches="tight")
plt.close(fig)
print("Saved fig_softmax.png")
