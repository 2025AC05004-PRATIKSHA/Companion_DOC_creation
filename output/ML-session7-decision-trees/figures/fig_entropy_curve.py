#!/usr/bin/env python3
import os, sys, matplotlib.pyplot as plt, numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../scripts")))
from figstyle import use_house_style, PALETTE
use_house_style()
p = np.linspace(0.0001, 0.9999, 500)
entropy = -p * np.log2(p) - (1 - p) * np.log2(1 - p)
fig, ax = plt.subplots(figsize=(6.5, 3.5))
ax.plot(p, entropy, color=PALETTE["blue"], lw=2.5, label="Entropy H(S)")
ax.axvline(0.5, color=PALETTE["red"], linestyle="--", lw=1.2, label="Max Impurity (p=0.5, H=1.0)")
ax.set_title("Binary Entropy Curve vs Class Probability")
ax.set_xlabel("Proportion of Positive Examples (p+)")
ax.set_ylabel("Entropy H(S) in Bits")
ax.grid(True, linestyle="--", alpha=0.5)
ax.legend(loc="lower center", fontsize=9)
plt.tight_layout(); plt.savefig(os.path.join(os.path.dirname(__file__), "fig_entropy_curve.png"), dpi=300)
