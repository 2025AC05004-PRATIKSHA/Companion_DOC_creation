#!/usr/bin/env python3
import os, sys, matplotlib.pyplot as plt, numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../scripts")))
from figstyle import use_house_style, PALETTE
use_house_style()
p = np.linspace(0.0001, 0.9999, 500)
entropy = -p * np.log2(p) - (1 - p) * np.log2(1 - p)
gini = 2 * p * (1 - p)
fig, ax = plt.subplots(figsize=(6.5, 3.5))
ax.plot(p, entropy/2, color=PALETTE["blue"], lw=2.0, label="Entropy (Scaled H/2)")
ax.plot(p, gini, color=PALETTE["green"], lw=2.0, label="Gini Index")
ax.set_title("Comparison of Node Impurity Measures")
ax.set_xlabel("Class Probability p+"); ax.set_ylabel("Impurity Score")
ax.grid(True, linestyle="--", alpha=0.5); ax.legend(loc="upper right", fontsize=9)
plt.tight_layout(); plt.savefig(os.path.join(os.path.dirname(__file__), "fig_impurity_comparison.png"), dpi=300)
