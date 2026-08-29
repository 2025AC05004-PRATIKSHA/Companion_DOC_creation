#!/usr/bin/env python3
import os, sys, matplotlib.pyplot as plt, numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../scripts")))
from figstyle import use_house_style, PALETTE
use_house_style()
theta = np.linspace(0, 2*np.pi, 500)
fig, ax = plt.subplots(figsize=(6.5, 3.5))
ax.plot(np.cos(theta), np.sin(theta), color=PALETTE["blue"], lw=2.0, label="L2 Euclidean (p=2)")
ax.plot([1, 0, -1, 0, 1], [0, 1, 0, -1, 0], color=PALETTE["green"], lw=2.0, label="L1 Manhattan (p=1)")
ax.set_aspect('equal'); ax.set_title("Unit Distance Contours for Minkowski Distance Norms")
ax.grid(True, linestyle="--", alpha=0.5); ax.legend(loc="upper right", fontsize=8)
plt.tight_layout(); plt.savefig(os.path.join(os.path.dirname(__file__), "fig_minkowski_norms.png"), dpi=300)
