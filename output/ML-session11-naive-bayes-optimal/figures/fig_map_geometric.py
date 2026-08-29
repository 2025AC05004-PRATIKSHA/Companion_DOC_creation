#!/usr/bin/env python3
import os, sys, matplotlib.pyplot as plt, numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../scripts")))
from figstyle import use_house_style, PALETTE
use_house_style()
x = np.linspace(0, 1, 300)
posterior_unnorm = 2 * x**2 * (1 - x)**2
fig, ax = plt.subplots(figsize=(6.5, 3.5))
ax.plot(x, posterior_unnorm, color=PALETTE["blue"], lw=2.5, label="MAP Objective g(x)")
ax.axvline(0.5, color=PALETTE["red"], linestyle="--", label="x_MAP = 0.5")
ax.set_title("MAP Parameter Estimation for Geometric Distribution (Y=3)")
ax.set_xlabel("Parameter x"); ax.set_ylabel("g(x) Value")
ax.grid(True, linestyle="--", alpha=0.5); ax.legend(loc="upper right", fontsize=8)
plt.tight_layout(); plt.savefig(os.path.join(os.path.dirname(__file__), "fig_map_geometric.png"), dpi=300)
