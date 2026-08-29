#!/usr/bin/env python3
import os, sys, matplotlib.pyplot as plt, numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../scripts")))
from figstyle import use_house_style, PALETTE
use_house_style()
np.random.seed(42)
x1_c0, y1_c0 = np.random.normal(2, 0.8, 15), np.random.normal(2, 0.8, 15)
x1_c1, y1_c1 = np.random.normal(5, 0.8, 15), np.random.normal(5, 0.8, 15)
query_x, query_y = 3.5, 3.5
fig, ax = plt.subplots(figsize=(6.5, 3.5))
ax.scatter(x1_c0, y1_c0, color=PALETTE["blue"], s=60, label="Class A")
ax.scatter(x1_c1, y1_c1, color=PALETTE["green"], s=60, label="Class B")
ax.scatter([query_x], [query_y], color=PALETTE["red"], s=120, marker="*", label="Query x_q", zorder=5)
circle1 = plt.Circle((query_x, query_y), 1.4, color=PALETTE["red"], fill=False, linestyle="--", lw=1.5, label="k=3")
ax.add_patch(circle1)
ax.set_title("k-Nearest Neighbors Classification for Query x_q")
ax.set_xlabel("x1"); ax.set_ylabel("x2"); ax.grid(True, linestyle="--", alpha=0.5); ax.legend(loc="upper left", fontsize=8)
plt.tight_layout(); plt.savefig(os.path.join(os.path.dirname(__file__), "fig_knn_classification.png"), dpi=300)
