#!/usr/bin/env python3
import os, sys, matplotlib.pyplot as plt, numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../scripts")))
from figstyle import use_house_style, PALETTE
use_house_style()
sizes = np.array([1200, 1500, 1700, 2000, 2300, 2600, 3000])
prices = np.array([250, 300, 340, 400, 430, 480, 550])
query_size = 1850; query_pred_k3 = (340 + 400 + 430) / 3.0
fig, ax = plt.subplots(figsize=(6.5, 3.5))
ax.scatter(sizes, prices, color=PALETTE["blue"], s=70, label="Training Houses")
ax.scatter([query_size], [query_pred_k3], color=PALETTE["red"], s=100, marker="^", label="Query (1850 sq ft)", zorder=5)
ax.set_title("k-NN Regression for House Price Prediction (k=3)")
ax.set_xlabel("Size (sq ft)"); ax.set_ylabel("Price ($1000s)"); ax.grid(True, linestyle="--", alpha=0.5); ax.legend(loc="upper left", fontsize=8)
plt.tight_layout(); plt.savefig(os.path.join(os.path.dirname(__file__), "fig_knn_regression.png"), dpi=300)
