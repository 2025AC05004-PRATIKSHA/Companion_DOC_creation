#!/usr/bin/env python3
import os, sys, matplotlib.pyplot as plt, numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../scripts")))
from figstyle import use_house_style, PALETTE
use_house_style()
nodes = np.arange(1, 50)
train_err = 0.45 * np.exp(-nodes/8)
test_err = np.maximum(0.12, 0.45 * np.exp(-nodes/12) + 0.0003 * (nodes - 15)**2)
fig, ax = plt.subplots(figsize=(6.5, 3.5))
ax.plot(nodes, train_err, color=PALETTE["blue"], lw=2.5, label="Training Error")
ax.plot(nodes, test_err, color=PALETTE["red"], lw=2.5, label="Test Error")
ax.axvline(15, color=PALETTE["green"], linestyle="--", lw=1.5, label="Optimal Tree Size")
ax.set_title("Decision Tree Overfitting: Training vs Test Error")
ax.set_xlabel("Number of Tree Nodes / Depth"); ax.set_ylabel("Error Rate")
ax.grid(True, linestyle="--", alpha=0.5); ax.legend(loc="upper right", fontsize=9)
plt.tight_layout(); plt.savefig(os.path.join(os.path.dirname(__file__), "fig_tree_overfitting.png"), dpi=300)
