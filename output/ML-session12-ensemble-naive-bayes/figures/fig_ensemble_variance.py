#!/usr/bin/env python3
import os, sys
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../scripts")))
from figstyle import use_house_style, PALETTE

use_house_style()

x = np.linspace(-3, 3, 300)

y1 = 1 / (1 + np.exp(-(1.2*x + 0.5)))
y2 = 1 / (1 + np.exp(-(0.8*x - 0.3)))
y3 = 1 / (1 + np.exp(-(1.5*x + 0.1)))
y_ensemble = (y1 + y2 + y3) / 3.0

fig, ax = plt.subplots(figsize=(6.5, 3.5))
ax.plot(x, y1, color=PALETTE["blue"], linestyle="--", alpha=0.6, label="Base Learner 1")
ax.plot(x, y2, color=PALETTE["purple"], linestyle="--", alpha=0.6, label="Base Learner 2")
ax.plot(x, y3, color=PALETTE["amber"], linestyle="--", alpha=0.6, label="Base Learner 3")
ax.plot(x, y_ensemble, color=PALETTE["red"], lw=2.5, label="Ensemble Average (Variance Reduced)")

ax.set_title("Bagging Ensemble Averaging Variance Reduction")
ax.set_xlabel("Feature x")
ax.set_ylabel("Predicted Probability")
ax.grid(True, linestyle="--", alpha=0.5)
ax.legend(loc="upper left", fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), "fig_ensemble_variance.png"), dpi=300)
