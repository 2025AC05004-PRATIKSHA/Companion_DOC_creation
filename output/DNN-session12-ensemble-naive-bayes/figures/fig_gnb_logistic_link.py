#!/usr/bin/env python3
import os, sys
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../scripts")))
from figstyle import use_house_style, PALETTE

use_house_style()

z = np.linspace(-6, 6, 300)
sigmoid = 1 / (1 + np.exp(-z))

fig, ax = plt.subplots(figsize=(6.5, 3.5))
ax.plot(z, sigmoid, color=PALETTE["blue"], lw=2.5, label="Logistic Sigmoid sigma(z) = 1/(1+e^-z)")
ax.axvline(0, color=PALETTE["red"], linestyle="--", lw=1.2, label="Decision Boundary (z=0)")
ax.axhline(0.5, color=PALETTE["red"], linestyle=":", lw=1.2)

ax.set_title("Gaussian Naïve Bayes Implied Sigmoid Form P(Y=1|X)")
ax.set_xlabel("Linear Log-Odds Ratio z = theta^T X")
ax.set_ylabel("Probability P(Y=1|X)")
ax.grid(True, linestyle="--", alpha=0.5)
ax.legend(loc="upper left", fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), "fig_gnb_logistic_link.png"), dpi=300)
