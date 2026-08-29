#!/usr/bin/env python3
import os, sys, matplotlib.pyplot as plt, numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../scripts")))
from figstyle import use_house_style, PALETTE
use_house_style()
x = np.linspace(-4, 4, 500)
gaussian_pdf = (1.0 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * x**2)
fig, ax = plt.subplots(figsize=(6.5, 3.5))
ax.plot(x, gaussian_pdf, color=PALETTE["blue"], lw=2.5, label="Gaussian N(mu=0, sigma=1)")
ax.set_title("Gaussian Probability Density Function for MLE")
ax.set_xlabel("Observation x"); ax.set_ylabel("Probability Density p(x)")
ax.grid(True, linestyle="--", alpha=0.5); ax.legend(loc="upper right", fontsize=8)
plt.tight_layout(); plt.savefig(os.path.join(os.path.dirname(__file__), "fig_gaussian_mle.png"), dpi=300)
