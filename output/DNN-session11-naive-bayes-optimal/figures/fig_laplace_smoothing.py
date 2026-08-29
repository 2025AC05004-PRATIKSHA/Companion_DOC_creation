#!/usr/bin/env python3
import os, sys, matplotlib.pyplot as plt, numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../scripts")))
from figstyle import use_house_style, PALETTE
use_house_style()
counts = np.arange(0, 10)
fig, ax = plt.subplots(figsize=(6.5, 3.5))
ax.plot(counts, counts/10.0, color=PALETTE["red"], marker="o", label="Raw Likelihood")
ax.plot(counts, (counts+1)/12.0, color=PALETTE["blue"], marker="s", label="Laplace Smoothed Probability")
ax.set_title("Effect of Laplace Smoothing on Small Sample Counts")
ax.set_xlabel("Observed Count n_c"); ax.set_ylabel("Estimated Probability P(x_i | y)")
ax.grid(True, linestyle="--", alpha=0.5); ax.legend(loc="upper left", fontsize=8)
plt.tight_layout(); plt.savefig(os.path.join(os.path.dirname(__file__), "fig_laplace_smoothing.png"), dpi=300)
