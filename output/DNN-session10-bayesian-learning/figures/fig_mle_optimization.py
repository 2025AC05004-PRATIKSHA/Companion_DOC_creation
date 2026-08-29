#!/usr/bin/env python3
import os, sys, matplotlib.pyplot as plt, numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../scripts")))
from figstyle import use_house_style, PALETTE
use_house_style()
theta = np.linspace(0.05, 0.95, 300)
log_L = 7 * np.log(theta) + 3 * np.log(1 - theta)
fig, ax = plt.subplots(figsize=(6.5, 3.5))
ax.plot(theta, log_L, color=PALETTE["purple"], lw=2.5, label="Log-Likelihood ln L(theta)")
ax.axvline(0.7, color=PALETTE["red"], linestyle="--", label="MLE theta_hat = 0.7")
ax.set_title("Maximizing Log-Likelihood Function for Coin Flip Data")
ax.set_xlabel("Parameter theta"); ax.set_ylabel("Log-Likelihood Value")
ax.grid(True, linestyle="--", alpha=0.5); ax.legend(loc="lower center", fontsize=8)
plt.tight_layout(); plt.savefig(os.path.join(os.path.dirname(__file__), "fig_mle_optimization.png"), dpi=300)
