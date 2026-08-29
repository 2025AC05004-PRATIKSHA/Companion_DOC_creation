#!/usr/bin/env python3
import os, sys, matplotlib.pyplot as plt, numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../scripts")))
from figstyle import use_house_style, PALETTE
use_house_style()
theta = np.linspace(0.01, 0.99, 500)
prior = 6 * theta * (1 - theta)
posterior = (theta**8 * (1 - theta)**4) / 0.00015
fig, ax = plt.subplots(figsize=(6.5, 3.5))
ax.plot(theta, prior, color=PALETTE["blue"], lw=2.0, label="Prior P(theta)")
ax.plot(theta, posterior, color=PALETTE["green"], lw=2.5, label="Posterior P(theta|D)")
ax.set_title("Bayesian Updating: Prior x Likelihood -> Posterior")
ax.set_xlabel("Parameter theta"); ax.set_ylabel("Probability Density")
ax.grid(True, linestyle="--", alpha=0.5); ax.legend(loc="upper left", fontsize=8)
plt.tight_layout(); plt.savefig(os.path.join(os.path.dirname(__file__), "fig_bayes_updating.png"), dpi=300)
