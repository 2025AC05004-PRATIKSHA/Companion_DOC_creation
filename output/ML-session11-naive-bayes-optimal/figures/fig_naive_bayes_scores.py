#!/usr/bin/env python3
import os, sys, matplotlib.pyplot as plt
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../scripts")))
from figstyle import use_house_style, PALETTE
use_house_style()
fig, ax = plt.subplots(figsize=(6.5, 3.2))
ax.bar(['Enjoy = Yes', 'Enjoy = No'], [0.0053, 0.0206], color=[PALETTE["green"], PALETTE["red"]])
ax.set_ylabel("Posterior Score P(X|Y) * P(Y)")
ax.set_title("Naïve Bayes Prediction Scores for Test Instance")
plt.tight_layout(); plt.savefig(os.path.join(os.path.dirname(__file__), "fig_naive_bayes_scores.png"), dpi=300)
