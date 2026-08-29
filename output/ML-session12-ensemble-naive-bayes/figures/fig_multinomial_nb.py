#!/usr/bin/env python3
import os, sys
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../scripts")))
from figstyle import use_house_style, PALETTE

use_house_style()

words = ['close', 'game', 'sports', 'clean']
p_sports = [2/8, 2/8, 2/8, 1/8]
p_notsports = [1/6, 1/6, 1/6, 2/6]

x = np.arange(len(words))
width = 0.35

fig, ax = plt.subplots(figsize=(6.5, 3.5))
ax.bar(x - width/2, p_sports, width, label='Sports Class', color=PALETTE["blue"])
ax.bar(x + width/2, p_notsports, width, label='Not Sports Class', color=PALETTE["red"])

ax.set_xticks(x)
ax.set_xticklabels(words)
ax.set_ylabel("Smoothed Word Probability P(w|c)")
ax.set_title("Multinomial Naïve Bayes Word Probabilities")
ax.grid(True, linestyle="--", alpha=0.5)
ax.legend(loc="upper right", fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), "fig_multinomial_nb.png"), dpi=300)
