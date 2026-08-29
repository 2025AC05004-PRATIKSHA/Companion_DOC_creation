#!/usr/bin/env python3
import os, sys, matplotlib.pyplot as plt
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../scripts")))
from figstyle import use_house_style, PALETTE
use_house_style()
temperatures = [64, 65, 68, 69, 70, 71, 72, 75, 80, 83, 85]
classes = [1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0]
fig, ax = plt.subplots(figsize=(7.0, 2.5))
for temp, cls in zip(temperatures, classes):
    color = PALETTE["green"] if cls == 1 else PALETTE["red"]
    ax.scatter(temp, 1, color=color, s=80, zorder=3)
ax.axvline(71.5, color=PALETTE["blue"], linestyle="--", lw=2, label="Best Split Threshold (T=71.5)")
ax.set_yticks([]); ax.set_xlabel("Temperature °F"); ax.set_title("Continuous Attribute Threshold Candidate Splitting")
ax.set_xlim(60, 90); ax.legend(loc="upper right", fontsize=9)
plt.tight_layout(); plt.savefig(os.path.join(os.path.dirname(__file__), "fig_continuous_split.png"), dpi=300)
