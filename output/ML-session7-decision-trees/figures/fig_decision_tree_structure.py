#!/usr/bin/env python3
import os, sys, matplotlib.pyplot as plt
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../scripts")))
from figstyle import use_house_style, PALETTE
use_house_style()
fig, ax = plt.subplots(figsize=(7.0, 3.5))
ax.axis("off")
ax.text(0.5, 0.85, "Outlook?", bbox=dict(boxstyle="round,pad=0.5", fc="#2C5AA0", ec="none"), color="white", weight="bold", ha="center")
ax.annotate("Sunny", xy=(0.2, 0.5), xytext=(0.5, 0.82), arrowprops=dict(arrowstyle="->", lw=1.5, color="#1A202C"))
ax.annotate("Overcast", xy=(0.5, 0.5), xytext=(0.5, 0.82), arrowprops=dict(arrowstyle="->", lw=1.5, color="#1A202C"))
ax.annotate("Rain", xy=(0.8, 0.5), xytext=(0.5, 0.82), arrowprops=dict(arrowstyle="->", lw=1.5, color="#1A202C"))
ax.text(0.2, 0.45, "Humidity?", bbox=dict(boxstyle="round,pad=0.5", fc="#2C5AA0", ec="none"), color="white", weight="bold", ha="center")
ax.text(0.5, 0.45, "Play = Yes", bbox=dict(boxstyle="round,pad=0.5", fc="#2E7D52", ec="none"), color="white", weight="bold", ha="center")
ax.text(0.8, 0.45, "Wind?", bbox=dict(boxstyle="round,pad=0.5", fc="#2C5AA0", ec="none"), color="white", weight="bold", ha="center")
ax.set_title("Decision Tree for PlayTennis Classification")
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
plt.tight_layout(); plt.savefig(os.path.join(os.path.dirname(__file__), "fig_decision_tree_structure.png"), dpi=300)
