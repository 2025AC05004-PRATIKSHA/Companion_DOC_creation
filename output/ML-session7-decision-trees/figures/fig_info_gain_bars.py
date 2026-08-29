#!/usr/bin/env python3
import os, sys, matplotlib.pyplot as plt
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../scripts")))
from figstyle import use_house_style, PALETTE
use_house_style()
attrs = ['Outlook', 'Humidity', 'Wind', 'Temperature']
gains = [0.246, 0.151, 0.048, 0.029]
fig, ax = plt.subplots(figsize=(6.5, 3.2))
bars = ax.bar(attrs, gains, color=[PALETTE["blue"], PALETTE["green"], PALETTE["amber"], PALETTE["purple"]])
ax.set_ylabel("Information Gain (Bits)")
ax.set_title("Information Gain Across Candidate Attributes (Root Node)")
ax.set_ylim(0, 0.3)
plt.tight_layout(); plt.savefig(os.path.join(os.path.dirname(__file__), "fig_info_gain_bars.png"), dpi=300)
