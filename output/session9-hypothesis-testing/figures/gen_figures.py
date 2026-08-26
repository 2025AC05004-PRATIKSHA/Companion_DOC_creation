import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../scripts")))
import numpy as np
from figstyle import use_house_style, shaded_normal, bars, PALETTE

use_house_style()
here = os.path.dirname(os.path.abspath(__file__))

# 1. Two-Tailed Rejection Region (\alpha = 0.05)
shaded_normal(0, 1, -1.96, 1.96, "Two-Tailed Hypothesis Test (\u03b1 = 0.05, Critical Z = \u00b11.96)",
              xlabel="Z-score", ylabel="Density",
              out=os.path.join(here, "fig_two_tailed.png"))

# 2. Right-Tailed Rejection Region (\alpha = 0.05)
shaded_normal(0, 1, -4.0, 1.645, "Right-Tailed Hypothesis Test (\u03b1 = 0.05, Critical Z = 1.645)",
              xlabel="Z-score", ylabel="Density",
              out=os.path.join(here, "fig_right_tailed.png"))

# 3. Type I and Type II Errors Comparison
bars(["Type I Error (\u03b1)", "Type II Error (\u03b2)", "Power (1 - \u03b2)"],
     [0.05, 0.15, 0.85],
     "Hypothesis Decision Risks & Test Power",
     xlabel="Metric / Error Type", ylabel="Probability",
     colors=[PALETTE["red"], PALETTE["amber"], PALETTE["green"]],
     out=os.path.join(here, "fig_errors.png"))

print("Session 9 figures generated successfully.")
