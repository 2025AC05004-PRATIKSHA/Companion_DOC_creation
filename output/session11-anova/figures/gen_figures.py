import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../scripts")))
import numpy as np
from figstyle import use_house_style, curve, bars, PALETTE

use_house_style()
here = os.path.dirname(os.path.abspath(__file__))

# 1. Fisher F-Distribution Rejection Region
from scipy.stats import f
x_f = np.linspace(0.01, 6, 250)
y_f = f.pdf(x_f, dfn=2, dfd=12)

curve(x_f, {"F(2, 12) Density": y_f},
      "One-Way ANOVA F-Distribution Rejection Region (\u03b1 = 0.05, Critical F = 3.89)",
      xlabel="F Statistic", ylabel="Density",
      fill_below="F(2, 12) Density",
      out=os.path.join(here, "fig_f_distribution.png"))

# 2. Between vs Within Group Variance Breakdown
bars(["SSTR (Treatment)", "SSE (Error)", "SST (Total)"],
     [24.0, 72.0, 96.0],
     "ANOVA Sum of Squares Breakdown (SST = SSTR + SSE)",
     xlabel="Variance Source", ylabel="Sum of Squares",
     colors=[PALETTE["blue"], PALETTE["amber"], PALETTE["purple"]],
     out=os.path.join(here, "fig_anova_ss.png"))

print("Session 11 figures generated successfully.")
