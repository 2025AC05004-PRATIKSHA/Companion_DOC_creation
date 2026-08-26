import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../scripts")))
import numpy as np
from figstyle import use_house_style, curve, shaded_normal, bars, PALETTE

use_house_style()
here = os.path.dirname(os.path.abspath(__file__))

# 1. CLT Convergence of Sample Means
x = np.linspace(-3, 3, 200)
y1 = np.exp(-0.5 * (x / 1.5)**2) / (1.5 * np.sqrt(2 * np.pi))
y2 = np.exp(-0.5 * (x / 0.8)**2) / (0.8 * np.sqrt(2 * np.pi))
y3 = np.exp(-0.5 * (x / 0.4)**2) / (0.4 * np.sqrt(2 * np.pi))

curve(x, {"n = 2": y1, "n = 10": y2, "n = 30": y3},
      "Central Limit Theorem: Distribution of Sample Means (\u03bc=0)",
      xlabel="Sample Mean (\u03bar{X})", ylabel="Density",
      out=os.path.join(here, "fig_clt.png"))

# 2. Standard Error vs Sample Size
ns = np.array([5, 10, 25, 50, 100, 200, 500])
sigma = 10.0
se = sigma / np.sqrt(ns)
curve(ns, {"Standard Error (\u03c3/\u221an)": se},
      "Standard Error Shrinks as Sample Size Increases",
      xlabel="Sample Size (n)", ylabel="Standard Error (SE)",
      fill_below="Standard Error (\u03c3/\u221an)",
      out=os.path.join(here, "fig_se_decay.png"))

# 3. Confidence Interval Alpha Margins
shaded_normal(0, 1, -1.96, 1.96, "95% Confidence Interval (\u03b1 = 0.05, Z_{\u03b1/2} = 1.96)",
              xlabel="Z-score", ylabel="Density",
              out=os.path.join(here, "fig_ci_95.png"))

print("Session 8 figures generated successfully.")
