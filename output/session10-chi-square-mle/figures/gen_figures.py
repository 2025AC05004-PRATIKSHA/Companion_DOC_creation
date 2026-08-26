import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../scripts")))
import numpy as np
from figstyle import use_house_style, curve, function_plot, PALETTE

use_house_style()
here = os.path.dirname(os.path.abspath(__file__))

# 1. Student's t-Distribution vs Standard Normal
from scipy.stats import t, norm
x = np.linspace(-3.5, 3.5, 250)
y_norm = norm.pdf(x)
y_t2 = t.pdf(x, df=2)
y_t10 = t.pdf(x, df=10)

curve(x, {"Normal (df=\u221e)": y_norm, "t-dist (df=10)": y_t10, "t-dist (df=2)": y_t2},
      "Student's t-Distribution Has Heavy Tails for Small Degrees of Freedom",
      xlabel="t or Z score", ylabel="Density",
      out=os.path.join(here, "fig_t_distribution.png"))

# 2. Chi-Square Distribution Shapes
from scipy.stats import chi2
x_c = np.linspace(0.01, 15, 250)
y_c2 = chi2.pdf(x_c, df=2)
y_c4 = chi2.pdf(x_c, df=4)
y_c8 = chi2.pdf(x_c, df=8)

curve(x_c, {"df = 2": y_c2, "df = 4": y_c4, "df = 8": y_c8},
      "Chi-Square Distribution Curves across Degrees of Freedom",
      xlabel="\u03c7\u00b2 Statistic", ylabel="Density",
      out=os.path.join(here, "fig_chi_square.png"))

# 3. Binomial Likelihood Function L(p)
p_grid = np.linspace(0, 1, 200)
# Example: 3 successes out of 10 trials
L_p = (p_grid**3) * ((1 - p_grid)**7)
function_plot(lambda p: (p**3) * ((1 - p)**7), (0, 1),
              "Binomial Likelihood Function L(p) for 3 Successes in 10 Trials",
              xlabel="Parameter p", ylabel="Likelihood L(p)",
              tangent_at=0.30, marks=[(0.30, "MLE \u0151 = 0.30")],
              out=os.path.join(here, "fig_mle_likelihood.png"))

print("Session 10 figures generated successfully.")
