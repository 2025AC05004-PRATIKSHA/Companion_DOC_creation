import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../scripts")))
import numpy as np
from figstyle import use_house_style, pmf_bar, shaded_normal, curve, PALETTE

use_house_style()
here = os.path.dirname(os.path.abspath(__file__))

# 1. Bernoulli PMF (p = 0.6)
pmf_bar([0, 1], [0.4, 0.6], "Bernoulli Distribution (p = 0.6)",
        xlabel="Outcome (0 = Failure, 1 = Success)", ylabel="Probability",
        out=os.path.join(here, "fig_bernoulli.png"))

# 2. Binomial PMF (n = 10, p = 0.5)
from scipy.stats import binom
xs = list(range(11))
ps = [float(binom.pmf(k, 10, 0.5)) for k in xs]
pmf_bar(xs, ps, "Binomial Distribution (n = 10, p = 0.5)",
        xlabel="Number of Successes (k)", ylabel="Probability",
        out=os.path.join(here, "fig_binomial.png"))

# 3. Poisson PMF (lambda = 4)
from scipy.stats import poisson
xs_p = list(range(12))
ps_p = [float(poisson.pmf(k, 4)) for k in xs_p]
pmf_bar(xs_p, ps_p, "Poisson Distribution (\u03bb = 4)",
        xlabel="Number of Events (k)", ylabel="Probability",
        out=os.path.join(here, "fig_poisson.png"))

# 4. Normal Curve 68-95-99.7 Rule
shaded_normal(0, 1, -1, 1, "Standard Normal Distribution (\u03bc = 0, \u03c3 = 1)",
              xlabel="Z-score", ylabel="Probability Density",
              out=os.path.join(here, "fig_normal.png"))

print("Session 7 figures generated successfully.")
