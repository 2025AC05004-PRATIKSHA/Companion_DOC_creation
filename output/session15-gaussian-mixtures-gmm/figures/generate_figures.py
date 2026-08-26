import sys
import os
fig_dir = os.path.dirname(__file__)
sys.path.insert(0, os.path.abspath(os.path.join(fig_dir, '../../../scripts')))

import matplotlib.pyplot as plt
import numpy as np
import figstyle

figstyle.use_house_style()
blue = figstyle.PALETTE['blue']
green = figstyle.PALETTE['green']
red = figstyle.PALETTE['red']
amber = figstyle.PALETTE['amber']

# Figure 1: Gaussian Mixture Distribution (Bimodal Step Counts)
fig, ax = plt.subplots(figsize=(6, 4))
x = np.linspace(0, 15, 200)

# Component 1 (sedentary): mean = 4.1, sd = 0.65
comp1 = 0.5 * (1 / (0.65 * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - 4.1) / 0.65)**2)
# Component 2 (active): mean = 10.4, sd = 0.83
comp2 = 0.5 * (1 / (0.83 * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - 10.4) / 0.83)**2)
mixture = comp1 + comp2

ax.plot(x, mixture, color=blue, linewidth=2.5, label='GMM Density Mixture')
ax.plot(x, comp1, '--', color=amber, label=r'Comp 1: Sedentary ($\mu_1=4.1, \pi_1=0.5$)')
ax.plot(x, comp2, '--', color=green, label=r'Comp 2: Active ($\mu_2=10.4, \pi_2=0.5$)')

# Rug plot for 10 student step counts
steps = np.array([3.2, 3.8, 4.1, 4.5, 4.9, 9.2, 9.8, 10.3, 10.7, 11.4])
ax.plot(steps, np.zeros_like(steps), '|', color=red, markersize=12, markeredgewidth=2, label='Student Step Observations')

ax.set_title('Bimodal Student Step Count GMM Density')
ax.set_xlabel('Daily Step Count (\'000 steps)')
ax.set_ylabel('Probability Density')
ax.legend(frameon=True, facecolor='white', edgecolor='none')
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'gmm_mixture_density.png'), dpi=300)
plt.close()

# Figure 2: Soft Responsibilities
fig, ax = plt.subplots(figsize=(6, 4))
resp1 = comp1 / mixture
resp2 = comp2 / mixture

ax.plot(x, resp1, color=amber, linewidth=2, label=r'Responsibility $\gamma_1(x)$ (Comp 1)')
ax.plot(x, resp2, color=green, linewidth=2, label=r'Responsibility $\gamma_2(x)$ (Comp 2)')
ax.axhline(0.5, color=red, linestyle='--', alpha=0.6, label='Decision Boundary ($\gamma=0.5$)')

ax.set_title('EM Algorithm Posterior Responsibilities')
ax.set_xlabel('Daily Step Count ($x$)')
ax.set_ylabel('Cluster Membership Probability')
ax.legend(frameon=True, facecolor='white', edgecolor='none')
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'em_responsibilities.png'), dpi=300)
plt.close()

print("Session 15 figures generated successfully.")
