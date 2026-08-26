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
muted = figstyle.PALETTE['muted']

# Figure 1: Non-linear regression fits
fig, ax = plt.subplots(figsize=(6, 4))
x = np.linspace(1, 10, 50)
y_poly = 2 + 0.5 * x**2 + np.random.normal(0, 2, 50)
y_log = 5 + 8 * np.log(x) + np.random.normal(0, 1, 50)

ax.scatter(x, y_poly, color=blue, alpha=0.6, label='Quadratic Data')
ax.plot(x, 2 + 0.5 * x**2, color=blue, linewidth=2, label=r'Polynomial Fit ($Y = a + bX + cX^2$)')

ax.scatter(x, y_log, color=red, alpha=0.6, label='Logarithmic Data')
ax.plot(x, 5 + 8 * np.log(x), color=red, linewidth=2, label=r'Logarithmic Fit ($Y = a + b\ln X$)')

ax.set_title('Non-Linear Regression Relationships')
ax.set_xlabel('Predictor ($X$)')
ax.set_ylabel('Response ($Y$)')
ax.legend(frameon=True, facecolor='white', edgecolor='none')
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'nonlinear_regression.png'), dpi=300)
plt.close()

# Figure 2: Time Series Components
fig, axes = plt.subplots(4, 1, figsize=(7, 6), sharex=True)

t = np.linspace(1, 24, 100)
trend = 0.5 * t + 10
seasonal = 3 * np.sin(2 * np.pi * t / 4)
cyclical = 4 * np.cos(2 * np.pi * t / 12)
noise = np.random.normal(0, 1, 100)

axes[0].plot(t, trend, color=blue, linewidth=2)
axes[0].set_ylabel('Secular Trend')
axes[0].set_title('Decomposition of Time Series Components')

axes[1].plot(t, seasonal, color=green, linewidth=2)
axes[1].set_ylabel('Seasonal')

axes[2].plot(t, cyclical, color=amber, linewidth=2)
axes[2].set_ylabel('Cyclical')

axes[3].plot(t, noise, color=red, linewidth=2)
axes[3].set_ylabel('Irregular')
axes[3].set_xlabel('Time (Quarters)')

for ax in axes:
    ax.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'time_series_components.png'), dpi=300)
plt.close()

# Figure 3: Stationarity Split-Half Test
fig, ax = plt.subplots(figsize=(6, 4))
t_pts = np.arange(1, 11)
y_vals = np.array([4, 6, 5, 7, 9, 8, 10, 12, 11, 13])

ax.plot(t_pts[:5], y_vals[:5], 'o-', color=blue, label=r'First Half ($t=1..5, \bar{y}_1=6.20$)')
ax.plot(t_pts[4:], y_vals[4:], 'o-', color=red, label=r'Second Half ($t=6..10, \bar{y}_2=10.80$)')
ax.axhline(6.20, color=blue, linestyle='--', alpha=0.7)
ax.axhline(10.80, color=red, linestyle='--', alpha=0.7)

ax.set_title('Stationarity Check: Split-Half Mean Comparison')
ax.set_xlabel('Time Period ($t$)')
ax.set_ylabel('Observed Series ($y_t$)')
ax.legend(frameon=True, facecolor='white', edgecolor='none')
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'stationarity_split.png'), dpi=300)
plt.close()

print("Session 12 figures generated successfully.")
