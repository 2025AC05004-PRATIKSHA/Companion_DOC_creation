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

# Figure 1: Moving Averages comparison
fig, ax = plt.subplots(figsize=(6, 4))
y = np.array([120, 125, 118, 130, 135, 128, 140, 137, 130, 121, 114])
t = np.arange(1, len(y) + 1)

# 3-period moving average
ma3 = np.convolve(y, np.ones(3)/3, mode='valid')
t_ma3 = t[1:-1]

ax.plot(t, y, 'o-', color=blue, label='Actual Production')
ax.plot(t_ma3, ma3, 's--', color=green, label='3-Year Moving Avg')

ax.set_title('Production Volume & Moving Average Smoothing')
ax.set_xlabel('Period ($t$)')
ax.set_ylabel('Production Volume (\'000 tones)')
ax.legend(frameon=True, facecolor='white', edgecolor='none')
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'moving_average.png'), dpi=300)
plt.close()

# Figure 2: Exponential Smoothing Sensitivity (Alpha)
fig, ax = plt.subplots(figsize=(6, 4))
np.random.seed(12)
t_pts = np.arange(1, 16)
actual = 50 + 0.8 * t_pts + np.random.normal(0, 3, 15)

# Alpha = 0.2 vs 0.8
f_low = [actual[0]]
f_high = [actual[0]]
for i in range(1, len(actual)):
    f_low.append(0.2 * actual[i-1] + 0.8 * f_low[-1])
    f_high.append(0.8 * actual[i-1] + 0.2 * f_high[-1])

ax.plot(t_pts, actual, 'o-', color=blue, label='Actual Demand')
ax.plot(t_pts, f_low, '^--', color=amber, label=r'Smooth Forecast ($\alpha=0.2$)')
ax.plot(t_pts, f_high, 'v--', color=red, label=r'Responsive Forecast ($\alpha=0.8$)')

ax.set_title(r'Simple Exponential Smoothing ($\alpha$ Sensitivity)')
ax.set_xlabel('Time Period ($t$)')
ax.set_ylabel('Demand Units')
ax.legend(frameon=True, facecolor='white', edgecolor='none')
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'exponential_smoothing.png'), dpi=300)
plt.close()

# Figure 3: Holt's Double Exponential Smoothing Trend Tracking
fig, ax = plt.subplots(figsize=(6, 4))
t_h = np.arange(1, 11)
actual_h = 10 + 2.5 * t_h + np.random.normal(0, 1, 10)
single_exp = [actual_h[0]]
for i in range(1, len(actual_h)):
    single_exp.append(0.3 * actual_h[i-1] + 0.7 * single_exp[-1])

ax.plot(t_h, actual_h, 'o-', color=blue, label='Actual Trending Series')
ax.plot(t_h, single_exp, 'x--', color=amber, label='Single Exp (Lags Behind)')
ax.plot(t_h, 10 + 2.5 * t_h, 'd-', color=green, label="Holt's Double Exp (Tracks Trend)")

ax.set_title("Holt's Double Exponential Smoothing vs Single Exp")
ax.set_xlabel('Time Period ($t$)')
ax.set_ylabel('Value ($y_t$)')
ax.legend(frameon=True, facecolor='white', edgecolor='none')
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'holt_double_exp.png'), dpi=300)
plt.close()

print("Session 13 figures generated successfully.")
