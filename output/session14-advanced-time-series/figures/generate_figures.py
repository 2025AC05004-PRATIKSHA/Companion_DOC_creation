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

# Figure 1: ACF vs PACF decay patterns
fig, axes = plt.subplots(1, 2, figsize=(7, 3.5))

lags = np.arange(1, 11)
acf_ar = 0.7**lags
pacf_ar = np.array([0.7, 0, 0, 0, 0, 0, 0, 0, 0, 0])

axes[0].bar(lags, acf_ar, color=blue, width=0.4)
axes[0].axhline(0, color=muted, linewidth=1)
axes[0].set_title('AR(1) ACF (Exponential Decay)')
axes[0].set_xlabel('Lag ($k$)')
axes[0].set_ylabel('Autocorrelation')

axes[1].bar(lags, pacf_ar, color=red, width=0.4)
axes[1].axhline(0, color=muted, linewidth=1)
axes[1].set_title('AR(1) PACF (Cuts Off at Lag $p=1$)')
axes[1].set_xlabel('Lag ($k$)')

for ax in axes:
    ax.set_ylim(-0.2, 1.0)
    ax.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'acf_pacf_patterns.png'), dpi=300)
plt.close()

# Figure 2: ARIMA vs SARIMAX forecast projection
fig, ax = plt.subplots(figsize=(6, 4))
t_hist = np.arange(90, 101)
y_hist = 100 + 0.6 * (t_hist - 90) + np.random.normal(0, 2, 11)

t_fore = np.array([101, 102])
y_arima = np.array([122.3, 123.22])
y_sarimax = np.array([129.5, 124.0])

ax.plot(t_hist, y_hist, 'o-', color=blue, label='Historical Series')
ax.plot(t_fore, y_arima, 's--', color=amber, label='ARIMA Forecast')
ax.plot(t_fore, y_sarimax, 'd--', color=green, label='SARIMAX (With Promo Shift)')

ax.set_title('Multi-Step ARIMA vs SARIMAX Forecasts')
ax.set_xlabel('Time Period ($t$)')
ax.set_ylabel('Target Series ($y_t$)')
ax.legend(frameon=True, facecolor='white', edgecolor='none')
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'arima_sarimax_forecast.png'), dpi=300)
plt.close()

# Figure 3: VAR System Interdependence
fig, ax = plt.subplots(figsize=(6, 4))
t_var = np.arange(95, 101)
gdp = np.array([1.8, 1.9, 2.1, 2.0, 1.9, 2.0])
infl = np.array([2.5, 2.7, 2.9, 3.1, 3.0, 3.0])

t_vfore = np.array([101])
gdp_f = np.array([1.6])
infl_f = np.array([2.4])

ax.plot(t_var, gdp, 'o-', color=blue, label='GDP Growth ($g_t$)')
ax.plot(t_var, infl, 's-', color=red, label='Inflation ($\pi_t$)')

ax.plot(t_vfore, gdp_f, 'o', color=blue, markersize=8, label='VAR(1) $g_{101}=1.6\%$')
ax.plot(t_vfore, infl_f, 's', color=red, markersize=8, label='VAR(1) $\pi_{101}=2.4\%$')

ax.set_title('Multivariate VAR(1) Joint System Forecast')
ax.set_xlabel('Quarter ($t$)')
ax.set_ylabel('Percentage Rate (%)')
ax.legend(frameon=True, facecolor='white', edgecolor='none')
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'var_system_forecast.png'), dpi=300)
plt.close()

print("Session 14 figures generated successfully.")
