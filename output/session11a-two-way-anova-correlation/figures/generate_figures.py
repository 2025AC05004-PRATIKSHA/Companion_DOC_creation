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

# Figure 1: Correlation comparison
fig, ax = plt.subplots(figsize=(6, 4))
np.random.seed(42)
x = np.linspace(10, 50, 20)
y1 = 1.2 * x + 10 + np.random.normal(0, 3, 20)
y2 = -0.9 * x + 80 + np.random.normal(0, 2, 20)

ax.scatter(x, y1, color=blue, label=r'Direct Positive ($r = +0.76$)', s=40)
ax.scatter(x, y2, color=red, label=r'Inverse Negative ($r = -0.94$)', s=40)
ax.set_title(r'Pearson Correlation ($r$) Comparison')
ax.set_xlabel('Predictor Variable ($X$)')
ax.set_ylabel('Response Variable ($Y$)')
ax.legend(frameon=True, facecolor='white', edgecolor='none')
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'correlation_scatter.png'), dpi=300)
plt.close()

# Figure 2: Simple Linear Regression fit
fig, ax = plt.subplots(figsize=(6, 4))
x_adv = np.array([2.0, 5.0, 4.0, 2.5, 3.0, 4.0, 2.5, 3.0])
y_rev = np.array([9.0, 16.0, 15.0, 11.0, 13.0, 14.0, 10.0, 12.0])

ax.scatter(x_adv, y_rev, color=blue, s=50, zorder=3, label='Observed Revenue')
x_line = np.linspace(1.5, 5.5, 100)
y_line = 4.88 + 2.36 * x_line
ax.plot(x_line, y_line, color=green, linewidth=2, label=r'Fitted Line: $\hat{Y} = 4.88 + 2.36X$')

ax.scatter([4.5], [15.5], color=red, s=70, zorder=4, label=r'Forecast at $X=4.5$ ($15.5k$)')
ax.set_title('Least Squares Regression Line')
ax.set_xlabel(r'Advertising Expenditure ($X$ in $\$1,000$s)')
ax.set_ylabel(r'Weekly Gross Revenue ($Y$ in $\$1,000$s)')
ax.legend(frameon=True, facecolor='white', edgecolor='none')
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'regression_line.png'), dpi=300)
plt.close()

# Figure 3: ANOVA Variance Partitioning
fig, ax = plt.subplots(figsize=(6, 3.5))
categories = ['Total\n(SST)', 'Row Factor\n(SSA)', 'Column Factor\n(SSB)', 'Error\n(SSE)']
values = [100, 45, 40, 15]
colors = [blue, green, amber, red]

bars = ax.bar(categories, values, color=colors, width=0.5)
ax.set_title('Variance Partitioning in Two-Way ANOVA')
ax.set_ylabel('Percentage of Variance (%)')
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2.0, yval + 1, f'{yval}%', ha='center', va='bottom', fontsize=9, fontweight='bold', color=muted)

ax.set_ylim(0, 115)
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'anova_partition.png'), dpi=300)
plt.close()

print("Session 11A figures generated successfully.")
