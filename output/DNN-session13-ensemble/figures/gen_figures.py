import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Ensure scripts directory is in sys.path for figstyle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../scripts')))
try:
    import figstyle
    figstyle.set_style()
except (ImportError, AttributeError):
    pass

fig_dir = os.path.dirname(__file__)

# Figure 1: Bias-Variance Reduction in Ensembles (Bagging vs Single Tree)
fig, ax = plt.subplots(figsize=(6, 4))
x = np.linspace(0, 10, 200)
y_true = np.sin(x)
np.random.seed(42)
y_single = y_true + np.random.normal(0, 0.4, size=len(x))
y_bagging = y_true + np.random.normal(0, 0.1, size=len(x))

ax.plot(x, y_true, 'k--', label='True Function f(x)', linewidth=2)
ax.plot(x, y_single, color='#e74c3c', alpha=0.6, label='Single High-Variance Tree')
ax.plot(x, y_bagging, color='#2ecc71', linewidth=2, label='Ensemble (Bagging 50 Trees)')
ax.set_title('Bagging Reduces Variance Without Increasing Bias')
ax.set_xlabel('Feature x')
ax.set_ylabel('Target y')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'fig_bagging_variance.png'), dpi=300)
plt.close()

# Figure 2: AdaBoost Sample Weight Updating
fig, ax = plt.subplots(figsize=(6, 4))
x_pts = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
y_pts = np.array([1, 1, 1, -1, -1, -1, -1, 1, 1, 1])
w_initial = np.ones(10) * 0.1
w_round1 = np.array([0.311, 0.311, 0.311, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01])

ax.scatter(x_pts[y_pts==1], np.ones(np.sum(y_pts==1)), s=w_initial[y_pts==1]*1500, color='#3498db', label='Class +1 (Initial)', alpha=0.7)
ax.scatter(x_pts[y_pts==-1], np.zeros(np.sum(y_pts==-1)), s=w_initial[y_pts==-1]*1500, color='#e74c3c', label='Class -1 (Initial)', alpha=0.7)
ax.scatter(x_pts[y_pts==1], np.ones(np.sum(y_pts==1))*0.8, s=w_round1[y_pts==1]*1500, color='#2ecc71', label='Class +1 (After Round 1 Misclass)', alpha=0.7)

ax.axvline(0.75, color='gray', linestyle=':', label='Stump Boundary (x <= 0.75)')
ax.set_yticks([0, 0.8, 1.0])
ax.set_yticklabels(['Class -1', 'Round 1 Boosted Wts', 'Class +1'])
ax.set_title('AdaBoost Sample Weight Re-weighting')
ax.set_xlabel('Sample Position x')
ax.legend(loc='lower left')
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'fig_adaboost_weights.png'), dpi=300)
plt.close()

# Figure 3: Gradient Boosting fitting Residuals
fig, ax = plt.subplots(figsize=(6, 4))
weights_true = np.array([88, 76, 56, 73, 77, 57])
pred_F0 = np.array([71.2]*6)
residuals_1 = weights_true - pred_F0
idx = np.arange(1, 7)

ax.bar(idx - 0.2, weights_true, width=0.4, label='Actual Weight (y)', color='#34495e')
ax.bar(idx + 0.2, residuals_1, width=0.4, label='Residual h1(x) = y - F0', color='#e67e22')
ax.axhline(71.2, color='#e74c3c', linestyle='--', label='Initial Prediction F0 = 71.2 kg')
ax.set_xticks(idx)
ax.set_xticklabels([f'Person {i}' for i in idx])
ax.set_ylabel('Weight (kg) / Residual (kg)')
ax.set_title('Gradient Boosting: Fitting Trees to Pseudo-Residuals')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'fig_gb_residuals.png'), dpi=300)
plt.close()

print('Figures generated for DNN-session13-ensemble successfully.')
