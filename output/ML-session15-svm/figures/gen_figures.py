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

# Figure 1: SVM Maximum Margin & Support Vectors
fig, ax = plt.subplots(figsize=(6, 4))
np.random.seed(42)

# Class +1 points
x_pos = np.array([[2, 3], [3, 4], [2, 5], [3.5, 3]])
# Class -1 points
x_neg = np.array([[0, 1], [1, 0.5], [1.5, 1], [0.5, 2]])

ax.scatter(x_pos[:, 0], x_pos[:, 1], color='#3498db', s=80, label='Class +1', zorder=3)
ax.scatter(x_neg[:, 0], x_neg[:, 1], color='#e74c3c', s=80, label='Class -1', zorder=3)

# Highlight Support Vectors
svs = np.array([[2, 3], [1.5, 1]])
ax.scatter(svs[:, 0], svs[:, 1], s=200, facecolors='none', edgecolors='#f59e0b', linewidths=2, label='Support Vectors', zorder=4)

# Hyperplane & Margin Lines (w1*x1 + w2*x2 + b = 0 => x2 = -x1 + 3.5)
x_line = np.linspace(-0.5, 4.5, 100)
ax.plot(x_line, -x_line + 3.75, color='#2c3e50', linewidth=2, label='Optimal Hyperplane w^T x + b = 0')
ax.plot(x_line, -x_line + 5.0, color='gray', linestyle='--', label='Margin Boundary (+1)')
ax.plot(x_line, -x_line + 2.5, color='gray', linestyle='--', label='Margin Boundary (-1)')

ax.set_title('SVM Maximum Margin Classifier & Support Vectors')
ax.set_xlabel('Feature x1')
ax.set_ylabel('Feature x2')
ax.legend(loc='upper right')
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'fig_svm_margin.png'), dpi=300)
plt.close()

# Figure 2: Non-linear Kernel Projection
fig, ax = plt.subplots(figsize=(6, 4))
r = np.linspace(0, 5, 100)
y_pos = r**2 + np.random.normal(0, 0.5, size=len(r))
y_neg = -(r**2) + np.random.normal(0, 0.5, size=len(r))

ax.scatter(r, y_pos, color='#3498db', alpha=0.6, label='Class +1 (Transformed)')
ax.scatter(r, y_neg, color='#e74c3c', alpha=0.6, label='Class -1 (Transformed)')
ax.axhline(0, color='#2ecc71', linewidth=2, linestyle='--', label='Linearly Separable Boundary in Phi(x) Space')
ax.set_title('Kernel Trick: Non-linear Mapping to Feature Space')
ax.set_xlabel('Radius r = ||x||')
ax.set_ylabel('Feature Space Projection Phi(x)')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'fig_kernel_trick.png'), dpi=300)
plt.close()

print('Figures generated for ML-session15-svm successfully.')
