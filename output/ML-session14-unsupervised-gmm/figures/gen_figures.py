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

# Figure 1: K-Means Clustering vs GMM Soft Boundaries
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
np.random.seed(42)

c1 = np.random.normal(loc=[2, 2], scale=0.6, size=(40, 2))
c2 = np.random.normal(loc=[5, 5], scale=0.9, size=(60, 2))

# Hard K-Means
ax1.scatter(c1[:, 0], c1[:, 1], color='#3498db', label='Cluster 1 (Hard)', alpha=0.8)
ax1.scatter(c2[:, 0], c2[:, 1], color='#e74c3c', label='Cluster 2 (Hard)', alpha=0.8)
ax1.set_title('K-Means: Hard Cluster Assignment')
ax1.set_xlabel('Feature x1')
ax1.set_ylabel('Feature x2')
ax1.legend()

# Soft GMM (Contour mapping)
ax2.scatter(c1[:, 0], c1[:, 1], color='#9b59b6', alpha=0.6, label='Data Points')
ax2.scatter(c2[:, 0], c2[:, 1], color='#9b59b6', alpha=0.6)
# Overlay Gaussian ellipses
from matplotlib.patches import Ellipse
e1 = Ellipse(xy=(2, 2), width=2.4, height=2.4, angle=0, edgecolor='#3498db', fc='none', lw=2, label='Component 1 Soft Contour')
e2 = Ellipse(xy=(5, 5), width=3.6, height=3.6, angle=0, edgecolor='#e74c3c', fc='none', lw=2, label='Component 2 Soft Contour')
ax2.add_patch(e1)
ax2.add_patch(e2)
ax2.set_title('GMM: Soft Probabilistic Membership')
ax2.set_xlabel('Feature x1')
ax2.set_ylabel('Feature x2')
ax2.legend()

plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'fig_kmeans_vs_gmm.png'), dpi=300)
plt.close()

# Figure 2: Expectation-Maximization Log-Likelihood Convergence
fig, ax = plt.subplots(figsize=(6, 4))
iters = np.arange(1, 11)
ll = -120 + 70 * (1 - np.exp(-0.5 * iters))

ax.plot(iters, ll, 'o-', color='#2ecc71', linewidth=2, markersize=8)
ax.set_title('EM Algorithm Log-Likelihood Convergence')
ax.set_xlabel('EM Iteration')
ax.set_ylabel('Observed Log-Likelihood L(theta)')
ax.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'fig_em_convergence.png'), dpi=300)
plt.close()

print('Figures generated for ML-session14-unsupervised-gmm successfully.')
