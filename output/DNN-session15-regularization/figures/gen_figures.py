import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../scripts')))
try:
    import figstyle
except ImportError:
    pass

def generate_regularization_geometry_fig():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    # L1 Diamond vs L2 Circle
    # L1: |w1| + |w2| <= 1
    l1_x = [1, 0, -1, 0, 1]
    l1_y = [0, 1, 0, -1, 0]

    # L2: w1^2 + w2^2 <= 1
    theta = np.linspace(0, 2*np.pi, 100)
    l2_x = np.cos(theta)
    l2_y = np.sin(theta)

    # Loss contours
    w1, w2 = np.meshgrid(np.linspace(-1.5, 2.5, 100), np.linspace(-1.5, 2.5, 100))
    loss = (w1 - 1.5)**2 + (w2 - 1.5)**2

    ax1.contour(w1, w2, loss, levels=7, colors='gray', linestyles='--')
    ax1.plot(l1_x, l1_y, 'b-', linewidth=2, label='L1 Constraint (|w1|+|w2|<=1)')
    ax1.plot(0, 1, 'ro', label='Corner Solution (Sparsity w1=0)')
    ax1.set_title("L1 Lasso Geometry (Sparsity Corner)")
    ax1.set_xlabel("w1")
    ax1.set_ylabel("w2")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left', fontsize=8)

    ax2.contour(w1, w2, loss, levels=7, colors='gray', linestyles='--')
    ax2.plot(l2_x, l2_y, 'g-', linewidth=2, label='L2 Constraint (w1^2+w2^2<=1)')
    ax2.plot(0.75, 0.75, 'go', label='Smooth Tangency')
    ax2.set_title("L2 Weight Decay Geometry (Smooth Shrinkage)")
    ax2.set_xlabel("w1")
    ax2.set_ylabel("w2")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper left', fontsize=8)

    out_path = os.path.join(os.path.dirname(__file__), 'regularization_geometry.png')
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

if __name__ == '__main__':
    generate_regularization_geometry_fig()
    print("Session 15 figures generated successfully.")
