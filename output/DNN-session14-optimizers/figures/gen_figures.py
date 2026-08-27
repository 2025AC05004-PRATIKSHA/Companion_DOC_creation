import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../scripts')))
try:
    import figstyle
except ImportError:
    pass

def generate_optimizer_trajectories_fig():
    fig, ax = plt.subplots(figsize=(8, 5))

    # 1D loss f(x) = x^2
    # 3 steps for all 6 optimizers from x_0 = 4.0, eta = 0.1
    steps = [0, 1, 2, 3]
    gd = [4.000, 3.200, 2.560, 2.048]
    mom = [4.000, 3.200, 1.840, 0.248]
    nag = [4.000, 3.200, 1.984, 0.712]
    adagrad = [4.000, 3.900, 3.830, 3.774]
    rmsprop = [4.000, 3.684, 3.464, 3.284]
    adam = [4.000, 3.900, 3.800, 3.700]

    ax.plot(steps, gd, marker='o', label='GD')
    ax.plot(steps, mom, marker='s', label='Momentum')
    ax.plot(steps, nag, marker='^', label='NAG')
    ax.plot(steps, adagrad, marker='d', label='Adagrad')
    ax.plot(steps, rmsprop, marker='x', label='RMSProp')
    ax.plot(steps, adam, marker='v', label='Adam')

    ax.set_title("3-Step Parameter Trajectories for f(x) = x² (x₀=4.0, η=0.1)")
    ax.set_xlabel("Step t")
    ax.set_ylabel("Parameter Value x_t")
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='upper right')

    out_path = os.path.join(os.path.dirname(__file__), 'optimizer_trajectories.png')
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

if __name__ == '__main__':
    generate_optimizer_trajectories_fig()
    print("Session 14 figures generated successfully.")
