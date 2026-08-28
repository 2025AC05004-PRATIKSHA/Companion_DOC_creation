import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Ensure scripts dir is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../scripts')))
try:
    import figstyle
except ImportError:
    pass

FIG_DIR = os.path.dirname(__file__)

def make_rnn_unroll():
    fig, ax = plt.subplots(figsize=(8, 3.5), dpi=300)
    ax.axis('off')

    # Draw unrolled RNN across 3 timesteps t-1, t, t+1
    steps = ['$t-1$', '$t$', '$t+1$']
    x_coords = [1, 3.5, 6]

    for i, (step, x) in enumerate(zip(steps, x_coords)):
        # Input node
        ax.add_patch(plt.Circle((x, 1), 0.4, color='#2b5c8f', ec='black', lw=1.5))
        ax.text(x, 1, f'$x_{{{step[1:-1]}}}$', color='white', ha='center', va='center', fontweight='bold')

        # Hidden node
        ax.add_patch(plt.Circle((x, 3), 0.4, color='#d95f02', ec='black', lw=1.5))
        ax.text(x, 3, f'$h_{{{step[1:-1]}}}$', color='white', ha='center', va='center', fontweight='bold')

        # Output node
        ax.add_patch(plt.Circle((x, 5), 0.4, color='#7570b3', ec='black', lw=1.5))
        ax.text(x, 5, f'$\hat{{y}}_{{{step[1:-1]}}}$', color='white', ha='center', va='center', fontweight='bold')

        # Vertical arrows
        ax.annotate('', xy=(x, 2.6), xytext=(x, 1.4), arrowprops=dict(arrowstyle="->", lw=1.5, color='#2b5c8f'))
        ax.text(x + 0.15, 2.0, '$W_{hx}$', color='#2b5c8f', fontsize=10, va='center')

        ax.annotate('', xy=(x, 4.6), xytext=(x, 3.4), arrowprops=dict(arrowstyle="->", lw=1.5, color='#7570b3'))
        ax.text(x + 0.15, 4.0, '$W_{yh}$', color='#7570b3', fontsize=10, va='center')

        # Horizontal recurrent arrows
        if i < len(x_coords) - 1:
            next_x = x_coords[i+1]
            ax.annotate('', xy=(next_x - 0.4, 3), xytext=(x + 0.4, 3), arrowprops=dict(arrowstyle="->", lw=1.5, color='#d95f02'))
            ax.text((x + next_x)/2, 3.2, '$W_{hh}$', color='#d95f02', fontsize=10, ha='center')

    ax.set_xlim(0, 7.5)
    ax.set_ylim(0, 6)
    plt.title('Unrolled Recurrent Neural Network Computational Graph', fontsize=12, pad=10)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'rnn_unroll.png'), bbox_inches='tight')
    plt.close()

def make_vanishing_gradient():
    fig, ax = plt.subplots(figsize=(7, 3.5), dpi=300)
    timesteps = np.arange(1, 11)
    # Derivative of tanh is <= 1, repeatedly multiplying leads to exponential decay
    gamma_09 = 0.9 ** timesteps
    gamma_05 = 0.5 ** timesteps
    gamma_02 = 0.2 ** timesteps

    ax.plot(timesteps, gamma_09, 'o-', label=r'Max gradient norm per step $\approx 0.9$', color='#2b5c8f', lw=2)
    ax.plot(timesteps, gamma_05, 's-', label=r'Typical gradient norm per step $\approx 0.5$', color='#d95f02', lw=2)
    ax.plot(timesteps, gamma_02, '^--', label=r'Severe bottleneck $\approx 0.2$', color='#e7298a', lw=2)

    ax.set_xlabel('Time Steps Backwards ($k$ steps in BPTT)', fontsize=10)
    ax.set_ylabel('Effective Gradient Magnitude', fontsize=10)
    ax.set_title('Vanishing Gradient Effect Over Temporal Steps', fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'vanishing_gradient.png'), bbox_inches='tight')
    plt.close()

def make_lstm_cell():
    fig, ax = plt.subplots(figsize=(7, 4), dpi=300)
    ax.axis('off')

    # Outer box for LSTM cell
    rect = plt.Rectangle((0.5, 0.5), 6, 4, fill=True, facecolor='#f0f4f8', edgecolor='#2b5c8f', lw=2)
    ax.add_patch(rect)
    ax.text(3.5, 4.2, 'LSTM Cell ($t$)', fontsize=12, fontweight='bold', ha='center', color='#2b5c8f')

    # Gate boxes
    gates = [('Forget Gate\n$f_t = \sigma(...)$', 1.2, 1.5, '#e41a1c'),
             ('Input Gate\n$i_t = \sigma(...)$', 2.8, 1.5, '#377eb8'),
             ('Candidate\n$\tilde{C}_t = \tanh(...)$', 4.3, 1.5, '#4daf4a'),
             ('Output Gate\n$o_t = \sigma(...)$', 5.6, 1.5, '#984ea3')]

    for title, x, y, col in gates:
        g_box = plt.Rectangle((x-0.6, y-0.4), 1.2, 0.8, fill=True, facecolor='white', edgecolor=col, lw=1.5)
        ax.add_patch(g_box)
        ax.text(x, y, title, fontsize=7, ha='center', va='center', color=col, fontweight='bold')

    # Cell State Highway
    ax.annotate('', xy=(6.5, 3.5), xytext=(0.5, 3.5), arrowprops=dict(arrowstyle="->", lw=2.5, color='#d95f02'))
    ax.text(0.2, 3.5, '$C_{t-1}$', fontsize=10, fontweight='bold', color='#d95f02', va='center')
    ax.text(6.8, 3.5, '$C_t$', fontsize=10, fontweight='bold', color='#d95f02', va='center')
    ax.text(3.5, 3.7, 'Constant Error Carousel (Cell State Highway)', fontsize=9, color='#d95f02', ha='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'lstm_cell.png'), bbox_inches='tight')
    plt.close()

def make_gru_cell():
    fig, ax = plt.subplots(figsize=(7, 3.5), dpi=300)
    ax.axis('off')

    rect = plt.Rectangle((0.5, 0.5), 6, 3.5, fill=True, facecolor='#fbf8f0', edgecolor='#d95f02', lw=2)
    ax.add_patch(rect)
    ax.text(3.5, 3.7, 'Gated Recurrent Unit (GRU)', fontsize=12, fontweight='bold', ha='center', color='#d95f02')

    gates = [('Reset Gate\n$r_t = \sigma(W_r x_t + U_r h_{t-1})$', 2.0, 1.8, '#e7298a'),
             ('Update Gate\n$z_t = \sigma(W_z x_t + U_z h_{t-1})$', 5.0, 1.8, '#7570b3')]

    for title, x, y, col in gates:
        g_box = plt.Rectangle((x-1.1, y-0.5), 2.2, 1.0, fill=True, facecolor='white', edgecolor=col, lw=1.5)
        ax.add_patch(g_box)
        ax.text(x, y, title, fontsize=8, ha='center', va='center', color=col, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'gru_cell.png'), bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    make_rnn_unroll()
    make_vanishing_gradient()
    make_lstm_cell()
    make_gru_cell()
    print("Session 10 figures generated successfully.")
