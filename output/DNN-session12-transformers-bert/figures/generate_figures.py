import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../scripts')))
try:
    import figstyle
except ImportError:
    pass

FIG_DIR = os.path.dirname(__file__)

def make_positional_encoding():
    fig, ax = plt.subplots(figsize=(7, 3.5), dpi=300)

    positions = np.arange(0, 50)
    d_model = 128

    pe_dim0 = np.sin(positions / (10000 ** (0 / d_model)))
    pe_dim4 = np.sin(positions / (10000 ** (4 / d_model)))
    pe_dim16 = np.sin(positions / (10000 ** (16 / d_model)))

    ax.plot(positions, pe_dim0, label='Dimension 0 (High Freq)', color='#2b5c8f', lw=2)
    ax.plot(positions, pe_dim4, label='Dimension 4 (Med Freq)', color='#d95f02', lw=2)
    ax.plot(positions, pe_dim16, label='Dimension 16 (Low Freq)', color='#7570b3', lw=2)

    ax.set_xlabel('Token Position ($pos$)', fontsize=10)
    ax.set_ylabel('Positional Encoding Value', fontsize=10)
    ax.set_title('Sinusoidal Positional Embeddings Across Frequencies', fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'positional_encoding.png'), bbox_inches='tight')
    plt.close()

def make_norm_comparison():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.5), dpi=300)

    # Batch Normalization diagram
    ax1.set_title('Batch Normalization', fontsize=11)
    ax1.axis('off')
    rect1 = plt.Rectangle((0.1, 0.1), 0.8, 0.8, facecolor='#f0f4f8', edgecolor='#2b5c8f', lw=2)
    ax1.add_patch(rect1)
    # Highlight vertical slices (across batch N)
    for i in range(3):
        ax1.add_patch(plt.Rectangle((0.2 + i*0.2, 0.15), 0.15, 0.7, facecolor='#377eb8', alpha=0.6))
    ax1.text(0.5, 0.02, 'Normalize across Batch (N)', ha='center', fontsize=9, color='#2b5c8f')

    # Layer Normalization diagram
    ax2.set_title('Layer Normalization', fontsize=11)
    ax2.axis('off')
    rect2 = plt.Rectangle((0.1, 0.1), 0.8, 0.8, facecolor='#fbf8f0', edgecolor='#d95f02', lw=2)
    ax2.add_patch(rect2)
    # Highlight horizontal slices (across features D)
    for j in range(3):
        ax2.add_patch(plt.Rectangle((0.15, 0.2 + j*0.2), 0.7, 0.15, facecolor='#ff7f00', alpha=0.6))
    ax2.text(0.5, 0.02, 'Normalize across Features (D)', ha='center', fontsize=9, color='#d95f02')

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'norm_comparison.png'), bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    make_positional_encoding()
    make_norm_comparison()
    print("Session 12 figures generated successfully.")
