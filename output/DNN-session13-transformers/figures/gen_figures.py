import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Add scripts directory to path to import figstyle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../scripts')))
try:
    import figstyle
except ImportError:
    pass

def generate_positional_encoding_fig():
    fig, ax = plt.subplots(figsize=(8, 4))
    pos = np.arange(20)
    d_model = 4

    # PE formula values
    pe = np.zeros((20, d_model))
    for p in pos:
        for i in range(d_model // 2):
            pe[p, 2*i] = np.sin(p / (10000 ** (2*i / d_model)))
            pe[p, 2*i + 1] = np.cos(p / (10000 ** (2*i / d_model)))

    for i in range(d_model):
        ax.plot(pos, pe[:, i], marker='o', label=f'Dim {i} ({"sin" if i%2==0 else "cos"})')

    ax.set_title("Positional Encoding Values Across Positions 0-19 (d_model=4)")
    ax.set_xlabel("Position in Sequence (pos)")
    ax.set_ylabel("Encoding Value PE(pos, i)")
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='upper right')

    out_path = os.path.join(os.path.dirname(__file__), 'pe_curves.png')
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def generate_attention_heatmap_fig():
    fig, ax = plt.subplots(figsize=(6, 5))
    words = ["I", "love", "NLP"]
    scores = np.array([[0.0, 2.0, 1.0],
                       [1.0, 3.0, 0.0],
                       [0.0, 1.0, 2.0]])
    # softmax row-wise
    exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

    im = ax.imshow(weights, cmap='Purples', vmin=0, vmax=1)
    ax.set_xticks(range(len(words)))
    ax.set_yticks(range(len(words)))
    ax.set_xticklabels(words)
    ax.set_yticklabels(words)
    ax.set_title("Self-Attention Weights Matrix (Softmax Rows)")

    for i in range(len(words)):
        for j in range(len(words)):
            ax.text(j, i, f"{weights[i, j]:.3f}", ha='center', va='center',
                    color='white' if weights[i, j] > 0.5 else 'black')

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    out_path = os.path.join(os.path.dirname(__file__), 'attention_heatmap.png')
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def generate_cross_attention_fig():
    fig, ax = plt.subplots(figsize=(7, 4.5))
    french = ["<sos>", "Mon", "nom", "est"]
    english = ["My", "name", "is", "Ravi"]

    weights = np.array([
        [0.257, 0.171, 0.236, 0.336],
        [0.409, 0.149, 0.194, 0.248],
        [0.447, 0.206, 0.189, 0.158],
        [0.317, 0.344, 0.222, 0.118]
    ])

    im = ax.imshow(weights, cmap='Blues', vmin=0, vmax=0.5)
    ax.set_xticks(range(len(english)))
    ax.set_yticks(range(len(french)))
    ax.set_xticklabels(english)
    ax.set_yticklabels(french)
    ax.set_xlabel("English Encoder Keys / Values")
    ax.set_ylabel("French Decoder Queries")
    ax.set_title("Cross-Attention Alignment Weights (French Queries -> English Keys)")

    for i in range(len(french)):
        for j in range(len(english)):
            ax.text(j, i, f"{weights[i, j]:.3f}", ha='center', va='center',
                    color='white' if weights[i, j] > 0.35 else 'black')

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    out_path = os.path.join(os.path.dirname(__file__), 'cross_attention.png')
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

if __name__ == '__main__':
    generate_positional_encoding_fig()
    generate_attention_heatmap_fig()
    generate_cross_attention_fig()
    print("Figures generated successfully.")
