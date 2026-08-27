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

def make_seq2seq_bottleneck():
    fig, ax = plt.subplots(figsize=(8, 3.5), dpi=300)
    ax.axis('off')

    # Encoder tokens
    words = ['The', 'cat', 'sat', 'on', 'the', 'mat']
    for i, w in enumerate(words):
        x = 0.8 + i * 1.0
        ax.add_patch(plt.Rectangle((x-0.35, 1.0), 0.7, 0.6, facecolor='#2b5c8f', edgecolor='black', lw=1.5))
        ax.text(x, 1.3, w, color='white', ha='center', va='center', fontweight='bold', fontsize=9)
        ax.annotate('', xy=(4.0, 2.5), xytext=(x, 1.6), arrowprops=dict(arrowstyle="->", color='#2b5c8f', lw=1.2))

    # Bottleneck Context Vector
    ax.add_patch(plt.Circle((4.0, 2.8), 0.5, facecolor='#d95f02', edgecolor='black', lw=2))
    ax.text(4.0, 2.8, 'Fixed Vector\nc', color='white', ha='center', va='center', fontweight='bold', fontsize=9)

    # Decoder tokens
    dec_words = ['Le', 'chat', 'etait', 'assis']
    for j, dw in enumerate(dec_words):
        x = 6.2 + j * 1.1
        ax.add_patch(plt.Rectangle((x-0.4, 1.0), 0.8, 0.6, facecolor='#7570b3', edgecolor='black', lw=1.5))
        ax.text(x, 1.3, dw, color='white', ha='center', va='center', fontweight='bold', fontsize=9)
        ax.annotate('', xy=(x, 1.6), xytext=(4.0, 2.8), arrowprops=dict(arrowstyle="->", color='#7570b3', lw=1.2))

    plt.title('Seq2Seq Encoder-Decoder Bottleneck Problem', fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'seq2seq_bottleneck.png'), bbox_inches='tight')
    plt.close()

def make_attention_heatmap():
    fig, ax = plt.subplots(figsize=(6, 4.5), dpi=300)

    src = ['The', 'cat', 'sat', 'on', 'the', 'mat']
    tgt = ['Le', 'chat', 'etait', 'assis', 'sur', 'le', 'tapis']

    weights = np.array([
        [0.8, 0.1, 0.05, 0.0, 0.05, 0.0],
        [0.05, 0.85, 0.05, 0.0, 0.0, 0.05],
        [0.0, 0.05, 0.8, 0.1, 0.0, 0.05],
        [0.0, 0.0, 0.75, 0.2, 0.0, 0.05],
        [0.0, 0.0, 0.1, 0.8, 0.1, 0.0],
        [0.1, 0.0, 0.0, 0.1, 0.7, 0.1],
        [0.0, 0.05, 0.0, 0.05, 0.1, 0.8]
    ])

    im = ax.imshow(weights, cmap='Blues')
    ax.set_xticks(np.arange(len(src)))
    ax.set_yticks(np.arange(len(tgt)))
    ax.set_xticklabels(src, fontsize=9)
    ax.set_yticklabels(tgt, fontsize=9)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    ax.set_xlabel('Source Sequence (Encoder)', fontsize=10)
    ax.set_ylabel('Target Sequence (Decoder)', fontsize=10)
    ax.set_title(r'Attention Alignment Weight Heatmap ($\alpha_{ij}$)', fontsize=11)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.ax.set_ylabel('Attention Weight', rotation=-90, va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'attention_heatmap.png'), bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    make_seq2seq_bottleneck()
    make_attention_heatmap()
    print("Session 11 figures generated successfully.")
