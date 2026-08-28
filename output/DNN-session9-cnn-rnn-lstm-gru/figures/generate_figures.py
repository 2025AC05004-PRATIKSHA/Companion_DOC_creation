#!/usr/bin/env python3
import sys
import os
import matplotlib.pyplot as plt
import numpy as np

# Allow script to import figstyle from scripts/ when run directly or via build_pdf.py
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../scripts')))

try:
    import figstyle
except ImportError:
    figstyle = None

fig_dir = os.path.dirname(__file__)

def make_cnn_patterns():
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_title("CNN Bottleneck & Residual Design Patterns", fontsize=12, pad=15)

    # Left box: Standard vs Bottleneck
    ax.text(0.2, 0.8, "Standard Conv 3x3\n(256 ch -> 256 ch)\n~590K params",
            ha='center', va='center', bbox=dict(boxstyle="round,pad=0.5", fc="#e1f5fe", ec="#0288d1", lw=2))
    ax.text(0.2, 0.2, "Bottleneck 1x1->3x3->1x1\n(256 -> 64 -> 256 ch)\n~70K params (8x reduction!)",
            ha='center', va='center', bbox=dict(boxstyle="round,pad=0.5", fc="#e8f5e9", ec="#388e3c", lw=2))

    # Right box: Skip Connection y = F(x) + x
    ax.text(0.7, 0.5, "Residual Block:\ny = F(x) + x\nIdentity path bypasses layers,\npreventing vanishing gradients",
            ha='center', va='center', bbox=dict(boxstyle="round,pad=0.6", fc="#fff3e0", ec="#f57c00", lw=2))

    ax.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "cnn_patterns.png"), dpi=200)
    plt.close()

def make_autoregressive_vs_latent():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))

    # Autoregressive
    ax1.set_title("Autoregressive Model", fontsize=11)
    ax1.plot([1, 2, 3, 4], [20, 22, 25, 24], 'o-', label="Observed T")
    ax1.plot(5, 24.0, 's', color='red', label="Predicted T5 = 0.8*T4 + 4")
    ax1.set_xlabel("Time step t")
    ax1.set_ylabel("Temperature (°C)")
    ax1.legend(fontsize=9)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Latent Variable
    ax2.set_title("Latent Variable Model (RNN)", fontsize=11)
    ax2.text(0.5, 0.7, "Hidden State h_t\n(Summary of past history x_1..x_t)",
             ha='center', va='center', bbox=dict(boxstyle="round,pad=0.5", fc="#ede7f6", ec="#512da8", lw=2))
    ax2.text(0.5, 0.3, "h_t = tanh(W_hh * h_{t-1} + W_hx * x_t + b_h)\no_t = W_ho * h_t + b_o",
             ha='center', va='center', bbox=dict(boxstyle="round,pad=0.5", fc="#e0f2f1", ec="#00796b", lw=2))
    ax2.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "autoregressive_vs_latent.png"), dpi=200)
    plt.close()

def make_rnn_unrolled():
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.set_title("Unrolled Vanilla Recurrent Neural Network", fontsize=12, pad=12)

    times = ["t=1", "t=2", "t=3"]
    for i, t in enumerate(times):
        x_pos = i * 2.5
        # Inputs
        ax.text(x_pos, 0, f"x_{i+1}", ha='center', va='center', bbox=dict(boxstyle="circle,pad=0.3", fc="#e1f5fe", ec="#0288d1"))
        # Hidden
        ax.text(x_pos, 1.5, f"h_{i+1}", ha='center', va='center', bbox=dict(boxstyle="circle,pad=0.4", fc="#ede7f6", ec="#512da8"))
        # Output
        ax.text(x_pos, 3, f"o_{i+1}", ha='center', va='center', bbox=dict(boxstyle="circle,pad=0.3", fc="#fff3e0", ec="#f57c00"))

        # Arrows vertical
        ax.annotate("", xy=(x_pos, 1.1), xytext=(x_pos, 0.4), arrowprops=dict(arrowstyle="->", lw=1.5, color="#0288d1"))
        ax.annotate("", xy=(x_pos, 2.6), xytext=(x_pos, 1.9), arrowprops=dict(arrowstyle="->", lw=1.5, color="#f57c00"))

        # Arrows horizontal (W_hh)
        if i < len(times) - 1:
            ax.annotate("", xy=(x_pos + 1.9, 1.5), xytext=(x_pos + 0.6, 1.5), arrowprops=dict(arrowstyle="->", lw=2, color="#512da8"))
            ax.text(x_pos + 1.25, 1.7, "W_hh", ha='center', fontsize=9, color="#512da8")

    ax.set_xlim(-1, 6.5)
    ax.set_ylim(-0.5, 3.5)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "rnn_unrolled.png"), dpi=200)
    plt.close()

def make_bptt_gradient_flow():
    fig, ax = plt.subplots(figsize=(7, 4))
    t_steps = np.arange(1, 21)
    # Gradient magnitude decay: (0.8 * 0.5)^k = 0.4^k
    grad_mag = (0.8 * 0.5)**t_steps

    ax.plot(t_steps, grad_mag, 'o-', color="#d32f2f", lw=2, label="Gradient magnitude ratio (0.4)^k")
    ax.set_yscale('log')
    ax.set_title("Vanishing Gradient Decay Across Time Steps in BPTT", fontsize=11)
    ax.set_xlabel("Steps backwards from output t=20")
    ax.set_ylabel("Gradient Magnitude (Log Scale)")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.axhline(1e-8, color="black", linestyle=":", label="Step 20 gradient ~ 10^-8")
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "bptt_gradient_flow.png"), dpi=200)
    plt.close()

def make_gru_architecture():
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_title("Gated Recurrent Unit (GRU) Internal Gate Flow", fontsize=12, pad=12)

    # Reset Gate box
    ax.text(0.25, 0.7, "Reset Gate (r_t)\nr_t = σ(W_r x_t + U_r h_{t-1} + b_r)\nControls how much past memory to drop",
            ha='center', va='center', bbox=dict(boxstyle="round,pad=0.5", fc="#ffebee", ec="#c62828", lw=1.5))

    # Update Gate box
    ax.text(0.75, 0.7, "Update Gate (z_t)\nz_t = σ(W_z x_t + U_z h_{t-1} + b_z)\nInterpolates between old h_{t-1} and candidate",
            ha='center', va='center', bbox=dict(boxstyle="round,pad=0.5", fc="#e3f2fd", ec="#1565c0", lw=1.5))

    # Candidate & Output box
    ax.text(0.5, 0.25, "Candidate: h~_t = tanh(W_h x_t + U_h (r_t ⊙ h_{t-1}) + b_h)\nFinal State: h_t = z_t ⊙ h_{t-1} + (1 - z_t) ⊙ h~_t",
            ha='center', va='center', bbox=dict(boxstyle="round,pad=0.5", fc="#e8f5e9", ec="#2e7d32", lw=2))

    ax.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "gru_architecture.png"), dpi=200)
    plt.close()

def make_lstm_architecture():
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    ax.set_title("LSTM Memory Cell Architecture: Long-Term vs Short-Term Flow", fontsize=12, pad=12)

    # Cell State Highway top
    ax.annotate("", xy=(8, 3.2), xytext=(0, 3.2), arrowprops=dict(arrowstyle="->", lw=3, color="#2e7d32"))
    ax.text(0.5, 3.5, "Cell State (c_t) Highway: c_t = f_t ⊙ c_{t-1} + i_t ⊙ c~_t", color="#2e7d32", weight="bold")

    # Gates
    ax.text(1.5, 1.5, "Forget Gate (f_t)\nf_t = σ(W_f [h_{t-1}, x_t])",
            ha='center', va='center', bbox=dict(boxstyle="round,pad=0.4", fc="#ffebee", ec="#c62828"))
    ax.text(4.25, 1.5, "Input Gate (i_t) & Candidate (c~_t)\ni_t = σ(W_i [h_{t-1}, x_t])\nc~_t = tanh(W_c [h_{t-1}, x_t])",
            ha='center', va='center', bbox=dict(boxstyle="round,pad=0.4", fc="#e3f2fd", ec="#1565c0"))
    ax.text(7.0, 1.5, "Output Gate (o_t)\no_t = σ(W_o [h_{t-1}, x_t])\nh_t = o_t ⊙ tanh(c_t)",
            ha='center', va='center', bbox=dict(boxstyle="round,pad=0.4", fc="#fff3e0", ec="#ef6c00"))

    ax.set_xlim(-0.5, 9)
    ax.set_ylim(0, 4)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "lstm_architecture.png"), dpi=200)
    plt.close()

def make_rnn_variants_comparison():
    fig, ax = plt.subplots(figsize=(8, 4))
    categories = ['Vanilla RNN', 'Bidirectional RNN', 'GRU', 'LSTM']
    param_counts = [2661392, 2628624, 274176, 365568] # Normalized example values from slides

    bars = ax.bar(categories, param_counts, color=['#7e57c2', '#42a5f5', '#66bb6a', '#ffa726'], width=0.5)
    ax.set_title("Parameter Count Comparison (Normalized Example)", fontsize=11)
    ax.set_ylabel("Total Parameters (d=100, h=256 / h=128)")
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)

    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 50000, f"{yval:,}", ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "rnn_variants_comparison.png"), dpi=200)
    plt.close()

if __name__ == "__main__":
    make_cnn_patterns()
    make_autoregressive_vs_latent()
    make_rnn_unrolled()
    make_bptt_gradient_flow()
    make_gru_architecture()
    make_lstm_architecture()
    make_rnn_variants_comparison()
    print("All figures successfully generated.")
