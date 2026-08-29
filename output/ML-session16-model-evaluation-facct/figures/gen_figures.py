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

# Figure 1: ROC Curve and AUC Area
fig, ax = plt.subplots(figsize=(6, 4))
fpr = np.linspace(0, 1, 100)
tpr = np.sqrt(fpr)  # Convex ROC curve with high AUC

ax.plot(fpr, tpr, color='#3498db', linewidth=2, label='Classifier ROC (AUC = 0.85)')
ax.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Random Chance (AUC = 0.50)')
ax.fill_between(fpr, tpr, alpha=0.2, color='#3498db')
ax.set_title('Receiver Operating Characteristic (ROC) Curve')
ax.set_xlabel('False Positive Rate (1 - Specificity)')
ax.set_ylabel('True Positive Rate (Sensitivity / Recall)')
ax.legend(loc='lower right')
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'fig_roc_auc.png'), dpi=300)
plt.close()

# Figure 2: Model Interpretability vs Performance Trade-off
fig, ax = plt.subplots(figsize=(6, 4))
models = ['Linear Reg', 'Decision Tree', 'KNN', 'Ensembles', 'Deep Learning']
interp = [90, 80, 70, 40, 20]
perf = [30, 50, 60, 85, 95]

ax.scatter(interp, perf, color='#e74c3c', s=100)
for i, txt in enumerate(models):
    ax.annotate(txt, (interp[i]+1, perf[i]+1), fontsize=9)

ax.plot(interp, perf, color='#9b59b6', linestyle=':', alpha=0.7)
ax.set_title('Interpretability vs Predictive Performance Trade-Off')
ax.set_xlabel('Model Interpretability / Transparency (%)')
ax.set_ylabel('Predictive Performance / Accuracy (%)')
ax.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'fig_tradeoff_facct.png'), dpi=300)
plt.close()

print('Figures generated for ML-session16-model-evaluation-facct successfully.')
