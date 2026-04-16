import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

attacks = ['LIE', 'MPAF', 'Sign-flip', 'CAMP(clip)', 'CAMP(perturb)']
x = np.arange(len(attacks))

# Detection Rate data (Cifar10, exp4, m=8)
catch = {
    'MAUD-Norm w=3':  [97.0, 100.0, 35.9, 100.0, 100.0],
    'MAUD-Norm w=5':  [100.0, 100.0, 44.6, 100.0, 100.0],
    'MAUD-Norm w=10': [100.0, 100.0, 54.2, 100.0, 100.0],
    'MAUD-Cos w=3':   [37.6, 100.0, 81.3, 100.0, 100.0],
    'MAUD-Cos w=5':   [49.5, 100.0, 56.1, 100.0, 100.0],
    'MAUD-Cos w=10':  [55.4, 100.0, 56.9, 100.0, 100.0],
    'Multi-Krum':     [0.0, 100.0, 100.0, 0.0, 0.0],
    'FLAME':          [50.0, 100.0, 62.6, 52.4, 51.7],
}

# False Positive Rate data
fp = {
    'MAUD-Norm w=3':  [10.4, 0.0, 57.4, 4.5, 0.0],
    'MAUD-Norm w=5':  [8.4, 0.0, 71.0, 1.1, 0.0],
    'MAUD-Norm w=10': [6.0, 0.0, 79.7, 0.3, 0.0],
    'MAUD-Cos w=3':   [38.9, 23.7, 31.1, 24.8, 24.6],
    'MAUD-Cos w=5':   [36.0, 20.6, 33.2, 21.8, 21.5],
    'MAUD-Cos w=10':  [33.1, 20.5, 31.1, 19.9, 19.6],
    'Multi-Krum':     [28.1, 3.1, 3.1, 28.1, 28.1],
    'FLAME':          [45.0, 33.8, 41.2, 45.3, 45.4],
}

# --- Plot 1: MAUD-Norm vs MAUD-Cosine (w=5) + baselines ---
selected = ['MAUD-Norm w=5', 'MAUD-Cos w=5', 'Multi-Krum', 'FLAME']
sel_labels = ['MAUD-Norm (N=5)', 'MAUD-Cosine (N=5)', 'Multi-Krum', 'FLAME']
sel_colors = ['#1565C0', '#E65100', '#616161', '#2E7D32']
sel_markers = ['o', 's', 'D', '^']

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

for data, title, ax in [(catch, '(a) Detection Rate', axes[0]),
                         (fp, '(b) False Positive Rate', axes[1])]:
    for name, label, c, m in zip(selected, sel_labels, sel_colors, sel_markers):
        ax.plot(x, data[name], label=label, color=c, marker=m, lw=2.5, ms=8)
    ax.set_xticks(x)
    ax.set_xticklabels(attacks, fontsize=10)
    ax.set_ylabel(title.split(') ')[1] + ' (%)', fontsize=11)
    # ax.set_title(title, fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc='best')

axes[0].set_ylim(-5, 110)
axes[1].set_ylim(-5, 85)
plt.tight_layout()
plt.savefig('./exp/maud_detection_fp.png', dpi=200, bbox_inches='tight')
plt.savefig('./exp/maud_detection_cifar10'
'.pdf', dpi=300, bbox_inches='tight')
print('Plot 1 (w=5 vs baselines) saved')
plt.close()
