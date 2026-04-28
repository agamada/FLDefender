"""
plot_detection_fp_auto.py
自动从实验日志解析 Detection Rate 和 False Positive Rate，绘制折线图。

Detection Rate (DR):  恶意客户端(ID 0~m-1)被过滤掉的比例，取所有轮次的平均值
False Positive Rate (FPR): 良性客户端(ID m~k-1)被过滤掉的比例，取所有轮次的平均值
"""

import re, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LOG_DIR = './exp/exp4/Cifar10'
M = 8    
K = 40   

ATTACKS = ['LIE', 'MPAF', 'Sign-flip', 'CAMP(clip)', 'CAMP(perturb)']
ATTACK_FILENAMES = {
    'LIE':           'LIE',
    'MPAF':          'MPAF',
    'Sign-flip':     'sign_flip',
    'CAMP(clip)':    'CAMP_clipping_v8',
    'CAMP(perturb)': 'CAMP_perturbation_v5',
}

FILTERS = {
    'MAUD-Norm w=3':  'maud-norm_{a}_w3.log',
    'MAUD-Norm w=5':  'maud-norm_{a}_w5.log',
    'MAUD-Norm w=10': 'maud-norm_{a}_w10.log',
    'MAUD-Cos w=3':   'maud-cosine_{a}_w3.log',
    'MAUD-Cos w=5':   'maud-cosine_{a}_w5.log',
    'MAUD-Cos w=10':  'maud-cosine_{a}_w10.log',
    'Multi-Krum':     'multi-krum_{a}.log',
    'FLAME':          'flame_{a}.log',
}


RE_SELECTED = re.compile(
    r'(?:select clients|selected \d+/\d+ clients):\s*\[([\d,\s]*)\]'
)


def parse_log(path):
    rounds = []
    with open(path) as f:
        for line in f:
            m = RE_SELECTED.search(line)
            if m:
                ids = [int(x) for x in re.findall(r'\d+', m.group(1))]
                rounds.append(ids)
    return rounds

def calc_dr_fpr(rounds):
    malicious = set(range(M))
    benign = set(range(M, K))
    drs, fprs = [], []
    for selected in rounds:
        excluded = set(range(K)) - set(selected)
        drs.append(len(malicious & excluded) / M * 100)
        fprs.append(len(benign & excluded) / len(benign) * 100)
    return round(np.mean(drs), 1), round(np.mean(fprs), 1)



catch = {k: [] for k in FILTERS}
fp    = {k: [] for k in FILTERS}
missing = []

for atk in ATTACKS:
    atk_fn = ATTACK_FILENAMES[atk]
    for fk, tpl in FILTERS.items():
        path = os.path.join(LOG_DIR, tpl.format(a=atk_fn))
        if not os.path.exists(path):
            missing.append(os.path.basename(path))
            catch[fk].append(None); fp[fk].append(None)
            continue
        rounds = parse_log(path)
        if not rounds:
            print(f'[WARN] no filter lines: {os.path.basename(path)}')
            catch[fk].append(None); fp[fk].append(None)
            continue
        dr, fpr = calc_dr_fpr(rounds)
        catch[fk].append(dr); fp[fk].append(fpr)

if missing:
    print(f'[WARN] {len(missing)} log(s) not found: {missing}')



def print_table(name, data):
    print(f'\n{name}:')
    print(f"{'Filter':<20}" + ''.join(f'{a:>15}' for a in ATTACKS))
    for k, vals in data.items():
        row = ''.join(f'{v if v is not None else "-":>15}' for v in vals)
        print(f'{k:<20}{row}')

print_table('Detection Rate (%)', catch)
print_table('False Positive Rate (%)', fp)



sel_keys    = ['MAUD-Norm w=5', 'MAUD-Cos w=5', 'Multi-Krum', 'FLAME']
sel_labels  = ['MAUD-Norm (N=5)', 'MAUD-Cosine (N=5)', 'Multi-Krum', 'FLAME']
sel_colors  = ['#1565C0', '#E65100', '#616161', '#2E7D32']
sel_markers = ['o', 's', 'D', '^']

x = np.arange(len(ATTACKS))
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

for data, ylabel, ax in [
    (catch, 'Detection Rate (%)',      axes[0]),
    (fp,    'False Positive Rate (%)', axes[1]),
]:
    for key, label, c, mk in zip(sel_keys, sel_labels, sel_colors, sel_markers):
        vals = data[key]
        xs = [x[i] for i, v in enumerate(vals) if v is not None]
        ys = [v for v in vals if v is not None]
        ax.plot(xs, ys, label=label, color=c, marker=mk, lw=2.5, ms=8)
    ax.set_xticks(x)
    ax.set_xticklabels(ATTACKS, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc='best')

axes[0].set_ylim(-5, 110)
axes[1].set_ylim(-5, 85)
plt.tight_layout()

os.makedirs('./exp', exist_ok=True)
plt.savefig('./exp/maud_detection_fp_auto.png', dpi=200, bbox_inches='tight')
plt.savefig('./exp/maud_detection_fp_auto.pdf', dpi=300, bbox_inches='tight')
print('\nPlot saved: exp/maud_detection_fp_auto.png / .pdf')
plt.close()
