import re
import matplotlib.pyplot as plt
import numpy as np
import os

log_dir = "./exp/exp3/UCIHAR"
methods = ["avg", "krum", "median", "trmean", "sad", "flame", "FLDetector"]
labels = ["FedAvg", "Krum", "Median", "TrMean", "SAD", "FLAME", "FLDetector"]
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2"]
linestyles = ["-", "--", "-.", ":", "-", "--", "-."]
markers = ["o", "s", "^", "D", "v", "P", "X"]

fig, ax = plt.subplots(figsize=(10, 6))

for method, label, color, ls, marker in zip(methods, labels, colors, linestyles, markers):
    log_file = os.path.join(log_dir, f"{method}_scale.log")
    asr_values = []
    with open(log_file, "r") as f:
        for line in f:
            m = re.search(r"Attack Success Rate:\s+([\d.]+)", line)
            if m:
                asr_values.append(float(m.group(1)))

    rounds = list(range(len(asr_values)))
    ax.plot(rounds, asr_values, label=label, color=color, linestyle=ls,
            marker=marker, markevery=10, markersize=6, linewidth=1.8)

ax.set_xlabel("Communication Round", fontsize=14)
ax.set_ylabel("Attack Success Rate (ASR)", fontsize=14)
ax.set_title("ASR under Scale + Label Flipping Attack (UCIHAR)", fontsize=15)
ax.legend(fontsize=11, loc="upper left")
ax.set_xlim(0, 100)
ax.set_ylim(-0.02, 1.02)
ax.grid(True, alpha=0.3)
ax.tick_params(labelsize=12)

plt.tight_layout()
plt.savefig("./exp/exp3/UCIHAR/asr_curve.png", dpi=200)
plt.savefig("./exp/exp3/UCIHAR/asr_curve.pdf")
print("Saved to exp/exp3/UCIHAR/asr_curve.png and asr_curve.pdf")
