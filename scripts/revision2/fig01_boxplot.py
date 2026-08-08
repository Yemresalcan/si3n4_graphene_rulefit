"""Figure 1: R2 distribution over the 30 repeated 70/30 splits.

Redrawn at the final printed width (131 mm) with horizontal orientation so that
the model names are read without rotation, and with the individual 30 split
outcomes overlaid so that the reader can see the resampling evidence directly
rather than only a summary box.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib.pyplot as plt
from fig_style import MODEL_COLOURS, TEXT_W_IN, apply_style, save

from paths import FIG_OUT as OUT, results_file

RESULTS = results_file("paper_30split_r2_raw.csv")

apply_style()
r2 = pd.read_csv(RESULTS)

order = [
    "XGBoost (Extra Features, Tuned)",
    "XGBoost (Default, Tuned)",
    "RuleFit (Tuned)",
    "RuleFit (Default)",
    "DNN Small (32-16)",
    "Few-Shot KNN",
    "Linear Regression",
]
labels = [
    "XGBoost\n(Extra Features, Tuned)",
    "XGBoost\n(Default, Tuned)$^{*}$",
    "RuleFit (Tuned)",
    "RuleFit (Default)",
    "DNN Small (32–16)",
    "Few-Shot KNN",
    "Linear Regression",
]

fig, ax = plt.subplots(figsize=(TEXT_W_IN, 3.9), layout="constrained")
positions = np.arange(len(order))[::-1]

bp = ax.boxplot(
    [r2[m].values for m in order],
    positions=positions,
    vert=False,
    widths=0.58,
    patch_artist=True,
    showfliers=False,
    medianprops=dict(color="black", linewidth=1.2),
    whiskerprops=dict(linewidth=0.7),
    capprops=dict(linewidth=0.7),
    boxprops=dict(linewidth=0.7),
)
for patch, m in zip(bp["boxes"], order):
    patch.set_facecolor(MODEL_COLOURS[m])
    patch.set_alpha(0.8)

rng = np.random.default_rng(42)
for pos, m in zip(positions, order):
    v = np.clip(r2[m].values, -0.015, None)
    jitter = rng.uniform(-0.15, 0.15, size=len(v))
    ax.scatter(v, pos + jitter, s=3.5, color="#1a1a1a", alpha=0.5,
               linewidths=0, zorder=3)

# median value printed at the right edge of every row
for pos, m in zip(positions, order):
    ax.text(1.055, pos, "%.3f" % np.median(r2[m].values),
            va="center", ha="center", fontsize=7.2)
ax.text(1.055, positions[0] + 0.72, "median", va="center", ha="center",
        fontsize=7.0, fontstyle="italic")

# models whose distribution leaves the plotted range
for pos, m in zip(positions, order):
    n_below = int((r2[m] < 0).sum())
    if n_below:
        ax.text(0.02, pos + 0.34,
                "%d/30 splits $R^2<0$ (min %.1f)" % (n_below, r2[m].min()),
                fontsize=6.4, va="bottom", ha="left", color="#8b0000")
        ax.plot([-0.012], [pos], marker="<", markersize=3.4, color="#8b0000",
                clip_on=False, zorder=4)

ax.set_yticks(positions)
ax.set_yticklabels(labels)
ax.set_ylim(-0.75, len(order) - 0.15)
ax.set_xlim(-0.02, 1.11)
ax.set_xticks(np.arange(0, 1.01, 0.2))
ax.set_xlabel("Coefficient of determination, $R^2$")
ax.axvline(0.0, color="#8b0000", linestyle=":", linewidth=0.8)
ax.grid(axis="x", linestyle="--", alpha=0.35)
ax.set_axisbelow(True)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
ax.spines["bottom"].set_bounds(-0.02, 1.0)

save(fig, "fig01_r2_distribution", OUT)

print("\nvalues for caption:")
for m in order:
    v = r2[m]
    q1, med, q3 = v.quantile([0.25, 0.5, 0.75])
    print("  %-34s median=%.3f IQR=%.3f [%.3f-%.3f] min=%.3f max=%.3f"
          % (m, med, q3 - q1, q1, q3, v.min(), v.max()))
rf = r2["RuleFit (Tuned)"]
print("  RuleFit > XGB(extra): %d/30 ; > XGB(default): %d/30 ; > KNN: %d/30"
      % ((rf > r2["XGBoost (Extra Features, Tuned)"]).sum(),
         (rf > r2["XGBoost (Default, Tuned)"]).sum(),
         (rf > r2["Few-Shot KNN"]).sum()))
