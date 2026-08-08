"""Figure 2: correlation structure of the 30-feature space.

Reviewer #1 (second round) noted that the in-cell numbers of the previous
correlation matrix were unreadable. A 31 x 31 annotated matrix cannot carry
legible digits inside a 131 mm text block: each cell is only ~12 pt wide, so a
four-character coefficient would need roughly 3 pt type. The matrix is therefore
shown as colour only, with legible variable names, and the coefficients the text
actually discusses are given numerically in panel (b) and in the caption.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib.pyplot as plt
from data_pipeline import TARGET, feature_matrix
from fig_style import TEXT_W_IN, apply_style, save, short
from matplotlib import colors as mcolors

from paths import FIG_OUT as OUT

apply_style()
X, y = feature_matrix()
corr = X.join(y).corr(method="pearson")
names = [short(c) for c in corr.columns]
n = len(names)

# sanity check against the previously published version of this figure
assert abs(corr.loc["Si3N4 (wt%)", TARGET] - 0.63) < 0.01
assert abs(corr.loc["α-Si3N4 content (%)", "β-Si3N4 Content (%)"] + 1.0) < 0.01

fig = plt.figure(figsize=(TEXT_W_IN, 6.75), layout="constrained")
gs = fig.add_gridspec(2, 1, height_ratios=[4.55, 2.05], hspace=0.02)

# ---------------------------------------------------------------- panel (a)
ax = fig.add_subplot(gs[0])
M = corr.values.copy()
M[np.triu_indices(n, 0)] = np.nan
cmap = plt.get_cmap("RdBu_r").copy()
cmap.set_bad("white")
im = ax.imshow(M, cmap=cmap, norm=mcolors.Normalize(-1, 1), aspect="equal")

for i in range(n):
    for j in range(i):
        if abs(M[i, j]) >= 0.9:
            ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                       edgecolor="black", linewidth=0.7))

ax.set_xticks(range(n))
ax.set_yticks(range(n))
ax.set_xticklabels(names, rotation=90, fontsize=6.2)
ax.set_yticklabels(names, fontsize=6.2)
ax.tick_params(length=1.5, pad=1.2)
for lbl in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
    if lbl.get_text() == short(TARGET):
        lbl.set_color("#8b0000")
        lbl.set_fontweight("bold")
ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
ax.grid(which="minor", color="white", linewidth=0.35)
ax.tick_params(which="minor", length=0)
for side in ax.spines.values():
    side.set_visible(False)
ax.set_title("(a) Pearson correlation matrix, 30 features and target",
             fontsize=8.2, pad=4, loc="left")

# The upper triangle is empty, so the colour bar is placed inside it.
cax = ax.inset_axes([0.74, 0.50, 0.032, 0.34])
cb = fig.colorbar(im, cax=cax, ticks=[-1, -0.5, 0, 0.5, 1])
cb.set_label("Pearson $r$", fontsize=7.5)
cb.ax.tick_params(labelsize=7, length=2)
cb.outline.set_linewidth(0.4)
ax.text(0.40, 0.97,
        "Black outline: $|r| > 0.9$\n(8 of the 900 feature pairs)",
        transform=ax.transAxes, fontsize=6.8, va="top", ha="left")

# ---------------------------------------------------------------- panel (b)
ax2 = fig.add_subplot(gs[1])
tgt = corr[TARGET].drop(TARGET).sort_values(key=abs, ascending=False).head(12)[::-1]
pos = np.arange(len(tgt))
cols = ["#b2182b" if v > 0 else "#2166ac" for v in tgt.values]
ax2.barh(pos, tgt.values, color=cols, height=0.68, linewidth=0.4,
         edgecolor="black", alpha=0.9)
for p, v in zip(pos, tgt.values):
    ax2.text(v + (0.025 if v > 0 else -0.025), p, "%+.2f" % v,
             va="center", ha="left" if v > 0 else "right", fontsize=6.8)
ax2.set_yticks(pos)
ax2.set_yticklabels([short(i) for i in tgt.index], fontsize=6.8)
ax2.set_xlim(-0.85, 0.85)
ax2.set_xticks(np.arange(-0.8, 0.81, 0.2))
ax2.tick_params(labelsize=7, length=2)
ax2.axvline(0, color="black", linewidth=0.6)
ax2.grid(axis="x", linestyle="--", alpha=0.3)
ax2.set_axisbelow(True)
ax2.set_xlabel("Pearson $r$ with Vickers hardness", fontsize=8)
for side in ("top", "right", "left"):
    ax2.spines[side].set_visible(False)
ax2.set_title("(b) Twelve strongest linear associations with the target",
              fontsize=8.2, pad=4, loc="left")

save(fig, "fig02_correlation_structure", OUT)

# ---------------------------------------------------------------- caption data
print("\nvalues for caption / Section 3.2:")
cm = corr.drop(index=TARGET, columns=TARGET).abs()
pairs = (cm.where(np.triu(np.ones(cm.shape), 1).astype(bool)).stack()
         .sort_values(ascending=False))
print("  feature-feature pairs: %d total, %d with |r|>0.9, %d with |r|>0.8"
      % (len(pairs), (pairs > 0.9).sum(), (pairs > 0.8).sum()))
for (a, b), v in pairs.head(8).items():
    print("    %-26s / %-26s |r| = %.3f" % (short(a), short(b), v))
print("  strongest |r| with target: %.3f (%s)"
      % (abs(corr[TARGET].drop(TARGET)).max(),
         short(abs(corr[TARGET].drop(TARGET)).idxmax())))
print("  n features with |r|>0.5 against target: %d"
      % (abs(corr[TARGET].drop(TARGET)) > 0.5).sum())
