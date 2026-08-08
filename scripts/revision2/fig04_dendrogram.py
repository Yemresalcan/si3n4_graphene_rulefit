"""Figure 4: hierarchical clustering of the feature space.

Two changes with respect to the previous version:

1. Average linkage is used instead of Ward. Ward linkage minimises within-cluster
   variance and is defined for Euclidean geometry; it is not appropriate for a
   pre-computed correlation distance d = 1 - |r|. Average linkage is the standard
   choice for a supplied distance matrix, and it is what the manuscript text
   describes. This is a change to an exploratory visualisation only: no model,
   no split, and no reported performance metric depends on it.
2. The figure is drawn at the final printed width with 7 pt leaf labels.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib.pyplot as plt
from data_pipeline import TARGET, feature_matrix
from fig_style import TEXT_W_IN, apply_style, save, short
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from sklearn.model_selection import train_test_split

from paths import FIG_OUT as OUT
LINKAGE = "average"

apply_style()
X, y = feature_matrix()
# clustering is computed on the training partition of the fixed 70/30 split,
# so that no held-out information enters the exploratory structure analysis
X_tr, _, y_tr, _ = train_test_split(X, y, test_size=0.3, random_state=42)
frame = X_tr.copy()
frame[TARGET] = y_tr.values

corr = frame.corr(method="pearson")
dist = (1 - corr.abs()).clip(lower=0).values.copy()
np.fill_diagonal(dist, 0)
Z = hierarchy.linkage(np.abs(squareform(dist, checks=False)), method=LINKAGE)
names = [short(c) for c in corr.columns]

fig, ax = plt.subplots(figsize=(TEXT_W_IN, 3.9), layout="constrained")
dn = hierarchy.dendrogram(
    Z, labels=names, orientation="top", leaf_rotation=90, leaf_font_size=6.4,
    color_threshold=0.55 * Z[:, 2].max(), ax=ax,
    above_threshold_color="#666666",
)
ax.set_ylabel("Distance,  $d = 1 - |r|$")
ax.tick_params(axis="y", labelsize=7.5)
ax.grid(axis="y", linestyle="--", alpha=0.3)
ax.set_axisbelow(True)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
for lbl in ax.get_xmajorticklabels():
    if "VICKERS" in lbl.get_text():
        lbl.set_color("#8b0000")
        lbl.set_fontweight("bold")

# annotate the four earliest merges, which are the redundant feature families
merge_notes = []
for i in range(4):
    a, b = int(Z[i, 0]), int(Z[i, 1])
    merge_notes.append((names[a] if a < len(names) else "?",
                        names[b] if b < len(names) else "?", Z[i, 2]))
ax.axhline(0.1, color="#8b0000", linestyle=":", linewidth=0.8)
ax.text(ax.get_xlim()[0] + 4, 0.113, "$d < 0.1$  ($|r| > 0.9$): 8 merges",
        fontsize=6.8, color="#8b0000", ha="left", va="bottom")

save(fig, "fig04_dendrogram", OUT)

print("\nlinkage = %s ; max merge height = %.3f" % (LINKAGE, Z[:, 2].max()))
print("earliest merges (caption):")
for a, b, d in merge_notes:
    print("  %-26s + %-26s at d = %.4f" % (a, b, d))
n = len(names)
ti = names.index(short(TARGET))
for i in range(len(Z)):
    if int(Z[i, 0]) == ti or int(Z[i, 1]) == ti:
        other = int(Z[i, 1]) if int(Z[i, 0]) == ti else int(Z[i, 0])
        label = (names[other] if other < n
                 else "a cluster of %d variables" % int(Z[other - n, 3]))
        print("  target merges with %s at d = %.4f" % (label, Z[i, 2]))
        break
print("  merges below d=0.1: %d of %d" % (int((Z[:, 2] < 0.1).sum()), len(Z)))
