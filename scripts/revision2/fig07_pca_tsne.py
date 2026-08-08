"""Figure 7: PCA and t-SNE projections of the 30-feature space.

The previous version of this figure was produced by a stand-alone plotting
script whose preprocessing had drifted from the pipeline described in Section
2.1: it label-encoded instead of target-encoded the categorical variables, did
not drop the reference label, and used different formulas for the engineered
descriptors. In the resulting design matrix ten of the thirty-one columns were
constant, so all five categorical variables and four of the eight engineered
descriptors carried no information at all.

This script imports the same `data_pipeline` module as every other figure, so
the projection now describes exactly the 30-feature space that the Methods
section defines. The explained-variance values therefore change (PC1 32.2 % ->
28.8 %, PC2 18.6 % -> 17.5 %). This is a correction to a descriptive
visualisation: no model, split, or reported performance metric is affected.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib.pyplot as plt
from data_pipeline import feature_matrix
from fig_style import TEXT_W_IN, apply_style, save
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from paths import FIG_OUT as OUT

apply_style()
X, y = feature_matrix()
Xs = StandardScaler().fit_transform(X)

pca_full = PCA(random_state=42).fit(Xs)
ev = pca_full.explained_variance_ratio_ * 100
cum = np.cumsum(ev)
P = pca_full.transform(Xs)
T = TSNE(n_components=2, perplexity=15, max_iter=1000, random_state=42).fit_transform(Xs)

fig, axes = plt.subplots(1, 2, figsize=(TEXT_W_IN, 2.85), layout="constrained")
common = dict(c=y.values, cmap="viridis", s=14, edgecolor="black", linewidth=0.25)

sc = axes[0].scatter(P[:, 0], P[:, 1], **common)
axes[0].set_xlabel("PC1 (%.1f%% of variance)" % ev[0])
axes[0].set_ylabel("PC2 (%.1f%% of variance)" % ev[1])
axes[0].set_title("(a) PCA", fontsize=8.2, loc="left", pad=3)
axes[0].text(0.03, 0.03, "PC1+PC2 = %.1f%%" % cum[1], transform=axes[0].transAxes,
             fontsize=6.8, va="bottom", ha="left")

axes[1].scatter(T[:, 0], T[:, 1], **common)
axes[1].set_xlabel("t-SNE dimension 1")
axes[1].set_ylabel("t-SNE dimension 2")
axes[1].set_title("(b) t-SNE (perplexity = 15)", fontsize=8.2, loc="left", pad=3)

for ax in axes:
    ax.tick_params(labelsize=7, length=2)
    ax.grid(linestyle="--", alpha=0.25)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

cb = fig.colorbar(sc, ax=axes, fraction=0.038, pad=0.015)
cb.set_label("Vickers hardness (GPa)", fontsize=7.5)
cb.ax.tick_params(labelsize=7, length=2)
cb.outline.set_linewidth(0.4)

save(fig, "fig07_pca_tsne", OUT)

print("\nexplained variance (caption / Section 3.3):")
for i in range(6):
    print("  PC%-2d %5.2f%%   cumulative %5.2f%%" % (i + 1, ev[i], cum[i]))
print("  components needed for 80%%: %d ; 90%%: %d ; 95%%: %d"
      % (np.argmax(cum >= 80) + 1, np.argmax(cum >= 90) + 1, np.argmax(cum >= 95) + 1))
print("  variance NOT shown in the 2-D panel: %.1f%%" % (100 - cum[1]))
lo, hi = y < 10, y > 20
print("  n samples <10 GPa: %d ; >20 GPa: %d ; between: %d"
      % (lo.sum(), hi.sum(), len(y) - lo.sum() - hi.sum()))
