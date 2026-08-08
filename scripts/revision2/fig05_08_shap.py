"""Figures 5 and 8: global and local SHAP explanations of the XGBoost model.

The explained model is the fixed-configuration XGBoost regressor used for the
interpretability analysis (n_estimators = 300, max_depth = 5, learning_rate =
0.05, subsample = colsample_bytree = 0.8), trained on the training partition of
the fixed 70/30 split. Its SHAP values reproduce the released
results/paper_shap_details.json exactly, which is asserted below.

Both figures are drawn at the final printed width so that no label is scaled
down by the LaTeX \\includegraphics call.
"""
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib.pyplot as plt
import shap
import xgboost as xgb
from data_pipeline import feature_matrix
from fig_style import TEXT_W_IN, apply_style, save, short
from sklearn.model_selection import train_test_split

from paths import FIG_OUT as OUT, results_file

DETAILS = results_file("paper_shap_details.json")
PARAMS = dict(n_estimators=300, max_depth=5, learning_rate=0.05,
              subsample=0.8, colsample_bytree=0.8)

apply_style()
X, y = feature_matrix()
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)
model = xgb.XGBRegressor(objective="reg:squarederror", random_state=42, **PARAMS)
model.fit(X_tr, y_tr)
pred = model.predict(X_te)
explainer = shap.TreeExplainer(model)
sv = explainer.shap_values(X_te)
base = float(explainer.expected_value)

stored = json.load(open(DETAILS, encoding="utf-8"))
assert abs(base - stored[0]["shap_base_value"]) < 5e-4, "base value drifted"
cols = list(X_te.columns)
for s in stored:
    assert abs(pred[s["test_idx"]] - s["xgb_pred_GPa"]) < 5e-3, "prediction drifted"
    for f in s["top5_shap"]:
        assert abs(sv[s["test_idx"], cols.index(f["feature"])] - f["shap_value"]) < 5e-4
print("reproduction check passed: base = %.4f GPa, %d test samples" % (base, len(X_te)))

# ------------------------------------------------------------------ Figure 5
mean_abs = pd.Series(np.abs(sv).mean(axis=0), index=cols).sort_values(ascending=False)
n_show = 12
sel = list(mean_abs.head(n_show).index)[::-1]

fig, (axl, axr) = plt.subplots(
    1, 2, figsize=(TEXT_W_IN, 3.6), width_ratios=[1.0, 1.35], layout="constrained")

pos = np.arange(len(sel))
share = 100 * mean_abs[sel] / mean_abs.sum()
axl.barh(pos, mean_abs[sel], height=0.62, color="#4a6fa5", alpha=0.9,
         edgecolor="black", linewidth=0.4)
for p, f in zip(pos, sel):
    axl.text(mean_abs[f] + 0.02, p, "%.3f  (%.1f%%)" % (mean_abs[f], share[f]),
             va="center", fontsize=6.2)
axl.set_yticks(pos)
axl.set_yticklabels([short(f) for f in sel], fontsize=6.6)
axl.set_xlim(0, 2.05)
axl.set_xlabel("mean $|$SHAP$|$ (GPa)", fontsize=7.5)
axl.tick_params(labelsize=7, length=2)
axl.grid(axis="x", linestyle="--", alpha=0.3)
axl.set_axisbelow(True)
for side in ("top", "right", "left"):
    axl.spines[side].set_visible(False)
axl.set_title("(a) global importance", fontsize=8.0, loc="left", pad=3)

rng = np.random.default_rng(0)
for p, f in enumerate(sel):
    j = cols.index(f)
    vals = sv[:, j]
    fv = X_te[f].values.astype(float)
    rank = (np.argsort(np.argsort(fv)) / max(len(fv) - 1, 1))
    axr.scatter(vals, p + rng.uniform(-0.22, 0.22, len(vals)), c=rank,
                cmap="coolwarm", s=6, linewidths=0.15, edgecolor="black", alpha=0.9)
axr.axvline(0, color="black", linewidth=0.6)
axr.set_yticks(pos)
axr.set_yticklabels([])
axr.set_ylim(-0.7, len(sel) - 0.3)
axr.set_xlabel("SHAP value (GPa)", fontsize=7.5)
axr.tick_params(labelsize=7, length=2)
axr.grid(axis="x", linestyle="--", alpha=0.3)
axr.set_axisbelow(True)
for side in ("top", "right", "left"):
    axr.spines[side].set_visible(False)
axr.set_title("(b) per-sample contributions", fontsize=8.0, loc="left", pad=3)
sm = plt.cm.ScalarMappable(cmap="coolwarm")
cb = fig.colorbar(sm, ax=axr, fraction=0.035, pad=0.02, ticks=[0, 1])
cb.ax.set_yticklabels(["low", "high"], fontsize=6.8)
cb.set_label("feature value (rank)", fontsize=6.8)
cb.outline.set_linewidth(0.4)

save(fig, "fig05_shap_summary", OUT)

# ------------------------------------------------------------------ Figure 8
points = stored[:3]
fig, axes = plt.subplots(3, 1, figsize=(TEXT_W_IN, 5.5), layout="constrained")
for ax, s in zip(axes, points):
    i = s["test_idx"]
    order = np.argsort(np.abs(sv[i]))[::-1][:6]
    top = pd.Series(sv[i, order], index=[cols[k] for k in order])
    # plotted bottom-to-top: the pooled remainder first, then increasing magnitude
    contrib = pd.concat([
        pd.Series({"all remaining features": float(sv[i].sum() - top.sum())}),
        top[::-1],
    ])
    pos = np.arange(len(contrib))
    ax.barh(pos, contrib.values, height=0.6, alpha=0.9, linewidth=0.4,
            edgecolor="black",
            color=["#b2182b" if v < 0 else "#2166ac" for v in contrib.values])
    for p, v in zip(pos, contrib.values):
        ax.text(v + (0.06 if v > 0 else -0.06), p, "%+.3f" % v, va="center",
                ha="left" if v > 0 else "right", fontsize=6.2)
    ax.set_yticks(pos)
    ax.set_yticklabels([short(f) if f in X.columns else f for f in contrib.index],
                       fontsize=6.4)
    ax.axvline(0, color="black", linewidth=0.6)
    ax.set_xlim(-4.0, 4.6)
    ax.tick_params(labelsize=7, length=2)
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.set_title(
        "P%d — measured %.1f GPa,  XGBoost %.3f GPa,  RuleFit %.3f GPa\n"
        "base value %.3f GPa,  total shift %+.3f GPa"
        % (s["point"], s["true_GPa"], s["xgb_pred_GPa"], s["rf_pred_GPa"],
           base, s["xgb_pred_GPa"] - base),
        fontsize=7.2, loc="left", pad=3)
axes[-1].set_xlabel("SHAP contribution to the XGBoost prediction (GPa)", fontsize=7.5)

save(fig, "fig08_shap_local", OUT)

# ------------------------------------------------------------------ caption data
print("\nvalues for captions / Section 3.2 and 3.4:")
total = mean_abs.sum()
for f in mean_abs.head(6).index:
    j = cols.index(f)
    print("  %-28s mean|SHAP| = %.4f GPa (%.1f%%), range [%+.3f, %+.3f]"
          % (short(f), mean_abs[f], 100 * mean_abs[f] / total,
             sv[:, j].min(), sv[:, j].max()))
print("  top-3 share = %.1f%% ; top-5 share = %.1f%%"
      % (100 * mean_abs.head(3).sum() / total, 100 * mean_abs.head(5).sum() / total))
for s in stored[:3]:
    i = s["test_idx"]
    top2 = np.sort(np.abs(sv[i]))[::-1][:2].sum()
    print("  P%d: total shift %+.3f GPa, two largest terms account for %.1f%% of |shift|"
          % (s["point"], s["xgb_pred_GPa"] - base,
             100 * top2 / max(abs(sv[i]).sum(), 1e-9)))
