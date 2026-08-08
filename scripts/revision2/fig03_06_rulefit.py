"""Figures 3 and 6: the two components of the fitted RuleFit model.

Figure 3 shows the retained linear terms, Figure 6 the full sparse model
(linear terms plus Boolean rules). Both are drawn at the final printed width.

The substantive change with respect to the previous version is that the rule
thresholds are now printed in physical units. RuleFit was fitted on standardised
training data, so the thresholds stored in the released rule table are z-scores
of the training partition; inverting that scaling turns "Graphene (wt. %) >
-0.059" into "Graphene > 2.63 wt%". The model itself is untouched.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib.pyplot as plt
from fig_style import TEXT_W_IN, apply_style, save
from rule_units import DISPLAY, label, physical_rule_table

from paths import FIG_OUT as OUT, TABLE_OUT
POS, NEG = "#2166ac", "#b2182b"

apply_style()
rules = physical_rule_table()
rules.to_csv(os.path.join(TABLE_OUT, "rulefit_rules_physical_units.csv"), index=False)
linear = rules[rules["type"] == "linear"].sort_values("importance")
boolean = rules[rules["type"] == "rule"]

# ------------------------------------------------------------------ Figure 3
fig, ax = plt.subplots(figsize=(TEXT_W_IN, 1.95), layout="constrained")
pos = np.arange(len(linear))
ax.barh(pos, linear["importance"], height=0.6, linewidth=0.4, edgecolor="black",
        color=[POS if c > 0 else NEG for c in linear["coef"]], alpha=0.9)
for p, (_, r) in zip(pos, linear.iterrows()):
    ax.text(r["importance"] + 0.015, p, "importance %.3f,  coefficient %+.3f"
            % (r["importance"], r["coef"]), va="center", fontsize=6.8)
ax.set_yticks(pos)
ax.set_yticklabels([DISPLAY.get(n, (n, ""))[0] for n in linear["rule"]], fontsize=7.5)
ax.set_xlim(0, 1.9)
ax.set_xlabel("RuleFit linear-term importance,  $|\\beta_j| \\cdot \\mathrm{std}(x_j)$")
ax.grid(axis="x", linestyle="--", alpha=0.3)
ax.set_axisbelow(True)
for side in ("top", "right", "left"):
    ax.spines[side].set_visible(False)
handles = [plt.Rectangle((0, 0), 1, 1, fc=POS, alpha=0.9),
           plt.Rectangle((0, 0), 1, 1, fc=NEG, alpha=0.9)]
ax.legend(handles, ["increases hardness", "decreases hardness"],
          loc="lower right", frameon=False, fontsize=6.8, handlelength=1.1)
save(fig, "fig03_rulefit_linear_terms", OUT)

# ------------------------------------------------------------------ Figure 6
# Rule descriptions are far too long to serve as tick labels, so each rule is
# written above its own bar and the full plot width is available for the text.
top = rules.nlargest(12, "importance").sort_values("importance")
fig, ax = plt.subplots(figsize=(TEXT_W_IN, 4.6), layout="constrained")
pos = np.arange(len(top))
ax.barh(pos, top["importance"], height=0.34, linewidth=0.4, edgecolor="black",
        color=[POS if c > 0 else NEG for c in top["coef"]], alpha=0.9)
for p, (_, r) in zip(pos, top.iterrows()):
    ax.text(0.004, p + 0.27, label(r["rule"], r["type"]), va="bottom", ha="left",
            fontsize=6.3)
    note = ("coefficient %+.3f" % r["coef"] if r["type"] == "linear"
            else "coefficient %+.3f,  support %.2f" % (r["coef"], r["support"]))
    ax.text(r["importance"] + 0.015, p, note, va="center", ha="left", fontsize=6.3)
ax.set_yticks([])
ax.set_ylim(-0.6, len(top) - 0.25)
ax.set_xlim(0, 1.30)
ax.set_xlabel("Term importance")
ax.grid(axis="x", linestyle="--", alpha=0.3)
ax.set_axisbelow(True)
for side in ("top", "right", "left"):
    ax.spines[side].set_visible(False)
ax.legend(handles, ["increases hardness", "decreases hardness"],
          loc="lower right", frameon=False, fontsize=6.8, handlelength=1.1)
save(fig, "fig06_rulefit_rules", OUT)

# ------------------------------------------------------------------ caption data
total = rules["importance"].sum()
print("\nvalues for captions / Section 3.2:")
# NOTE: the released rule table stores the twenty highest-importance non-zero
# terms of the fitted model, so every share below is a share of those twenty.
print("  released rule table: %d linear terms + %d Boolean rules (top 20 by importance)"
      % (len(linear), len(boolean)))
print("  within those twenty, linear terms carry %.1f%% of the importance, rules %.1f%%"
      % (100 * linear["importance"].sum() / total,
         100 * boolean["importance"].sum() / total))
print("  rule support: min %.3f, max %.3f, mean %.3f"
      % (boolean["support"].min(), boolean["support"].max(), boolean["support"].mean()))
print("  strongest three terms carry %.1f%% of the total importance"
      % (100 * rules.nlargest(3, "importance")["importance"].sum() / total))
print("\n  physical-unit rules quoted in the text:")
for _, r in rules.nlargest(6, "importance").iterrows():
    print("    coef %+7.3f  supp %.3f  %s" % (r["coef"], r["support"],
                                              r["rule_physical_units"]))
