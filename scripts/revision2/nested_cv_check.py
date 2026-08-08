"""Leakage-free hyperparameter-selection robustness check.

This script does NOT replace the protocol used in the manuscript. The published
head-to-head comparison deliberately reproduces the evaluation protocol of the
baseline study so that the two works remain directly comparable; changing that
protocol would itself introduce a bias against the original method.

What this adds is an independent check: the RuleFit grid is re-selected inside
each training partition only, using an inner 5-fold cross-validation, and the
held-out test partition of that split is touched exactly once, for the final
score. Running it on all 30 repeated partitions gives a selection-bias-free
counterpart to Table 3, so the reader can see how much of the reported
performance depends on the selection step.
"""
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from data_pipeline import feature_matrix
from rulefit import RuleFit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler

from paths import TABLE_OUT as OUT
GRID = [(ts, mr) for ts in (2, 3, 4, 6, 8) for mr in (50, 100, 200, 500)]
PUBLISHED = (6, 500)


def fit_score(ts, mr, A, ytr, B, yte, names):
    m = RuleFit(tree_size=ts, max_rules=mr, rfmode="regress", random_state=42)
    m.fit(A, ytr, feature_names=names)
    p = m.predict(B)
    return (r2_score(yte, p), mean_absolute_error(yte, p),
            float(np.sqrt(mean_squared_error(yte, p))))


def select_on_training_only(A, ytr, names, n_folds=5):
    """Grid search scored purely inside the training partition."""
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    best, best_score = None, -np.inf
    table = []
    for ts, mr in GRID:
        scores = []
        for tr_idx, va_idx in kf.split(A):
            try:
                r2, _, _ = fit_score(ts, mr, A[tr_idx], ytr[tr_idx],
                                     A[va_idx], ytr[va_idx], names)
            except Exception:
                r2 = np.nan
            scores.append(r2)
        mean_score = float(np.nanmean(scores))
        table.append({"tree_size": ts, "max_rules": mr, "cv_R2": mean_score})
        if mean_score > best_score:
            best, best_score = (ts, mr), mean_score
    return best, best_score, pd.DataFrame(table)


X, y = feature_matrix()
names = list(X.columns)

# ------------------------------------------------------------------ fixed split
print("=== fixed 70/30 split (random_state=42) ===", flush=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)
sc = StandardScaler()
A, B = sc.fit_transform(X_tr), sc.transform(X_te)
cfg, cv_r2, grid_table = select_on_training_only(A, y_tr.values, names)
grid_table.to_csv(os.path.join(OUT, "nested_cv_grid_fixed_split.csv"), index=False)
fixed_nested = fit_score(cfg[0], cfg[1], A, y_tr.values, B, y_te.values, names)
fixed_published = fit_score(PUBLISHED[0], PUBLISHED[1], A, y_tr.values, B, y_te.values, names)
print("  training-only selection -> tree_size=%d, max_rules=%d (inner CV R2 = %.4f)"
      % (cfg[0], cfg[1], cv_r2))
print("  nested-CV configuration on held-out test: R2=%.4f MAE=%.4f RMSE=%.4f" % fixed_nested)
print("  published configuration (6, 500)        : R2=%.4f MAE=%.4f RMSE=%.4f" % fixed_published,
      flush=True)

# ------------------------------------------------------------ 30 repeated splits
print("\n=== 30 repeated 70/30 splits, selection inside each training set ===",
      flush=True)
rows = []
t0 = time.time()
for i in range(30):
    Xtr_i, Xte_i, ytr_i, yte_i = train_test_split(X, y, test_size=0.3, random_state=i)
    s = StandardScaler()
    Ai, Bi = s.fit_transform(Xtr_i), s.transform(Xte_i)
    cfg_i, cv_i, _ = select_on_training_only(Ai, ytr_i.values, names)
    r2, mae, rmse = fit_score(cfg_i[0], cfg_i[1], Ai, ytr_i.values, Bi, yte_i.values, names)
    r2_p, mae_p, rmse_p = fit_score(PUBLISHED[0], PUBLISHED[1], Ai, ytr_i.values,
                                    Bi, yte_i.values, names)
    rows.append({"split": i, "tree_size": cfg_i[0], "max_rules": cfg_i[1],
                 "inner_cv_R2": cv_i, "R2": r2, "MAE": mae, "RMSE": rmse,
                 "R2_published_cfg": r2_p, "MAE_published_cfg": mae_p,
                 "RMSE_published_cfg": rmse_p})
    print("  split %2d/30  selected (%d, %3d)  nested R2=%.4f  published-cfg R2=%.4f  [%.0fs]"
          % (i + 1, cfg_i[0], cfg_i[1], r2, r2_p, time.time() - t0), flush=True)

df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, "nested_cv_30splits.csv"), index=False)

summary = {
    "nested_R2_mean": float(df["R2"].mean()), "nested_R2_sd": float(df["R2"].std()),
    "nested_MAE_mean": float(df["MAE"].mean()), "nested_MAE_sd": float(df["MAE"].std()),
    "nested_RMSE_mean": float(df["RMSE"].mean()), "nested_RMSE_sd": float(df["RMSE"].std()),
    "published_cfg_R2_mean": float(df["R2_published_cfg"].mean()),
    "published_cfg_R2_sd": float(df["R2_published_cfg"].std()),
    "published_cfg_MAE_mean": float(df["MAE_published_cfg"].mean()),
    "times_published_cfg_selected": int(((df["tree_size"] == PUBLISHED[0])
                                         & (df["max_rules"] == PUBLISHED[1])).sum()),
    "fixed_split_nested": dict(zip(("R2", "MAE", "RMSE"), fixed_nested)),
    "fixed_split_published_cfg": dict(zip(("R2", "MAE", "RMSE"), fixed_published)),
    "fixed_split_selected_cfg": {"tree_size": cfg[0], "max_rules": cfg[1]},
}
with open(os.path.join(OUT, "nested_cv_summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print("\n=== SUMMARY ===")
print("  nested-CV over 30 splits : R2 = %.4f +/- %.4f, MAE = %.4f +/- %.4f"
      % (summary["nested_R2_mean"], summary["nested_R2_sd"],
         summary["nested_MAE_mean"], summary["nested_MAE_sd"]))
print("  published cfg, same splits: R2 = %.4f +/- %.4f"
      % (summary["published_cfg_R2_mean"], summary["published_cfg_R2_sd"]))
print("  (6, 500) selected by inner CV in %d of 30 splits"
      % summary["times_published_cfg_selected"])
print("  selection frequency:")
print(df.groupby(["tree_size", "max_rules"]).size().sort_values(ascending=False).to_string())
