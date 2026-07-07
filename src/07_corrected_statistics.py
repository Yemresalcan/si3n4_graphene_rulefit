"""Corrected statistics for reviewer response:
- Bonferroni and Holm corrected p-values for the 18 paired one-tailed t-tests
- Cohen's d_z effect sizes
- 95% CI of paired differences (RuleFit Tuned minus comparator)
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import numpy as np
import pandas as pd
from scipy import stats

BASE = r'results/tables'
raw = {m: pd.read_csv(rf'{BASE}/paper_30split_{m}_raw.csv') for m in ['r2', 'mae', 'rmse']}

REF = 'RuleFit (Tuned)'
comparators = [c for c in raw['r2'].columns if c != REF]

rows = []
for metric, df in raw.items():
    for comp in comparators:
        a = df[REF].values
        b = df[comp].values
        diff = a - b
        n = len(diff)
        mean_d = diff.mean()
        sd_d = diff.std(ddof=1)
        se = sd_d / np.sqrt(n)
        tcrit = stats.t.ppf(0.975, n - 1)
        ci_lo, ci_hi = mean_d - tcrit * se, mean_d + tcrit * se
        t_stat = mean_d / se
        # one-tailed: R2 -> RuleFit higher is better (H1: diff>0); MAE/RMSE -> lower better (H1: diff<0)
        if metric == 'r2':
            p_one = stats.t.sf(t_stat, n - 1)
        else:
            p_one = stats.t.cdf(t_stat, n - 1)
        d_z = mean_d / sd_d
        rows.append({
            'Metric': metric.upper(), 'Comparator': comp, 'n': n,
            'MeanDiff': mean_d, 'CI95_lo': ci_lo, 'CI95_hi': ci_hi,
            't': t_stat, 'p_one_tailed': p_one, 'Cohens_dz': d_z,
        })

res = pd.DataFrame(rows)
m = len(res)  # 18 tests
res['p_bonferroni'] = np.minimum(res['p_one_tailed'] * m, 1.0)
# Holm
order = np.argsort(res['p_one_tailed'].values)
holm = np.empty(m)
prev = 0.0
for rank, idx in enumerate(order):
    val = min((m - rank) * res['p_one_tailed'].values[idx], 1.0)
    prev = max(prev, val)
    holm[idx] = prev
res['p_holm'] = holm
res['sig_bonf_0.05'] = res['p_bonferroni'] < 0.05

pd.set_option('display.width', 200)
print(res.round(4).to_string(index=False))
res.to_csv(rf'{BASE}/paper_table4_corrected_stats.csv', index=False)
print('\nsaved -> paper_table4_corrected_stats.csv')
