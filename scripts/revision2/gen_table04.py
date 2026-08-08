"""Rebuild Table 4 so that it fits the 31 pc text block and reports effect sizes.

The previous layout placed seven numeric columns plus a long model name on one
row, which overflowed the page (Reviewer #1, second round). It also led with
t statistics and uncorrected p values, whereas Reviewer #1 asked in the first
round for effect sizes and confidence intervals to carry the argument.

The table is therefore transposed: one block per metric, one row per comparator,
and the columns are the paired mean difference, its 95% confidence interval,
Cohen's d_z, and the uncorrected and Bonferroni-corrected p values. Every number
is read straight from results/paper_table4_corrected_stats.csv, so the table
cannot drift from the analysis output.
"""
import os
import sys

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from paths import TABLE_OUT, results_file

SRC = results_file("paper_table4_corrected_stats.csv")
OUT = os.path.join(TABLE_OUT, "table04_ttest.tex")

SHORT = {
    "XGBoost (Default, Tuned)": "XGBoost (Default Features, Tuned)$^{*}$",
    "XGBoost (Extra Features, Tuned)": "XGBoost (Extra Features, Tuned)",
    "Linear Regression": "Linear Regression",
    "RuleFit (Default)": "RuleFit (Default)",
    "Few-Shot KNN": "Few-Shot KNN",
    "DNN Small (32-16)": "DNN Small (32--16)",
}
ORDER = list(SHORT)
METRICS = [("R2", "$R^2$  (positive difference favours RuleFit)"),
           ("MAE", "MAE, GPa  (negative difference favours RuleFit)"),
           ("RMSE", "RMSE, GPa  (negative difference favours RuleFit)")]


def num(x, digits=3, signed=True):
    fmt = "%+.*f" if signed else "%.*f"
    return (fmt % (digits, x)).replace("-", "$-$")


def pval(p):
    if p < 0.001:
        return "$<$0.001"
    if p >= 0.9995:
        return "1.000"
    return "%.3f" % p


df = pd.read_csv(SRC)
lines = [
    r"\begin{table}[h]",
    r"\footnotesize",
    r"\centering",
    r"\renewcommand{\arraystretch}{1.15}",
    r"\setlength{\tabcolsep}{4pt}",
    r"\caption{PLACEHOLDER}",
    r"\label{tab:ttest}",
    r"\begin{tabular}{@{}lrcrrr@{}}",
    r"\hline",
    r"\textbf{Comparator} & \textbf{Mean diff.} & \textbf{95\% CI} & "
    r"\textbf{$d_z$} & \textbf{$p$} & \textbf{$p_{\mathrm{Bonf}}$} \\ \hline",
]

for metric, header in METRICS:
    block = df[df["Metric"] == metric].set_index("Comparator")
    lines.append(r"\multicolumn{6}{@{}l}{\textit{%s}} \\[1pt]" % header)
    for key in ORDER:
        r = block.loc[key]
        cells = [
            SHORT[key],
            num(r["MeanDiff"]),
            "[%s, %s]" % (num(r["CI95_lo"], signed=False),
                          num(r["CI95_hi"], signed=False)),
            num(r["Cohens_dz"], 2),
            pval(r["p_one_tailed"]),
            pval(r["p_bonferroni"]),
        ]
        if str(r["sig_bonf_0.05"]) == "True":
            # bold each cell separately; \textbf cannot span an alignment tab
            cells = [r"\textbf{%s}" % c for c in cells]
        lines.append(" & ".join(cells) + r" \\")
    # \noalign works in a plain tabular; \addlinespace would require booktabs
    lines.append(r"\hline" if metric == METRICS[-1][0] else r"\noalign{\vskip 3pt}")

lines += [
    r"\end{tabular}",
    r"\vspace{1mm}",
    r"\end{tabular_note_placeholder}",
]
lines[-1] = (
    r"\scriptsize{One-tailed paired $t$-tests of RuleFit (Tuned) against each "
    r"comparator over the same 30 random 70/30 partitions ($n=30$ paired "
    r"observations per test). Mean diff.\ is the paired mean of "
    r"RuleFit\,$-$\,comparator. $p_{\mathrm{Bonf}}$ applies Bonferroni correction "
    r"across all 18 tests. Bold rows remain significant after correction. "
    r"$^{*}$Reimplemented XGBoost baseline following the published "
    r"state-of-the-art strategy of Qadir et al.~\cite{qadir2024predicting}.}"
)
lines.append(r"\end{table}")

with open(OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")

print("wrote", OUT)
print("\nwidest data row, character count:")
for key in ORDER:
    r = df[(df["Metric"] == "R2") & (df["Comparator"] == key)].iloc[0]
    print("  %-42s %d chars" % (SHORT[key], len(SHORT[key])))
print("\ncolumns: Comparator | Mean diff. | 95%% CI | d_z | p | p_Bonf  (6 vs previous 8)")
sig = df[df["sig_bonf_0.05"].astype(str) == "True"]
print("significant after Bonferroni: %d of %d tests -> %s"
      % (len(sig), len(df), ", ".join(sorted(set(sig["Comparator"])))))
