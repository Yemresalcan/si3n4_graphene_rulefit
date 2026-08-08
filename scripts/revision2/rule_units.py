"""Recover physical-unit thresholds for the published RuleFit rule set.

RuleFit was fitted on StandardScaler-transformed training data of the fixed
70/30 split, so every threshold stored in results/rulefit_top_rules.csv is a
z-score of that training partition. Inverting the scaler recovers the original
physical threshold exactly. No model is refitted and no reported metric changes;
only the presentation of the already-published rules becomes interpretable.
"""
import os
import re
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_pipeline import feature_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from paths import RESULTS, results_file

RULES_CSV = results_file("rulefit_top_rules.csv")
COND = re.compile(r"^(.*?)\s*(<=|>)\s*(-?\d+\.?\d*(?:e-?\d+)?)$")

# display name and physical unit for every variable that appears in a rule
DISPLAY = {
    "Sintering Pressure (MPa)": ("Sint. pressure", "MPa"),
    "Sintering Temperature (°C)": ("Sint. temperature", "°C"),
    "Sintering Time (min)": ("Sint. time", "min"),
    "Graphene (wt. %)": ("Graphene", "wt%"),
    "Si3N4 (wt%)": ("Si$_3$N$_4$", "wt%"),
    "Density (%)": ("Density", "%"),
    "Load (N)": ("Load", "N"),
    "Thickness of graphene (nm)": ("GNP thickness", "nm"),
    "Surface Area or Diameter or lateral size of graphene (um)": ("GNP lateral size", "µm"),
    "Milling time (hour)": ("Milling time", "h"),
    "Initial particle size of starting Si3N4 powder (μm)": ("Particle size", "µm"),
    "Content of sintering additive 1 (wt.% or vol.%)": ("Additive 1", "wt%"),
    "Content of sintering additive 2 (wt.% or vol.%)": ("Additive 2", "wt%"),
    "α-Si3N4 content (%)": ("α post-sint.", "%"),
    "β-Si3N4 Content (%)": ("β post-sint.", "%"),
    "Type of Graphene_encoded": ("GNP type (enc.)", ""),
    "Type of Sintering additive 1_encoded": ("Additive 1 type (enc.)", ""),
    "Type of Sintering additive 2_encoded": ("Additive 2 type (enc.)", ""),
    "Milling type_encoded": ("Milling type (enc.)", ""),
    "Sintering Technique_encoded": ("Sint. technique (enc.)", ""),
    "sintering_intensity": ("Sint. intensity", ""),
    "sintering_energy": ("Sint. energy", ""),
    "sintering_dose": ("Sint. dose", ""),
    "total_additive": ("Total additive", "wt%"),
    "additive_ratio": ("Additive ratio", ""),
    "si3n4_to_additive": ("Si$_3$N$_4$/additive", ""),
    "graphene_aspect_ratio": ("GNP aspect ratio", ""),
    "graphene_volume_proxy": ("GNP volume proxy", ""),
}


def _scaler():
    X, y = feature_matrix()
    X_tr, _, _, _ = train_test_split(X, y, test_size=0.3, random_state=42)
    sc = StandardScaler().fit(X_tr)
    return (pd.Series(sc.mean_, index=X.columns),
            pd.Series(sc.scale_, index=X.columns))


MEAN, STD = _scaler()


def _fmt(value):
    a = abs(value)
    if a >= 1e4:
        return "%.3g" % value
    if a >= 100:
        return "%.0f" % value
    if a >= 10:
        return "%.1f" % value
    return "%.3g" % value


def conditions(rule, math=True):
    """Split a rule into readable, physical-unit conditions."""
    out = []
    for part in rule.split(" & "):
        m = COND.match(part.strip())
        if not m:
            out.append(part.strip())
            continue
        feat, op, z = m.group(1).strip(), m.group(2), float(m.group(3))
        name, unit = DISPLAY.get(feat, (feat, ""))
        if feat in MEAN.index:
            value = MEAN[feat] + z * STD[feat]
            txt = "%s %s %s" % (name, "\u2264" if op == "<=" else ">", _fmt(value))
            if unit:
                txt += " " + unit
        else:
            txt = "%s %s %.3g" % (name, op, z)
        out.append(txt)
    return out


def label(rule, rule_type, max_conditions=3):
    if rule_type == "linear":
        name, unit = DISPLAY.get(rule, (rule, ""))
        return "linear term:  %s" % name
    parts = conditions(rule)
    text = "  &  ".join(parts[:max_conditions])
    if len(parts) > max_conditions:
        text += "  (+%d)" % (len(parts) - max_conditions)
    return text


def physical_rule_table():
    rules = pd.read_csv(RULES_CSV)
    rules["rule_physical_units"] = [
        " & ".join(conditions(r)) if t == "rule" else r
        for r, t in zip(rules["rule"], rules["type"])
    ]
    return rules


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    tbl = physical_rule_table()
    from paths import TABLE_OUT

    out = os.path.join(TABLE_OUT, "rulefit_rules_physical_units.csv")
    tbl.to_csv(out, index=False)
    for _, r in tbl.nlargest(12, "importance").iterrows():
        print("%-6s coef=%+7.3f supp=%.3f imp=%.3f  %s"
              % (r["type"], r["coef"], r["support"], r["importance"],
                 r["rule_physical_units"]))
    print("\nwrote", out)
