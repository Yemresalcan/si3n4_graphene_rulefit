"""Single source of truth for the feature space used in the manuscript.

This mirrors, line for line, the preprocessing described in Section 2.1:
target encoding of the five categorical variables, exclusion of the reference
label, and the eight physically motivated engineered descriptors, giving a
30-feature design matrix. Every figure script imports this module so that all
figures describe the same feature space as the Methods section.
"""
import pandas as pd

from paths import DATA

TARGET = "Vickers hardness (GPa)"


def load_encoded():
    """82 x 22 target-encoded design matrix (reference label excluded)."""
    data = pd.read_excel(DATA, sheet_name="Vickers Hardness")
    if "Ref." in data.columns:
        data = data.drop(columns=["Ref."])
    text_columns = [c for c in data.columns if data[c].map(type).eq(str).any()]
    enc = data.copy()
    for col in text_columns:
        enc[col + "_encoded"] = enc[col].map(enc.groupby(col)[TARGET].mean())
    return enc.drop(columns=text_columns)


def add_features(df):
    """The eight engineered descriptors defined in Section 2.1."""
    df = df.copy()
    thickness = [c for c in df.columns if "Thickness" in c][0]
    surface = [c for c in df.columns if "Surface Area" in c][0]
    graphene = [c for c in df.columns if "Graphene (wt" in c][0]
    temp = [c for c in df.columns if "Sintering Temperature" in c][0]
    time = [c for c in df.columns if "Sintering Time" in c][0]
    press = [c for c in df.columns if "Sintering Pressure" in c][0]
    add1 = [c for c in df.columns if "additive 1" in c and "Content" in c][0]
    add2 = [c for c in df.columns if "additive 2" in c and "Content" in c][0]
    si3n4 = [c for c in df.columns if "Si3N4 (wt" in c][0]
    df["graphene_aspect_ratio"] = df[surface] / (df[thickness] + 0.01)
    df["graphene_volume_proxy"] = df[graphene] * df[thickness] * df[surface]
    df["sintering_energy"] = df[temp] * df[time]
    df["sintering_intensity"] = df[temp] * df[press]
    df["sintering_dose"] = df[temp] * df[time] * df[press]
    df["total_additive"] = df[add1] + df[add2]
    df["additive_ratio"] = df[add1] / (df[add2] + 0.01)
    df["si3n4_to_additive"] = df[si3n4] / (df["total_additive"] + 0.01)
    return df


def feature_matrix():
    """Return (X, y) with the 30-feature space used throughout the paper."""
    fe = add_features(load_encoded())
    return fe.drop(columns=[TARGET]), fe[TARGET]
