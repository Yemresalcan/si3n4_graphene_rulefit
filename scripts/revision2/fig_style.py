"""Shared plotting style for Ceramics International revision 2.

Key point addressed by Reviewer #1: figures were previously drawn on 13-17 inch
canvases and then scaled down to the 131 mm (31 pc) single-column text block of
the journal template, shrinking every label by a factor of 2.7-3.4. All figures
are now drawn at their FINAL printed width so that no down-scaling occurs and
the smallest type in any figure stays at or above 7 pt.
"""
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Greek letters appear in several feature names; the Windows console default
# code page cannot encode them.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# sn-jnl single-column text block: text={31pc, 194.25mm}
TEXT_W_IN = 31 * 12 / 72.27  # 31 pc -> 5.147 in (131 mm)
TEXT_H_IN = 194.25 / 25.4    # 7.648 in

RC = {
    "font.family": "DejaVu Sans",
    "font.size": 8.0,
    "axes.titlesize": 9.0,
    "axes.titleweight": "bold",
    "axes.labelsize": 8.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5,
    "figure.titlesize": 9.5,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "lines.linewidth": 1.0,
    "grid.linewidth": 0.4,
    # NOTE: deliberately NOT 'tight'. A tight bounding box silently enlarges the
    # canvas past the 131 mm text block, which is exactly how the previous
    # figures ended up being down-scaled by the LaTeX \includegraphics call.
    "savefig.bbox": None,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}

# Consistent model colours across every figure in the paper.
MODEL_COLOURS = {
    "XGBoost (Default, Tuned)": "#3b7dd8",
    "XGBoost (Extra Features, Tuned)": "#7ab0f0",
    "RuleFit (Tuned)": "#2ca05a",
    "RuleFit (Default)": "#8fd0a8",
    "Few-Shot KNN": "#e8a33d",
    "DNN Small (32-16)": "#d9534f",
    "Linear Regression": "#9aa0a6",
}

# Compact display names, used so that axis labels stay readable at 7-8 pt.
SHORT_NAMES = {
    "Si3N4 (wt%)": "Si$_3$N$_4$ wt%",
    "α content in Si3N4 starting powder (%)": "α powder",
    " β content in Si3N4 starting powder (%)": "β powder",
    "β content in Si3N4 starting powder (%)": "β powder",
    "Initial particle size of starting Si3N4 powder (μm)": "Particle size",
    "Graphene (wt. %)": "Graphene wt%",
    "Thickness of graphene (nm)": "GNP thickness",
    "Surface Area or Diameter or lateral size of graphene (um)": "GNP lateral size",
    "Content of sintering additive 1 (wt.% or vol.%)": "Additive 1 wt%",
    "Content of sintering additive 2 (wt.% or vol.%)": "Additive 2 wt%",
    "Milling time (hour)": "Milling time",
    "Sintering Temperature (°C)": "Sint. temperature",
    "Sintering Time (min)": "Sint. time",
    "Sintering Pressure (MPa)": "Sint. pressure",
    "Density (%)": "Density",
    "α-Si3N4 content (%)": "α post-sint.",
    "β-Si3N4 Content (%)": "β post-sint.",
    "Load (N)": "Indentation load",
    "Type of Graphene_encoded": "GNP type (enc.)",
    "Type of Sintering additive 1_encoded": "Additive 1 type (enc.)",
    "Type of Sintering additive 2_encoded": "Additive 2 type (enc.)",
    "Milling type_encoded": "Milling type (enc.)",
    "Sintering Technique_encoded": "Sint. technique (enc.)",
    "graphene_aspect_ratio": "GNP aspect ratio*",
    "graphene_volume_proxy": "GNP volume proxy*",
    "sintering_energy": "Sint. energy*",
    "sintering_intensity": "Sint. intensity*",
    "sintering_dose": "Sint. dose*",
    "total_additive": "Total additive*",
    "additive_ratio": "Additive ratio*",
    "si3n4_to_additive": "Si$_3$N$_4$/additive*",
    "Vickers hardness (GPa)": "VICKERS HARDNESS",
}


def short(name):
    return SHORT_NAMES.get(name, name)


def apply_style():
    plt.rcParams.update(RC)


def save(fig, stem, outdir):
    """Write both a vector PDF (typesetter source) and a 600 dpi PNG."""
    import os

    for ext, kw in ((".pdf", {}), (".png", {"dpi": 600})):
        fig.savefig(os.path.join(outdir, stem + ext), **kw)
    w, h = fig.get_size_inches()
    plt.close(fig)
    scale = w / TEXT_W_IN
    print("wrote %s.pdf / %s.png  canvas %.2f x %.2f in (%.0f%% of text width)"
          % (stem, stem, w, h, 100 * scale))
    if scale > 1.001:
        print("  WARNING: wider than the text block; labels would be shrunk by "
              "%.2fx when placed at \\textwidth" % scale)
