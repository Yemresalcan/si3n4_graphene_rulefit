"""Regenerate boxplot with y-axis 0-1, clipping unstable models."""
import warnings, os
warnings.filterwarnings('ignore')
os.makedirs('results', exist_ok=True)

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Load 30-split R² data
r2 = pd.read_csv('results/paper_30split_r2_raw.csv')
print("Columns:", list(r2.columns))
print("Shape:", r2.shape)

# Model display names (order matters)
col_order = list(r2.columns)
print("Column order:", col_order)

# Build box data
box_data   = [r2[c].values for c in col_order]
box_labels = col_order

# Color by category
category_colors = {
    'XGBoost (Default, Tuned)':        '#3498db',
    'XGBoost (Extra Features, Tuned)': '#3498db',
    'Linear Regression':               '#95a5a6',
    'RuleFit (Tuned)':                 '#2ecc71',
    'RuleFit (Default)':               '#2ecc71',
    'Few-Shot KNN':                    '#f39c12',
    'DNN Small (32-16)':               '#e74c3c',
}
default_color = '#95a5a6'

fig, ax = plt.subplots(figsize=(14, 8))
fig.subplots_adjust(bottom=0.32, top=0.90, left=0.08, right=0.97)

bp = ax.boxplot(box_data,
                labels=box_labels,
                patch_artist=True,
                vert=True,
                showfliers=True,
                flierprops=dict(marker='o', markersize=4, alpha=0.5))

for i, lbl in enumerate(box_labels):
    color = category_colors.get(lbl, default_color)
    bp['boxes'][i].set_facecolor(color)
    bp['boxes'][i].set_alpha(0.85)
    bp['medians'][i].set_color('black')
    bp['medians'][i].set_linewidth(2)

# Y-axis: 0 to 1
ax.set_ylim(0.0, 1.05)
ax.set_yticks(np.arange(0.0, 1.1, 0.1))
ax.yaxis.grid(True, linestyle='--', alpha=0.5)
ax.set_axisbelow(True)

# Reference line
ax.axhline(y=0.8, color='red', linestyle='--', linewidth=1.2, alpha=0.7, label='R² = 0.8')

ax.set_ylabel('R²', fontsize=12)
ax.set_title('R² Distributions Across 30 Random 70/30 Train–Test Splits', fontsize=13, fontweight='bold', pad=10)
ax.set_xticklabels(box_labels, rotation=40, ha='right', fontsize=10)

# Note about unstable models
ax.text(0.5, -0.30,
        '* Linear Regression (mean R² = −6.58 ± 37.56) and DNN Small (mean R² = −9.24 ± 53.43)\n'
        '  are clipped to the [0, 1] range; their distributions extend far below zero.',
        transform=ax.transAxes, ha='center', fontsize=8.5,
        style='italic', color='#555555')

# Legend
legend_elements = [
    Patch(facecolor='#3498db', label='XGBoost (Ensemble)'),
    Patch(facecolor='#2ecc71', label='RuleFit (Rule-based)'),
    Patch(facecolor='#f39c12', label='Few-Shot KNN'),
    Patch(facecolor='#e74c3c', label='DNN Small'),
    Patch(facecolor='#95a5a6', label='Linear Regression'),
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=9, framealpha=0.8)

out = 'results/boxplot_r2_distribution.png'
plt.savefig(out, dpi=180, bbox_inches='tight')
plt.close()
print(f"Saved: {out}")
