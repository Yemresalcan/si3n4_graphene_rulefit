"""
Task 5: SHAP Waterfall plots for P1, P2, P3
Task 6: RuleFit rule extraction figure + discussion text
"""
import warnings, os
warnings.filterwarnings('ignore')
os.makedirs('results', exist_ok=True)

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
import xgboost as xgb
import shap

# ── 1. Load & prepare data (same pipeline as paper) ─────────────────────────
df = pd.read_excel('data.xlsx')
target_col = 'Vickers hardness (GPa)'
y = df[target_col].values

cat_cols = [c for c in df.columns if df[c].dtype == object]
df_enc = df.copy()
le_map = {}
for c in cat_cols:
    le = LabelEncoder()
    df_enc[c] = le.fit_transform(df[c].astype(str))
    le_map[c] = le

X_raw = df_enc.drop(columns=[target_col]).apply(pd.to_numeric, errors='coerce').fillna(0)
feature_names_orig = list(X_raw.columns)

# Feature engineering
def safe(col_hint):
    matches = [c for c in X_raw.columns if col_hint.lower() in c.lower()]
    return X_raw[matches[0]].values.astype(float) if matches else np.zeros(len(X_raw))

g   = safe('graphene (wt')
asp = safe('aspect ratio')
lth = safe('length')
dia = safe('diameter')
tmp = safe('sintering temperature')
prs = safe('sintering pressure')
tim = safe('sintering time')
a1  = safe('additive 1')
a2  = safe('additive 2')
si  = safe('si3n4')

ar    = np.where(dia > 0, lth / (dia + 1e-9), asp)
gvp   = g * ar
se    = tmp * prs * tim
si_f  = tmp * prs
sd    = tmp * tim
ta    = a1 + a2
ratio = np.where((ta + si) > 0, ta / (ta + si + 1e-9), np.zeros(len(X_raw)))
s2a   = np.where(ta > 0, si / (ta + 1e-9), si)

fe_names = ['graphene_aspect_ratio', 'graphene_volume_proxy',
            'sintering_energy', 'sintering_intensity',
            'sintering_dose', 'total_additive',
            'additive_ratio', 'si3n4_to_additive']
fe_vals  = [ar, gvp, se, si_f, sd, ta, ratio, s2a]

X_fe = np.hstack([X_raw.values] + [v.reshape(-1,1) for v in fe_vals])
all_feat_names = feature_names_orig + fe_names

# Fixed split (same random_state=42 as paper)
X_tr, X_te, y_tr, y_te = train_test_split(X_fe, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_tr_sc = scaler.fit_transform(X_tr)
X_te_sc  = scaler.transform(X_te)

# Train XGBoost (tuned params from paper)
xgb_model = xgb.XGBRegressor(
    n_estimators=300, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    random_state=42, verbosity=0
)
xgb_model.fit(X_tr_sc, y_tr)

# ── 2. SHAP waterfall for P1, P2, P3 (using saved values from paper) ────────
print("Building SHAP waterfall from saved data...")

with open('results/paper_shap_details.json') as f:
    shap_details = json.load(f)

# Feature name cleanup
def clean_feat(name):
    return (name.replace('Type of Graphene_encoded', 'Graphene Type (enc.)')
                .replace('Sintering Pressure (MPa)', 'Sint. Pressure (MPa)')
                .replace('Sintering Temperature', 'Sint. Temp.')
                .replace('graphene_volume_proxy', 'Graphene Vol. Proxy')
                .replace('graphene_aspect_ratio', 'Graphene Aspect Ratio')
                .replace('sintering_intensity', 'Sint. Intensity')
                .replace('sintering_dose', 'Sint. Dose')
                .replace('sintering_energy', 'Sint. Energy')
                .replace('si3n4_to_additive', 'Si3N4/Additive')
                .replace('total_additive', 'Total Additive'))

fig, axes = plt.subplots(1, 3, figsize=(18, 7))
fig.subplots_adjust(left=0.05, right=0.97, top=0.86, bottom=0.10, wspace=0.55)

for ax, pt in zip(axes, shap_details[:3]):
    pnum = pt['point']
    base = pt['shap_base_value']
    xgb_pred = pt['xgb_pred_GPa']
    true_val  = pt['true_GPa']
    rf_pred   = pt['rf_pred_GPa']

    # Top-5 saved + "Others" remainder
    top5 = pt['top5_shap']
    top5_sum = sum(s['shap_value'] for s in top5)
    others_val = (xgb_pred - base) - top5_sum

    features = [clean_feat(s['feature']) for s in top5] + ['Others']
    values   = [s['shap_value'] for s in top5] + [others_val]

    # Sort by absolute value descending (keep Others at bottom)
    order = sorted(range(5), key=lambda i: abs(values[i]), reverse=True)
    feat_sorted = [features[i] for i in order] + ['Others']
    val_sorted  = [values[i]  for i in order] + [others_val]

    colors = ['#d73027' if v < 0 else '#4575b4' for v in val_sorted]
    colors[-1] = '#aaaaaa'  # Others = grey

    y_pos = np.arange(len(feat_sorted))
    bars  = ax.barh(y_pos, val_sorted, color=colors,
                    edgecolor='white', height=0.65, alpha=0.88)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(feat_sorted, fontsize=9.5)
    ax.axvline(0, color='black', linewidth=0.9)
    ax.set_xlabel('SHAP value (GPa)', fontsize=10)
    ax.set_title(
        f'P{pnum}  |  True: {true_val} GPa\n'
        f'XGB: {xgb_pred} GPa   RF: {rf_pred} GPa   Base: {base} GPa',
        fontsize=9.5, fontweight='bold', pad=6)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    # Annotate (inside bar if too close to edge)
    xlim = ax.get_xlim()
    xrange = xlim[1] - xlim[0]
    for bar, val in zip(bars, val_sorted):
        sign = '+' if val >= 0 else ''
        if val >= 0:
            offset = xrange * 0.02
            ha = 'left'
            txt_x = val + offset
        else:
            offset = xrange * 0.02
            ha = 'right'
            txt_x = val - offset
        ax.text(txt_x, bar.get_y() + bar.get_height()/2,
                f'{sign}{val:.3f}', va='center', ha=ha, fontsize=8.5)

fig.suptitle('SHAP Feature Contributions for P1, P2, P3  (XGBoost, base = 14.933 GPa)',
             fontsize=13, fontweight='bold', y=0.98)

# Shared legend
from matplotlib.patches import Patch
fig.legend(handles=[Patch(facecolor='#4575b4', label='Positive contribution (↑ hardness)'),
                    Patch(facecolor='#d73027', label='Negative contribution (↓ hardness)'),
                    Patch(facecolor='#aaaaaa', label='Others (remaining features)')],
           loc='lower center', ncol=3, fontsize=9, bbox_to_anchor=(0.5, -0.02))

out5 = 'results/shap_waterfall_p123.png'
fig.savefig(out5, dpi=180, bbox_inches='tight')
plt.close()
print(f"Saved: {out5}")

# ── 3. RuleFit rule extraction figure ───────────────────────────────────────
print("Creating RuleFit rule extraction figure...")

rules_df = pd.read_csv('results/rulefit_top_rules.csv')
print(rules_df.head(15).to_string().encode('ascii', errors='replace').decode())

# Keep top 12 by importance
top_rules = rules_df.nlargest(12, 'importance').copy()

# Clean rule labels: shorten to readable form
def clean_label(row):
    if row['type'] == 'linear':
        return f"Linear: {row['rule']}"
    # For rules: extract feature names, drop scaled thresholds
    import re
    parts = row['rule'].split(' & ')
    clean_parts = []
    for p in parts[:3]:  # max 3 conditions
        # extract feature name and direction
        m = re.match(r'(.+?)\s*(<=|>)\s*[-\d.]+', p.strip())
        if m:
            feat = m.group(1).strip()
            # shorten long names
            feat = feat.replace('Sintering Temperature (°C)', 'Sint. Temp.')
            feat = feat.replace('Sintering Pressure (MPa)', 'Sint. Pressure')
            feat = feat.replace('Type of Graphene_encoded', 'Graphene Type')
            feat = feat.replace('Graphene (wt. %)', 'Graphene %')
            feat = feat.replace('Si3N4 (wt%)', 'Si₃N₄ %')
            feat = feat.replace('Density (%)', 'Density')
            feat = feat.replace('Load (N)', 'Load')
            feat = feat.replace('Thickness of graphene (nm)', 'Graphene Thick.')
            feat = feat.replace('sintering_intensity', 'Sint. Intensity')
            feat = feat.replace('si3n4_to_additive', 'Si₃N₄/Additive')
            feat = feat.replace('total_additive', 'Total Additive')
            feat = feat.replace('graphene_aspect_ratio', 'Graphene AR')
            feat = feat.replace('graphene_volume_proxy', 'Graphene Vol.')
            clean_parts.append(f"{feat} {m.group(2)}")
        else:
            clean_parts.append(p.strip()[:25])
    label = ' & '.join(clean_parts)
    if len(parts) > 3:
        label += f' (+{len(parts)-3} more)'
    return label

top_rules['label'] = top_rules.apply(clean_label, axis=1)
top_rules['coef_sign'] = top_rules['coef'].apply(lambda x: 'Positive' if x > 0 else 'Negative')

# Sort by importance
top_rules = top_rules.sort_values('importance', ascending=True)

colors = ['#d73027' if s == 'Negative' else '#2166ac'
          for s in top_rules['coef_sign']]

fig2, ax2 = plt.subplots(figsize=(12, 7))
fig2.subplots_adjust(left=0.38, right=0.95, top=0.90, bottom=0.12)

bars = ax2.barh(range(len(top_rules)), top_rules['importance'].values,
                color=colors, edgecolor='white', height=0.65, alpha=0.85)
ax2.set_yticks(range(len(top_rules)))
ax2.set_yticklabels(top_rules['label'].values, fontsize=9.5)
ax2.set_xlabel('Rule Importance (|coef| × support)', fontsize=11)
ax2.set_title('RuleFit: Top 12 Extracted Rules\n(Blue = positive effect on hardness, Red = negative effect)',
              fontsize=12, fontweight='bold', pad=8)
ax2.axvline(0, color='black', linewidth=0.8)
ax2.grid(axis='x', alpha=0.3, linestyle='--')
ax2.set_axisbelow(True)

# Annotate support
for bar, (_, row) in zip(bars, top_rules.iterrows()):
    ax2.text(bar.get_width() + 0.005,
             bar.get_y() + bar.get_height()/2,
             f'sup={row["support"]:.2f}  coef={row["coef"]:+.3f}',
             va='center', fontsize=8, color='#333333')

from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#2166ac', label='Positive (↑ hardness)'),
                   Patch(facecolor='#d73027', label='Negative (↓ hardness)')]
ax2.legend(handles=legend_elements, loc='lower right', fontsize=9)

out6 = 'results/rulefit_rules_extracted.png'
fig2.savefig(out6, dpi=180, bbox_inches='tight')
plt.close()
print(f"Saved: {out6}")

print("\nDone.")
