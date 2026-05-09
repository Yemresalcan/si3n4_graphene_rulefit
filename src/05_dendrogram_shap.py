"""Dendrogram, SHAP, LIME, RuleFit Case Study"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import json

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform as sp_squareform
import xgboost as xgb
from rulefit import RuleFit
import shap
import lime
import lime.lime_tabular

# --- veri yukle ---
data = pd.read_excel('data.xlsx', sheet_name='Vickers Hardness')
if 'Ref.' in data.columns:
    data = data.drop(columns=['Ref.'])
tc = 'Vickers hardness (GPa)'
encoded_data = data.copy()
text_columns = data.select_dtypes(include=['object']).columns.tolist()
for col in text_columns:
    means = encoded_data.groupby(col)[tc].mean()
    encoded_data[col + '_encoded'] = encoded_data[col].map(means)
encoded_data = encoded_data.drop(columns=text_columns)
X_orig = encoded_data.drop(columns=[tc])
y = encoded_data[tc]

def add_features(df):
    df = df.copy()
    thickness_col = [c for c in df.columns if 'Thickness' in c][0]
    surface_col   = [c for c in df.columns if 'Surface Area' in c][0]
    graphene_col  = [c for c in df.columns if 'Graphene (wt' in c][0]
    temp_col      = [c for c in df.columns if 'Sintering Temperature' in c][0]
    time_col      = [c for c in df.columns if 'Sintering Time' in c][0]
    press_col     = [c for c in df.columns if 'Sintering Pressure' in c][0]
    add1_col      = [c for c in df.columns if 'additive 1' in c and 'Content' in c][0]
    add2_col      = [c for c in df.columns if 'additive 2' in c and 'Content' in c][0]
    si3n4_col     = [c for c in df.columns if 'Si3N4 (wt' in c][0]
    df['graphene_aspect_ratio'] = df[surface_col] / (df[thickness_col] + 0.01)
    df['graphene_volume_proxy'] = df[graphene_col] * df[thickness_col] * df[surface_col]
    df['sintering_energy']      = df[temp_col] * df[time_col]
    df['sintering_intensity']   = df[temp_col] * df[press_col]
    df['sintering_dose']        = df[temp_col] * df[time_col] * df[press_col]
    df['total_additive']        = df[add1_col] + df[add2_col]
    df['additive_ratio']        = df[add1_col] / (df[add2_col] + 0.01)
    df['si3n4_to_additive']     = df[si3n4_col] / (df['total_additive'] + 0.01)
    return df

encoded_fe = add_features(encoded_data)
X_fe = encoded_fe.drop(columns=[tc])

X_tr_70, X_te_70, y_tr_70, y_te_70 = train_test_split(X_fe, y, test_size=0.3, random_state=42)
scaler_70 = StandardScaler()
X_tr_sc_70 = scaler_70.fit_transform(X_tr_70)
X_te_sc_70 = scaler_70.transform(X_te_70)

def evaluate(y_true, y_pred):
    return (round(r2_score(y_true, y_pred), 4),
            round(mean_absolute_error(y_true, y_pred), 4),
            round(np.sqrt(mean_squared_error(y_true, y_pred)), 4))

# ============================================================
# DENDROGRAM
# ============================================================
print("=== DENDROGRAM ===")
X_with_label = X_tr_70.copy()
X_with_label['Vickers Hardness (GPa)'] = y_tr_70.values
corr = X_with_label.corr(method='pearson')
dist_arr = (1 - corr.abs()).clip(lower=0).values.copy()
np.fill_diagonal(dist_arr, 0)
condensed = sp_squareform(dist_arr, checks=False)
condensed = np.abs(condensed)
linkage = hierarchy.linkage(condensed, method='ward')
col_names = list(corr.columns)
rename_map = {
    'Si3N4 (wt%)': 'Si3N4 wt%',
    'alpha content in Si3N4 starting powder (%)': 'alpha-Si3N4',
    'beta content in Si3N4 starting powder (%)': 'beta-Si3N4',
    'Initial particle size of starting Si3N4 powder (\u00b5m)': 'Particle Size',
    'Graphene (wt.%)': 'Graphene wt%',
    'Thickness of graphene (nm)': 'GNP Thickness',
    'Surface Area or Diameter of graphene (\u00b5m)': 'GNP Surface',
    'Content of sintering additive 1 (wt.% or vol.%)': 'Additive 1 wt%',
    'Content of sintering additive 2 (wt.% or vol.%)': 'Additive 2 wt%',
    'Milling time (hour)': 'Milling Time',
    'Sintering Temperature (\u00b0C)': 'Sint. Temp',
    'Sintering Time (min)': 'Sint. Time',
    'Sintering Pressure (MPa)': 'Sint. Press',
    'Density (%)': 'Density %',
    'alpha-Si3N4 content (%)': 'alpha post',
    'beta-Si3N4 Content (%)': 'beta post',
    'Load (N)': 'Load',
    'Type of Graphene_encoded': 'GNP Type',
    'Type of Sintering additive 1_encoded': 'Add1 Type',
    'Type of Sintering additive 2_encoded': 'Add2 Type',
    'Milling type_encoded': 'Milling Type',
    'Sintering Technique_encoded': 'Sint. Tech',
    'graphene_aspect_ratio': 'GNP Aspect',
    'graphene_volume_proxy': 'GNP Volume',
    'sintering_energy': 'Sint. Energy',
    'sintering_intensity': 'Sint. Intensity',
    'sintering_dose': 'Sint. Dose',
    'total_additive': 'Total Add.',
    'additive_ratio': 'Add. Ratio',
    'si3n4_to_additive': 'Si3N4/Add.',
    'Vickers Hardness (GPa)': 'HARDNESS (target)',
}
short = [rename_map.get(c, c[:18]) for c in col_names]
fig, ax = plt.subplots(figsize=(18, 8))
hierarchy.dendrogram(
    linkage, labels=short, orientation='top',
    leaf_rotation=90, leaf_font_size=7,
    color_threshold=0.55 * max(linkage[:, 2]), ax=ax
)
ax.set_title('Hierarchical Feature Clustering Dendrogram\n(red = Vickers Hardness target variable)',
             fontsize=12)
ax.set_xlabel('Features', fontsize=10)
ax.set_ylabel('Distance (1 - |Pearson r|)', fontsize=10)
for lbl in ax.get_xmajorticklabels():
    if 'HARDNESS' in lbl.get_text():
        lbl.set_color('red')
        lbl.set_fontweight('bold')
plt.tight_layout()
plt.savefig('results/paper_dendrogram.png', dpi=150, bbox_inches='tight')
plt.close()
print("Dendrogram kaydedildi.")

# ============================================================
# MODEL EGITIM (70/30)
# ============================================================
print("\n=== MODEL EGITIM ===")
xgb_params = {'n_estimators': 300, 'max_depth': 5, 'learning_rate': 0.05,
              'subsample': 0.8, 'colsample_bytree': 0.8}
xgb_main = xgb.XGBRegressor(objective='reg:squarederror', random_state=42, **xgb_params)
xgb_main.fit(X_tr_70, y_tr_70)
y_pred_xgb = xgb_main.predict(X_te_70)
r2, mae, rmse = evaluate(y_te_70, y_pred_xgb)
print(f"XGB: R2={r2}, MAE={mae}, RMSE={rmse}")

rf_main = RuleFit(tree_size=6, max_rules=500, rfmode='regress', random_state=42)
rf_main.fit(X_tr_70.values, y_tr_70.values, feature_names=list(X_te_70.columns))
y_pred_rf = rf_main.predict(X_te_70.values)
r2, mae, rmse = evaluate(y_te_70, y_pred_rf)
print(f"RuleFit: R2={r2}, MAE={mae}, RMSE={rmse}")

# 5 datapoint sec
errors = np.abs(y_te_70.values - y_pred_rf)
sorted_idx = np.argsort(errors)
n = len(sorted_idx)
sel_idx = list(sorted_idx[:2]) + list(sorted_idx[n//2 - 1:n//2 + 1]) + [sorted_idx[-1]]
print(f"Secilen indeksler: {sel_idx}")
for j, idx in enumerate(sel_idx):
    cat = 'Good' if errors[idx] < 1.5 else ('Medium' if errors[idx] < 3.5 else 'High')
    print(f"  P{j+1} idx={idx}: True={y_te_70.values[idx]:.2f}, RF={y_pred_rf[idx]:.2f}, "
          f"XGB={y_pred_xgb[idx]:.2f}, err={errors[idx]:.2f} [{cat}]")

# ============================================================
# SHAP
# ============================================================
print("\n=== SHAP ===")
explainer = shap.TreeExplainer(xgb_main)
shap_vals_all = explainer.shap_values(X_te_70)
base_val = float(explainer.expected_value)

plt.figure(figsize=(10, 7))
shap.summary_plot(shap_vals_all, X_te_70, show=False, max_display=15)
plt.title('SHAP Feature Importance (XGBoost, Test Set)', fontsize=12)
plt.tight_layout()
plt.savefig('results/paper_shap_summary.png', dpi=150, bbox_inches='tight')
plt.close()
print("SHAP summary plot kaydedildi.")

shap_details = []
for j, idx in enumerate(sel_idx):
    sv = shap_vals_all[idx]
    top5_i = np.argsort(np.abs(sv))[-5:][::-1]
    top5 = [(list(X_te_70.columns)[k], round(float(sv[k]), 4),
             round(float(X_te_70.iloc[idx, k]), 4)) for k in top5_i]
    shap_details.append({
        'point': j + 1,
        'test_idx': int(idx),
        'true_GPa': round(float(y_te_70.values[idx]), 3),
        'xgb_pred_GPa': round(float(y_pred_xgb[idx]), 3),
        'rf_pred_GPa': round(float(y_pred_rf[idx]), 3),
        'shap_base_value': round(base_val, 3),
        'top5_shap': [{'feature': f, 'shap_value': sv, 'feature_value': fv}
                      for f, sv, fv in top5],
    })
    summary = ', '.join([f"{f.split('(')[0].strip()}({sv:+.3f})" for f, sv, fv in top5[:3]])
    print(f"  P{j+1}: {summary}")

with open('results/paper_shap_details.json', 'w', encoding='utf-8') as f:
    json.dump(shap_details, f, ensure_ascii=False, indent=2)
print("SHAP details kaydedildi.")

# ============================================================
# LIME
# ============================================================
print("\n=== LIME ===")
lime_exp = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_tr_70.values,
    feature_names=list(X_tr_70.columns),
    mode='regression',
    random_state=42,
)
lime_details = []
for j, idx in enumerate(sel_idx):
    exp = lime_exp.explain_instance(
        data_row=X_te_70.iloc[idx].values,
        predict_fn=xgb_main.predict,
        num_features=5,
    )
    contribs = exp.as_list()
    lime_details.append({
        'point': j + 1,
        'test_idx': int(idx),
        'true_GPa': round(float(y_te_70.values[idx]), 3),
        'xgb_pred_GPa': round(float(y_pred_xgb[idx]), 3),
        'rf_pred_GPa': round(float(y_pred_rf[idx]), 3),
        'top5_lime': [{'condition': c, 'contribution': round(v, 4)} for c, v in contribs],
    })
    summary = ', '.join([f"({v:+.3f})" for c, v in contribs[:3]])
    print(f"  P{j+1}: LIME contribs: {summary}")

with open('results/paper_lime_details.json', 'w', encoding='utf-8') as f:
    json.dump(lime_details, f, ensure_ascii=False, indent=2)
print("LIME details kaydedildi.")

# ============================================================
# RULEFIT CASE STUDY
# ============================================================
print("\n=== RULEFIT CASE STUDY ===")
rules = rf_main.get_rules()
rules = rules[rules.coef != 0].copy()
rules_s = rules.reindex(rules.importance.abs().sort_values(ascending=False).index)
top_rules  = rules_s[rules_s.type == 'rule'].head(8)
top_linear = rules_s[rules_s.type == 'linear'].head(8)
feat_names = list(X_te_70.columns)

case_study = []
for j, idx in enumerate(sel_idx):
    row_vals = X_te_70.iloc[idx]
    true_v = float(y_te_70.values[idx])
    rf_v   = float(y_pred_rf[idx])
    xgb_v  = float(y_pred_xgb[idx])
    error  = abs(true_v - rf_v)
    cat = 'Good (<1.5 GPa)' if error < 1.5 else ('Medium (1.5-3.5 GPa)' if error < 3.5 else 'High (>3.5 GPa)')

    # Linear contributions
    lc_list = []
    for _, lr in top_linear.iterrows():
        fn = lr.rule
        if fn in feat_names:
            fval = float(row_vals[fn])
            lc_list.append({
                'feature': fn,
                'value': round(fval, 4),
                'coefficient': round(float(lr.coef), 4),
                'contribution': round(float(lr.coef) * fval, 4),
            })

    # Rule activations
    ra_list = []
    for _, rr in top_rules.iterrows():
        rule_str = rr.rule
        conds = rule_str.split(' & ')
        activated = True
        for cond in conds:
            parts = cond.strip().split(' ')
            if len(parts) >= 3:
                fn = ' '.join(parts[:-2])
                op = parts[-2]
                try:
                    val = float(parts[-1])
                    if fn in feat_names:
                        fv = float(row_vals[fn])
                        if   op == '>':  activated = activated and (fv >  val)
                        elif op == '<=': activated = activated and (fv <= val)
                        elif op == '<':  activated = activated and (fv <  val)
                        elif op == '>=': activated = activated and (fv >= val)
                except Exception:
                    pass
        ra_list.append({
            'rule': rule_str[:100],
            'coefficient': round(float(rr.coef), 4),
            'importance': round(float(rr.importance), 4),
            'activated': activated,
        })

    cs = {
        'point': j + 1,
        'test_idx': int(idx),
        'true_GPa': round(true_v, 3),
        'rf_pred_GPa': round(rf_v, 3),
        'xgb_pred_GPa': round(xgb_v, 3),
        'error_GPa': round(error, 3),
        'category': cat,
        'key_features': {
            'Graphene_wt%': round(float(row_vals.get('Graphene (wt.%)', 0)), 3),
            'Density_%': round(float(row_vals.get('Density (%)', 0)), 3),
            'Sintering_Temp_C': round(float(row_vals.get('Sintering Temperature (\u00b0C)', 0)), 1),
            'Sintering_Press_MPa': round(float(row_vals.get('Sintering Pressure (MPa)', 0)), 1),
        },
        'linear_contributions': lc_list[:5],
        'rule_activations': ra_list[:5],
    }
    case_study.append(cs)
    print(f"\n  P{j+1} True={true_v:.2f} GPa, RF={rf_v:.2f} GPa, err={error:.2f} [{cat}]")
    print(f"  Key: graphene={cs['key_features']['Graphene_wt%']} wt%, "
          f"density={cs['key_features']['Density_%']}%, "
          f"T={cs['key_features']['Sintering_Temp_C']}C, "
          f"P={cs['key_features']['Sintering_Press_MPa']}MPa")
    for lc in lc_list[:3]:
        print(f"  LINEAR {lc['feature'][:35]}: val={lc['value']}, "
              f"coef={lc['coefficient']:+.4f}, contrib={lc['contribution']:+.4f}")
    for ra in ra_list[:3]:
        print(f"  RULE [{ra['activated']}] coef={ra['coefficient']:+.4f}: {ra['rule'][:60]}")

with open('results/paper_case_study.json', 'w', encoding='utf-8') as f:
    json.dump(case_study, f, ensure_ascii=False, indent=2)

print("\n=== TUM ANALIZLER TAMAMLANDI ===")
import os
new_files = [f for f in os.listdir('results') if f.startswith('paper_')]
for f in sorted(new_files):
    print(f"  results/{f}")
