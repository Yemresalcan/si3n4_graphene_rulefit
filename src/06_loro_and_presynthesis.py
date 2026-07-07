"""Reviewer-requested sensitivity analyses:
1) Leave-One-Reference-Out (LORO) validation over the 16 literature sources,
   with target encoding fitted only on the training folds (leakage-aware).
2) Pre-synthesis feature ablation: 30 random 70/30 splits after removing
   post-sintering descriptors (density, alpha/beta phase content).
Models: RuleFit (Tuned, tree_size=6, max_rules=500) and
        XGBoost (Extra Features, Tuned via RandomizedSearchCV on the 70/30 split,
        same protocol as the paper pipeline).
"""
import sys, io, warnings, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.stats import randint, uniform
import xgboost as xgb
from rulefit import RuleFit

DATA = r'data/data.xlsx'
OUT = r'results/tables'
TC = 'Vickers hardness (GPa)'

data = pd.read_excel(DATA, sheet_name='Vickers Hardness')
refs = data['Ref.'].astype(str).values
data = data.drop(columns=['Ref.'])
text_columns = data.select_dtypes(include=['object']).columns.tolist()


def target_encode_fit(train_df):
    maps = {}
    for col in text_columns:
        maps[col] = train_df.groupby(col)[TC].mean()
    global_mean = train_df[TC].mean()
    return maps, global_mean


def target_encode_apply(df, maps, global_mean):
    out = df.copy()
    for col in text_columns:
        out[col + '_encoded'] = out[col].map(maps[col]).fillna(global_mean)
    return out.drop(columns=text_columns)


def add_features(df):
    df = df.copy()
    thickness_col = [c for c in df.columns if 'Thickness' in c][0]
    surface_col = [c for c in df.columns if 'Surface Area' in c][0]
    graphene_col = [c for c in df.columns if 'Graphene (wt' in c][0]
    temp_col = [c for c in df.columns if 'Sintering Temperature' in c][0]
    time_col = [c for c in df.columns if 'Sintering Time' in c][0]
    press_col = [c for c in df.columns if 'Sintering Pressure' in c][0]
    add1_col = [c for c in df.columns if 'additive 1' in c and 'Content' in c][0]
    add2_col = [c for c in df.columns if 'additive 2' in c and 'Content' in c][0]
    si3n4_col = [c for c in df.columns if 'Si3N4 (wt' in c][0]
    df['graphene_aspect_ratio'] = df[surface_col] / (df[thickness_col] + 0.01)
    df['graphene_volume_proxy'] = df[graphene_col] * df[thickness_col] * df[surface_col]
    df['sintering_energy'] = df[temp_col] * df[time_col]
    df['sintering_intensity'] = df[temp_col] * df[press_col]
    df['sintering_dose'] = df[temp_col] * df[time_col] * df[press_col]
    df['total_additive'] = df[add1_col] + df[add2_col]
    df['additive_ratio'] = df[add1_col] / (df[add2_col] + 0.01)
    df['si3n4_to_additive'] = df[si3n4_col] / (df['total_additive'] + 0.01)
    return df


xgb_param_dist = {
    'n_estimators': randint(100, 1001),
    'max_depth': randint(3, 31),
    'learning_rate': uniform(0.01, 0.30),
    'gamma': uniform(0, 1.0),
    'min_child_weight': randint(1, 21),
    'subsample': uniform(0.6, 0.4),
    'colsample_bytree': uniform(0.6, 0.4),
}

# --- Reproduce paper's 70/30 XGB (Extra Features) tuning to fix hyperparameters ---
print('Tuning XGBoost (Extra Features) on the 70/30 split (paper protocol)...', flush=True)
maps_all, gm_all = target_encode_fit(data)  # paper pipeline encoded globally for tuning
enc_all = target_encode_apply(data, maps_all, gm_all)
fe_all = add_features(enc_all)
X_fe_all = fe_all.drop(columns=[TC])
y_all = fe_all[TC]
Xtr, Xte, ytr, yte = train_test_split(X_fe_all, y_all, test_size=0.3, random_state=42)
rs = RandomizedSearchCV(xgb.XGBRegressor(objective='reg:squarederror', random_state=42),
                        xgb_param_dist, n_iter=50, cv=5,
                        scoring='neg_mean_squared_error', random_state=42, n_jobs=-1)
rs.fit(Xtr, ytr)
xgb_fe_best_params = rs.best_params_
print('Best params:', xgb_fe_best_params, flush=True)


def eval_pooled(y_true, y_pred):
    return (r2_score(y_true, y_pred),
            mean_absolute_error(y_true, y_pred),
            np.sqrt(mean_squared_error(y_true, y_pred)))


# ============================================================
# 1) LEAVE-ONE-REFERENCE-OUT
# ============================================================
print('\n=== LORO over', len(np.unique(refs)), 'references ===', flush=True)
loro_rows = []
pooled = {'RuleFit (Tuned)': ([], []), 'XGBoost (Extra Features, Tuned)': ([], [])}

for ref in sorted(np.unique(refs), key=lambda r: -np.sum(refs == r)):
    t0 = time.time()
    test_mask = refs == ref
    tr_df, te_df = data[~test_mask], data[test_mask]
    maps, gm = target_encode_fit(tr_df)
    tr_enc = add_features(target_encode_apply(tr_df, maps, gm))
    te_enc = add_features(target_encode_apply(te_df, maps, gm))
    Xtr_f, ytr_f = tr_enc.drop(columns=[TC]), tr_enc[TC]
    Xte_f, yte_f = te_enc.drop(columns=[TC]), te_enc[TC]

    row = {'Ref': ref, 'n_test': int(test_mask.sum())}
    # RuleFit tuned
    m = RuleFit(tree_size=6, max_rules=500, rfmode='regress', random_state=42)
    m.fit(Xtr_f.values, ytr_f.values, feature_names=list(Xtr_f.columns))
    yp = m.predict(Xte_f.values)
    row['RF_MAE'] = mean_absolute_error(yte_f, yp)
    pooled['RuleFit (Tuned)'][0].extend(yte_f.tolist())
    pooled['RuleFit (Tuned)'][1].extend(np.asarray(yp).tolist())
    # XGB FE tuned
    m = xgb.XGBRegressor(objective='reg:squarederror', random_state=42, **xgb_fe_best_params)
    m.fit(Xtr_f, ytr_f)
    yp = m.predict(Xte_f)
    row['XGB_MAE'] = mean_absolute_error(yte_f, yp)
    pooled['XGBoost (Extra Features, Tuned)'][0].extend(yte_f.tolist())
    pooled['XGBoost (Extra Features, Tuned)'][1].extend(np.asarray(yp).tolist())

    loro_rows.append(row)
    print(f"  Ref {ref}: n={row['n_test']}, RF_MAE={row['RF_MAE']:.3f}, "
          f"XGB_MAE={row['XGB_MAE']:.3f} ({time.time()-t0:.1f}s)", flush=True)

loro_df = pd.DataFrame(loro_rows)
loro_df.to_csv(rf'{OUT}/paper_loro_per_reference.csv', index=False)

print('\nLORO pooled out-of-fold metrics:')
summary_rows = []
for name, (yt, yp) in pooled.items():
    r2, mae, rmse = eval_pooled(np.array(yt), np.array(yp))
    summary_rows.append({'Model': name, 'R2_pooled': r2, 'MAE_pooled': mae, 'RMSE_pooled': rmse})
    print(f'  {name}: R2={r2:.4f}, MAE={mae:.4f}, RMSE={rmse:.4f}', flush=True)
pd.DataFrame(summary_rows).to_csv(rf'{OUT}/paper_loro_summary.csv', index=False)

# ============================================================
# 2) PRE-SYNTHESIS ABLATION (30 random 70/30 splits)
# ============================================================
print('\n=== Pre-synthesis ablation (density + alpha/beta phase removed) ===', flush=True)
post_cols = [c for c in data.columns
             if c.startswith('Density') or 'Si3N4 content' in c or 'Si3N4 Content' in c]
print('Removed post-sintering columns:', post_cols, flush=True)

res = {'RuleFit (Tuned)': {'r2': [], 'mae': [], 'rmse': []},
       'XGBoost (Extra Features, Tuned)': {'r2': [], 'mae': [], 'rmse': []}}
enc_pre = add_features(target_encode_apply(data, maps_all, gm_all)).drop(columns=post_cols)
X_pre = enc_pre.drop(columns=[TC])
y_pre = enc_pre[TC]
print(f'Pre-synthesis feature count: {X_pre.shape[1]}', flush=True)

for i in range(30):
    Xtr_p, Xte_p, ytr_p, yte_p = train_test_split(X_pre, y_pre, test_size=0.3, random_state=i)
    m = RuleFit(tree_size=6, max_rules=500, rfmode='regress', random_state=42)
    m.fit(Xtr_p.values, ytr_p.values, feature_names=list(Xtr_p.columns))
    r2, mae, rmse = eval_pooled(yte_p, m.predict(Xte_p.values))
    res['RuleFit (Tuned)']['r2'].append(r2)
    res['RuleFit (Tuned)']['mae'].append(mae)
    res['RuleFit (Tuned)']['rmse'].append(rmse)
    m = xgb.XGBRegressor(objective='reg:squarederror', random_state=42, **xgb_fe_best_params)
    m.fit(Xtr_p, ytr_p)
    r2, mae, rmse = eval_pooled(yte_p, m.predict(Xte_p))
    res['XGBoost (Extra Features, Tuned)']['r2'].append(r2)
    res['XGBoost (Extra Features, Tuned)']['mae'].append(mae)
    res['XGBoost (Extra Features, Tuned)']['rmse'].append(rmse)
    if (i + 1) % 5 == 0:
        print(f'  split {i+1}/30 done', flush=True)

rows = []
for name, d in res.items():
    rows.append({'Model': name,
                 'R2_mean': np.mean(d['r2']), 'R2_SD': np.std(d['r2'], ddof=1),
                 'MAE_mean': np.mean(d['mae']), 'MAE_SD': np.std(d['mae'], ddof=1),
                 'RMSE_mean': np.mean(d['rmse']), 'RMSE_SD': np.std(d['rmse'], ddof=1)})
abl = pd.DataFrame(rows)
abl.to_csv(rf'{OUT}/paper_presynthesis_ablation.csv', index=False)
print('\nPre-synthesis ablation (mean ± SD over 30 splits):')
print(abl.round(4).to_string(index=False))
print('\nDONE', flush=True)
