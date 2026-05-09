"""
Kapsamli Analiz Scripti — Hocanin 6 Istegi
Gorev 1: SOTA + FE XGBoost + 20 Model tablolari
Gorev 2: Few-Shot sistematik (learning curve)
Gorev 3: 30 split kararlilik analizi
Gorev 4: Student t-test (RuleFit vs digerleri)
Gorev 5: RuleFit kural cikarimi
Gorev 6: Heatmap, ANOVA, F-test
"""
import sys, io, os, time, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
warnings.filterwarnings('ignore')
os.makedirs('results', exist_ok=True)

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import f_oneway, ttest_rel

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge, HuberRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor,
                               AdaBoostRegressor, ExtraTreesRegressor, BaggingRegressor)
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from rulefit import RuleFit
import xgboost as xgb
from scipy.stats import randint, uniform

plt.rcParams['font.size'] = 11
plt.rcParams['figure.dpi'] = 150

# ============================================================
# VERI YUKLEME VE HAZIRLAMA
# ============================================================
print("=" * 70)
print("VERI YUKLEME")
print("=" * 70, flush=True)

data = pd.read_excel('data.xlsx', sheet_name='Vickers Hardness')
tc = 'Vickers hardness (GPa)'
if 'Ref.' in data.columns:
    data = data.drop(columns=['Ref.'])

# Target Encoding
text_columns = data.select_dtypes(include=['object']).columns.tolist()
encoded_data = data.copy()
for col in text_columns:
    means = encoded_data.groupby(col)[tc].mean()
    encoded_data[col + '_encoded'] = encoded_data[col].map(means)
encoded_data = encoded_data.drop(columns=text_columns)

# ORIJINAL veri (FE yok) — SOTA icin
X_orig = encoded_data.drop(columns=[tc])
y_orig = encoded_data[tc]

# Feature Engineering
ed = encoded_data.copy()
ed['graphene_aspect_ratio'] = ed['Surface Area or Diameter or lateral size of graphene (um)'] / (ed['Thickness of graphene (nm)'] + 0.01)
ed['graphene_volume_proxy'] = ed['Graphene (wt. %)'] * ed['Thickness of graphene (nm)'] * ed['Surface Area or Diameter or lateral size of graphene (um)']
ed['sintering_energy'] = ed['Sintering Temperature (°C)'] * ed['Sintering Time (min)']
ed['sintering_intensity'] = ed['Sintering Temperature (°C)'] * ed['Sintering Pressure (MPa)']
ed['sintering_dose'] = ed['Sintering Temperature (°C)'] * ed['Sintering Time (min)'] * ed['Sintering Pressure (MPa)']
ed['total_additive'] = ed['Content of sintering additive 1 (wt.% or vol.%)'] + ed['Content of sintering additive 2 (wt.% or vol.%)']
ed['additive_ratio'] = ed['Content of sintering additive 1 (wt.% or vol.%)'] / (ed['Content of sintering additive 2 (wt.% or vol.%)'] + 0.01)
ed['si3n4_to_additive'] = ed['Si3N4 (wt%)'] / (ed['total_additive'] + 0.01)

X_fe = ed.drop(columns=[tc])
y_fe = ed[tc]

print(f"Orijinal: {X_orig.shape[1]} ozellik, FE: {X_fe.shape[1]} ozellik, {len(y_fe)} ornek", flush=True)

# ============================================================
# GOREV 1: SOTA + FE XGBOOST + 20 MODEL TABLOLARI
# ============================================================
print("\n" + "=" * 70)
print("GOREV 1: SOTA KARSILASTIRMA TABLOLARI")
print("=" * 70, flush=True)

# Tablo 1: SOTA XGBoost (orijinal features, RandomizedSearchCV)
Xtr_o, Xte_o, ytr_o, yte_o = train_test_split(X_orig, y_orig, test_size=0.3, random_state=42)

param_dist = {
    'n_estimators': randint(100, 1000),
    'max_depth': randint(3, 30),
    'learning_rate': uniform(0.01, 0.3),
    'gamma': uniform(0, 1),
    'min_child_weight': randint(1, 20),
    'subsample': uniform(0.6, 0.4),
    'colsample_bytree': uniform(0.6, 0.4),
}

xgb_base = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
rs_sota = RandomizedSearchCV(xgb_base, param_dist, n_iter=50, cv=5,
                              scoring='neg_mean_squared_error', random_state=42, n_jobs=1)
rs_sota.fit(Xtr_o, ytr_o)
yp_sota = rs_sota.predict(Xte_o)

sota_r2 = r2_score(yte_o, yp_sota)
sota_mae = mean_absolute_error(yte_o, yp_sota)
sota_rmse = np.sqrt(mean_squared_error(yte_o, yp_sota))

print(f"\nTablo 1 — SOTA XGBoost (Orijinal {X_orig.shape[1]} ozellik, RandomizedSearchCV):")
print(f"  MAE={sota_mae:.4f}, RMSE={sota_rmse:.4f}, R2={sota_r2:.4f}")
print(f"  Best params: {rs_sota.best_params_}", flush=True)

# Tablo 2: Ayni XGBoost + FE
Xtr_f, Xte_f, ytr_f, yte_f = train_test_split(X_fe, y_fe, test_size=0.3, random_state=42)

rs_fe = RandomizedSearchCV(xgb.XGBRegressor(objective='reg:squarederror', random_state=42),
                            param_dist, n_iter=50, cv=5,
                            scoring='neg_mean_squared_error', random_state=42, n_jobs=1)
rs_fe.fit(Xtr_f, ytr_f)
yp_fe = rs_fe.predict(Xte_f)

fe_r2 = r2_score(yte_f, yp_fe)
fe_mae = mean_absolute_error(yte_f, yp_fe)
fe_rmse = np.sqrt(mean_squared_error(yte_f, yp_fe))

print(f"\nTablo 2 — XGBoost + FE ({X_fe.shape[1]} ozellik, RandomizedSearchCV):")
print(f"  MAE={fe_mae:.4f}, RMSE={fe_rmse:.4f}, R2={fe_r2:.4f}", flush=True)

# Kaydet
t1 = pd.DataFrame({
    'Yaklasim': ['SOTA (Orijinal XGBoost)', 'XGBoost + Feature Engineering'],
    'Ozellik Sayisi': [X_orig.shape[1], X_fe.shape[1]],
    'MAE': [round(sota_mae, 4), round(fe_mae, 4)],
    'RMSE': [round(sota_rmse, 4), round(fe_rmse, 4)],
    'R2': [round(sota_r2, 4), round(fe_r2, 4)],
})
t1.to_csv('results/tablo1_sota_vs_fe.csv', index=False)
print("\n[Tablo 1-2 kaydedildi: results/tablo1_sota_vs_fe.csv]", flush=True)


# ============================================================
# GOREV 2: FEW-SHOT SISTEMATIK ANALIZ
# ============================================================
print("\n" + "=" * 70)
print("GOREV 2: FEW-SHOT SISTEMATIK ANALIZ")
print("=" * 70, flush=True)

# Sabit test seti (random_state=42)
Xtr_fs, Xte_fs, ytr_fs, yte_fs = train_test_split(X_fe, y_fe, test_size=0.3, random_state=42)
sc_fs = StandardScaler()
Xtr_fs_s = sc_fs.fit_transform(Xtr_fs)
Xte_fs_s = sc_fs.transform(Xte_fs)

k_values = [1, 5, 10, 15, 20, 30, 40, 50, len(Xtr_fs)]
n_repeats = 30
fs_results = []

# RuleFit icin few-shot
for k in k_values:
    r2_list = []
    rmse_list = []
    mae_list = []
    for seed in range(n_repeats):
        np.random.seed(seed)
        if k >= len(Xtr_fs):
            idx = np.arange(len(Xtr_fs))
        else:
            idx = np.random.choice(len(Xtr_fs), size=k, replace=False)

        Xtr_sub = Xtr_fs_s[idx]
        ytr_sub = ytr_fs.values[idx]

        try:
            rf = RuleFit(tree_size=4, max_rules=100, rfmode='regress', random_state=seed)
            rf.fit(Xtr_sub, ytr_sub, feature_names=X_fe.columns.tolist())
            yp = rf.predict(Xte_fs_s)
            r2_list.append(r2_score(yte_fs, yp))
            rmse_list.append(np.sqrt(mean_squared_error(yte_fs, yp)))
            mae_list.append(mean_absolute_error(yte_fs, yp))
        except:
            pass

    if r2_list:
        fs_results.append({
            'k': k, 'n_runs': len(r2_list),
            'R2_mean': np.mean(r2_list), 'R2_std': np.std(r2_list),
            'RMSE_mean': np.mean(rmse_list), 'RMSE_std': np.std(rmse_list),
            'MAE_mean': np.mean(mae_list), 'MAE_std': np.std(mae_list),
        })
        print(f"  k={k:3d}: R2={np.mean(r2_list):.4f}±{np.std(r2_list):.4f} "
              f"RMSE={np.mean(rmse_list):.4f}±{np.std(rmse_list):.4f} ({len(r2_list)} runs)", flush=True)

fs_df = pd.DataFrame(fs_results)
fs_df.to_csv('results/tablo2_fewshot.csv', index=False)

# Few-shot grafik
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].errorbar(fs_df['k'], fs_df['R2_mean'], yerr=fs_df['R2_std'],
                  marker='o', capsize=5, linewidth=2, color='#2ecc71')
axes[0].set_xlabel('Egitim Ornegi Sayisi (k)')
axes[0].set_ylabel('R² (ortalama ± std)')
axes[0].set_title('Few-Shot Learning Curve — RuleFit R²')
axes[0].grid(True, alpha=0.3)

axes[1].errorbar(fs_df['k'], fs_df['RMSE_mean'], yerr=fs_df['RMSE_std'],
                  marker='s', capsize=5, linewidth=2, color='#e74c3c')
axes[1].set_xlabel('Egitim Ornegi Sayisi (k)')
axes[1].set_ylabel('RMSE (ortalama ± std)')
axes[1].set_title('Few-Shot Learning Curve — RuleFit RMSE')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/fewshot_learning_curve.png', bbox_inches='tight')
plt.close()
print("[Few-shot grafik kaydedildi: results/fewshot_learning_curve.png]", flush=True)


# ============================================================
# GOREV 3: 30 FARKLI SPLIT ILE KARARLILIK ANALIZI
# ============================================================
print("\n" + "=" * 70)
print("GOREV 3: 30 SPLIT KARARLILIK ANALIZI")
print("=" * 70, flush=True)

def get_models():
    """Tum 20 modeli dondur (isim, model, scaled_mi)"""
    return [
        ('Linear Regression', LinearRegression(), False),
        ('Ridge', Ridge(alpha=1.0), False),
        ('Lasso', Lasso(alpha=1.0), False),
        ('ElasticNet', ElasticNet(alpha=1.0, l1_ratio=0.5), False),
        ('Decision Tree', DecisionTreeRegressor(), False),
        ('Random Forest', RandomForestRegressor(n_estimators=100, random_state=42), False),
        ('SVR (Linear)', SVR(kernel='linear', C=1.0), True),
        ('Gradient Boosting', GradientBoostingRegressor(), False),
        ('XGBoost', xgb.XGBRegressor(objective='reg:squarederror', random_state=42), False),
        ('AdaBoost', AdaBoostRegressor(n_estimators=100, random_state=42), False),
        ('Extra Trees', ExtraTreesRegressor(n_estimators=100, random_state=42), False),
        ('Bagging', BaggingRegressor(n_estimators=100, random_state=42), False),
        ('Bayesian Ridge', BayesianRidge(), True),
        ('GPR', GaussianProcessRegressor(kernel=ConstantKernel(1.0)*Matern(1.0, nu=2.5),
                                          n_restarts_optimizer=3, random_state=42, alpha=0.1), True),
        ('Huber', HuberRegressor(max_iter=500), True),
        ('KNN (Few-Shot)', KNeighborsRegressor(n_neighbors=3, metric='euclidean', weights='distance'), True),
        ('DNN Small (32-16)', MLPRegressor(hidden_layer_sizes=(32,16), max_iter=2000, random_state=42, early_stopping=True), True),
        ('DNN Medium (64-32)', MLPRegressor(hidden_layer_sizes=(64,32), max_iter=2000, random_state=42, early_stopping=True), True),
        ('DNN Large (128-64-32)', MLPRegressor(hidden_layer_sizes=(128,64,32), max_iter=2000, random_state=42, early_stopping=True), True),
    ]

n_splits = 30
model_names = [m[0] for m in get_models()] + ['RuleFit']
all_r2 = {name: [] for name in model_names}
all_rmse = {name: [] for name in model_names}
all_mae = {name: [] for name in model_names}

for split_i in range(n_splits):
    t0 = time.time()
    Xtr, Xte, ytr, yte = train_test_split(X_fe, y_fe, test_size=0.3, random_state=split_i)
    sc = StandardScaler()
    Xtr_s = sc.fit_transform(Xtr)
    Xte_s = sc.transform(Xte)

    for name, model, scaled in get_models():
        try:
            a, b = (Xtr_s, Xte_s) if scaled else (Xtr, Xte)
            model.fit(a, ytr)
            yp = model.predict(b)
            all_r2[name].append(r2_score(yte, yp))
            all_rmse[name].append(np.sqrt(mean_squared_error(yte, yp)))
            all_mae[name].append(mean_absolute_error(yte, yp))
        except:
            all_r2[name].append(np.nan)
            all_rmse[name].append(np.nan)
            all_mae[name].append(np.nan)

    # RuleFit
    try:
        rf = RuleFit(tree_size=4, max_rules=100, rfmode='regress', random_state=42)
        rf.fit(Xtr_s, ytr.values, feature_names=X_fe.columns.tolist())
        yp = rf.predict(Xte_s)
        all_r2['RuleFit'].append(r2_score(yte, yp))
        all_rmse['RuleFit'].append(np.sqrt(mean_squared_error(yte, yp)))
        all_mae['RuleFit'].append(mean_absolute_error(yte, yp))
    except:
        all_r2['RuleFit'].append(np.nan)
        all_rmse['RuleFit'].append(np.nan)
        all_mae['RuleFit'].append(np.nan)

    elapsed = time.time() - t0
    print(f"  Split {split_i+1}/{n_splits} tamamlandi ({elapsed:.1f}s)", flush=True)

# Ozet tablo
summary = []
for name in model_names:
    r2_arr = np.array(all_r2[name])
    rmse_arr = np.array(all_rmse[name])
    mae_arr = np.array(all_mae[name])
    valid = ~np.isnan(r2_arr)
    summary.append({
        'Model': name,
        'R2_mean': np.nanmean(r2_arr), 'R2_std': np.nanstd(r2_arr),
        'RMSE_mean': np.nanmean(rmse_arr), 'RMSE_std': np.nanstd(rmse_arr),
        'MAE_mean': np.nanmean(mae_arr), 'MAE_std': np.nanstd(mae_arr),
        'Valid_runs': int(valid.sum()),
    })

summary_df = pd.DataFrame(summary).sort_values('R2_mean', ascending=False).reset_index(drop=True)
summary_df.to_csv('results/tablo3_30split_summary.csv', index=False)

print("\n30-Split Ozet Tablo:")
print(summary_df[['Model', 'R2_mean', 'R2_std', 'RMSE_mean', 'RMSE_std']].to_string(index=False))
print("\n[Kaydedildi: results/tablo3_30split_summary.csv]", flush=True)

# Raw data kaydet
r2_df = pd.DataFrame(all_r2)
r2_df.to_csv('results/30split_r2_raw.csv', index=False)
rmse_df = pd.DataFrame(all_rmse)
rmse_df.to_csv('results/30split_rmse_raw.csv', index=False)


# ============================================================
# GOREV 4: STUDENT T-TEST (RULEFIT VS DIGERLERI)
# ============================================================
print("\n" + "=" * 70)
print("GOREV 4: STUDENT T-TEST (RuleFit vs Digerleri)")
print("=" * 70, flush=True)

rulefit_r2 = np.array(all_r2['RuleFit'])
ttest_results = []

for name in model_names:
    if name == 'RuleFit':
        continue
    other_r2 = np.array(all_r2[name])

    # NaN filtreleme
    valid = ~(np.isnan(rulefit_r2) | np.isnan(other_r2))
    if valid.sum() < 5:
        continue

    t_stat, p_value = ttest_rel(rulefit_r2[valid], other_r2[valid])
    diff = np.mean(rulefit_r2[valid]) - np.mean(other_r2[valid])
    ttest_results.append({
        'Model': name,
        'RuleFit_R2_mean': round(np.mean(rulefit_r2[valid]), 4),
        'Model_R2_mean': round(np.mean(other_r2[valid]), 4),
        'Fark': round(diff, 4),
        't_statistic': round(t_stat, 4),
        'p_value': round(p_value, 6),
        'Anlamli (p<0.05)': 'Evet' if p_value < 0.05 else 'Hayir',
    })

ttest_df = pd.DataFrame(ttest_results).sort_values('p_value')
ttest_df.to_csv('results/tablo4_ttest.csv', index=False)

print(ttest_df.to_string(index=False))
print("\n[Kaydedildi: results/tablo4_ttest.csv]", flush=True)


# ============================================================
# GOREV 5: RULEFIT KURAL CIKARIMI VE DERINLESTIRME
# ============================================================
print("\n" + "=" * 70)
print("GOREV 5: RULEFIT KURAL CIKARIMI")
print("=" * 70, flush=True)

# Ana modeli egit (rs=42 split)
Xtr_r, Xte_r, ytr_r, yte_r = train_test_split(X_fe, y_fe, test_size=0.3, random_state=42)
sc_r = StandardScaler()
Xtr_r_s = sc_r.fit_transform(Xtr_r)
Xte_r_s = sc_r.transform(Xte_r)

# Farkli tree_size ve max_rules dene
hp_results = []
for ts in [2, 3, 4, 6, 8]:
    for mr in [50, 100, 200, 500]:
        try:
            rf = RuleFit(tree_size=ts, max_rules=mr, rfmode='regress', random_state=42)
            rf.fit(Xtr_r_s, ytr_r.values, feature_names=X_fe.columns.tolist())
            yp = rf.predict(Xte_r_s)
            r2 = r2_score(yte_r, yp)
            hp_results.append({'tree_size': ts, 'max_rules': mr, 'R2': round(r2, 4)})
        except:
            pass

hp_df = pd.DataFrame(hp_results).sort_values('R2', ascending=False)
hp_df.to_csv('results/rulefit_hyperparams.csv', index=False)
print("RuleFit Hiperparametre Arama:")
print(hp_df.head(10).to_string(index=False), flush=True)

# En iyi parametrelerle modeli egit
best_hp = hp_df.iloc[0]
best_rf = RuleFit(tree_size=int(best_hp['tree_size']), max_rules=int(best_hp['max_rules']),
                   rfmode='regress', random_state=42)
best_rf.fit(Xtr_r_s, ytr_r.values, feature_names=X_fe.columns.tolist())

# Kurallari cikart
rules = best_rf.get_rules()
rules = rules[rules['coef'] != 0].sort_values('importance', ascending=False)
rules_top = rules.head(20)
rules_top.to_csv('results/rulefit_top_rules.csv', index=False)

print(f"\nEn iyi parametreler: tree_size={int(best_hp['tree_size'])}, max_rules={int(best_hp['max_rules'])}, R2={best_hp['R2']}")
print(f"\nToplam aktif kural: {len(rules)}")
print(f"\nEn onemli 15 kural:")
print(rules[['rule', 'type', 'coef', 'importance']].head(15).to_string(index=False), flush=True)

# Feature importance bar chart
linear_rules = rules[rules['type'] == 'linear'].head(15)
fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(linear_rules['rule'], linear_rules['importance'], color='#3498db')
ax.set_xlabel('Importance')
ax.set_title('RuleFit — Feature Importance (Linear Terms)')
ax.invert_yaxis()
plt.tight_layout()
plt.savefig('results/rulefit_feature_importance.png', bbox_inches='tight')
plt.close()

# Rule importance
rule_rules = rules[rules['type'] == 'rule'].head(15)
fig, ax = plt.subplots(figsize=(12, 7))
short_rules = [r[:60] + '...' if len(r) > 60 else r for r in rule_rules['rule']]
ax.barh(short_rules, rule_rules['importance'], color='#e74c3c')
ax.set_xlabel('Importance')
ax.set_title('RuleFit — En Onemli Kurallar')
ax.invert_yaxis()
plt.tight_layout()
plt.savefig('results/rulefit_rules_importance.png', bbox_inches='tight')
plt.close()
print("\n[RuleFit grafikleri kaydedildi]", flush=True)


# ============================================================
# GOREV 6: HEATMAP, ANOVA, F-TEST
# ============================================================
print("\n" + "=" * 70)
print("GOREV 6: HEATMAP, ANOVA, F-TEST")
print("=" * 70, flush=True)

# 6a: Korelasyon Heatmap
print("\n6a: Korelasyon Heatmap...", flush=True)
corr = X_fe.join(y_fe).corr()

fig, ax = plt.subplots(figsize=(18, 15))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, square=True, linewidths=0.5, ax=ax,
            annot_kws={'size': 7})
ax.set_title('Ozellik Korelasyon Matrisi (Pearson)', fontsize=14)
plt.tight_layout()
plt.savefig('results/heatmap_correlation.png', bbox_inches='tight')
plt.close()

# 6b: Model Performans Heatmap (30 split x 20 model)
print("6b: Model Performans Heatmap...", flush=True)
r2_matrix = pd.DataFrame(all_r2)
# Modelleri ortalama R2'ye gore sirala
col_order = r2_matrix.mean().sort_values(ascending=False).index.tolist()
r2_matrix = r2_matrix[col_order]

fig, ax = plt.subplots(figsize=(20, 10))
sns.heatmap(r2_matrix.T, annot=True, fmt='.2f', cmap='RdYlGn',
            center=0.7, linewidths=0.5, ax=ax, annot_kws={'size': 7})
ax.set_xlabel('Split No (0-29)')
ax.set_ylabel('Model')
ax.set_title('30 Split x 20 Model — R² Performans Heatmap', fontsize=14)
plt.tight_layout()
plt.savefig('results/heatmap_model_performance.png', bbox_inches='tight')
plt.close()

# 6c: ANOVA F-test
print("6c: ANOVA F-test...", flush=True)
# Tum modellerin R² degerlerini topla
r2_groups = []
group_names = []
for name in col_order:
    arr = np.array(all_r2[name])
    valid = arr[~np.isnan(arr)]
    if len(valid) >= 5:
        r2_groups.append(valid)
        group_names.append(name)

f_stat, p_value_anova = f_oneway(*r2_groups)
print(f"\nANOVA F-test (tum modeller arasi):")
print(f"  F-statistic = {f_stat:.4f}")
print(f"  p-value = {p_value_anova:.2e}")
print(f"  Anlamli fark var mi? {'EVET' if p_value_anova < 0.05 else 'HAYIR'}", flush=True)

# Kategori bazli ANOVA
categories = {
    'Klasik ML': ['Linear Regression', 'Ridge', 'Lasso', 'ElasticNet', 'Decision Tree', 'Random Forest', 'SVR (Linear)'],
    'Ensemble': ['Gradient Boosting', 'XGBoost', 'AdaBoost', 'Extra Trees', 'Bagging'],
    'Kucuk Veri': ['RuleFit', 'Bayesian Ridge', 'GPR', 'Huber', 'KNN (Few-Shot)'],
    'Deep Learning': ['DNN Small (32-16)', 'DNN Medium (64-32)', 'DNN Large (128-64-32)'],
}

cat_r2 = {}
for cat, models in categories.items():
    vals = []
    for m in models:
        if m in all_r2:
            arr = np.array(all_r2[m])
            vals.extend(arr[~np.isnan(arr)].tolist())
    cat_r2[cat] = vals

cat_groups = list(cat_r2.values())
f_cat, p_cat = f_oneway(*cat_groups)
print(f"\nANOVA F-test (kategoriler arasi):")
print(f"  F-statistic = {f_cat:.4f}")
print(f"  p-value = {p_cat:.2e}", flush=True)

# 6d: Boxplot karsilastirma
print("6d: Boxplot...", flush=True)
fig, ax = plt.subplots(figsize=(16, 8))
box_data = []
box_labels = []
for name in col_order:
    arr = np.array(all_r2[name])
    valid = arr[~np.isnan(arr)]
    box_data.append(valid)
    box_labels.append(name)

bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True, vert=True)
colors_map = {
    'Deep Learning': '#e74c3c', 'Ensemble': '#3498db',
    'Kucuk Veri': '#2ecc71', 'Klasik ML': '#95a5a6'
}
for i, name in enumerate(box_labels):
    cat = 'Klasik ML'
    for c, models in categories.items():
        if name in models:
            cat = c
            break
    bp['boxes'][i].set_facecolor(colors_map.get(cat, '#95a5a6'))

ax.set_xticklabels(box_labels, rotation=45, ha='right', fontsize=9)
ax.set_ylabel('R²')
ax.set_title('30 Split — Model R² Dagilimi (Boxplot)', fontsize=14)
ax.axhline(y=0.8, color='red', linestyle='--', alpha=0.5, label='R²=0.8')
ax.grid(True, alpha=0.2)

from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#e74c3c', label='Deep Learning'),
                   Patch(facecolor='#3498db', label='Ensemble'),
                   Patch(facecolor='#2ecc71', label='Kucuk Veri'),
                   Patch(facecolor='#95a5a6', label='Klasik ML')]
ax.legend(handles=legend_elements, loc='lower right')

plt.tight_layout()
plt.savefig('results/boxplot_r2_distribution.png', bbox_inches='tight')
plt.close()

# ANOVA sonuc tablosu
anova_results = {
    'Test': ['ANOVA (Tum Modeller)', 'ANOVA (Kategoriler)'],
    'F-statistic': [round(f_stat, 4), round(f_cat, 4)],
    'p-value': [f'{p_value_anova:.2e}', f'{p_cat:.2e}'],
    'Anlamli': ['Evet' if p_value_anova < 0.05 else 'Hayir',
                'Evet' if p_cat < 0.05 else 'Hayir'],
}
pd.DataFrame(anova_results).to_csv('results/anova_results.csv', index=False)

print("\n" + "=" * 70)
print("TUM GOREVLER TAMAMLANDI!")
print("=" * 70)
print("\nOlusan dosyalar:")
for f in sorted(os.listdir('results')):
    print(f"  results/{f}")
print("\nBitti!", flush=True)
