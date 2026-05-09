"""PCA + t-SNE combined figure — label positioning fixed."""
import warnings, os
warnings.filterwarnings('ignore')
os.makedirs('results', exist_ok=True)

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# ── Load data ──────────────────────────────────────────────────────────────
df = pd.read_excel('data.xlsx')
target_col = 'Vickers hardness (GPa)'
y = df[target_col].values

# Encode categoricals
cat_cols = [c for c in df.columns if df[c].dtype == object]
df_enc = df.copy()
for c in cat_cols:
    df_enc[c] = LabelEncoder().fit_transform(df[c].astype(str))

X_raw = df_enc.drop(columns=[target_col]).apply(pd.to_numeric, errors='coerce').fillna(0)

# Feature engineering (same as paper)
col = list(X_raw.columns)
def ci(name):
    matches = [c for c in col if name.lower() in c.lower()]
    return matches[0] if matches else None

def safe(name):
    c = ci(name)
    return X_raw[c].values.astype(float) if c else np.zeros(len(X_raw))

g   = safe('graphene content')
asp = safe('aspect')
lth = safe('length')
dia = safe('diameter')
tmp = safe('sintering temperature')
prs = safe('sintering pressure')
tim = safe('sintering time')
a1  = safe('additive 1')
a2  = safe('additive 2')
si  = safe('si3n4')

n = len(X_raw)
ar    = np.where(dia > 0, lth / (dia + 1e-9), asp)
gvp   = g * ar
se    = tmp * prs * tim
si_f  = tmp * prs
sd    = tmp * tim
ta    = a1 + a2
ratio = np.where((ta + si) > 0, ta / (ta + si + 1e-9), 0)
s2a   = np.where(ta > 0, si / (ta + 1e-9), si)

X = np.hstack([X_raw.values,
               ar.reshape(-1,1), gvp.reshape(-1,1),
               se.reshape(-1,1), si_f.reshape(-1,1),
               sd.reshape(-1,1), ta.reshape(-1,1),
               ratio.reshape(-1,1), s2a.reshape(-1,1)])

# Scale
X_sc = StandardScaler().fit_transform(X)

# PCA
pca   = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_sc)
var1, var2 = pca.explained_variance_ratio_ * 100

# t-SNE
tsne   = TSNE(n_components=2, perplexity=15, max_iter=1000, random_state=42)
X_tsne = tsne.fit_transform(X_sc)

# ── Plot ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.subplots_adjust(left=0.07, right=0.88, top=0.90, bottom=0.12, wspace=0.35)

vmin, vmax = y.min(), y.max()

# PCA panel
sc1 = axes[0].scatter(X_pca[:, 0], X_pca[:, 1],
                      c=y, cmap='RdYlBu_r', s=55, alpha=0.85,
                      vmin=vmin, vmax=vmax, edgecolors='k', linewidths=0.3)
axes[0].set_xlabel(f'PC1 ({var1:.1f}% variance)', fontsize=11)
axes[0].set_ylabel(f'PC2 ({var2:.1f}% variance)', fontsize=11)
axes[0].set_title('PCA', fontsize=13, fontweight='bold', pad=8)
axes[0].tick_params(labelsize=9)

# t-SNE panel
sc2 = axes[1].scatter(X_tsne[:, 0], X_tsne[:, 1],
                      c=y, cmap='RdYlBu_r', s=55, alpha=0.85,
                      vmin=vmin, vmax=vmax, edgecolors='k', linewidths=0.3)
axes[1].set_xlabel('t-SNE Dimension 1', fontsize=11)
axes[1].set_ylabel('t-SNE Dimension 2', fontsize=11)
axes[1].set_title('t-SNE (perplexity = 15)', fontsize=13, fontweight='bold', pad=8)
axes[1].tick_params(labelsize=9)

# Shared colorbar — outside on the right
cbar_ax = fig.add_axes([0.90, 0.15, 0.025, 0.70])
cbar = fig.colorbar(sc2, cax=cbar_ax)
cbar.set_label('Vickers Hardness (GPa)', fontsize=11, labelpad=10)
cbar.ax.tick_params(labelsize=9)

out = 'results/pca_tsne_combined.png'
fig.savefig(out, dpi=180, bbox_inches='tight')
plt.close()
print(f"Saved: {out}")
print(f"PCA variance: PC1={var1:.1f}%, PC2={var2:.1f}%")
