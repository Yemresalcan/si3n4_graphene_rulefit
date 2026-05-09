# Interpretable Machine Learning for Vickers Hardness Prediction of Graphene-Added Si₃N₄ Ceramics

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7%2B-orange)](https://xgboost.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/SHAP-0.44%2B-purple)](https://shap.readthedocs.io/)
[![RuleFit](https://img.shields.io/badge/RuleFit-0.3%2B-red)](https://github.com/christophM/rulefit)

> **Paper:** *Interpretable Machine Learning Framework for Predicting Vickers Hardness of Graphene-Added Si₃N₄ Ceramics*

---

## Overview

This repository contains the full source code, dataset, and results for an interpretable machine learning (ML) study predicting the **Vickers hardness** of **graphene-reinforced Si₃N₄ ceramics**. The dataset comprises **82 experimentally characterised samples** sourced from the literature, covering a hardness range of **0.46–27.00 GPa**.

The central contribution is a **tuned RuleFit model** that matches the predictive accuracy of black-box models (XGBoost, DNN) while producing **human-readable, physically interpretable rules** — directly actionable for ceramic process design.

---

## Key Results

| Model | R² (70/30) | MAE (GPa) | RMSE (GPa) |
|---|---|---|---|
| **RuleFit (Tuned)** | **0.854** | **1.477** | **1.955** |
| XGBoost (Extra Features) | 0.835 | 1.599 | 2.074 |
| DNN Small (32-16) | 0.836 | 1.519 | 2.071 |
| XGBoost (Default) | 0.817 | 1.757 | 2.187 |
| RuleFit (Default) | 0.825 | 1.504 | 2.135 |
| ElasticNet | 0.636 | 2.338 | 3.081 |

> RuleFit (Tuned) achieves the best performance on the 70/30 split and demonstrates stable R² = 0.769 ± 0.089 across 30 random splits.

---

## Methodology

```
Raw Data (82 samples, 22 features)
         │
         ▼
Target Encoding (6 categorical variables)
         │
         ▼
Feature Engineering (+8 features → 30 total)
  • graphene_aspect_ratio    • sintering_energy
  • graphene_volume_proxy    • sintering_intensity
  • sintering_dose           • total_additive
  • additive_ratio           • si3n4_to_additive
         │
         ▼
Model Training & Evaluation
  ┌──────────────────────────────────────┐
  │  Fixed 80/20 split (Table 1)        │
  │  Fixed 70/30 split (Table 2)        │
  │  30-replicate stability (Table 3)   │
  │  One-tailed paired t-test (Table 4) │
  └──────────────────────────────────────┘
         │
         ▼
Interpretability Analysis
  • RuleFit rule extraction
  • SHAP (TreeExplainer on XGBoost)
  • LIME (LimeTabularExplainer)
  • SHAP waterfall for P1, P2, P3
```

---

## Repository Structure

```
├── data/
│   └── data.xlsx                      # Dataset (82 samples, 24 columns)
│
├── src/
│   ├── 01_train_and_evaluate.py       # Full training pipeline, all tables
│   ├── 02_pca_tsne.py                 # PCA and t-SNE visualisation
│   ├── 03_boxplot.py                  # R² distribution boxplot (30 splits)
│   ├── 04_shap_waterfall_and_rules.py # SHAP waterfall + RuleFit rule chart
│   └── 05_dendrogram_shap.py          # Feature dendrogram + SHAP summary
│
├── notebooks/
│   ├── vickers_hardness_prediction.ipynb  # Main analysis notebook
│   └── model_analysis.ipynb               # Extended model analysis
│
├── results/
│   ├── figures/                       # All generated figures (PNG)
│   │   ├── pca_tsne_combined.png
│   │   ├── boxplot_r2_distribution.png
│   │   ├── heatmap_correlation.png
│   │   ├── paper_dendrogram.png
│   │   ├── paper_shap_summary.png
│   │   ├── rulefit_feature_importance.png
│   │   ├── rulefit_rules_extracted.png
│   │   ├── shap_waterfall_p123.png
│   │   ├── fewshot_learning_curve.png
│   │   └── heatmap_model_performance.png
│   │
│   └── tables/                        # All result tables (CSV)
│       ├── paper_table1_80_20.csv
│       ├── paper_table2_70_30.csv
│       ├── paper_table3_30split.csv
│       ├── paper_table4_onetail_ttest.csv
│       ├── paper_30split_r2_raw.csv
│       └── rulefit_top_rules.csv
│
├── paper/
│   └── sn-article.tex                 # LaTeX source (Springer Nature)
│
├── requirements.txt
└── README.md
```

---

## Figures

### Data Structure
<p align="center">
  <img src="results/figures/pca_tsne_combined.png" width="800"/>
</p>

*PCA (PC1=32.2%, PC2=18.6%) and t-SNE projections coloured by Vickers hardness. Low-hardness (blue) and high-hardness (red) samples form distinct clusters in t-SNE space.*

---

### Model Stability (30 Random Splits)
<p align="center">
  <img src="results/figures/boxplot_r2_distribution.png" width="750"/>
</p>

*R² distributions across 30 random 70/30 splits. XGBoost and RuleFit show narrow IQRs; Linear Regression and DNN exhibit extreme instability (clipped to [0,1] — their actual distributions extend far below zero).*

---

### RuleFit Extracted Rules
<p align="center">
  <img src="results/figures/rulefit_rules_extracted.png" width="750"/>
</p>

*Top 12 rules from the tuned RuleFit model. Blue = hardness-increasing, Red = hardness-decreasing. The most important rule: high graphene content combined with insufficient sintering intensity strongly degrades hardness (coef = −2.133).*

---

### SHAP Waterfall Analysis
<p align="center">
  <img src="results/figures/shap_waterfall_p123.png" width="800"/>
</p>

*SHAP feature contributions for P1 (low hardness, 6.5 GPa), P2 (high hardness, 18.3 GPa), and P3 (moderate hardness, 13.3 GPa). Base value = 14.933 GPa.*

---

### Feature Importance & Clustering
<p float="left" align="center">
  <img src="results/figures/rulefit_feature_importance.png" width="380"/>
  <img src="results/figures/paper_dendrogram.png" width="380"/>
</p>

*Left: RuleFit feature importance (sintering pressure, graphene content, and density are top predictors). Right: Ward-linkage dendrogram revealing three feature families — sintering process parameters, material composition, and graphene morphology.*

---

## Installation

```bash
# Clone the repository
git clone https://github.com/Yemresalcan/si3n4_graphene_rulefit.git
cd si3n4_graphene_rulefit

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

### Run the full training pipeline
```bash
python src/01_train_and_evaluate.py
```
Outputs: all result CSVs and figures to `results/`

### Generate individual figures
```bash
python src/02_pca_tsne.py          # PCA + t-SNE
python src/03_boxplot.py           # Stability boxplot
python src/04_shap_waterfall_and_rules.py  # SHAP waterfall + rules
python src/05_dendrogram_shap.py   # Dendrogram + SHAP summary
```

### Run interactively
```bash
jupyter notebook notebooks/
```

---

## Dataset

| Property | Value |
|---|---|
| Samples | 82 |
| Original features | 22 |
| Engineered features | 8 |
| Total features | 30 |
| Target | Vickers hardness (GPa) |
| Target range | 0.46 – 27.00 GPa |
| Categorical variables | 6 (target-encoded) |

The dataset (`data/data.xlsx`) aggregates published experimental results for graphene-added Si₃N₄ ceramics, covering variations in graphene content, graphene type, sintering technique, temperature, pressure, time, and additive composition.

---

## Top Extracted RuleFit Rules

| Rule | Coefficient | Support | Physical Meaning |
|---|---|---|---|
| Linear: Graphene (wt.%) | −1.682 | 1.00 | Globally, more graphene → lower hardness |
| Load > th & Graphene Type > th & Density > th | +1.626 | 0.43 | High load + quality graphene + density → higher hardness |
| Graphene% > th & Sint. Intensity ≤ th | −2.133 | 0.11 | Excess graphene + low sintering intensity → severe hardness loss |
| Linear: Graphene Type (encoded) | +0.598 | 1.00 | Higher-quality graphene type globally increases hardness |
| Sint. Temp > th & Si₃N₄% ≤ th | −1.017 | 0.25 | High temperature with low Si₃N₄ reduces hardness |

---

## Requirements

```
pandas>=1.5
numpy>=1.23
scikit-learn>=1.2
xgboost>=1.7
shap>=0.44
rulefit>=0.3
lime>=0.2
matplotlib>=3.6
seaborn>=0.12
scipy>=1.9
openpyxl>=3.0
```

---

## Citation

If you use this code or dataset, please cite:

```bibtex
@article{salcan2025rulefit,
  title   = {Interpretable Machine Learning Framework for Predicting
             Vickers Hardness of Graphene-Added Si$_3$N$_4$ Ceramics},
  author  = {Salcan, Yunus Emre and others},
  journal = {Journal of Materials Science},
  year    = {2025},
  note    = {Under review}
}
```

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
