#!/usr/bin/env python3
"""Step 13 - Permutation feature importance (PFI) versus MDI (Sections 2.8 & 3.6).

Computes Permutation Feature Importance (PFI) to verify that the dominance of
stoichiometric Lp norms under Mean Decrease in Impurity (MDI) is not an artifact
of cardinality bias.

Protocol:
  - The tuned Random Forest is trained strictly on the TRAINING set (3,645 samples).
  - PFI is evaluated exclusively on the VALIDATION set (1,041 samples, 30 random repeats).
  - The held-out test set (521 samples) is untouched.

Outputs
  figures/revision/fig_pfi_top20.png
  figures/revision/fig_mdi_vs_pfi_comparison.png
  figures/publication/fig_pfi_top20.png, fig_mdi_vs_pfi_comparison.{png,pdf}
  results/permutation_importance_full.csv
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tco_stability.paths import ensure_project_root
ensure_project_root()

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from time import time

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, r2_score

from tco_stability.features import parse_composition, featurize_composition_enhanced

RNG_SEED = 42
np.random.seed(RNG_SEED)

os.makedirs("figures/revision", exist_ok=True)
os.makedirs("results", exist_ok=True)

print("=" * 70)
print("STEP 13: Permutation Feature Importance Analysis (Validation Set)")
print("=" * 70)

print("\n[1/5] Loading and featurizing data...")
df = pd.read_csv("data/processed/mp_doped_oxides_cleaned.csv")
unsupported = {'Pm', 'Pu', 'Tc', 'Np', 'Pa'}
mask = df['elements'].apply(lambda x: not any(e.strip() in unsupported for e in x.split(',')))
df = df[mask].reset_index(drop=True)

feature_records, skipped = [], []
for idx, row in df.iterrows():
    comp = parse_composition(row['composition'])
    try:
        feature_records.append(featurize_composition_enhanced(comp))
    except (KeyError, ZeroDivisionError):
        skipped.append(idx)

df = df.drop(index=skipped).reset_index(drop=True)
df_features = pd.DataFrame(feature_records)
feature_names = list(df_features.columns)
y = df['energy_above_hull'].values
print(f"  {len(df)} samples, {len(feature_names)} features")

print("\n[2/5] Splitting data (composition-based, seed=42)...")
unique_formulas = df['formula_pretty'].unique().copy()
np.random.shuffle(unique_formulas)
n_total = len(unique_formulas)
n_test = int(round(0.10 * n_total))
n_val = int(round(0.20 * n_total))

test_formulas = set(unique_formulas[:n_test])
val_formulas = set(unique_formulas[n_test:n_test + n_val])
train_formulas = set(unique_formulas[n_test + n_val:])

train_mask = df['formula_pretty'].isin(train_formulas)
val_mask = df['formula_pretty'].isin(val_formulas)
test_mask = df['formula_pretty'].isin(test_formulas)

X_train, y_train = df_features[train_mask].values, y[train_mask]
X_val, y_val = df_features[val_mask].values, y[val_mask]
X_test, y_test = df_features[test_mask].values, y[test_mask]
print(f"  Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s = scaler.transform(X_val)
X_test_s = scaler.transform(X_test)

print("\n[3/5] Training tuned RF on training set only (preserving validation & test integrity)...")
rf_train_only = RandomForestRegressor(
    n_estimators=200, max_depth=None, min_samples_split=2,
    min_samples_leaf=1, max_features='sqrt', random_state=RNG_SEED, n_jobs=-1
)
rf_train_only.fit(X_train_s, y_train)

y_pred_val = rf_train_only.predict(X_val_s)
mae_val = mean_absolute_error(y_val, y_pred_val)
r2_val = r2_score(y_val, y_pred_val)
print(f"  Validation MAE = {mae_val:.4f} eV/atom, R² = {r2_val:.4f} (Test set untouched)")

print("\n[4/5] Computing MDI and Validation-Set Permutation (PFI) importances...")

mdi_imp = rf_train_only.feature_importances_
mdi_df = pd.DataFrame({
    'feature': feature_names,
    'mdi_importance': mdi_imp
}).sort_values('mdi_importance', ascending=False).reset_index(drop=True)

t0 = time()
perm_result = permutation_importance(
    rf_train_only, X_val_s, y_val,
    n_repeats=30,
    random_state=RNG_SEED,
    scoring='neg_mean_absolute_error',
    n_jobs=-1
)
dt = time() - t0
print(f"  Permutation importance computed in {dt:.1f}s (30 repeats × {len(feature_names)} features)")

pfi_df = pd.DataFrame({
    'feature': feature_names,
    'pfi_mean': perm_result.importances_mean,
    'pfi_std': perm_result.importances_std,
}).sort_values('pfi_mean', ascending=False).reset_index(drop=True)

merged = mdi_df.merge(pfi_df, on='feature')
merged.to_csv("results/permutation_importance_full.csv", index=False)
print("  Full results saved to results/permutation_importance_full.csv")

print("\n" + "=" * 70)
print("KEY COMPARISON: MDI vs PFI Top 10 (Validation Set)")
print("=" * 70)

print("\n  MDI Top 10:")
for i, row in mdi_df.head(10).iterrows():
    cat = "STOICH" if "stoich" in row['feature'] else "ELEC" if "elec_" in row['feature'] else "IONIC" if "ionic" in row['feature'] else "ELEM"
    print(f"    {i+1:2d}. {row['feature']:35s}  {row['mdi_importance']:.4f}  [{cat}]")

print("\n  PFI Top 10 (unbiased, on validation set):")
for i, row in pfi_df.head(10).iterrows():
    cat = "STOICH" if "stoich" in row['feature'] else "ELEC" if "elec_" in row['feature'] else "IONIC" if "ionic" in row['feature'] else "ELEM"
    print(f"    {i+1:2d}. {row['feature']:35s}  {row['pfi_mean']:.4f} ± {row['pfi_std']:.4f}  [{cat}]")

n_stoich_mdi = sum(1 for f in mdi_df.head(10)['feature'] if 'stoich' in f)
n_stoich_pfi = sum(1 for f in pfi_df.head(10)['feature'] if 'stoich' in f)
print(f"\n  Stoichiometric features in MDI top 10: {n_stoich_mdi}")
print(f"  Stoichiometric features in PFI top 10: {n_stoich_pfi}")

print("\n[5/5] Generating figures...")
os.makedirs("figures/publication", exist_ok=True)
os.makedirs("figures/revision", exist_ok=True)


def get_category_color(fname):
    if 'stoich' in fname: return '#E53935'
    elif 'elec_' in fname: return '#EF9A9A'
    elif 'ionic' in fname: return '#7B1FA2'
    else: return '#1E88E5'


top20_pfi = pfi_df.head(20).iloc[::-1]

fig, ax = plt.subplots(figsize=(8, 7))
colors = [get_category_color(f) for f in top20_pfi['feature']]
ax.barh(range(20), top20_pfi['pfi_mean'], xerr=top20_pfi['pfi_std'],
        color=colors, edgecolor='white', height=0.7, capsize=2)
ax.set_yticks(range(20))
ax.set_yticklabels(top20_pfi['feature'], fontsize=8)
ax.set_xlabel("Permutation Feature Importance\n(increase in validation MAE when feature is shuffled)", fontsize=10)
ax.set_title("Top 20 Features — Permutation Importance (Validation Set)", fontsize=12, fontweight='bold')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#1E88E5', label='Elemental property'),
    Patch(facecolor='#E53935', label='Stoichiometric'),
    Patch(facecolor='#EF9A9A', label='Electronic structure'),
    Patch(facecolor='#7B1FA2', label='Ionic compound'),
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=8)

plt.tight_layout()
plt.savefig("figures/revision/fig_pfi_top20.png", dpi=300, bbox_inches='tight')
plt.savefig("figures/publication/fig_pfi_top20.png", dpi=300, bbox_inches='tight')
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(14, 7))

top20_mdi = mdi_df.head(20).iloc[::-1]
colors_mdi = [get_category_color(f) for f in top20_mdi['feature']]
axes[0].barh(range(20), top20_mdi['mdi_importance'], color=colors_mdi, edgecolor='white', height=0.7)
axes[0].set_yticks(range(20))
axes[0].set_yticklabels(top20_mdi['feature'], fontsize=7)
axes[0].set_xlabel("MDI (Mean Decrease in Impurity)", fontsize=10)
axes[0].set_title("(a) MDI Feature Importance", fontsize=11, fontweight='bold')
axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)

top20_pfi_plot = pfi_df.head(20).iloc[::-1]
colors_pfi = [get_category_color(f) for f in top20_pfi_plot['feature']]
axes[1].barh(range(20), top20_pfi_plot['pfi_mean'], xerr=top20_pfi_plot['pfi_std'],
             color=colors_pfi, edgecolor='white', height=0.7, capsize=2)
axes[1].set_yticks(range(20))
axes[1].set_yticklabels(top20_pfi_plot['feature'], fontsize=7)
axes[1].set_xlabel("PFI (Increase in validation MAE)", fontsize=10)
axes[1].set_title("(b) Permutation Feature Importance (Validation Set)", fontsize=11, fontweight='bold')
axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)
axes[1].legend(handles=legend_elements, loc='lower right', fontsize=7)

plt.tight_layout()
plt.savefig("figures/revision/fig_mdi_vs_pfi_comparison.png", dpi=300, bbox_inches='tight')
plt.savefig("figures/publication/fig_mdi_vs_pfi_comparison.png", dpi=300, bbox_inches='tight')
plt.savefig("figures/publication/fig_mdi_vs_pfi_comparison.pdf", bbox_inches='tight')
plt.close(fig)
print("  Saved: figures/publication/fig_mdi_vs_pfi_comparison.{png,pdf}")

print("\n" + "=" * 70)
print("PERMUTATION IMPORTANCE ANALYSIS COMPLETE")
print("=" * 70)