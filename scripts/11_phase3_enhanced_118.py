"""Step 11 - Phase 3 enhanced: 118 Magpie-style features and three ensemble models.

Expands the composition-based feature vector from 105 to 118 features (Ward et al., 2016)
and compares Random Forest, Gradient Boosting (GBR), and XGBoost:
  1. Elemental property statistics (105 features)
  2. Stoichiometric attributes     (7 features)
  3. Electronic-structure fractions (4 features)
  4. Ionic-compound attributes     (2 features)

Follows the composition-based splitting and single test-evaluation protocol of
Wang et al., Chem. Mater. 2020, 32, 4954-4965.

Outputs
  data/processed/feature_importances_enhanced.csv
  data/processed/grid_search_results_enhanced.csv
  results/test_predictions.csv, results/final_metrics_summary.csv
  figures/phase3_enhanced/05E..10E_*.png
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
from itertools import product

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

from tco_stability.features import (
    PROP_NAMES,
    parse_composition,
    featurize_composition_enhanced,
)

try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    print("  WARNING: XGBoost not installed. Run: pip install xgboost")
    print("    Continuing without XGBoost...")
    HAS_XGBOOST = False

RNG_SEED = 42
np.random.seed(RNG_SEED)

print("=" * 70)
print("PHASE 3 ENHANCED: 118 Magpie Features + RF / GBR / XGBoost")
print("=" * 70)
print("\n[SETUP] Loading data and applying 118-feature Magpie featurization...")

df = pd.read_csv("data/processed/mp_doped_oxides_cleaned.csv")
unsupported = {'Pm', 'Pu', 'Tc', 'Np', 'Pa'}
mask = df['elements'].apply(
    lambda x: not any(e.strip() in unsupported for e in x.split(','))
)
df = df[mask].reset_index(drop=True)

feature_records = []
skipped = []
for idx, row in df.iterrows():
    comp = parse_composition(row['composition'])
    try:
        feats = featurize_composition_enhanced(comp)
        feature_records.append(feats)
    except (KeyError, ZeroDivisionError):
        skipped.append(idx)

df = df.drop(index=skipped).reset_index(drop=True)
df_features = pd.DataFrame(feature_records)
feature_names = list(df_features.columns)
y = df['energy_above_hull'].values

n_elem_prop = 15 * 7
n_stoich = len([f for f in feature_names if f.startswith('stoich_')])
n_elec = len([f for f in feature_names if f.startswith('elec_')])
n_ionic = len([f for f in feature_names if f.startswith('ionic_')])
n_total_features = len(feature_names)

print(f"\n  Feature breakdown (Magpie categories):")
print(f"    Category 1 - Elemental property stats:  {n_elem_prop} features")
print(f"    Category 2 - Stoichiometric attributes: {n_stoich} features")
print(f"    Category 3 - Electronic structure frac: {n_elec} features")
print(f"    Category 4 - Ionic compound attributes: {n_ionic} features")
print(f"    {'-' * 45}")
print(f"    TOTAL:                                  {n_total_features} features")

unique_formulas = df['formula_pretty'].unique().copy()
np.random.seed(RNG_SEED)
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

X_train_raw = df_features[train_mask].values
X_val_raw = df_features[val_mask].values
X_test_raw = df_features[test_mask].values
y_train = y[train_mask]
y_val = y[val_mask]
y_test = y[test_mask]

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
X_val = scaler.transform(X_val_raw)
X_test = scaler.transform(X_test_raw)

print(f"\n  Data splits: {len(X_train)} train / {len(X_val)} val / {len(X_test)} test")
print(f"  Total features: {X_train.shape[1]}")

os.makedirs("figures/phase3_enhanced", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)
os.makedirs("results", exist_ok=True)


# =============================================================================
# STEP 3.1: FEATURE IMPORTANCE (118 features)
# =============================================================================
print("\n" + "=" * 70)
print("[STEP 3.1] FEATURE IMPORTANCE - 118-Feature Set")
print("=" * 70)

rf_baseline = RandomForestRegressor(
    n_estimators=200, max_depth=None, min_samples_split=5,
    min_samples_leaf=2, max_features='sqrt',
    random_state=RNG_SEED, n_jobs=-1
)
rf_baseline.fit(X_train, y_train)

importances = rf_baseline.feature_importances_
feat_imp = pd.DataFrame({
    'feature': feature_names,
    'importance': importances
}).sort_values('importance', ascending=False).reset_index(drop=True)


def get_category(feat_name):
    if feat_name.startswith('stoich_'):
        return 'Stoichiometric'
    elif feat_name.startswith('elec_'):
        return 'Electronic Structure'
    elif feat_name.startswith('ionic_'):
        return 'Ionic Compound'
    else:
        return 'Elemental Property'


feat_imp['category'] = feat_imp['feature'].apply(get_category)

print("\n  Top 25 most important features (with category):")
print("  " + "-" * 65)
for i, row in feat_imp.head(25).iterrows():
    cat_tag = f"  [{row['category']}]" if row['category'] != 'Elemental Property' else ""
    bar = "#" * int(row['importance'] / feat_imp['importance'].max() * 30)
    print(f"  {i+1:2d}. {row['feature']:30s} {row['importance']:.4f} {bar}{cat_tag}")

new_features = feat_imp[feat_imp['category'] != 'Elemental Property']
print(f"\n  Non-elemental feature rankings:")
print(f"  {'-' * 60}")
for _, row in new_features.iterrows():
    rank = feat_imp.index[feat_imp['feature'] == row['feature']].tolist()[0] + 1
    print(f"    Rank {rank:3d}: {row['feature']:30s} ({row['importance']:.4f}) [{row['category']}]")

feat_imp.to_csv("data/processed/feature_importances_enhanced.csv", index=False)

top_n = 25
top_feats = feat_imp.head(top_n)

fig, ax = plt.subplots(figsize=(11, 9))
cat_colors = {
    'Elemental Property': plt.cm.viridis(0.6),
    'Stoichiometric': '#E91E63',
    'Electronic Structure': '#FF9800',
    'Ionic Compound': '#9C27B0',
}
colors = [cat_colors[c] for c in top_feats['category'].values[::-1]]

bars = ax.barh(range(top_n), top_feats['importance'].values[::-1],
               color=colors, edgecolor='white', linewidth=0.5)
ax.set_yticks(range(top_n))
ax.set_yticklabels(top_feats['feature'].values[::-1], fontsize=9)
ax.set_xlabel("Feature Importance (MDI)", fontsize=12)
ax.set_title("Top 25 Feature Importances - 118 Magpie Features\n"
             "(Random Forest, Mean Decrease in Impurity)",
             fontsize=13, fontweight='bold')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=v, label=k) for k, v in cat_colors.items()
                   if k in top_feats['category'].values]
ax.legend(handles=legend_elements, loc='lower right', fontsize=9)

plt.tight_layout()
plt.savefig("figures/phase3_enhanced/05E_feature_importance_enhanced.png", dpi=150, bbox_inches="tight")
print("\n  Saved: figures/phase3_enhanced/05E_feature_importance_enhanced.png")

category_importance = {}
for _, row in feat_imp.iterrows():
    if row['category'] != 'Elemental Property':
        cat = row['category']
        category_importance[cat] = category_importance.get(cat, 0) + row['importance']
    else:
        for prop in PROP_NAMES:
            if row['feature'].startswith(prop):
                category_importance[prop] = category_importance.get(prop, 0) + row['importance']
                break

cat_imp = pd.Series(category_importance).sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(10, 8))
n_cats = len(cat_imp)
colors_cat = []
for name in cat_imp.index:
    if name == 'Stoichiometric':
        colors_cat.append('#E91E63')
    elif name == 'Electronic Structure':
        colors_cat.append('#FF9800')
    elif name == 'Ionic Compound':
        colors_cat.append('#9C27B0')
    else:
        colors_cat.append(plt.cm.plasma(0.2 + 0.6 * len(colors_cat) / n_cats))

ax.barh(range(n_cats), cat_imp.values, color=colors_cat,
        edgecolor='white', linewidth=0.5)
ax.set_yticks(range(n_cats))
labels = [n.replace('_', ' ').title() if n not in ['Stoichiometric', 'Electronic Structure', 'Ionic Compound']
          else f"* {n}" for n in cat_imp.index]
ax.set_yticklabels(labels, fontsize=10)
ax.set_xlabel("Cumulative Feature Importance", fontsize=12)
ax.set_title("Feature Importance by Category - 118 Magpie Set",
             fontsize=13, fontweight='bold')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig("figures/phase3_enhanced/06E_importance_by_category_enhanced.png", dpi=150, bbox_inches="tight")
print("  Saved: figures/phase3_enhanced/06E_importance_by_category_enhanced.png")


# =============================================================================
# STEP 3.2: PHYSICAL INTERPRETATION
# =============================================================================
print("\n" + "=" * 70)
print("[STEP 3.2] PHYSICAL INTERPRETATION - Feature Families")
print("=" * 70)

INTERPRETATIONS_NEW = {
    "stoichiometric": "Compositional complexity - the Lp norms capture how unevenly "
                      "the constituent elements are distributed across the composition. "
                      "Compositions close to a binary oxide have higher Lp norms than "
                      "equimolar multicomponent mixtures (Ward et al., 2016).",
    "electronic_structure": "Orbital character - the fraction of s/p/d/f valence electrons "
                            "encodes whether the compound is dominated by main-group "
                            "(s/p) or transition-metal (d) bonding.",
    "ionic_compound": "Ionic character - the maximum electronegativity difference captures "
                      "bond polarity and electrostatic stabilization in multicomponent "
                      "oxides (Hautier et al., 2014).",
}

for cat, interp in INTERPRETATIONS_NEW.items():
    print(f"\n  * {cat.replace('_', ' ').title()}:")
    words = interp.split()
    line = "    "
    for w in words:
        if len(line) + len(w) > 75:
            print(line)
            line = "    "
        line += w + " "
    print(line)


# =============================================================================
# STEP 3.3: HYPERPARAMETER TUNING (118 features)
# =============================================================================
print("\n" + "=" * 70)
print("[STEP 3.3] HYPERPARAMETER TUNING - 118 Features")
print("=" * 70)

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
}

grid_results = []
total_combos = 1
for v in param_grid.values():
    total_combos *= len(v)
print(f"  Total combinations: {total_combos}")

t0 = time()
combo_idx = 0
for n_est, depth, split, leaf in product(
    param_grid['n_estimators'], param_grid['max_depth'],
    param_grid['min_samples_split'], param_grid['min_samples_leaf']
):
    combo_idx += 1
    model = RandomForestRegressor(
        n_estimators=n_est, max_depth=depth,
        min_samples_split=split, min_samples_leaf=leaf,
        max_features='sqrt', random_state=RNG_SEED, n_jobs=-1
    )
    model.fit(X_train, y_train)

    mae_tr = mean_absolute_error(y_train, model.predict(X_train))
    r2_tr = r2_score(y_train, model.predict(X_train))
    mae_v = mean_absolute_error(y_val, model.predict(X_val))
    r2_v = r2_score(y_val, model.predict(X_val))

    grid_results.append({
        'n_estimators': n_est, 'max_depth': depth,
        'min_samples_split': split, 'min_samples_leaf': leaf,
        'mae_train': mae_tr, 'mae_val': mae_v,
        'r2_train': r2_tr, 'r2_val': r2_v,
        'overfit_gap': r2_tr - r2_v
    })

    if combo_idx <= 2 or combo_idx % 36 == 0 or combo_idx == total_combos:
        depth_str = str(depth) if depth else "None"
        print(f"  [{combo_idx:3d}/{total_combos}] n={n_est}, d={depth_str}, "
              f"s={split}, l={leaf} -> Val MAE={mae_v:.4f}")

dt = time() - t0
print(f"\n  Grid search completed in {dt:.1f}s")

df_grid = pd.DataFrame(grid_results)
best_idx = df_grid['mae_val'].idxmin()
best = df_grid.loc[best_idx]

print(f"\n  BEST RF hyperparameters (by validation MAE):")
print(f"    n_estimators      = {int(best['n_estimators'])}")
depth_display = int(best['max_depth']) if pd.notna(best['max_depth']) else None
print(f"    max_depth         = {depth_display}")
print(f"    min_samples_split = {int(best['min_samples_split'])}")
print(f"    min_samples_leaf  = {int(best['min_samples_leaf'])}")
print(f"    Val MAE = {best['mae_val']:.4f}, Val R² = {best['r2_val']:.4f}")

df_grid.to_csv("data/processed/grid_search_results_enhanced.csv", index=False)

best_params = {
    'n_estimators': int(best['n_estimators']),
    'max_depth': int(best['max_depth']) if pd.notna(best['max_depth']) else None,
    'min_samples_split': int(best['min_samples_split']),
    'min_samples_leaf': int(best['min_samples_leaf']),
    'max_features': 'sqrt',
    'random_state': RNG_SEED,
    'n_jobs': -1,
}

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
for n_est in param_grid['n_estimators']:
    subset = df_grid[(df_grid['n_estimators'] == n_est) &
                     (df_grid['min_samples_split'] == int(best['min_samples_split'])) &
                     (df_grid['min_samples_leaf'] == int(best['min_samples_leaf']))]
    depths = subset['max_depth'].fillna(999).values
    sort_idx = np.argsort(depths)
    depth_labels = [str(int(d)) if d < 999 else 'None' for d in depths[sort_idx]]

    ax.plot(range(len(depth_labels)), subset['mae_train'].values[sort_idx],
            'o--', alpha=0.7, label=f'Train (n={n_est})', markersize=5)
    ax.plot(range(len(depth_labels)), subset['mae_val'].values[sort_idx],
            's-', alpha=0.7, label=f'Val (n={n_est})', markersize=5)

ax.set_xticks(range(len(depth_labels)))
ax.set_xticklabels(depth_labels)
ax.set_xlabel("max_depth", fontsize=12)
ax.set_ylabel("MAE (eV/atom)", fontsize=12)
ax.set_title("Overfitting Diagnostic - 118 Features\nTraining vs Validation MAE", fontsize=11)
ax.legend(fontsize=8, ncol=2)

ax = axes[1]
ax.scatter(df_grid['r2_train'], df_grid['r2_val'], alpha=0.5, s=20, c='#2196F3')
ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='No overfitting')
ax.set_xlabel("Training R²", fontsize=12)
ax.set_ylabel("Validation R²", fontsize=12)
ax.set_title("Train vs Val R² - 118 Features", fontsize=11)
ax.legend()

plt.tight_layout()
plt.savefig("figures/phase3_enhanced/07E_overfitting_diagnostic_enhanced.png", dpi=150, bbox_inches="tight")
print("  Saved: figures/phase3_enhanced/07E_overfitting_diagnostic_enhanced.png")

print("\n  5-Fold Cross-Validation on best RF and GBR configurations...")
kf = KFold(n_splits=5, shuffle=True, random_state=RNG_SEED)
cv_mae, cv_r2 = [], []
cv_mae_gbr, cv_r2_gbr = [], []

for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train)):
    X_tr_cv, X_val_cv = X_train[tr_idx], X_train[val_idx]
    y_tr_cv, y_val_cv = y_train[tr_idx], y_train[val_idx]

    model_cv = RandomForestRegressor(**best_params)
    model_cv.fit(X_tr_cv, y_tr_cv)
    y_pred_cv = model_cv.predict(X_val_cv)
    fold_mae = mean_absolute_error(y_val_cv, y_pred_cv)
    fold_r2 = r2_score(y_val_cv, y_pred_cv)
    cv_mae.append(fold_mae)
    cv_r2.append(fold_r2)

    gbr_cv = GradientBoostingRegressor(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        min_samples_split=5, min_samples_leaf=3, subsample=0.8,
        random_state=RNG_SEED,
    )
    gbr_cv.fit(X_tr_cv, y_tr_cv)
    y_pred_gbr_cv = gbr_cv.predict(X_val_cv)
    fold_mae_gbr = mean_absolute_error(y_val_cv, y_pred_gbr_cv)
    fold_r2_gbr = r2_score(y_val_cv, y_pred_gbr_cv)
    cv_mae_gbr.append(fold_mae_gbr)
    cv_r2_gbr.append(fold_r2_gbr)

    winner = "RF" if fold_mae < fold_mae_gbr else "GBR"
    print(f"    Fold {fold+1}: RF MAE = {fold_mae:.4f} (R² = {fold_r2:.4f}) | "
          f"GBR MAE = {fold_mae_gbr:.4f} (R² = {fold_r2_gbr:.4f})  [Best: {winner}]")

print(f"\n    RF  CV Mean MAE = {np.mean(cv_mae):.4f} ± {np.std(cv_mae):.4f} | "
      f"Mean R² = {np.mean(cv_r2):.4f} ± {np.std(cv_r2):.4f}")
print(f"    GBR CV Mean MAE = {np.mean(cv_mae_gbr):.4f} ± {np.std(cv_mae_gbr):.4f} | "
      f"Mean R² = {np.mean(cv_r2_gbr):.4f} ± {np.std(cv_r2_gbr):.4f}")


# =============================================================================
# STEP 3.4: MODEL COMPARISON - RF vs GBR vs XGBoost
# =============================================================================
print("\n" + "=" * 70)
print("[STEP 3.4] THREE-MODEL COMPARISON: RF vs GBR vs XGBoost")
print("=" * 70)

print("\n  Training GradientBoostingRegressor...")
gbr = GradientBoostingRegressor(
    n_estimators=300, max_depth=5, learning_rate=0.05,
    min_samples_split=5, min_samples_leaf=3, subsample=0.8,
    random_state=RNG_SEED,
)
t0 = time()
gbr.fit(X_train, y_train)
dt_gbr = time() - t0

mae_gbr_train = mean_absolute_error(y_train, gbr.predict(X_train))
r2_gbr_train = r2_score(y_train, gbr.predict(X_train))
mae_gbr_val = mean_absolute_error(y_val, gbr.predict(X_val))
r2_gbr_val = r2_score(y_val, gbr.predict(X_val))

print(f"  GBR Training time: {dt_gbr:.1f}s")
print(f"  GBR Train: MAE = {mae_gbr_train:.4f}, R² = {r2_gbr_train:.4f}")
print(f"  GBR Val:   MAE = {mae_gbr_val:.4f}, R² = {r2_gbr_val:.4f}")

if HAS_XGBOOST:
    print("\n  Training XGBRegressor...")
    xgb = XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=RNG_SEED,
        n_jobs=-1,
        verbosity=0,
    )
    t0 = time()
    xgb.fit(X_train, y_train)
    dt_xgb = time() - t0

    mae_xgb_train = mean_absolute_error(y_train, xgb.predict(X_train))
    r2_xgb_train = r2_score(y_train, xgb.predict(X_train))
    mae_xgb_val = mean_absolute_error(y_val, xgb.predict(X_val))
    r2_xgb_val = r2_score(y_val, xgb.predict(X_val))

    print(f"  XGB Training time: {dt_xgb:.1f}s")
    print(f"  XGB Train: MAE = {mae_xgb_train:.4f}, R² = {r2_xgb_train:.4f}")
    print(f"  XGB Val:   MAE = {mae_xgb_val:.4f}, R² = {r2_xgb_val:.4f}")

print(f"\n  {'='*65}")
print(f"  MODEL COMPARISON SUMMARY (Validation Set)")
print(f"  {'='*65}")
print(f"  {'Model':<25s} {'MAE':>10s} {'R²':>10s} {'Overfit Gap':>12s}")
print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*12}")

dummy = DummyRegressor(strategy="mean")
dummy.fit(X_train, y_train)
mae_dummy = mean_absolute_error(y_val, dummy.predict(X_val))
r2_dummy = r2_score(y_val, dummy.predict(X_val))
print(f"  {'Dummy (baseline)':<25s} {mae_dummy:10.4f} {r2_dummy:10.4f} {'N/A':>12s}")

mae_rf_base_val = mean_absolute_error(y_val, rf_baseline.predict(X_val))
r2_rf_base_val = r2_score(y_val, rf_baseline.predict(X_val))
r2_rf_base_tr = r2_score(y_train, rf_baseline.predict(X_train))
print(f"  {'RF (default)':<25s} {mae_rf_base_val:10.4f} {r2_rf_base_val:10.4f} {r2_rf_base_tr - r2_rf_base_val:12.4f}")
print(f"  {'RF (tuned)':<25s} {best['mae_val']:10.4f} {best['r2_val']:10.4f} {best['overfit_gap']:12.4f}")
print(f"  {'Gradient Boosting':<25s} {mae_gbr_val:10.4f} {r2_gbr_val:10.4f} {r2_gbr_train - r2_gbr_val:12.4f}")
if HAS_XGBOOST:
    print(f"  {'XGBoost':<25s} {mae_xgb_val:10.4f} {r2_xgb_val:10.4f} {r2_xgb_train - r2_xgb_val:12.4f}")
print(f"  {'='*65}")


# =============================================================================
# STEP 3.5: FINAL EVALUATION ON THE HELD-OUT TEST SET
# =============================================================================
print("\n" + "=" * 70)
print("[STEP 3.5] FINAL EVALUATION ON HELD-OUT TEST SET")
print("=" * 70)

candidates = {
    'RF (tuned)': best['mae_val'],
    'GBR': mae_gbr_val,
}
if HAS_XGBOOST:
    candidates['XGBoost'] = mae_xgb_val

best_model_name = min(candidates, key=candidates.get)
print(f"\n  Validation MAE comparison:")
for name, mae_v in candidates.items():
    marker = " <- BEST" if name == best_model_name else ""
    print(f"    {name:<20s}: {mae_v:.4f}{marker}")

if best_model_name == 'RF (tuned)':
    final_model = RandomForestRegressor(**best_params)
elif best_model_name == 'GBR':
    final_model = GradientBoostingRegressor(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        min_samples_split=5, min_samples_leaf=3, subsample=0.8,
        random_state=RNG_SEED,
    )
else:
    final_model = XGBRegressor(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
        gamma=0.1, reg_alpha=0.1, reg_lambda=1.0,
        random_state=RNG_SEED, n_jobs=-1, verbosity=0,
    )

print(f"\n  Re-training {best_model_name} on combined train+val data...")
X_trainval = np.vstack([X_train, X_val])
y_trainval = np.concatenate([y_train, y_val])
final_model.fit(X_trainval, y_trainval)

y_pred_test = final_model.predict(X_test)
mae_test = mean_absolute_error(y_test, y_pred_test)
r2_test = r2_score(y_test, y_pred_test)
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))

dummy_test = DummyRegressor(strategy="mean")
dummy_test.fit(X_trainval, y_trainval)
mae_dummy_test = mean_absolute_error(y_test, dummy_test.predict(X_test))
improvement_test = (mae_dummy_test - mae_test) / mae_dummy_test * 100

print("\n  Evaluating all three models on the held-out test set...")
all_test_results = {}

rf_final = RandomForestRegressor(**best_params)
rf_final.fit(X_trainval, y_trainval)
y_pred_rf = rf_final.predict(X_test)
all_test_results['RF (tuned)'] = {
    'mae': mean_absolute_error(y_test, y_pred_rf),
    'rmse': np.sqrt(mean_squared_error(y_test, y_pred_rf)),
    'r2': r2_score(y_test, y_pred_rf),
    'preds': y_pred_rf,
}

gbr_final = GradientBoostingRegressor(
    n_estimators=300, max_depth=5, learning_rate=0.05,
    min_samples_split=5, min_samples_leaf=3, subsample=0.8,
    random_state=RNG_SEED,
)
gbr_final.fit(X_trainval, y_trainval)
y_pred_gbr = gbr_final.predict(X_test)
all_test_results['GBR'] = {
    'mae': mean_absolute_error(y_test, y_pred_gbr),
    'rmse': np.sqrt(mean_squared_error(y_test, y_pred_gbr)),
    'r2': r2_score(y_test, y_pred_gbr),
    'preds': y_pred_gbr,
}

if HAS_XGBOOST:
    xgb_final = XGBRegressor(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
        gamma=0.1, reg_alpha=0.1, reg_lambda=1.0,
        random_state=RNG_SEED, n_jobs=-1, verbosity=0,
    )
    xgb_final.fit(X_trainval, y_trainval)
    y_pred_xgb = xgb_final.predict(X_test)
    all_test_results['XGBoost'] = {
        'mae': mean_absolute_error(y_test, y_pred_xgb),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred_xgb)),
        'r2': r2_score(y_test, y_pred_xgb),
        'preds': y_pred_xgb,
    }

print(f"\n  All models - TEST set performance (n = {len(y_test)}):")
print(f"  {'Model':<20s} {'MAE':>10s} {'RMSE':>10s} {'R²':>10s}")
print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*10}")
for name, res in all_test_results.items():
    print(f"  {name:<20s} {res['mae']:10.4f} {res['rmse']:10.4f} {res['r2']:10.4f}")

pred_df = pd.DataFrame({
    "formula_pretty": df[test_mask]["formula_pretty"].values,
    "y_test": y_test,
    "y_pred_dummy": dummy_test.predict(X_test),
    "y_pred_rf": all_test_results["RF (tuned)"]["preds"],
    "y_pred_gbr": all_test_results["GBR"]["preds"],
})
if HAS_XGBOOST and "XGBoost" in all_test_results:
    pred_df["y_pred_xgb"] = all_test_results["XGBoost"]["preds"]
pred_df.to_csv("results/test_predictions.csv", index=False)

summary_rows = [
    {
        "model": "DummyRegressor",
        "val_mae": mae_dummy,
        "val_r2": r2_dummy,
        "test_mae": mae_dummy_test,
        "test_rmse": np.sqrt(mean_squared_error(y_test, dummy_test.predict(X_test))),
        "test_r2": r2_score(y_test, dummy_test.predict(X_test)),
    },
    {
        "model": "RF_default_118",
        "val_mae": mae_rf_base_val,
        "val_r2": r2_rf_base_val,
        "test_mae": np.nan,
        "test_rmse": np.nan,
        "test_r2": np.nan,
    },
    {
        "model": "RF_tuned_118",
        "val_mae": best["mae_val"],
        "val_r2": best["r2_val"],
        "test_mae": all_test_results["RF (tuned)"]["mae"],
        "test_rmse": all_test_results["RF (tuned)"]["rmse"],
        "test_r2": all_test_results["RF (tuned)"]["r2"],
    },
    {
        "model": "GBR_118",
        "val_mae": mae_gbr_val,
        "val_r2": r2_gbr_val,
        "test_mae": all_test_results["GBR"]["mae"],
        "test_rmse": all_test_results["GBR"]["rmse"],
        "test_r2": all_test_results["GBR"]["r2"],
    },
]
if HAS_XGBOOST and "XGBoost" in all_test_results:
    summary_rows.append({
        "model": "XGBoost_118",
        "val_mae": mae_xgb_val,
        "val_r2": r2_xgb_val,
        "test_mae": all_test_results["XGBoost"]["mae"],
        "test_rmse": all_test_results["XGBoost"]["rmse"],
        "test_r2": all_test_results["XGBoost"]["r2"],
    })
pd.DataFrame(summary_rows).to_csv("results/final_metrics_summary.csv", index=False)
print("  Saved: results/test_predictions.csv and results/final_metrics_summary.csv")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

ax = axes[0]
ax.scatter(y_test, y_pred_test, alpha=0.5, s=20, c='#4CAF50', edgecolors='none')
lims = [0, max(y_test.max(), y_pred_test.max()) * 1.05]
ax.plot(lims, lims, 'k--', linewidth=1, label='Ideal (y=x)')
coeffs = np.polyfit(y_test, y_pred_test, 1)
x_fit = np.linspace(lims[0], lims[1], 100)
ax.plot(x_fit, np.poly1d(coeffs)(x_fit), 'r-', alpha=0.7, linewidth=1, label='Linear fit')
ax.set_xlim(lims); ax.set_ylim(lims)
ax.set_xlabel("Actual $E_{hull}$ (eV/atom)", fontsize=12)
ax.set_ylabel("Predicted $E_{hull}$ (eV/atom)", fontsize=12)
ax.set_title(f"Test Set - {best_model_name} (118 Features)\n"
             f"MAE={mae_test:.4f}, R²={r2_test:.4f}", fontsize=11)
ax.legend(fontsize=9)
ax.set_aspect('equal')

residuals_test = y_pred_test - y_test
ax = axes[1]
ax.hist(residuals_test, bins=40, color='#4CAF50', edgecolor='white', alpha=0.8)
ax.axvline(x=0, color='k', linestyle='--', linewidth=1)
ax.set_xlabel("Residual (eV/atom)", fontsize=12)
ax.set_ylabel("Count", fontsize=12)
ax.set_title(f"Test Residual Distribution\n"
             f"Mean={residuals_test.mean():.4f}, Std={residuals_test.std():.4f}", fontsize=11)

plt.tight_layout()
plt.savefig("figures/phase3_enhanced/08E_final_test_parity_enhanced.png", dpi=150, bbox_inches="tight")
print("\n  Saved: figures/phase3_enhanced/08E_final_test_parity_enhanced.png")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
model_names_bar = ['Dummy\n(baseline)', 'RF\n(tuned)', 'GBR']
mae_vals_bar = [mae_dummy, best['mae_val'], mae_gbr_val]
colors_bar = ['#9E9E9E', '#1976D2', '#FF9800']

if HAS_XGBOOST:
    model_names_bar.append('XGBoost')
    mae_vals_bar.append(mae_xgb_val)
    colors_bar.append('#4CAF50')

bars = ax.bar(model_names_bar, mae_vals_bar, color=colors_bar, edgecolor='white', width=0.6)
ax.set_ylabel("Validation MAE (eV/atom)", fontsize=12)
ax.set_title("Model Comparison: Validation MAE\n(118-Feature Set)", fontsize=13, fontweight='bold')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
for bar, val in zip(bars, mae_vals_bar):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
            f'{val:.4f}', ha='center', va='bottom', fontsize=10)

ax = axes[1]
test_names = list(all_test_results.keys())
test_maes = [all_test_results[n]['mae'] for n in test_names]
test_colors = ['#1976D2', '#FF9800']
if HAS_XGBOOST:
    test_colors.append('#4CAF50')

bars2 = ax.bar(test_names, test_maes, color=test_colors[:len(test_names)],
               edgecolor='white', width=0.5)
ax.set_ylabel("Test MAE (eV/atom)", fontsize=12)
ax.set_title("Final Test Performance: All Models\n(Trained on train+val, evaluated once)", fontsize=13, fontweight='bold')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
for bar, val in zip(bars2, test_maes):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
            f'{val:.4f}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig("figures/phase3_enhanced/09E_model_comparison_enhanced.png", dpi=150, bbox_inches="tight")
print("  Saved: figures/phase3_enhanced/09E_model_comparison_enhanced.png")

n_models = len(all_test_results)
fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5.5))
if n_models == 1:
    axes = [axes]

model_colors = {'RF (tuned)': '#1976D2', 'GBR': '#FF9800', 'XGBoost': '#4CAF50'}

for ax, (name, res) in zip(axes, all_test_results.items()):
    color = model_colors.get(name, '#666666')
    ax.scatter(y_test, res['preds'], alpha=0.4, s=15, c=color, edgecolors='none')
    lims = [0, max(y_test.max(), max(res['preds'])) * 1.05]
    ax.plot(lims, lims, 'k--', linewidth=1, alpha=0.5)
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("Actual $E_{hull}$ (eV/atom)", fontsize=11)
    ax.set_ylabel("Predicted $E_{hull}$ (eV/atom)", fontsize=11)
    ax.set_title(f"{name}\nMAE={res['mae']:.4f}, R²={res['r2']:.4f}", fontsize=11)
    ax.set_aspect('equal')

plt.suptitle("Test Set Parity - Three-Model Comparison (118 Features)",
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig("figures/phase3_enhanced/10E_multi_model_parity.png", dpi=150, bbox_inches="tight")
print("  Saved: figures/phase3_enhanced/10E_multi_model_parity.png")

print("\n" + "=" * 70)
print("PHASE 3 ENHANCED - SUMMARY")
print("=" * 70)
print(f"  Features           : {n_total_features} (105 elemental + {n_stoich} stoich + {n_elec} elec + {n_ionic} ionic)")
print(f"  5-Fold CV (RF)     : MAE = {np.mean(cv_mae):.4f} ± {np.std(cv_mae):.4f}, R² = {np.mean(cv_r2):.4f} ± {np.std(cv_r2):.4f}")
print(f"  5-Fold CV (GBR)    : MAE = {np.mean(cv_mae_gbr):.4f} ± {np.std(cv_mae_gbr):.4f}, R² = {np.mean(cv_r2_gbr):.4f} ± {np.std(cv_r2_gbr):.4f}")
print(f"  Test Set Baseline  : MAE = {mae_dummy_test:.4f} eV/atom")
print(f"  Test Set RF (tuned): MAE = {all_test_results['RF (tuned)']['mae']:.4f} eV/atom, RMSE = {all_test_results['RF (tuned)']['rmse']:.4f}, R² = {all_test_results['RF (tuned)']['r2']:.4f} (-{improvement_test:.1f}% vs baseline)")
print(f"  Test Set GBR       : MAE = {all_test_results['GBR']['mae']:.4f} eV/atom, RMSE = {all_test_results['GBR']['rmse']:.4f}, R² = {all_test_results['GBR']['r2']:.4f}")
if HAS_XGBOOST and 'XGBoost' in all_test_results:
    print(f"  Test Set XGBoost   : MAE = {all_test_results['XGBoost']['mae']:.4f} eV/atom, RMSE = {all_test_results['XGBoost']['rmse']:.4f}, R² = {all_test_results['XGBoost']['r2']:.4f}")
print("=" * 70)