"""Step 10 - Phase 3: interpretation, hyperparameter tuning and final test (105 features).

Steps
  3.1  Feature importance (MDI) extraction and plots
  3.2  Physical interpretation: linking features to materials science
  3.3  Hyperparameter tuning (grid search) and 5-fold cross-validation
  3.4  Comparison with Gradient Boosting
  3.5  Final evaluation on the held-out test set

Follows Wang et al., Chem. Mater. 2020, 32, 4954-4965.

Outputs
  data/processed/feature_importances.csv, data/processed/grid_search_results.csv
  figures/phase3/05..09_*.png
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
    featurize_composition,
)

# Reproducibility
RNG_SEED = 42
np.random.seed(RNG_SEED)

# =============================================================================
# Rebuild the Phase 2 pipeline (load, featurize, split, scale)
# =============================================================================
print("=" * 70)
print("PHASE 3: MODEL INTERPRETATION, TUNING & FINAL EVALUATION")
print("=" * 70)
print("\n[SETUP] Reproducing Phase 2 pipeline...")

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
        feats = featurize_composition(comp)
        feature_records.append(feats)
    except (KeyError, ZeroDivisionError):
        skipped.append(idx)

df = df.drop(index=skipped).reset_index(drop=True)
df_features = pd.DataFrame(feature_records)
feature_names = list(df_features.columns)
y = df['energy_above_hull'].values

# Same split as Phase 2
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

print(f"  Data: {len(X_train)} train / {len(X_val)} val / {len(X_test)} test")
print(f"  Features: {X_train.shape[1]}")

os.makedirs("figures/phase3", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)


# =============================================================================
# STEP 3.1: FEATURE IMPORTANCE
# =============================================================================
print("\n" + "=" * 70)
print("[STEP 3.1] FEATURE IMPORTANCE ANALYSIS")
print("=" * 70)

# Baseline RF, identical to the Phase 2 model
rf_baseline = RandomForestRegressor(
    n_estimators=200, max_depth=None, min_samples_split=5,
    min_samples_leaf=2, max_features='sqrt',
    random_state=RNG_SEED, n_jobs=-1
)
rf_baseline.fit(X_train, y_train)

# MDI = Mean Decrease in Impurity
importances = rf_baseline.feature_importances_
feat_imp = pd.DataFrame({
    'feature': feature_names,
    'importance': importances
}).sort_values('importance', ascending=False).reset_index(drop=True)

print("\n  Top 20 most important features:")
print("  " + "-" * 50)
for i, row in feat_imp.head(20).iterrows():
    bar = "█" * int(row['importance'] / feat_imp['importance'].max() * 30)
    print(f"  {i+1:2d}. {row['feature']:30s} {row['importance']:.4f} {bar}")

# Full ranking
feat_imp.to_csv("data/processed/feature_importances.csv", index=False)

# ── Figure 5: top-20 feature importances ─────────────────────────────────────
top_n = 20
top_feats = feat_imp.head(top_n)

fig, ax = plt.subplots(figsize=(10, 8))
colors = plt.cm.viridis(np.linspace(0.3, 0.9, top_n))[::-1]
bars = ax.barh(range(top_n), top_feats['importance'].values[::-1],
               color=colors, edgecolor='white', linewidth=0.5)
ax.set_yticks(range(top_n))
ax.set_yticklabels(top_feats['feature'].values[::-1], fontsize=10)
ax.set_xlabel("Feature Importance (MDI)", fontsize=12)
ax.set_title("Top 20 Feature Importances — Random Forest\n(Mean Decrease in Impurity)",
             fontsize=13, fontweight='bold')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig("figures/phase3/05_feature_importance_RF.png", dpi=150, bbox_inches="tight")
print("\n  Saved: figures/phase3/05_feature_importance_RF.png")


# =============================================================================
# STEP 3.2: PHYSICAL INTERPRETATION
# =============================================================================
print("\n" + "=" * 70)
print("[STEP 3.2] PHYSICAL INTERPRETATION")
print("=" * 70)

# What each elemental property means for stability
INTERPRETATIONS = {
    "electronegativity": "Bond ionicity — large electronegativity differences between "
                         "dopant and host indicate ionic character mismatch, destabilizing "
                         "the lattice. Central to Pauling's rules for crystal stability.",
    "atomic_radius": "Size mismatch — atomic radius differences drive lattice strain "
                     "when dopants substitute for host cations. Goldschmidt tolerance "
                     "factor-like effect in oxide frameworks.",
    "ionization_energy": "Ease of electron removal — reflects oxidation state flexibility. "
                         "Dopants with very different ionization energies may create "
                         "charge compensation defects that affect stability.",
    "electron_affinity": "Tendency to accept electrons — related to reducibility. "
                         "Important for n-type TCO dopants (e.g., Al in ZnO) where "
                         "the dopant must donate electrons to the conduction band.",
    "n_valence": "Total valence electrons — governs bonding capacity and coordination "
                 "preference. Mismatched valence creates charge imbalance requiring "
                 "compensating defects (vacancies, interstitials).",
    "n_d": "d-electron count — transition metal dopants with partially filled "
           "d-orbitals introduce crystal field effects and possible magnetic "
           "ordering that can stabilize or destabilize the structure.",
    "n_f": "f-electron count — rare earth dopants with f-electrons introduce "
           "strong correlation effects. Important for lanthanide-doped TCOs.",
    "group": "Periodic table group — encodes the number of valence electrons and "
             "typical oxidation states. Directly relates to chemical bonding type.",
    "row": "Periodic table period — correlates with atomic size and orbital extent. "
           "Captures the difference between 3d, 4d, and 5d transition metals.",
    "density_elem": "Elemental density — proxy for atomic packing and metallic/covalent "
                    "character. Heavy elements pack differently in oxide frameworks.",
    "melting_point": "Melting point — proxy for bond strength. Elements with high melting "
                     "points form stronger bonds, potentially stabilizing the compound.",
    "atomic_number": "Atomic number — captures general trends across the periodic table. "
                     "Often correlates with multiple physical properties simultaneously.",
    "atomic_mass": "Atomic mass — related to vibrational properties (phonons). Heavy "
                   "atoms reduce phonon frequencies, affecting thermodynamic stability.",
    "n_s": "s-electron count — relevant for main-group elements. Paired s-electrons "
           "in post-transition metals (In, Sn, Pb) create the 'lone pair' effect.",
    "n_p": "p-electron count — governs bonding in main-group elements. p-orbital "
           "hybridization with oxygen 2p determines the conduction band character in TCOs.",
}

print("\n  Key feature → Physical mechanism mapping:")
print("  " + "=" * 65)
for i, row in feat_imp.head(10).iterrows():
    feat = row['feature']
    # Find the base property name
    base_prop = None
    for prop in PROP_NAMES:
        if feat.startswith(prop):
            base_prop = prop
            break
    stat_type = feat.replace(base_prop + "_", "") if base_prop else "unknown"

    interp = INTERPRETATIONS.get(base_prop, "No interpretation available.")
    print(f"\n  {i+1}. {feat} (importance: {row['importance']:.4f})")
    print(f"     Statistic: {stat_type}")
    print(f"     Physics: {interp[:120]}...")

# ── Figure 6: importance by property category ────────────────────────────────
category_importance = {}
for _, row in feat_imp.iterrows():
    for prop in PROP_NAMES:
        if row['feature'].startswith(prop):
            category_importance[prop] = category_importance.get(prop, 0) + row['importance']
            break

cat_imp = pd.Series(category_importance).sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(9, 7))
colors_cat = plt.cm.plasma(np.linspace(0.2, 0.85, len(cat_imp)))
ax.barh(range(len(cat_imp)), cat_imp.values, color=colors_cat,
        edgecolor='white', linewidth=0.5)
ax.set_yticks(range(len(cat_imp)))
ax.set_yticklabels([n.replace('_', ' ').title() for n in cat_imp.index], fontsize=10)
ax.set_xlabel("Cumulative Feature Importance", fontsize=12)
ax.set_title("Feature Importance by Elemental Property Category\n"
             "(Sum of all statistics per property)",
             fontsize=13, fontweight='bold')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig("figures/phase3/06_importance_by_category.png", dpi=150, bbox_inches="tight")
print("\n  Saved: figures/phase3/06_importance_by_category.png")


# =============================================================================
# STEP 3.3: HYPERPARAMETER TUNING
# =============================================================================
print("\n" + "=" * 70)
print("[STEP 3.3] HYPERPARAMETER TUNING")
print("=" * 70)

# ── 3.3a: grid search for the Random Forest ──────────────────────────────────
print("\n  Grid search for RandomForest hyperparameters...")
print("  Paper: 'train numerous models with different hyperparameters")
print("  using the same training set, evaluate on the same validation set'")

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
print(f"\n  Total combinations: {total_combos}")
print(f"  {'n_est':>6s} {'depth':>6s} {'split':>6s} {'leaf':>6s} | {'MAE_tr':>8s} {'MAE_val':>8s} {'R2_tr':>8s} {'R2_val':>8s}")
print(f"  {'-'*6} {'-'*6} {'-'*6} {'-'*6} | {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

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

    depth_str = str(depth) if depth else "None"
    grid_results.append({
        'n_estimators': n_est, 'max_depth': depth,
        'min_samples_split': split, 'min_samples_leaf': leaf,
        'mae_train': mae_tr, 'mae_val': mae_v,
        'r2_train': r2_tr, 'r2_val': r2_v,
        'overfit_gap': r2_tr - r2_v
    })

    # Print only the first few combinations and periodic updates
    if combo_idx <= 3 or combo_idx % 36 == 0 or combo_idx == total_combos:
        print(f"  {n_est:>6d} {depth_str:>6s} {split:>6d} {leaf:>6d} | "
              f"{mae_tr:8.4f} {mae_v:8.4f} {r2_tr:8.4f} {r2_v:8.4f}")

dt = time() - t0
print(f"\n  Grid search completed in {dt:.1f}s")

df_grid = pd.DataFrame(grid_results)

# Best configuration by validation MAE
best_idx = df_grid['mae_val'].idxmin()
best = df_grid.loc[best_idx]
print(f"\n  BEST hyperparameters (by validation MAE):")
print(f"    n_estimators     = {int(best['n_estimators'])}")
print(f"    max_depth        = {best['max_depth']}")
print(f"    min_samples_split = {int(best['min_samples_split'])}")
print(f"    min_samples_leaf  = {int(best['min_samples_leaf'])}")
print(f"    Val MAE = {best['mae_val']:.4f}, Val R² = {best['r2_val']:.4f}")
print(f"    Train MAE = {best['mae_train']:.4f}, Train R² = {best['r2_train']:.4f}")
print(f"    Overfit gap (R²): {best['overfit_gap']:.4f}")

df_grid.to_csv("data/processed/grid_search_results.csv", index=False)

# ── Figure 7: overfitting diagnostic ─────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Panel 1: train vs validation MAE by max_depth
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
ax.set_title("Overfitting Diagnostic:\nTraining vs Validation MAE", fontsize=11)
ax.legend(fontsize=8, ncol=2)

# Panel 2: train vs validation R²
ax = axes[1]
ax.scatter(df_grid['r2_train'], df_grid['r2_val'], alpha=0.5, s=20, c='#2196F3')
ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='No overfitting')
ax.set_xlabel("Training R²", fontsize=12)
ax.set_ylabel("Validation R²", fontsize=12)
ax.set_title("Train vs Val R² Across All Configurations\n(closer to diagonal = less overfitting)",
             fontsize=11)
ax.legend()

plt.tight_layout()
plt.savefig("figures/phase3/07_overfitting_diagnostic.png", dpi=150, bbox_inches="tight")
print("  Saved: figures/phase3/07_overfitting_diagnostic.png")


# ── 3.3b: cross-validation of the best RF ────────────────────────────────────
print("\n  5-Fold Cross-Validation on best RF configuration...")
print("  Paper: 'k-fold CV gives a more accurate estimate of the model's")
print("  true predictive performance compared to evaluating on one single split'")

best_params = {
    'n_estimators': int(best['n_estimators']),
    'max_depth': int(best['max_depth']) if pd.notna(best['max_depth']) else None,
    'min_samples_split': int(best['min_samples_split']),
    'min_samples_leaf': int(best['min_samples_leaf']),
    'max_features': 'sqrt',
    'random_state': RNG_SEED,
    'n_jobs': -1,
}

# CV on the training data only (the test set stays untouched)
kf = KFold(n_splits=5, shuffle=True, random_state=RNG_SEED)
cv_mae = []
cv_r2 = []

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
    print(f"    Fold {fold+1}: MAE = {fold_mae:.4f}, R² = {fold_r2:.4f}")

print(f"\n    CV Mean MAE = {np.mean(cv_mae):.4f} ± {np.std(cv_mae):.4f}")
print(f"    CV Mean R²  = {np.mean(cv_r2):.4f} ± {np.std(cv_r2):.4f}")


# =============================================================================
# STEP 3.4: GRADIENT BOOSTING COMPARISON
# =============================================================================
print("\n" + "=" * 70)
print("[STEP 3.4] GRADIENT BOOSTING COMPARISON")
print("=" * 70)

print("\n  Training GradientBoostingRegressor for comparison...")
gbr = GradientBoostingRegressor(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    min_samples_split=5,
    min_samples_leaf=3,
    subsample=0.8,
    random_state=RNG_SEED,
)
t0 = time()
gbr.fit(X_train, y_train)
dt_gbr = time() - t0

y_pred_gbr_train = gbr.predict(X_train)
y_pred_gbr_val = gbr.predict(X_val)

mae_gbr_train = mean_absolute_error(y_train, y_pred_gbr_train)
r2_gbr_train = r2_score(y_train, y_pred_gbr_train)
mae_gbr_val = mean_absolute_error(y_val, y_pred_gbr_val)
r2_gbr_val = r2_score(y_val, y_pred_gbr_val)

print(f"  Training time: {dt_gbr:.1f}s")
print(f"  Train: MAE = {mae_gbr_train:.4f}, R² = {r2_gbr_train:.4f}")
print(f"  Val:   MAE = {mae_gbr_val:.4f}, R² = {r2_gbr_val:.4f}")


# =============================================================================
# STEP 3.5: FINAL EVALUATION ON THE HELD-OUT TEST SET
# =============================================================================
print("\n" + "=" * 70)
print("[STEP 3.5] FINAL EVALUATION ON HELD-OUT TEST SET")
print("=" * 70)
print("\n  Paper: 'test data may only be used in order to evaluate a model's")
print("  performance as a final step, after the model has been finalized.'")
print("  'Do not evaluate your model on the held-out test data set until")
print("  you have finished tuning your model and it is ready for publication.'")

# Pick the better model on the validation set: tuned RF or GBR
print("\n  Model comparison on VALIDATION set:")
print(f"    RF (tuned):   MAE = {best['mae_val']:.4f}, R² = {best['r2_val']:.4f}")
print(f"    GBR:          MAE = {mae_gbr_val:.4f}, R² = {r2_gbr_val:.4f}")

if best['mae_val'] <= mae_gbr_val:
    print("\n  → Selected: Tuned Random Forest")
    final_model = RandomForestRegressor(**best_params)
    final_model_name = "RandomForest (tuned)"
else:
    print("\n  → Selected: Gradient Boosting Regressor")
    final_model = GradientBoostingRegressor(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        min_samples_split=5, min_samples_leaf=3, subsample=0.8,
        random_state=RNG_SEED,
    )
    final_model_name = "GradientBoosting"

# Recommended practice: retrain the chosen model on train+val combined
print(f"\n  Re-training {final_model_name} on combined train+val data...")
X_trainval = np.vstack([X_train, X_val])
y_trainval = np.concatenate([y_train, y_val])
final_model.fit(X_trainval, y_trainval)

# FINAL test evaluation (done exactly once)
y_pred_test = final_model.predict(X_test)
mae_test = mean_absolute_error(y_test, y_pred_test)
r2_test = r2_score(y_test, y_pred_test)
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))

# Dummy baseline on the test set, for reference
dummy = DummyRegressor(strategy="mean")
dummy.fit(X_trainval, y_trainval)
y_pred_dummy_test = dummy.predict(X_test)
mae_dummy_test = mean_absolute_error(y_test, y_pred_dummy_test)
r2_dummy_test = r2_score(y_test, y_pred_dummy_test)

improvement_test = (mae_dummy_test - mae_test) / mae_dummy_test * 100

print(f"\n  ╔══════════════════════════════════════════════════════╗")
print(f"  ║         FINAL TEST SET RESULTS                       ║")
print(f"  ╠══════════════════════════════════════════════════════╣")
print(f"  ║  Model: {final_model_name:42s}  ║")
print(f"  ║  MAE  = {mae_test:.4f} eV/atom                            ║")
print(f"  ║  RMSE = {rmse_test:.4f} eV/atom                            ║")
print(f"  ║  R²   = {r2_test:.4f}                                    ║")
print(f"  ║  Baseline MAE = {mae_dummy_test:.4f} eV/atom                   ║")
print(f"  ║  Improvement over baseline: {improvement_test:.1f}%                  ║")
print(f"  ╚══════════════════════════════════════════════════════╝")

# ── Figure 8: final test-set parity plot ─────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Panel 1: parity plot
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
ax.set_title(f"Test Set — {final_model_name}\nMAE={mae_test:.4f}, R²={r2_test:.4f}", fontsize=11)
ax.legend(fontsize=9)
ax.set_aspect('equal')

# Panel 2: residual histogram
residuals_test = y_pred_test - y_test
ax = axes[1]
ax.hist(residuals_test, bins=40, color='#4CAF50', edgecolor='white', alpha=0.8)
ax.axvline(x=0, color='k', linestyle='--', linewidth=1)
ax.set_xlabel("Residual (eV/atom)", fontsize=12)
ax.set_ylabel("Count", fontsize=12)
ax.set_title(f"Test Set Residual Distribution\n"
             f"Mean={residuals_test.mean():.4f}, Std={residuals_test.std():.4f}", fontsize=11)

plt.tight_layout()
plt.savefig("figures/phase3/08_final_test_parity.png", dpi=150, bbox_inches="tight")
print("\n  Saved: figures/phase3/08_final_test_parity.png")

# ── Figure 9: model comparison bar chart ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))

model_names = ['Dummy\n(baseline)', 'RF\n(Phase 2)', 'RF\n(tuned)', 'GBR']
mae_vals_all = [mae_dummy_test,
                mean_absolute_error(y_val, rf_baseline.predict(X_val)),
                best['mae_val'],
                mae_gbr_val]
colors_bar = ['#9E9E9E', '#2196F3', '#1976D2', '#FF9800']

bars = ax.bar(model_names, mae_vals_all, color=colors_bar, edgecolor='white', width=0.6)
ax.set_ylabel("Validation MAE (eV/atom)", fontsize=12)
ax.set_title("Model Comparison: Validation MAE", fontsize=13, fontweight='bold')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Value labels on top of the bars
for bar, val in zip(bars, mae_vals_all):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
            f'{val:.4f}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig("figures/phase3/09_model_comparison.png", dpi=150, bbox_inches="tight")
print("  Saved: figures/phase3/09_model_comparison.png")


# =============================================================================
# FINAL SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print("PHASE 3 COMPLETE — SUMMARY")
print("=" * 70)

print(f"""
  Feature Importance:
    Top features: {', '.join(feat_imp.head(5)['feature'].tolist())}
    Top property category: {cat_imp.index[-1].replace('_', ' ').title()}

  Hyperparameter Tuning:
    Searched {total_combos} RF configurations
    Best: n_estimators={int(best['n_estimators'])}, max_depth={best['max_depth']},
          min_samples_split={int(best['min_samples_split'])}, min_samples_leaf={int(best['min_samples_leaf'])}

  Cross-Validation (5-fold):
    MAE = {np.mean(cv_mae):.4f} ± {np.std(cv_mae):.4f}
    R²  = {np.mean(cv_r2):.4f} ± {np.std(cv_r2):.4f}

  Final Test Set Performance ({final_model_name}):
    MAE  = {mae_test:.4f} eV/atom
    RMSE = {rmse_test:.4f} eV/atom
    R²   = {r2_test:.4f}
    Improvement over baseline: {improvement_test:.1f}%
""")
print("=" * 70)
