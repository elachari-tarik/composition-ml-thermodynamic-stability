"""Step 09 - Phase 2: featurization and baseline modeling (105 features).

Steps
  2.1  Composition-based feature vectors (CBFV): 15 elemental properties x 7 statistics
  2.2  Train / validation / test split by unique composition (70 / 20 / 10)
  2.3  Feature scaling (StandardScaler, fit on the training set only)
  2.4  Baseline (DummyRegressor) and Random Forest, trained and evaluated

Follows Wang et al., Chem. Mater. 2020, 32, 4954-4965.

Outputs
  data/splits/{train,val,test}.csv
  data/processed/phase2_model_results.csv
  figures/phase2/03_parity_plot_RF.png, figures/phase2/04_residual_plot_RF.png
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
from time import time

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

from tco_stability.features import (
    PROP_NAMES,
    STAT_NAMES,
    parse_composition,
    featurize_composition,
)

# Reproducibility
RNG_SEED = 42
np.random.seed(RNG_SEED)

# ── Load cleaned data ────────────────────────────────────────────────────────
print("\n[STEP 2.1] Loading data and generating CBFVs...")
df = pd.read_csv("data/processed/mp_doped_oxides_cleaned.csv")
print(f"  Loaded: {len(df)} entries")

# Drop entries containing elements that are missing from our property table
unsupported = {'Pm', 'Pu', 'Tc', 'Np', 'Pa'}
mask = df['elements'].apply(
    lambda x: not any(e.strip() in unsupported for e in x.split(','))
)
df = df[mask].reset_index(drop=True)
print(f"  After removing unsupported elements: {len(df)} entries")

# Generate the features
print("  Featurizing compositions...")
t0 = time()
feature_records = []
skipped = []
for idx, row in df.iterrows():
    comp = parse_composition(row['composition'])
    try:
        feats = featurize_composition(comp)
        feature_records.append(feats)
    except (KeyError, ZeroDivisionError) as e:
        skipped.append(idx)

df = df.drop(index=skipped).reset_index(drop=True)
df_features = pd.DataFrame(feature_records)
dt = time() - t0
print(f"  Generated {df_features.shape[1]} features for {len(df_features)} compositions ({dt:.1f}s)")
if skipped:
    print(f"  Skipped {len(skipped)} entries due to parsing errors")

feature_names = list(df_features.columns)
print(f"  Example features: {feature_names[:5]} ... {feature_names[-3:]}")

# =============================================================================
# STEP 2.2: TRAIN / VALIDATION / TEST SPLIT
# =============================================================================
# Critical rule: split by UNIQUE COMPOSITION to prevent data leakage. Phase 1
# already left one entry per formula, so a plain random split on formulas is safe.
# Proportions: ~70% train, ~20% validation, ~10% test.
# =============================================================================

print("\n[STEP 2.2] Splitting data by unique composition...")

y = df['energy_above_hull'].values

# Unique formulas, shuffled with the global seed set above
unique_formulas = df['formula_pretty'].unique().copy()
np.random.shuffle(unique_formulas)

n_total = len(unique_formulas)
n_test = int(round(0.10 * n_total))
n_val = int(round(0.20 * n_total))
n_train = n_total - n_val - n_test

test_formulas = set(unique_formulas[:n_test])
val_formulas = set(unique_formulas[n_test:n_test + n_val])
train_formulas = set(unique_formulas[n_test + n_val:])

train_mask = df['formula_pretty'].isin(train_formulas)
val_mask = df['formula_pretty'].isin(val_formulas)
test_mask = df['formula_pretty'].isin(test_formulas)

X_train = df_features[train_mask].values
X_val = df_features[val_mask].values
X_test = df_features[test_mask].values
y_train = y[train_mask]
y_val = y[val_mask]
y_test = y[test_mask]

# Keep the metadata of each split for later analyses
df_train = df[train_mask].reset_index(drop=True)
df_val = df[val_mask].reset_index(drop=True)
df_test = df[test_mask].reset_index(drop=True)

print(f"  Train: {len(X_train)} samples ({len(X_train)/n_total*100:.1f}%)")
print(f"  Val:   {len(X_val)} samples ({len(X_val)/n_total*100:.1f}%)")
print(f"  Test:  {len(X_test)} samples ({len(X_test)/n_total*100:.1f}%)")

# No formula may appear in two splits
leak_1 = train_formulas & val_formulas
leak_2 = train_formulas & test_formulas
leak_3 = val_formulas & test_formulas
print(f"\n  Data leakage check:")
print(f"    Train ∩ Val:  {len(leak_1)} common formulas {'✓' if len(leak_1)==0 else '✗ LEAKAGE!'}")
print(f"    Train ∩ Test: {len(leak_2)} common formulas {'✓' if len(leak_2)==0 else '✗ LEAKAGE!'}")
print(f"    Val ∩ Test:   {len(leak_3)} common formulas {'✓' if len(leak_3)==0 else '✗ LEAKAGE!'}")

# Save the splits so every later analysis uses exactly the same partition
os.makedirs("data/splits", exist_ok=True)
df_train.to_csv("data/splits/train.csv", index=False)
df_val.to_csv("data/splits/val.csv", index=False)
df_test.to_csv("data/splits/test.csv", index=False)
print("  Splits saved to data/splits/")

# =============================================================================
# STEP 2.3: SCALING
# =============================================================================
# Inputs are scaled to zero mean and unit variance for stable training. The scaler
# is fit on the TRAINING data only, then applied to validation and test.
# =============================================================================

print("\n[STEP 2.3] Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)
print(f"  Scaler fit on training set: mean≈0, std≈1")
print(f"  Train feature range: [{X_train_scaled.min():.2f}, {X_train_scaled.max():.2f}]")

# =============================================================================
# STEP 2.4: BASELINE + RANDOM FOREST
# =============================================================================

print("\n[STEP 2.4] Training models...")
print("-" * 55)

# ── Baseline: DummyRegressor ─────────────────────────────────────────────────
# A dummy model is a cheap sanity check: if the real models do not clearly beat it,
# something went wrong.

dummy = DummyRegressor(strategy="mean")
dummy.fit(X_train_scaled, y_train)
y_pred_dummy_val = dummy.predict(X_val_scaled)

mae_dummy = mean_absolute_error(y_val, y_pred_dummy_val)
r2_dummy = r2_score(y_val, y_pred_dummy_val)
rmse_dummy = np.sqrt(mean_squared_error(y_val, y_pred_dummy_val))

print(f"\n  DummyRegressor (baseline - predicts mean):")
print(f"    MAE  = {mae_dummy:.4f} eV/atom")
print(f"    RMSE = {rmse_dummy:.4f} eV/atom")
print(f"    R²   = {r2_dummy:.4f}")

# ── Random Forest ────────────────────────────────────────────────────────────
print(f"\n  Training RandomForestRegressor...")
t0 = time()
rf = RandomForestRegressor(
    n_estimators=200,
    max_depth=None,        # let the trees grow fully
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',   # standard choice for RF
    random_state=RNG_SEED,
    n_jobs=-1,             # all CPU cores
)
rf.fit(X_train_scaled, y_train)
dt = time() - t0
print(f"    Training time: {dt:.1f}s")

# Training-set scores
y_pred_train = rf.predict(X_train_scaled)
mae_train = mean_absolute_error(y_train, y_pred_train)
r2_train = r2_score(y_train, y_pred_train)
rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))

# Validation-set scores
y_pred_val = rf.predict(X_val_scaled)
mae_val = mean_absolute_error(y_val, y_pred_val)
r2_val = r2_score(y_val, y_pred_val)
rmse_val = np.sqrt(mean_squared_error(y_val, y_pred_val))

print(f"\n  RandomForestRegressor results:")
print(f"    {'':12s} {'MAE':>10s} {'RMSE':>10s} {'R²':>10s}")
print(f"    {'Training':12s} {mae_train:10.4f} {rmse_train:10.4f} {r2_train:10.4f}")
print(f"    {'Validation':12s} {mae_val:10.4f} {rmse_val:10.4f} {r2_val:10.4f}")
print(f"    {'Dummy (ref)':12s} {mae_dummy:10.4f} {rmse_dummy:10.4f} {r2_dummy:10.4f}")

improvement = (mae_dummy - mae_val) / mae_dummy * 100
print(f"\n    → RF reduces MAE by {improvement:.1f}% vs baseline")

# =============================================================================
# Parity plot (predicted vs. actual)
# =============================================================================

print("\n[VISUALIZATION] Generating parity plot...")
os.makedirs("figures/phase2", exist_ok=True)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# ── Panel 1: training set ────────────────────────────────────────────────────
ax = axes[0]
ax.scatter(y_train, y_pred_train, alpha=0.3, s=12, c='#2196F3', edgecolors='none')
lims = [0, max(y_train.max(), y_pred_train.max()) * 1.05]
ax.plot(lims, lims, 'k--', linewidth=1, label='Ideal (y=x)')

# Linear fit
coeffs = np.polyfit(y_train, y_pred_train, 1)
fit_line = np.poly1d(coeffs)
x_fit = np.linspace(lims[0], lims[1], 100)
ax.plot(x_fit, fit_line(x_fit), 'r-', alpha=0.7, linewidth=1, label='Linear fit')

ax.set_xlim(lims)
ax.set_ylim(lims)
ax.set_xlabel("Actual $E_{hull}$ (eV/atom)", fontsize=12)
ax.set_ylabel("Predicted $E_{hull}$ (eV/atom)", fontsize=12)
ax.set_title(f"Training Set\nMAE={mae_train:.4f}, R²={r2_train:.4f}", fontsize=11)
ax.legend(fontsize=9)
ax.set_aspect('equal')

# ── Panel 2: validation set ──────────────────────────────────────────────────
ax = axes[1]
ax.scatter(y_val, y_pred_val, alpha=0.4, s=15, c='#FF5722', edgecolors='none')
lims = [0, max(y_val.max(), y_pred_val.max()) * 1.05]
ax.plot(lims, lims, 'k--', linewidth=1, label='Ideal (y=x)')

coeffs = np.polyfit(y_val, y_pred_val, 1)
fit_line = np.poly1d(coeffs)
x_fit = np.linspace(lims[0], lims[1], 100)
ax.plot(x_fit, fit_line(x_fit), 'r-', alpha=0.7, linewidth=1, label='Linear fit')

ax.set_xlim(lims)
ax.set_ylim(lims)
ax.set_xlabel("Actual $E_{hull}$ (eV/atom)", fontsize=12)
ax.set_ylabel("Predicted $E_{hull}$ (eV/atom)", fontsize=12)
ax.set_title(f"Validation Set\nMAE={mae_val:.4f}, R²={r2_val:.4f}", fontsize=11)
ax.legend(fontsize=9)
ax.set_aspect('equal')

fig.suptitle("Random Forest: Predicted vs Actual Energy Above Hull",
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig("figures/phase2/03_parity_plot_RF.png", dpi=150, bbox_inches="tight")
print("  Saved: figures/phase2/03_parity_plot_RF.png")

# ── Residual plot ────────────────────────────────────────────────────────────
fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))

residuals_val = y_pred_val - y_val
ax = axes2[0]
ax.scatter(y_val, residuals_val, alpha=0.4, s=12, c='#FF5722', edgecolors='none')
ax.axhline(y=0, color='k', linestyle='--', linewidth=1)
ax.set_xlabel("Actual $E_{hull}$ (eV/atom)", fontsize=12)
ax.set_ylabel("Residual (Pred - Actual) (eV/atom)", fontsize=12)
ax.set_title("Residuals vs Actual (Validation)", fontsize=11)

ax = axes2[1]
ax.hist(residuals_val, bins=50, color='#FF5722', edgecolor='white', alpha=0.8)
ax.axvline(x=0, color='k', linestyle='--', linewidth=1)
ax.set_xlabel("Residual (eV/atom)", fontsize=12)
ax.set_ylabel("Count", fontsize=12)
ax.set_title(f"Residual Distribution\nMean={residuals_val.mean():.4f}, Std={residuals_val.std():.4f}",
             fontsize=11)

plt.tight_layout()
plt.savefig("figures/phase2/04_residual_plot_RF.png", dpi=150, bbox_inches="tight")
print("  Saved: figures/phase2/04_residual_plot_RF.png")

# =============================================================================
# Save the results summary
# =============================================================================
results = {
    "model": ["DummyRegressor", "RandomForest"],
    "MAE_val": [mae_dummy, mae_val],
    "RMSE_val": [rmse_dummy, rmse_val],
    "R2_val": [r2_dummy, r2_val],
    "MAE_train": [None, mae_train],
    "R2_train": [None, r2_train],
}
df_results = pd.DataFrame(results)
df_results.to_csv("data/processed/phase2_model_results.csv", index=False)

print("\n" + "=" * 70)
print("PHASE 2 COMPLETE")
print("=" * 70)
print(f"  Features: {df_features.shape[1]} CBFVs from {len(PROP_NAMES)} elemental properties × {len(STAT_NAMES)} statistics")
print(f"  Split: {len(X_train)} train / {len(X_val)} val / {len(X_test)} test")
print(f"  Baseline MAE: {mae_dummy:.4f} eV/atom")
print(f"  RF Val MAE:   {mae_val:.4f} eV/atom (R²={r2_val:.4f})")
print(f"  Improvement:  {improvement:.1f}% over baseline")
print("=" * 70)
