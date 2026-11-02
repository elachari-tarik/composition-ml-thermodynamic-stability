"""Step 17 - Save the trained models (reproducibility package).

The manuscript states that the repository contains "the parameters of the trained
models". This step retrains the three final models exactly as in step 11 and stores
them, together with the scaler and the feature names, in models/.

Pipeline (identical to step 11)
  118 features -> split by unique formula (seed 42, 70/20/10) -> StandardScaler fit on
  the training set -> hyperparameters of the random forest taken from the grid search
  (data/processed/grid_search_results_enhanced.csv, best validation MAE) -> every model
  retrained on train+val.

Run it after step 11. Output: models/*.joblib and models/model_summary.json
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tco_stability.paths import ensure_project_root
ensure_project_root()

import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

from tco_stability.features import parse_composition, featurize_composition_enhanced

try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    print("  ⚠ XGBoost not installed: the XGBoost model will not be saved.")
    HAS_XGBOOST = False

RNG_SEED = 42
np.random.seed(RNG_SEED)

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

# ── Load, featurize, split, scale ────────────────────────────────────────────
print("[1/3] Rebuilding the 118-feature pipeline...")
df = pd.read_csv("data/processed/mp_doped_oxides_cleaned.csv")
unsupported = {'Pm', 'Pu', 'Tc', 'Np', 'Pa'}
mask = df['elements'].apply(
    lambda x: not any(e.strip() in unsupported for e in x.split(','))
)
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

scaler = StandardScaler()
X_train = scaler.fit_transform(df_features[train_mask].values)
X_val = scaler.transform(df_features[val_mask].values)
X_test = scaler.transform(df_features[test_mask].values)
y_train, y_val, y_test = y[train_mask], y[val_mask], y[test_mask]
print(f"  {len(X_train)} train / {len(X_val)} val / {len(X_test)} test, {len(feature_names)} features")

X_trainval = np.vstack([X_train, X_val])
y_trainval = np.concatenate([y_train, y_val])

# ── Best random-forest hyperparameters from the grid search of step 11 ───────
print("\n[2/3] Reading the best RF hyperparameters from the grid search...")
df_grid = pd.read_csv("data/processed/grid_search_results_enhanced.csv")
best = df_grid.loc[df_grid['mae_val'].idxmin()]
best_params = {
    'n_estimators': int(best['n_estimators']),
    'max_depth': int(best['max_depth']) if pd.notna(best['max_depth']) else None,
    'min_samples_split': int(best['min_samples_split']),
    'min_samples_leaf': int(best['min_samples_leaf']),
    'max_features': 'sqrt',
    'random_state': RNG_SEED,
    'n_jobs': -1,
}
print(f"  {best_params}")

# ── Train the final models on train+val and save them ────────────────────────
print("\n[3/3] Training final models on train+val and saving...")
models = {
    'random_forest': RandomForestRegressor(**best_params),
    'gradient_boosting': GradientBoostingRegressor(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        min_samples_split=5, min_samples_leaf=3, subsample=0.8,
        random_state=RNG_SEED,
    ),
}
if HAS_XGBOOST:
    models['xgboost'] = XGBRegressor(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
        gamma=0.1, reg_alpha=0.1, reg_lambda=1.0,
        random_state=RNG_SEED, n_jobs=-1, verbosity=0,
    )

summary = {'n_features': len(feature_names),
           'n_trainval': int(len(X_trainval)),
           'n_test': int(len(X_test)),
           'rf_best_params': best_params,
           'test_metrics': {}}

for name, model in models.items():
    model.fit(X_trainval, y_trainval)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    summary['test_metrics'][name] = {'mae': float(mae), 'r2': float(r2)}
    joblib.dump(model, f"{MODELS_DIR}/{name}.joblib")
    print(f"  {name:<18s} test MAE = {mae:.4f}, R² = {r2:.4f}  -> {MODELS_DIR}/{name}.joblib")

joblib.dump(scaler, f"{MODELS_DIR}/scaler.joblib")
joblib.dump(feature_names, f"{MODELS_DIR}/feature_names.joblib")
with open(f"{MODELS_DIR}/model_summary.json", "w") as fh:
    json.dump(summary, fh, indent=2)
print(f"\n  Saved scaler, feature names and summary to {MODELS_DIR}/")
