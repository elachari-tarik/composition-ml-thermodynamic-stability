"""Step 06 - Sensitivity of the Ehull model to the Ef >= 0 filter (Section 2.2).

Evaluates the effect of keeping vs. removing the 58 raw entries with non-negative
formation energy (Ef >= 0), corresponding to 16 unique formulas after polymorph
deduplication.

Method
  - 118-feature extended Magpie representation and Random Forest regressor
    (n_estimators=200, max_depth=None, min_samples_split=2, min_samples_leaf=1,
    max_features='sqrt', seed=42).
  - Protocol 1 (fixed test set): models trained WITH (n=5207) and WITHOUT (n=5223)
    the Ef >= 0 filter are evaluated on the exact same 521 held-out test formulas.
  - Protocol 2 (full re-split): a fresh 70/20/10 composition split of the unfiltered
    dataset.

Output: results/sensitivity_ef_results.csv
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tco_stability.paths import ensure_project_root
ensure_project_root()

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from tco_stability.features import (
    UNSUPPORTED_ELEMENTS as UNSUPPORTED,
    parse_composition,
    featurize_composition_enhanced,
)

RNG_SEED = 42
np.random.seed(RNG_SEED)


def prepare_dataset(filter_positive_ef=True):
    """Load the raw CSV, clean it with or without the Ef < 0 filter, and featurize."""
    df = pd.read_csv("data/raw/mp_doped_oxides_raw.csv")
    df = df.dropna(subset=["energy_above_hull"])

    n_pos_ef = int((df["formation_energy_per_atom"] >= 0).sum())
    if filter_positive_ef:
        df = df[df["formation_energy_per_atom"] < 0]

    df = df[df["energy_above_hull"] <= 2.0]
    df = df.sort_values("energy_above_hull").drop_duplicates(
        subset="formula_pretty", keep="first"
    )

    mask = df["elements"].apply(
        lambda x: not any(e.strip() in UNSUPPORTED for e in str(x).split(","))
    )
    df = df[mask].reset_index(drop=True)

    feature_records, skipped = [], []
    for idx, row in df.iterrows():
        try:
            comp = parse_composition(row["composition"])
            feature_records.append(featurize_composition_enhanced(comp))
        except (KeyError, ZeroDivisionError, ValueError):
            skipped.append(idx)

    df = df.drop(index=skipped).reset_index(drop=True)
    df_features = pd.DataFrame(feature_records)
    return df, df_features, n_pos_ef


if __name__ == "__main__":
    print("=" * 70)
    print("SENSITIVITY ANALYSIS: IMPACT OF POSITIVE FORMATION ENERGY FILTER (Ef >= 0)")
    print("=" * 70)

    rf_params = dict(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        random_state=RNG_SEED,
        n_jobs=-1,
    )

    # 1. Filtered dataset (standard pipeline: Ef < 0)
    df_filt, X_filt_df, n_pos_raw = prepare_dataset(filter_positive_ef=True)
    y_filt = df_filt["energy_above_hull"].values

    unique_formulas = df_filt["formula_pretty"].unique().copy()
    np.random.seed(RNG_SEED)
    np.random.shuffle(unique_formulas)
    n_total = len(unique_formulas)
    n_test = int(round(0.10 * n_total))
    n_val = int(round(0.20 * n_total))

    test_formulas = set(unique_formulas[:n_test])
    val_formulas = set(unique_formulas[n_test : n_test + n_val])
    train_formulas = set(unique_formulas[n_test + n_val :])

    train_mask_f = df_filt["formula_pretty"].isin(train_formulas)
    val_mask_f = df_filt["formula_pretty"].isin(val_formulas)
    test_mask_f = df_filt["formula_pretty"].isin(test_formulas)

    scaler_f = StandardScaler()
    X_tr_f = scaler_f.fit_transform(X_filt_df[train_mask_f].values)
    X_va_f = scaler_f.transform(X_filt_df[val_mask_f].values)
    X_te_f = scaler_f.transform(X_filt_df[test_mask_f].values)

    X_tv_f = np.vstack([X_tr_f, X_va_f])
    y_tv_f = np.concatenate([y_filt[train_mask_f], y_filt[val_mask_f]])
    y_te_f = y_filt[test_mask_f]

    rf_f = RandomForestRegressor(**rf_params)
    rf_f.fit(X_tv_f, y_tv_f)
    pred_f = rf_f.predict(X_te_f)

    mae_filt = mean_absolute_error(y_te_f, pred_f)
    rmse_filt = np.sqrt(mean_squared_error(y_te_f, pred_f))
    r2_filt = r2_score(y_te_f, pred_f)

    # 2. Unfiltered dataset (Ef >= 0 entries retained)
    df_unfilt, X_unfilt_df, _ = prepare_dataset(filter_positive_ef=False)
    y_unfilt = df_unfilt["energy_above_hull"].values

    # Protocol 1: score on the exact same held-out test formulas (521 formulas)
    test_mask_u_fixed = df_unfilt["formula_pretty"].isin(test_formulas)
    trainval_mask_u_fixed = ~test_mask_u_fixed

    scaler_u_fixed = StandardScaler()
    X_tv_u_fixed = scaler_u_fixed.fit_transform(X_unfilt_df[trainval_mask_u_fixed].values)
    X_te_u_fixed = scaler_u_fixed.transform(X_unfilt_df[test_mask_u_fixed].values)
    y_tv_u_fixed = y_unfilt[trainval_mask_u_fixed]
    y_te_u_fixed = y_unfilt[test_mask_u_fixed]

    rf_u_fixed = RandomForestRegressor(**rf_params)
    rf_u_fixed.fit(X_tv_u_fixed, y_tv_u_fixed)
    pred_u_fixed = rf_u_fixed.predict(X_te_u_fixed)

    mae_unfilt_fixed = mean_absolute_error(y_te_u_fixed, pred_u_fixed)
    rmse_unfilt_fixed = np.sqrt(mean_squared_error(y_te_u_fixed, pred_u_fixed))
    r2_unfilt_fixed = r2_score(y_te_u_fixed, pred_u_fixed)

    # Protocol 2: independent 70/20/10 split of the unfiltered dataset
    u_formulas = df_unfilt["formula_pretty"].unique().copy()
    np.random.seed(RNG_SEED)
    np.random.shuffle(u_formulas)
    n_u_test = int(round(0.10 * len(u_formulas)))
    n_u_val = int(round(0.20 * len(u_formulas)))
    u_test_set = set(u_formulas[:n_u_test])
    u_val_set = set(u_formulas[n_u_test : n_u_test + n_u_val])
    u_train_set = set(u_formulas[n_u_test + n_u_val :])

    tr_m = df_unfilt["formula_pretty"].isin(u_train_set)
    va_m = df_unfilt["formula_pretty"].isin(u_val_set)
    te_m = df_unfilt["formula_pretty"].isin(u_test_set)

    scaler_u_resplit = StandardScaler()
    X_tr_ur = scaler_u_resplit.fit_transform(X_unfilt_df[tr_m].values)
    X_va_ur = scaler_u_resplit.transform(X_unfilt_df[va_m].values)
    X_te_ur = scaler_u_resplit.transform(X_unfilt_df[te_m].values)
    X_tv_ur = np.vstack([X_tr_ur, X_va_ur])
    y_tv_ur = np.concatenate([y_unfilt[tr_m], y_unfilt[va_m]])
    y_te_ur = y_unfilt[te_m]

    rf_u_resplit = RandomForestRegressor(**rf_params)
    rf_u_resplit.fit(X_tv_ur, y_tv_ur)
    pred_u_resplit = rf_u_resplit.predict(X_te_ur)

    mae_unfilt_resplit = mean_absolute_error(y_te_ur, pred_u_resplit)
    rmse_unfilt_resplit = np.sqrt(mean_squared_error(y_te_ur, pred_u_resplit))
    r2_unfilt_resplit = r2_score(y_te_ur, pred_u_resplit)

    diff_fixed = abs(mae_filt - mae_unfilt_fixed)
    diff_resplit = abs(mae_filt - mae_unfilt_resplit)

    print(f"Raw entries with Ef >= 0                : {n_pos_raw}")
    print(f"Featurized dataset size (WITH filter)   : {len(df_filt)}")
    print(f"Featurized dataset size (WITHOUT filter): {len(df_unfilt)} (+{len(df_unfilt) - len(df_filt)} unique formulas)")
    print("-" * 70)
    print(f"1. WITH Ef < 0 filter (Standard, n_test={len(y_te_f)}):")
    print(f"   MAE = {mae_filt:.4f} eV/atom | RMSE = {rmse_filt:.4f} | R2 = {r2_filt:.4f}")
    print(f"2. WITHOUT filter (Same {len(y_te_u_fixed)} test formulas, n_test={len(y_te_u_fixed)}):")
    print(f"   MAE = {mae_unfilt_fixed:.4f} eV/atom | RMSE = {rmse_unfilt_fixed:.4f} | R2 = {r2_unfilt_fixed:.4f}")
    print(f"   -> Absolute MAE difference (same test set): {diff_fixed:.5f} eV/atom")
    print(f"3. WITHOUT filter (Re-shuffled test split, n_test={len(y_te_ur)}):")
    print(f"   MAE = {mae_unfilt_resplit:.4f} eV/atom | RMSE = {rmse_unfilt_resplit:.4f} | R2 = {r2_unfilt_resplit:.4f}")
    print(f"   -> Absolute MAE difference (re-split)     : {diff_resplit:.5f} eV/atom")
    print("=" * 70)

    os.makedirs("results", exist_ok=True)
    pd.DataFrame([
        {
            "protocol": "with_ef_filter_standard",
            "n_total": len(df_filt),
            "n_trainval": len(y_tv_f),
            "n_test": len(y_te_f),
            "test_mae": mae_filt,
            "test_rmse": rmse_filt,
            "test_r2": r2_filt,
            "abs_mae_diff_vs_standard": 0.0,
        },
        {
            "protocol": "without_ef_filter_same_test_set",
            "n_total": len(df_unfilt),
            "n_trainval": len(y_tv_u_fixed),
            "n_test": len(y_te_u_fixed),
            "test_mae": mae_unfilt_fixed,
            "test_rmse": rmse_unfilt_fixed,
            "test_r2": r2_unfilt_fixed,
            "abs_mae_diff_vs_standard": diff_fixed,
        },
        {
            "protocol": "without_ef_filter_resplit",
            "n_total": len(df_unfilt),
            "n_trainval": len(y_tv_ur),
            "n_test": len(y_te_ur),
            "test_mae": mae_unfilt_resplit,
            "test_rmse": rmse_unfilt_resplit,
            "test_r2": r2_unfilt_resplit,
            "abs_mae_diff_vs_standard": diff_resplit,
        },
    ]).to_csv("results/sensitivity_ef_results.csv", index=False)
    print("Saved: results/sensitivity_ef_results.csv")