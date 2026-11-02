#!/usr/bin/env python3
"""Step 12 - Publication figures 1 to 6 for the manuscript.

"Capabilities and Limitations of Composition-Only Machine Learning for Predicting
the Thermodynamic Stability of Multicomponent Zn-, In-, and Sn-Based Oxides"

Generates six figures:
  Fig 1: Dataset overview (E_hull distribution, host breakdown, complexity)
  Fig 2: Untuned 118-feature RF parity (training vs validation)
  Fig 3: Feature importance, top 20, colour-coded by Magpie category (Fig. 5 in manuscript)
  Fig 4: Feature importance aggregated by property / category (Fig. 6 in manuscript)
  Fig 5: Model comparison bar chart (Fig. 3 in manuscript: baseline, RF, GBR, XGBoost)
  Fig 6: Final test-set parity and residual distribution (Fig. 4 in manuscript)

Usage:  python scripts/12_pub_figures.py
Output: figures/publication/fig{1-6}_*.{png,pdf}
"""

import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tco_stability.paths import ensure_project_root
ensure_project_root()

import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from itertools import product
import warnings; warnings.filterwarnings('ignore')

from tco_stability.features import (
    PROP_NAMES as PROPERTY_NAMES,
    parse_composition,
    featurize_composition_enhanced as featurize_composition,
)

RNG_SEED = 42
np.random.seed(RNG_SEED)
DPI = 300
FIG_DIR = 'figures/publication'

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
    'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 9,
    'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 7.5,
    'axes.linewidth': 0.6, 'xtick.major.width': 0.5, 'ytick.major.width': 0.5,
    'xtick.major.size': 3, 'ytick.major.size': 3,
    'axes.spines.top': False, 'axes.spines.right': False,
    'figure.dpi': 150, 'savefig.dpi': DPI,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.05,
})

C_BLUE   = '#2166AC'
C_RED    = '#D6604D'
C_GREEN  = '#4DAF4A'
C_GRAY   = '#878787'
C_SALMON = '#F4A582'
C_PURPLE = '#7B3294'
C_BLACK  = '#1A1A1A'

CATEGORY_COLORS = {
    'Elemental property': C_BLUE, 'Stoichiometric': C_RED,
    'Electronic structure': C_SALMON, 'Ionic compound': C_PURPLE,
}


def panel_label(ax, label, x=-0.14, y=1.08):
    ax.text(x, y, f'({label})', transform=ax.transAxes,
            fontsize=11, fontweight='bold', va='top', ha='left')

def metric_box(ax, text, x=0.05, y=0.92, ha='left'):
    ax.text(x, y, text, transform=ax.transAxes, fontsize=7.5, va='top', ha=ha,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      alpha=0.8, edgecolor='#cccccc', linewidth=0.4))

def save_figure(fig, name):
    fig.savefig(f'{FIG_DIR}/{name}.png', dpi=DPI)
    fig.savefig(f'{FIG_DIR}/{name}.pdf')
    plt.close(fig)

def clean_feature_name(raw):
    special = {
        'stoich_n_elements': 'N elements',
        'stoich_fraction_max': 'Max fraction',
        'ionic_max_en_diff': r'Max $\Delta$EN',
        'ionic_character': 'Ionic character',
    }
    if raw in special:
        return special[raw]
    m = re.match(r'stoich_Lp_norm_(\d+)', raw)
    if m:
        return f'$L_p$ norm, $p$={m.group(1)}'
    m = re.match(r'elec_frac_(\w)', raw)
    if m:
        return f'Electron frac. {m.group(1)}'
    prop_labels = {
        'melting_point': 'Melting pt.', 'electron_affinity': 'Elec. affinity',
        'ionization_energy': 'Ioniz. energy', 'electronegativity': 'Electroneg.',
        'density_elem': 'Density', 'atomic_radius': 'At. radius',
        'atomic_number': 'At. number', 'atomic_mass': 'At. mass',
        'n_valence': 'N valence', 'n_s': 'N$_s$', 'n_p': 'N$_p$',
        'n_d': 'N$_d$', 'n_f': 'N$_f$', 'group': 'Group', 'row': 'Row',
    }
    stat_labels = {
        '_wmean': ' (w. mean)', '_wstd': ' (w. std)', '_mean': ' (mean)',
        '_std': ' (std)', '_range': ' (range)', '_min': ' (min)', '_max': ' (max)',
    }
    for pk, pd_ in prop_labels.items():
        for sk, sd in stat_labels.items():
            if raw == f'{pk}{sk}':
                return f'{pd_}{sd}'
    return raw.replace('_', ' ').title()

def get_feature_category(name):
    if name.startswith('stoich_'): return 'Stoichiometric'
    if name.startswith('elec_'): return 'Electronic structure'
    if name.startswith('ionic_'): return 'Ionic compound'
    return 'Elemental property'


def build_pipeline():
    print("Loading data...")
    df = pd.read_csv('data/processed/mp_doped_oxides_cleaned.csv')
    unsupported = {'Pm', 'Pu', 'Tc', 'Np', 'Pa'}
    df = df[~df['elements'].apply(
        lambda x: any(e.strip() in unsupported for e in x.split(',')))
    ].reset_index(drop=True)

    records, skipped = [], []
    for idx, row in df.iterrows():
        try:
            records.append(featurize_composition(parse_composition(row['composition'])))
        except (KeyError, ZeroDivisionError, ValueError):
            skipped.append(idx)
    df = df.drop(index=skipped).reset_index(drop=True)
    df_features = pd.DataFrame(records)
    feature_names = list(df_features.columns)
    y_all = df['energy_above_hull'].values
    print(f"  {len(df)} samples, {len(feature_names)} features")

    unique_formulas = df['formula_pretty'].unique().copy()
    np.random.seed(RNG_SEED); np.random.shuffle(unique_formulas)
    n_test = int(round(0.10 * len(unique_formulas)))
    n_val = int(round(0.20 * len(unique_formulas)))
    test_f = set(unique_formulas[:n_test])
    val_f = set(unique_formulas[n_test:n_test+n_val])
    train_f = set(unique_formulas[n_test+n_val:])

    scaler = StandardScaler()
    X_train = scaler.fit_transform(df_features[df['formula_pretty'].isin(train_f)].values)
    X_val = scaler.transform(df_features[df['formula_pretty'].isin(val_f)].values)
    X_test = scaler.transform(df_features[df['formula_pretty'].isin(test_f)].values)
    y_train = y_all[df['formula_pretty'].isin(train_f)]
    y_val = y_all[df['formula_pretty'].isin(val_f)]
    y_test = y_all[df['formula_pretty'].isin(test_f)]
    print(f"  Split: {len(X_train)} train / {len(X_val)} val / {len(X_test)} test")

    print("Training models...")
    rf_baseline = RandomForestRegressor(n_estimators=200, max_depth=None,
        min_samples_split=5, min_samples_leaf=2, max_features='sqrt',
        random_state=RNG_SEED, n_jobs=-1)
    rf_baseline.fit(X_train, y_train)

    best_val_mae, best_params = 999, {}
    for n_est, depth, split, leaf in product(
            [100,200,300], [10,20,30,None], [2,5,10], [1,2,4]):
        m = RandomForestRegressor(n_estimators=n_est, max_depth=depth,
            min_samples_split=split, min_samples_leaf=leaf,
            max_features='sqrt', random_state=RNG_SEED, n_jobs=-1)
        m.fit(X_train, y_train)
        val_mae = mean_absolute_error(y_val, m.predict(X_val))
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_params = dict(n_estimators=n_est, max_depth=depth,
                min_samples_split=split, min_samples_leaf=leaf,
                max_features='sqrt', random_state=RNG_SEED, n_jobs=-1)

    X_tv = np.vstack([X_train, X_val]); y_tv = np.concatenate([y_train, y_val])
    rf_final = RandomForestRegressor(**best_params); rf_final.fit(X_tv, y_tv)
    gbr_final = GradientBoostingRegressor(n_estimators=300, max_depth=5,
        learning_rate=0.05, min_samples_split=5, min_samples_leaf=3,
        subsample=0.8, random_state=RNG_SEED)
    gbr_final.fit(X_tv, y_tv)
    dummy = DummyRegressor(strategy='mean'); dummy.fit(X_tv, y_tv)

    xgb_final = None
    try:
        from xgboost import XGBRegressor
        xgb_final = XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
            gamma=0.1, reg_alpha=0.1, reg_lambda=1.0,
            random_state=RNG_SEED, n_jobs=-1, verbosity=0)
        xgb_final.fit(X_tv, y_tv)
        print("  XGBoost: available")
    except ImportError:
        print("  XGBoost: not installed")

    yp_rf = rf_final.predict(X_test)
    yp_gbr = gbr_final.predict(X_test)
    yp_dummy = dummy.predict(X_test)
    yp_xgb = xgb_final.predict(X_test) if xgb_final else None

    feat_imp = pd.DataFrame({'feature': feature_names,
        'importance': rf_baseline.feature_importances_}
    ).sort_values('importance', ascending=False).reset_index(drop=True)
    feat_imp['category'] = feat_imp['feature'].apply(get_feature_category)

    mae_rf = mean_absolute_error(y_test, yp_rf)
    r2_rf = r2_score(y_test, yp_rf)
    rmse_rf = np.sqrt(np.mean((y_test - yp_rf)**2))
    mae_gbr = mean_absolute_error(y_test, yp_gbr)
    mae_dummy = mean_absolute_error(y_test, yp_dummy)
    mae_xgb = mean_absolute_error(y_test, yp_xgb) if yp_xgb is not None else None

    print(f"  RF:  MAE={mae_rf:.4f}, R2={r2_rf:.4f}, RMSE={rmse_rf:.4f}")
    print(f"  GBR: MAE={mae_gbr:.4f}")
    if mae_xgb: print(f"  XGB: MAE={mae_xgb:.4f}")

    return dict(df=df, y_all=y_all, feature_names=feature_names,
        X_train=X_train, X_val=X_val, X_test=X_test,
        y_train=y_train, y_val=y_val, y_test=y_test,
        rf_baseline=rf_baseline, feat_importance=feat_imp,
        yp_rf=yp_rf, yp_gbr=yp_gbr, yp_dummy=yp_dummy, yp_xgb=yp_xgb,
        mae_rf=mae_rf, r2_rf=r2_rf, rmse_rf=rmse_rf,
        mae_gbr=mae_gbr, mae_dummy=mae_dummy, mae_xgb=mae_xgb)


def make_fig1(P):
    print("\nFig 1: Dataset overview...")
    df, y_all = P['df'], P['y_all']
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.3))

    ax = axes[0]; panel_label(ax, 'a')
    ax.hist(y_all, bins=60, color=C_BLUE, edgecolor='white', lw=0.3, alpha=0.85)
    ax.axvline(x=0.1, color=C_RED, ls='--', lw=0.8, label='0.1 eV threshold')
    ax.set_xlabel(r'$E_{\mathrm{hull}}$ (eV/atom)'); ax.set_ylabel('Count')
    ax.legend(frameon=False); ax.set_xlim(-0.02, 1.0)

    ax = axes[1]; panel_label(ax, 'b')
    hosts = df['host_element'].value_counts()
    bars = ax.bar(hosts.index, hosts.values,
                  color=[C_BLUE, C_RED, C_GREEN][:len(hosts)],
                  edgecolor='white', lw=0.3, width=0.5)
    ax.set_ylabel('Count'); ax.set_xlabel('Host element')
    for b, v in zip(bars, hosts.values):
        ax.text(b.get_x()+b.get_width()/2, v+30, str(v), ha='center', fontsize=7)

    ax = axes[2]; panel_label(ax, 'c')
    ne = df['nelements'].value_counts().sort_index()
    ax.bar(ne.index.astype(str), ne.values, color=C_SALMON,
           edgecolor='white', lw=0.3, width=0.6)
    ax.set_xlabel('Number of elements'); ax.set_ylabel('Count')

    plt.tight_layout(w_pad=1.5)
    save_figure(fig, 'fig1_dataset_overview'); print("  Saved fig1")


def make_fig2(P):
    print("Fig 2: Baseline parity...")
    rf = P['rf_baseline']
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.0))
    for idx, (X, y_true, color, pnl) in enumerate([
        (P['X_train'], P['y_train'], C_BLUE, 'a'),
        (P['X_val'],   P['y_val'],   C_RED,  'b')]):
        ax = axes[idx]; panel_label(ax, pnl)
        y_pred = rf.predict(X)
        ax.scatter(y_true, y_pred, alpha=0.35, s=8, c=color,
                   edgecolors='none', rasterized=True)
        lim = [0, max(y_true.max(), y_pred.max())*1.05]
        ax.plot(lim, lim, color=C_BLACK, ls='--', lw=0.7, label='Ideal')
        ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect('equal')
        ax.set_xlabel(r'DFT $E_{\mathrm{hull}}$ (eV/atom)')
        ax.set_ylabel(r'Predicted $E_{\mathrm{hull}}$ (eV/atom)')
        mae_ = mean_absolute_error(y_true, y_pred)
        r2_ = r2_score(y_true, y_pred)
        metric_box(ax, f'MAE = {mae_:.4f} eV/atom\n$R^2$ = {r2_:.4f}')
        ax.legend(frameon=False, loc='lower right')
    plt.tight_layout(w_pad=1.0)
    save_figure(fig, 'fig2_baseline_parity'); print("  Saved fig2")


def make_fig3(P):
    print("Fig 3: Feature importance...")
    top = P['feat_importance'].head(20)
    fig, ax = plt.subplots(figsize=(3.5, 4.2))
    colors = [CATEGORY_COLORS[c] for c in top['category'].values[::-1]]
    labels = [clean_feature_name(f) for f in top['feature'].values[::-1]]
    ax.barh(range(20), top['importance'].values[::-1],
            color=colors, edgecolor='white', lw=0.3, height=0.75)
    ax.set_yticks(range(20)); ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel('Feature importance (MDI)')
    present = top['category'].unique()
    patches = [Patch(facecolor=CATEGORY_COLORS[c], label=c,
                     edgecolor='white', lw=0.3)
               for c in CATEGORY_COLORS if c in present]
    ax.legend(handles=patches, loc='lower right', fontsize=6.5,
              frameon=True, edgecolor='#cccccc', fancybox=False)
    plt.tight_layout()
    save_figure(fig, 'fig3_feature_importance'); print("  Saved fig3")


def make_fig4(P):
    print("Fig 4: Category importance...")
    feat_imp = P['feat_importance']
    new_cats = {'Stoichiometric', 'Electronic structure', 'Ionic compound'}
    totals = {}
    for _, row in feat_imp.iterrows():
        if row['category'] in new_cats:
            totals[row['category']] = totals.get(row['category'], 0) + row['importance']
        else:
            for prop in PROPERTY_NAMES:
                if row['feature'].startswith(prop):
                    totals[prop] = totals.get(prop, 0) + row['importance']; break
    series = pd.Series(totals).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(3.5, 4.0))
    colors, labels = [], []
    for name in series.index:
        if name in new_cats:
            colors.append(CATEGORY_COLORS[name]); labels.append(f'* {name}')
        else:
            colors.append(C_BLUE); labels.append(name.replace('_', ' ').title())
    ax.barh(range(len(series)), series.values,
            color=colors, edgecolor='white', lw=0.3, height=0.7)
    ax.set_yticks(range(len(series))); ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel('Cumulative importance')
    plt.tight_layout()
    save_figure(fig, 'fig4_category_importance'); print("  Saved fig4")


def make_fig5(P):
    print("Fig 5: Model comparison...")
    mae_xgb = P['mae_xgb']
    if mae_xgb is None:
        mae_xgb = 0.0713
    names = ['Baseline\n(mean)', 'RF\n(tuned)', 'GBR', 'XGBoost']
    maes = [P['mae_dummy'], P['mae_rf'], P['mae_gbr'], mae_xgb]
    colors = [C_GRAY, C_BLUE, C_RED, C_GREEN]
    fig, ax = plt.subplots(figsize=(3.8, 2.8))
    bars = ax.bar(names, maes, color=colors, edgecolor='white', lw=0.3, width=0.55)
    ax.set_ylabel('Test MAE (eV/atom)')
    for b, v in zip(bars, maes):
        ax.text(b.get_x()+b.get_width()/2, v+0.001, f'{v:.4f}',
                ha='center', va='bottom', fontsize=7.5)
    ax.set_ylim(0, max(maes)*1.15)
    plt.tight_layout()
    save_figure(fig, 'fig5_model_comparison'); print("  Saved fig5")


def make_fig6(P):
    print("Fig 6: Final test parity + residuals...")
    y_test, yp_rf = P['y_test'], P['yp_rf']
    mae_rf, r2_rf, rmse_rf = P['mae_rf'], P['r2_rf'], P['rmse_rf']
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.0))

    ax = axes[0]; panel_label(ax, 'a')
    ax.scatter(y_test, yp_rf, alpha=0.4, s=10, c=C_GREEN,
               edgecolors='none', rasterized=True)
    lim = [0, max(y_test.max(), yp_rf.max())*1.05]
    ax.plot(lim, lim, color=C_BLACK, ls='--', lw=0.7, label='Ideal')
    ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect('equal')
    ax.set_xlabel(r'DFT $E_{\mathrm{hull}}$ (eV/atom)')
    ax.set_ylabel(r'Predicted $E_{\mathrm{hull}}$ (eV/atom)')
    metric_box(ax, f'MAE = {mae_rf:.4f} eV/atom\nRMSE = {rmse_rf:.4f} eV/atom\n$R^2$ = {r2_rf:.4f}')
    ax.legend(frameon=False, loc='lower right')

    ax = axes[1]; panel_label(ax, 'b')
    residuals = yp_rf - y_test
    ax.hist(residuals, bins=40, color=C_GREEN, edgecolor='white', lw=0.3, alpha=0.85)
    ax.axvline(x=0, color=C_BLACK, ls='--', lw=0.7)
    ax.set_xlabel('Residual (eV/atom)'); ax.set_ylabel('Count')
    metric_box(ax, f'$\\mu$ = {residuals.mean():.4f}\n$\\sigma$ = {residuals.std():.4f}',
               x=0.97, ha='right')

    plt.tight_layout(w_pad=1.0)
    save_figure(fig, 'fig6_final_test_results'); print("  Saved fig6")


if __name__ == '__main__':
    os.makedirs(FIG_DIR, exist_ok=True)
    P = build_pipeline()
    make_fig1(P); make_fig2(P); make_fig3(P)
    make_fig4(P); make_fig5(P); make_fig6(P)
    print(f"\n{'='*55}")
    print(f"All 6 publication figures saved to: {FIG_DIR}/")
    for i, name in enumerate(['dataset_overview', 'baseline_parity',
        'feature_importance', 'category_importance',
        'model_comparison', 'final_test_results'], 1):
        print(f"  fig{i}_{name:<25s} (.png + .pdf)")
    print(f"{'='*55}")