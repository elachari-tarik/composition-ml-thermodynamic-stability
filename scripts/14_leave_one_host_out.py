#!/usr/bin/env python3
"""Step 14 - Leave-one-host-out extrapolation validation (Section 3.7, Figures 8 & 9).

Tests whether the composition-only Random Forest model generalizes to unseen host
chemistries or primarily interpolates within known host families:
  Experiment 1: train on (In + Sn) oxides -> test on Zn oxides (n = 2,605)
  Experiment 2: train on (Zn + Sn) oxides -> test on In oxides (n = 946)
  Experiment 3: train on (Zn + In) oxides -> test on Sn oxides (n = 1,656)

Output: figures/publication/figS1_leave_one_host_out.{png,pdf} (Figure 8 in manuscript)
        figures/publication/figS2_loho_comparison.{png,pdf}    (Figure 9 in manuscript)
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tco_stability.paths import ensure_project_root
ensure_project_root()

import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

from tco_stability.features import (
    parse_composition,
    featurize_composition_enhanced as featurize_composition,
)

import warnings; warnings.filterwarnings('ignore')

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
C_BLACK  = '#1A1A1A'
HOST_COLORS = {'Zn': C_BLUE, 'In': C_RED, 'Sn': C_GREEN}


def panel_label(ax, label, x=-0.14, y=1.08):
    ax.text(x, y, f'({label})', transform=ax.transAxes,
            fontsize=11, fontweight='bold', va='top', ha='left')


def metric_box(ax, text, x=0.05, y=0.92, ha='left'):
    ax.text(x, y, text, transform=ax.transAxes, fontsize=7.5, va='top', ha=ha,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      alpha=0.8, edgecolor='#cccccc', linewidth=0.4))


if __name__ == '__main__':
    os.makedirs(FIG_DIR, exist_ok=True)

    print("=" * 65)
    print("STEP 14: LEAVE-ONE-HOST-OUT VALIDATION (Section 3.7)")
    print("=" * 65)

    print("\nLoading data...")
    df = pd.read_csv('data/processed/mp_doped_oxides_cleaned.csv')
    unsupported = {'Pm', 'Pu', 'Tc', 'Np', 'Pa'}
    df = df[~df['elements'].apply(
        lambda x: any(e.strip() in unsupported for e in x.split(',')))
    ].reset_index(drop=True)

    print("Featurizing...")
    records, skipped = [], []
    for idx, row in df.iterrows():
        try:
            records.append(featurize_composition(parse_composition(row['composition'])))
        except (KeyError, ZeroDivisionError, ValueError):
            skipped.append(idx)
    df = df.drop(index=skipped).reset_index(drop=True)
    X_all = pd.DataFrame(records).values
    y_all = df['energy_above_hull'].values
    hosts = df['host_element'].values
    print(f"  {len(df)} samples, {X_all.shape[1]} features")
    print(f"  Zn: {(hosts=='Zn').sum()}, In: {(hosts=='In').sum()}, Sn: {(hosts=='Sn').sum()}")

    rf_params = dict(n_estimators=200, max_depth=None, min_samples_split=2,
                     min_samples_leaf=1, max_features='sqrt',
                     random_state=RNG_SEED, n_jobs=-1)

    host_list = ['Zn', 'In', 'Sn']
    results = {}

    for held_out in host_list:
        train_mask = hosts != held_out
        test_mask = hosts == held_out

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_all[train_mask])
        X_test = scaler.transform(X_all[test_mask])
        y_train = y_all[train_mask]
        y_test = y_all[test_mask]

        rf = RandomForestRegressor(**rf_params)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(np.mean((y_test - y_pred)**2))
        dummy_mae = mean_absolute_error(y_test, np.full_like(y_test, y_train.mean()))

        results[held_out] = dict(
            y_test=y_test, y_pred=y_pred, mae=mae, r2=r2, rmse=rmse,
            dummy_mae=dummy_mae, n_train=len(y_train), n_test=len(y_test))

        print(f"\n  Hold out: {held_out}-oxides")
        print(f"    Train: {len(y_train)} samples ({', '.join(h for h in host_list if h != held_out)})")
        print(f"    Test:  {len(y_test)} samples ({held_out})")
        print(f"    MAE  = {mae:.4f} eV/atom")
        print(f"    RMSE = {rmse:.4f} eV/atom")
        print(f"    R²   = {r2:.4f}")
        print(f"    Baseline MAE = {dummy_mae:.4f}")
        print(f"    Improvement  = {(1 - mae/dummy_mae)*100:.1f}%")

    print(f"\n{'='*65}")
    print("COMPARISON: Leave-One-Host-Out vs Standard Composition Split")
    print(f"{'='*65}")
    print("  Standard split (composition-based):")
    print("    Test MAE = 0.0632 eV/atom, R² = 0.5596 (RF) / 0.6056 (GBR)")
    print("\n  Leave-one-host-out:")
    for h in host_list:
        r = results[h]
        print(f"    Hold out {h}: MAE = {r['mae']:.4f}, R² = {r['r2']:.4f}")
    avg_mae = np.mean([results[h]['mae'] for h in host_list])
    avg_r2 = np.mean([results[h]['r2'] for h in host_list])
    print(f"\n    Average:     MAE = {avg_mae:.4f}, R² = {avg_r2:.4f}")

    print("\nGenerating Figure S1 (Manuscript Fig. 8)...")
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.8))

    for idx, held_out in enumerate(host_list):
        ax = axes[idx]
        panel_label(ax, chr(97 + idx))
        r = results[held_out]

        ax.scatter(r['y_test'], r['y_pred'], alpha=0.35, s=8,
                   c=HOST_COLORS[held_out], edgecolors='none', rasterized=True)
        lim = [0, max(r['y_test'].max(), r['y_pred'].max()) * 1.05]
        ax.plot(lim, lim, color=C_BLACK, ls='--', lw=0.7)
        ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect('equal')
        ax.set_xlabel(r'DFT $E_{\mathrm{hull}}$ (eV/atom)')
        if idx == 0:
            ax.set_ylabel(r'Predicted $E_{\mathrm{hull}}$ (eV/atom)')

        title = f'Test: {held_out}-oxides (n={r["n_test"]})'
        ax.set_title(title, fontsize=8)

        metric_box(ax,
            f'MAE = {r["mae"]:.4f}\n$R^2$ = {r["r2"]:.4f}',
            x=0.05, y=0.92)

    plt.tight_layout(w_pad=1.0)
    fig.savefig(f'{FIG_DIR}/figS1_leave_one_host_out.png', dpi=DPI)
    fig.savefig(f'{FIG_DIR}/figS1_leave_one_host_out.pdf')
    plt.close(fig)
    print(f"  Saved: {FIG_DIR}/figS1_leave_one_host_out.{{png,pdf}}")

    print("Generating Figure S2 (Manuscript Fig. 9)...")
    fig, ax = plt.subplots(figsize=(4.0, 2.8))

    x_pos = np.arange(4)
    maes = [avg_mae] + [results[h]['mae'] for h in host_list]
    colors = ['#878787'] + [HOST_COLORS[h] for h in host_list]
    labels = ['Average'] + [f'Hold {h}' for h in host_list]

    bars = ax.bar(x_pos, maes, color=colors, edgecolor='white', lw=0.3, width=0.55)
    for b, v in zip(bars, maes):
        ax.text(b.get_x() + b.get_width()/2, v + 0.001,
                f'{v:.4f}', ha='center', va='bottom', fontsize=7.5)

    ax.axhline(y=0.0632, color=C_BLACK, ls='--', lw=0.7, alpha=0.6)
    ax.text(3.5, 0.064, 'Std. split\n(0.0632)', fontsize=6.5,
            ha='right', va='bottom', color=C_BLACK, alpha=0.7)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel('Test MAE (eV/atom)')
    ax.set_ylim(0, max(maes) * 1.2)
    ax.set_title('Leave-One-Host-Out vs Standard Split', fontsize=9)

    plt.tight_layout()
    fig.savefig(f'{FIG_DIR}/figS2_loho_comparison.png', dpi=DPI)
    fig.savefig(f'{FIG_DIR}/figS2_loho_comparison.pdf')
    plt.close(fig)
    print(f"  Saved: {FIG_DIR}/figS2_loho_comparison.{{png,pdf}}")

    print(f"\n{'='*65}")
    print("LEAVE-ONE-HOST-OUT VALIDATION COMPLETE")
    print(f"{'='*65}")