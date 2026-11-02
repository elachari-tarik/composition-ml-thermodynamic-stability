#!/usr/bin/env python3
"""Step 15 - Sensitivity of Magpie composition-based features to dopant concentration (Section 3.8).

Quantifies how strongly the 118-feature Magpie vector shifts as a function of dopant
concentration from 0.1% to 50%:
  1. Take an undoped binary host oxide (ZnO, In2O3, SnO2).
  2. Substitute a cation-site dopant (Al, Ga, Sn, F, Sb) at concentrations from 0.1% to 50%.
  3. Compute the 118 Magpie features at each concentration.
  4. Measure the Euclidean (L2) distance relative to the pristine host.
  5. Decompose the shift into stoichiometric and composition-weighted mean contributions.

Output
  figures/publication/figS3_dopant_sensitivity.{png,pdf} (Figure 10 in manuscript)
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

from tco_stability.features import featurize_composition_enhanced as featurize_composition

import warnings; warnings.filterwarnings('ignore')

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
C_PURPLE = '#7B3294'
C_ORANGE = '#E6AB02'
C_GRAY   = '#878787'
C_BLACK  = '#1A1A1A'


def get_feature_category(name):
    if name.startswith('stoich_'): return 'Stoichiometric'
    if name.startswith('elec_'): return 'Electronic'
    if name.startswith('ionic_'): return 'Ionic'
    return 'Elemental'


def panel_label(ax, label, x=-0.14, y=1.08):
    ax.text(x, y, f'({label})', transform=ax.transAxes,
            fontsize=11, fontweight='bold', va='top', ha='left')


if __name__ == '__main__':
    os.makedirs(FIG_DIR, exist_ok=True)

    print("=" * 65)
    print("STEP 15: DOPANT SENSITIVITY ANALYSIS (Section 3.8)")
    print("=" * 65)

    systems = [
        {'host_comp': {'Zn': 1, 'O': 1}, 'dopant': 'Al', 'label': 'ZnO:Al (AZO)',
         'color': C_BLUE, 'ls': '-'},
        {'host_comp': {'Zn': 1, 'O': 1}, 'dopant': 'Ga', 'label': 'ZnO:Ga (GZO)',
         'color': C_PURPLE, 'ls': '--'},
        {'host_comp': {'In': 2, 'O': 3}, 'dopant': 'Sn', 'label': r'In$_2$O$_3$:Sn (ITO)',
         'color': C_RED, 'ls': '-'},
        {'host_comp': {'Sn': 1, 'O': 2}, 'dopant': 'F', 'label': r'SnO$_2$:F (FTO)',
         'color': C_GREEN, 'ls': '-'},
        {'host_comp': {'Sn': 1, 'O': 2}, 'dopant': 'Sb', 'label': r'SnO$_2$:Sb (ATO)',
         'color': C_ORANGE, 'ls': '--'},
    ]

    concentrations = np.array([0.001, 0.005, 0.01, 0.02, 0.03, 0.05,
                                0.10, 0.15, 0.20, 0.30, 0.40, 0.50])

    print("\nComputing feature vectors across dopant concentrations...")
    all_results = []

    for system in systems:
        host = system['host_comp']
        dopant = system['dopant']
        label = system['label']

        ref_features = featurize_composition(host)
        ref_vec = np.array(list(ref_features.values()))
        feature_names = list(ref_features.keys())

        l2_distances = []
        max_feature_changes = []
        stoich_changes = []
        weighted_changes = []

        for conc in concentrations:
            doped = dict(host)
            cation = [e for e in host if e != 'O'][0]
            original_amount = doped[cation]
            dopant_amount = original_amount * conc
            doped[cation] = original_amount - dopant_amount
            doped[dopant] = dopant_amount

            doped_features = featurize_composition(doped)
            doped_vec = np.array(list(doped_features.values()))

            diff = doped_vec - ref_vec
            l2 = np.linalg.norm(diff)
            l2_distances.append(l2)

            safe_ref = np.where(np.abs(ref_vec) > 1e-10, ref_vec, 1.0)
            rel_changes = np.abs(diff / safe_ref)
            max_feature_changes.append(np.max(rel_changes))

            stoich_idx = [i for i, n in enumerate(feature_names) if n.startswith('stoich_')]
            stoich_change = np.linalg.norm(diff[stoich_idx])
            stoich_changes.append(stoich_change)

            wmean_idx = [i for i, n in enumerate(feature_names) if n.endswith('_wmean')]
            wmean_change = np.linalg.norm(diff[wmean_idx])
            weighted_changes.append(wmean_change)

        all_results.append({
            'label': label, 'color': system['color'], 'ls': system['ls'],
            'dopant': dopant, 'l2': l2_distances,
            'max_rel': max_feature_changes,
            'stoich': stoich_changes, 'wmean': weighted_changes,
        })

        for ci, conc in enumerate(concentrations):
            if conc in [0.01, 0.05, 0.10]:
                print(f"\n  {label} at {conc*100:.0f}% dopant:")
                print(f"    L2 distance:             {l2_distances[ci]:.4f}")
                print(f"    Stoichiometric shift:    {stoich_changes[ci]:.4f}")
                print(f"    Weighted-mean shift:     {weighted_changes[ci]:.4f}")
                print(f"    Max relative change:     {max_feature_changes[ci]:.4f} ({max_feature_changes[ci]*100:.1f}%)")

    print(f"\n{'='*65}")
    print("DETAILED: Feature shifts for 1% Al in ZnO")
    print(f"{'='*65}")
    host = {'Zn': 1, 'O': 1}
    doped_1pct = {'Zn': 0.99, 'O': 1, 'Al': 0.01}
    ref_f = featurize_composition(host)
    doped_f = featurize_composition(doped_1pct)

    changes = []
    for feat in ref_f:
        ref_val = ref_f[feat]
        doped_val = doped_f[feat]
        abs_change = abs(doped_val - ref_val)
        rel_change = abs_change / abs(ref_val) if abs(ref_val) > 1e-10 else float('inf')
        changes.append((feat, ref_val, doped_val, abs_change, rel_change))

    changes.sort(key=lambda x: x[4], reverse=True)
    print(f"\n  {'Feature':<35s} {'Undoped':>10s} {'1% Al':>10s} {'Abs Δ':>10s} {'Rel Δ':>8s}")
    print("  " + "-" * 78)
    for feat, ref_val, doped_val, abs_c, rel_c in changes[:20]:
        cat = get_feature_category(feat)
        rel_str = f"{rel_c*100:.1f}%" if rel_c < 100 else "new"
        print(f"  {feat:<35s} {ref_val:10.4f} {doped_val:10.4f} {abs_c:10.4f} {rel_str:>8s}  [{cat}]")

    tiny = sum(1 for _, _, _, _, r in changes if r < 0.01)
    small = sum(1 for _, _, _, _, r in changes if r < 0.05)
    print(f"\n  Features changing <1%:  {tiny}/{len(changes)} ({tiny/len(changes)*100:.0f}%)")
    print(f"  Features changing <5%:  {small}/{len(changes)} ({small/len(changes)*100:.0f}%)")

    print(f"\n{'='*65}")
    print("Generating Figure S3 (Manuscript Fig. 10)...")

    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.8))

    ax = axes[0]; panel_label(ax, 'a')
    for r in all_results:
        ax.plot(concentrations * 100, r['l2'], color=r['color'], ls=r['ls'],
                lw=1.2, marker='o', ms=3, label=r['label'])
    ax.set_xlabel('Dopant concentration (%)')
    ax.set_ylabel(r'$\|\Delta \mathbf{f}\|_2$ (L2 distance)')
    ax.set_xlim(0, 52)
    ax.set_xticks([0, 10, 20, 30, 40, 50])
    ax.set_ylim(0, 820)
    ax.axvspan(0, 5, alpha=0.15, color=C_GRAY)
    ax.text(2.5, 255, 'Typical\ndoping\nrange',
            fontsize=4.5, color='#222222', ha='center', va='center',
            style='italic', fontweight='semibold', linespacing=1.2)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 0.995), ncol=3,
              fontsize=5, frameon=True, edgecolor='#cccccc', fancybox=False,
              framealpha=0.9, handlelength=0.9, handletextpad=0.3,
              borderpad=0.3, columnspacing=0.5, labelspacing=0.2)

    ax = axes[1]; panel_label(ax, 'b')
    r0 = all_results[0]
    ax.plot(concentrations * 100, r0['stoich'], color=C_RED, ls='-',
            lw=1.2, marker='s', ms=3, label='Stoichiometric features')
    ax.plot(concentrations * 100, r0['wmean'], color=C_BLUE, ls='-',
            lw=1.2, marker='o', ms=3, label='Weighted-mean features')
    ax.set_xlabel('Dopant concentration (%)')
    ax.set_ylabel(r'Feature shift ($\|\Delta\|_2$)')
    ax.set_xlim(0, 52)
    ax.set_xticks([0, 10, 20, 30, 40, 50])
    ax.axvspan(0, 5, alpha=0.08, color=C_GRAY)
    ax.legend(fontsize=6, frameon=True, edgecolor='#cccccc', loc='upper left',
              fancybox=False)
    ax.set_title('ZnO:Al breakdown', fontsize=8)

    ax = axes[2]; panel_label(ax, 'c')
    thresholds_pct = [1, 5, 10]
    host = {'Zn': 1, 'O': 1}
    ref_f = featurize_composition(host)
    ref_vec = np.array(list(ref_f.values()))

    for thresh, color, marker in zip(thresholds_pct, [C_GREEN, C_ORANGE, C_RED], ['o', 's', '^']):
        counts = []
        for conc in concentrations:
            doped = {'Zn': 1 - conc, 'O': 1, 'Al': conc}
            doped_vec = np.array(list(featurize_composition(doped).values()))
            safe_ref = np.where(np.abs(ref_vec) > 1e-10, ref_vec, 1.0)
            rel_changes = np.abs((doped_vec - ref_vec) / safe_ref)
            n_above = np.sum(rel_changes > thresh / 100.0)
            counts.append(n_above)
        ax.plot(concentrations * 100, counts, color=color, ls='-',
                lw=1.2, marker=marker, ms=3, label=f'>{thresh}% change')

    ax.set_xlabel('Dopant concentration (%)')
    ax.set_ylabel('N features above threshold')
    ax.set_xlim(0, 52)
    ax.set_xticks([0, 10, 20, 30, 40, 50])
    ax.axhline(y=118, color=C_GRAY, ls=':', lw=0.5, alpha=0.5)
    ax.text(50, 118, '118 total', fontsize=6, ha='right', va='bottom', color=C_GRAY)
    ax.axvspan(0, 5, alpha=0.08, color=C_GRAY)
    ax.legend(fontsize=6, frameon=True, edgecolor='#cccccc', loc='upper left',
              fancybox=False)
    ax.set_title('ZnO:Al feature sensitivity', fontsize=8)

    plt.tight_layout(w_pad=1.2)
    fig.savefig(f'{FIG_DIR}/figS3_dopant_sensitivity.png', dpi=DPI)
    fig.savefig(f'{FIG_DIR}/figS3_dopant_sensitivity.pdf')
    plt.close(fig)
    print(f"  Saved: {FIG_DIR}/figS3_dopant_sensitivity.{{png,pdf}}")

    print(f"\n{'='*65}")
    print("DOPANT SENSITIVITY ANALYSIS COMPLETE")
    print(f"{'='*65}")