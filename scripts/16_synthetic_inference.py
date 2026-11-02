#!/usr/bin/env python3
"""Step 16 - Synthetic dopant inference test (Section 3.8, Figure 11).

Tests directly whether the trained Random Forest model resolves dilute dopant
concentration differences (0.1% to 50%) across ZnO, In2O3, and SnO2 hosts doped
with Al, Ga, F, and Sb:
  1. Generate synthetic compositions for ZnO, In2O3, and SnO2 doped with
     Al, Ga, F, and Sb at concentrations from 0.1% to 50%.
  2. Encode each composition into the 118-feature Magpie vector.
  3. Predict Ehull using the Random Forest trained on train+val (4,686 samples).
  4. Plot predicted Ehull vs dopant concentration (Figure 11: three hosts, 0-10%).

Outputs
  figures/publication/fig_synthetic_inference_all_hosts.{png,pdf}
  figures/revision/fig_synthetic_inference_all_hosts.{png,eps}
  figures/revision/fig_synthetic_inference_ZnO_full_range.png
  results/synthetic_inference_predictions.csv
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

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

from tco_stability.features import (
    ELEM_PROPS,
    parse_composition,
    featurize_composition_enhanced,
)

RNG_SEED = 42
np.random.seed(RNG_SEED)

os.makedirs("figures/revision", exist_ok=True)
os.makedirs("figures/publication", exist_ok=True)
os.makedirs("results", exist_ok=True)

print("=" * 70)
print("STEP 16: Synthetic Dopant Inference Test (Section 3.8)")
print("=" * 70)

print("\n[1/4] Rebuilding pipeline and training model...")
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

X_train = df_features[train_mask].values
X_val = df_features[val_mask].values
y_train, y_val = y[train_mask], y[val_mask]

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s = scaler.transform(X_val)

X_trainval = np.vstack([X_train_s, X_val_s])
y_trainval = np.concatenate([y_train, y_val])

rf_final = RandomForestRegressor(
    n_estimators=200, max_depth=None, min_samples_split=2,
    min_samples_leaf=1, max_features='sqrt', random_state=RNG_SEED, n_jobs=-1
)
rf_final.fit(X_trainval, y_trainval)
print(f"  Model trained on {len(X_trainval)} samples, {len(feature_names)} features")

print("\n[2/4] Generating synthetic doped compositions...")

HOST_SYSTEMS = {
    "ZnO":    {"host_cation": "Zn", "composition": {"Zn": 1.0, "O": 1.0}},
    "In2O3":  {"host_cation": "In", "composition": {"In": 2.0, "O": 3.0}},
    "SnO2":   {"host_cation": "Sn", "composition": {"Sn": 1.0, "O": 2.0}},
}

DOPANTS = ["Al", "Ga", "F", "Sb"]

CONCENTRATIONS = np.concatenate([
    np.arange(0.001, 0.05, 0.002),   # 0.1% to 5% in 0.2% steps
    np.arange(0.05, 0.51, 0.01),     # 5% to 50% in 1% steps
])

all_results = []

for host_name, host_info in HOST_SYSTEMS.items():
    host_cation = host_info["host_cation"]
    base_comp = host_info["composition"].copy()
    cation_amount = base_comp[host_cation]

    for dopant in DOPANTS:
        if dopant not in ELEM_PROPS:
            continue

        for conc in CONCENTRATIONS:
            doped_comp = base_comp.copy()
            dopant_amount = cation_amount * conc
            doped_comp[host_cation] = cation_amount - dopant_amount
            doped_comp[dopant] = dopant_amount

            doped_comp = {k: v for k, v in doped_comp.items() if v > 1e-10}

            try:
                feats = featurize_composition_enhanced(doped_comp)
                feat_vec = np.array([feats[fn] for fn in feature_names]).reshape(1, -1)
                feat_scaled = scaler.transform(feat_vec)
                pred_ehull = rf_final.predict(feat_scaled)[0]

                all_results.append({
                    'host': host_name,
                    'dopant': dopant,
                    'concentration_pct': conc * 100,
                    'predicted_ehull': pred_ehull,
                    'formula': f"{host_cation}{cation_amount - dopant_amount:.3f}{dopant}{dopant_amount:.3f}O{base_comp['O']:.0f}",
                })
            except (KeyError, ValueError):
                pass

df_results = pd.DataFrame(all_results)
df_results.to_csv("results/synthetic_inference_predictions.csv", index=False)
print(f"  Generated {len(df_results)} synthetic predictions")
print("  Saved to results/synthetic_inference_predictions.csv")

print("\n[3/4] Analyzing concentration sensitivity of predictions...")
print("\n  Predicted Ehull at key concentrations for ZnO:Al:")
zno_al = df_results[(df_results['host'] == 'ZnO') & (df_results['dopant'] == 'Al')]
for target_conc in [0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0]:
    row = zno_al.iloc[(zno_al['concentration_pct'] - target_conc).abs().argsort()[:1]]
    print(f"    {target_conc:5.1f}% Al -> predicted Ehull = {row['predicted_ehull'].values[0]:.4f} eV/atom")

for host_name in HOST_SYSTEMS:
    for dopant in DOPANTS:
        subset = df_results[(df_results['host'] == host_name) & (df_results['dopant'] == dopant)]
        if len(subset) == 0:
            continue
        low_range = subset[(subset['concentration_pct'] >= 1) & (subset['concentration_pct'] <= 5)]
        high_range = subset[(subset['concentration_pct'] >= 5) & (subset['concentration_pct'] <= 50)]
        if len(low_range) > 0 and len(high_range) > 0:
            low_span = low_range['predicted_ehull'].max() - low_range['predicted_ehull'].min()
            high_span = high_range['predicted_ehull'].max() - high_range['predicted_ehull'].min()
            print(f"  {host_name}:{dopant:2s}  1-5% range: {low_span:.4f} eV/atom | 5-50% range: {high_span:.4f} eV/atom")

print("\n[4/4] Generating figures...")

colors_dopant = {'Al': '#E53935', 'Ga': '#1E88E5', 'F': '#43A047', 'Sb': '#FB8C00'}
markers = {'Al': 'o', 'Ga': 's', 'F': '^', 'Sb': 'D'}

# Auxiliary plot: ZnO full range (0-50%)
fig, ax = plt.subplots(figsize=(9, 6))
for dopant in DOPANTS:
    subset = df_results[(df_results['host'] == 'ZnO') & (df_results['dopant'] == dopant)]
    if len(subset) == 0:
        continue
    ax.plot(subset['concentration_pct'], subset['predicted_ehull'],
            color=colors_dopant.get(dopant, 'gray'), marker=markers.get(dopant, 'o'),
            markersize=3, linewidth=1.5, label=f'ZnO:{dopant}', alpha=0.8)

ax.axvspan(0, 5, alpha=0.1, color='gray', label='Typical doping range (0-5%)')
ax.set_xlabel("Dopant Concentration (% of cation site)", fontsize=12)
ax.set_ylabel("Predicted $E_{hull}$ (eV/atom)", fontsize=12)
ax.set_title("Synthetic Inference Test: ZnO with Various Dopants (0-50%)", fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xlim(0, 52)
plt.tight_layout()
plt.savefig("figures/revision/fig_synthetic_inference_ZnO_full_range.png", dpi=300, bbox_inches='tight')
plt.close(fig)

# Main Manuscript Figure 11: all three hosts zoomed on 0-10%
fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)

for ax_idx, (host_name, host_info) in enumerate(HOST_SYSTEMS.items()):
    ax = axes[ax_idx]
    for dopant in DOPANTS:
        subset = df_results[(df_results['host'] == host_name) & (df_results['dopant'] == dopant)]
        subset_zoom = subset[subset['concentration_pct'] <= 10]
        if len(subset_zoom) == 0:
            continue
        ax.plot(subset_zoom['concentration_pct'], subset_zoom['predicted_ehull'],
                color=colors_dopant.get(dopant, 'gray'), marker=markers.get(dopant, 'o'),
                markersize=4, linewidth=1.5, label=f'{dopant}', alpha=0.8)

    ax.axvspan(0, 5, alpha=0.08, color='gray')
    ax.set_xlabel("Dopant Concentration (%)", fontsize=10)
    ax.set_title(f"{host_name}", fontsize=12, fontweight='bold')
    ax.legend(fontsize=8, title="Dopant")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlim(0, 10.5)

axes[0].set_ylabel("Predicted $E_{hull}$ (eV/atom)", fontsize=11)
fig.suptitle("Synthetic Inference: Predicted Stability vs Dopant Concentration (0-10%)",
             fontsize=13, fontweight='bold', y=1.02)

plt.tight_layout()
plt.savefig("figures/revision/fig_synthetic_inference_all_hosts.png", dpi=300, bbox_inches='tight')
plt.savefig("figures/revision/fig_synthetic_inference_all_hosts.eps", bbox_inches='tight')
plt.savefig("figures/publication/fig_synthetic_inference_all_hosts.png", dpi=300, bbox_inches='tight')
plt.savefig("figures/publication/fig_synthetic_inference_all_hosts.pdf", bbox_inches='tight')
plt.close(fig)
print("  Saved: figures/publication/fig_synthetic_inference_all_hosts.{png,pdf}")

print("\n" + "=" * 70)
print("SYNTHETIC INFERENCE TEST COMPLETE")
print("=" * 70)