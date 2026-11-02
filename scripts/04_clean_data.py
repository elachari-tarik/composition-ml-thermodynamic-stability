"""Step 04 - Clean the raw Materials Project data.

Phase 1, step 1.4. Follows the data-cleaning practices of Wang et al.
(Chem. Mater. 2020, 32, 4954-4965): document every step, keep both the raw and the
cleaned copy, and deal explicitly with missing values, outliers and unrealistic entries.

Outputs
  data/processed/mp_doped_oxides_cleaned.csv
  data/processed/cleaning_log.txt
  figures/01_data_exploration.png, figures/02_ehull_by_host.png
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tco_stability.paths import ensure_project_root
ensure_project_root()

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # no display needed, we only save figures
import matplotlib.pyplot as plt
import seaborn as sns

# Reproducibility seed
RNG_SEED = 42
np.random.seed(RNG_SEED)

# =============================================================================
# Step 0: Load raw data
# =============================================================================
print("=" * 70)
print("PHASE 1, STEP 1.4: DATA CLEANING & INITIAL EXPLORATION")
print("=" * 70)

raw_path = "data/raw/mp_doped_oxides_raw.csv"
df_raw = pd.read_csv(raw_path)
print(f"\n[LOAD] Raw data: {df_raw.shape[0]} rows, {df_raw.shape[1]} columns")

# Work on a copy so the raw frame stays intact for the final comparison
df = df_raw.copy()
cleaning_log = []  # every cleaning step is written here

# =============================================================================
# Step 1: Missing values
# =============================================================================
print("\n--- Step 1: Missing Values ---")
missing = df.isnull().sum()
missing_cols = missing[missing > 0]
if len(missing_cols) == 0:
    print("  ✓ No missing values found")
    cleaning_log.append("No missing values found in any column.")
else:
    print(f"  ! Missing values found in: {dict(missing_cols)}")
    # Rows without a target value are useless, drop them
    before = len(df)
    df = df.dropna(subset=["energy_above_hull"])
    dropped = before - len(df)
    if dropped > 0:
        cleaning_log.append(f"Dropped {dropped} rows with missing energy_above_hull.")
        print(f"  Dropped {dropped} rows with missing target variable")

# =============================================================================
# Step 2: Positive formation energy
# =============================================================================
print("\n--- Step 2: Positive Formation Energy Filter ---")
# Compounds with positive formation energy are thermodynamically unstable
# relative to their elements. Some may be genuine metastable phases, but many are
# unreliable calculations, so they are removed.
pos_fe_count = (df["formation_energy_per_atom"] >= 0).sum()
print(f"  Entries with positive formation energy: {pos_fe_count}")
before = len(df)
df = df[df["formation_energy_per_atom"] < 0]
cleaning_log.append(
    f"Removed {before - len(df)} entries with positive formation energy "
    f"(thermodynamically unstable relative to elements)."
)
print(f"  Removed {before - len(df)} entries. Remaining: {len(df)}")

# =============================================================================
# Step 3: Extreme energy-above-hull values
# =============================================================================
print("\n--- Step 3: Energy Above Hull Outlier Filter ---")
# Very high Ehull usually means a hypothetical structure nobody could synthesize.
# 2.0 eV/atom is deliberately generous (most studies use 0.5-1.0 eV/atom); we stay
# inclusive here.
EHULL_THRESHOLD = 2.0
extreme_count = (df["energy_above_hull"] > EHULL_THRESHOLD).sum()
print(f"  Entries with Ehull > {EHULL_THRESHOLD} eV/atom: {extreme_count}")
before = len(df)
df = df[df["energy_above_hull"] <= EHULL_THRESHOLD]
cleaning_log.append(
    f"Removed {before - len(df)} entries with energy_above_hull > "
    f"{EHULL_THRESHOLD} eV/atom (highly unstable/hypothetical structures)."
)
print(f"  Removed {before - len(df)} entries. Remaining: {len(df)}")

# =============================================================================
# Step 4: Polymorphs (several structures per formula)
# =============================================================================
print("\n--- Step 4: Polymorph Handling ---")
polymorph_counts = df["formula_pretty"].value_counts()
n_polymorphs = (polymorph_counts > 1).sum()
max_poly = polymorph_counts.max()
print(f"  Formulas with multiple entries (polymorphs): {n_polymorphs}")
print(f"  Maximum entries for one formula: {max_poly} ({polymorph_counts.idxmax()})")

# Keep the most stable polymorph (lowest energy_above_hull), the most meaningful
# choice for a stability model
before = len(df)
df = df.sort_values(["energy_above_hull", "material_id"]).drop_duplicates(
    subset="formula_pretty", keep="first"
).reset_index(drop=True)
cleaning_log.append(
    f"For {n_polymorphs} formulas with multiple polymorphs, kept only the "
    f"most stable structure (lowest energy_above_hull). "
    f"Reduced from {before} to {len(df)} entries."
)
print(f"  Kept most stable polymorph per formula: {before} → {len(df)} entries")

# =============================================================================
# Step 5: Data integrity checks
# =============================================================================
print("\n--- Step 5: Data Integrity Checks ---")
checks_passed = True

# Energy above hull cannot be negative
if (df["energy_above_hull"] < 0).any():
    print("  ✗ Negative energy_above_hull values found!")
    checks_passed = False
else:
    print("  ✓ energy_above_hull: all non-negative")

# Neither can the band gap
if (df["band_gap"] < 0).any():
    print("  ✗ Negative band_gap values found!")
    checks_passed = False
else:
    print("  ✓ band_gap: all non-negative")

# Density and volume must be strictly positive
for col in ["density", "volume"]:
    if (df[col] <= 0).any():
        print(f"  ✗ Non-positive {col} values found!")
        checks_passed = False
    else:
        print(f"  ✓ {col}: all positive")

if checks_passed:
    cleaning_log.append("All data integrity checks passed.")

# =============================================================================
# Step 6: Summary of the cleaned data
# =============================================================================
print("\n" + "=" * 70)
print("CLEANED DATA SUMMARY")
print("=" * 70)
print(f"\nDataset shape: {df.shape}")
print(f"Unique formulas: {df['formula_pretty'].nunique()}")
print(f"Number of elements range: {df['nelements'].min()} - {df['nelements'].max()}")

print(f"\nTarget variable (energy_above_hull) statistics:")
print(df["energy_above_hull"].describe().to_string())

print(f"\nFormation energy statistics:")
print(df["formation_energy_per_atom"].describe().to_string())

print(f"\nBand gap distribution:")
print(f"  Metallic (gap=0):      {(df['band_gap'] == 0).sum()}")
print(f"  Semiconductor (gap>0): {(df['band_gap'] > 0).sum()}")

print(f"\nCrystal system distribution:")
print(df["crystal_system"].value_counts().to_string())

print(f"\nNumber of elements distribution:")
print(df["nelements"].value_counts().sort_index().to_string())

print(f"\nHost element distribution:")
print(df["host_element"].value_counts().to_string())

print(f"\nStable (on hull) vs metastable:")
print(df["is_stable"].value_counts().to_string())

# =============================================================================
# Step 7: Save the cleaned data and the cleaning log
# =============================================================================
os.makedirs("data/processed", exist_ok=True)

clean_path = "data/processed/mp_doped_oxides_cleaned.csv"
df.to_csv(clean_path, index=False)
print(f"\n[SAVE] Cleaned data saved to: {clean_path}")

# The log documents the cleaning for reproducibility
log_path = "data/processed/cleaning_log.txt"
with open(log_path, "w") as f:
    f.write("Data Cleaning Log\n")
    f.write("=" * 50 + "\n")
    f.write(f"Raw data: {raw_path} ({df_raw.shape[0]} rows)\n")
    f.write(f"Cleaned data: {clean_path} ({df.shape[0]} rows)\n")
    f.write(f"Total rows removed: {df_raw.shape[0] - df.shape[0]}\n\n")
    f.write("Steps performed:\n")
    for i, step in enumerate(cleaning_log, 1):
        f.write(f"  {i}. {step}\n")
print(f"[SAVE] Cleaning log saved to: {log_path}")

# =============================================================================
# Step 8: Exploratory figures
# =============================================================================
os.makedirs("figures", exist_ok=True)

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("Cleaned Dataset: Exploratory Analysis", fontsize=14, fontweight="bold")

# 1. Target distribution
ax = axes[0, 0]
ax.hist(df["energy_above_hull"], bins=60, color="#2196F3", edgecolor="white", alpha=0.85)
ax.set_xlabel("Energy Above Hull (eV/atom)")
ax.set_ylabel("Count")
ax.set_title("Target: Energy Above Hull")
ax.axvline(x=0.1, color="red", linestyle="--", linewidth=1, label="0.1 eV threshold")
ax.legend(fontsize=8)

# 2. Formation energy
ax = axes[0, 1]
ax.hist(df["formation_energy_per_atom"], bins=60, color="#4CAF50", edgecolor="white", alpha=0.85)
ax.set_xlabel("Formation Energy (eV/atom)")
ax.set_ylabel("Count")
ax.set_title("Formation Energy per Atom")

# 3. Band gap
ax = axes[0, 2]
ax.hist(df["band_gap"], bins=60, color="#FF9800", edgecolor="white", alpha=0.85)
ax.set_xlabel("Band Gap (eV)")
ax.set_ylabel("Count")
ax.set_title("Band Gap Distribution")

# 4. Ehull vs formation energy
ax = axes[1, 0]
scatter = ax.scatter(
    df["formation_energy_per_atom"],
    df["energy_above_hull"],
    c=df["nelements"], cmap="viridis", alpha=0.3, s=8
)
ax.set_xlabel("Formation Energy (eV/atom)")
ax.set_ylabel("Energy Above Hull (eV/atom)")
ax.set_title("Ehull vs Formation Energy")
plt.colorbar(scatter, ax=ax, label="# Elements")

# 5. Number of elements
ax = axes[1, 1]
nel_counts = df["nelements"].value_counts().sort_index()
ax.bar(nel_counts.index, nel_counts.values, color="#9C27B0", edgecolor="white")
ax.set_xlabel("Number of Elements")
ax.set_ylabel("Count")
ax.set_title("Compositional Complexity")
ax.set_xticks(nel_counts.index)

# 6. Crystal systems
ax = axes[1, 2]
cs_counts = df["crystal_system"].value_counts()
ax.barh(cs_counts.index, cs_counts.values, color="#00BCD4", edgecolor="white")
ax.set_xlabel("Count")
ax.set_title("Crystal System Distribution")

plt.tight_layout()
fig_path = "figures/01_data_exploration.png"
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
print(f"[SAVE] Exploration figure saved to: {fig_path}")

# Extra figure: Ehull by host element (box plot)
fig2, ax2 = plt.subplots(figsize=(8, 5))
host_data = [df[df["host_element"] == h]["energy_above_hull"] for h in ["Zn", "In", "Sn"]]
bp = ax2.boxplot(host_data, tick_labels=["Zn-oxides", "In-oxides", "Sn-oxides"],
                  patch_artist=True, showfliers=True,
                  flierprops=dict(marker=".", markersize=2, alpha=0.3))
colors = ["#2196F3", "#4CAF50", "#FF9800"]
for patch, color in zip(bp["boxes"], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax2.set_ylabel("Energy Above Hull (eV/atom)")
ax2.set_title("Stability Distribution by Host Element")
fig2_path = "figures/02_ehull_by_host.png"
plt.savefig(fig2_path, dpi=150, bbox_inches="tight")
print(f"[SAVE] Host element figure saved to: {fig2_path}")

print("\n" + "=" * 70)
print("DATA CLEANING COMPLETE")
print(f"Raw: {df_raw.shape[0]} rows → Cleaned: {df.shape[0]} rows")
print(f"Rows removed: {df_raw.shape[0] - df.shape[0]} "
      f"({(df_raw.shape[0] - df.shape[0]) / df_raw.shape[0] * 100:.1f}%)")
print("=" * 70)
