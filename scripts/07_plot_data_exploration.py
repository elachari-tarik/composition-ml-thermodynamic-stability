"""Step 07 - Exploratory overview of the cleaned dataset (6-panel figure).

Phase 1, step 1.4. Output: figures/phase1/01_data_exploration.png
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
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Load cleaned data ────────────────────────────────────────────────────────
df = pd.read_csv("data/processed/mp_doped_oxides_cleaned.csv")
print(f"Loaded cleaned data: {df.shape[0]} rows, {df.shape[1]} columns")

# ── 2x3 figure ───────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("Cleaned Dataset: Exploratory Analysis",
             fontsize=14, fontweight="bold")

# Panel 1: energy above hull (the target)
ax = axes[0, 0]
ax.hist(df["energy_above_hull"], bins=60,
        color="#2196F3", edgecolor="white", alpha=0.85)
ax.set_xlabel("Energy Above Hull (eV/atom)")
ax.set_ylabel("Count")
ax.set_title("Target: Energy Above Hull")
ax.axvline(x=0.1, color="red", linestyle="--", linewidth=1,
           label="0.1 eV threshold")
ax.legend(fontsize=8)

# Panel 2: formation energy
ax = axes[0, 1]
ax.hist(df["formation_energy_per_atom"], bins=60,
        color="#4CAF50", edgecolor="white", alpha=0.85)
ax.set_xlabel("Formation Energy (eV/atom)")
ax.set_ylabel("Count")
ax.set_title("Formation Energy per Atom")

# Panel 3: band gap
ax = axes[0, 2]
ax.hist(df["band_gap"], bins=60,
        color="#FF9800", edgecolor="white", alpha=0.85)
ax.set_xlabel("Band Gap (eV)")
ax.set_ylabel("Count")
ax.set_title("Band Gap Distribution")

# Panel 4: Ehull vs formation energy, coloured by number of elements
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

# Panel 5: number of elements (compositional complexity)
ax = axes[1, 1]
nel_counts = df["nelements"].value_counts().sort_index()
ax.bar(nel_counts.index, nel_counts.values,
       color="#9C27B0", edgecolor="white")
ax.set_xlabel("Number of Elements")
ax.set_ylabel("Count")
ax.set_title("Compositional Complexity")
ax.set_xticks(nel_counts.index)

# Panel 6: crystal systems
ax = axes[1, 2]
cs_counts = df["crystal_system"].value_counts()
ax.barh(cs_counts.index, cs_counts.values,
        color="#00BCD4", edgecolor="white")
ax.set_xlabel("Count")
ax.set_title("Crystal System Distribution")

os.makedirs("figures/phase1", exist_ok=True)
plt.tight_layout()
fig_path = "figures/phase1/01_data_exploration.png"
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Figure saved to: {fig_path}")
