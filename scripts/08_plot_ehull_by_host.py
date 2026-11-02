"""Step 08 - Energy above hull by host element (box plot).

Phase 1, step 1.4. Compares the stability distribution of Zn-, In- and Sn-based oxides.
Output: figures/phase1/02_ehull_by_host.png
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

# Summary statistics per host element
print("\nEnergy Above Hull statistics by host element:")
print("-" * 55)
for host in ["Zn", "In", "Sn"]:
    subset = df[df["host_element"] == host]["energy_above_hull"]
    print(f"  {host}-oxides: n={len(subset):>4d}, "
          f"mean={subset.mean():.4f}, "
          f"median={subset.median():.4f}, "
          f"std={subset.std():.4f}")

# ── Box plot ─────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))

host_labels = ["Zn-oxides", "In-oxides", "Sn-oxides"]
host_data = [
    df[df["host_element"] == "Zn"]["energy_above_hull"],
    df[df["host_element"] == "In"]["energy_above_hull"],
    df[df["host_element"] == "Sn"]["energy_above_hull"],
]

bp = ax.boxplot(
    host_data,
    tick_labels=host_labels,
    patch_artist=True,           # filled boxes
    showfliers=True,             # show outliers
    flierprops=dict(
        marker=".",
        markersize=2,
        alpha=0.3
    ),
    medianprops=dict(
        color="black",
        linewidth=1.5
    ),
)

colors = ["#2196F3", "#4CAF50", "#FF9800"]
for patch, color in zip(bp["boxes"], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax.set_ylabel("Energy Above Hull (eV/atom)")
ax.set_title("Stability Distribution by Host Element")

# Reference line at 0.1 eV, a common synthesizability threshold
ax.axhline(y=0.1, color="red", linestyle="--", linewidth=1,
           alpha=0.7, label="0.1 eV threshold")
ax.legend(fontsize=9, loc="upper right")

os.makedirs("figures/phase1", exist_ok=True)
plt.tight_layout()
fig_path = "figures/phase1/02_ehull_by_host.png"
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nFigure saved to: {fig_path}")
