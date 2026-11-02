"""Step 03 - Query the Materials Project for oxides of the three TCO host elements.

Phase 1, step 1.3. Every oxide that contains Zn, In or Sn (together with O) is
downloaded, which captures both the pure hosts and their doped variants. Entries found
through more than one host query are de-duplicated by material_id. The result is stored
unchanged as the raw dataset.

Requires the MP_API_KEY environment variable.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tco_stability.paths import ensure_project_root
ensure_project_root()

import pandas as pd
from mp_api.client import MPRester
from tco_stability.config import get_api_key

# =============================================================================
# Configuration
# =============================================================================
# Host elements for TCOs
HOST_ELEMENTS = ["Zn", "In", "Sn"]

# Common dopants for TCOs (the list can be expanded)
DOPANT_ELEMENTS = [
    "Al", "Ga", "B",     # Group 13 (common n-type dopants for ZnO)
    "Si", "Ge", "Ti",    # Group 14 / transition metals
    "Sb", "Bi",          # Group 15
    "F", "Cl",           # Halogens (anion-site dopants)
    "Nb", "Ta", "Mo", "W",  # Transition metals
    "Cu", "Ag",          # Potential p-type dopants
    "Mg", "Ca", "Sr",    # Alkaline earth
    "La", "Ce", "Y",     # Rare earth
    "Cd", "Co", "Ni", "Mn", "Fe", "Cr", "V"  # Other transition metals
]

# Fields to retrieve
FIELDS = [
    "material_id",
    "formula_pretty",
    "composition",
    "composition_reduced",
    "elements",
    "nelements",
    "energy_above_hull",
    "formation_energy_per_atom",
    "band_gap",
    "nsites",
    "volume",
    "density",
    "symmetry",
    "is_stable",
]


# =============================================================================
# Query helpers
# =============================================================================
def query_host_oxides(mpr, host_element):
    """Download every oxide containing the given host element (pure and doped)."""
    print(f"\nQuerying oxides containing {host_element}...")

    docs = mpr.materials.summary.search(
        elements=[host_element, "O"],  # must contain the host and oxygen
        fields=FIELDS,
    )

    print(f"  Found {len(docs)} entries for {host_element}-O systems")
    return docs


def doc_to_dict(doc):
    """Flatten a SummaryDoc into a plain dictionary for DataFrame creation."""
    return {
        "material_id": str(doc.material_id),
        "formula_pretty": doc.formula_pretty,
        "composition": str(doc.composition),
        "elements": ", ".join(sorted(str(e) for e in doc.elements)),
        "nelements": doc.nelements,
        "energy_above_hull": doc.energy_above_hull,
        "formation_energy_per_atom": doc.formation_energy_per_atom,
        "band_gap": doc.band_gap,
        "nsites": doc.nsites,
        "volume": doc.volume,
        "density": doc.density,
        "crystal_system": doc.symmetry.crystal_system.value if doc.symmetry else None,
        "space_group_symbol": doc.symmetry.symbol if doc.symmetry else None,
        "space_group_number": doc.symmetry.number if doc.symmetry else None,
        "is_stable": doc.is_stable,
    }


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    all_records = []

    with MPRester(get_api_key()) as mpr:
        for host in HOST_ELEMENTS:
            docs = query_host_oxides(mpr, host)
            for doc in docs:
                record = doc_to_dict(doc)
                record["host_element"] = host  # remember which query found it
                all_records.append(record)

    df = pd.DataFrame(all_records)
    print(f"\nTotal records before deduplication: {len(df)}")

    # Same material found through different host queries -> keep the first hit
    df = df.drop_duplicates(subset="material_id", keep="first")
    print(f"Total records after deduplication: {len(df)}")

    print(f"\nNumber of unique formulas: {df['formula_pretty'].nunique()}")
    print(f"Number of elements range: {df['nelements'].min()} to {df['nelements'].max()}")
    print(f"\nEnergy above hull statistics:")
    print(df["energy_above_hull"].describe())

    # Save the raw data untouched (data-version-control best practice)
    os.makedirs("data/raw", exist_ok=True)
    raw_path = "data/raw/mp_doped_oxides_raw.csv"
    df.to_csv(raw_path, index=False)
    print(f"\nRaw data saved to: {raw_path}")
    print(f"Shape: {df.shape}")
