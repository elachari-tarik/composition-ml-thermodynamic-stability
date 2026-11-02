"""Step 02 - Test the connection to the Materials Project API.

Fetches the ZnO entries and prints the first few, as a quick check that the API key
(MP_API_KEY environment variable) works.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tco_stability.paths import ensure_project_root
ensure_project_root()

from mp_api.client import MPRester
from tco_stability.config import get_api_key

with MPRester(get_api_key()) as mpr:
    # One well-known material: ZnO (zinc oxide)
    docs = mpr.materials.summary.search(
        formula="ZnO",
        fields=["material_id", "formula_pretty", "energy_above_hull",
                "formation_energy_per_atom", "band_gap"]
    )

    print(f"Found {len(docs)} ZnO entries\n")
    for doc in docs[:5]:  # first five are enough for a sanity check
        print(f"  ID: {doc.material_id}")
        print(f"  Formula: {doc.formula_pretty}")
        print(f"  Energy above hull: {doc.energy_above_hull:.4f} eV/atom")
        print(f"  Formation energy: {doc.formation_energy_per_atom:.4f} eV/atom")
        print(f"  Band gap: {doc.band_gap:.2f} eV")
        print()
