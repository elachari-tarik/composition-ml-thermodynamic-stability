"""Step 05 - Audit the cleaned dataset: strict dilute-doped hosts vs. multicomponent oxides.

Classifies the cleaned Materials Project oxides into three groups:
  1. strict dilute-doped binary host oxides (ZnO, In2O3, SnO2),
  2. general ternary, quaternary, and higher-order multicomponent oxides,
  3. borderline or edge-case compositions (exported to ambiguous_cases.csv).

Read-only on the source CSV and deterministic (seed = 42).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tco_stability.paths import ensure_project_root
ensure_project_root()

import re
import numpy as np
import pandas as pd
from tco_stability.features import UNSUPPORTED_ELEMENTS

RNG_SEED = 42
np.random.seed(RNG_SEED)

# Nominal anion-to-cation ratios of the binary hosts: ZnO (1:1), In2O3 (3:2), SnO2 (2:1)
HOST_IDEAL_RATIO = {
    "Zn": 1.0,
    "In": 1.5,
    "Sn": 2.0,
}

# Unsupported radioactive elements removed before the 5,207 featurization step
ANION_SPECIES = {"O", "F", "Cl", "Br", "I", "S", "Se", "Te", "N", "P"}


def parse_composition(comp_str):
    """Parse a composition string (e.g. 'Zn40 O40') into an {element: amount} dict."""
    pairs = re.findall(r"([A-Z][a-z]?)(\d*\.?\d+)", str(comp_str))
    return {elem: float(count) for elem, count in pairs if float(count) > 0}


def classify_entry(comp_dict, host_threshold=0.90, stoich_tol=0.10):
    """Classify a composition into 'dilute_doped_host', 'ternary_quaternary_general', or 'ambiguous'."""
    elements = list(comp_dict.keys())
    n_elem = len(elements)

    anions = {k: v for k, v in comp_dict.items() if k in ANION_SPECIES}
    cations = {k: v for k, v in comp_dict.items() if k not in ANION_SPECIES}

    total_anions = sum(anions.values())
    total_cations = sum(cations.values())

    if total_cations <= 0 or total_anions <= 0:
        return "ambiguous", "invalid_ion_partition", {
            "dom_cation": None,
            "dom_cation_frac": np.nan,
            "anion_cation_ratio": np.nan,
        }

    dom_cation = max(cations.items(), key=lambda kv: kv[1])[0]
    dom_cation_amount = cations[dom_cation]
    dom_cation_frac = dom_cation_amount / total_cations
    anion_cation_ratio = total_anions / total_cations

    details = {
        "dom_cation": dom_cation,
        "dom_cation_frac": round(dom_cation_frac, 6),
        "anion_cation_ratio": round(anion_cation_ratio, 6),
    }

    # Case 1: binary reference oxide (nelements == 2)
    if n_elem < 3:
        return "ambiguous", "binary_undoped_edge_case", details

    # Case 2: dominant cation is Zn, In, or Sn
    if dom_cation in HOST_IDEAL_RATIO:
        ideal_ratio = HOST_IDEAL_RATIO[dom_cation]
        rel_ratio_diff = abs(anion_cation_ratio - ideal_ratio) / ideal_ratio

        if len(cations) == 1 and len(anions) >= 2 and "O" in anions:
            o_frac_in_anions = anions["O"] / total_anions
            if o_frac_in_anions > host_threshold and rel_ratio_diff <= stoich_tol:
                return "dilute_doped_host", f"{dom_cation}_anionic_doped", details
            return "ambiguous", "anionic_doped_high_conc_or_stoich_mismatch", details

        if dom_cation_frac > host_threshold:
            if rel_ratio_diff <= stoich_tol:
                return "dilute_doped_host", f"{dom_cation}_cationic_doped", details
            return (
                "ambiguous",
                "dominant_host_gt_90pct_but_stoich_mismatch",
                details,
            )

        if 0.80 < dom_cation_frac <= host_threshold and rel_ratio_diff <= stoich_tol:
            return (
                "ambiguous",
                "borderline_host_fraction_80_to_90pct",
                details,
            )

        return "ternary_quaternary_general", "multicomponent_oxide", details

    # Case 3: dominant cation is another metal (> 90%) with Zn/In/Sn as minority species
    if dom_cation_frac > host_threshold:
        return (
            "ambiguous",
            "dilute_oxide_with_other_dominant_cation",
            details,
        )

    return "ternary_quaternary_general", "multicomponent_oxide", details


def count_host_structures(
    csv_path="data/processed/mp_doped_oxides_cleaned.csv",
    ambiguous_out_path="data/processed/ambiguous_cases.csv",
    exclude_unsupported=True,
    host_threshold=0.90,
    stoich_tol=0.10,
):
    """Read the cleaned dataset, classify every composition, and export borderline cases."""
    csv_file = Path(csv_path)
    if not csv_file.exists():
        raise FileNotFoundError(f"Input CSV not found: {csv_file.resolve()}")

    df = pd.read_csv(csv_file)
    n_loaded = len(df)

    if exclude_unsupported and "elements" in df.columns:
        mask_valid = df["elements"].apply(
            lambda x: not any(
                e.strip() in UNSUPPORTED_ELEMENTS for e in str(x).split(",")
            )
        )
        df = df[mask_valid].reset_index(drop=True)

    records_ambiguous = []
    n_dilute = 0
    n_general = 0
    n_ambiguous = 0

    n_naive_any_metal_gt90 = 0
    n_naive_zn_in_sn_gt90 = 0
    host_breakdown = {"Zn": 0, "In": 0, "Sn": 0}

    for idx, row in df.iterrows():
        comp_str = row.get("composition", "")
        comp_dict = parse_composition(comp_str)

        metals_only = {k: v for k, v in comp_dict.items() if k != "O"}
        if metals_only:
            tot_m = sum(metals_only.values())
            dom_m, max_m = max(metals_only.items(), key=lambda kv: kv[1])
            if len(comp_dict) >= 3 and (max_m / tot_m) > host_threshold:
                n_naive_any_metal_gt90 += 1
                if dom_m in HOST_IDEAL_RATIO:
                    n_naive_zn_in_sn_gt90 += 1

        category, reason, details = classify_entry(
            comp_dict, host_threshold=host_threshold, stoich_tol=stoich_tol
        )

        if category == "dilute_doped_host":
            n_dilute += 1
            host_breakdown[details["dom_cation"]] += 1
        elif category == "ternary_quaternary_general":
            n_general += 1
        else:
            n_ambiguous += 1
            records_ambiguous.append(
                {
                    "row_index": idx,
                    "material_id": row.get("material_id", ""),
                    "formula_pretty": row.get("formula_pretty", ""),
                    "composition": comp_str,
                    "nelements": row.get("nelements", len(comp_dict)),
                    "host_element_col": row.get("host_element", ""),
                    "dominant_cation": details["dom_cation"],
                    "dominant_cation_fraction": details["dom_cation_frac"],
                    "anion_to_cation_ratio": details["anion_cation_ratio"],
                    "ambiguity_reason": reason,
                }
            )

    n_total = len(df)
    assert n_dilute + n_general + n_ambiguous == n_total, "Partition mismatch!"

    out_file = Path(ambiguous_out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    df_amb = pd.DataFrame(records_ambiguous)
    df_amb.to_csv(out_file, index=False)

    print("=" * 72)
    print("HOST vs MULTICOMPONENT OXIDE AUDIT SUMMARY (Section 3.1)")
    print("=" * 72)
    print(f"Source file (read-only)         : {csv_file}")
    print(f"Rows in cleaned CSV             : {n_loaded}")
    print(f"Rows analyzed (featurized set)  : {n_total} (exclude_unsupported={exclude_unsupported})")
    print("-" * 72)
    print(f"Strict dilute-doped hosts       : {n_dilute} ({100.0 * n_dilute / n_total:.2f}%)")
    print(f"  - ZnO-type dilute doped       : {host_breakdown['Zn']}")
    print(f"  - In2O3-type dilute doped     : {host_breakdown['In']}")
    print(f"  - SnO2-type dilute doped      : {host_breakdown['Sn']}")
    print(f"Multicomponent oxide phases     : {n_general} ({100.0 * n_general / n_total:.2f}%)")
    print(f"Borderline / binary edge cases  : {n_ambiguous} ({100.0 * n_ambiguous / n_total:.2f}%)")
    print(f"Total featurized compositions   : {n_total}")
    print("-" * 72)
    print("Comparison with relaxed definitions (nelements >= 3, no stoich check):")
    print(f"  - Zn/In/Sn > 90% of non-O     : {n_naive_zn_in_sn_gt90} ({100.0 * n_naive_zn_in_sn_gt90 / n_total:.2f}%)")
    print(f"  - Any non-O > 90% of non-O    : {n_naive_any_metal_gt90} ({100.0 * n_naive_any_metal_gt90 / n_total:.2f}%)")
    print("-" * 72)
    print(f"Borderline cases exported to    : {out_file}")
    print("=" * 72)

    return n_dilute


if __name__ == "__main__":
    count_host_structures()