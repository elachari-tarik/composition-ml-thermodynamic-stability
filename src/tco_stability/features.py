"""Composition-based feature vectors (Magpie-style) used by every script in the project.

Two featurizers live here, and both reproduce the numbers used in the paper:

* ``featurize_composition``           -> 105 features (15 properties x 7 statistics)
* ``featurize_composition_enhanced``  -> 118 features (the 105 above + 7 stoichiometric
                                         + 4 electronic-structure + 2 ionic attributes,
                                         following Ward et al., npj Comput. Mater. 2016)

The elemental property table below is the single source of truth. It used to be
copy-pasted into several scripts; keeping one copy guarantees that every
experiment sees exactly the same numbers.
"""

import re

import numpy as np

# Columns of each ELEM_PROPS row:
#   [atomic_number, atomic_mass, row, group, electronegativity (Pauling),
#    atomic_radius (pm), ionization_energy (eV), electron_affinity (eV),
#    n_valence, n_s, n_p, n_d, n_f, melting_point (K), density (g/cm3)]
ELEM_PROPS = {
    "H":  [1, 1.008, 1, 1, 2.20, 53, 13.598, 0.754, 1, 1, 0, 0, 0, 14.01, 0.00009],
    "Li": [3, 6.941, 2, 1, 0.98, 167, 5.392, 0.618, 1, 1, 0, 0, 0, 453.69, 0.534],
    "Be": [4, 9.012, 2, 2, 1.57, 112, 9.323, 0.0, 2, 2, 0, 0, 0, 1560.0, 1.85],
    "B":  [5, 10.81, 2, 13, 2.04, 87, 8.298, 0.277, 3, 2, 1, 0, 0, 2349.0, 2.34],
    "C":  [6, 12.01, 2, 14, 2.55, 67, 11.260, 1.263, 4, 2, 2, 0, 0, 3823.0, 2.267],
    "N":  [7, 14.01, 2, 15, 3.04, 56, 14.534, -0.07, 5, 2, 3, 0, 0, 63.15, 0.0013],
    "O":  [8, 16.00, 2, 16, 3.44, 48, 13.618, 1.461, 6, 2, 4, 0, 0, 54.36, 0.00143],
    "F":  [9, 19.00, 2, 17, 3.98, 42, 17.423, 3.401, 7, 2, 5, 0, 0, 53.53, 0.0017],
    "Na": [11, 22.99, 3, 1, 0.93, 190, 5.139, 0.548, 1, 1, 0, 0, 0, 370.87, 0.971],
    "Mg": [12, 24.31, 3, 2, 1.31, 145, 7.646, 0.0, 2, 2, 0, 0, 0, 923.0, 1.738],
    "Al": [13, 26.98, 3, 13, 1.61, 118, 5.986, 0.441, 3, 2, 1, 0, 0, 933.47, 2.70],
    "Si": [14, 28.09, 3, 14, 1.90, 111, 8.152, 1.385, 4, 2, 2, 0, 0, 1687.0, 2.33],
    "P":  [15, 30.97, 3, 15, 2.19, 98, 10.487, 0.746, 5, 2, 3, 0, 0, 317.30, 1.82],
    "S":  [16, 32.06, 3, 16, 2.58, 88, 10.360, 2.077, 6, 2, 4, 0, 0, 388.36, 2.067],
    "Cl": [17, 35.45, 3, 17, 3.16, 79, 12.968, 3.613, 7, 2, 5, 0, 0, 171.6, 0.0032],
    "K":  [19, 39.10, 4, 1, 0.82, 243, 4.341, 0.501, 1, 1, 0, 0, 0, 336.53, 0.862],
    "Ca": [20, 40.08, 4, 2, 1.00, 194, 6.113, 0.025, 2, 2, 0, 0, 0, 1115.0, 1.54],
    "Sc": [21, 44.96, 4, 3, 1.36, 184, 6.562, 0.188, 3, 2, 0, 1, 0, 1814.0, 2.989],
    "Ti": [22, 47.87, 4, 4, 1.54, 176, 6.828, 0.079, 4, 2, 0, 2, 0, 1941.0, 4.54],
    "V":  [23, 50.94, 4, 5, 1.63, 171, 6.746, 0.525, 5, 2, 0, 3, 0, 2183.0, 6.11],
    "Cr": [24, 52.00, 4, 6, 1.66, 166, 6.767, 0.666, 6, 1, 0, 5, 0, 2180.0, 7.15],
    "Mn": [25, 54.94, 4, 7, 1.55, 161, 7.434, 0.0, 7, 2, 0, 5, 0, 1519.0, 7.44],
    "Fe": [26, 55.85, 4, 8, 1.83, 156, 7.902, 0.151, 8, 2, 0, 6, 0, 1811.0, 7.874],
    "Co": [27, 58.93, 4, 9, 1.88, 152, 7.881, 0.662, 9, 2, 0, 7, 0, 1768.0, 8.90],
    "Ni": [28, 58.69, 4, 10, 1.91, 149, 7.640, 1.156, 10, 2, 0, 8, 0, 1728.0, 8.908],
    "Cu": [29, 63.55, 4, 11, 1.90, 145, 7.726, 1.235, 11, 1, 0, 10, 0, 1357.77, 8.96],
    "Zn": [30, 65.38, 4, 12, 1.65, 142, 9.394, 0.0, 12, 2, 0, 10, 0, 692.68, 7.134],
    "Ga": [31, 69.72, 4, 13, 1.81, 136, 5.999, 0.43, 3, 2, 1, 10, 0, 302.91, 5.907],
    "Ge": [32, 72.63, 4, 14, 2.01, 125, 7.900, 1.233, 4, 2, 2, 10, 0, 1211.40, 5.323],
    "As": [33, 74.92, 4, 15, 2.18, 114, 9.789, 0.804, 5, 2, 3, 10, 0, 1090.0, 5.776],
    "Se": [34, 78.96, 4, 16, 2.55, 103, 9.752, 2.021, 6, 2, 4, 10, 0, 494.0, 4.809],
    "Br": [35, 79.90, 4, 17, 2.96, 94, 11.814, 3.364, 7, 2, 5, 10, 0, 265.8, 3.122],
    "Rb": [37, 85.47, 5, 1, 0.82, 265, 4.177, 0.486, 1, 1, 0, 0, 0, 312.46, 1.532],
    "Sr": [38, 87.62, 5, 2, 0.95, 219, 5.695, 0.048, 2, 2, 0, 0, 0, 1050.0, 2.64],
    "Y":  [39, 88.91, 5, 3, 1.22, 212, 6.217, 0.307, 3, 2, 0, 1, 0, 1799.0, 4.469],
    "Zr": [40, 91.22, 5, 4, 1.33, 206, 6.634, 0.426, 4, 2, 0, 2, 0, 2128.0, 6.506],
    "Nb": [41, 92.91, 5, 5, 1.60, 198, 6.759, 0.893, 5, 1, 0, 4, 0, 2750.0, 8.57],
    "Mo": [42, 95.96, 5, 6, 2.16, 190, 7.092, 0.746, 6, 1, 0, 5, 0, 2896.0, 10.22],
    "Ru": [44, 101.07, 5, 8, 2.20, 178, 7.361, 1.05, 8, 1, 0, 7, 0, 2607.0, 12.37],
    "Rh": [45, 102.91, 5, 9, 2.28, 173, 7.459, 1.137, 9, 1, 0, 8, 0, 2237.0, 12.41],
    "Pd": [46, 106.42, 5, 10, 2.20, 169, 8.337, 0.557, 10, 0, 0, 10, 0, 1828.05, 12.02],
    "Ag": [47, 107.87, 5, 11, 1.93, 165, 7.576, 1.302, 11, 1, 0, 10, 0, 1234.93, 10.501],
    "Cd": [48, 112.41, 5, 12, 1.69, 161, 8.994, 0.0, 12, 2, 0, 10, 0, 594.22, 8.65],
    "In": [49, 114.82, 5, 13, 1.78, 156, 5.786, 0.404, 3, 2, 1, 10, 0, 429.75, 7.31],
    "Sn": [50, 118.71, 5, 14, 1.96, 145, 7.344, 1.112, 4, 2, 2, 10, 0, 505.08, 7.287],
    "Sb": [51, 121.76, 5, 15, 2.05, 133, 8.609, 1.047, 5, 2, 3, 10, 0, 903.78, 6.685],
    "Te": [52, 127.60, 5, 16, 2.10, 123, 9.010, 1.971, 6, 2, 4, 10, 0, 722.66, 6.232],
    "I":  [53, 126.90, 5, 17, 2.66, 115, 10.451, 3.059, 7, 2, 5, 10, 0, 386.85, 4.93],
    "Cs": [55, 132.91, 6, 1, 0.79, 298, 3.894, 0.472, 1, 1, 0, 0, 0, 301.59, 1.873],
    "Ba": [56, 137.33, 6, 2, 0.89, 253, 5.212, 0.145, 2, 2, 0, 0, 0, 1000.0, 3.594],
    "La": [57, 138.91, 6, 3, 1.10, 195, 5.577, 0.47, 3, 2, 0, 1, 0, 1193.0, 6.145],
    "Ce": [58, 140.12, 6, 3, 1.12, 185, 5.539, 0.55, 4, 2, 0, 1, 1, 1068.0, 6.77],
    "Pr": [59, 140.91, 6, 3, 1.13, 247, 5.473, 0.5, 5, 2, 0, 0, 3, 1208.0, 6.773],
    "Nd": [60, 144.24, 6, 3, 1.14, 206, 5.525, 0.5, 6, 2, 0, 0, 4, 1297.0, 7.008],
    "Sm": [62, 150.36, 6, 3, 1.17, 238, 5.644, 0.5, 8, 2, 0, 0, 6, 1345.0, 7.52],
    "Eu": [63, 151.96, 6, 3, 1.20, 231, 5.670, 0.5, 9, 2, 0, 0, 7, 1099.0, 5.244],
    "Gd": [64, 157.25, 6, 3, 1.20, 233, 6.150, 0.5, 10, 2, 0, 1, 7, 1585.0, 7.90],
    "Tb": [65, 158.93, 6, 3, 1.20, 225, 5.864, 0.5, 11, 2, 0, 0, 9, 1629.0, 8.23],
    "Dy": [66, 162.50, 6, 3, 1.22, 228, 5.939, 0.5, 12, 2, 0, 0, 10, 1680.0, 8.55],
    "Ho": [67, 164.93, 6, 3, 1.23, 226, 6.022, 0.5, 13, 2, 0, 0, 11, 1734.0, 8.795],
    "Er": [68, 167.26, 6, 3, 1.24, 226, 6.108, 0.5, 14, 2, 0, 0, 12, 1802.0, 9.066],
    "Tm": [69, 168.93, 6, 3, 1.25, 222, 6.184, 0.5, 15, 2, 0, 0, 13, 1818.0, 9.321],
    "Yb": [70, 173.05, 6, 3, 1.10, 222, 6.254, 0.5, 16, 2, 0, 0, 14, 1097.0, 6.90],
    "Lu": [71, 174.97, 6, 3, 1.27, 217, 5.426, 0.5, 17, 2, 0, 1, 14, 1925.0, 9.84],
    "Hf": [72, 178.49, 6, 4, 1.30, 208, 6.825, 0.0, 4, 2, 0, 2, 14, 2506.0, 13.31],
    "Ta": [73, 180.95, 6, 5, 1.50, 200, 7.550, 0.322, 5, 2, 0, 3, 14, 3290.0, 16.654],
    "W":  [74, 183.84, 6, 6, 2.36, 193, 7.864, 0.816, 6, 2, 0, 4, 14, 3695.0, 19.25],
    "Re": [75, 186.21, 6, 7, 1.90, 188, 7.834, 0.15, 7, 2, 0, 5, 14, 3459.0, 21.02],
    "Os": [76, 190.23, 6, 8, 2.20, 185, 8.438, 1.1, 8, 2, 0, 6, 14, 3306.0, 22.587],
    "Ir": [77, 192.22, 6, 9, 2.20, 180, 8.967, 1.565, 9, 2, 0, 7, 14, 2719.0, 22.562],
    "Pt": [78, 195.08, 6, 10, 2.28, 177, 8.959, 2.128, 10, 1, 0, 9, 14, 2041.4, 21.46],
    "Au": [79, 196.97, 6, 11, 2.54, 174, 9.226, 2.309, 11, 1, 0, 10, 14, 1337.33, 19.282],
    "Hg": [80, 200.59, 6, 12, 2.00, 171, 10.437, 0.0, 12, 2, 0, 10, 14, 234.43, 13.534],
    "Tl": [81, 204.38, 6, 13, 1.62, 156, 6.108, 0.377, 3, 2, 1, 10, 14, 577.0, 11.85],
    "Pb": [82, 207.2, 6, 14, 2.33, 154, 7.417, 0.364, 4, 2, 2, 10, 14, 600.61, 11.342],
    "Bi": [83, 208.98, 6, 15, 2.02, 143, 7.286, 0.946, 5, 2, 3, 10, 14, 544.7, 9.807],
    "Th": [90, 232.04, 7, 3, 1.30, 179, 6.307, 0.0, 4, 2, 0, 2, 0, 2115.0, 11.72],
    "U":  [92, 238.03, 7, 3, 1.38, 175, 6.194, 0.0, 6, 2, 0, 1, 3, 1405.3, 18.95],
}

PROP_NAMES = [
    "atomic_number", "atomic_mass", "row", "group", "electronegativity",
    "atomic_radius", "ionization_energy", "electron_affinity",
    "n_valence", "n_s", "n_p", "n_d", "n_f", "melting_point", "density_elem",
]

STAT_NAMES = ["mean", "std", "min", "max", "range", "wmean", "wstd"]

# Column positions inside each ELEM_PROPS row
IDX_EN = PROP_NAMES.index("electronegativity")
IDX_VALENCE = PROP_NAMES.index("n_valence")
IDX_N_S = PROP_NAMES.index("n_s")
IDX_N_P = PROP_NAMES.index("n_p")
IDX_N_D = PROP_NAMES.index("n_d")
IDX_N_F = PROP_NAMES.index("n_f")

# Elements missing from ELEM_PROPS; compounds containing them are dropped
UNSUPPORTED_ELEMENTS = {"Pm", "Pu", "Tc", "Np", "Pa"}


def parse_composition(comp_str):
    """Turn a composition string such as 'Ca12 Zn12 Te8 O48' into {'Ca': 12.0, ...}."""
    pairs = re.findall(r"([A-Z][a-z]?)(\d+\.?\d*)", str(comp_str))
    return {elem: float(count) for elem, count in pairs}


def featurize_composition(comp_dict):
    """105-feature vector: 7 statistics for each of the 15 elemental properties.

    Statistics: mean, std, min, max, range, and the stoichiometry-weighted mean/std.
    """
    elements = list(comp_dict.keys())
    amounts = np.array([comp_dict[e] for e in elements])
    fractions = amounts / amounts.sum()

    features = {}
    for p_idx, prop_name in enumerate(PROP_NAMES):
        values = np.array([ELEM_PROPS[e][p_idx] for e in elements])

        features[f"{prop_name}_mean"] = np.mean(values)
        features[f"{prop_name}_std"] = np.std(values) if len(values) > 1 else 0.0
        features[f"{prop_name}_min"] = np.min(values)
        features[f"{prop_name}_max"] = np.max(values)
        features[f"{prop_name}_range"] = np.max(values) - np.min(values)

        wmean = np.average(values, weights=fractions)
        features[f"{prop_name}_wmean"] = wmean
        wvar = np.average((values - wmean) ** 2, weights=fractions)
        features[f"{prop_name}_wstd"] = np.sqrt(wvar)

    return features


def featurize_composition_enhanced(comp_dict):
    """118-feature vector: the 105 elemental statistics plus three extra families.

    * Stoichiometric (7): number of elements, Lp norms (p = 2, 3, 5, 7, 10) of the
      fraction vector, and the largest fraction.
    * Electronic structure (4): composition-weighted fraction of s, p, d, f valence electrons.
    * Ionic (2): largest electronegativity difference and the Pauling ionic character.
    """
    elements = list(comp_dict.keys())
    amounts = np.array([comp_dict[e] for e in elements])
    fractions = amounts / amounts.sum()
    n_elem = len(elements)

    features = {}

    # Family 1: elemental property statistics (105 features)
    for p_idx, prop_name in enumerate(PROP_NAMES):
        values = np.array([ELEM_PROPS[e][p_idx] for e in elements])
        features[f"{prop_name}_mean"] = np.mean(values)
        features[f"{prop_name}_std"] = np.std(values) if n_elem > 1 else 0.0
        features[f"{prop_name}_min"] = np.min(values)
        features[f"{prop_name}_max"] = np.max(values)
        features[f"{prop_name}_range"] = np.max(values) - np.min(values)
        wmean = np.average(values, weights=fractions)
        features[f"{prop_name}_wmean"] = wmean
        wvar = np.average((values - wmean) ** 2, weights=fractions)
        features[f"{prop_name}_wstd"] = np.sqrt(wvar)

    # Family 2: stoichiometric attributes (7 features)
    features["stoich_n_elements"] = float(n_elem)
    for p in [2, 3, 5, 7, 10]:
        features[f"stoich_Lp_norm_{p}"] = np.sum(fractions ** p) ** (1.0 / p)
    features["stoich_fraction_max"] = np.max(fractions)

    # Family 3: electronic-structure fractions (4 features)
    total_valence = np.array([ELEM_PROPS[e][IDX_VALENCE] for e in elements])
    safe_total = np.where(total_valence > 0, total_valence, 1.0)
    for orbital, idx in [("s", IDX_N_S), ("p", IDX_N_P), ("d", IDX_N_D), ("f", IDX_N_F)]:
        orbital_counts = np.array([ELEM_PROPS[e][idx] for e in elements])
        features[f"elec_frac_{orbital}"] = np.average(
            orbital_counts / safe_total, weights=fractions
        )

    # Family 4: ionic-compound attributes (2 features)
    en_values = np.array([ELEM_PROPS[e][IDX_EN] for e in elements])
    if n_elem > 1:
        max_en_diff = max(
            abs(en_values[i] - en_values[j])
            for i in range(n_elem)
            for j in range(i + 1, n_elem)
        )
        features["ionic_max_en_diff"] = max_en_diff
        features["ionic_character"] = 1.0 - np.exp(-0.25 * max_en_diff ** 2)
    else:
        features["ionic_max_en_diff"] = 0.0
        features["ionic_character"] = 0.0

    return features
