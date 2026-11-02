# Computational Framework and Methodology

This document summarizes the data processing, feature engineering, and evaluation rules implemented in `src/tco_stability/` and `scripts/01` through `17`.

---

## 1. Data Acquisition and Four-Step Cleaning

### Materials Project Query (`scripts/03_query_materials.py`)
We queried the Materials Project database via `pymatgen` for all inorganic oxides containing oxygen (`O`), at least one of the three host cations (`Zn`, `In`, or `Sn`), and at least three distinct elements in total.
- **Initial records across the three host queries:** 7,376 entries (Zn: 3,635; In: 1,156; Sn: 2,585).
- **Unique material IDs after deduplication:** 7,116 entries.

### Cleaning Protocol (`scripts/04_clean_data.py`)
1. **Non-negative formation energy filter ($E_f < 0$):** Removed 58 entries with $E_f \ge 0$, which are unstable relative to their elemental standard states. In `scripts/06_sensitivity_ef.py`, we verified on the same 521 test formulas that training with ($n = 5,207$) and without ($n = 5,223$) this filter gives test MAEs of `0.0644` and `0.0654 eV/atom` (difference of `0.0010 eV/atom`, and `0.0022 eV/atom` under a full re-split).
2. **Extreme outlier removal:** Removed 2 entries with $E_{\text{hull}} > 2.0\text{ eV/atom}$.
3. **Polymorph deduplication:** For chemical formulas with multiple entries (906 formulas, with $\text{SnO}_2$ having up to 30 polymorphs), we kept only the lowest-energy polymorph (lowest $E_{\text{hull}}$) per unique `formula_pretty`. This leaves **5,216 cleaned oxide formulas**.
4. **Elemental property coverage:** Before featurization, 9 entries containing radioactive elements without complete property data (`Pm`, `Pu`, `Tc`, `Np`, `Pa`) were excluded, giving **5,207 featurized compositions** (Zn: 2,605; Sn: 1,656; In: 946).

### Dilute Host-Dopant vs. Multicomponent Phases (`scripts/05_count_host_structures.py`)
Out of the 5,207 featurized compositions, only **1 entry (0.02%)** strictly corresponds to a dilute-doped binary host lattice ($\text{ZnO}$, $\text{In}_2\text{O}_3$, or $\text{SnO}_2$ retaining $>90\%$ of the host cation with an anion-to-cation ratio within $\pm 10\%$ of the parent binary oxide), and **19 entries (0.36%)** contain any single metal at $>90\%$ of the non-oxygen content. By contrast, **4,987 entries (95.8%)** are stoichiometric ternary, quaternary, or higher-order multicomponent oxides.

---

## 2. Extended Magpie Featurization (`src/tco_stability/features.py`)

Each chemical formula is converted into a 118-dimensional vector without external featurization dependencies (`matminer` is not required; the 78-element property table is embedded directly in `features.py`):

1. **Elemental Property Statistics (105 features):**  
   For 15 elemental properties (atomic number, atomic mass, row, group, Pauling electronegativity, atomic radius, first ionization energy, electron affinity, valence electrons, $s/p/d/f$ valence electrons, melting point, and density), we compute 7 statistics: unweighted mean, standard deviation, minimum, maximum, and range, plus the composition-weighted mean (`wmean`) and composition-weighted standard deviation (`wstd`).
2. **Stoichiometric Attributes (7 features):**  
   Number of distinct elements, maximum elemental fraction, and the $L_p$ norms $\left(\sum_i f_i^p\right)^{1/p}$ for $p \in \{2, 3, 5, 7, 10\}$.
3. **Electronic Structure Fractions (4 features):**  
   Composition-weighted average fraction of valence electrons in $s$, $p$, $d$, and $f$ orbitals.
4. **Ionic Compound Attributes (2 features):**  
   Maximum pairwise electronegativity difference ($\Delta\text{EN}$) and Pauling ionic character:
   $$\text{ionic character} = 1 - \exp\left(-0.25\,\Delta\text{EN}^2\right)$$

---

## 3. Data Splitting and No-Leakage Protocol

- **Composition-based split (`seed = 42`):** Unique `formula_pretty` strings are shuffled and divided into **70% training (3,645 samples)**, **20% validation (1,041 samples)**, and **10% test (521 samples)**.
- **Feature standardization:** `StandardScaler` is fit strictly on the 3,645 training samples and applied to transform the validation and test sets.
- **Model selection and tuning (`scripts/11_phase3_enhanced_118.py`):**
  - Hyperparameter tuning (108-combination grid search for Random Forest) and model comparison are performed on the validation set (`1,041` samples).
  - 5-fold cross-validation is run on the training set (`3,645` samples) to compare RF (`MAE = 0.0665 ± 0.0019 eV/atom`) and GBR (`MAE = 0.0670 ± 0.0022 eV/atom`).
- **Validation-set Permutation Feature Importance (`scripts/13_permutation_importance.py`):**  
  Computed with 30 random repeats exclusively on the **validation set (`1,041` samples)** using the tuned RF trained on the `3,645` training samples, keeping the final test set untouched.
- **Single Final Test Evaluation (`scripts/11_phase3_enhanced_118.py`):**  
  After fixing all hyperparameters, the models are retrained on the combined training + validation set (`4,686` samples) and evaluated **once** on the held-out test set (`521` samples).