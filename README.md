# Capabilities and Limitations of Composition-Only Machine Learning for Predicting the Thermodynamic Stability of Multicomponent Zn-, In-, and Sn-Based Oxides

**Authors:** T. El-Achari, L. B. Drissi, I. Mallek-Zouari, F. Goumrhar, W. Azouzi  
**Persistent Deposit (Zenodo):** [https://doi.org/10.5281/zenodo.19430860](https://doi.org/10.5281/zenodo.19430860)

This repository contains the data, custom Python scripts, trained model weights, and publication figures for our study on predicting the DFT-computed energy above the convex hull ($E_{\text{hull}}$) of multicomponent Zn-, In-, and Sn-based oxides from chemical composition alone.

---

## Repository Structure

```text
tco_stability_repo/
├── run_all.py                        # Master script running the pipeline in manuscript order
├── environment.yml                   # Conda environment specification (tco_ml)
├── requirements.txt                  # Pinned Python package versions
├── .env.example                      # Template for the Materials Project API key
├── src/
│   └── tco_stability/                # Shared library (single source of truth)
│       ├── __init__.py
│       ├── config.py                 # Reads MP_API_KEY from environment variables
│       ├── paths.py                  # Project root and directory path resolution
│       └── features.py               # 78-element property table + 105 & 118 Magpie featurizers
├── scripts/                          # Sequential analysis steps (01 to 17)
│   ├── 01_check_env.py               # Verify Python packages and environment
│   ├── 02_test_api.py                # Test connection to the Materials Project API
│   ├── 03_query_materials.py         # Query raw Zn-, In-, and Sn-based oxides (7,376 records)
│   ├── 04_clean_data.py              # 4-step cleaning and polymorph deduplication (5,216 entries)
│   ├── 05_count_host_structures.py   # Count strict dilute-doped hosts vs multicomponent phases
│   ├── 06_sensitivity_ef.py          # Sensitivity check with and without positive-Ef filtering
│   ├── 07_plot_data_exploration.py   # Exploratory dataset distributions
│   ├── 08_plot_ehull_by_host.py      # Ehull statistics by primary host element (Zn, In, Sn)
│   ├── 09_phase2_baseline.py         # Baseline DummyRegressor and initial 105-feature RF
│   ├── 10_phase3_tuning_105.py       # Hyperparameter grid search on the 105-feature set
│   ├── 11_phase3_enhanced_118.py     # Full 118-feature training, 5-fold CV, RF / GBR / XGBoost
│   ├── 12_pub_figures.py             # Generate main manuscript Figures 1 to 6
│   ├── 13_permutation_importance.py  # Validation-set Permutation Feature Importance (Figure 7)
│   ├── 14_leave_one_host_out.py      # Leave-One-Host-Out extrapolation experiment (Figures 8 & 9)
│   ├── 15_dopant_sensitivity.py      # L2 feature-shift analysis across doping levels (Figure 10)
│   ├── 16_synthetic_inference.py     # Synthetic dopant concentration inference test (Figure 11)
│   ├── 17_save_models.py             # Export trained scaler and model weights to models/
│   ├── verify_outputs.py             # Check regenerated CSV outputs against reference/
│   └── tools/
│       └── run_100x.py               # Repeated-split stability check
├── notebooks/
│   └── reproduce.ipynb               # Step-by-step interactive walkthrough of the pipeline
├── data/
│   ├── raw/                          # Raw Materials Project query (mp_doped_oxides_raw.csv)
│   ├── processed/                    # Cleaned dataset (mp_doped_oxides_cleaned.csv)
│   └── splits/                       # Composition-based splits (train.csv, val.csv, test.csv)
├── models/                           # Serialized scaler and trained RF, GBR, XGBoost models
├── results/                          # Output tables, grid search logs, and feature importances
├── figures/                          # Generated figures (300 DPI PNG and vector PDF)
├── reference/                        # Reference CSV outputs for automated verification
└── docs/
    ├── METHODS.md                    # Summary of the data cleaning, features, and evaluation protocol
    └── LITERATURE.md                 # Key methodological references
```

---

## Installation and Environment

We used Python 3.11 for all calculations. You can recreate the environment using either `conda` or `pip`:

```bash
# Option 1: Conda
conda env create -f environment.yml
conda activate tco_ml

# Option 2: Pip (exact package versions used for the paper)
pip install -r requirements.txt
```

---

## Data Access and API Key

- **Using the archived dataset (recommended for exact reproduction):**  
  The raw dataset (`data/raw/mp_doped_oxides_raw.csv`) and cleaned dataset (`data/processed/mp_doped_oxides_cleaned.csv`) are included in this repository. By default, `run_all.py` uses the archived raw file and skips live API queries (steps `02` and `03`) so that results can be reproduced without a Materials Project account.

- **Running a fresh query from the Materials Project:**  
  If you want to re-query the database from scratch, copy `.env.example` to `.env`, add your personal API key from [next-gen.materialsproject.org](https://next-gen.materialsproject.org/api), and pass `--refresh-raw`:

```bash
cp .env.example .env
export MP_API_KEY="your_32_character_api_key"
python run_all.py --refresh-raw
```

*(Note: Because the Materials Project database is updated periodically, a fresh query may return a slightly different number of entries than our archived snapshot).*

---

## Reproducing the Results and Figures

To run the entire pipeline from data cleaning to figure generation (with a fixed random seed of `42`):

```bash
python run_all.py
```

You can also run any individual script directly from the project root:

```bash
python scripts/11_phase3_enhanced_118.py
python scripts/13_permutation_importance.py
```

To verify that your local run matches the published tables within floating-point tolerance ($10^{-9}$):

```bash
python scripts/verify_outputs.py
```

### Summary of Key Numbers (Seed = 42)
- **Dataset:** 7,376 raw records $\rightarrow$ 7,116 unique material IDs $\rightarrow$ 5,216 cleaned unique formulas $\rightarrow$ **5,207 featurized compositions** (Zn: 2,605; Sn: 1,656; In: 946).
- **Composition-based split:** 3,645 training (70.0%), 1,041 validation (20.0%), and 521 test (10.0%) compositions, with zero shared chemical formulas across splits.
- **Final Held-Out Test Set Performance (521 compositions, evaluated once):**
  - **Mean-prediction baseline (DummyRegressor):** MAE = `0.1040 eV/atom` ($R^2 \approx 0$)
  - **Tuned Random Forest:** MAE = `0.0632 eV/atom`, RMSE = `0.1087 eV/atom`, $R^2 = 0.5596$
  - **Gradient Boosting (GBR):** MAE = `0.0632 eV/atom`, RMSE = `0.1029 eV/atom`, $R^2 = 0.6056$
  - **XGBoost:** MAE = `0.0713 eV/atom`, $R^2 = 0.5384$

---

## Citation

If you use this dataset, code, or workflow in your work, please cite the associated manuscript and Zenodo archive:

> T. El-Achari, L. B. Drissi, I. Mallek-Zouari, F. Goumrhar, and W. Azouzi, *"Capabilities and Limitations of Composition-Only Machine Learning for Predicting the Thermodynamic Stability of Multicomponent Zn-, In-, and Sn-Based Oxides"*, Zenodo (2026). DOI: [10.5281/zenodo.19430860](https://doi.org/10.5281/zenodo.19430860)