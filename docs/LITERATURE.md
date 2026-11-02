# Key Methodological References

This document lists the primary references that guide our feature engineering, data splitting, and evaluation design.

1. **Best Practices & Composition-Based Splitting**  
   Wang, A. Y.-T. et al., *"Machine Learning for Materials Scientists: An Introductory Guide toward Best Practices"*, Chem. Mater. **2020**, 32, 4954–4965.  
   - Basis for splitting by unique chemical formula (`formula_pretty`), fitting scalers on the training set only, comparing against a `DummyRegressor` baseline, and evaluating the held-out test set only once.

2. **Extended Magpie Composition-Based Feature Vector (CBFV)**  
   Ward, L. et al., *"A general-purpose machine learning framework for predicting properties of inorganic materials"*, npj Comput. Mater. **2016**, 2, 16028.  
   - Basis for our 118 composition-only features combining elemental property statistics, stoichiometric $L_p$ norms, valence orbital fractions, and Pauling ionic character.

3. **Dataset Size & Tree-Based Ensemble Selection**  
   Jain, A., *"Machine learning in materials research: Developments over the last decade and challenges for the future"*, Curr. Opin. Solid State Mater. Sci. **2024**, 33, 101189.  
   - Shows that for materials datasets below ~10,000 entries, tree-based ensemble models (Random Forest, Gradient Boosting, XGBoost) match or outperform deep learning architectures.

4. **Extrapolation Limits in Transparent Conducting Materials**  
   Ottomano, F. et al., *"Assessing data-driven predictions of band gap and electrical conductivity for transparent conducting materials"*, Digital Discovery **2025**, 4, 1794–1811.  
   Meredig, B. et al., *"Can machine learning identify the next high-temperature superconductor? Examining extrapolation performance for materials discovery"*, Mol. Syst. Des. Eng. **2018**, 3, 819–825.  
   - Basis for our Leave-One-Host-Out (LOHO) validation protocol (`scripts/14_leave_one_host_out.py`) testing cross-family generalization across Zn-, In-, and Sn-based oxides.

5. **Dilute Dopant Sensitivity in Fractional Representations**  
   Wang, A. Y.-T. et al., *"Compositionally restricted attention-based network for materials property predictions"*, npj Comput. Mater. **2021**, 7, 77.  
   - Motivates our $L_2$ feature-shift analysis (`scripts/15_dopant_sensitivity.py`) and synthetic dopant inference test (`scripts/16_synthetic_inference.py`) in the 0.1–10% doping regime.

6. **Thermodynamic Stability ($E_{\text{hull}}$) and Benchmark Context**  
   Sun, W. et al., *"The thermodynamic scale of inorganic crystalline metastability"*, Sci. Adv. **2016**, 2, e1600225.  
   Bartel, C. J. et al., *"A critical examination of compound stability predictions from machine-learned formation energies"*, npj Comput. Mater. **2020**, 6, 97.  
   Riebesell, J. et al., *"A framework to evaluate machine learning crystal stability predictions"*, Nature Mach. Intell. **2025**, 7, 836–847.  
   - Context for the `0.1 eV/atom` synthesizability heuristic and the distinction between composition-only pre-screening and structure-aware models.