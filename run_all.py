#!/usr/bin/env python3
"""run_all.py - Master script: runs the full pipeline in manuscript order.

Executes every script in scripts/ following the order of the manuscript (Sections 2.1 to 3.8)
and generates all tables, metrics, trained model weights, and publication figures.

If the archived raw dataset (data/raw/mp_doped_oxides_raw.csv) is already present, the live
Materials Project API queries (steps 02 and 03) are skipped so the dataset remains frozen.
Pass `--refresh-raw` to download a fresh copy (requires the MP_API_KEY environment variable).

Logging
  - Main log      : logs/run_all_<YYYYmmdd_HHMMSS>.log   (complete console output)
  - Per-step logs : logs/steps/<NN>_<script>.log         (stdout + stderr of each step)

Usage
    python run_all.py
    python run_all.py --refresh-raw
"""

import subprocess
import sys
from pathlib import Path
from time import time
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

RAW_CSV = PROJECT_ROOT / "data" / "raw" / "mp_doped_oxides_raw.csv"
REFRESH_RAW = "--refresh-raw" in sys.argv

# Optional "--from-step N" argument to resume from step N (1-based)
START_FROM_STEP = 1
for i, arg in enumerate(sys.argv):
    if arg == "--from-step" and i + 1 < len(sys.argv):
        START_FROM_STEP = int(sys.argv[i + 1])
    elif arg.startswith("--from-step="):
        START_FROM_STEP = int(arg.split("=", 1)[1])

# ----- Logging setup ---------------------------------------------------------
LOGS_DIR = PROJECT_ROOT / "logs"
STEPS_LOG_DIR = LOGS_DIR / "steps"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
STEPS_LOG_DIR.mkdir(parents=True, exist_ok=True)

RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
MAIN_LOG_PATH = LOGS_DIR / f"run_all_{RUN_TIMESTAMP}.log"


class Tee:
    """Send every write to both console and log file."""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            try:
                s.write(data)
                s.flush()
            except Exception:
                pass

    def flush(self):
        for s in self.streams:
            try:
                s.flush()
            except Exception:
                pass

    def isatty(self):
        return False


# Pipeline steps mapped to manuscript sections
PIPELINE_STEPS = [
    # --- PHASE 1: environment and data acquisition (Section 2.1) ---
    ("01_check_env.py", "Section 2.1 - Environment check", True),
    ("02_test_api.py", "Section 2.1 - MP API connection test", REFRESH_RAW or not RAW_CSV.exists()),
    ("03_query_materials.py", "Section 2.1 - MP API data acquisition", REFRESH_RAW or not RAW_CSV.exists()),

    # --- PHASE 1: cleaning, sensitivity and EDA (Sections 2.2 & 3.1) ---
    ("04_clean_data.py", "Section 2.2 - Data cleaning & polymorph deduplication", True),
    ("05_count_host_structures.py", "Section 3.1 - Dilute host vs multicomponent oxide count", True),
    ("06_sensitivity_ef.py", "Section 2.2 - Sensitivity analysis on Ef >= 0 filtering", True),
    ("07_plot_data_exploration.py", "Section 3.1 - Exploratory data analysis plots", True),
    ("08_plot_ehull_by_host.py", "Section 3.1 - Ehull distribution by host element", True),

    # --- PHASE 2: 105-feature baseline (Sections 2.3-2.6 & 3.2) ---
    ("09_phase2_baseline.py", "Section 3.2 - Initial 105-feature baseline RF", True),

    # --- PHASE 3: tuning, 118-feature models, CV (Sections 3.2-3.5) ---
    ("10_phase3_tuning_105.py", "Section 3.2 - 105-feature tuning & interpretation", True),
    ("11_phase3_enhanced_118.py", "Sections 3.3-3.5 - 118-feature RF/GBR/XGBoost, 5-fold CV & final test", True),
    ("12_pub_figures.py", "Sections 3.1-3.6 - Publication figures (Fig 1 to Fig 6)", True),

    # --- ADDITIONAL ANALYSES: PFI, extrapolation, dopant sensitivity (Sections 3.6-3.8) ---
    ("13_permutation_importance.py", "Sections 2.8 & 3.6 - Validation-set PFI analysis (Fig 7)", True),
    ("14_leave_one_host_out.py", "Section 3.7 - Leave-one-host-out extrapolation (Fig 8 & Fig 9)", True),
    ("15_dopant_sensitivity.py", "Section 3.8 - Magpie L2 feature sensitivity to doping (Fig 10)", True),
    ("16_synthetic_inference.py", "Section 3.8 - Synthetic dilute dopant inference test (Fig 11)", True),

    # --- ARCHIVE: trained models ---
    ("17_save_models.py", "Reproducibility archive - save trained models to models/", True),
]


def resolve_script_path(script_name: str) -> Path:
    return SCRIPTS_DIR / script_name


def run_script_streaming(script_path: Path, step_idx: int):
    """Run one script as a subprocess and stream output to console and step log."""
    import os

    step_log_path = STEPS_LOG_DIR / f"{step_idx:02d}_{script_path.stem}.log"
    print(f"  [LOG] Per-step log: {step_log_path.relative_to(PROJECT_ROOT)}")

    # Environment variables to prevent Windows cp1252 Unicode crashes and
    # Python 3.15 multiprocessing pipe race conditions ("concurrent send_bytes()")
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["JOBLIB_MULTIPROCESSING"] = "0"

    with open(step_log_path, "w", encoding="utf-8", buffering=1) as step_fh:
        proc = subprocess.Popen(
            [sys.executable, str(script_path)],
            cwd=str(PROJECT_ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            step_fh.write(line)
        proc.wait()

    return proc.returncode, step_log_path


def main():
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    log_fh = open(MAIN_LOG_PATH, "w", encoding="utf-8", buffering=1)

    sys.stdout = Tee(original_stdout, log_fh)
    sys.stderr = Tee(original_stderr, log_fh)

    python_exe = sys.executable
    t_start = time()

    try:
        print("=" * 78)
        print(" STARTING COMPLETE REPRODUCIBLE PIPELINE EXECUTION")
        print("=" * 78)
        print(f"Start time        : {datetime.now().isoformat(timespec='seconds')}")
        print(f"Python executable : {python_exe}")
        print(f"Scripts directory : {SCRIPTS_DIR}")
        print(f"Project root (cwd): {PROJECT_ROOT}")
        print(f"Raw dataset exists: {RAW_CSV.exists()} ({RAW_CSV})")
        print(f"Main log file     : {MAIN_LOG_PATH}")
        print(f"Per-step logs dir : {STEPS_LOG_DIR}")
        print("=" * 78)

        for idx, (script_name, description, should_run) in enumerate(PIPELINE_STEPS, 1):
            script_path = resolve_script_path(script_name)
            print(f"\n[{idx:02d}/{len(PIPELINE_STEPS):02d}] {script_path.name} — {description}")
            print("-" * 78)

            if idx < START_FROM_STEP:
                print(f"  [SKIP] Step {idx:02d} < --from-step {START_FROM_STEP}")
                continue

            if not should_run:
                print(f"  [SKIP] Archived raw data already exists at {RAW_CSV.name}.")
                print("         (Use `python run_all.py --refresh-raw` to re-query Materials Project)")
                continue

            if not script_path.exists():
                print(f"  [ERROR] Script not found: {script_path}")
                sys.exit(1)

            t0 = time()
            returncode, step_log_path = run_script_streaming(script_path, idx)
            elapsed = time() - t0

            if returncode != 0:
                print(f"\n  [!] ERROR in {script_path.name} (exit code = {returncode}).")
                print(f"      See log: {step_log_path}")
                print("      Pipeline execution aborted.")
                sys.exit(returncode)

            print(f"  [OK] Completed {script_path.name} in {elapsed:.1f}s")

        total_time = time() - t_start
        print("\n" + "=" * 78)
        print(f" PIPELINE COMPLETED SUCCESSFULLY IN {total_time:.1f}s")
        print(f" End time: {datetime.now().isoformat(timespec='seconds')}")
        print(" Key outputs generated:")
        print("   - results/final_metrics_summary.csv")
        print("   - results/test_predictions.csv")
        print("   - results/sensitivity_ef_results.csv")
        print("   - results/permutation_importance_full.csv")
        print("   - data/processed/ambiguous_cases.csv")
        print("   - figures/publication/ (all manuscript figures)")
        print("   - models/ (trained models, scaler, feature names)")
        print(f" Full execution log : {MAIN_LOG_PATH}")
        print(f" Per-step logs      : {STEPS_LOG_DIR}")
        print("=" * 78)

    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_fh.close()


if __name__ == "__main__":
    main()