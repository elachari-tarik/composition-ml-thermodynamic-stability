#!/usr/bin/env python3
"""run_100x.py - Run the full pipeline N times, with the outputs of each run kept apart.

For every run it
  - cleans results/, figures/, logs/, models/ and data/processed/ (unless --no-clean),
  - executes run_all.py,
  - copies those folders into runs/<session>/run_NNN/,
  - writes runs/<session>/run_NNN/status.txt.

Usage (from the project root):
    python scripts/tools/run_100x.py                      # 100 runs, fresh session
    python scripts/tools/run_100x.py -n 10                # 10 runs
    python scripts/tools/run_100x.py --start 51           # runs 51..150
    python scripts/tools/run_100x.py --resume             # resume the latest session
    python scripts/tools/run_100x.py --session test1 -n 3 # named session
    python scripts/tools/run_100x.py --dry-run            # show the plan only
    python scripts/tools/run_100x.py --no-clean           # keep outputs between runs
    python scripts/tools/run_100x.py --stop-on-error      # abort on the first failed run
"""

import argparse
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from time import time, sleep

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from tco_stability.paths import find_project_root

# ---------- locate the project root and run_all.py ----------
PROJECT_ROOT = find_project_root(__file__)


def find_run_all():
    c = PROJECT_ROOT / "run_all.py"
    return c if c.exists() else None


RUN_ALL = find_run_all()
RUNS_ROOT = PROJECT_ROOT / "runs"

OUTPUT_DIRS = ["results", "figures", "logs", "models"]
DATA_SUBDIRS = ["processed"]


# ---------- helpers ----------
def clean_outputs(dry_run=False):
    targets = [PROJECT_ROOT / d for d in OUTPUT_DIRS]
    targets += [PROJECT_ROOT / "data" / s for s in DATA_SUBDIRS]
    for p in targets:
        if p.exists():
            if dry_run:
                print(f"  [dry] would remove {p.relative_to(PROJECT_ROOT)}")
            else:
                shutil.rmtree(p)


def collect_outputs(run_dir: Path, dry_run=False):
    run_dir.mkdir(parents=True, exist_ok=True)
    for d in OUTPUT_DIRS:
        src = PROJECT_ROOT / d
        if src.exists():
            if dry_run:
                print(f"  [dry] would copy {d}/ -> {run_dir.name}/{d}/")
            else:
                shutil.copytree(src, run_dir / d, dirs_exist_ok=True)
    for sub in DATA_SUBDIRS:
        src = PROJECT_ROOT / "data" / sub
        if src.exists():
            dst = run_dir / "data" / sub
            if dry_run:
                print(f"  [dry] would copy data/{sub}/ -> {run_dir.name}/data/{sub}/")
            else:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(src, dst, dirs_exist_ok=True)


def write_status(run_dir: Path, status: str, returncode: int,
                 elapsed: float, extra: str = ""):
    ts = datetime.now().isoformat(timespec="seconds")
    content = (
        f"status     : {status}\n"
        f"returncode : {returncode}\n"
        f"elapsed_s  : {elapsed:.1f}\n"
        f"finished_at: {ts}\n"
    )
    if extra:
        content += extra
    (run_dir / "status.txt").write_text(content, encoding="utf-8")


def run_pipeline(orchestrator_log, extra_args=()):
    """Run run_all.py, streaming its output to the console and to the orchestrator log."""
    cmd = [sys.executable, str(RUN_ALL), *extra_args]
    proc = subprocess.Popen(
        cmd,
        cwd=str(PROJECT_ROOT),
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
        if orchestrator_log is not None:
            orchestrator_log.write(line)
            orchestrator_log.flush()
    proc.wait()
    return proc.returncode


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(
        description="Run the full pipeline N times, with the outputs of each run kept apart."
    )
    ap.add_argument("-n", "--n", type=int, default=100,
                    help="Number of runs (default: 100).")
    ap.add_argument("--start", type=int, default=1,
                    help="Starting run index (default: 1).")
    ap.add_argument("--session", type=str, default=None,
                    help="Session folder name (default: session_<timestamp>).")
    ap.add_argument("--resume", action="store_true",
                    help="Resume the most recent session, skipping SUCCESS runs.")
    ap.add_argument("--no-clean", action="store_true",
                    help="Do NOT clean output dirs before each run.")
    ap.add_argument("--refresh-raw", action="store_true",
                    help="Pass --refresh-raw to run_all.py (WARNING: hits MP API each run).")
    ap.add_argument("--stop-on-error", action="store_true",
                    help="Abort on the first failed run.")
    ap.add_argument("--extra-args", type=str, default="",
                    help="Additional space-separated args forwarded to run_all.py.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Show the plan without executing anything.")
    args = ap.parse_args()

    if RUN_ALL is None:
        print("[ERROR] Could not find run_all.py.")
        sys.exit(1)

    # ----- session directory -----
    if args.resume:
        sessions = sorted(RUNS_ROOT.glob("session_*"))
        if not sessions:
            print(f"[ERROR] --resume given but no session found under {RUNS_ROOT}.")
            sys.exit(1)
        session_dir = sessions[-1]
        print(f"[resume] Using latest session: {session_dir.name}")
    else:
        name = args.session or f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        session_dir = RUNS_ROOT / name
        if session_dir.exists() and not args.dry_run:
            print(f"[ERROR] Session directory already exists: {session_dir}")
            print("        Use --resume to continue it, or pick another --session name.")
            sys.exit(1)

    if args.dry_run:
        print("=" * 78)
        print(" DRY RUN — no changes will be made")
        print("=" * 78)
        print(f" Project root : {PROJECT_ROOT}")
        print(f" run_all.py   : {RUN_ALL}")
        print(f" Session dir  : {session_dir}")
        print(f" N runs       : {args.n} (start={args.start})")
        print(f" Clean each run: {not args.no_clean}")
        for i in range(args.start, args.start + args.n):
            print(f"  -> run_{i:03d}")
        return

    session_dir.mkdir(parents=True, exist_ok=True)
    orchestrator_log_path = session_dir / "orchestrator.log"
    orchestrator_log = open(orchestrator_log_path, "w", encoding="utf-8", buffering=1)

    print("=" * 78)
    print(" MULTI-RUN ORCHESTRATOR")
    print("=" * 78)
    print(f" Project root       : {PROJECT_ROOT}")
    print(f" run_all.py         : {RUN_ALL}")
    print(f" Session directory  : {session_dir}")
    print(f" Orchestrator log   : {orchestrator_log_path}")
    print(f" Runs               : {args.n} (start={args.start})")
    print(f" Clean between runs : {not args.no_clean}")
    print(f" Resume mode        : {args.resume}")
    print(f" Refresh raw data   : {args.refresh_raw}")
    try:
        _, _, free = shutil.disk_usage(session_dir.parent)
        print(f" Free disk space    : {free / (1024**3):.1f} GB")
    except Exception:
        pass
    print("=" * 78)

    if args.refresh_raw:
        print("[!] WARNING: --refresh-raw will hit the Materials Project API on EVERY run.")
        print("    This can exhaust your API quota and take a very long time.")
        sleep(2)

    extra = []
    if args.refresh_raw:
        extra.append("--refresh-raw")
    if args.extra_args:
        extra.extend(args.extra_args.split())

    failures = []
    completed = 0
    skipped = 0
    first_elapsed = None
    t_all = time()

    try:
        for i in range(args.start, args.start + args.n):
            run_dir = session_dir / f"run_{i:03d}"

            if args.resume and (run_dir / "status.txt").exists():
                if "status     : SUCCESS" in (run_dir / "status.txt").read_text(encoding="utf-8"):
                    print(f"\n[{i:03d}] SKIP (already SUCCESS)")
                    skipped += 1
                    continue

            if run_dir.exists():
                shutil.rmtree(run_dir)
            run_dir.mkdir()

            print(f"\n[{i:03d}] Starting run -> {run_dir.relative_to(PROJECT_ROOT)}")
            print("-" * 78)

            if not args.no_clean:
                clean_outputs()

            t0 = time()
            rc = -1
            try:
                rc = run_pipeline(orchestrator_log, extra)
            except KeyboardInterrupt:
                print("\n[!] Interrupted by user. Saving partial outputs...")
                collect_outputs(run_dir)
                write_status(run_dir, "INTERRUPTED", -1, time() - t0)
                raise
            except Exception as e:
                print(f"[!] Exception during subprocess: {e}")

            elapsed = time() - t0
            if first_elapsed is None:
                first_elapsed = elapsed

            collect_outputs(run_dir)
            status = "SUCCESS" if rc == 0 else "FAILED"
            write_status(run_dir, status, rc, elapsed)
            print(f"[{i:03d}] {status} in {elapsed:.1f}s")

            if rc != 0:
                failures.append((i, rc))
                if args.stop_on_error:
                    print("[!] --stop-on-error: aborting loop.")
                    break
            else:
                completed += 1

            # ETA
            remaining = args.start + args.n - i - 1  # iterations after this one
            if remaining > 0 and first_elapsed is not None:
                eta_h = first_elapsed * remaining / 3600
                print(f"  [ETA] ~{eta_h:.1f} h remaining for {remaining} run(s)")

    except KeyboardInterrupt:
        print("\n[!] Session interrupted by user.")
    finally:
        total = time() - t_all
        print("\n" + "=" * 78)
        print(f" SESSION SUMMARY — {session_dir.name}")
        print("=" * 78)
        print(f" Runs requested  : {args.n}")
        print(f" Successful      : {completed}")
        print(f" Skipped (resume): {skipped}")
        print(f" Failed          : {len(failures)}")
        for i, rc in failures:
            print(f"   - run_{i:03d} (rc={rc})")
        print(f" Total wall time : {total/60:.1f} min ({total/3600:.2f} h)")
        print(f" Session dir     : {session_dir}")
        print("=" * 78)
        orchestrator_log.close()


if __name__ == "__main__":
    main()
