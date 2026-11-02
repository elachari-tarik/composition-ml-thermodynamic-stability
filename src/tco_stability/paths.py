"""Project-root handling.

Every script calls ``ensure_project_root()`` first so that relative paths such as
``data/processed/...`` resolve the same way no matter where the script is launched from.
The root is the folder that contains ``environment.yml``.
"""

import os
from pathlib import Path


def find_project_root(start=None):
    here = Path(start or __file__).resolve()
    for parent in [here] + list(here.parents):
        if (parent / "environment.yml").is_file():
            return parent
    raise RuntimeError("Could not locate the project root (no environment.yml found).")


def ensure_project_root(start=None):
    root = find_project_root(start)
    os.chdir(root)
    print(f"Working directory: {os.getcwd()}")
    return root
