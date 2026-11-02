"""Step 01 - Check that the Python environment has everything the pipeline needs.

Prints the interpreter version and the version of each required library.
"""

import sys

print(f"Python: {sys.version}")

libs = ['pymatgen', 'matminer', 'sklearn', 'pandas', 'numpy',
        'matplotlib', 'seaborn', 'mp_api', 'tqdm', 'xgboost']

for lib in libs:
    try:
        mod = __import__(lib)
        print(f"  ✓ {lib}: {mod.__version__}")
    except ImportError:
        print(f"  ✗ {lib}: NOT INSTALLED")
    except AttributeError:
        print(f"  ✓ {lib}: installed (no __version__)")
