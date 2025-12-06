import sys
from pathlib import Path

# Ensure the repository root is on sys.path so tests can import the top-level
# `src` package (e.g. `import src.data.series_config`). When running pytest
# from various working directories, Python may not include the project root
# on sys.path, leading to ModuleNotFoundError: No module named 'src'.
ROOT = Path(__file__).resolve().parent.parent
root_str = str(ROOT)
if root_str not in sys.path:
    sys.path.insert(0, root_str)
