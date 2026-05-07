"""
pytest configuration: add the project root to sys.path so tests can import
project modules (pages.data.retrieve_data, pages.data.grammarFunctions, etc.)
without installing the project as a package.

Heavy dependencies (umap, numba, leidenalg, igraph, seaborn) are stubbed out
in sys.modules before any test file imports grammarFunctions.py, which avoids
the llvmlite/numba JIT compilation overhead during the test collection phase.
"""
import sys
import types
import os

# ---------------------------------------------------------------------------
# 1. Add project root to sys.path
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ---------------------------------------------------------------------------
# 2. Stub out heavy / JIT-compiled libraries that are not needed by the
#    pure functions under test.  This must happen before any test module
#    imports pages.data.grammarFunctions.
# ---------------------------------------------------------------------------
def _make_stub(name):
    mod = types.ModuleType(name)
    mod.__spec__ = None
    return mod


_STUBS = [
    "umap",
    "umap.umap_",
    "numba",
    "numba.core",
    "numba.core.types",
    "llvmlite",
    "llvmlite.ir",
    "llvmlite.ir.values",
    "leidenalg",
    "igraph",
    "seaborn",
]

for _name in _STUBS:
    if _name not in sys.modules:
        sys.modules[_name] = _make_stub(_name)

# umap.UMAP is referenced directly; give the stub a no-op class
sys.modules["umap"].UMAP = type("UMAP", (), {"fit_transform": lambda self, x: x})  # type: ignore
