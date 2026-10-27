"""Pytest bootstrap: regenerate e2e sample meshes when they are missing.

The e2e suites (``test_sample_meshes_e2e.py``, ``test_triangle_intersection.py``)
depend on deterministic fixture files under ``<repo>/e2e/sample_meshes/``. The
fixtures are cheap to rebuild and intentionally not committed; the first test
session recreates them with ``scripts/generate_sample_meshes.py`` (skipping any
files that already exist, so user-made edits are preserved).
"""

import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_DIR = REPO_ROOT / "e2e" / "sample_meshes"
GENERATOR = REPO_ROOT / "scripts" / "generate_sample_meshes.py"

# Written last by the generator; its absence means fixtures are incomplete.
_SENTINEL = SAMPLE_DIR / "isolated_vertices" / "isolated_vertices_surface.vtk"


@pytest.fixture(scope="session", autouse=True)
def sample_meshes():
    if not _SENTINEL.exists():
        spec = importlib.util.spec_from_file_location(
            "generate_sample_meshes", GENERATOR)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        written = module.generate(force=False)
        assert written, f"fixture generation wrote nothing under {SAMPLE_DIR}"
    return SAMPLE_DIR
