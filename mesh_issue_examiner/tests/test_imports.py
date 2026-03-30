"""
Import-chain smoke tests.

Verifies that every module in the package can be imported without errors
and that no circular dependencies or module-level side-effects cause
surprises at import time. This guards an entire *class* of bugs:

  - Circular imports between any pair of modules
  - Module-level code that raises on import (e.g. double registration)
  - Missing __init__.py re-exports breaking downstream imports
  - Singleton construction failures
"""

import importlib
import os
import sys

import pytest


def _discover_modules(base_dir):
    """Find all .py modules under base_dir and return their dotted import names.

    Skips test files, __pycache__, setup.py, and top-level scripts.
    """
    skip_names = {"setup.py", "conftest.py"}
    modules = []
    for root, dirs, files in os.walk(base_dir):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for f in files:
            if f in skip_names or f.startswith("test_") or not f.endswith(".py"):
                continue
            rel = os.path.relpath(os.path.join(root, f), base_dir)
            # Skip files at the package root (setup.py, __main__.py, etc.)
            if os.sep not in rel and not rel.startswith("__init__"):
                continue
            dotted = rel.replace(os.sep, ".")[:-3]  # strip .py
            if dotted.endswith(".__init__"):
                dotted = dotted[:-9]
            if dotted == "__init__":
                continue
            modules.append(dotted)
    return modules


class TestEveryModuleImportsCleanly:
    """Walk every sub-module and import it.

    If any module has a circular import, a missing dependency at module
    level, or code that raises during definition, this will catch it.
    """

    def test_all_submodules_import(self):
        """Every .py module in the package must import without error."""
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        modules = _discover_modules(base_dir)
        assert len(modules) > 10, f"Expected many modules, found {len(modules)}"
        for dotted in modules:
            try:
                importlib.import_module(dotted)
            except Exception as e:
                pytest.fail(f"Failed to import {dotted}: {type(e).__name__}: {e}")

    def test_adapters_subpackage_imports(self):
        import adapters.base_adapter
        import adapters.pyvista_adapter
        import adapters.trimesh_adapter
        import adapters.pymeshlab_adapter

    def test_operations_subpackage_imports(self):
        import operations.base
        import operations.operation_registry
        import operations._registry
        import operations.topological.nonmanifold_edges

    def test_core_subpackage_imports(self):
        import core.models
        import core.exceptions

    def test_mcp_subpackage_imports(self):
        import mcp.server
        import mcp.handlers
        import mcp.tools

    def test_utils_subpackage_imports(self):
        import utils.mesh_loader
        import utils.metrics


class TestNoCircularImports:
    """Verify key import paths that are prone to cycles."""

    def test_operations_base_then_registry(self):
        """Importing base then registry must not raise ImportError."""
        import operations.base
        import operations.operation_registry

    def test_operations_registry_then_base(self):
        """Importing registry then base must not raise ImportError."""
        import operations.operation_registry
        import operations.base

    def test_get_registry_works_after_all_operations_imports(self):
        """get_registry() must return a valid OperationRegistry after
        importing the full operations subpackage."""
        import operations.base
        import operations.operation_registry
        import operations._registry
        import operations.topological.nonmanifold_edges

        from operations.base import get_registry
        from operations.operation_registry import OperationRegistry

        reg = get_registry()
        assert isinstance(reg, OperationRegistry)

    def test_get_registry_is_singleton(self):
        from operations.base import get_registry
        assert get_registry() is get_registry()


class TestModuleLevelSideEffects:
    """Guard against module-level code that causes side effects at import."""

    def test_nonmanifold_edges_self_registers_in_global_registry(self):
        """Operations that self-register at module level must appear in the
        global registry after import. If a future operation forgets its
        module-level register call, this test catches it."""
        from operations.operation_registry import registry

        assert "topology.detect_nonmanifold_edges" in registry

    def test_mcp_server_import_does_not_raise(self):
        """Importing the MCP server must not trigger uncaught exceptions.

        Guards against the class of bugs where module-level code in
        server.py triggers registration failures or import errors.
        """
        from mcp.server import MeshIssueExaminerServer
        assert callable(MeshIssueExaminerServer)
