"""
Comprehensive unit tests for operations/base.py and operations/operation_registry.py.

Tests OperationMetadata, BaseOperation, QualityOperation, OperationError,
module-level registry functions, OperationCategory, OperationRegistry,
workflow adjacency priming, reset_registry, and registry injection into
MeshIssueExaminerServer.
"""

import asyncio

import pytest
import numpy as np

from mesh_issue_examiner.core.models import (
    MeshInput,
    MeshIssue,
    QualityMetrics,
    IssueCategory,
    IssueSeverity,
    OperationConfig,
    InspectionResult,
)
from mesh_issue_examiner.mcp.server import MCPRequest, MeshIssueExaminerServer
from mesh_issue_examiner.operations import _registry as _registry_module
from mesh_issue_examiner.operations import operation_registry as operation_registry_module
from mesh_issue_examiner.operations.base import (
    BaseOperation,
    OperationMetadata,
    QualityOperation,
    OperationError,
    get_registry,
    register_operation,
    _register_decorator,
)
from mesh_issue_examiner.operations.operation_registry import (
    OperationRegistry,
    OperationCategory,
    reset_registry,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mesh():
    """Create a minimal valid MeshInput for testing."""
    return MeshInput(
        vertices=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float),
        faces=np.array([[0, 1, 2]]),
    )


class MockOperation(BaseOperation):
    """Concrete BaseOperation subclass for testing."""

    def __init__(
        self,
        op_id="mock.op",
        category=IssueCategory.TOPOLOGICAL,
        tags=None,
        libraries=None,
        config=None,
        requires_adjacency=False,
    ):
        super().__init__(config=config)
        self._op_id = op_id
        self._category = category
        self._tags = tags or ["test"]
        self._libraries = libraries or ["trimesh"]
        self._requires_adjacency = requires_adjacency

    @property
    def metadata(self):
        return OperationMetadata(
            operation_id=self._op_id,
            display_name=f"Mock {self._op_id}",
            description="Mock operation for testing",
            category=self._category,
            supported_libraries=self._libraries,
            recommended_library=self._libraries[0],
            requires_adjacency=self._requires_adjacency,
            tags=self._tags,
        )

    def execute(self, mesh):
        return []


class FailingOperation(BaseOperation):
    """Operation whose execute() always raises."""

    @property
    def metadata(self):
        return OperationMetadata(
            operation_id="mock.fail",
            display_name="Failing Op",
            description="Always raises",
            category=IssueCategory.GEOMETRIC,
            supported_libraries=["trimesh"],
            recommended_library="trimesh",
        )

    def execute(self, mesh):
        raise RuntimeError("deliberate failure")


class MockQualityOperation(QualityOperation):
    """Concrete QualityOperation subclass for testing."""

    @property
    def metric_name(self):
        return "test_metric"

    def compute(self, mesh):
        return QualityMetrics(
            operation_id="quality.test",
            metric_name=self.metric_name,
            values=np.array([0.5, 0.8, 0.3]),
            threshold=self.get_threshold(),
        )

    def get_threshold(self):
        return 0.4


# ===================================================================
# Part 1: operations/base.py
# ===================================================================


class TestOperationMetadata:
    """Tests for the OperationMetadata dataclass."""

    def test_construction_with_all_fields(self):
        meta = OperationMetadata(
            operation_id="test.op",
            display_name="Test Op",
            description="A test operation",
            category=IssueCategory.GEOMETRIC,
            supported_libraries=["trimesh", "pyvista"],
            recommended_library="trimesh",
            requires_adjacency=True,
            complexity="O(n^2)",
            tags=["geometry", "surface"],
        )
        assert meta.operation_id == "test.op"
        assert meta.display_name == "Test Op"
        assert meta.description == "A test operation"
        assert meta.category is IssueCategory.GEOMETRIC
        assert meta.supported_libraries == ["trimesh", "pyvista"]
        assert meta.recommended_library == "trimesh"
        assert meta.requires_adjacency is True
        assert meta.complexity == "O(n^2)"
        assert meta.tags == ["geometry", "surface"]

    def test_defaults(self):
        meta = OperationMetadata(
            operation_id="x",
            display_name="X",
            description="X",
            category=IssueCategory.TOPOLOGICAL,
            supported_libraries=["trimesh"],
            recommended_library="trimesh",
        )
        assert meta.requires_adjacency is False
        assert meta.complexity == "O(n)"
        assert meta.tags == []

    def test_to_dict(self):
        meta = OperationMetadata(
            operation_id="d.op",
            display_name="D",
            description="D op",
            category=IssueCategory.ORIENTATION,
            supported_libraries=["pyvista"],
            recommended_library="pyvista",
            requires_adjacency=True,
            complexity="O(n log n)",
            tags=["orient"],
        )
        d = meta.to_dict()
        assert d["operation_id"] == "d.op"
        assert d["display_name"] == "D"
        assert d["description"] == "D op"
        assert d["category"] == "orientation"
        assert d["supported_libraries"] == ["pyvista"]
        assert d["recommended_library"] == "pyvista"
        assert d["requires_adjacency"] is True
        assert d["complexity"] == "O(n log n)"
        assert d["tags"] == ["orient"]


# -- BaseOperation --


class TestBaseOperation:
    """Tests for the BaseOperation ABC."""

    def test_concrete_subclass_returns_empty_list(self):
        op = MockOperation()
        mesh = _make_mesh()
        result = op.execute(mesh)
        assert result == []

    def test_run_increments_execution_count(self):
        op = MockOperation()
        mesh = _make_mesh()
        assert op._execution_count == 0
        op.run(mesh)
        assert op._execution_count == 1
        op.run(mesh)
        assert op._execution_count == 2

    def test_run_tracks_timing(self):
        op = MockOperation()
        mesh = _make_mesh()
        assert op._total_time == 0.0
        op.run(mesh)
        assert op._total_time > 0

    def test_run_catches_exceptions_and_wraps_in_operation_error(self):
        op = FailingOperation()
        mesh = _make_mesh()
        with pytest.raises(OperationError) as exc_info:
            op.run(mesh)
        assert "mock.fail" in str(exc_info.value)
        # The original exception should be chained
        assert exc_info.value.__cause__ is not None
        assert isinstance(exc_info.value.__cause__, RuntimeError)

    def test_statistics_no_executions_average_time_zero(self):
        op = MockOperation()
        stats = op.statistics
        assert stats["execution_count"] == 0
        assert stats["total_time"] == 0.0
        assert stats["avg_time"] == 0  # divide-by-zero guard

    def test_statistics_after_executions(self):
        op = MockOperation()
        mesh = _make_mesh()
        op.run(mesh)
        op.run(mesh)
        stats = op.statistics
        assert stats["execution_count"] == 2
        assert stats["total_time"] > 0
        assert stats["avg_time"] > 0

    def test_repr_includes_class_name(self):
        op = MockOperation(op_id="mock.op")
        r = repr(op)
        assert "MockOperation" in r
        assert "mock.op" in r

    def test_custom_config_is_stored(self):
        cfg = OperationConfig(tolerance=0.01, quality_threshold=0.5)
        op = MockOperation(config=cfg)
        assert op.config is cfg
        assert op.config.tolerance == 0.01
        assert op.config.quality_threshold == 0.5

    def test_default_config_when_none(self):
        op = MockOperation(config=None)
        assert isinstance(op.config, OperationConfig)


# -- QualityOperation --


class TestQualityOperation:
    """Tests for the QualityOperation ABC."""

    def test_severity_critical_above_0_5(self):
        op = MockQualityOperation()
        assert op.get_recommended_severity(0.6) is IssueSeverity.CRITICAL
        assert op.get_recommended_severity(0.51) is IssueSeverity.CRITICAL
        assert op.get_recommended_severity(1.0) is IssueSeverity.CRITICAL

    def test_severity_error_above_0_2(self):
        op = MockQualityOperation()
        assert op.get_recommended_severity(0.21) is IssueSeverity.ERROR
        assert op.get_recommended_severity(0.5) is IssueSeverity.ERROR

    def test_severity_warning_above_0_05(self):
        op = MockQualityOperation()
        assert op.get_recommended_severity(0.06) is IssueSeverity.WARNING
        assert op.get_recommended_severity(0.2) is IssueSeverity.WARNING

    def test_severity_info_at_or_below_0_05(self):
        op = MockQualityOperation()
        assert op.get_recommended_severity(0.05) is IssueSeverity.INFO
        assert op.get_recommended_severity(0.01) is IssueSeverity.INFO
        assert op.get_recommended_severity(0.0) is IssueSeverity.INFO

    def test_compute_and_threshold(self):
        op = MockQualityOperation()
        mesh = _make_mesh()
        metrics = op.compute(mesh)
        assert metrics.metric_name == "test_metric"
        assert op.get_threshold() == 0.4


# -- OperationError --


class TestOperationError:
    """Tests for the OperationError exception class."""

    def test_is_exception_subclass(self):
        assert issubclass(OperationError, Exception)

    def test_can_be_raised_and_caught(self):
        with pytest.raises(OperationError):
            raise OperationError("something went wrong")

    def test_message(self):
        try:
            raise OperationError("test message")
        except OperationError as e:
            assert "test message" in str(e)


# -- Module-level functions --


class TestModuleLevelFunctions:
    """Tests for get_registry, register_operation, _register_decorator."""

    def test_get_registry_returns_registry(self):
        reg = get_registry()
        assert isinstance(reg, OperationRegistry)

    def test_get_registry_returns_singleton(self):
        r1 = get_registry()
        r2 = get_registry()
        assert r1 is r2

    def test_register_operation_registers_instance(self):
        # register_operation registers with the global singleton.
        # We verify it does not raise and returns the operation.
        op = MockOperation(op_id="__test_register_op__")
        result = register_operation(op)
        assert result is op
        # Clean up the global registry
        get_registry().unregister("__test_register_op__")

    def test_register_decorator_registers_class(self):
        @_register_decorator
        class _DecoratedOp(BaseOperation):
            @property
            def metadata(self):
                return OperationMetadata(
                    operation_id="__decorated_op__",
                    display_name="Decorated",
                    description="",
                    category=IssueCategory.QUALITY,
                    supported_libraries=["trimesh"],
                    recommended_library="trimesh",
                )

            def execute(self, mesh):
                return []

        # The class should still be returned
        assert issubclass(_DecoratedOp, BaseOperation)
        # It should be in the global registry
        assert "__decorated_op__" in get_registry()
        # Clean up
        get_registry().unregister("__decorated_op__")


# ===================================================================
# Part 2: operations/operation_registry.py
# ===================================================================


class TestOperationCategory:
    """Tests for the OperationCategory dataclass."""

    def test_construction(self):
        cat = OperationCategory(name="topological")
        assert cat.name == "topological"
        assert cat.operations == {}

    def test_register_and_get(self):
        cat = OperationCategory(name="geometric")
        op = MockOperation(op_id="cat.op1", category=IssueCategory.GEOMETRIC)
        cat.register(op)
        assert cat.get("cat.op1") is op
        assert cat.get("nonexistent") is None

    def test_list_operations(self):
        cat = OperationCategory(name="quality")
        op1 = MockOperation(op_id="c.q1", category=IssueCategory.QUALITY)
        op2 = MockOperation(op_id="c.q2", category=IssueCategory.QUALITY)
        cat.register(op1)
        cat.register(op2)
        metas = cat.list_operations()
        ids = {m.operation_id for m in metas}
        assert ids == {"c.q1", "c.q2"}

    def test_len(self):
        cat = OperationCategory(name="orientation")
        assert len(cat) == 0
        cat.register(MockOperation(op_id="o.1", category=IssueCategory.ORIENTATION))
        assert len(cat) == 1

    def test_iter(self):
        cat = OperationCategory(name="topological")
        op = MockOperation(op_id="t.1", category=IssueCategory.TOPOLOGICAL)
        cat.register(op)
        assert list(cat) == [op]


class TestOperationRegistryFresh:
    """Tests for OperationRegistry with fresh instances for isolation."""

    def _fresh_registry(self):
        """Always return a brand-new OperationRegistry."""
        return OperationRegistry()

    # -- 1. Fresh registry creation --

    def test_fresh_registry_is_empty(self):
        reg = self._fresh_registry()
        assert len(reg) == 0
        assert reg.list_all() == []

    # -- 2. Register a mock operation --

    def test_register_operation(self):
        reg = self._fresh_registry()
        op = MockOperation(op_id="reg.test")
        result = reg.register(op)
        assert result is reg  # returns self for chaining
        assert "reg.test" in reg
        assert len(reg) == 1

    # -- 3. Register duplicate ID raises ValueError --

    def test_register_duplicate_is_idempotent(self):
        reg = self._fresh_registry()
        op1 = MockOperation(op_id="dup.id")
        op2 = MockOperation(op_id="dup.id")
        reg.register(op1)
        reg.register(op2)  # idempotent: should not raise
        assert len(reg) == 1

    # -- 4. Unregister existing returns True --

    def test_unregister_existing(self):
        reg = self._fresh_registry()
        op = MockOperation(op_id="unreg.me")
        reg.register(op)
        assert reg.unregister("unreg.me") is True
        assert "unreg.me" not in reg
        assert len(reg) == 0

    # -- 5. Unregister nonexistent returns False --

    def test_unregister_nonexistent(self):
        reg = self._fresh_registry()
        assert reg.unregister("no.such.op") is False

    # -- 6. Get existing operation --

    def test_get_existing(self):
        reg = self._fresh_registry()
        op = MockOperation(op_id="get.me")
        reg.register(op)
        assert reg.get("get.me") is op

    # -- 7. Get nonexistent returns None --

    def test_get_nonexistent(self):
        reg = self._fresh_registry()
        assert reg.get("nope") is None

    # -- 8. __contains__ --

    def test_contains(self):
        reg = self._fresh_registry()
        op = MockOperation(op_id="cont.op")
        reg.register(op)
        assert "cont.op" in reg
        assert "other" not in reg

    # -- 9. __len__ --

    def test_len(self):
        reg = self._fresh_registry()
        assert len(reg) == 0
        reg.register(MockOperation(op_id="len.1"))
        assert len(reg) == 1
        reg.register(MockOperation(op_id="len.2"))
        assert len(reg) == 2

    # -- 10. __iter__ --

    def test_iter(self):
        reg = self._fresh_registry()
        op1 = MockOperation(op_id="iter.1")
        op2 = MockOperation(op_id="iter.2")
        reg.register(op1)
        reg.register(op2)
        ops = list(reg)
        assert len(ops) == 2
        assert op1 in ops
        assert op2 in ops

    # -- 11. list_all() --

    def test_list_all(self):
        reg = self._fresh_registry()
        reg.register(MockOperation(op_id="la.1"))
        reg.register(MockOperation(op_id="la.2"))
        metas = reg.list_all()
        ids = {m.operation_id for m in metas}
        assert ids == {"la.1", "la.2"}

    # -- 12. list_by_category --

    def test_list_by_category(self):
        reg = self._fresh_registry()
        reg.register(MockOperation(op_id="cat.topo", category=IssueCategory.TOPOLOGICAL))
        reg.register(MockOperation(op_id="cat.geom", category=IssueCategory.GEOMETRIC))
        topo_metas = reg.list_by_category(IssueCategory.TOPOLOGICAL)
        topo_ids = {m.operation_id for m in topo_metas}
        assert "cat.topo" in topo_ids
        assert "cat.geom" not in topo_ids

    # -- 13. list_by_library --

    def test_list_by_library(self):
        reg = self._fresh_registry()
        reg.register(MockOperation(op_id="lib.tri", libraries=["trimesh"]))
        reg.register(MockOperation(op_id="lib.pv", libraries=["pyvista"]))
        reg.register(MockOperation(op_id="lib.both", libraries=["trimesh", "pyvista"]))
        tri_metas = reg.list_by_library("trimesh")
        tri_ids = {m.operation_id for m in tri_metas}
        assert "lib.tri" in tri_ids
        assert "lib.both" in tri_ids
        assert "lib.pv" not in tri_ids

    # -- 14. list_by_tag --

    def test_list_by_tag(self):
        reg = self._fresh_registry()
        reg.register(MockOperation(op_id="tag.a", tags=["alpha", "beta"]))
        reg.register(MockOperation(op_id="tag.b", tags=["beta"]))
        reg.register(MockOperation(op_id="tag.c", tags=["gamma"]))
        alpha_metas = reg.list_by_tag("alpha")
        alpha_ids = {m.operation_id for m in alpha_metas}
        assert alpha_ids == {"tag.a"}

    # -- 15. create_workflow with valid IDs returns callable --

    def test_create_workflow_valid_ids(self):
        reg = self._fresh_registry()
        reg.register(MockOperation(op_id="wf.1"))
        reg.register(MockOperation(op_id="wf.2"))
        wf = reg.create_workflow(["wf.1", "wf.2"])
        assert callable(wf)

    # -- 16. create_workflow with invalid ID raises ValueError --

    def test_create_workflow_invalid_id(self):
        reg = self._fresh_registry()
        reg.register(MockOperation(op_id="wf.ok"))
        with pytest.raises(ValueError, match="Unknown operation"):
            reg.create_workflow(["wf.ok", "wf.missing"])

    # -- 17. Workflow execution returns InspectionResult --

    def test_workflow_execution_returns_inspection_result(self):
        reg = self._fresh_registry()
        reg.register(MockOperation(op_id="wfexec.1"))
        wf = reg.create_workflow(["wfexec.1"])
        mesh = _make_mesh()
        result = wf(mesh)
        assert isinstance(result, InspectionResult)
        assert result.processing_time >= 0

    # -- 18. create_full_workflow runs all registered ops --

    def test_create_full_workflow(self):
        reg = self._fresh_registry()
        reg.register(MockOperation(op_id="full.1"))
        reg.register(MockOperation(op_id="full.2"))
        wf = reg.create_full_workflow()
        mesh = _make_mesh()
        result = wf(mesh)
        assert isinstance(result, InspectionResult)
        # Both operations ran (they return empty issues, so just verify no error)
        assert result.processing_time >= 0

    # -- 19. get_recommended_library --

    def test_get_recommended_library(self):
        reg = self._fresh_registry()
        reg.register(MockOperation(
            op_id="rec.lib",
            libraries=["trimesh", "pyvista"],
        ))
        assert reg.get_recommended_library("rec.lib") == "trimesh"
        assert reg.get_recommended_library("nonexistent") is None

    # -- 20. get_statistics --

    def test_get_statistics_empty_registry(self):
        reg = self._fresh_registry()
        stats = reg.get_statistics()
        assert stats["total_operations"] == 0
        assert stats["total_executions"] == 0
        assert stats["total_time"] == 0.0

    def test_get_statistics_with_operations(self):
        reg = self._fresh_registry()
        op = MockOperation(op_id="stats.1")
        reg.register(op)
        mesh = _make_mesh()
        op.run(mesh)
        op.run(mesh)
        stats = reg.get_statistics()
        assert stats["total_operations"] == 1
        assert stats["total_executions"] == 2
        assert stats["total_time"] > 0
        # by_category should contain all IssueCategory values
        for cat in IssueCategory:
            assert cat.value in stats["by_category"]

    # -- 21. get_by_category --

    def test_get_by_category(self):
        reg = self._fresh_registry()
        op = MockOperation(op_id="gbc.1", category=IssueCategory.GEOMETRIC)
        reg.register(op)
        cat = reg.get_by_category(IssueCategory.GEOMETRIC)
        assert isinstance(cat, OperationCategory)
        assert cat.name == "geometric"
        assert cat.get("gbc.1") is op

    # -- Additional edge cases --

    def test_unregister_removes_from_category(self):
        """Verify unregister removes from both _operations and category."""
        reg = self._fresh_registry()
        op = MockOperation(op_id="unreg.cat", category=IssueCategory.QUALITY)
        reg.register(op)
        cat = reg.get_by_category(IssueCategory.QUALITY)
        assert len(cat) == 1
        reg.unregister("unreg.cat")
        assert len(cat) == 0

    def test_create_category_workflow(self):
        reg = self._fresh_registry()
        reg.register(MockOperation(
            op_id="cw.topo", category=IssueCategory.TOPOLOGICAL
        ))
        reg.register(MockOperation(
            op_id="cw.geom", category=IssueCategory.GEOMETRIC
        ))
        wf = reg.create_category_workflow(IssueCategory.TOPOLOGICAL)
        mesh = _make_mesh()
        result = wf(mesh)
        assert isinstance(result, InspectionResult)

    def test_create_workflow_with_config(self):
        reg = self._fresh_registry()
        op = MockOperation(op_id="wf.cfg")
        reg.register(op)
        original_cfg = op.config
        cfg = OperationConfig(tolerance=0.5)
        wf = reg.create_workflow(["wf.cfg"], config=cfg)
        # The workflow should be created without mutating the singleton's config
        assert wf is not None
        assert op.config is original_cfg

    def test_register_method_chaining(self):
        reg = self._fresh_registry()
        result = (
            reg
            .register(MockOperation(op_id="chain.1"))
            .register(MockOperation(op_id="chain.2"))
            .register(MockOperation(op_id="chain.3"))
        )
        assert result is reg
        assert len(reg) == 3

    def test_categories_initialized_for_all_issue_categories(self):
        reg = self._fresh_registry()
        for cat in IssueCategory:
            cat_obj = reg.get_by_category(cat)
            assert isinstance(cat_obj, OperationCategory)
            assert cat_obj.name == cat.value


# ===================================================================
# Part 3: workflow adjacency priming (audit 3.3)
# ===================================================================


class AdjacencyRecordingOperation(MockOperation):
    """MockOperation recording whether adjacency was primed at execute()."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.adjacency_seen = None

    def execute(self, mesh):
        self.adjacency_seen = mesh._adjacency_cache.get("edge_face")
        return []


class TestWorkflowAdjacencyPriming:
    """create_workflow must prime adjacency for requires_adjacency ops."""

    def test_requires_adjacency_primes_before_run(self):
        reg = OperationRegistry()
        op = AdjacencyRecordingOperation(
            op_id="adj.needs", requires_adjacency=True
        )
        reg.register(op)
        mesh = _make_mesh()

        result = reg.create_workflow(["adj.needs"])(mesh)

        assert isinstance(result, InspectionResult)
        # Adjacency was available BEFORE execute() ran.
        assert op.adjacency_seen is not None
        assert op.adjacency_seen is mesh.get_edge_face_adjacency()

    def test_no_adjacency_flag_does_not_prime(self):
        reg = OperationRegistry()
        op = AdjacencyRecordingOperation(
            op_id="adj.skip", requires_adjacency=False
        )
        reg.register(op)
        mesh = _make_mesh()

        reg.create_workflow(["adj.skip"])(mesh)

        assert op.adjacency_seen is None

    def test_adjacency_shared_across_multiple_hungry_ops(self):
        """Multiple adjacency-hungry ops see the SAME memoized map."""
        reg = OperationRegistry()
        op1 = AdjacencyRecordingOperation(
            op_id="adj.one", requires_adjacency=True
        )
        op2 = AdjacencyRecordingOperation(
            op_id="adj.two", requires_adjacency=True
        )
        reg.register(op1)
        reg.register(op2)
        mesh = _make_mesh()

        reg.create_workflow(["adj.one", "adj.two"])(mesh)

        assert op1.adjacency_seen is not None
        assert op1.adjacency_seen is op2.adjacency_seen


# ===================================================================
# Part 4: reset_registry (audit 3.1)
# ===================================================================


class TestResetRegistry:
    """reset_registry swaps in a fresh global registry and re-resolves
    the lazy accessor. The original global is restored afterwards so
    other tests in the session keep the built-in operations."""

    def test_reset_rebinds_global_and_lazy_accessor(self):
        old = operation_registry_module.registry
        try:
            new = reset_registry()

            assert isinstance(new, OperationRegistry)
            assert new is not old
            assert operation_registry_module.registry is new
            # Lazy accessor (used by base.register_operation) re-resolves.
            assert get_registry() is new
            assert _registry_module.get_registry() is new
            # Fresh registry starts empty; old instance is untouched.
            assert len(new) == 0
            assert "mock.op" not in old  # sanity: old still usable

            # New registrations land in the new registry only.
            register_operation(MockOperation(op_id="reset.op"))
            assert "reset.op" in new
            assert "reset.op" not in old

            # Servers constructed afterwards pick up the new registry.
            server = MeshIssueExaminerServer(auto_register_operations=False)
            assert server.registry is new
        finally:
            operation_registry_module.registry = old
            _registry_module._clear_cache()
        assert get_registry() is old
        assert operation_registry_module.registry is old


# ===================================================================
# Part 5: registry injection into MeshIssueExaminerServer (audit 3.1)
# ===================================================================


class TestMeshIssueExaminerServerRegistryInjection:
    """MeshIssueExaminerServer must honor an injected registry."""

    def test_default_registry_is_current_global(self):
        server = MeshIssueExaminerServer(auto_register_operations=False)
        assert server.registry is operation_registry_module.registry

    def test_injected_registry_used_for_inspect(self):
        reg = OperationRegistry()
        reg.register(MockOperation(op_id="injected.op"))
        server = MeshIssueExaminerServer(
            auto_register_operations=False, registry=reg
        )
        assert server.registry is reg
        assert server.registry is not operation_registry_module.registry

        server._mesh_cache["m"] = _make_mesh()
        req = MCPRequest(
            method="mesh_inspect",
            params={"mesh_id": "m", "operations": ["injected.op"]},
        )
        resp = asyncio.run(server.handle_request(req))
        assert resp.error is None
        assert resp.result["summary"]["total_issues"] == 0

    def test_injected_registry_unknown_operation_rejected(self):
        """Validation runs against the injected registry, not the global."""
        reg = OperationRegistry()  # empty: knows no built-in ids
        server = MeshIssueExaminerServer(
            auto_register_operations=False, registry=reg
        )
        server._mesh_cache["m"] = _make_mesh()
        req = MCPRequest(
            method="mesh_inspect",
            params={"mesh_id": "m", "operations": ["no.such.op"]},
        )
        resp = asyncio.run(server.handle_request(req))
        assert resp.error is not None
        assert "Unknown operation" in resp.error["message"]
