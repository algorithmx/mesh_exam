"""
Thread-safety tests for core.models.MeshInput and operations.base (audit 6.1).

Covers:
- Concurrent get_native() across threads and libraries
- Per-instance native cache eviction under contention (_MAX_CACHE_SIZE)
- invalidate_cache() racing get_native()
- Concurrent BaseOperation.run() sharing one MeshInput
- The process-wide native-object LRU tracker (aggregate bound, LRU
  eviction order, weakref cleanup of dead entries)
- Concurrent get_edge_face_adjacency() memoization + invalidation

Uses monkeypatched dummy adapters so no heavy mesh libraries
(pyvista/trimesh/pymeshlab) are imported.
"""

import gc
import threading

import numpy as np
import pytest

from mesh_issue_examiner.core import models
from mesh_issue_examiner.core.models import IssueCategory, MeshInput
from mesh_issue_examiner.operations.base import BaseOperation, OperationMetadata


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mesh(mesh_id="thread_test"):
    """Tetrahedron surface mesh: 4 faces, 6 edges, each shared by 2 faces."""
    return MeshInput(
        vertices=np.array(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float
        ),
        faces=np.array([[0, 1, 2], [0, 1, 3], [1, 2, 3], [0, 2, 3]]),
        mesh_id=mesh_id,
    )


class DummyAdapter:
    """Minimal adapter stand-in: from_canonical returns a tagged string."""

    def __init__(self, tag):
        self.tag = tag

    def from_canonical(self, mesh):
        return f"native::{self.tag}::{mesh.mesh_id}"


@pytest.fixture
def dummy_factory(monkeypatch):
    """Patch adapters.factory.get_adapter to serve DummyAdapter instances.

    core.models imports ``get_adapter`` lazily inside get_native(), so
    patching the factory module attribute takes effect immediately.

    Returns:
        The dict of created dummy adapters (library -> DummyAdapter).
    """
    from mesh_issue_examiner.adapters import factory

    created = {}

    def fake_get_adapter(library):
        if library not in created:
            created[library] = DummyAdapter(library)
        return created[library]

    monkeypatch.setattr(factory, "get_adapter", fake_get_adapter)
    return created


def _run_threads(targets):
    """Run zero-arg callables as threads and assert none raise.

    Args:
        targets: list of zero-argument callables, one per thread.
    """
    errors = []

    def wrap(fn):
        def run():
            try:
                fn()
            except Exception as exc:  # pragma: no cover - failure path
                errors.append(exc)
        return run

    threads = [threading.Thread(target=wrap(fn)) for fn in targets]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
        assert not t.is_alive(), "worker thread hung"
    assert not errors, f"worker exceptions: {errors!r}"


# ---------------------------------------------------------------------------
# 1. Concurrent get_native() across ~8 threads x several libraries
# ---------------------------------------------------------------------------

class TestConcurrentGetNative:
    """Concurrent get_native() must be exception-free and per-library exact."""

    def test_no_exceptions_and_correct_object_per_library(self, dummy_factory):
        mesh = _make_mesh()
        libraries = ["libA", "libB", "libC", "libD"]

        def make_worker():
            def worker():
                for _ in range(200):
                    for lib in libraries:
                        obj = mesh.get_native(lib)
                        assert obj == f"native::{lib}::{mesh.mesh_id}"
            return worker

        _run_threads([make_worker() for _ in range(8)])


# ---------------------------------------------------------------------------
# 2. Eviction stress: more libraries than _MAX_CACHE_SIZE, concurrently
# ---------------------------------------------------------------------------

class TestPerInstanceCacheEvictionUnderContention:
    """Per-instance native cache must respect _MAX_CACHE_SIZE under races."""

    def test_cache_respects_max_size_concurrently(self, dummy_factory):
        assert MeshInput._MAX_CACHE_SIZE < 12
        mesh = _make_mesh()
        libraries = [f"stress{i}" for i in range(12)]

        def make_worker(i):
            def worker():
                for _ in range(50):
                    mesh.get_native(libraries[i % len(libraries)])
            return worker

        _run_threads([make_worker(i) for i in range(8)])
        assert len(mesh._native_objects) <= MeshInput._MAX_CACHE_SIZE


# ---------------------------------------------------------------------------
# 3. invalidate_cache() racing get_native()
# ---------------------------------------------------------------------------

class TestInvalidateRacingGetNative:
    """invalidate_cache() racing get_native() must never raise."""

    def test_no_exceptions(self, dummy_factory):
        mesh = _make_mesh()

        def reader():
            for _ in range(400):
                mesh.get_native("race_lib")

        def invalidator():
            for _ in range(400):
                mesh.invalidate_cache()

        _run_threads([invalidator, reader, reader, reader])


# ---------------------------------------------------------------------------
# 4. Concurrent BaseOperation.run() sharing one MeshInput
# ---------------------------------------------------------------------------

class TrivialOperation(BaseOperation):
    """No-op operation that touches the shared mesh's native cache."""

    @property
    def metadata(self):
        return OperationMetadata(
            operation_id="thread.noop",
            display_name="Trivial",
            description="Touches the native cache, returns no issues",
            category=IssueCategory.TOPOLOGICAL,
            supported_libraries=["dummy"],
            recommended_library="dummy",
        )

    def execute(self, mesh):
        mesh.get_native("op_lib")
        return []


class TestConcurrentOperationRun:
    """Concurrent BaseOperation.run() on a shared mesh must not raise."""

    def test_shared_mesh_concurrent_runs(self, dummy_factory):
        mesh = _make_mesh()
        op = TrivialOperation()

        def make_worker():
            def worker():
                for _ in range(100):
                    assert op.run(mesh) == []
            return worker

        _run_threads([make_worker() for _ in range(8)])


# ---------------------------------------------------------------------------
# 5. Global native-object LRU tracker
# ---------------------------------------------------------------------------

class TestGlobalNativeCacheLimit:
    """Aggregate bound: LRU eviction + weakref cleanup of dead entries."""

    def test_lru_eviction_and_weakref_cleanup(self, dummy_factory, monkeypatch):
        monkeypatch.setattr(MeshInput, "GLOBAL_CACHE_LIMIT", 2)

        # Neutralize native caches of meshes still alive from earlier
        # tests in this session so aggregate accounting is deterministic.
        with models._global_mesh_lru_lock:
            leftovers = [ref() for ref in list(models._global_mesh_lru.values())]
        for leftover in leftovers:
            if leftover is not None:
                leftover.invalidate_cache()

        meshes = []
        for i in range(5):
            m = _make_mesh(f"lru{i}")
            meshes.append(m)
            m.get_native("glib")  # construction triggers enforcement

        # Aggregate cached natives stay within the global limit...
        total = sum(len(m._native_objects) for m in meshes)
        assert total <= MeshInput.GLOBAL_CACHE_LIMIT
        # ...the least-recently-used mesh (meshes[0]) was evicted...
        assert len(meshes[0]._native_objects) == 0
        # ...while the most recently used meshes keep their cache.
        assert len(meshes[3]._native_objects) == 1
        assert len(meshes[4]._native_objects) == 1

        # Weakref cleanup: drop every reference; the tracker must not
        # hold dead entries afterwards.
        keys = [id(m) for m in meshes]
        del meshes, leftovers, m
        gc.collect()
        with models._global_mesh_lru_lock:
            dead = [
                k for k, ref in models._global_mesh_lru.items()
                if ref() is None
            ]
            tracked = list(models._global_mesh_lru.keys())
        assert dead == []
        assert all(k not in tracked for k in keys)


# ---------------------------------------------------------------------------
# 6. get_edge_face_adjacency() concurrency + invalidation
# ---------------------------------------------------------------------------

class TestAdjacencyConcurrency:
    """Adjacency memoization must be race-free and invalidated on mutation."""

    def test_concurrent_calls_return_same_object(self):
        mesh = _make_mesh()
        results = [None] * 8

        def make_worker(i):
            def worker():
                results[i] = mesh.get_edge_face_adjacency()
            return worker

        _run_threads([make_worker(i) for i in range(8)])
        first = results[0]
        assert all(r is first for r in results)
        # Tetrahedron surface: 6 unique edges, each incident to 2 faces.
        assert len(first) == 6
        assert all(len(faces) == 2 for faces in first.values())

    def test_faces_mutation_invalidates_adjacency(self):
        mesh = _make_mesh()
        adj1 = mesh.get_edge_face_adjacency()
        assert mesh._adjacency_cache["edge_face"] is adj1

        mesh.faces = np.array([[0, 1, 2]])
        assert "edge_face" not in mesh._adjacency_cache

        adj2 = mesh.get_edge_face_adjacency()
        assert adj2 is not adj1
        assert set(adj2) == {(0, 1), (0, 2), (1, 2)}
