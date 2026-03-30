"""
Unit tests for mesh adapters.

Tests parsing, validation, and edge-case handling in the adapter layer
without requiring actual third-party libraries (pyvista, trimesh, pymeshlab).

Generalized test scope:
  - Adapter instantiation and interface compliance
  - get_faces parsing with malformed / edge-case inputs
  - Round-trip create→get consistency
  - Defensive behavior on invalid data

Regression guard for Bug #4 (missing n_verts validation), but also covers
the broader class of "adapter silently corrupts data on edge-case input."
"""

import numpy as np
import pytest

from adapters.pyvista_adapter import PyVistaAdapter
from adapters.base_adapter import MeshAdapter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakePolyData:
    """Minimal stand-in for pyvista.PolyData exposing only .faces."""

    def __init__(self, faces_array):
        self.faces = np.array(faces_array, dtype=np.int64)


# ---------------------------------------------------------------------------
# Adapter interface compliance
# ---------------------------------------------------------------------------


class TestAdapterInterface:
    """Verify that all adapters satisfy the MeshAdapter ABC contract."""

    def test_pyvista_adapter_is_mesh_adapter_subclass(self):
        assert issubclass(PyVistaAdapter, MeshAdapter)

    def test_pyvista_adapter_library_name(self):
        adapter = PyVistaAdapter()
        assert adapter.library_name == "pyvista"

    def test_pyvista_adapter_has_all_required_methods(self):
        """Every abstract method on MeshAdapter must be implemented."""
        required = [
            "library_name", "load_mesh", "create_mesh", "get_vertices",
            "get_faces", "is_watertight", "is_manifold", "get_boundary_edges",
            "get_nonmanifold_edges", "get_duplicate_vertices",
            "compute_face_areas", "compute_face_normals",
            "detect_self_intersections", "compute_cell_quality",
        ]
        adapter = PyVistaAdapter()
        for method_name in required:
            assert hasattr(adapter, method_name), (
                f"PyVistaAdapter missing method: {method_name}"
            )


# ---------------------------------------------------------------------------
# PyVistaAdapter.get_faces — parsing correctness
# ---------------------------------------------------------------------------


class TestPyVistaGetFacesCorrectness:
    """Verify get_faces correctly decompresses PyVista's packed face array."""

    def setup_method(self):
        self.adapter = PyVistaAdapter()

    def test_single_triangle(self):
        mesh = _FakePolyData([3, 10, 20, 30])
        faces = self.adapter.get_faces(mesh)
        assert faces.shape == (1, 3)
        np.testing.assert_array_equal(faces[0], [10, 20, 30])

    def test_two_triangles(self):
        mesh = _FakePolyData([3, 0, 1, 2, 3, 3, 4, 5])
        faces = self.adapter.get_faces(mesh)
        assert faces.shape == (2, 3)
        np.testing.assert_array_equal(faces[0], [0, 1, 2])
        np.testing.assert_array_equal(faces[1], [3, 4, 5])

    def test_many_triangles(self):
        """Parsing 100 triangles should produce correct shapes."""
        flat = []
        expected = []
        for i in range(100):
            v = [i * 3, i * 3 + 1, i * 3 + 2]
            flat.extend([3] + v)
            expected.append(v)
        mesh = _FakePolyData(flat)
        faces = self.adapter.get_faces(mesh)
        assert faces.shape == (100, 3)
        np.testing.assert_array_equal(faces, expected)

    def test_quad_faces(self):
        """Four-vertex faces (quads) should be decompressed correctly."""
        mesh = _FakePolyData([4, 0, 1, 2, 3])
        faces = self.adapter.get_faces(mesh)
        assert faces.shape == (1, 4)
        np.testing.assert_array_equal(faces[0], [0, 1, 2, 3])

    def test_empty_mesh(self):
        mesh = _FakePolyData([])
        faces = self.adapter.get_faces(mesh)
        assert isinstance(faces, np.ndarray)
        assert len(faces) == 0


# ---------------------------------------------------------------------------
# PyVistaAdapter.get_faces — malformed input handling
#
# This is the generalization of Bug #4: any invalid n_verts value must not
# cause silent data corruption, infinite loops, or crashes.
# ---------------------------------------------------------------------------


class TestPyVistaGetFacesMalformedInput:
    """Verify get_faces handles corrupt or edge-case face arrays gracefully."""

    def setup_method(self):
        self.adapter = PyVistaAdapter()

    def test_zero_nverts_stops_cleanly(self):
        """n_verts=0 should stop parsing without error or infinite loop."""
        mesh = _FakePolyData([3, 0, 1, 2, 0])
        faces = self.adapter.get_faces(mesh)
        assert faces.shape[0] == 1
        np.testing.assert_array_equal(faces[0], [0, 1, 2])

    def test_negative_nverts_stops_cleanly(self):
        """n_verts < 0 should stop parsing without error."""
        mesh = _FakePolyData([3, 0, 1, 2, -1, 5, 6, 7])
        faces = self.adapter.get_faces(mesh)
        assert faces.shape[0] == 1

    def test_truncated_face_array_raises(self):
        """If the array ends mid-face, get_faces raises ValueError.

        The n_verts count says 3 but only 1 vertex follows before the
        array ends. The partial face [10] has different length than the
        first face [0,1,2], so np.array() raises ValueError due to
        inhomogeneous shape. This documents the actual behavior: the
        adapter does not currently handle truncated data gracefully.
        """
        mesh = _FakePolyData([3, 0, 1, 2, 3, 10])  # second face truncated
        with pytest.raises(ValueError):
            self.adapter.get_faces(mesh)

    def test_result_faces_are_independent_of_source(self):
        """Modifying the returned array must not mutate the source mesh.faces.

        Guards against returning numpy views instead of copies.
        """
        mesh = _FakePolyData([3, 0, 1, 2])
        faces = self.adapter.get_faces(mesh)
        original = mesh.faces.copy()
        faces[0][0] = 999
        np.testing.assert_array_equal(mesh.faces, original)


# ---------------------------------------------------------------------------
# PyVistaAdapter.create_mesh → get_faces round-trip
# ---------------------------------------------------------------------------


class TestPyVistaRoundTrip:
    """Verify create_mesh → get_faces round-trip consistency.

    If pyvista is installed, create a mesh and verify get_faces recovers
    the original faces. If pyvista is not installed, skip.
    """

    def test_round_trip_triangle_mesh(self):
        pytest.importorskip("pyvista")
        import pyvista as pv

        adapter = PyVistaAdapter()
        verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
        faces = np.array([[0, 1, 2]])

        pv_mesh = adapter.create_mesh(verts, faces)
        recovered = adapter.get_faces(pv_mesh)
        np.testing.assert_array_equal(recovered, faces)

    def test_round_trip_multi_face_mesh(self):
        pytest.importorskip("pyvista")
        import pyvista as pv

        adapter = PyVistaAdapter()
        verts = np.array([
            [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]
        ], dtype=float)
        faces = np.array([[0, 1, 2], [0, 1, 3], [1, 2, 3]])

        pv_mesh = adapter.create_mesh(verts, faces)
        recovered = adapter.get_faces(pv_mesh)
        np.testing.assert_array_equal(recovered, faces)
