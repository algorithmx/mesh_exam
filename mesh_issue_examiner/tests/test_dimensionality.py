"""
Tests for utils/dimensionality.py — mesh dimensionality detection.
"""

import pytest
import numpy as np

from mesh_issue_examiner.core.models import MeshDimensionality, MeshInput
from mesh_issue_examiner.utils.dimensionality import (
    detect_dimensionality,
    detect_mesh_dimensionality,
    _check_coplanarity,
)


# ---------------------------------------------------------------------------
# MeshDimensionality enum
# ---------------------------------------------------------------------------

class TestMeshDimensionalityEnum:
    """Tests for the MeshDimensionality enum."""

    def test_values(self):
        assert MeshDimensionality.PURE_2D.value == "pure_2d"
        assert MeshDimensionality.SURFACE_2D.value == "surface_2d"
        assert MeshDimensionality.VOLUMETRIC_3D.value == "volumetric_3d"
        assert MeshDimensionality.MIXED_2D_3D.value == "mixed_2d_3d"

    def test_member_count(self):
        assert len(MeshDimensionality) == 4


# ---------------------------------------------------------------------------
# _check_coplanarity (tested indirectly via detect_dimensionality mostly,
# but also directly for edge cases)
# ---------------------------------------------------------------------------

class TestCheckCoplanarity:

    def test_planar_xy_vertices(self):
        verts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        assert _check_coplanarity(verts, 1e-10) is True

    def test_unit_cube_not_planar(self):
        verts = np.array([
            [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
        ])
        assert _check_coplanarity(verts, 1e-10) is False

    def test_single_vertex(self):
        assert _check_coplanarity(np.array([[0.0, 0.0, 0.0]]), 1e-10) is True

    def test_two_vertices(self):
        verts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        assert _check_coplanarity(verts, 1e-10) is True

    def test_tiny_z_perturbation(self):
        verts = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 1e-12],
        ])
        assert _check_coplanarity(verts, 1e-10) is True

    def test_large_perturbation_not_planar(self):
        verts = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        assert _check_coplanarity(verts, 1e-10) is False


# ---------------------------------------------------------------------------
# detect_dimensionality
# ---------------------------------------------------------------------------

class TestDetectDimensionality:

    def _planar_triangles(self):
        verts = np.array([
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ])
        faces = np.array([[0, 1, 2], [1, 3, 2]])
        return verts, faces

    def _cube_surface(self):
        verts = np.array([
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
        ], dtype=float)
        faces = np.array([
            [0, 1, 2], [0, 2, 3],
            [4, 6, 5], [4, 7, 6],
            [0, 4, 5], [0, 5, 1],
            [2, 6, 7], [2, 7, 3],
            [0, 3, 7], [0, 7, 4],
            [1, 5, 6], [1, 6, 2],
        ])
        return verts, faces

    def test_planar_surface_is_pure_2d(self):
        verts, faces = self._planar_triangles()
        result = detect_dimensionality(verts, surface_cells=faces)
        assert result == MeshDimensionality.PURE_2D

    def test_cube_surface_is_surface_2d(self):
        verts, faces = self._cube_surface()
        result = detect_dimensionality(verts, surface_cells=faces)
        assert result == MeshDimensionality.SURFACE_2D

    def test_volume_only_is_volumetric_3d(self):
        verts = np.array([
            [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
        ], dtype=float)
        tets = np.array([[0, 1, 2, 3]])
        result = detect_dimensionality(verts, volume_cells=tets)
        assert result == MeshDimensionality.VOLUMETRIC_3D

    def test_mixed_surface_and_volume(self):
        verts = np.array([
            [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0],
        ], dtype=float)
        faces = np.array([[0, 1, 4]])
        tets = np.array([[0, 1, 2, 3]])
        result = detect_dimensionality(
            verts, surface_cells=faces, volume_cells=tets
        )
        assert result == MeshDimensionality.MIXED_2D_3D

    def test_bare_planar_vertices_is_pure_2d(self):
        verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        result = detect_dimensionality(verts)
        assert result == MeshDimensionality.PURE_2D

    def test_bare_nonplanar_vertices_is_surface_2d(self):
        verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
        result = detect_dimensionality(verts)
        assert result == MeshDimensionality.SURFACE_2D

    def test_empty_surface_cells_treated_as_none(self):
        verts, _ = self._cube_surface()
        empty_faces = np.array([]).reshape(0, 3)
        result = detect_dimensionality(verts, surface_cells=empty_faces)
        assert result == MeshDimensionality.SURFACE_2D

    def test_empty_volume_cells_treated_as_none(self):
        verts, faces = self._cube_surface()
        empty_tets = np.array([]).reshape(0, 4)
        result = detect_dimensionality(
            verts, surface_cells=faces, volume_cells=empty_tets
        )
        assert result == MeshDimensionality.SURFACE_2D

    def test_raises_on_empty_vertices(self):
        with pytest.raises(ValueError, match="must not be empty"):
            detect_dimensionality(np.array([]).reshape(0, 3))

    def test_raises_on_wrong_shape(self):
        with pytest.raises(ValueError, match="Nx3"):
            detect_dimensionality(np.array([[1.0, 2.0]]))

    def test_raises_on_non_numpy(self):
        with pytest.raises(ValueError, match="numpy array"):
            detect_dimensionality([[0, 0, 0], [1, 0, 0]])

    def test_custom_coplanarity_threshold(self):
        verts = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1e-6],
        ])
        # Strict threshold: not planar
        assert detect_dimensionality(
            verts, coplanarity_threshold=1e-12
        ) == MeshDimensionality.SURFACE_2D
        # Loose threshold: planar
        assert detect_dimensionality(
            verts, coplanarity_threshold=1e-3
        ) == MeshDimensionality.PURE_2D


# ---------------------------------------------------------------------------
# detect_mesh_dimensionality (MeshInput convenience wrapper)
# ---------------------------------------------------------------------------

class TestDetectMeshDimensionality:

    def test_planar_meshinput(self):
        verts = np.array([
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0], [1.0, 1.0, 0.0],
        ])
        faces = np.array([[0, 1, 2], [1, 3, 2]])
        mesh = MeshInput(vertices=verts, faces=faces)
        assert detect_mesh_dimensionality(mesh) == MeshDimensionality.PURE_2D

    def test_cube_meshinput(self):
        verts = np.array([
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
        ], dtype=float)
        faces = np.array([
            [0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6],
            [0, 4, 5], [0, 5, 1], [2, 6, 7], [2, 7, 3],
            [0, 3, 7], [0, 7, 4], [1, 5, 6], [1, 6, 2],
        ])
        mesh = MeshInput(vertices=verts, faces=faces)
        assert detect_mesh_dimensionality(mesh) == MeshDimensionality.SURFACE_2D

    def test_propagates_threshold(self):
        verts = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1e-6],
        ])
        faces = np.array([[0, 1, 2], [0, 1, 3]])
        mesh = MeshInput(vertices=verts, faces=faces)
        assert detect_mesh_dimensionality(
            mesh, coplanarity_threshold=1e-3
        ) == MeshDimensionality.PURE_2D
