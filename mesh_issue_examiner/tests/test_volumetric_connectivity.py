"""
Unit tests for volumetric connectivity validation operations:

- topology_3d.detect_duplicate_cells
- topology_3d.detect_orphaned_cells
- topology_3d.check_boundary_closure

All meshes are constructed in-process (MeshInput with volume_cells);
orphaned-cell meshes cannot be serialized to any on-disk format because
the invalid indices are exactly the defect under test.
"""

import numpy as np
import pytest

from mesh_issue_examiner.core.models import MeshInput, IssueSeverity
from mesh_issue_examiner.operations.operation_registry import registry

EMPTY_FACES = np.zeros((0, 3), dtype=np.int64)


# ---------------------------------------------------------------------------
# Mesh builders
# ---------------------------------------------------------------------------

def _hex_grid_mesh(nx: int, ny: int, nz: int, mesh_id: str = "grid") -> MeshInput:
    """Structured hex grid with VTK hex8 node ordering (corners first)."""
    vertices = np.array(
        [
            (x, y, z)
            for z in range(nz + 1)
            for y in range(ny + 1)
            for x in range(nx + 1)
        ],
        dtype=float,
    )

    def vid(i, j, k):
        return k * (ny + 1) * (nx + 1) + j * (nx + 1) + i

    cells = []
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                cells.append([
                    vid(i, j, k), vid(i + 1, j, k),
                    vid(i + 1, j + 1, k), vid(i, j + 1, k),
                    vid(i, j, k + 1), vid(i + 1, j, k + 1),
                    vid(i + 1, j + 1, k + 1), vid(i, j + 1, k + 1),
                ])
    return MeshInput(
        vertices=vertices,
        faces=EMPTY_FACES,
        mesh_id=mesh_id,
        volume_cells={"hex8": np.array(cells, dtype=np.int64)},
    )


def _t_junction_mesh(mesh_id: str = "leaky") -> MeshInput:
    """Non-conforming block interface (T-junction) with shared vertices.

    Block A: [0,1]x[0,2]x[0,2] (big hex).
    Block B: [1,2]x[0,1]x[0,2] (hex covering HALF of A's right face).

    The blocks share the vertices (1,0,0) and (1,0,2) but no face keys
    match, so both interface faces are boundary faces and the shared
    boundary edge is used by 4 faces (non-manifold boundary edge): the
    volume does not bound a watertight solid.
    """
    vertices = np.array(
        [
            [0, 0, 0], [1, 0, 0], [1, 2, 0], [0, 2, 0],   # 0-3: A bottom
            [0, 0, 2], [1, 0, 2], [1, 2, 2], [0, 2, 2],   # 4-7: A top
            [2, 0, 0], [2, 1, 0], [1, 1, 0],              # 8-10: B bottom
            [2, 0, 2], [2, 1, 2], [1, 1, 2],              # 11-13: B top
        ],
        dtype=float,
    )
    cell_a = [0, 1, 2, 3, 4, 5, 6, 7]
    cell_b = [1, 8, 9, 10, 5, 11, 12, 13]
    return MeshInput(
        vertices=vertices,
        faces=EMPTY_FACES,
        mesh_id=mesh_id,
        volume_cells={"hex8": np.array([cell_a, cell_b], dtype=np.int64)},
    )


def _surface_only_mesh() -> MeshInput:
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
    faces = np.array([[0, 1, 2]])
    return MeshInput(vertices=vertices, faces=faces, mesh_id="surface_only")


# ---------------------------------------------------------------------------
# Registry presence
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_new_operations_registered(self):
        for op_id in (
            "topology_3d.detect_duplicate_cells",
            "topology_3d.detect_orphaned_cells",
            "topology_3d.check_boundary_closure",
        ):
            assert registry.get(op_id) is not None, op_id


# ---------------------------------------------------------------------------
# DuplicateCellDetection
# ---------------------------------------------------------------------------

class TestDuplicateCellDetection:

    def test_clean_grid_no_issues(self):
        op = registry.get("topology_3d.detect_duplicate_cells")
        assert op.run(_hex_grid_mesh(2, 1, 1)) == []

    def test_duplicate_cell_detected(self):
        mesh = _hex_grid_mesh(2, 1, 1)
        cells = mesh.volume_cells["hex8"]
        mesh = MeshInput(
            vertices=mesh.vertices,
            faces=EMPTY_FACES,
            mesh_id="dup",
            volume_cells={"hex8": np.vstack([cells, cells[0]])},
        )
        op = registry.get("topology_3d.detect_duplicate_cells")
        issues = op.run(mesh)
        assert len(issues) == 1
        m = issues[0].metrics
        assert m["duplicate_cell_count"] == 1
        assert m["total_cell_count"] == 3
        assert m["unique_cell_count"] == 2
        # 1/3 = 33% -> ERROR by standard severity thresholds
        assert issues[0].severity == IssueSeverity.ERROR
        # Location points at the duplicated (third) cell
        assert issues[0].locations[0].element_index == 2

    def test_duplicate_ignored_winding_order(self):
        """Reordered vertex lists still produce the same canonical key."""
        mesh = _hex_grid_mesh(1, 1, 1)
        cell = mesh.volume_cells["hex8"][0]
        reordered = np.array([cell[4], cell[5], cell[6], cell[7],
                              cell[0], cell[1], cell[2], cell[3]])
        mesh = MeshInput(
            vertices=mesh.vertices,
            faces=EMPTY_FACES,
            mesh_id="dup_wound",
            volume_cells={"hex8": np.vstack([cell, reordered])},
        )
        op = registry.get("topology_3d.detect_duplicate_cells")
        issues = op.run(mesh)
        assert len(issues) == 1
        assert issues[0].metrics["duplicate_cell_count"] == 1

    def test_surface_only_mesh_skipped(self):
        op = registry.get("topology_3d.detect_duplicate_cells")
        assert op.run(_surface_only_mesh()) == []


# ---------------------------------------------------------------------------
# OrphanedCellDetection
# ---------------------------------------------------------------------------

class TestOrphanedCellDetection:

    def test_clean_grid_no_issues(self):
        op = registry.get("topology_3d.detect_orphaned_cells")
        assert op.run(_hex_grid_mesh(2, 2, 2)) == []

    def test_out_of_range_index_detected(self):
        mesh = _hex_grid_mesh(1, 1, 1)
        n_vertices = mesh.n_vertices
        cells = mesh.volume_cells["hex8"].copy()
        cells[0][7] = n_vertices  # one past the last valid vertex index
        mesh = MeshInput(
            vertices=mesh.vertices,
            faces=EMPTY_FACES,
            mesh_id="orphan",
            volume_cells={"hex8": cells},
        )
        op = registry.get("topology_3d.detect_orphaned_cells")
        issues = op.run(mesh)
        assert len(issues) == 1
        assert issues[0].severity == IssueSeverity.CRITICAL
        assert issues[0].metrics["orphaned_cell_count"] == 1
        assert issues[0].metrics["n_vertices"] == n_vertices
        assert issues[0].locations[0].adjacent_elements["invalid_vertex_indices"] == [n_vertices]

    def test_negative_index_detected(self):
        mesh = _hex_grid_mesh(1, 1, 1)
        cells = mesh.volume_cells["hex8"].copy()
        cells[0][0] = -1
        mesh = MeshInput(
            vertices=mesh.vertices,
            faces=EMPTY_FACES,
            mesh_id="neg",
            volume_cells={"hex8": cells},
        )
        op = registry.get("topology_3d.detect_orphaned_cells")
        issues = op.run(mesh)
        assert len(issues) == 1
        assert issues[0].severity == IssueSeverity.CRITICAL

    def test_surface_only_mesh_skipped(self):
        op = registry.get("topology_3d.detect_orphaned_cells")
        assert op.run(_surface_only_mesh()) == []


# ---------------------------------------------------------------------------
# BoundaryClosureCheck
# ---------------------------------------------------------------------------

class TestBoundaryClosureCheck:

    def test_single_hex_is_watertight(self):
        op = registry.get("topology_3d.check_boundary_closure")
        assert op.run(_hex_grid_mesh(1, 1, 1)) == []

    def test_filled_grid_is_watertight(self):
        op = registry.get("topology_3d.check_boundary_closure")
        assert op.run(_hex_grid_mesh(2, 2, 2)) == []

    def test_grid_minus_corner_cell_still_closed(self):
        """Removing a corner cell reshapes the boundary but leaves it a
        closed surface -- this is NOT a leak by the edge-manifold test."""
        mesh = _hex_grid_mesh(2, 2, 2)
        cells = mesh.volume_cells["hex8"]
        # First cell of the grid iteration is the (0,0,0) corner cell.
        mesh = MeshInput(
            vertices=mesh.vertices,
            faces=EMPTY_FACES,
            mesh_id="minus_corner",
            volume_cells={"hex8": cells[1:]},
        )
        op = registry.get("topology_3d.check_boundary_closure")
        assert op.run(mesh) == []

    def test_t_junction_interface_detected(self):
        op = registry.get("topology_3d.check_boundary_closure")
        issues = op.run(_t_junction_mesh())
        assert len(issues) == 1
        m = issues[0].metrics
        assert m["is_watertight"] is False
        assert m["nonmanifold_boundary_edge_count"] >= 1
        assert issues[0].severity in (
            IssueSeverity.WARNING, IssueSeverity.ERROR, IssueSeverity.CRITICAL,
        )

    def test_surface_only_mesh_skipped(self):
        op = registry.get("topology_3d.check_boundary_closure")
        assert op.run(_surface_only_mesh()) == []
