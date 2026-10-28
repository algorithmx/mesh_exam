"""
Unit tests for material loading via ``load_mesh_from_file``.

Covers the ``material_array_name`` plumbing and, critically, the alignment
between per-cell data arrays (grid cell order) and the canonical
``volume_cells`` enumeration (grouped by element type in order of first
occurrence).  Misalignment silently attributes materials to the wrong
cells, which is why it is pinned here explicitly.
"""

import numpy as np
import pytest

from mesh_issue_examiner.utils.mesh_loader import load_mesh_from_file


def _vtk_type(element_type: str) -> int:
    from mesh_issue_examiner.utils.cell_types import get_cell_type_map
    return get_cell_type_map()[element_type]


def _write_interleaved_grid(path, materials) -> None:
    """Interleaved [hex, tet, hex, tet] grid with per-cell material data.

    hex1 = unit cube at origin, tet1 = tet at x=10, hex2 = cube at x=20,
    tet2 = tet at x=30.  The grid cell order interleaves element types on
    purpose: a naive grid-order material array would NOT match the
    canonical per-type volume_cells enumeration.
    """
    import pyvista as pv

    cube_at = lambda o: [
        [o, 0, 0], [o + 1, 0, 0], [o + 1, 1, 0], [o, 1, 0],
        [o, 0, 1], [o + 1, 0, 1], [o + 1, 1, 1], [o, 1, 1],
    ]
    tet_at = lambda o: [[o, 0, 0], [o + 1, 0, 0], [o, 1, 0], [o, 0, 1]]

    vertices = np.array(
        cube_at(0.0) + tet_at(10.0) + cube_at(20.0) + tet_at(30.0), float
    )
    conn = [
        list(range(0, 8)),      # hex1 (grid cell 0)
        list(range(8, 12)),     # tet1 (grid cell 1)
        list(range(12, 20)),    # hex2 (grid cell 2)
        list(range(20, 24)),    # tet2 (grid cell 3)
    ]
    counts = [len(c) for c in conn]
    cells = np.concatenate(
        [np.r_[n, np.asarray(c, dtype=np.int64)] for n, c in zip(counts, conn)]
    )
    celltypes = np.array(
        [_vtk_type("hex8"), _vtk_type("tet4"),
         _vtk_type("hex8"), _vtk_type("tet4")], dtype=np.uint8
    )
    grid = pv.UnstructuredGrid(cells, celltypes, vertices)
    grid.cell_data["material"] = np.asarray(materials)
    grid.save(str(path))


class TestMaterialArrayAlignment:

    def test_interleaved_grid_reordered_to_block_order(self, tmp_path):
        """Grid-order materials must be remapped onto the per-type
        volume_cells enumeration: hex block first, then tet block."""
        path = tmp_path / "interleaved.vtu"
        _write_interleaved_grid(path, materials=[10, 20, 11, 21])

        mesh = load_mesh_from_file(str(path), material_array_name="material")

        assert mesh.has_volume_cells
        # First-occurrence order: hex8 seen at grid cell 0, tet4 at cell 1
        assert mesh.cell_types == ["hex8", "tet4"]
        assert mesh.n_volume_cells == 4
        # Grid order would be [10, 20, 11, 21]; block order is [10, 11, 20, 21]
        np.testing.assert_array_equal(mesh.cell_materials, [10, 11, 20, 21])

    def test_block_alignment_via_geometry(self, tmp_path):
        """Independent geometric check: the material of the tet at x=30
        (grid cell 3, material 21) must follow that cell into the tet4
        block's second slot regardless of enumeration assumptions."""
        path = tmp_path / "interleaved.vtu"
        _write_interleaved_grid(path, materials=[10, 20, 11, 21])

        mesh = load_mesh_from_file(str(path), material_array_name="material")

        tets = mesh.volume_cells["tet4"]
        assert tets.shape == (2, 4)
        # tet2's corner x-coordinates are 30..31 -> material 21
        x_of = lambda row: float(np.min(mesh.vertices[row][:, 0]))
        tet2_idx = 0 if x_of(tets[0]) > 25 else 1
        # tet4 block occupies global indices [2, 3]
        assert mesh.get_material_for_cell(2 + tet2_idx) == 21
        assert mesh.get_material_for_cell(2 + (1 - tet2_idx)) == 20

    def test_no_material_array_leaves_cells_unset(self, tmp_path):
        path = tmp_path / "interleaved.vtu"
        _write_interleaved_grid(path, materials=[10, 20, 11, 21])

        mesh = load_mesh_from_file(str(path))

        assert mesh.cell_materials is None
        assert not mesh.has_multiple_materials()

    def test_unknown_array_name_is_safe(self, tmp_path):
        path = tmp_path / "interleaved.vtu"
        _write_interleaved_grid(path, materials=[1, 1, 1, 1])

        mesh = load_mesh_from_file(str(path), material_array_name="nope")

        assert mesh.cell_materials is None


class TestNastranMaterialExtraction:

    def test_psolid_materials_follow_element_order(self, tmp_path):
        """PSOLID PID->MID mapping produces cell_materials aligned with the
        parser's element order (ascending EID in this deck)."""
        deck = """$ two-material tet deck
BEGIN BULK
GRID,1,0,0.0,0.0,0.0
GRID,2,0,1.0,0.0,0.0
GRID,3,0,0.0,1.0,0.0
GRID,4,0,0.0,0.0,1.0
GRID,5,0,5.0,0.0,0.0
GRID,6,0,6.0,0.0,0.0
GRID,7,0,5.0,1.0,0.0
GRID,8,0,5.0,0.0,1.0
PSOLID,10,100
PSOLID,11,200
CTETRA,1,10,1,2,3,4
CTETRA,2,11,5,6,7,8
ENDDATA
"""
        path = tmp_path / "two_mat.nas"
        path.write_text(deck, encoding="ascii")

        mesh = load_mesh_from_file(str(path), material_array_name="material")

        assert mesh.has_volume_cells
        assert mesh.cell_types == ["tet4"]
        np.testing.assert_array_equal(mesh.cell_materials, [100, 200])
        assert mesh.material_names == {100: "Material_100", 200: "Material_200"}
        assert mesh.has_multiple_materials()
        assert mesh.validate_materials() == []

    def test_deck_without_psolid_leaves_cells_unset(self, tmp_path):
        deck = """BEGIN BULK
GRID,1,0,0.0,0.0,0.0
GRID,2,0,1.0,0.0,0.0
GRID,3,0,0.0,1.0,0.0
GRID,4,0,0.0,0.0,1.0
CTETRA,1,1,1,2,3,4
ENDDATA
"""
        path = tmp_path / "no_psolid.nas"
        path.write_text(deck, encoding="ascii")

        mesh = load_mesh_from_file(str(path), material_array_name="material")

        assert mesh.has_volume_cells
        assert mesh.cell_materials is None
