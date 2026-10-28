"""
Unit tests for the native Gmsh .msh 2.2 ASCII parser (utils/gmsh_parser.py)
and the mesh_loader .msh routing (native first, meshio fallback).
"""

from pathlib import Path

import numpy as np
import pytest

from mesh_issue_examiner.utils.gmsh_parser import parse_msh


def _write_deck(tmp_path: Path, sections: str) -> str:
    path = tmp_path / "mesh.msh"
    path.write_text(sections, encoding="ascii")
    return str(path)


_HEADER = "$MeshFormat\n2.2 0 8\n$EndMeshFormat\n"

_UNIT_TET_NODES = (
    "$Nodes\n4\n"
    "1 0.0 0.0 0.0\n2 1.0 0.0 0.0\n3 0.0 1.0 0.0\n4 0.0 0.0 1.0\n"
    "$EndNodes\n"
)


class TestGmshParserBasics:
    def test_linear_mixed_parse(self, tmp_path):
        deck = (_HEADER
                + "$Nodes\n23\n"
                + "".join(
                    f"{i} {float(i)} 0.0 0.0\n" for i in range(1, 24))
                + "$EndNodes\n"
                + "$Elements\n4\n"
                + "1 4 2 1 1 1 2 3 4\n"          # tet4
                + "2 5 2 1 1 5 6 7 8 9 10 11 12\n"  # hex8
                + "3 6 2 1 1 13 14 15 16 17 18\n"   # prism6
                + "4 7 2 1 1 19 20 21 22 23\n"      # pyramid5
                + "$EndElements\n")
        vertices, faces, cells, mats, names = parse_msh(
            _write_deck(tmp_path, deck))

        assert vertices.shape == (23, 3)
        assert list(cells) == ["tet4", "hex8", "prism6", "pyramid5"]
        assert cells["tet4"].shape == (1, 4)
        assert cells["hex8"].shape == (1, 8)
        assert cells["prism6"].shape == (1, 6)
        assert cells["pyramid5"].shape == (1, 5)
        # Connectivity remapped to sorted 0-based indices.
        np.testing.assert_array_equal(cells["tet4"][0], [0, 1, 2, 3])
        # Boundary surfaces: tet 4 + hex 12 + prism 8 + pyramid 6 faces.
        assert faces.shape == (4 + 12 + 8 + 6, 3)
        np.testing.assert_array_equal(mats, [1, 1, 1, 1])

    def test_stacked_tets_share_interior_face(self, tmp_path):
        deck = (_HEADER
                + "$Nodes\n5\n"
                + "1 0.0 0.0 0.0\n2 1.0 0.0 0.0\n3 0.0 1.0 0.0\n"
                + "4 0.0 0.0 1.0\n5 0.0 0.0 -1.0\n"
                + "$EndNodes\n"
                + "$Elements\n2\n"
                + "1 4 2 1 1 1 2 3 4\n"
                + "2 4 2 1 1 5 2 3 4\n"
                + "$EndElements\n")
        _, faces, cells, _, _ = parse_msh(_write_deck(tmp_path, deck))

        assert cells["tet4"].shape == (2, 4)
        # 8 face instances; both copies of the shared (2,3,4) face are
        # internal -> 6 boundary faces (bipyramid surface).
        assert faces.shape == (6, 3)

    def test_quad_surface_triangulated(self, tmp_path):
        deck = (_HEADER
                + _UNIT_TET_NODES
                + "$Elements\n1\n"
                + "1 3 2 1 1 1 2 3 4\n"
                + "$EndElements\n")
        _, faces, cells, _, _ = parse_msh(_write_deck(tmp_path, deck))

        assert cells == {}
        assert faces.shape == (2, 3)
        np.testing.assert_array_equal(faces, [[0, 1, 2], [0, 2, 3]])


class TestGmshQuadraticOrdering:
    """Gmsh node order -> canonical (VTK) order permutations."""

    def test_tet10_midside_permutation(self, tmp_path):
        # Gmsh midside order: (0,1),(1,2),(0,2),(0,3),(2,3),(1,3) — the
        # last two entries are swapped vs VTK.  Node coordinates make the
        # permutation observable: canonical position 8 must be the
        # (1,3)-edge midpoint (0.5,0,0.5) and position 9 the (2,3)-edge
        # midpoint (0,0.5,0.5).
        deck = (_HEADER
                + "$Nodes\n10\n"
                + "1 0.0 0.0 0.0\n2 1.0 0.0 0.0\n3 0.0 1.0 0.0\n"
                + "4 0.0 0.0 1.0\n"
                + "5 0.5 0.0 0.0\n6 0.5 0.5 0.0\n7 0.0 0.5 0.0\n"
                + "8 0.0 0.0 0.5\n"
                + "9 0.0 0.5 0.5\n10 0.5 0.0 0.5\n"
                + "$EndNodes\n"
                + "$Elements\n1\n"
                + "1 11 2 1 1 1 2 3 4 5 6 7 8 9 10\n"
                + "$EndElements\n")
        vertices, _, cells, _, _ = parse_msh(_write_deck(tmp_path, deck))

        conn = cells["tet10"][0]
        assert cells["tet10"].shape == (1, 10)
        np.testing.assert_allclose(vertices[conn[8]], [0.5, 0.0, 0.5])
        np.testing.assert_allclose(vertices[conn[9]], [0.0, 0.5, 0.5])

    def _all_quadratic_deck(self):
        specs = [
            (4, 4), (5, 8), (6, 6), (7, 5),          # linear volume types
            (11, 10), (17, 20), (12, 27),            # tet10/hex20/hex27
            (13, 18), (14, 14),                      # wedge18/pyramid14
        ]
        node_lines, element_lines = [], []
        nid = 1
        for eid, (code, width) in enumerate(specs, start=1):
            ids = []
            for _ in range(width):
                node_lines.append(f"{nid} {nid}.0 {nid * 2}.0 {nid * 3}.0")
                ids.append(nid)
                nid += 1
            element_lines.append(
                f"{eid} {code} 2 1 1 " + " ".join(str(i) for i in ids))
        return (_HEADER
                + f"$Nodes\n{len(node_lines)}\n"
                + "\n".join(node_lines) + "\n$EndNodes\n"
                + f"$Elements\n{len(element_lines)}\n"
                + "\n".join(element_lines) + "\n$EndElements\n")

    def test_matches_meshio_canonical_connectivity(self, tmp_path):
        """The native parser must produce exactly what meshio produces
        for the same file (linear + all constructible quadratic types),
        proving the Gmsh->VTK permutations are equivalent."""
        meshio = pytest.importorskip("meshio")
        from mesh_issue_examiner.adapters.meshio_adapter import MeshioAdapter

        path = _write_deck(tmp_path, self._all_quadratic_deck())
        _, _, native_cells, _, _ = parse_msh(path)
        meshio_cells = MeshioAdapter().to_canonical(
            meshio.read(path)).volume_cells

        assert list(native_cells) == list(meshio_cells)
        for etype, conn in native_cells.items():
            assert meshio_cells[etype].shape == conn.shape, etype
            np.testing.assert_array_equal(
                conn, meshio_cells[etype], err_msg=etype)

    def test_wedge15_and_pyramid13_parsed_natively(self, tmp_path):
        """meshio 5.3.x cannot even construct wedge15/pyramid13 cell
        blocks (KeyError); the native parser handles them."""
        deck = (_HEADER
                + "$Nodes\n28\n"
                + "".join(f"{i} {i}.0 0.0 0.0\n" for i in range(1, 29))
                + "$EndNodes\n"
                + "$Elements\n2\n"
                + "1 18 2 1 1 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15\n"
                + "2 19 2 1 1 16 17 18 19 20 21 22 23 24 25 26 27 28\n"
                + "$EndElements\n")
        _, _, cells, _, _ = parse_msh(_write_deck(tmp_path, deck))

        assert list(cells) == ["prism15", "pyramid13"]
        assert cells["prism15"].shape == (1, 15)
        assert cells["pyramid13"].shape == (1, 13)


class TestGmshMaterials:
    def test_physical_tags_follow_block_order(self, tmp_path):
        # Interleaved tet/hex elements with alternating physical tags:
        # volume_cells groups by type (first occurrence), and
        # cell_materials must follow that same enumeration.
        deck = (_HEADER
                + "$PhysicalNames\n2\n"
                + '3 1 "fluid"\n3 2 "solid"\n'
                + "$EndPhysicalNames\n"
                + "$Nodes\n10\n"
                + "".join(f"{i} {i}.0 0.0 0.0\n" for i in range(1, 11))
                + "$EndNodes\n"
                + "$Elements\n4\n"
                + "1 4 2 1 1 1 2 3 4\n"
                + "2 5 2 2 2 1 2 3 4 5 6 7 8\n"
                + "3 4 2 2 2 5 6 7 8\n"
                + "4 5 2 1 1 1 2 3 4 5 6 7 8\n"
                + "$EndElements\n")
        _, _, cells, mats, names = parse_msh(_write_deck(tmp_path, deck))

        assert list(cells) == ["tet4", "hex8"]
        np.testing.assert_array_equal(mats, [1, 2, 2, 1])
        assert names == {1: "fluid", 2: "solid"}

    def test_physical_name_with_spaces(self, tmp_path):
        deck = (_HEADER
                + '$PhysicalNames\n1\n3 7 "fluid core"\n$EndPhysicalNames\n'
                + _UNIT_TET_NODES
                + "$Elements\n1\n1 4 2 7 1 1 2 3 4\n$EndElements\n")
        _, _, _, mats, names = parse_msh(_write_deck(tmp_path, deck))

        np.testing.assert_array_equal(mats, [7])
        assert names == {7: "fluid core"}

    def test_untagged_elements_yield_no_materials(self, tmp_path):
        deck = (_HEADER
                + _UNIT_TET_NODES
                + "$Elements\n1\n1 4 0 1 2 3 4\n$EndElements\n")
        _, _, cells, mats, names = parse_msh(_write_deck(tmp_path, deck))

        assert list(cells) == ["tet4"]
        assert mats is None
        assert names is None


class TestGmshDeclines:
    def test_binary_declined(self, tmp_path):
        deck = "$MeshFormat\n2.2 1 8\n$EndMeshFormat\n" + _UNIT_TET_NODES
        with pytest.raises(ValueError, match="[Bb]inary"):
            parse_msh(_write_deck(tmp_path, deck))

    def test_version_4_declined(self, tmp_path):
        deck = "$MeshFormat\n4.1 0 8\n$EndMeshFormat\n"
        with pytest.raises(ValueError, match="4.1"):
            parse_msh(_write_deck(tmp_path, deck))

    def test_missing_sections_declined(self, tmp_path):
        with pytest.raises(ValueError):
            parse_msh(_write_deck(tmp_path, "$MeshFormat\n2.2 0 8\n"))


class TestGmshLoaderRouting:
    @pytest.fixture()
    def ascii_deck(self, tmp_path):
        deck = (_HEADER
                + '$PhysicalNames\n2\n3 1 "fluid"\n3 2 "solid"\n'
                + "$EndPhysicalNames\n"
                + "$Nodes\n18\n"
                + "1 0.0 0.0 0.0\n2 1.0 0.0 0.0\n3 0.0 1.0 0.0\n"
                + "4 0.0 0.0 1.0\n"
                + "5 0.5 0.0 0.0\n6 0.5 0.5 0.0\n7 0.0 0.5 0.0\n"
                + "8 0.0 0.0 0.5\n9 0.0 0.5 0.5\n10 0.5 0.0 0.5\n"
                + "11 5.0 0.0 0.0\n12 6.0 0.0 0.0\n13 6.0 1.0 0.0\n"
                + "14 5.0 1.0 0.0\n15 5.0 0.0 1.0\n16 6.0 0.0 1.0\n"
                + "17 6.0 1.0 1.0\n18 5.0 1.0 1.0\n"
                + "$EndNodes\n"
                + "$Elements\n2\n"
                + "1 11 2 1 1 1 2 3 4 5 6 7 8 9 10\n"
                + "2 5 2 2 2 11 12 13 14 15 16 17 18\n"
                + "$EndElements\n")
        return _write_deck(tmp_path, deck)

    def test_native_path_loads_with_materials(self, ascii_deck):
        from mesh_issue_examiner.utils.mesh_loader import load_mesh_from_file

        mesh = load_mesh_from_file(
            ascii_deck, material_array_name="gmsh:physical")

        assert mesh.source_format == "msh"
        assert mesh.has_volume_cells
        assert set(mesh.cell_types) == {"tet10", "hex8"}
        assert mesh.n_volume_cells == 2
        np.testing.assert_array_equal(mesh.cell_materials, [1, 2])
        assert mesh.material_names == {1: "fluid", 2: "solid"}

    def test_native_path_without_material_request(self, ascii_deck):
        from mesh_issue_examiner.utils.mesh_loader import load_mesh_from_file

        mesh = load_mesh_from_file(ascii_deck)

        assert mesh.has_volume_cells
        assert mesh.cell_materials is None

    def test_fallback_to_meshio_for_binary(self, tmp_path):
        """A binary .msh the native parser declines must still load
        through the meshio fallback path."""
        meshio = pytest.importorskip("meshio")
        from mesh_issue_examiner.utils.mesh_loader import (
            MeshLoadError, load_mesh_from_file)

        path = tmp_path / "binary.msh"
        meshio.write(str(path), meshio.Mesh(
            points=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
                            float),
            cells=[("tetra", np.array([[0, 1, 2, 3]]))],
        ))

        mesh = load_mesh_from_file(str(path))
        assert mesh.source_format == "msh"
        assert mesh.has_volume_cells
        assert "tet4" in mesh.cell_types

    def test_unreadable_msh_raises_mesh_load_error(self, tmp_path):
        from mesh_issue_examiner.utils.mesh_loader import (
            MeshLoadError, load_mesh_from_file)

        path = tmp_path / "garbage.msh"
        path.write_text("not a gmsh file at all\n", encoding="ascii")

        with pytest.raises(MeshLoadError):
            load_mesh_from_file(str(path))
