"""Unit tests for the NASTRAN bulk-data parser (nas_parser.py).

Covers CTETRA (existing behavior) plus the CHEXA/CPENTA extension with
continuation cards, in both free-field and short-field formats.
"""

from pathlib import Path

import numpy as np
import pytest

from mesh_issue_examiner.utils.nas_parser import parse_nas, write_nas


def _write(tmp_path: Path, content: str) -> str:
    p = tmp_path / "deck.nas"
    p.write_text(content, encoding="ascii")
    return str(p)


FREE_FIELD_CHEXA20 = """\
BEGIN BULK
GRID,1,0,0.0,0.0,0.0
GRID,2,0,1.0,0.0,0.0
GRID,3,0,1.0,1.0,0.0
GRID,4,0,0.0,1.0,0.0
GRID,5,0,0.0,0.0,1.0
GRID,6,0,1.0,0.0,1.0
GRID,7,0,1.0,1.0,1.0
GRID,8,0,0.0,1.0,1.0
GRID,9,0,0.5,0.0,0.0
GRID,10,0,1.0,0.5,0.0
GRID,11,0,0.5,1.0,0.0
GRID,12,0,0.0,0.5,0.0
GRID,13,0,0.5,0.0,1.0
GRID,14,0,1.0,0.5,1.0
GRID,15,0,0.5,1.0,1.0
GRID,16,0,0.0,0.5,1.0
GRID,17,0,0.0,0.0,0.5
GRID,18,0,1.0,0.0,0.5
GRID,19,0,1.0,1.0,0.5
GRID,20,0,0.0,1.0,0.5
CHEXA,1,100,1,2,3,4,5,6,+H1
+H1,7,8,9,10,11,12,13,14,+H2
+H2,15,16,17,18,19,20
ENDDATA
"""


class TestChexaParsing:
    def test_free_field_20_node_chexa_retained_as_hex20(self, tmp_path):
        """20-node CHEXA: all 20 nodes reach volume_cells as hex20.

        Recalibrated 2026-10-28 (Phase 5): midside nodes are retained —
        NASTRAN's midside ordering (bottom ring, top ring, verticals)
        matches VTK's QUADRATIC_HEXAHEDRON ordering, so the row maps
        straight through.  Boundary faces stay corner-based.
        """
        v, f, vc = parse_nas(_write(tmp_path, FREE_FIELD_CHEXA20))
        assert set(vc.keys()) == {"hex20"}
        assert vc["hex20"].shape == (1, 20)
        # All 20 GRID nodes used (sorted ids 1..20 -> indices 0..19).
        assert len(v) == 20
        np.testing.assert_array_equal(
            vc["hex20"][0], list(range(20)))
        # 6 corner quad faces -> 12 boundary triangles (midside nodes
        # do not change face identity)
        assert f.shape == (12, 3)

    def test_short_field_chexa8_with_continuation(self, tmp_path):
        """Short-field 8-node CHEXA split over two cards."""
        deck = (
            "BEGIN BULK\n"
            "GRID    1       0       0.0     0.0     0.0\n"
            "GRID    2       0       1.0     0.0     0.0\n"
            "GRID    3       0       1.0     1.0     0.0\n"
            "GRID    4       0       0.0     1.0     0.0\n"
            "GRID    5       0       0.0     0.0     1.0\n"
            "GRID    6       0       1.0     0.0     1.0\n"
            "GRID    7       0       1.0     1.0     1.0\n"
            "GRID    8       0       0.0     1.0     1.0\n"
            "CHEXA   1       100     1       2       3       4       5       6       +H1\n"
            "+H1     7       8\n"
            "ENDDATA\n"
        )
        v, f, vc = parse_nas(_write(tmp_path, deck))
        assert set(vc.keys()) == {"hex8"}
        assert vc["hex8"].shape == (1, 8)
        np.testing.assert_array_equal(vc["hex8"][0], list(range(8)))
        assert f.shape == (12, 3)

    def test_two_chexa_sharing_face_boundary_extracted(self, tmp_path):
        """Two CHEXAs sharing a quad: the shared quad's triangles must be
        removed from the boundary surface (each appears twice)."""
        deck = (
            "BEGIN BULK\n"
            + "".join(
                f"GRID,{gid},0,{x},{y},{z}\n"
                for gid, (x, y, z) in enumerate([
                    (0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),   # 1-4
                    (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1),   # 5-8
                    (2, 0, 0), (2, 1, 0), (2, 0, 1), (2, 1, 1),   # 9-12
                ], start=1)
            )
            + "CHEXA,1,100,1,2,3,4,5,6,7,8\n"
            # Second hex spans x=1..2; its left face (2,3,7,6) is shared
            # with hex1's right face. Both cards list the quad starting at
            # G2, so the two triangle keys cancel in boundary extraction.
            "CHEXA,2,100,2,3,10,9,6,7,12,11\n"
            "ENDDATA\n"
        )
        v, f, vc = parse_nas(_write(tmp_path, deck))
        assert vc["hex8"].shape == (2, 8)
        # 2 * 12 triangles total; the shared quad contributes 2 duplicated
        # triangle keys (4 triangles) which cancel -> 20 boundary faces.
        assert f.shape == (20, 3)


class TestCpentaParsing:
    def test_free_field_cpenta6_with_continuation(self, tmp_path):
        deck = (
            "BEGIN BULK\n"
            "GRID,1,0,0.0,0.0,0.0\n"
            "GRID,2,0,1.0,0.0,0.0\n"
            "GRID,3,0,0.5,1.0,0.0\n"
            "GRID,4,0,0.0,0.0,1.0\n"
            "GRID,5,0,1.0,0.0,1.0\n"
            "GRID,6,0,0.5,1.0,1.0\n"
            "CPENTA,1,100,1,2,3,4,5,+P1\n"
            "+P1,6\n"
            "ENDDATA\n"
        )
        v, f, vc = parse_nas(_write(tmp_path, deck))
        assert set(vc.keys()) == {"prism6"}
        assert vc["prism6"].shape == (1, 6)
        np.testing.assert_array_equal(vc["prism6"][0], list(range(6)))
        # 2 triangle caps + 3 quads -> 2 + 6 = 8 boundary faces
        assert f.shape == (8, 3)

    def test_cpenta15_midside_retained_as_prism15(self, tmp_path):
        """15-node CPENTA: full connectivity retained as prism15.

        Recalibrated 2026-10-28 (Phase 5): NASTRAN midside ordering
        (bottom edges, top edges, verticals) matches VTK's
        QUADRATIC_WEDGE ordering; boundary faces stay corner-based.
        """
        deck = (
            "BEGIN BULK\n"
            "GRID,1,0,0.0,0.0,0.0\n"
            "GRID,2,0,1.0,0.0,0.0\n"
            "GRID,3,0,0.5,1.0,0.0\n"
            "GRID,4,0,0.0,0.0,1.0\n"
            "GRID,5,0,1.0,0.0,1.0\n"
            "GRID,6,0,0.5,1.0,1.0\n"
            # bottom edges (1-2),(2-3),(3-1)
            "GRID,7,0,0.5,0.0,0.0\n"
            "GRID,8,0,0.75,0.5,0.0\n"
            "GRID,9,0,0.25,0.5,0.0\n"
            # top edges (4-5),(5-6),(6-4)
            "GRID,10,0,0.5,0.0,1.0\n"
            "GRID,11,0,0.75,0.5,1.0\n"
            "GRID,12,0,0.25,0.5,1.0\n"
            # verticals (1-4),(2-5),(3-6)
            "GRID,13,0,0.0,0.0,0.5\n"
            "GRID,14,0,1.0,0.0,0.5\n"
            "GRID,15,0,0.5,1.0,0.5\n"
            "CPENTA,1,100,1,2,3,4,5,+P1\n"
            "+P1,6,7,8,9,10,11,+P2\n"
            "+P2,12,13,14,15\n"
            "ENDDATA\n"
        )
        v, f, vc = parse_nas(_write(tmp_path, deck))
        assert set(vc.keys()) == {"prism15"}
        assert vc["prism15"].shape == (1, 15)
        assert len(v) == 15
        np.testing.assert_array_equal(
            vc["prism15"][0], list(range(15)))
        # 2 corner caps + 3 corner quads -> 8 boundary triangles
        assert f.shape == (8, 3)

    def test_partial_midside_form_falls_back_to_corners(self, tmp_path):
        """A CPENTA listing only some midside nodes (no registered
        9-node type) reduces to the prism6 corner connectivity."""
        deck = (
            "BEGIN BULK\n"
            "GRID,1,0,0.0,0.0,0.0\n"
            "GRID,2,0,1.0,0.0,0.0\n"
            "GRID,3,0,0.5,1.0,0.0\n"
            "GRID,4,0,0.0,0.0,1.0\n"
            "GRID,5,0,1.0,0.0,1.0\n"
            "GRID,6,0,0.5,1.0,1.0\n"
            "GRID,7,0,0.5,0.0,0.0\n"
            "GRID,8,0,0.75,0.5,0.0\n"
            "GRID,9,0,0.25,0.5,0.0\n"
            "CPENTA,1,100,1,2,3,4,5,+P1\n"
            "+P1,6,7,8,9\n"
            "ENDDATA\n"
        )
        v, f, vc = parse_nas(_write(tmp_path, deck))
        assert set(vc.keys()) == {"prism6"}
        assert len(v) == 6  # unlisted-in-cells midside nodes dropped


class TestMixedDeck:
    def test_tet_hex_prism_mix(self, tmp_path):
        deck = (
            "BEGIN BULK\n"
            "GRID,1,0,0.0,0.0,0.0\n"
            "GRID,2,0,1.0,0.0,0.0\n"
            "GRID,3,0,0.0,1.0,0.0\n"
            "GRID,4,0,0.0,0.0,1.0\n"
            "CTETRA,1,100,1,2,3,4\n"
            "GRID,5,0,5.0,0.0,0.0\n"
            "GRID,6,0,6.0,0.0,0.0\n"
            "GRID,7,0,6.0,1.0,0.0\n"
            "GRID,8,0,5.0,1.0,0.0\n"
            "GRID,9,0,5.0,0.0,1.0\n"
            "GRID,10,0,6.0,0.0,1.0\n"
            "GRID,11,0,6.0,1.0,1.0\n"
            "GRID,12,0,5.0,1.0,1.0\n"
            "CHEXA,2,100,5,6,7,8,9,10,11,12\n"
            "GRID,13,0,10.0,0.0,0.0\n"
            "GRID,14,0,11.0,0.0,0.0\n"
            "GRID,15,0,10.5,1.0,0.0\n"
            "GRID,16,0,10.0,0.0,1.0\n"
            "GRID,17,0,11.0,0.0,1.0\n"
            "GRID,18,0,10.5,1.0,1.0\n"
            "CPENTA,3,100,13,14,15,16,17,18\n"
            "ENDDATA\n"
        )
        v, f, vc = parse_nas(_write(tmp_path, deck))
        assert set(vc.keys()) == {"tet4", "hex8", "prism6"}
        assert vc["tet4"].shape == (1, 4)
        assert vc["hex8"].shape == (1, 8)
        assert vc["prism6"].shape == (1, 6)
        # 4 + 12 + 8 boundary triangles from the three disjoint cells
        assert f.shape == (24, 3)


class TestContinuationEdgeCases:
    def test_continuation_marker_not_treated_as_grid(self, tmp_path):
        """A comment line after element cards must not break parsing."""
        deck = (
            "BEGIN BULK\n"
            "GRID,1,0,0.0,0.0,0.0\n"
            "GRID,2,0,1.0,0.0,0.0\n"
            "GRID,3,0,0.0,1.0,0.0\n"
            "GRID,4,0,0.0,0.0,1.0\n"
            "CTETRA,1,100,1,2,3,4\n"
            "$ comment line\n"
            "ENDDATA\n"
        )
        v, f, vc = parse_nas(_write(tmp_path, deck))
        assert set(vc.keys()) == {"tet4"}

    def test_malformed_chexa_skipped(self, tmp_path):
        """A CHEXA with too few nodes is skipped without error."""
        deck = (
            "BEGIN BULK\n"
            "GRID,1,0,0.0,0.0,0.0\n"
            "GRID,2,0,1.0,0.0,0.0\n"
            "GRID,3,0,0.0,1.0,0.0\n"
            "GRID,4,0,0.0,0.0,1.0\n"
            "CTETRA,1,100,1,2,3,4\n"
            "CHEXA,2,100,1,2,3,+X\n"
            "+X,4\n"
            "ENDDATA\n"
        )
        v, f, vc = parse_nas(_write(tmp_path, deck))
        assert set(vc.keys()) == {"tet4"}  # malformed CHEXA dropped


# =============================================================================
# Write-back (write_nas) and round-trips
# =============================================================================

_UNIT_TET = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], float)


def _unit_hex20():
    """Unit hex with straight midside nodes in canonical (VTK) order."""
    corners = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
    ], float)
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7),
             (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
    vertices = np.vstack(
        [corners, [(corners[a] + corners[b]) / 2 for a, b in edges]])
    cells = np.array([list(range(20))])
    return vertices, cells


class TestWriteNas:
    def test_card_structure(self, tmp_path):
        path = tmp_path / "out.nas"
        write_nas(_UNIT_TET, np.zeros((0, 3), int),
                  {"tet4": np.array([[0, 1, 2, 3]])}, str(path))
        text = path.read_text(encoding="ascii")

        lines = text.strip().splitlines()
        assert lines[0].startswith("$")
        assert lines[1] == "BEGIN BULK"
        assert lines[-1] == "ENDDATA"
        assert "GRID,1,0,0.0,0.0,0.0" in lines
        assert "CTETRA,101,1,1,2,3,4" in lines

    def test_hex20_continuation_cards(self, tmp_path):
        vertices, cells = _unit_hex20()
        path = tmp_path / "out.nas"
        write_nas(vertices, np.zeros((0, 3), int), {"hex20": cells},
                  str(path))
        lines = path.read_text(encoding="ascii").strip().splitlines()

        parent = [ln for ln in lines if ln.startswith("CHEXA")]
        conts = [ln for ln in lines if ln.startswith("+")]
        assert len(parent) == 1
        # parent holds 8 corners + continuation marker; the 12 midsides
        # span two continuation cards (8 + 4 fields).
        assert parent[0] == "CHEXA,101,1,1,2,3,4,5,6,7,8,+C1011"
        assert len(conts) == 2
        assert conts[0].startswith("+C1011,")
        assert conts[1].startswith("+C1012,")
        assert conts[0].endswith(",+C1012")

        v2, f2, vc2 = parse_nas(str(path))
        assert set(vc2) == {"hex20"}
        np.testing.assert_array_equal(vc2["hex20"], cells)
        np.testing.assert_array_equal(v2, vertices)

    def test_surface_only_mesh_writes_ctria3(self, tmp_path):
        faces = np.array([[0, 1, 2], [0, 2, 3]])
        path = tmp_path / "out.nas"
        write_nas(_UNIT_TET, faces, {}, str(path))
        lines = path.read_text(encoding="ascii").strip().splitlines()

        assert "CTRIA3,101,1,1,2,3" in lines
        assert "CTRIA3,102,1,1,3,4" in lines

        v2, f2, vc2 = parse_nas(str(path))
        assert vc2 == {}
        np.testing.assert_array_equal(f2, faces)

    def test_unsupported_type_raises(self, tmp_path):
        with pytest.raises(ValueError, match="tet10"):
            write_nas(np.zeros((10, 3)), np.zeros((0, 3), int),
                      {"tet10": np.array([list(range(10))])},
                      str(tmp_path / "out.nas"))

    def test_round_trip_nas_mixed_volume_fixture(self, tmp_path):
        """parse -> write -> parse of the e2e mixed deck preserves
        vertices, boundary faces and all volume cells (hex20 retained)."""
        fixture = (Path(__file__).resolve().parents[2] / "e2e"
                   / "sample_meshes" / "nas_mixed_volume"
                   / "nas_mixed_volume.nas")
        if not fixture.exists():  # fixtures generated by conftest
            pytest.skip("nas_mixed_volume fixture not generated")

        v1, f1, vc1 = parse_nas(str(fixture))
        out = tmp_path / "roundtrip.nas"
        write_nas(v1, f1, vc1, str(out))
        v2, f2, vc2 = parse_nas(str(out))

        np.testing.assert_array_equal(v1, v2)
        np.testing.assert_array_equal(f1, f2)
        assert list(vc2) == list(vc1)
        for etype in vc1:
            np.testing.assert_array_equal(vc1[etype], vc2[etype],
                                          err_msg=etype)

    def test_save_mesh_to_file_dispatches_to_write_nas(self, tmp_path):
        """save_mesh_to_file routes .nas through write_nas (quadratic
        forms kept), unlike the meshio path which cannot write hex20
        cards."""
        from mesh_issue_examiner.utils.mesh_loader import (
            load_mesh_from_file, save_mesh_to_file)

        fixture = (Path(__file__).resolve().parents[2] / "e2e"
                   / "sample_meshes" / "nas_mixed_volume"
                   / "nas_mixed_volume.nas")
        if not fixture.exists():
            pytest.skip("nas_mixed_volume fixture not generated")

        mesh = load_mesh_from_file(str(fixture))
        out = tmp_path / "saved.nas"
        save_mesh_to_file(mesh, str(out))

        reloaded = load_mesh_from_file(str(out))
        assert reloaded.source_format == "nas"
        assert set(reloaded.cell_types) == set(mesh.cell_types)
        for etype in mesh.cell_types:
            np.testing.assert_array_equal(
                mesh.volume_cells[etype],
                reloaded.volume_cells[etype], err_msg=etype)
        np.testing.assert_array_equal(mesh.vertices, reloaded.vertices)
        np.testing.assert_array_equal(mesh.faces, reloaded.faces)
