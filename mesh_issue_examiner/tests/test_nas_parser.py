"""Unit tests for the NASTRAN bulk-data parser (nas_parser.py).

Covers CTETRA (existing behavior) plus the CHEXA/CPENTA extension with
continuation cards, in both free-field and short-field formats.
"""

from pathlib import Path

import numpy as np
import pytest

from mesh_issue_examiner.utils.nas_parser import parse_nas


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
