"""
End-to-end tests loading real mesh files from e2e/sample_meshes/
through the full pipeline: load -> inspect -> get_issues -> get_quality_metrics.

The fixture files are deterministic and committed to the repository;
``tests/conftest.py`` regenerates any that are missing (via
scripts/generate_sample_meshes.py, which skips files that already exist).
To rebuild them by hand:

    python scripts/generate_sample_meshes.py --force   # recreate + verify
    python scripts/generate_sample_meshes.py --verify  # verify only

Run with:
    python -m pytest mesh_issue_examiner/tests/test_sample_meshes_e2e.py -v
"""

from pathlib import Path

import numpy as np
import pytest
import trimesh

from mesh_issue_examiner.mcp.server import MeshIssueExaminerServer, MCPRequest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAMPLE_DIR = Path(__file__).resolve().parents[2] / "e2e" / "sample_meshes"


def _mesh_file(subdir: str, fmt: str = "stl", file_override: str = None) -> str:
    """Absolute path to a sample mesh file."""
    if file_override:
        p = SAMPLE_DIR / subdir / file_override
        if p.exists():
            return str(p)
    for ext in (fmt, "stl", "obj"):
        p = SAMPLE_DIR / subdir / f"{subdir}.{ext}"
        if p.exists():
            return str(p)
    raise FileNotFoundError(f"No mesh file found for {subdir}")


def _server():
    return MeshIssueExaminerServer(auto_register_operations=True)


async def _load_inspect_get_issues(server, subdir, fmt="stl", file_override=None):
    """Full pipeline: load -> inspect -> get_issues. Returns three dicts."""
    req = MCPRequest(method="mesh_load", params={"file_path": _mesh_file(subdir, fmt, file_override)})
    resp = await server.handle_request(req)
    assert resp.error is None, f"load failed for {subdir}: {resp.error}"
    load = resp.result

    mesh_id = load["mesh_id"]
    req = MCPRequest(method="mesh_inspect", params={"mesh_id": mesh_id})
    resp = await server.handle_request(req)
    assert resp.error is None, f"inspect failed for {subdir}: {resp.error}"
    inspect = resp.result

    req = MCPRequest(method="mesh_get_issues", params={"mesh_id": mesh_id})
    resp = await server.handle_request(req)
    assert resp.error is None, f"get_issues failed for {subdir}: {resp.error}"
    issues = resp.result

    return load, inspect, issues


# ===================================================================
# 1. Truly clean meshes — zero topological AND quality issues
# ===================================================================

PERFECT_MESHES = [
    "perfect_cube_surface",
    "perfect_hexahedron",
    "torus",
    "pyramid",
    "wedge",
]


class TestCleanMeshes:
    """Well-formed meshes: inspection should pass with zero issues."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("mesh_name", PERFECT_MESHES)
    async def test_passes_inspection(self, mesh_name):
        server = _server()
        load, inspect, issues = await _load_inspect_get_issues(server, mesh_name)
        assert inspect["passed"] is True, (
            f"{mesh_name}: expected passed=True, "
            f"total_issues={inspect['summary']['total_issues']}"
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize("mesh_name", PERFECT_MESHES)
    async def test_zero_issues(self, mesh_name):
        server = _server()
        _, _, issues = await _load_inspect_get_issues(server, mesh_name)
        assert issues["total_count"] == 0, (
            f"{mesh_name}: expected 0 issues, got {issues['total_count']}"
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize("mesh_name", PERFECT_MESHES)
    async def test_has_vertices_and_faces(self, mesh_name):
        server = _server()
        load, _, _ = await _load_inspect_get_issues(server, mesh_name)
        assert load["n_vertices"] > 0
        assert load["n_faces"] > 0


# ===================================================================
# 2. Tessellated meshes — clean topology but some quality issues
# ===================================================================

TESSELLATED_MESHES = [
    "icosphere_l1",  # 80 triangles, some below angle thresholds
    "icosphere_l2",  # 320 triangles, more quality variation
]


class TestTessellatedMeshes:
    """Icospheres have clean topology but quality metrics may flag some faces."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("mesh_name", TESSELLATED_MESHES)
    async def test_has_vertices_and_faces(self, mesh_name):
        server = _server()
        load, _, _ = await _load_inspect_get_issues(server, mesh_name)
        assert load["n_vertices"] > 0
        assert load["n_faces"] > 0

    @pytest.mark.asyncio
    @pytest.mark.parametrize("mesh_name", TESSELLATED_MESHES)
    async def test_has_quality_issues(self, mesh_name):
        """Tessellated meshes produce cell-quality violations from face shape.

        The z-stretched icospheres must trip the min-angle threshold of
        quality.compute_cell_quality (not merely the edge-dihedral op); see
        SAMPLE_MESH_AUDIT.md and scripts/tune_sample_meshes.py.
        """
        server = _server()
        _, inspect, issues = await _load_inspect_get_issues(server, mesh_name)
        assert issues["total_count"] > 0, (
            f"{mesh_name}: expected quality issues from tessellation"
        )
        op_ids = {i["operation_id"] for i in issues["issues"]}
        assert any("quality.compute_cell_quality" in oid for oid in op_ids), (
            f"{mesh_name}: expected a quality.compute_cell_quality issue, "
            f"got {op_ids}"
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize("mesh_name", TESSELLATED_MESHES)
    async def test_issues_are_present(self, mesh_name):
        """Tessellated meshes produce issues (topological or quality)."""
        server = _server()
        _, _, issues = await _load_inspect_get_issues(server, mesh_name)
        assert issues["total_count"] > 0


# ===================================================================
# 3. Defective meshes — detected by current operations
# ===================================================================

DEFECTIVE_CASES = {
    "non_manifold_edge": {
        "expect_passed": False,
        "expect_keywords": ["nonmanifold"],
    },
    "non_manifold_vertex": {
        "expect_passed": True,
        "expect_keywords": ["nonmanifold", "vertices"],
    },
    "degenerate_triangle": {
        "expect_passed": False,
        "expect_keywords": ["aspect_ratio", "min_angle", "max_angle"],
    },
    "self_intersecting": {
        "expect_passed": False,
        "expect_keywords": ["detect_self_intersections"],
    },
    # Old geometry of self_intersecting, kept as the contact-inclusive
    # counterpart: a vertex touches the interior of another triangle's
    # edge with no proper crossing (see TestSelfIntersectionGeometry).
    "self_intersecting_point_contact": {
        "expect_passed": False,
        "expect_keywords": ["detect_self_intersections"],
    },
    "varying_quality": {
        "expect_passed": False,
        "expect_keywords": ["aspect_ratio"],
    },
    "flipped_normals": {
        "expect_passed": False,
        "expect_keywords": ["orientation", "inconsistent"],
    },
    "disconnected_components": {
        "expect_passed": False,
        "expect_keywords": ["disconnected"],
    },
    "isolated_vertices": {
        "expect_passed": False,
        "expect_keywords": ["isolated"],
        "format": "vtk",
        "file_override": "isolated_vertices_surface.vtk",
    },
    "duplicate_vertices": {
        "expect_passed": False,
        "expect_keywords": ["duplicate", "faces"],
    },
    "perfect_tetrahedron": {
        "expect_passed": False,
        "expect_keywords": ["orientation", "inconsistent"],
    },
}


class TestDefectiveMeshes:
    """Meshes with known defects that the system detects."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("mesh_name", list(DEFECTIVE_CASES))
    async def test_fails_inspection(self, mesh_name):
        server = _server()
        case = DEFECTIVE_CASES[mesh_name]
        fmt = case.get("format", "stl")
        fov = case.get("file_override")
        _, inspect, _ = await _load_inspect_get_issues(server, mesh_name, fmt, fov)
        expected = case["expect_passed"]
        assert inspect["passed"] == expected, (
            f"{mesh_name}: expected passed={expected}, "
            f"got {inspect['passed']}, "
            f"issues={inspect['summary']['total_issues']}"
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize("mesh_name", list(DEFECTIVE_CASES))
    async def test_has_issues(self, mesh_name):
        server = _server()
        case = DEFECTIVE_CASES[mesh_name]
        fmt = case.get("format", "stl")
        fov = case.get("file_override")
        _, inspect, _ = await _load_inspect_get_issues(server, mesh_name, fmt, fov)
        assert inspect["summary"]["total_issues"] > 0

    @pytest.mark.asyncio
    @pytest.mark.parametrize("mesh_name", list(DEFECTIVE_CASES))
    async def test_issue_keywords(self, mesh_name):
        """Check that expected operation IDs appear in the issues list."""
        case = DEFECTIVE_CASES[mesh_name]
        if not case["expect_keywords"]:
            pytest.skip(f"{mesh_name}: no specific keywords to check")

        server = _server()
        fmt = case.get("format", "stl")
        fov = case.get("file_override")
        _, _, issues = await _load_inspect_get_issues(server, mesh_name, fmt, fov)
        op_ids = {i["operation_id"] for i in issues["issues"]}

        for kw in case["expect_keywords"]:
            assert any(kw in oid for oid in op_ids), (
                f"{mesh_name}: expected '{kw}' in operation IDs, got {op_ids}"
            )


# ===================================================================
# 3b. Volumetric (.vtu) meshes — topology_3d connectivity defects
# ===================================================================

VOLUME_CASES = {
    # Clean 2x2x2 hex grid: zero issues.  The face-winding tables were
    # normalised to the outward convention (2026-10-28) so stacked hex
    # bottom faces no longer trip check_orientation_consistency, and the
    # 2D surface ops are volume-aware so the interior node is not
    # flagged as isolated/irregular/disconnected.
    "closed_volume": {
        "expect_passed": True,
        "expect_zero": True,
        "expect_keywords": [],
        "forbid_keywords": ["duplicate_cells", "orphaned_cells",
                            "boundary_closure", "nonmanifold_faces",
                            "check_orientation_consistency"],
    },
    "duplicate_cell": {
        "expect_passed": False,
        "expect_keywords": ["duplicate_cells", "boundary_closure"],
        "forbid_keywords": ["orphaned_cells"],
    },
    # The T-junction still fires boundary_closure but at sub-failing
    # severity, so the mesh passes since the boundary-layer recalibration.
    "leaky_volume": {
        "expect_passed": True,
        "expect_keywords": ["boundary_closure"],
        "forbid_keywords": ["duplicate_cells", "orphaned_cells",
                            "nonmanifold_faces"],
    },
    # Clean quadratic (tet10) volume: connectivity ops stay silent;
    # quality metrics come from the corner-linearised grid (Phase 4)
    # with real values and zero violations.
    "tet10_volume": {
        "expect_passed": True,
        "expect_keywords": [],
        "forbid_keywords": ["duplicate_cells", "orphaned_cells",
                            "boundary_closure", "nonmanifold_faces",
                            "check_orientation_consistency"],
    },
    # Single squashed tet10: sliver fires via corner linearisation,
    # attributed to tet10; connectivity ops stay silent.
    "quadratic_sliver": {
        "expect_passed": False,
        "expect_keywords": ["assess_geometric_validity"],
        "forbid_keywords": ["duplicate_cells", "orphaned_cells",
                            "boundary_closure", "nonmanifold_faces",
                            "check_orientation_consistency"],
    },
}


class TestVolumeMeshes:
    """Hex8 .vtu volumes exercising the volumetric connectivity ops."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("mesh_name", list(VOLUME_CASES))
    async def test_volume_mesh_loads(self, mesh_name):
        server = _server()
        load, inspect, _ = await _load_inspect_get_issues(server, mesh_name, fmt="vtu")
        assert load["n_vertices"] > 0
        total = inspect["summary"]["total_issues"]
        if VOLUME_CASES[mesh_name].get("expect_zero"):
            assert total == 0
        else:
            assert total > 0

    @pytest.mark.asyncio
    @pytest.mark.parametrize("mesh_name", list(VOLUME_CASES))
    async def test_volume_connectivity_contract(self, mesh_name):
        """Keyword contracts for the volumetric connectivity ops."""
        case = VOLUME_CASES[mesh_name]
        server = _server()
        _, inspect, issues = await _load_inspect_get_issues(server, mesh_name, fmt="vtu")
        assert inspect["passed"] == case["expect_passed"]
        op_ids = {i["operation_id"] for i in issues["issues"]}
        for kw in case["expect_keywords"]:
            assert any(kw in oid for oid in op_ids), (
                f"{mesh_name}: expected '{kw}' in operation IDs, got {op_ids}"
            )
        for kw in case["forbid_keywords"]:
            assert not any(kw in oid for oid in op_ids), (
                f"{mesh_name}: unexpected '{kw}' in operation IDs, got {op_ids}"
            )

    @pytest.mark.asyncio
    async def test_volume_quality_metrics_surfaced(self):
        """Phase-1 regression: metrics surface end-to-end for volumes too."""
        server = _server()
        load, _, _ = await _load_inspect_get_issues(server, "closed_volume", fmt="vtu")
        resp = await server.handle_request(MCPRequest(
            method="mesh_get_quality_metrics",
            params={"mesh_id": load["mesh_id"]},
        ))
        assert resp.error is None, f"get_quality_metrics failed: {resp.error}"
        assert resp.result["metric_count"] >= 1, (
            f"expected quality metrics for closed_volume, "
            f"got {resp.result['metric_count']}"
        )

    @pytest.mark.asyncio
    async def test_tet10_quality_metrics_linearized(self):
        """Phase-4: quadratic volumes get real (corner-linearised) metrics.

        vtkCellQuality cannot evaluate quadratic cells, so the quality op
        recomputes on the corner-only tet4 grid; values must surface via
        mesh_get_quality_metrics tagged as linearised, matching the
        corner-equivalent geometry (unit tets: volume 1/6 each).
        """
        server = _server()
        load, inspect, _ = await _load_inspect_get_issues(
            server, "tet10_volume", fmt="vtu")
        assert inspect["passed"] is True
        resp = await server.handle_request(MCPRequest(
            method="mesh_get_quality_metrics",
            params={"mesh_id": load["mesh_id"]},
        ))
        assert resp.error is None, f"get_quality_metrics failed: {resp.error}"
        by_name = {m["metric_name"]: m for m in resp.result["metrics"]}
        for name in ("scaled_jacobian", "volume", "aspect_ratio"):
            assert name in by_name, sorted(by_name)
            m = by_name[name]
            assert m["metadata"].get("linearized") is True, name
            assert m["violation_count"] == 0, name
            assert m["total_count"] == 2, name
        assert by_name["scaled_jacobian"]["statistics"]["min"] > 0.5
        assert abs(by_name["volume"]["statistics"]["min"] - 1 / 6) < 1e-9

    @pytest.mark.asyncio
    async def test_tet10_topology_matches_tet4_corners(self):
        """tet10 topology must match the corner-equivalent tet4 mesh.

        VTK quadratic node ordering places corner nodes first, so the
        corner-only tet4 mesh with identical corner ordering must produce
        exactly the same connectivity/orientation results.
        """
        import numpy as np
        from mesh_issue_examiner.core.models import MeshInput
        from mesh_issue_examiner.operations.operation_registry import registry

        compare_ops = [
            "topology_3d.detect_duplicate_cells",
            "topology_3d.detect_orphaned_cells",
            "topology_3d.check_boundary_closure",
            "topology_3d.detect_nonmanifold_faces",
            "quality_3d.check_orientation_consistency",
        ]

        server = _server()
        _, inspect, issues = await _load_inspect_get_issues(
            server, "tet10_volume", fmt="vtu")
        assert inspect["passed"] is True
        tet10_op_ids = {
            i["operation_id"] for i in issues["issues"]
            if i["operation_id"] in compare_ops
        }

        # Corner-equivalent tet4 mesh (same corners/order as tet10_volume;
        # see scripts/generate_sample_meshes.py::tet10_volume).
        corners = np.array([
            [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0.25, 0.25, -1],
        ], float)
        tet4_mesh = MeshInput(
            vertices=corners,
            faces=np.zeros((0, 3), dtype=int),
            mesh_id="tet4_corner_equivalent",
            volume_cells={"tet4": np.array([[0, 1, 2, 3], [1, 0, 2, 4]])},
        )
        result4 = registry.create_workflow(compare_ops)(tet4_mesh)
        tet4_op_ids = {i.operation_id for i in result4.issues}

        assert tet10_op_ids == tet4_op_ids == set(), (
            f"topology mismatch: tet10={tet10_op_ids}, tet4={tet4_op_ids}"
        )


# ===================================================================
# 3c. Quadratic sliver (.vtu): corner-linearised geometric validity
# ===================================================================

class TestQuadraticSliver:
    """quadratic_sliver exercises sliver detection on quadratic cells.

    vtkCellQuality cannot evaluate tet10 cells, so geometric validity
    linearises to the corner tet4 grid; the squashed cell must be
    reported with sliver_tet10_count attribution.
    """

    @pytest.mark.asyncio
    async def test_sliver_fires_with_tet10_attribution(self):
        server = _server()
        _, inspect, issues = await _load_inspect_get_issues(
            server, "quadratic_sliver", fmt="vtu")
        assert inspect["passed"] is False

        sliver = [
            i for i in issues["issues"]
            if "assess_geometric_validity" in i["operation_id"]
            and "sliver" in i["message"]
        ]
        assert sliver, (
            f"expected a sliver issue, got "
            f"{[i['operation_id'] for i in issues['issues']]}")
        metrics = sliver[0]["metrics"]
        assert metrics["sliver_count"] == 1
        assert metrics["sliver_tet10_count"] == 1
        assert sliver[0]["metadata"].get("linearized") is True


# ===================================================================
# 3c. NASTRAN mixed volumetric deck (.nas): CTETRA + CHEXA + CPENTA
# ===================================================================

class TestNastranMixedVolume:
    """The nas_mixed_volume fixture exercises CHEXA/CPENTA parsing."""

    @pytest.mark.asyncio
    async def test_mixed_element_types_parsed(self):
        """CTETRA/CHEXA/CPENTA map to tet4/hex20/prism6 volume cells.

        The fixture's CHEXA uses the 20-node form with continuation cards;
        since Phase 5 (2026-10-28) the parser retains the midside nodes as
        hex20 (30 vertices: 4 tet + 20 hex + 6 wedge; NASTRAN midside
        ordering matches VTK so no reordering is needed).
        """
        from mesh_issue_examiner.utils.mesh_loader import load_mesh_from_file

        mesh = load_mesh_from_file(_mesh_file("nas_mixed_volume", fmt="nas"))
        assert mesh.has_volume_cells
        assert set(mesh.cell_types) == {"tet4", "hex20", "prism6"}
        assert mesh.volume_cells["tet4"].shape == (1, 4)
        assert mesh.volume_cells["hex20"].shape == (1, 20)
        assert mesh.volume_cells["prism6"].shape == (1, 6)
        assert mesh.n_vertices == 30  # midside nodes retained
        assert mesh.n_volume_cells == 3

    @pytest.mark.asyncio
    async def test_pipeline_runs_and_connectivity_ops_silent(self):
        """Full MCP pipeline on the .nas deck: three disjoint cells are
        detected as disconnected, but no connectivity defects fire."""
        server = _server()
        load, inspect, issues = await _load_inspect_get_issues(
            server, "nas_mixed_volume", fmt="nas")
        assert load["format"] == "nas"
        assert load["n_vertices"] == 30
        assert inspect["passed"] is False

        op_ids = {i["operation_id"] for i in issues["issues"]}
        assert any("disconnected" in oid for oid in op_ids), op_ids
        for kw in ("duplicate_cells", "orphaned_cells", "boundary_closure",
                   "nonmanifold_faces", "check_orientation_consistency"):
            assert not any(kw in oid for oid in op_ids), (
                f"nas_mixed_volume: unexpected '{kw}' in operation IDs: {op_ids}"
            )

    @pytest.mark.asyncio
    async def test_retained_hex20_gets_linearized_quality(self):
        """Phase 4 + 5: the retained hex20 cell receives corner-linearised
        quality metrics (unit hex corners -> volume 1.0, no violations)."""
        server = _server()
        load, _, _ = await _load_inspect_get_issues(
            server, "nas_mixed_volume", fmt="nas")
        resp = await server.handle_request(MCPRequest(
            method="mesh_get_quality_metrics",
            params={"mesh_id": load["mesh_id"]},
        ))
        assert resp.error is None, f"get_quality_metrics failed: {resp.error}"
        metrics = resp.result["metrics"]
        lin = [m for m in metrics if m["metadata"].get("linearized")]
        assert lin, [m["metric_name"] for m in metrics]
        assert {"hex20"} <= set(lin[0]["metadata"]["linearized_types"])
        by_name = {m["metric_name"]: m for m in metrics}
        assert by_name["volume"]["total_count"] == 3   # tet + hex + wedge
        assert by_name["volume"]["violation_count"] == 0
        assert abs(by_name["volume"]["statistics"]["max"] - 1.0) < 1e-9


# ===================================================================
# 3c-bis. Gmsh 2.2 ASCII mixed volumetric deck (.msh): native reader
# ===================================================================

class TestGmshMixedVolume:
    """The gmsh_mixed_volume fixture exercises the native .msh parser.

    A tet10 (Gmsh node order, reordered to canonical VTK order on load)
    and a hex8 in two physical groups; loaded through the native Gmsh
    2.2 ASCII parser with physical-group materials.
    """

    @pytest.mark.asyncio
    async def test_mixed_element_types_parsed(self):
        """tet10 (gmsh order) + hex8 parse into canonical volume cells
        with physical-group materials attached."""
        import numpy as np
        from mesh_issue_examiner.utils.mesh_loader import load_mesh_from_file

        mesh = load_mesh_from_file(
            _mesh_file("gmsh_mixed_volume", fmt="msh"),
            material_array_name="gmsh:physical",
        )
        assert mesh.source_format == "msh"
        assert mesh.has_volume_cells
        assert set(mesh.cell_types) == {"tet10", "hex8"}
        assert mesh.volume_cells["tet10"].shape == (1, 10)
        assert mesh.volume_cells["hex8"].shape == (1, 8)
        assert mesh.n_vertices == 18
        assert mesh.n_volume_cells == 2
        np.testing.assert_array_equal(mesh.cell_materials, [1, 2])
        assert mesh.material_names == {1: "fluid", 2: "solid"}

    @pytest.mark.asyncio
    async def test_tet10_midside_nodes_reordered_to_vtk(self):
        """Gmsh and VTK disagree on the last two tet10 midside entries;
        the parser must swap them so canonical position 8 is the (1,3)-
        edge midpoint and position 9 the (2,3)-edge midpoint."""
        import numpy as np
        from mesh_issue_examiner.utils.mesh_loader import load_mesh_from_file

        mesh = load_mesh_from_file(
            _mesh_file("gmsh_mixed_volume", fmt="msh"))
        conn = mesh.volume_cells["tet10"][0]
        np.testing.assert_allclose(mesh.vertices[conn[8]], [0.5, 0.0, 0.5])
        np.testing.assert_allclose(mesh.vertices[conn[9]], [0.0, 0.5, 0.5])

    @pytest.mark.asyncio
    async def test_pipeline_runs_and_connectivity_ops_silent(self):
        """Full MCP pipeline on the .msh deck: two disjoint bodies are
        detected as disconnected, but no connectivity defects fire."""
        server = _server()
        load, inspect, issues = await _load_inspect_get_issues(
            server, "gmsh_mixed_volume", fmt="msh")
        assert load["format"] == "msh"
        assert load["n_vertices"] == 18
        assert inspect["passed"] is False

        op_ids = {i["operation_id"] for i in issues["issues"]}
        assert any("disconnected" in oid for oid in op_ids), op_ids
        for kw in ("duplicate_cells", "orphaned_cells", "boundary_closure",
                   "nonmanifold_faces", "check_orientation_consistency"):
            assert not any(kw in oid for oid in op_ids), (
                f"gmsh_mixed_volume: unexpected '{kw}' in operation IDs: {op_ids}"
            )

    @pytest.mark.asyncio
    async def test_mcp_load_with_materials_reports_groups(self):
        """mesh_load with material_array_name surfaces the physical
        groups as the materials summary block."""
        server = _server()
        resp = await server.handle_request(MCPRequest(
            method="mesh_load",
            params={
                "file_path": _mesh_file("gmsh_mixed_volume", fmt="msh"),
                "material_array_name": "gmsh:physical",
            },
        ))
        assert resp.error is None, f"load failed: {resp.error}"
        mats = resp.result["materials"]
        assert mats["count"] == 2
        assert mats["ids"] == [1, 2]
        assert mats["n_cells_assigned"] == 2


# ===================================================================
# 3c-ter. Optional I/O formats (CGNS / Exodus): verified delegation
# ===================================================================

class TestOptionalIOFormats:
    """cgns_volume / exo_volume exercise meshio's format delegation.

    The fixtures (the clean 2x2x2 hex volume) are only written — and
    these tests only run — when the delegate dependency is installed:
    h5py for CGNS, netCDF4 for Exodus.  Without the dependency the
    loader reports a clean MeshLoadError (covered by the gmsh garbage
    test path).
    """

    @pytest.mark.asyncio
    async def test_cgns_volume_loads_and_passes(self):
        pytest.importorskip("h5py")
        server = _server()
        load, inspect, issues = await _load_inspect_get_issues(
            server, "cgns_volume", fmt="cgns")
        assert load["format"] == "cgns"
        # meshio's CGNS support is tetra-only; the fixture is the
        # two-unit-tet mesh (5 vertices, 2 cells) and it is fully clean.
        assert load["n_vertices"] == 5
        assert inspect["passed"] is True
        assert issues["total_count"] == 0

    @pytest.mark.asyncio
    async def test_cgns_volume_cells_populated(self):
        pytest.importorskip("h5py")
        from mesh_issue_examiner.utils.mesh_loader import load_mesh_from_file

        mesh = load_mesh_from_file(_mesh_file("cgns_volume", fmt="cgns"))
        assert mesh.has_volume_cells
        assert set(mesh.cell_types) == {"tet4"}
        assert mesh.volume_cells["tet4"].shape == (2, 4)

    @pytest.mark.asyncio
    async def test_exo_volume_loads_and_passes(self):
        pytest.importorskip("netCDF4")
        server = _server()
        load, inspect, issues = await _load_inspect_get_issues(
            server, "exo_volume", fmt="exo")
        assert load["format"] == "exo"
        assert load["n_vertices"] == 27
        assert inspect["passed"] is True
        assert issues["total_count"] == 0

    @pytest.mark.asyncio
    async def test_exo_volume_cells_populated(self):
        pytest.importorskip("netCDF4")
        from mesh_issue_examiner.utils.mesh_loader import load_mesh_from_file

        mesh = load_mesh_from_file(_mesh_file("exo_volume", fmt="exo"))
        assert mesh.has_volume_cells
        assert "hex8" in mesh.cell_types
        assert mesh.volume_cells["hex8"].shape == (8, 8)


# ===================================================================
# 3d. Two-material volumetric mesh (.vtu with material cell data)
# ===================================================================

class TestTwoMaterialVolume:
    """The two_material_volume fixture exercises the material loading path.

    Requires mesh_load(..., material_array_name="material") — without it
    the multi-material operations see no cell_materials and stay silent.
    """

    @pytest.mark.asyncio
    async def test_materials_loaded_and_aligned(self):
        """cell_materials follow volume_cells block order [1, 2, 1, 2].

        The generator builds a 2x2x1 hex grid with materials split along x;
        cells are enumerated k-major then j then i, so the expected array
        (one value per cell) is [1, 2, 1, 2].
        """
        import numpy as np
        from mesh_issue_examiner.utils.mesh_loader import load_mesh_from_file

        mesh = load_mesh_from_file(
            _mesh_file("two_material_volume", fmt="vtu"),
            material_array_name="material",
        )
        assert mesh.has_volume_cells
        assert mesh.n_volume_cells == 4
        mats = mesh.cell_materials
        assert mats is not None
        np.testing.assert_array_equal(mats, [1, 2, 1, 2])
        assert mesh.has_multiple_materials()
        assert mesh.validate_materials() == []

    @pytest.mark.asyncio
    async def test_mcp_load_returns_materials_summary(self):
        """mesh_load with material_array_name reports the materials block."""
        server = _server()
        resp = await server.handle_request(MCPRequest(
            method="mesh_load",
            params={"file_path": _mesh_file("two_material_volume", fmt="vtu"),
                    "material_array_name": "material"},
        ))
        assert resp.error is None, f"load failed: {resp.error}"
        load = resp.result
        assert load["materials"]["count"] == 2
        assert load["materials"]["ids"] == [1, 2]
        assert load["materials"]["n_cells_assigned"] == 4

    @pytest.mark.asyncio
    async def test_pipeline_reports_material_ops(self):
        """With materials loaded, detect_material_interfaces and
        per_material_statistics fire; connectivity ops stay silent."""
        server = _server()
        resp = await server.handle_request(MCPRequest(
            method="mesh_load",
            params={"file_path": _mesh_file("two_material_volume", fmt="vtu"),
                    "material_array_name": "material"},
        ))
        assert resp.error is None
        mesh_id = resp.result["mesh_id"]

        resp = await server.handle_request(MCPRequest(
            method="mesh_inspect", params={"mesh_id": mesh_id}))
        assert resp.error is None
        resp = await server.handle_request(MCPRequest(
            method="mesh_get_issues", params={"mesh_id": mesh_id}))
        assert resp.error is None
        issues = resp.result

        op_ids = {i["operation_id"] for i in issues["issues"]}
        assert any("detect_material_interfaces" in oid for oid in op_ids), op_ids
        assert any("per_material_statistics" in oid for oid in op_ids), op_ids
        for kw in ("duplicate_cells", "orphaned_cells", "boundary_closure",
                   "nonmanifold_faces"):
            assert not any(kw in oid for oid in op_ids), (
                f"two_material_volume: unexpected '{kw}' in operation IDs"
            )

    @pytest.mark.asyncio
    async def test_without_material_array_material_ops_silent(self):
        """Loading without material_array_name leaves cell_materials unset,
        so the multi-material operations report nothing (backwards compat)."""
        server = _server()
        _, _, issues = await _load_inspect_get_issues(
            server, "two_material_volume", fmt="vtu")
        op_ids = {i["operation_id"] for i in issues["issues"]}
        assert not any("material" in oid for oid in op_ids), op_ids


# ===================================================================
# 3e. Boundary-layer volumetric meshes (.vtu prism6 stack + tet4 cap)
# ===================================================================

class TestBoundaryLayerVolume:
    """boundary_layer / boundary_layer_bad_growth exercise the wall-seed
    calibration and growth-ratio detection end-to-end.

    The clean stack (growth 1.15) must keep quality_3d.check_boundary_layer
    silent; the defective stack (3x jump in layer 3) must fire it.
    """

    @pytest.mark.asyncio
    async def test_clean_stack_boundary_layer_op_silent(self):
        """The calibrated BL check reports nothing on the clean stack."""
        from mesh_issue_examiner.operations.operation_registry import registry
        from mesh_issue_examiner.utils.mesh_loader import load_mesh_from_file

        mesh = load_mesh_from_file(_mesh_file("boundary_layer", fmt="vtu"))
        assert set(mesh.cell_types) == {"prism6", "tet4"}
        assert mesh.n_volume_cells == 6

        op = registry.get("quality_3d.check_boundary_layer")
        issues = op.execute(mesh)
        assert issues == [], (
            f"clean BL stack must be silent, got: "
            f"{[(i.operation_id, i.message) for i in issues]}"
        )

    @pytest.mark.asyncio
    async def test_bad_growth_fires_high_severity_growth_issue(self):
        """The 3x layer jump produces a high-severity growth_ratio issue.

        Centroid-metric ratios [1.075, 2.14, 1.61, 1.15]: 2 of 4
        transitions violate 1.2 (rate 0.5 -> ERROR severity).
        """
        from mesh_issue_examiner.core.models import IssueSeverity
        from mesh_issue_examiner.operations.operation_registry import registry
        from mesh_issue_examiner.utils.mesh_loader import load_mesh_from_file

        mesh = load_mesh_from_file(
            _mesh_file("boundary_layer_bad_growth", fmt="vtu"))
        op = registry.get("quality_3d.check_boundary_layer")
        issues = op.execute(mesh)

        growth = [i for i in issues
                  if i.metadata.get("check") == "growth_ratio"]
        assert len(growth) == 1, (
            f"expected exactly one growth_ratio issue, got "
            f"{[(i.metadata, i.message) for i in issues]}"
        )
        assert growth[0].severity in (
            IssueSeverity.CRITICAL, IssueSeverity.ERROR
        )
        assert growth[0].metrics["max_growth_ratio"] > 1.2
        # Heights 0.2, 0.23, 0.69 ... -> centroid-metric ratio ~2.14
        assert growth[0].metrics["max_growth_ratio"] > 2.0

    @pytest.mark.asyncio
    async def test_pipeline_clean_stack_bl_op_absent(self):
        """Full MCP pipeline: the clean stack passes inspection since the
        winding recalibration (2026-10-28) — only sub-failing 2D surface
        warnings remain — and never shows the boundary-layer op."""
        server = _server()
        _, inspect, issues = await _load_inspect_get_issues(
            server, "boundary_layer", fmt="vtu")
        assert inspect["passed"] is True
        op_ids = {i["operation_id"] for i in issues["issues"]}
        assert not any("check_boundary_layer" in oid for oid in op_ids), (
            f"boundary_layer: unexpected BL op in {op_ids}"
        )

    @pytest.mark.asyncio
    async def test_pipeline_bad_growth_reports_bl_op(self):
        """Full MCP pipeline: the defective stack surfaces the BL op."""
        server = _server()
        _, inspect, issues = await _load_inspect_get_issues(
            server, "boundary_layer_bad_growth", fmt="vtu")
        assert inspect["passed"] is False
        op_ids = {i["operation_id"] for i in issues["issues"]}
        assert any("check_boundary_layer" in oid for oid in op_ids), op_ids


# ===================================================================
# 4. Open surface meshes — pass topology but may have quality issues
# ===================================================================

OPEN_MESHES = [
    "open_boundary",
    "holey_plane",
    "multi_hole_plane",
    "subdivided_plane",
    "cylinder",
    "perfect_triangle",
    "perfect_square",
]

CLOSED_MESHES = [
    "perfect_cube_surface",
    "torus",
    # Hollow shell: outer cube wound outward, inner cube wound inward,
    # so the enclosed solid is the space BETWEEN the cubes.
    "shell_structure",
]


class TestOpenMeshes:
    """Open meshes: should complete inspection without error."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("mesh_name", OPEN_MESHES)
    async def test_loads_and_inspects(self, mesh_name):
        server = _server()
        load, inspect, _ = await _load_inspect_get_issues(server, mesh_name)
        assert load["n_vertices"] > 0
        assert load["n_faces"] > 0
        # Just verify the pipeline completes
        assert "passed" in inspect

    @pytest.mark.asyncio
    @pytest.mark.parametrize("mesh_name", OPEN_MESHES)
    async def test_detects_boundary_edges(self, mesh_name):
        """Open meshes should have boundary edges detected."""
        server = _server()
        load, inspect, issues = await _load_inspect_get_issues(server, mesh_name)
        
        # Find boundary edge issues
        boundary_issues = [
            i for i in issues["issues"]
            if "boundary" in i["operation_id"]
        ]
        
        assert len(boundary_issues) >= 1, (
            f"{mesh_name}: Expected boundary edge detection, "
            f"got issues: {[i['operation_id'] for i in issues['issues']]}")
        
        # Verify the issue has correct structure
        issue = boundary_issues[0]
        assert issue["metrics"]["is_watertight"] is False
        assert issue["metrics"]["boundary_edge_count"] > 0


class TestClosedMeshes:
    """Closed/watertight meshes: should have no boundary edges."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("mesh_name", CLOSED_MESHES)
    async def test_no_boundary_edges(self, mesh_name):
        """Closed meshes should have no boundary edges."""
        server = _server()
        load, inspect, issues = await _load_inspect_get_issues(server, mesh_name)
        
        # Find boundary edge issues
        boundary_issues = [
            i for i in issues["issues"]
            if "boundary" in i["operation_id"]
        ]
        
        assert len(boundary_issues) == 0, (
            f"{mesh_name}: Expected no boundary edges for closed mesh, "
            f"but found {len(boundary_issues)} boundary issue(s)")

    @pytest.mark.asyncio
    async def test_shell_structure_is_hollow_shell(self):
        """shell_structure encloses the space between two cubes.

        The outer unit cube is wound outward and the inner half-size cube
        is wound inward, so the signed volume is 1 - 0.125 = 0.875 — a
        true outer-minus-inner solid, not two nested outward-wound cubes
        (which would sum to 1.125).
        """
        m = trimesh.load(_mesh_file("shell_structure"))
        m.merge_vertices()
        assert m.is_watertight
        assert abs(m.volume - 0.875) < 1e-6, (
            f"expected hollow-shell volume 0.875, got {m.volume}"
        )


# ===================================================================
# 4b. Self-intersection geometry (independent of the examiner)
# ===================================================================


def _interior_pierce_points(subdir: str) -> list:
    """Strict interior edge-pierce points between non-adjacent faces.

    Self-contained barycentric predicate (does NOT use the examiner's
    detector, which is the code under test): an edge pierces the STRICT
    interior of a triangle when its endpoints lie strictly on opposite
    sides of the triangle's plane and the hit point is strictly inside
    the triangle.
    """
    m = trimesh.load(_mesh_file(subdir))
    m.merge_vertices()
    V, F = np.asarray(m.vertices, float), np.asarray(m.faces)
    pts = []
    for i in range(len(F)):
        for j in range(len(F)):
            if i == j or set(F[i]) & set(F[j]):
                continue                      # adjacent: ordinary contact
            T1, T2 = V[F[i]], V[F[j]]
            n = np.cross(T2[1] - T2[0], T2[2] - T2[0])
            for a, b in ((T1[0], T1[1]), (T1[1], T1[2]), (T1[2], T1[0])):
                da, db = np.dot(a - T2[0], n), np.dot(b - T2[0], n)
                if da * db >= -1e-15:
                    continue                  # not strictly opposite sides
                t = da / (da - db)
                p = a + t * (b - a)
                e1, e2 = T2[1] - T2[0], T2[2] - T2[0]
                ap = p - T2[0]
                d00, d01, d11 = (np.dot(e1, e1), np.dot(e1, e2),
                                 np.dot(e2, e2))
                d20, d21 = np.dot(ap, e1), np.dot(ap, e2)
                den = d00 * d11 - d01 * d01
                u = (d11 * d20 - d01 * d21) / den
                v = (d00 * d21 - d01 * d20) / den
                eps = 1e-9
                if eps < u and eps < v and u + v < 1 - eps:
                    pts.append(p)
    return pts


class TestSelfIntersectionGeometry:
    """Geometry-level ground truth for the self-intersection fixtures.

    These checks are independent of the examiner's detector (see
    SAMPLE_MESH_AUDIT.md): the crossing fixture must genuinely cross
    (positive-length intersection segment), while the point-contact
    fixture must have zero proper crossings by construction.
    """

    def test_crossing_fixture_has_positive_length_segment(self):
        pts = _interior_pierce_points("self_intersecting")
        assert len(pts) >= 2, (
            f"expected >=2 strict interior pierce points, got {len(pts)}"
        )
        dmax = max(float(np.linalg.norm(p - q)) for p in pts for q in pts)
        assert dmax > 0.1, (
            f"intersection segment must have positive length, max pairwise "
            f"pierce-point distance is {dmax}"
        )

    def test_point_contact_fixture_has_no_proper_crossing(self):
        pts = _interior_pierce_points("self_intersecting_point_contact")
        assert pts == [], (
            f"point-contact fixture must have no proper crossings, got "
            f"{len(pts)} pierce point(s)"
        )


# ===================================================================
# 5. Category-scoped inspection
# ===================================================================


class TestCategoryInspection:
    """Run inspection on specific categories only."""

    @pytest.mark.asyncio
    async def test_topological_only_on_nonmanifold_edge(self):
        server = _server()
        load, inspect, _ = await _load_inspect_get_issues(
            server,
            "non_manifold_edge",
        )
        assert inspect["passed"] is False
        assert inspect["summary"]["total_issues"] >= 1

    @pytest.mark.asyncio
    async def test_quality_only_on_perfect_mesh(self):
        server = _server()
        req = MCPRequest(
            method="mesh_load",
            params={"file_path": _mesh_file("perfect_triangle")},
        )
        load = (await server.handle_request(req)).result
        mesh_id = load["mesh_id"]

        req = MCPRequest(
            method="mesh_inspect",
            params={"mesh_id": mesh_id, "categories": ["quality"]},
        )
        resp = await server.handle_request(req)
        assert resp.result["passed"] is True


# ===================================================================
# 6. Filtering and pagination
# ===================================================================


class TestFilteringAndPagination:
    """Test issue filtering and pagination on degenerate_triangle."""

    @pytest.mark.asyncio
    async def test_severity_filter(self):
        server = _server()
        load, _, _ = await _load_inspect_get_issues(server, "degenerate_triangle")
        mesh_id = load["mesh_id"]

        resp = await server.handle_request(MCPRequest(
            method="mesh_get_issues",
            params={"mesh_id": mesh_id, "severity": "warning"},
        ))
        result = resp.result
        assert result["total_count"] >= 1
        assert all(i["severity"] == "warning" for i in result["issues"])

    @pytest.mark.asyncio
    async def test_category_filter(self):
        server = _server()
        load, _, _ = await _load_inspect_get_issues(server, "degenerate_triangle")
        mesh_id = load["mesh_id"]

        resp = await server.handle_request(MCPRequest(
            method="mesh_get_issues",
            params={"mesh_id": mesh_id, "category": "quality"},
        ))
        result = resp.result
        assert result["total_count"] >= 1
        assert all(i["category"] == "quality" for i in result["issues"])

    @pytest.mark.asyncio
    async def test_pagination_limit(self):
        server = _server()
        load, _, _ = await _load_inspect_get_issues(server, "degenerate_triangle")
        mesh_id = load["mesh_id"]

        resp = await server.handle_request(MCPRequest(
            method="mesh_get_issues",
            params={"mesh_id": mesh_id, "limit": 1},
        ))
        result = resp.result
        assert result["returned_count"] == 1
        assert result["total_count"] >= 2

    @pytest.mark.asyncio
    async def test_pagination_offset(self):
        server = _server()
        load, _, _ = await _load_inspect_get_issues(server, "degenerate_triangle")
        mesh_id = load["mesh_id"]

        p0 = (await server.handle_request(MCPRequest(
            method="mesh_get_issues",
            params={"mesh_id": mesh_id, "limit": 100},
        ))).result
        p1 = (await server.handle_request(MCPRequest(
            method="mesh_get_issues",
            params={"mesh_id": mesh_id, "limit": 100, "offset": 1},
        ))).result
        assert p1["offset"] == 1
        assert p1["total_count"] == p0["total_count"]
        assert p1["returned_count"] == p0["returned_count"] - 1


# ===================================================================
# 7. Cache management
# ===================================================================


class TestCacheManagement:

    @pytest.mark.asyncio
    async def test_clear_specific_mesh(self):
        server = _server()
        req = MCPRequest(
            method="mesh_load",
            params={"file_path": _mesh_file("perfect_triangle")},
        )
        load = (await server.handle_request(req)).result
        mesh_id = load["mesh_id"]
        assert mesh_id in server._mesh_cache

        resp = await server.handle_request(MCPRequest(
            method="mesh_clear_cache",
            params={"mesh_id": mesh_id},
        ))
        assert resp.result["status"] == "cleared"
        assert mesh_id not in server._mesh_cache

    @pytest.mark.asyncio
    async def test_clear_all(self):
        server = _server()
        for name in ["perfect_triangle", "perfect_square"]:
            await server.handle_request(MCPRequest(
                method="mesh_load",
                params={"file_path": _mesh_file(name)},
            ))

        assert len(server._mesh_cache) == 2
        resp = await server.handle_request(MCPRequest(
            method="mesh_clear_cache", params={},
        ))
        assert resp.result["status"] == "cleared"
        assert len(server._mesh_cache) == 0
        assert len(server._result_cache) == 0
