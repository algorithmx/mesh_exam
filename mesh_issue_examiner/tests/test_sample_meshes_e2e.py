"""
End-to-end tests loading real mesh files from e2e/sample_meshes/
through the full pipeline: load -> inspect -> get_issues -> get_quality_metrics.

The fixture files are deterministic and NOT committed (e2e/ is gitignored).
They are regenerated automatically on the first test session by
``tests/conftest.py`` (via scripts/generate_sample_meshes.py, which skips
files that already exist). To rebuild them by hand:

    python scripts/generate_sample_meshes.py --force   # recreate + verify
    python scripts/generate_sample_meshes.py --verify  # verify only

Run with:
    conda run -n PEEC python -m pytest mesh_issue_examiner/tests/test_sample_meshes_e2e.py -v
"""

from pathlib import Path

import pytest

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
        """Tessellated meshes produce quality violations from face shape."""
        server = _server()
        _, inspect, issues = await _load_inspect_get_issues(server, mesh_name)
        assert issues["total_count"] > 0, (
            f"{mesh_name}: expected quality issues from tessellation"
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
        "expect_keywords": [],
    },
    "varying_quality": {
        "expect_passed": False,
        "expect_keywords": [],
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
    # Clean 2x2x2 hex grid: the three connectivity ops must stay silent.
    # The legacy quality_3d ops (check_orientation_consistency,
    # check_boundary_layer) misfire on regular hex grids, so passed is
    # False for every volume fixture (see docs/volumetric-mesh-gaps-report.md).
    "closed_volume": {
        "expect_passed": False,
        "expect_keywords": [],
        "forbid_keywords": ["duplicate_cells", "orphaned_cells",
                            "boundary_closure", "nonmanifold_faces"],
    },
    "duplicate_cell": {
        "expect_passed": False,
        "expect_keywords": ["duplicate_cells", "boundary_closure"],
        "forbid_keywords": ["orphaned_cells"],
    },
    "leaky_volume": {
        "expect_passed": False,
        "expect_keywords": ["boundary_closure"],
        "forbid_keywords": ["duplicate_cells", "orphaned_cells",
                            "nonmanifold_faces"],
    },
    # Clean quadratic (tet10) volume: connectivity ops stay silent and
    # vtkCellQuality's -1 sentinel for quadratic tets must not be read as
    # an inverted cell.
    "tet10_volume": {
        "expect_passed": True,
        "expect_keywords": [],
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
        assert inspect["summary"]["total_issues"] > 0

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
# 3c. NASTRAN mixed volumetric deck (.nas): CTETRA + CHEXA + CPENTA
# ===================================================================

class TestNastranMixedVolume:
    """The nas_mixed_volume fixture exercises CHEXA/CPENTA parsing."""

    @pytest.mark.asyncio
    async def test_mixed_element_types_parsed(self):
        """CTETRA/CHEXA/CPENTA map to tet4/hex8/prism6 volume cells.

        The fixture's CHEXA uses the 20-node form with continuation cards;
        the parser must keep the 8 corners and drop the midside nodes
        (18 vertices: 4 tet + 8 hex + 6 wedge corners).
        """
        from mesh_issue_examiner.utils.mesh_loader import load_mesh_from_file

        mesh = load_mesh_from_file(_mesh_file("nas_mixed_volume", fmt="nas"))
        assert mesh.has_volume_cells
        assert set(mesh.cell_types) == {"tet4", "hex8", "prism6"}
        assert mesh.volume_cells["tet4"].shape == (1, 4)
        assert mesh.volume_cells["hex8"].shape == (1, 8)
        assert mesh.volume_cells["prism6"].shape == (1, 6)
        assert mesh.n_vertices == 18  # midside nodes dropped
        assert mesh.n_volume_cells == 3

    @pytest.mark.asyncio
    async def test_pipeline_runs_and_connectivity_ops_silent(self):
        """Full MCP pipeline on the .nas deck: three disjoint cells are
        detected as disconnected, but no connectivity defects fire."""
        server = _server()
        load, inspect, issues = await _load_inspect_get_issues(
            server, "nas_mixed_volume", fmt="nas")
        assert load["format"] == "nas"
        assert load["n_vertices"] == 18
        assert inspect["passed"] is False

        op_ids = {i["operation_id"] for i in issues["issues"]}
        assert any("disconnected" in oid for oid in op_ids), op_ids
        for kw in ("duplicate_cells", "orphaned_cells", "boundary_closure",
                   "nonmanifold_faces", "check_orientation_consistency"):
            assert not any(kw in oid for oid in op_ids), (
                f"nas_mixed_volume: unexpected '{kw}' in operation IDs: {op_ids}"
            )


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
    "shell_structure",  # Thin-walled cube, actually watertight
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
