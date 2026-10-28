"""
Large-mesh performance smoke test: ~1M hex8 cells through a full inspect.

Deselected from the default suite via ``pytest.ini`` (``-m "not slow"``);
run explicitly with::

    python -m pytest mesh_issue_examiner/tests/test_large_mesh_smoke.py -m slow -q

The mesh is a 100x100x100 structured hex8 grid (1,000,000 cells,
1,030,301 vertices) built in-memory with numpy — pyvista's
``ImageData.cast_to_unstructured_grid`` produces VTK VOXEL cells, which
the adapter's type map does not extract as volume cells, so the grid is
constructed directly in canonical form.  The boundary surface is the
six outer slabs, triangulated with outward-consistent orientation.
"""

import asyncio

import numpy as np
import pytest

from mesh_issue_examiner.core.models import MeshInput
from mesh_issue_examiner.mcp.server import MeshIssueExaminerServer, MCPRequest

pytestmark = pytest.mark.slow

N = 100  # N^3 hex8 cells


def _structured_hex_mesh(n: int) -> MeshInput:
    """Structured n^3 hex8 grid in canonical form (unit-spacing cells)."""
    side = n + 1
    axis = np.arange(side, dtype=np.float64)

    # Vertex ids follow ravel order of the meshgrid below: i*side^2 + j*side + k
    def vid(i, j, k):
        return (i * side + j) * side + k

    gx, gy, gz = np.meshgrid(axis, axis, axis, indexing="ij")
    vertices = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])
    assert vertices.shape == (side ** 3, 3)

    # hex8 connectivity (VTK corner order: bottom ring then top ring)
    ci, cj, ck = np.meshgrid(
        np.arange(n), np.arange(n), np.arange(n), indexing="ij")
    ci, cj, ck = ci.ravel(), cj.ravel(), ck.ravel()
    cells = np.column_stack([
        vid(ci, cj, ck), vid(ci + 1, cj, ck),
        vid(ci + 1, cj + 1, ck), vid(ci, cj + 1, ck),
        vid(ci, cj, ck + 1), vid(ci + 1, cj, ck + 1),
        vid(ci + 1, cj + 1, ck + 1), vid(ci, cj + 1, ck + 1),
    ]).astype(np.int64)

    # Boundary surface: the six outer slabs, quads triangulated as
    # (a,b,c)+(a,c,d) with outward-oriented winding.
    def quads(a, b, c, d):
        return np.vstack([
            np.column_stack([a, b, c]),
            np.column_stack([a, c, d]),
        ]).astype(np.int64)

    idx = np.arange(n)
    ii, jj = np.meshgrid(idx, idx, indexing="ij")
    ii, jj = ii.ravel(), jj.ravel()
    jk1, jk2 = np.meshgrid(idx, idx, indexing="ij")
    jk1, jk2 = jk1.ravel(), jk2.ravel()

    faces = np.vstack([
        quads(vid(0 * jk1, jk1, jk2),          vid(0 * jk1, jk1, jk2 + 1),
              vid(0 * jk1, jk1 + 1, jk2 + 1),  vid(0 * jk1, jk1 + 1, jk2)),
        quads(vid(n + 0 * jk1, jk1, jk2),          vid(n + 0 * jk1, jk1 + 1, jk2),
              vid(n + 0 * jk1, jk1 + 1, jk2 + 1),  vid(n + 0 * jk1, jk1, jk2 + 1)),
        quads(vid(ii, 0 * jj, jj),          vid(ii + 1, 0 * jj, jj),
              vid(ii + 1, 0 * jj, jj + 1),  vid(ii, 0 * jj, jj + 1)),
        quads(vid(ii, n + 0 * jj, jj),          vid(ii, n + 0 * jj, jj + 1),
              vid(ii + 1, n + 0 * jj, jj + 1),  vid(ii + 1, n + 0 * jj, jj)),
        quads(vid(ii, jj, 0),          vid(ii, jj + 1, 0),
              vid(ii + 1, jj + 1, 0),  vid(ii + 1, jj, 0)),
        quads(vid(ii, jj, n),          vid(ii + 1, jj, n),
              vid(ii + 1, jj + 1, n),  vid(ii, jj + 1, n)),
    ])

    return MeshInput(
        vertices=vertices,
        faces=faces,
        mesh_id=f"hex_grid_{n}cubed",
        source_format="arrays",
        volume_cells={"hex8": cells},
    )


class TestMillionHexFullInspect:
    """A 100^3 hex8 grid must survive a full inspection run."""

    def test_mesh_shape(self):
        """Construction sanity before the (expensive) inspect."""
        mesh = _structured_hex_mesh(3)
        assert mesh.n_vertices == 4 ** 3
        assert mesh.volume_cells["hex8"].shape == (27, 8)
        assert mesh.faces.shape == (6 * 3 * 3 * 2, 3)

    def test_full_inspect_completes_and_passes(self):
        mesh = _structured_hex_mesh(N)
        assert mesh.n_volume_cells == N ** 3
        assert mesh.n_vertices == (N + 1) ** 3

        server = MeshIssueExaminerServer()
        server._mesh_cache[mesh.mesh_id] = mesh

        resp = asyncio.run(server.handle_request(MCPRequest(
            method="mesh_inspect", params={"mesh_id": mesh.mesh_id})))
        assert resp.error is None, resp.error
        assert resp.result["passed"] is True, resp.result["summary"]
        assert resp.result["processing_time"] > 0
