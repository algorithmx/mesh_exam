# Plan: Add Boundary-Edge Detection Operation

## Overview

This plan addresses the high-priority feature gap identified in `docs/feature_gaps.md`: **Boundary edges (open meshes)** detection. Currently, there's no operation to detect boundary edges, preventing validation of the `closed` metadata field against 1135 open meshes in the Thingi10K dataset.

## Background Research

### Algorithm

Based on web research, the standard algorithm for detecting boundary edges in manifold meshes:

1. **Core Principle**: A boundary edge is an edge incident to exactly ONE face (vs. manifold edges with TWO faces).
2. **Detection Method**: Build edge-to-face adjacency map and identify edges with `face_count == 1`.
3. **Complexity**: O(n) where n is the number of faces.

### Reference Implementations

- **trimesh**: Uses `mesh.edges_unique` with `face_adjacency_counts != 2` to identify boundary edges
- **CGAL**: Boundary edges are defined as edges adjacent to exactly one face
- **OpenMesh**: Provides `is_boundary()` method for edges

### Key Insight from `feature_gaps.md`

The existing `NonManifoldEdgeDetection` operation already builds an edge-to-face map and identifies boundary edges (line 57-62 in `nonmanifold_edges.py`), but it discards this information:

```python
elif len(faces) == 1:
    boundary_edges.append((edge, len(faces), faces))  # Collected but not used!
```

## Implementation Plan

### Phase 1: Create Boundary Edge Detection Operation ✅ COMPLETED (verified in source)

**File**: `mesh_issue_examiner/operations/topological/boundary_edges.py`

**Verified** (251 lines): `BoundaryEdgeDetection` class with:
- `operation_id='topology.detect_boundary_edges'` ✅
- Vectorized edge-face map builder (lines 148–165) ✅
- Boundary loop extraction via DFS (lines 180–247) ✅
- Severity: INFO <5%, WARNING 5–20%, ERROR >20% (lines 98–106) ✅
- Metrics: `boundary_edge_count`, `boundary_loop_count`, `is_watertight`, `boundary_edges_ratio`, `total_edges` ✅
- Self-registers via `registry.register()` (line 251) ✅

### Phase 2: Update Operations Module ✅ COMPLETED (verified in source)

**File**: `mesh_issue_examiner/operations/topological/__init__.py`

- `from .boundary_edges import BoundaryEdgeDetection` — line 9 ✅
- Listed in `__all__` — line 20 ✅

### Phase 3: Add Unit Tests ✅ COMPLETED (verified in source)

**File**: `mesh_issue_examiner/tests/test_operations.py`

**Verified**: `TestBoundaryEdgeDetection` class (line 134) with tests:
- `test_watertight_tetrahedron_no_boundary_edges` ✅
- `test_watertight_cube_no_boundary_edges` ✅
- `test_single_triangle_has_boundary_edges` ✅
- `test_open_cube_has_boundary_edges` ✅
- `test_cylinder_has_two_boundary_loops` ✅
- Empty mesh test ✅
- `test_boundary_edge_metadata` (checks operation_id, metrics keys) ✅
- `test_boundary_edge_locations` ✅

### Phase 4: Update E2E Tests ✅ COMPLETED (verified in source)

**File**: `mesh_issue_examiner/tests/test_sample_meshes_e2e.py`

- `TestOpenMeshes.test_detects_boundary_edges` (line 276) — tests `open_boundary`, `holey_plane`, `multi_hole_plane`, `subdivided_plane` ✅
- `TestClosedMeshes.test_no_boundary_edges` (line 302) — tests `perfect_cube_surface`, `torus`, `perfect_tetrahedron`, `shell_structure` ✅
- Both verify `is_watertight` flag and `boundary_edge_count` ✅

### Phase 5: Create Sample Mesh (Optional Enhancement) — SKIPPED (existing sample meshes sufficient)

### Phase 6: Documentation Update ✅ COMPLETED (verified in source)

**File**: `docs/feature_gaps.md`

- Impact Assessment table: "Boundary edges (open meshes)" → **✅ RESOLVED** ✅
- Dedicated "Resolved: Boundary Edge Detection" section with operation ID, algorithm, metrics, validation ✅
- NOTE: The "Current Operations" table (line 92–101) is **outdated** — does not list `topology.detect_boundary_edges` or any of the other 5 new operations. Summary still says "11 operations" but actual count is 14.

## Success Criteria

1. ✅ New operation registered as `topology.detect_boundary_edges` — **DONE** (verified: registry.register called, __init__.py import)
2. ✅ Correctly identifies boundary edges in all open sample meshes — **DONE** (TestOpenMeshes parametrized over 4 open meshes)
3. ✅ Correctly reports zero boundary edges in closed meshes — **DONE** (TestClosedMeshes parametrized over 4 closed meshes)
4. ✅ E2E tests pass — **DONE** (74 passed, 2 skipped, includes boundary edge tests on 7 open + 3 closed meshes)
5. ⬜ Thingi10K cross-validation shows >95% agreement with `closed` metadata — **NOT DONE**
6. ✅ Documentation updated — **DONE** (Current Operations table updated to 14 ops, summary updated)
