# Implementation Plan: Surface Orientation Check

## Overview

Add a surface orientation consistency check operation to detect faces with inconsistent winding order (flipped normals) in surface meshes. This addresses the feature gap documented in `docs/feature_gaps.md` section 2 (Flipped / inconsistent normals).

**Target Operation ID:** `topology.detect_inconsistent_orientation`

**Category:** `IssueCategory.ORIENTATION`

---

## Background & Research

### The Problem
- Surface meshes should have consistent face orientation (all normals pointing consistently)
- Inconsistent normals break downstream tasks: rendering, simulation, slicing, boolean operations
- The existing `quality_3d.check_orientation_consistency` only works on volumetric cells
- Thingi10K dataset: 10.3% of meshes have orientation issues (8965/9995 are properly oriented)

### Algorithm (from research)

**Core Principle:** For two adjacent faces sharing an edge, the shared edge must be traversed in opposite directions:
- Face A has edge with vertices in order (a → b)
- Adjacent Face B must have the same edge in order (b → a)
- If both have (a → b) or both have (b → a), one face has flipped orientation

**Implementation Steps:**
1. Build edge-to-faces adjacency map
2. For each edge shared by exactly 2 faces, compare vertex ordering
3. Perform BFS/DFS traversal from a seed face to identify connected components
4. Track faces where winding consistency is violated
5. Report inconsistent faces with locations and metrics

---

## Precise References

### 1. StackOverflow Algorithm Description
**Source:** StackOverflow Answer - "how to check if all the faces face outward"  
**URL:** https://stackoverflow.com/questions/77927993/how-to-check-if-all-the-faces-face-outward  
**Access Date:** 2026-04-10  
**Key Algorithm:**
> "To process a face, first add it to the 'closed set'. Now examine all faces it shares an edge with. If a face has an edge with vertices (in counterclockwise order) `a,b`, then the other face (and it's manifold, so there should be only one at most) with an edge with those vertices should have them in the order `b,a`. If the other face has them in the order `a,b`, flip its winding."

**Core Concept:** Edge traversal direction must be opposite for adjacent faces.

### 2. Academic Paper - Borodin et al.
**Citation:**
```
Pavel Borodin, Gabriel Zachmann, Reinhard Klein.
"Consistent Normal Orientation for Polygonal Meshes"
Computer Graphics International (CGI), June 16-19, 2004, Crete, Greece.
IEEE Computer Society Press, pp. 18-25.
```

**URLs:**
- Paper PDF: https://user.informatik.uni-bremen.de/zach/papers/orientation_print.pdf
- Project Page: https://user.math.uni-bremen.de/zach/papers/orientation.html

**Key Concepts from Paper:**
- Section 3.1 (Outline): "Since the orientation of each polygon's normal can be encoded in its vertex order, the problem of orientation of normals is equivalent to the problem of ordering vertices in polygons consistently."
- Section 3.2 (Detection of Patches): "As boundary edges we define all edges which are incident to only one polygon... we build the neighbourhood information for the mesh... divide it into a set of manifold surface patches... For each patch we consistently orient all polygons belonging to it"
- Section 3.3 (Boundary Coherence): "The idea of the boundary coherence coefficient is that it can give a hint as to which patches should probably be oriented consistently."

**BibTeX:**
```bibtex
@INPROCEEDINGS{Zach04a,
    author = "Pavel Borodin and Gabriel Zachmann and Reinhard Klein",
    title = "Consistent Normal Orientation for Polygonal Meshes",
    booktitle = "Computer Graphics International (CGI)",
    year = "2004",
    month = "June 16--19",
    publisher = "IEEE Computer Society Press",
    address = "Crete, Greece",
    pages = "18--25",
    keywords = "normal orientation, mesh repair, visibility, patch connectivity, modelling"
}
```

### 3. Related Work - CGAL Documentation
**Source:** CGAL Polygon Mesh Processing - Orientation  
**URL:** https://doc.cgal.org/latest/Polygon_mesh_processing/index.html  
**Concept:** "Orientation: checking or fixing the orientation of a polygon soup"

### 4. Industry Context - MeshInspector
**Source:** MeshInspector Knowledge Base  
**URL:** https://meshinspector.com/knowledge-base/mesh-repair/convenient-mesh-normal-orientation-with-meshinspector/  
**Key Point:** "Proper normal orientation is step zero for many downstream operations, which includes mesh repair, boolean operations, segmentation, and other geometry-processing algorithms."

---

## Implementation Details

### 1. New File: `mesh_issue_examiner/operations/topological/surface_orientation.py` ✅ COMPLETED (verified in source)

**Verified** (423 lines): `SurfaceOrientationCheck` class with:
- `operation_id='topology.detect_inconsistent_orientation'` (line 106) ✅
- `category=IssueCategory.ORIENTATION` ✅
- Module docstring cites StackOverflow + Borodin et al. (lines 1–57) ✅
- `_build_edge_face_map()` — edge-to-(face_idx, edge_direction) map (lines 233–286) ✅
- `_are_directions_opposite()` — core StackOverflow algorithm (lines 288–306) ✅
- `_build_face_adjacency()` — face adjacency graph (lines 308–333) ✅
- `_find_connected_components()` — BFS per Borodin et al. patches (lines 335–378) ✅
- Handles non-manifold edges (skips >2 faces), boundary edges (skips 1 face), single faces ✅
- Metrics: `inconsistent_face_count`, `total_faces`, `violation_rate`, `inconsistent_edge_count`, `affected_component_count`, `total_component_count` ✅
- Self-registers via `registry.register()` (line 423) ✅

### 2. Update `mesh_issue_examiner/operations/topological/__init__.py` ✅ COMPLETED (verified in source)

- `from .surface_orientation import SurfaceOrientationCheck` — line 10 ✅
- Listed in `__all__` — line 21 ✅

### 3. Update E2E Test: `mesh_issue_examiner/tests/test_sample_meshes_e2e.py` ✅ COMPLETED (verified in source)

- `"flipped_normals"` in `DEFECTIVE_CASES` dict (line 173) with `expect_passed: False` and keywords `["orientation", "inconsistent"]` ✅
- `TestUndetectedDefects` class and `UNDETECTED_DEFECTS` — removed entirely ✅

### 4. Test File: `mesh_issue_examiner/tests/test_surface_orientation.py` ✅ COMPLETED (verified in source)

**Verified** (382 lines) with test classes `TestSurfaceOrientationCheck` and `TestSurfaceOrientationEdgeCases`:
- `test_consistent_mesh_no_issues` — two triangles with consistent winding ✅
- `test_inconsistent_orientation_detected` — two triangles same winding on shared edge ✅
- `test_single_face_no_issues` ✅
- `test_boundary_edges_skipped` — per Borodin et al. ✅
- `test_non_manifold_edges_skipped` — 3 faces sharing one edge ✅
- `test_disconnected_components` — independent analysis per patch ✅
- `test_cube_consistent` — 12-face triangulated cube, no issues ✅
- `test_flipped_face_in_cube` — one face flipped, detects >=2 inconsistent faces ✅
- `test_empty_mesh` ✅
- `test_metrics_structure` — verifies all metric keys and values ✅
- `test_operation_metadata` — verifies operation_id, category, tags ✅
- `test_polygonal_faces` — mixed quad/tri faces ✅
- `test_degenerate_face_handling` — collinear points ✅
- `test_large_mesh_performance` — 50x50 grid (5000 triangles) ✅

### 5. Update Documentation: `docs/feature_gaps.md` ✅ COMPLETED (verified in source)

- Section 2 "Flipped / inconsistent normals" → **✅ RESOLVED** ✅
- Includes Operation ID, implementation date, algorithm references (StackOverflow + Borodin et al.), metrics, validation ✅
- Impact Assessment table: "Flipped normals / orientation" → **✅ RESOLVED** ✅
- NOTE: The "Current Operations" table (lines 92–101) does **not** list `topology.detect_inconsistent_orientation` — table is outdated

---

## Sample Mesh Analysis

**File:** `e2e/sample_meshes/flipped_normals/flipped_normals.obj`
```
v 0.000000 0.000000 0.000000  # v0
v 1.000000 0.000000 0.000000  # v1
v 0.500000 1.000000 0.000000  # v2
v 1.500000 0.000000 0.000000  # v3
v 2.500000 0.000000 0.000000  # v4
v 2.000000 1.000000 0.000000  # v5
f 1 2 3  # Face 0: (v0,v1,v2) - normal points +Z
f 4 6 5  # Face 1: (v3,v5,v4) - normal points -Z (flipped!)
```

Face 0 normal: +Z (counter-clockwise winding)  
Face 1 normal: -Z (clockwise winding - inconsistent!)

The two faces don't share an edge, so they form separate components. The operation should report that the mesh has 2 components with inconsistent orientations between them.

---

## File Structure

```
mesh_issue_examiner/
├── operations/
│   └── topological/
│       ├── __init__.py              # Add import
│       ├── surface_orientation.py   # NEW (with reference citations)
│       └── ...
├── tests/
│   ├── test_sample_meshes_e2e.py    # Update test expectations
│   └── test_surface_orientation.py  # NEW (unit tests)
└── ...
docs/
    └── feature_gaps.md              # Update status with references
```

---

## Verification Steps

1. Run unit tests:
   ```bash
   conda run -n PEEC python -m pytest mesh_issue_examiner/tests/test_surface_orientation.py -v
   ```

2. Run e2e tests:
   ```bash
   conda run -n PEEC python -m pytest mesh_issue_examiner/tests/test_sample_meshes_e2e.py::TestDefectiveMeshes -v
   ```

3. Verify flipped_normals is now detected:
   ```bash
   conda run -n PEEC python -m pytest mesh_issue_examiner/tests/test_sample_meshes_e2e.py::TestDefectiveMeshes::test_fails_inspection[flipped_normals] -v
   ```

---

## Dependencies

- numpy: For array operations
- Existing mesh_issue_examiner core modules: models, base operation, registry

---

## Edge Cases to Handle

1. **Non-manifold edges:** Skip edges with >2 incident faces (non-manifold) - per Borodin et al.
2. **Boundary edges:** Skip edges with only 1 incident face (boundary)
3. **Single face meshes:** No shared edges to validate against
4. **Degenerate faces:** Zero-area faces (skip or report separately)
5. **Disconnected components:** Each component can have independent orientation (Borodin et al. patch concept)

---

## Reference Summary Table

| Reference | Type | Key Contribution | URL |
|-----------|------|------------------|-----|
| StackOverflow Answer | Practical Algorithm | Edge traversal direction check | https://stackoverflow.com/questions/77927993/how-to-check-if-all-the-faces-face-outward |
| Borodin et al. 2004 | Academic Paper | Formal definition, patch-based orientation | https://user.informatik.uni-bremen.de/zach/papers/orientation_print.pdf |
| CGAL PMP | Library Documentation | Polygon soup orientation | https://doc.cgal.org/latest/Polygon_mesh_processing/ |
| MeshInspector | Industry Tool | Context and motivation | https://meshinspector.com/knowledge-base/mesh-repair/ |

---

## Estimated Effort

- Implementation: ~2-3 hours
- Unit tests: ~1 hour
- E2E test updates: ~30 minutes
- Documentation: ~30 minutes

**Total: ~4-5 hours**
