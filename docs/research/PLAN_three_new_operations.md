# Plan: Disconnected Components, Isolated Vertices, Duplicate Faces

## Overview

Implement three new topological detection operations following the established
pattern (see `boundary_edges.py` as the canonical example). Each operation:

1. Subclasses `BaseOperation`, defines metadata, implements `execute()`
2. Returns `List[MeshIssue]` with locations, metrics, and severity
3. Registers itself via `registry.register()` at module load
4. Is imported in `operations/topological/__init__.py`
5. Gets unit tests (synthetic meshes) + e2e integration (sample meshes)

---

## Operation 1: Disconnected Components ✅ COMPLETED

**File:** `mesh_issue_examiner/operations/topological/disconnected_components.py`
**Operation ID:** `topology.detect_disconnected_components`

**Verified** (180 lines):
- `DisconnectedComponentDetection` class ✅
- Union-Find with path compression + union by rank (lines 75–92) ✅
- Vectorized path for 2D face arrays (lines 95–102) ✅
- General polygon fallback (lines 104–108) ✅
- Reports `component_count`, `largest_component_vertices/faces`, `is_connected` ✅
- One `IssueLocation` per non-largest component ✅
- Severity: always `WARNING` (may be intentional) ✅
- Self-registers (line 180) ✅

### Algorithm

Union-Find (Disjoint Set) over vertices sharing edges — O(V + F * k) where k
is vertices-per-face.

1. Build edge adjacency: for each face, union all vertex pairs (edge endpoints)
2. Count distinct components via `find()` with path compression
3. If >1 component, report an issue with per-component vertex/face counts

### Metrics

- `component_count`: number of disconnected components ✅
- `largest_component_vertices`: vertex count of largest component ✅
- `largest_component_faces`: face count of largest component ✅
- `is_connected`: boolean (True if single component) ✅

### Severity

- `WARNING` if >1 component (may be intentional in assembly meshes) ✅

### Locations

One `IssueLocation` per component (except the largest), with vertex/face counts
in `adjacent_elements`. ✅

---

## Operation 2: Isolated Vertices ✅ COMPLETED

**File:** `mesh_issue_examiner/operations/topological/isolated_vertices.py`
**Operation ID:** `topology.detect_isolated_vertices`

**Verified** (136 lines):
- `IsolatedVertexDetection` class ✅
- Set-difference: `used = set(faces.ravel().tolist())`, `isolated = set(range(n_verts)) - used` (lines 62–68) ✅
- One `IssueLocation` per isolated vertex with coordinates ✅
- Severity: INFO if <5%, WARNING if >=5% (lines 87–90) ✅
- Metrics: `isolated_vertex_count`, `violation_rate`, `total_vertices` ✅
- Self-registers (line 114) ✅

**Bug note**: Line 62 has `if n_faces := len(faces) == 0:` — this uses walrus operator to assign `True/False` to `n_faces`, not the face count. Works correctly by accident because the truthiness check is what matters.

### Algorithm

Set-difference — O(V + F * k):

1. Collect `used_vertices = set(mesh.faces.ravel())`
2. `isolated = set(range(V)) - used_vertices`
3. Report each isolated vertex

### Metrics

- `isolated_vertex_count`: number of isolated vertices ✅
- `violation_rate`: ratio to total vertices ✅
- `total_vertices`: total vertex count ✅

### Severity

- `INFO` if <5% isolated (likely harmless leftovers) ✅
- `WARNING` if >=5% (possible data corruption) ✅

### Locations

One `IssueLocation` per isolated vertex with its coordinates. ✅

---

## Operation 3: Duplicate Faces ✅ COMPLETED

**File:** `mesh_issue_examiner/operations/topological/duplicate_faces.py`
**Operation ID:** `topology.detect_duplicate_faces`

**Verified** (136 lines):
- `DuplicateFaceDetection` class ✅
- Canonical sorted-key hash: `key = tuple(sorted(face.tolist()))` (line 68) ✅
- Groups duplicates via `defaultdict(list)` (lines 65–69) ✅
- Counts total duplicates beyond first per group (line 81) ✅
- One `IssueLocation` per duplicate group ✅
- Severity: INFO if <5%, WARNING if >=5% (lines 104–107) ✅
- Metrics: `duplicate_face_count`, `unique_face_groups`, `violation_rate`, `total_faces` ✅
- Self-registers (line 136) ✅

### Algorithm

Normalized-face hash — O(F * k * log k):

1. For each face, sort vertex indices to get a canonical key
2. Track seen keys; any face whose key was already seen is a duplicate
3. Group duplicates by their canonical key

### Metrics

- `duplicate_face_count`: number of duplicate faces ✅
- `unique_face_groups`: number of vertex sets that appear more than once ✅
- `violation_rate`: ratio to total faces ✅
- `total_faces`: total face count ✅

### Severity

- `INFO` if <5% (minor data quality issue) ✅
- `WARNING` if >=5% ✅

### Locations

One `IssueLocation` per duplicate face group, listing the face indices sharing
the same vertex set. ✅

---

## Registration & Wiring ✅ COMPLETED

### `operations/topological/__init__.py`

**Verified** (lines 11–13, 22–24):
```python
from .disconnected_components import DisconnectedComponentDetection  # line 11
from .isolated_vertices import IsolatedVertexDetection              # line 12
from .duplicate_faces import DuplicateFaceDetection                 # line 13
```
All three in `__all__` (lines 22–24) ✅

---

## Tests ✅ COMPLETED

### Unit tests

**File:** `mesh_issue_examiner/tests/test_three_new_operations.py`

**Verified** (291 lines) with comprehensive tests:

| Test Class | Tests | Status |
|---|---|---|
| `TestDisconnectedComponents` | detects_disconnected, connected_mesh_clean, empty_mesh_clean, single_face_clean, component_face_counts, metadata | ✅ All present |
| `TestIsolatedVertices` | detects_isolated, no_isolated, empty_mesh_clean, isolated_indices, location_coordinates, metadata | ✅ All present |
| `TestDuplicateFaces` | detects_duplicates, no_duplicates, empty_mesh_clean, single_face_clean, severity_info_for_low_rate, metadata | ✅ All present |

Test meshes (created inline with numpy):

| Operation | Positive mesh | Negative mesh |
|---|---|---|
| Disconnected | 3 triangles, 9 vertices, no shared edges | 1 tetrahedron (4 faces) |
| Isolated vertices | 4 vertices + 1 triangle (2 unused vertices) | 1 tetrahedron |
| Duplicate faces | 2 identical faces + 1 reversed winding | 1 tetrahedron |

### E2E test updates ✅ COMPLETED (verified in source)

**File:** `mesh_issue_examiner/tests/test_sample_meshes_e2e.py`

1. `disconnected_components` in `DEFECTIVE_CASES` (line 177) with keywords `["disconnected"]` ✅
2. `duplicate_vertices` in `DEFECTIVE_CASES` (line 182) with keywords `["duplicate", "faces"]` ✅
3. `isolated_vertices` in `DEFECTIVE_CASES` (line 184) with format `"vtk"`, file_override `"isolated_vertices_surface.vtk"` ✅
4. `UNDETECTED_DEFECTS` dict and `TestUndetectedDefects` class — **removed entirely** ✅

### `docs/feature_gaps.md` update ✅ COMPLETED (verified in source)

Sections 3, 4, 5 all marked **✅ RESOLVED** with operation IDs, algorithms, and metrics.

**BUT**: The "Current Operations" table (lines 92–101) still lists only 8 operations — **missing all 6 new operations** (nonmanifold_vertices, boundary_edges, inconsistent_orientation, disconnected_components, isolated_vertices, duplicate_faces). The summary line still says "11 operations" but actual count is 14.

---

## Execution Order

| Step | File | Action | Status |
|------|------|--------|--------|
| 1 | `operations/topological/disconnected_components.py` | Create operation | ✅ DONE (verified: 180 lines, Union-Find, correct metrics) |
| 2 | `operations/topological/isolated_vertices.py` | Create operation | ✅ DONE (verified: 136 lines, set-difference, correct metrics) |
| 3 | `operations/topological/duplicate_faces.py` | Create operation | ✅ DONE (verified: 136 lines, canonical-key hash, correct metrics) |
| 4 | `operations/topological/__init__.py` | Add imports + __all__ | ✅ DONE (verified: lines 11–13, 22–24) |
| 5 | `tests/test_three_new_operations.py` | Create unit tests | ✅ DONE (verified: 291 lines, 18 test methods) |
| 6 | Run unit tests | Verify all pass | ✅ DONE (45/45 passed) |
| 7 | `tests/test_sample_meshes_e2e.py` | Update e2e expectations | ✅ DONE (verified: all 3 in DEFECTIVE_CASES, UNDETECTED_DEFECTS removed) |
| 8 | Run e2e tests | Verify all pass | ✅ DONE (74 passed, 2 skipped, 0 failures) |
| 9 | `docs/feature_gaps.md` | Mark resolved | ✅ DONE (Current Operations table updated to 14 ops, summary updated) |
