# Gaps in 3D Volumetric Mesh Quality Check Code

**Report Date:** 2026-04-10
**Last Updated:** 2026-10-28
**Component:** mesh_issue_examiner - Volumetric (3D) Mesh Operations
**Status:** 11 Gaps Closed · 1 Partially Closed · 1 Open

---

## Executive Summary

The mesh_issue_examiner framework has foundational support for volumetric (3D) mesh quality assessment with significant progress toward production readiness. All critical gaps for basic volumetric mesh operations have been addressed, with support for all 4 linear element types (tet4, hex8, prism6, pyramid5) in key operations.

### Key Findings
- **5 Critical Gaps** closed — all basic volumetric operations now support all element types
- **5 Adapters** — **4 support volumetric meshes** (PyVista, Meshio, Vedo, PyMeshLab); **1 extracts surface boundary** (Trimesh)
- NonManifoldFaceDetection **implemented and registered**
- Winding Consistency **extended to all element types** (tet4, hex8, prism6, pyramid5)
- **4 linear + 7 quadratic element types** now supported (tet4/10, hex8/20/27, prism6/15, wedge18, pyramid5/13/14)
- **Quadratic topology checks enabled** via corner-first face resolution (`_face_tables.py`): non-manifold faces, orientation winding, duplicate/orphaned cells, boundary closure all analyze quadratic cells through their corner nodes
- **Sliver detection extended** to all 4 linear element types (tet4, hex8, prism6, pyramid5)
- **4 new topological operations** added: DisconnectedComponentDetection, IsolatedVertexDetection, DuplicateFaceDetection, DuplicateVertexDetection
- **3 new volumetric connectivity operations** added (2026-10-27): DuplicateCellDetection, OrphanedCellDetection, BoundaryClosureCheck
- **6 extended quality metrics** implemented (warpage, equiangle_skew, stretch, taper, radius_ratio, collapse_ratio) with thresholds and element applicability
- **Quadratic quality via corner linearization** (2026-10-28): vtkCellQuality's -1 sentinel on quadratic cells is bypassed by recomputing metrics on corner-only linear grids, tagged `linearized` in metric metadata; unblocks pyramid14 (pyramid5 corners)
- **Boundary-layer checks implemented and calibrated** (2026-10-28): `quality_3d.check_boundary_layer` with wall-seed gating, growth-ratio and coverage checks; clean prism stacks pass, defective growth fires ERROR; hex grids stay silent
- **Multi-material end-to-end** (2026-10-28): `material_array_name` exposed on both MCP layers; interface detection, interface quality and per-material statistics fire on the `two_material_volume` fixture
- **Clean volumes pass full inspection** (2026-10-28): face-winding tables normalised to the outward convention and volume-aware 2D surface ops, so `closed_volume` reports zero issues
- **NASTRAN higher-order retention** (2026-10-28): CHEXA20 → hex20 and CPENTA15 → prism15 with midside nodes kept (ordering matches VTK); corner reduction only for forms with no registered type
- **Remaining open** — parallelization for very large meshes only; dedicated Gmsh/CGNS/Exodus readers stay delegated to meshio by design

---

## 🔴 Critical Gaps (Blocking Production Use)

### 1. NonManifoldFaceDetection Operation ✅ IMPLEMENTED

**Location:** `mesh_issue_examiner/operations/volumetric/nonmanifold_faces.py`

**Status:** ✅ **COMPLETED** (2026-03-31)

**Implementation Summary:**
- Detects faces shared by more than 2 volume cells
- Supports all 4 element types: tet4, hex8, prism6, pyramid5
- Uses sorted vertex tuples for face matching
- Reports cell indices, face vertices, and severity based on violation rate

**Key Features:**
1. `ELEMENT_FACES` dictionary defines face topologies for each element type (shared with `_face_tables.py`)
2. `_build_face_to_cell_map()` creates adjacency using canonical (sorted) face keys
3. Identifies faces shared by >2 cells as non-manifold
4. Creates detailed `IssueLocation` objects with face vertices and sharing cell count

**Tests Added:** 13 comprehensive tests covering:
- Single elements (no issues)
- Two cells sharing a face (manifold)
- Three cells sharing a face (non-manifold detection)
- All element types (tet4, hex8, prism6, pyramid5)
- Issue metadata and metrics validation

**Registration:**
- Operation is now registered: `topology_3d.detect_nonmanifold_faces`
- Added to `mcp/server.py` imports for auto-registration

**Implementation Summary:**
- Detects faces shared by more than 2 volume cells
- Supports all 4 element types: tet4, hex8, prism6, pyramid5
- Uses sorted vertex tuples for face matching
- Reports cell indices, face vertices, and severity based on violation rate

**Key Features:**
1. `ELEMENT_FACES` dictionary defines face topologies for each element type
2. `_build_face_to_cell_map()` creates adjacency using canonical (sorted) face keys
3. Identifies faces shared by >2 cells as non-manifold
4. Creates detailed `IssueLocation` objects with face vertices and sharing cell count

**Tests Added:** 13 comprehensive tests covering:
- Single elements (no issues)
- Two cells sharing a face (manifold)
- Three cells sharing a face (non-manifold detection)
- All element types (tet4, hex8, prism6, pyramid5)
- Issue metadata and metrics validation

**Registration:**
- Operation is now registered: `topology_3d.detect_nonmanifold_faces`
- Added to `mcp/server.py` imports for auto-registration

---

### 2. Adapter Support Limited to PyVista Only ✅ RESOLVED

**Location:** `mesh_issue_examiner/adapters/`

**Status:** ✅ **COMPLETED** (2026-04-10)

| Adapter | Surface Mesh | Volumetric Mesh | Status |
|---------|-------------|-----------------|--------|
| PyVista | ✅ Full | ✅ Full | Production Ready |
| Meshio | ✅ Full | ✅ Full (via PyVista caching) | Production Ready |
| Vedo | ✅ Full | ✅ Full (via PyVista caching) | Production Ready |
| PyMeshLab | ✅ Full | ✅ Full (via PyVista caching) | Production Ready |
| Trimesh | ✅ Full | ⚠️ Surface boundary extraction | Graceful degradation |

**What was done:**
- Meshio, Vedo, and PyMeshLab adapters gained full volumetric support with PyVista caching for quality metrics
- Trimesh adapter extracts surface boundary from volumetric inputs via PyVista delegation
- Capability-based architecture (`AdapterCapability` flag enum) added for adapter selection
- Standardized exception hierarchy and factory functions added
- All adapters now declare their capabilities via the `capabilities` property

**Capability Matrix:**
- PyVista: `SURFACE_MESH | VOLUMETRIC_MESH | FILE_IO | QUALITY_METRICS | TOPOLOGY_CHECKS | SELF_INTERSECTION`
- Meshio: `SURFACE_MESH | VOLUMETRIC_MESH | FILE_IO | QUALITY_METRICS | TOPOLOGY_CHECKS`
- Vedo: `SURFACE_MESH | VOLUMETRIC_MESH | FILE_IO | QUALITY_METRICS | TOPOLOGY_CHECKS`
- PyMeshLab: `SURFACE_MESH | VOLUMETRIC_MESH | FILE_IO | QUALITY_METRICS | TOPOLOGY_CHECKS`
- Trimesh: `SURFACE_MESH | FILE_IO | QUALITY_METRICS | TOPOLOGY_CHECKS` (no volumetric mesh support)

---

### 3. Sliver Detection Only Works for tet4 Elements ✅ RESOLVED

**Location:** `mesh_issue_examiner/operations/volumetric/geometric_validity.py`

**Status:** ✅ **COMPLETED** (2026-03-31)

**Implementation Summary:**
All four linear element types now have dedicated sliver detectors:

| Element Type | Method | Edge Count |
|--------------|--------|------------|
| `tet4` | `_detect_sliver_tets()` | 6 edges |
| `hex8` | `_detect_sliver_hexes()` | 12 edges |
| `prism6` | `_detect_sliver_prisms()` | 9 edges |
| `pyramid5` | `_detect_sliver_pyramids()` | 8 edges |

A `detector_map` dictionary dispatches each element type to its detector, and `_compute_element_offset()` replaces the old tet-specific offset helper. Each detector follows the same pattern: compute edge lengths from vertex coordinates, compute `volume / max_edge³` ratio, and flag cells below threshold.

---

### 4. Winding Consistency Extended to All Element Types ✅ RESOLVED

**Location:** `mesh_issue_examiner/operations/volumetric/orientation_consistency.py`

**Status:** ✅ **COMPLETED** (2026-04-10)

**Implementation Summary:**
The `OrientationConsistencyCheck` operation now supports all 4 linear element types:

| Element Type | Method | Status |
|--------------|--------|--------|
| `tet4` | `_signed_volume_tet4()` | ✅ Supported |
| `hex8` | `_signed_volume_decomposed()` | ✅ Supported |
| `prism6` | `_signed_volume_decomposed()` | ✅ Supported |
| `pyramid5` | `_signed_volume_decomposed()` | ✅ Supported |

**Key Features:**
1. **Face extraction** — Uses `ELEMENT_FACES` from `_face_tables.py` for all element types
2. **Face winding comparison** — `_faces_are_reversed()` handles both triangular and quadrilateral faces using directed-edge method
3. **Signed volume computation** — Vectorized for tet4, decomposed for hex8/prism6/pyramid5
4. **Mixed-element support** — Correctly processes meshes with multiple element types

**Tests Added:**
- `test_hex_only_mesh_not_silently_skipped` — Verifies hex8 meshes are processed
- `test_prism_only_mesh_runs` — Verifies prism6 meshes are processed
- `test_pyramid_only_mesh_runs` — Verifies pyramid5 meshes are processed
- `test_two_hexes_inconsistent_winding_detected` — Verifies hex8 winding detection
- `test_two_prisms_sharing_tri_face_consistent` — Verifies prism6 winding
- `test_two_pyramids_sharing_tri_face_consistent` — Verifies pyramid5 winding
- `test_mixed_tet_prism_winding_inconsistent` — Verifies mixed-element support

**Impact:**
- Hex meshes with inconsistent winding are now detected
- Mixed meshes have complete orientation validation
- All element types are properly checked for consistent face winding

---

## 🟡 Medium Priority Gaps

### 5. Limited Element Type Support ✅ RESOLVED

**Location:** `mesh_issue_examiner/utils/cell_types.py`

**Status:** ✅ **COMPLETED** (2026-04-10)

**What was done:**
All linear and quadratic element types added to `ELEMENT_NODE_COUNT`; complete quadratic types (hex27, wedge18, pyramid14) added 2026-10-27:

| Element Type | Nodes | Status |
|--------------|-------|--------|
| `tet4` | 4 | ✅ Supported |
| `hex8` | 8 | ✅ Supported |
| `prism6` | 6 | ✅ Supported |
| `pyramid5` | 5 | ✅ Supported |
| `tet10` | 10 | ✅ Supported |
| `hex20` | 20 | ✅ Supported |
| `prism15` | 15 | ✅ Supported |
| `pyramid13` | 13 | ✅ Supported |
| `hex27` | 27 | ✅ Supported (VTK TRIQUADRATIC_HEXAHEDRON) |
| `wedge18` | 18 | ✅ Supported (VTK BIQUADRATIC_QUADRATIC_WEDGE) |
| `pyramid14` | 14 | ✅ Registered — connectivity-level only (no VTK cell) |

**Corner-first quadratic face resolution (2026-10-27):** `_face_tables.py` now defines `ELEMENT_CORNER_COUNT` and `LINEAR_BASE_TYPE`, and `get_face_definitions()` resolves any quadratic element to its linear base type so face-based ops (non-manifold faces, winding, duplicate cells, orphaned cells, boundary closure) analyze quadratic cells through their corner nodes. Midside nodes are ignored for face identity, which is correct because two conforming cells share corners iff they share midside nodes.

**Element Type Support in Operations:**

| Operation | Linear (4) | Quadratic (tet10/hex20/prism15/pyramid13) | Complete quadratic (hex27/wedge18) | pyramid14 |
|-----------|------------|------------------------------------------|-------------------------------------|-----------|
| NonManifoldFaceDetection | ✅ | ✅ (corner faces) | ✅ (corner faces) | ✅ (corner faces) |
| OrientationConsistency | ✅ | ✅ (corner faces) | ✅ (corner faces) | ✅ (corner faces) |
| DuplicateCellDetection | ✅ | ✅ | ✅ | ✅ |
| OrphanedCellDetection | ✅ | ✅ | ✅ | ✅ |
| BoundaryClosureCheck | ✅ | ✅ | ✅ | ✅ |
| GeometricValidity (sliver) | ✅ | ⚠️ linear detectors only (quadratic blocks not visited) | ⚠️ linear detectors only | ⚠️ not visited (no detector) |
| VolumeCellQuality | ✅ | ✅ (corner linearization, tagged) | ✅ (corner linearization, tagged) | ✅ (corner linearization via pyramid5) |

**Known limitations:**
1. ~~**VTK quality sentinel**~~ — **resolved 2026-10-28**: `volume_cell_quality.py` detects quadratic element types, rebuilds a corner-only linear grid (`_linearized_volume_cells`) and computes every metric there, attributing values to the original cells with a `linearized` provenance tag in the `QualityMetrics` metadata. Midside curvature is ignored, so linearised quadratic metrics are approximations bounded by the corner-equivalence unit test. The inverted-cell check still filters any residual -1 sentinel.
2. **pyramid14 has no VTK cell** — VTK's tri-quadratic pyramid has 19 nodes, not 14. pyramid14 remains connectivity-level for native conversion (`MeshConversionError`), but 3D quality now reaches it through its pyramid5 corners (linearized). Self-intersection and other strictly native checks still skip gracefully.
3. ~~**hex8 winding convention**~~ — **resolved 2026-10-28**: `ELEMENT_FACES` was recalibrated so every face tuple follows the outward right-hand rule under VTK node ordering (hex8 bottom `(3,2,1,0)`, prism6 bottom `(2,1,0)`, pyramid5 base `(3,2,1,0)`, tet4 normalised). Stacked-cell interfaces now list shared faces in opposing directions and clean grids pass `check_orientation_consistency`.

---

### 6. No Volumetric Mesh File Loaders ⚠️ PARTIALLY RESOLVED

**Location:** `mesh_issue_examiner/utils/mesh_loader.py`, `mesh_issue_examiner/utils/nas_parser.py`

**Status:** ⚠️ **PARTIALLY COMPLETED** (2026-10-27)

**What was done:**
- `load_mesh_from_file()` now accepts `extract_volume_cells=True` parameter
- Uses adapter's `to_canonical()` which handles volumetric mesh extraction
- NASTRAN parser (`nas_parser.py`) extracts CTETRA, CHEXA, and CPENTA volumetric cells with boundary face decomposition (2026-10-27)
- CHEXA/CPENTA continuation cards supported in both free-field and short-field formats; since 2026-10-28 the 20-node CHEXA / 15-node CPENTA forms are retained as hex20/prism15 with midside nodes kept (NASTRAN midside ordering matches VTK quadratic ordering; corner reduction applies only to forms with no registered element type)
- FORMAT_SUPPORT expanded to include .msh, .cgns, .exo, .xdmf, .med, .su2, .ugrid, .inp, and many more
- `save_mesh_to_file()` with volumetric-aware library selection (prefers meshio)
- New `nas_mixed_volume` fixture (CTETRA + 20-node CHEXA + CPENTA) with pinned e2e contract and unit tests (`test_nas_parser.py`)

**Current workflow now works:**
```python
# Automatic volumetric extraction
mesh = load_mesh_from_file("simulation.vtu")  # Returns MeshInput with volume_cells
mesh = load_mesh_from_file("model.nas")       # CTETRA/CHEXA/CPENTA extracted
```

**Remaining gaps:**
- No dedicated Gmsh reader (relies on meshio delegation)
- No dedicated CGNS/Exodus readers (relies on meshio)

(The higher-order NASTRAN bullet closed on 2026-10-28: CHEXA20/CPENTA15 are retained as hex20/prism15.)

**Effort Estimate for remaining:** Medium (2-3 days)

---

### 7. Cell Connectivity Validation ✅ RESOLVED

**Description:**
Basic connectivity validation has been addressed through new topological operations.

**What was done:**
- `DisconnectedComponentDetection` — Identifies separate connected components
- `IsolatedVertexDetection` — Finds vertices not referenced by any face
- `DuplicateFaceDetection` — Identifies faces with identical vertex sets
- `DuplicateVertexDetection` — Identifies vertices with identical or nearly identical coordinates

**Complete Check Coverage:**

| Check | Description | Impact | Status |
|-------|-------------|--------|--------|
| **Duplicate Cells** | Cells with identical vertex lists | Wasted memory, solver confusion | ✅ Implemented (`topology_3d.detect_duplicate_cells`) |
| **Orphaned Cells** | Cells referencing vertex indices ≥ n_vertices | Segmentation fault risk | ✅ Implemented (`topology_3d.detect_orphaned_cells`) |
| **Dangling Faces** | Boundary faces with incorrect adjacency | Open volume, solver failure | ✅ Covered by boundary closure check |
| **Negative Volume** | Cells with inverted orientation | Negative Jacobians | ✅ Covered by orientation check |
| **Zero Volume** | Degenerate/collapsed cells | Solver singularities | ✅ Covered by geometric validity |
| **Face Mismatch** | Shared faces with different vertex count | Topology corruption | ✅ Covered by boundary closure (non-conforming interfaces) |
| **Disconnected Components** | Separate connected regions | May be intentional | ✅ Implemented |
| **Isolated Vertices** | Vertices not referenced by any face | Wasted memory | ✅ Implemented |
| **Duplicate Faces** | Faces with identical vertex sets | Rendering artifacts | ✅ Implemented |
| **Duplicate Vertices** | Vertices with identical coordinates | Topology issues | ✅ Implemented |

**Current State:**
- `GeometricValidityAssessment` checks for degenerate (near-zero) volumes
- `OrientationConsistencyCheck` checks for negative/inverted volumes
- `DisconnectedComponentDetection` identifies disconnected regions
- `IsolatedVertexDetection` finds unreferenced vertices
- `DuplicateFaceDetection` finds duplicate faces
- `DuplicateVertexDetection` finds duplicate vertices (with configurable tolerance; numpy fallback for meshes with no native representation)
- `DuplicateCellDetection` finds cells with identical vertex sets — `topology_3d.detect_duplicate_cells` (added 2026-10-27)
- `OrphanedCellDetection` finds cells referencing out-of-range vertex indices — `topology_3d.detect_orphaned_cells` (added 2026-10-27)
- `BoundaryClosureCheck` verifies watertightness (every boundary edge shared by exactly two boundary faces) — `topology_3d.check_boundary_closure` (added 2026-10-27)

**Remaining Gaps:** None. All three remaining items (duplicate cells, orphaned cells, boundary closure) were implemented in Phase 2, with defective fixtures (`duplicate_cell`, `leaky_volume`) and pinned e2e contracts covering each.

---

### 8. Incomplete Quality Metrics Coverage ✅ RESOLVED

**Description:**
Volumetric quality metrics were limited to those directly supported by VTK's `cell_quality` filter, and quadratic cells received no metrics at all (VTK's -1 sentinel).

**Status:** ✅ **COMPLETED** (2026-10-28)

**What was done:**
- **6 extended metrics** implemented in `volume_cell_quality.py` (2026-04) with industry-standard thresholds and per-element applicability (`METRIC_ELEMENT_SUPPORT`): `warpage`, `equiangle_skew`, `stretch`, `taper`, `radius_ratio`, `collapse_ratio`; `min/max_dihedral_angle` available via the surface edge ops
- **Quadratic quality via corner linearization** (2026-10-28): when the mesh contains quadratic element types, `_linearized_volume_cells()` rebuilds a corner-only linear grid (VTK corner-first ordering; `LINEAR_BASE_TYPE`/`ELEMENT_CORNER_COUNT`), all metrics are computed there and attributed to the original cells with a `linearized` provenance tag in the `QualityMetrics` metadata and issue metadata.  The -1 sentinel filtering is preserved everywhere
- **pyramid14 quality unblocked**: no VTK cell exists, but the linearized pyramid5 corner grid carries its metrics
- Every computed metric is surfaced through `mesh_get_quality_metrics` (distributions, thresholds, violation counts)

**Validation:** `test_extended_quality_metrics.py` (metric applicability, thresholds, higher-is-worse direction), `TestLinearizedQuality` in `test_higher_order_elements.py` (no sentinels, provenance tag, exact corner-equivalence on tet10, pyramid14 metrics, mixed-mesh cell ordering), e2e `test_tet10_quality_metrics_linearized` and `test_retained_hex20_gets_linearized_quality` (MCP metrics on the tet10 and NASTRAN hex20 fixtures)

**Residual limitation:** linearized quadratic metrics approximate midside curvature by straight edges — labeled in metric metadata and bounded by the corner-equivalence tolerance test.  Sliver detection (`geometric_validity.py`) still visits linear element blocks only; quadratic slivers are covered indirectly by the linearized `scaled_jacobian`/`volume` thresholds.

---

### 9. No Boundary Layer Quality Checks ✅ RESOLVED

**Description:**
No specialized checks for anisotropic boundary layer meshes common in CFD.

**Status:** ✅ **COMPLETED** (2026-10-28)

**What was done:** `quality_3d.check_boundary_layer` (`boundary_layer_quality.py`) implements wall-seed identification (anisotropy-gated: stacked-cell height ≪ in-plane extent via `_inplane_edge_length`), layer tracing, growth-ratio verification (threshold 1.2, ERROR/CRITICAL by violation rate), layer-count and wall-coverage checks with transition-cell classification.

**Calibration (2026-10-28):** the seed gate requires BL-typical anisotropy, so plain hex grids yield no seeds — `closed_volume`/`duplicate_cell`/`leaky_volume` keep the op silent; genuine prism stacks are detected.

**Validation:** unit tests in `test_boundary_layer_quality.py` (seed calibration on 2×2×2 hex grids and growing stacks, 36 tests); e2e fixtures `boundary_layer` (clean 5-layer r=1.15 stack + tet cap: op silent, inspection passes) and `boundary_layer_bad_growth` (3× jump: growth ERROR fires, `max_growth_ratio` ≈ 2.14), pinned in the generator CONTRACTS and `TestBoundaryLayerVolume`.

---

### 10. Missing Multi-Material Interface Checks ✅ RESOLVED

**Description:**
No support for meshes with multiple materials or regions.

**Status:** ✅ **COMPLETED** (2026-10-28)

**What was done:**
- `MeshInput.cell_materials` populated via `material_array_name` on `load_mesh_from_file` (VTU cell data) and PSOLID-based extraction for NASTRAN (`material_reader.py`); per-block ordering alignment covered by dedicated tests
- Operations `quality_3d.detect_material_interfaces`, `quality_3d.check_interface_quality`, `quality_3d.per_material_statistics` (unit-tested in `test_multi_material.py`)
- `material_array_name` exposed on **both MCP layers** (business handler + FastMCP `mesh_load`) with a materials summary (count, ids, cells assigned) in the load response — previously unreachable through MCP (2026-10-28)

**Validation:** e2e `TestTwoMaterialVolume` on the `two_material_volume.vtu` fixture: materials load aligned ([1,2,1,2]), interface/statistics ops fire at INFO, connectivity ops stay silent; loading without the parameter keeps the ops silent (backwards compatible).

---

## 🟢 Lower Priority Gaps

### 11. Hardcoded tet4 Face Definitions ✅ RESOLVED

**Location:** `mesh_issue_examiner/operations/volumetric/nonmanifold_faces.py:27-56`

**Status:** ✅ **COMPLETED** (2026-03-31)

**Implementation Summary:**
The `ELEMENT_FACES` dict in `nonmanifold_faces.py` now provides generic face definitions for all 4 linear element types. This can be reused by other operations (e.g., orientation consistency).

```python
ELEMENT_FACES: Dict[str, List[Tuple[int, ...]]] = {
    'tet4': [(0,1,2), (0,2,3), (0,3,1), (1,3,2)],
    'hex8': [(0,1,2,3), (4,5,6,7), (0,1,5,4), (1,2,6,5), (2,3,7,6), (3,0,4,7)],
    'prism6': [(0,1,2), (3,4,5), (0,1,4,3), (1,2,5,4), (2,0,3,5)],
    'pyramid5': [(0,1,2,3), (0,1,4), (1,2,4), (2,3,4), (3,0,4)],
}
```

---

### 12. No Parallel Processing for Large Meshes ❌ STILL OPEN

**Description:**
All volumetric operations are single-threaded. No support for parallel processing of large meshes (>10M cells).

**Current State:**
```python
# Sequential loop over all cells
for quality_type in self.quality_types:
    values = adapter.compute_cell_quality(grid, quality_type)
    # ... process sequentially
```

**Opportunities for Parallelization:**
1. Independent quality metric computation per cell
2. Parallel face adjacency building
3. Parallel sliver detection across element batches

**Implementation Options:**
1. `multiprocessing` for CPU-bound quality calculations
2. `numba` JIT compilation for hot loops
3. `numpy` vectorization improvements

**Effort Estimate:** High (5-7 days)

---

### 13. Limited Reporting Granularity ✅ RESOLVED

**Description:**
Issues were reported at cell level only, with no vertex-level or edge-level diagnostics and no metric surfacing.

**Status:** ✅ **COMPLETED** (2026-10-28)

**What was done:**
- **Vertex level**: `topology.detect_vertex_valence` (irregular-valence detection with triangle/quad thresholds); on volume meshes, cell-referenced interior nodes absent from the extracted surface are excluded from the irregular mask (2026-10-28)
- **Edge level**: `topology.detect_edge_dihedral_angle` and `topology.detect_edge_length_ratio`; boundary-edge detection with watertightness metrics
- **Metric surfacing**: every computed metric is returned through `mesh_get_quality_metrics` with statistics, thresholds, violation counts and provenance metadata (e.g. `linearized` for corner-linearized quadratic metrics, 2026-10-28)
- **Volume-aware 2D ops** (2026-10-28): isolated-vertex, disconnected-component and valence checks account for volume-cell references, so interior nodes of closed volumes are no longer misreported

**Validation:** fixtures across the e2e suite pin vertex/edge behavior (icospheres, varying_quality, open/closed surfaces, closed_volume zero-issue sentinel); metrics surfaced for every volumetric fixture.

---

## Test Coverage Gaps

Current tests in `test_volumetric_operations.py` cover:
- ✅ Basic tetrahedral mesh quality
- ✅ Hexahedral cube quality
- ✅ Degenerate cell detection
- ✅ Sliver detection for all 4 linear element types (tet4, hex8, prism6, pyramid5)
- ✅ Orientation consistency for all 4 linear element types
- ✅ Non-manifold face detection for all 4 linear element types
- ✅ Volume cell properties
- ✅ Mixed-element mesh support

Current tests in `test_three_new_operations.py` cover:
- ✅ Disconnected component detection
- ✅ Isolated vertex detection
- ✅ Duplicate face detection

**Missing Test Coverage:**

| Test | Priority | Description |
|------|----------|-------------|
| ~~Higher-order elements~~ | Medium | ✅ Covered by `test_higher_order_elements.py` (registration, corner face tables, ops on quadratic meshes, pyramid14 degradation, linearized quality) plus tet10/hex20 e2e fixtures |
| Large mesh performance | Medium | >1M cell mesh handling |
| ~~All quality metrics~~ | Medium | ✅ Covered by `test_extended_quality_metrics.py` (per-metric applicability, thresholds, direction) and the linearized-quality tests |
| ~~File loading~~ | High | ✅ `.vtu` volume fixtures + `.nas` e2e (`TestNastranMixedVolume`, hex20 retention) + `.msh`/`.cgns`/`.exo` via meshio delegation |
| Duplicate vertex detection | Medium | Tests for DuplicateVertexDetection operation |

---

## Appendix: Supported vs. Needed Features

### Element Types

| Type | Supported | Needed | Notes |
|------|-----------|--------|-------|
| tet4 | ✅ | ✅ | Full support in all operations |
| hex8 | ✅ | ✅ | Full support in all operations |
| prism6 | ✅ | ✅ | Full support in all operations |
| pyramid5 | ✅ | ✅ | Full support in all operations |
| tet10 | ✅ | ✅ | Topology via corner-first faces; quality via corner linearization (tagged) |
| hex20 | ✅ | ✅ | Topology via corner-first faces; quality via corner linearization (tagged); retained from CHEXA20 |
| prism15 | ✅ | ✅ | Topology via corner-first faces; quality via corner linearization (tagged); retained from CPENTA15 |
| pyramid13 | ✅ | ✅ | Topology via corner-first faces; quality via corner linearization (tagged) |
| hex27 | ✅ | ✅ | Registered (2026-10-27); topology via corners; quality via corner linearization |
| wedge18 | ✅ | ✅ | Registered (2026-10-27); topology via corners; quality via corner linearization |
| pyramid14 | ✅ | ⚠️ | Registered (2026-10-27) — no VTK cell; topology + linearized quality via pyramid5 corners |

### Quality Metrics

| Metric | tet4 | hex8 | prism6 | pyramid5 | Quadratic (via corner linearization) |
|--------|------|------|--------|----------|--------------------------------------|
| Volume | ✅ | ✅ | ✅ | ✅ | ✅ (tagged `linearized`) |
| Aspect Ratio | ✅ | ✅ | ✅ | ✅ | ✅ (tagged `linearized`) |
| Scaled Jacobian | ✅ | ✅ | ✅ | ✅ | ✅ (tagged `linearized`) |
| Condition | ✅ | ✅ | ✅ | ✅ | ✅ (tagged `linearized`) |
| Skewness | ✅ | ✅ | ✅ | ✅ | ✅ (tagged `linearized`) |
| Sliver Detection | ✅ | ✅ | ✅ | ✅ | ⚠️ linear blocks only (indirect via linearized scaled_jacobian/volume) |
| Winding Check | ✅ | ✅ | ✅ | ✅ | ✅ (corner faces, outward convention since 2026-10-28) |
| Warpage | N/A | ✅ | ✅ | ✅ | ✅ (tagged `linearized`) |
| Equiangle Skew | ✅ | ✅ | ✅ | ✅ | ✅ (tagged `linearized`) |
| Stretch | ✅ | ✅ | ✅ | ✅ | ✅ (tagged `linearized`) |
| Taper | N/A | ✅ | N/A | N/A | ✅ hex20/hex27 (tagged `linearized`) |
| Radius Ratio | ✅ | N/A | N/A | N/A | ✅ tet10 (tagged `linearized`) |
| Collapse Ratio | ✅ | ✅ | ✅ | ✅ | ✅ (tagged `linearized`) |

### Topological Operations

| Operation | Surface | Volumetric | Status |
|-----------|---------|------------|--------|
| NonManifoldEdgeDetection | ✅ | N/A | ✅ Implemented |
| NonManifoldFaceDetection | N/A | ✅ | ✅ Implemented |
| SelfIntersectionDetection | ✅ | ⚠️ | ✅ Implemented |
| DuplicateVertexDetection | ✅ | ✅ | ✅ Implemented |
| DuplicateFaceDetection | ✅ | ✅ | ✅ Implemented |
| IsolatedVertexDetection | ✅ | ✅ | ✅ Implemented |
| DisconnectedComponentDetection | ✅ | ✅ | ✅ Implemented |
| OrientationConsistencyCheck | N/A | ✅ | ✅ Implemented |
| DuplicateCellDetection | N/A | ✅ | ✅ Implemented (2026-10-27) |
| OrphanedCellDetection | N/A | ✅ | ✅ Implemented (2026-10-27) |
| BoundaryClosureCheck | N/A | ✅ | ✅ Implemented (2026-10-27) |

### File Formats

| Format | Read | Write | Volume Cells | Notes |
|--------|------|-------|--------------|-------|
| .vtu | ✅ (via adapter) | ✅ (meshio) | ✅ | Full support via PyVista/Meshio |
| .vtk | ✅ (via adapter) | ✅ (meshio) | ✅ | Full support via PyVista/Meshio |
| .nas/.bdf | ✅ (custom parser) | ❌ | ✅ (CTETRA/CHEXA/CPENTA) | NASTRAN parser extracts all three volume cards; 20/15-node forms retained as hex20/prism15 (2026-10-28) |
| .msh | ✅ (meshio) | ✅ (meshio) | ✅ | Gmsh format via meshio |
| .cgns | ✅ (meshio) | ❌ | ✅ | CGNS format via meshio |
| .exo | ✅ (meshio) | ❌ | ✅ | Exodus format via meshio |
| .xdmf | ✅ (meshio) | ✅ (meshio) | ✅ | XDMF format via meshio |
| .inp | ✅ (meshio) | ❌ | ✅ | Abaqus format via meshio |
| .stl | ✅ (surface only) | ✅ | ❌ | Surface mesh format |
| .obj | ✅ (surface only) | ✅ | ❌ | Surface mesh format |
| .ply | ✅ (surface only) | ✅ | ❌ | Surface mesh format |

---

*Report generated by code analysis of mesh_issue_examiner framework*
*Last updated: 2026-10-28 — progress audit against current codebase (phases 1-6: multi-material e2e, boundary-layer calibration, winding recalibration, quadratic quality linearization, NASTRAN higher-order retention)*
