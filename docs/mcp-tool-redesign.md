# MCP Tool Interface Redesign

> Derived from gap analysis against `mcp-server-best-practices.md` sections 2–3.
> Date: 2026-04-08.

## Problem Statement

The current `MeshIssueExaminerServer` exposes 5 tools through a flat design that conflates execution with data retrieval, returns unbounded output, and carries dead parameters. Specific issues:

- `mesh_inspect` accepts `categories` but ignores it (dead code).
- `mesh_inspect` uses the magic string `["all"]` instead of default-by-omission.
- `mesh_inspect` returns the entire `InspectionResult` — potentially megabytes of JSON.
- `mesh_get_result` has no filtering or pagination.
- `list_operations` is reference data exposed as a tool instead of a resource.
- No MCP resources are defined (section 2.4.2).
- Tool descriptions are too terse for LLM consumption (section 3.2.1).

## Design Principles

1. **Separate execution from retrieval.** `mesh_inspect` runs operations and returns a summary; dedicated retrieval tools provide drill-down.
2. **Bound all output.** Paginate issues, filter metrics, return summaries by default.
3. **Use resources for reference data.** Operation catalog and mesh metadata are URI-addressable, not tools.
4. **No magic strings.** Default behavior is expressed by omitting parameters, not by passing sentinel values.
5. **Mutual exclusion where appropriate.** `categories` and `operations` on `mesh_inspect` cannot both be provided.
6. **LLM-friendly descriptions.** Every tool documents trigger conditions, parameters, return structure, error cases, and usage examples (section 3.2.3).

## Tool Catalog (5 tools)

### 1. `mesh_load`

Load a mesh file into the server cache.

```python
@mcp.tool()
async def mesh_load(
    file_path: str,
    mesh_id: str | None = None,
    library: str | None = None,
) -> str:
    """Load a mesh file for inspection.

    Reads a mesh from disk (STL, OBJ, PLY, VTK, NAS, and 30+ other
    formats) and stores it in the server cache.  Returns a short
    summary needed to decide what to inspect next.

    Args:
        file_path: Absolute path to the mesh file on the server
            filesystem.
        mesh_id: Custom identifier.  Defaults to filename stem.
        library: Preferred loader — "trimesh" | "pyvista" | "pymeshlab".
            Auto-detected when omitted.

    Returns:
        JSON with mesh_id, vertex/face counts, bounding box, format,
        and whether volume cells were found.

    Errors:
        FileNotFoundError — path does not exist.
        UnsupportedFormatError — file extension not in supported list.
        MeshLoadError — file exists but could not be parsed.

    Example::

        result = mesh_load(file_path="bracket.stl")
        # {"mesh_id": "bracket", "n_vertices": 12000, "n_faces": 24000, ...}
    """
```

**Change from current:** Description enriched per section 3.2.1. Signature unchanged.

---

### 2. `mesh_inspect`

Execute inspection operations on a loaded mesh. Returns summary only.

```python
@mcp.tool()
async def mesh_inspect(
    mesh_id: str,
    categories: list[str] | None = None,
    operations: list[str] | None = None,
) -> str:
    """Execute inspection operations on a previously loaded mesh.

    Provide ONE of ``categories`` or ``operations`` to run a subset;
    omit both to run every registered operation.

    Args:
        mesh_id: Identifier returned by mesh_load.
        categories: Run all operations in these categories.
            Each value must be one of: "topological", "geometric",
            "quality", "orientation".
        operations: Run these specific operation IDs.
            Discover IDs via the mesh://operations resource.

    Returns:
        JSON summary with severity counts, pass/fail flag, category
        breakdown, and processing time.  DOES NOT include individual
        issues or quality metrics — use mesh_get_issues and
        mesh_get_quality_metrics to drill down.

    Errors:
        ValueError — mesh_id not found in cache.
        ValueError — unknown operation ID.
        ValueError — both categories and operations provided
            (mutual exclusion).

    Example::

        # Run everything
        mesh_inspect(mesh_id="bracket")

        # Run just topological checks
        mesh_inspect(mesh_id="bracket", categories=["topological"])

        # Run one specific operation
        mesh_inspect(mesh_id="bracket",
                     operations=["topology.detect_nonmanifold_edges"])
    """
```

**Changes from current:**

| Issue | Before | After |
|---|---|---|
| Magic string `["all"]` | `operations=["all"]` | Omit parameter to run all |
| Dead `categories` param | Accepted, silently ignored | Implemented via `registry.create_category_workflow()` |
| Ambiguous both-provided | Both accepted, only `operations` used | Error: mutual exclusion enforced |
| Unbounded output | Returns full `InspectionResult` | Returns **summary only** |
| Operation discovery | Must call `list_operations` tool | Via `mesh://operations` resource |

**Handler logic:**

```python
async def _handle_inspect(self, params: dict) -> dict:
    mesh_id = params["mesh_id"]
    categories = params.get("categories")
    operations = params.get("operations")

    if categories is not None and operations is not None:
        raise ValueError(
            "Provide 'categories' or 'operations', not both."
        )

    mesh = self._mesh_cache.get(mesh_id)
    if mesh is None:
        raise ValueError(f"Mesh not found: {mesh_id}")

    if operations is not None:
        # Validate IDs against registry
        available = {op.operation_id for op in registry.list_all()}
        invalid = [o for o in operations if o not in available]
        if invalid:
            raise ValueError(f"Unknown operation(s): {invalid}")
        workflow = registry.create_workflow(operations, config=self.config)
    elif categories is not None:
        # Build workflow from categories
        cat_enums = [IssueCategory(c) for c in categories]
        op_ids = []
        for cat in cat_enums:
            op_ids.extend(
                op.operation_id
                for op in registry.list_by_category(cat)
            )
        workflow = registry.create_workflow(op_ids, config=self.config)
    else:
        # Default: run all
        workflow = registry.create_full_workflow(config=self.config)

    result = workflow(mesh)
    self._result_cache[mesh_id] = result

    # Return summary only
    return {
        "mesh_id": mesh_id,
        "passed": result.passed,
        "processing_time": result.processing_time,
        "summary": result.summary,
    }
```

---

### 3. `mesh_get_issues`

Retrieve detected issues with filtering and pagination. Replaces `mesh_get_result`.

```python
@mcp.tool()
def mesh_get_issues(
    mesh_id: str,
    severity: str | None = None,
    category: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> str:
    """Retrieve detected issues from a completed inspection.

    Call after mesh_inspect.  Supports filtering and pagination to
    control response size.

    Args:
        mesh_id: Identifier from mesh_load.
        severity: Filter — "critical" | "error" | "warning" | "info".
            Returns all severities when omitted.
        category: Filter — "topological" | "geometric" | "quality"
            | "orientation".  Returns all categories when omitted.
        limit: Max issues to return (1–200, default 50).
        offset: Skip this many issues for pagination.

    Returns:
        JSON with total_count (matching filter), returned_count,
        offset, and the issue list.  Each issue includes severity,
        category, message, and up to 10 location references.

    Errors:
        ValueError — mesh_id not found or no inspection run yet.

    Example::

        # First page of critical issues
        mesh_get_issues(mesh_id="bracket", severity="critical")
        # {"total_count": 3, "returned_count": 3, "offset": 0,
        #  "issues": [...]}

        # Paginated geometric issues
        mesh_get_issues(mesh_id="bracket", category="geometric",
                        limit=20, offset=0)
    """
```

**Why this replaces `mesh_get_result`:** The old tool returns the entire `InspectionResult` (issues + metrics + summary) with no filtering or pagination. A mesh with 10k issues would dump megabytes into the MCP response.

**Handler logic:**

```python
def _handle_get_issues(self, params: dict) -> dict:
    mesh_id = params["mesh_id"]
    result = self._result_cache.get(mesh_id)
    if result is None:
        raise ValueError(f"No inspection result for mesh: {mesh_id}")

    issues = result.issues

    # Filter
    if "severity" in params:
        sev = IssueSeverity(params["severity"])
        issues = [i for i in issues if i.severity == sev]
    if "category" in params:
        cat = IssueCategory(params["category"])
        issues = [i for i in issues if i.category == cat]

    total = len(issues)
    limit = max(1, min(params.get("limit", 50), 200))
    offset = max(0, params.get("offset", 0))
    page = issues[offset : offset + limit]

    return {
        "total_count": total,
        "returned_count": len(page),
        "offset": offset,
        "issues": [i.to_dict() for i in page],
    }
```

---

### 4. `mesh_get_quality_metrics`

Retrieve quality metric summaries. Extracted from the old `mesh_get_result`.

```python
@mcp.tool()
def mesh_get_quality_metrics(
    mesh_id: str,
    metric_name: str | None = None,
) -> str:
    """Retrieve quality metrics from a completed inspection.

    Call after mesh_inspect.  Returns per-metric statistics (min,
    max, mean, median, std, p5, p95) and violation counts.

    Args:
        mesh_id: Identifier from mesh_load.
        metric_name: Specific metric to retrieve (e.g. "aspect_ratio",
            "scaled_jacobian").  Returns all metrics when omitted.

    Returns:
        JSON array of metric objects.  Each contains metric_name,
        statistics, threshold, violation_count, and total_count.

    Errors:
        ValueError — mesh_id not found or no inspection run yet.

    Example::

        mesh_get_quality_metrics(mesh_id="bracket")
        # [{"metric_name": "aspect_ratio", "violation_count": 12,
        #   "statistics": {"mean": 1.3, ...}}, ...]

        mesh_get_quality_metrics(mesh_id="bracket",
                                 metric_name="aspect_ratio")
    """
```

**Why a separate tool:** Quality metrics are a fundamentally different data shape from issues (statistical summaries vs. discrete defect locations). Mixing them forces the LLM to parse a monolith.

---

### 5. `mesh_clear_cache`

```python
@mcp.tool()
def mesh_clear_cache(mesh_id: str | None = None) -> str:
    """Free cached mesh data and inspection results.

    Call to release memory or force a fresh reload.

    Args:
        mesh_id: Specific mesh to clear.  If omitted, clears all
            cached meshes and results.

    Returns:
        JSON with "status": "cleared" and list of cleared IDs.

    Example::

        mesh_clear_cache()                   # clear all
        mesh_clear_cache(mesh_id="bracket")  # clear one
    """
```

**Change from current:** Description enriched. Signature unchanged.

---

## Resource Catalog (3 resources)

### `mesh://operations`

```python
@mcp.resource("mesh://operations")
def operations_catalog() -> str:
    """Catalog of all available inspection operations.

    Returns JSON array of operation descriptors including operation_id,
    display_name, description, category, supported_libraries,
    recommended_library, and complexity.
    """
```

**Why a resource instead of a tool:** The operations catalog is reference data — it changes only when the server code changes, not per-request. It needs no parameters. Tools are for *actions*; resources are for *data access* (section 2.4.2). This also removes one tool from the LLM's decision space.

### `mesh://{mesh_id}/info`

```python
@mcp.resource("mesh://{mesh_id}/info")
def mesh_info(mesh_id: str) -> str:
    """Mesh metadata: vertex/face counts, bounding box, format,
    volume cell types.

    Available after mesh_load.
    """
```

### `mesh://{mesh_id}/summary`

```python
@mcp.resource("mesh://{mesh_id}/summary")
def inspection_summary(mesh_id: str) -> str:
    """Inspection summary: severity counts, pass/fail flag,
    category breakdown, processing time.

    Available after mesh_inspect.
    """
```

---

## What Was Removed

| Removed | Reason |
|---|---|
| `list_operations` tool | Replaced by `mesh://operations` resource (read-only reference data, not an action) |
| `mesh_get_result` tool | Replaced by `mesh_get_issues` (paginated, filterable) + `mesh_get_quality_metrics` (separate concern) |
| `operations=["all"]` magic | Replaced by omitting the parameter (default-by-omission) |
| Dead `categories` param | **Fixed** — now implemented via `registry.create_category_workflow()`, with mutual exclusion vs `operations` |

---

## LLM Workflow

```
Step 1 — Load
    mesh_load(file_path="bracket.stl")
    → {"mesh_id": "bracket", "n_vertices": 12000, "n_faces": 24000, ...}

Step 2 — Inspect (returns summary only)
    mesh_inspect(mesh_id="bracket")
    → {"passed": false, "summary": {"critical": 3, "error": 12, ...},
       "processing_time": 1.2}

Step 3 — Drill into issues (paginated)
    mesh_get_issues(mesh_id="bracket", severity="critical")
    → {"total_count": 3, "returned_count": 3, "issues": [...]}

Step 4 — Drill into metrics (optional)
    mesh_get_quality_metrics(mesh_id="bracket")
    → [{"metric_name": "aspect_ratio", "violation_count": 12, ...}]
```

Every response is bounded. The LLM drills down only when needed.

---

## Concrete Examiner Capabilities

The following section documents every inspection operation the server
can run — what each one detects, what mesh types it applies to, what
libraries it requires, and what it returns.  This is the ground truth
derived from the source code and should be used as the reference when
writing tool descriptions or the `mesh://operations` resource payload.

### Surface Operations (2D meshes — triangular / polygonal faces)

#### `topology.detect_nonmanifold_edges`

- **Source:** `operations/topological/nonmanifold_edges.py`
- **Category:** topological
- **What it detects:** Edges shared by more than two faces.  In a valid
  2-manifold mesh every edge must be incident to exactly two faces.
  Boundary edges (1 face) are normal for open surfaces and are not
  flagged.
- **Algorithm:** Builds an edge-to-face adjacency map from the face
  array.  Vectorised fast path for triangular meshes (3 edges/face).
  Falls back to generic loop for polygonal faces.
- **Applicable to:** Any mesh with faces (surface or volumetric surface
  boundary).
- **Requires volume cells:** No.
- **Supported libraries:** trimesh, pyvista, pymeshlab (uses only
  `mesh.faces` and `mesh.vertices` — no library-specific calls).
- **Recommended library:** pyvista.
- **Complexity:** O(n).
- **Return shape:** Single `MeshIssue` with all non-manifold edges as
  locations.  Each location carries the edge vertices, face count,
  and incident face indices.  Severity scales with violation rate
  (fraction of faces involved).
- **Special notes:** Location `element_index` is -1 (edges have no
  global index).  Coordinates are taken from the first vertex of the
  edge.

#### `topology.detect_duplicate_vertices`

- **Source:** `operations/topological/duplicate_vertices.py`
- **Category:** topological
- **What it detects:** Vertices whose coordinates are identical or
  within a configurable tolerance (default `1e-6`).  Reports the
  total count and duplicate rate.
- **Algorithm:** Delegates to `adapter.get_duplicate_vertices()`.
  Each library implements its own spatial index (KD-tree, hash, etc.).
- **Applicable to:** Any mesh with vertices.
- **Requires volume cells:** No.
- **Supported libraries:** trimesh, pyvista, pymeshlab, meshio, vedo.
- **Recommended library:** pyvista.
- **Complexity:** O(n log n).
- **Constructor params:** `tolerance: float = 1e-6`.
- **Return shape:** Single `MeshIssue`.  Severity: capped at ERROR
  (never CRITICAL) unless >50% of vertices are duplicates.
- **Caveat:** trimesh auto-merges vertices on load, so duplicate
  detection always returns 0 when meshes are loaded via trimesh.
  Use `library='pyvista'` in `mesh_load` for reliable detection.

#### `topology.detect_self_intersections`

- **Source:** `operations/topological/self_intersection.py`
- **Category:** topological
- **What it detects:** Triangles that intersect non-adjacent triangles
  in the same mesh.  Always reported as CRITICAL.
- **Algorithm:** Möller–Trumbore triangle–triangle intersection test
  with KD-tree spatial acceleration.  Uses
  `adapter.detect_self_intersections(method='manual')`.
- **Applicable to:** Surface meshes (triangular faces).
- **Requires volume cells:** No.
- **Supported libraries:** pyvista only.
- **Recommended library:** pyvista.
- **Complexity:** O(n log n).
- **Return shape:** One `MeshIssue` per intersecting face (not pair).
  Each issue has CRITICAL severity, the face index, and the paired
  face index in metrics.  Already-reported faces are deduplicated.
- **Special notes:** Silently returns empty list on adapter failure
  (catches all exceptions).  `min_area_threshold = 1e-10` filters
  sliver intersections.

### Volumetric Operations (3D meshes — tet4 / hex8 / prism6 / pyramid5)

All volumetric operations return an empty list when `mesh.has_volume_cells`
is False.  They are safe to run on surface-only meshes — they simply
produce no output.

#### `topology_3d.detect_nonmanifold_faces`

- **Source:** `operations/volumetric/nonmanifold_faces.py`
- **Category:** topological
- **What it detects:** Faces shared by more than two volumetric cells.
  In a valid manifold volume mesh every internal face must be shared
  by exactly two cells.
- **Algorithm:** Builds a face-to-cell adjacency map using
  `ELEMENT_FACES` lookup tables (tet4 → 4 triangular faces, hex8 →
  6 quad faces, prism6 → 5 faces, pyramid5 → 5 faces).  Canonical
  face key is sorted vertex indices.  Reports faces with >2 cells.
- **Applicable to:** Volumetric meshes only.
- **Requires volume cells:** Yes (returns `[]` otherwise).
- **Supported libraries:** pyvista (uses `mesh.volume_cells` and
  `ELEMENT_FACES` tables — no VTK calls).
- **Recommended library:** pyvista.
- **Complexity:** O(n).
- **Return shape:** Single `MeshIssue`.  Locations are capped at
  `_MAX_LOCATIONS = 100`.  Each location carries face centroid
  coordinates, face vertices, and the sharing cell count/indices.
  Metadata includes `affected_cell_types` and `max_sharing_count`.
- **Element types supported:** tet4, hex8, prism6, pyramid5
  (via `_face_tables.ELEMENT_FACES`).

#### `quality_3d.assess_geometric_validity`

- **Source:** `operations/volumetric/geometric_validity.py`
- **Category:** geometric
- **What it detects:** Two classes of volumetric defects:
  1. **Degenerate cells** (CRITICAL): `|volume| < degenerate_tolerance`
     (default `1e-12`).  Collapsed or inverted cells.
  2. **Sliver cells** (WARNING): `volume / max_edge_length³ < sliver_threshold`
     (default `1e-6`).  Flat, needle-like, or poorly shaped cells
     that are not degenerate.
- **Algorithm:** Computes cell volumes via
  `adapter.compute_cell_quality(grid, 'volume')`.  For slivers,
  computes edge lengths per element type (tet4: 6 edges, hex8: 12
  edges, prism6: 9 edges, pyramid5: 8 edges) and checks the
  volume-to-max-edge-cubed ratio.
- **Applicable to:** Volumetric meshes only.
- **Requires volume cells:** Yes (returns `[]` otherwise).
- **Supported libraries:** pyvista only.
- **Recommended library:** pyvista.
- **Complexity:** O(n).
- **Constructor params:** `degenerate_tolerance: float = 1e-12`,
  `sliver_threshold: float = 1e-6`.
- **Return shape:** Up to two `MeshIssue` objects (one for degenerate,
  one for sliver).  Locations capped at 100 per issue.  Metrics
  include per-element-type sliver counts.

#### `quality.compute_cell_quality`

- **Source:** `operations/quality/cell_quality.py`
- **Category:** quality
- **What it detects:** Surface mesh elements (triangles/polygons) whose
  shape quality metrics fall outside acceptable thresholds.  Reports
  violation counts per metric.
- **Default metrics and thresholds:**

  | Metric | Threshold | Direction | Meaning |
  |---|---|---|---|
  | `aspect_ratio` | 10.0 | higher is worse | longest/shortest edge ratio |
  | `min_angle` | 15.0° | lower is worse | smallest interior angle |
  | `max_angle` | 120.0° | higher is worse | largest interior angle |

- **All valid metric names:** `aspect_ratio`, `min_angle`,
  `max_angle`, `scaled_jacobian`, `condition`, `shape`, `area`,
  `distortion`, `radius_ratio`.
- **Algorithm:** Converts to native mesh, calls
  `adapter.compute_cell_quality(native_mesh, quality_type)` for
  each metric.  Filters out VTK sentinel value `-1`.  Violation
  direction depends on metric (HIGHER_IS_WORSE set: max_angle,
  condition, aspect_ratio).
- **Applicable to:** Any mesh with faces.  Works on surface triangles
  and polygonal faces.
- **Requires volume cells:** No.
- **Supported libraries:** pyvista (primary), trimesh (fallback).
- **Recommended library:** pyvista.
- **Complexity:** O(n) per metric.
- **Constructor params:** `quality_types: list[str]` (default:
  `['aspect_ratio', 'min_angle', 'max_angle']`).
- **Return shape:** One `MeshIssue` per metric that has violations.
  Operation ID is suffixed: `quality.compute_cell_quality.aspect_ratio`.
  For `min_angle`, severity is capped at WARNING and percentile
  breakdown (p1, p5, p10, p20, p50) is included in metadata.
- **Special notes:** `skewness` is NOT valid for triangular meshes
  (VTK returns -1).  Use `scaled_jacobian` or `shape` instead.

#### `quality_3d.compute_volume_cell_quality`

- **Source:** `operations/volumetric/volume_cell_quality.py`
- **Category:** quality
- **What it detects:** Volumetric cells whose quality metrics fall
  below (or above, for condition number) acceptable thresholds.
- **Default metrics and thresholds:**

  | Metric | Threshold | Direction | Meaning |
  |---|---|---|---|
  | `scaled_jacobian` | 0.1 | lower is worse | ≤0 = invalid, <0.1 = poor |
  | `volume` | 1e-12 | lower is worse | near-zero = degenerate |
  | `aspect_ratio` | 0.1 | lower is worse | VTK convention (1 = ideal) |
  | `condition` | 100.0 | higher is worse | condition number |
  | `shape` | 0.3 | lower is worse | 1 = ideal |

- **All valid metric names:** `scaled_jacobian`, `volume`,
  `aspect_ratio`, `condition`, `shape`, `skewness`, `skew`,
  `orthogonality`.
- **Algorithm:** Converts to PyVista `UnstructuredGrid`, calls
  `adapter.compute_cell_quality(grid, quality_type)` for each metric.
- **Applicable to:** Volumetric meshes only.
- **Requires volume cells:** Yes (returns `[]` otherwise).
- **Supported libraries:** pyvista only.
- **Recommended library:** pyvista.
- **Complexity:** O(n) per metric.
- **Constructor params:** `quality_types: list[str]` (default:
  `['scaled_jacobian', 'volume', 'aspect_ratio', 'condition',
  'shape']`).
- **Return shape:** One `MeshIssue` per metric that has violations.
  Operation ID suffixed: `quality_3d.compute_volume_cell_quality.scaled_jacobian`.
  Locations capped at 100.  Metadata includes p5/p95 percentiles.

#### `quality_3d.check_orientation_consistency`

- **Source:** `operations/volumetric/orientation_consistency.py`
- **Category:** orientation
- **What it detects:** Two classes of orientation defects in
  volumetric meshes:
  1. **Inverted elements** (CRITICAL): Cells with negative signed
     volume (`volume < -tolerance`, tolerance default `1e-6` from
     `OperationConfig`).
  2. **Inconsistent winding** (ERROR): For each internal face shared
     by two cells, the face vertices should appear in opposite
     winding order.  Uses directed-edge method: every edge (u→v)
     in one cell must appear as (v→u) in the adjacent cell.
- **Algorithm:**
  - Inverted: Gets cell volumes from `grid.cell_quality('volume')`
    or falls back to manual signed-volume computation
    (`_compute_signed_volumes`).  Supports tet4 (vectorised
    determinant), hex8 (5-tet decomposition), prism6 (3-tet
    decomposition), pyramid5 (2-tet decomposition).
  - Winding: Builds face-to-cell map using `ELEMENT_FACES` tables.
    For each shared face, checks that both cells list the face
    with opposite directed edges.
- **Applicable to:** Volumetric meshes only.
- **Requires volume cells:** Yes (returns `[]` otherwise).
- **Supported libraries:** pyvista only.
- **Recommended library:** pyvista.
- **Complexity:** O(n log n) (face-to-cell map construction).
- **Return shape:** Up to two `MeshIssue` objects.  Locations capped
  at `_MAX_LOCATIONS = 100`.  Inverted cells always CRITICAL.
  Inconsistent winding cells always ERROR.
- **Element types supported:** tet4, hex8, prism6, pyramid5, and
  mixed-element meshes.

### Operations Summary Table

| Operation ID | Category | Mesh Type | Libraries | Complexity | Requires Vol. Cells |
|---|---|---|---|---|---|
| `topology.detect_nonmanifold_edges` | topological | surface | trimesh, pyvista, pymeshlab | O(n) | No |
| `topology.detect_duplicate_vertices` | topological | surface | trimesh, pyvista, pymeshlab, meshio, vedo | O(n log n) | No |
| `topology.detect_self_intersections` | topological | surface | pyvista | O(n log n) | No |
| `topology_3d.detect_nonmanifold_faces` | topological | volumetric | pyvista | O(n) | Yes |
| `quality_3d.assess_geometric_validity` | geometric | volumetric | pyvista | O(n) | Yes |
| `quality.compute_cell_quality` | quality | surface | pyvista, trimesh | O(n)/metric | No |
| `quality_3d.compute_volume_cell_quality` | quality | volumetric | pyvista | O(n)/metric | Yes |
| `quality_3d.check_orientation_consistency` | orientation | volumetric | pyvista | O(n log n) | Yes |

### Quality Metric Reference

#### Surface Metrics (via `quality.compute_cell_quality`)

| Metric | Threshold | Direction | Range | Notes |
|---|---|---|---|---|
| `aspect_ratio` | 10.0 | higher is worse | [1, ∞) | 1 = ideal equilateral |
| `min_angle` | 15.0° | lower is worse | (0°, 60°] | 60° = ideal for tri; severity capped at WARNING |
| `max_angle` | 120.0° | higher is worse | [60°, 180°) | 60° = ideal for tri |
| `scaled_jacobian` | 0.5 | lower is worse | [-1, 1] | 1 = ideal |
| `condition` | 100.0 | higher is worse | [1, ∞) | 1 = ideal |
| `shape` | 0.25 | lower is worse | (0, 1] | 1 = ideal |
| `area` | 1e-10 | lower is worse | (0, ∞) | near-zero = degenerate |
| `distortion` | 0.5 | lower is worse | (0, 1] | 1 = ideal |
| `radius_ratio` | 2.0 | higher is worse | [1, ∞) | 1 = ideal |
| `skewness` | N/A | — | — | NOT valid for triangles (VTK returns -1) |

#### Volumetric Metrics (via `quality_3d.compute_volume_cell_quality`)

| Metric | Threshold | Direction | Range | Notes |
|---|---|---|---|---|
| `scaled_jacobian` | 0.1 | lower is worse | [-1, 1] | ≤0 = invalid element |
| `volume` | 1e-12 | lower is worse | (-∞, ∞) | negative = inverted |
| `aspect_ratio` | 0.1 | lower is worse | (0, 1] | VTK convention (1 = ideal) |
| `condition` | 100.0 | higher is worse | [1, ∞) | lower = better |
| `shape` | 0.3 | lower is worse | (0, 1] | 1 = ideal |
| `skewness` | 0.8 | higher is worse | [0, 1] | equiangular skewness |
| `orthogonality` | 0.15 | lower is worse | (0, 1] | face normal orthogonality |

#### Geometric Validity Metrics (via `quality_3d.assess_geometric_validity`)

| Check | Threshold | Severity | Notes |
|---|---|---|---|
| Degenerate cells | `|volume| < 1e-12` | CRITICAL | collapsed/inverted |
| Sliver cells | `volume / max_edge³ < 1e-6` | WARNING | flat/needle-like |

### Supported Element Types

| Element | Node Count | VTK CellType | Operations Using It |
|---|---|---|---|
| tet4 | 4 | TETRA | nonmanifold_faces, orientation_consistency, geometric_validity, volume_cell_quality |
| hex8 | 8 | HEXAHEDRON | nonmanifold_faces, orientation_consistency, geometric_validity, volume_cell_quality |
| prism6 | 6 | WEDGE | nonmanifold_faces, orientation_consistency, geometric_validity, volume_cell_quality |
| pyramid5 | 5 | PYRAMID | nonmanifold_faces, orientation_consistency, geometric_validity, volume_cell_quality |
| tet10 | 10 | QUADRATIC_TETRA | cell_types.py recognizes, no operations yet |
| hex20 | 20 | QUADRATIC_HEXAHEDRON | cell_types.py recognizes, no operations yet |
| prism15 | 15 | QUADRATIC_WEDGE | cell_types.py recognizes, no operations yet |
| pyramid13 | 13 | QUADRATIC_PYRAMID | cell_types.py recognizes, no operations yet |

### Adapter Capabilities Matrix

| Capability | trimesh | pyvista | pymeshlab | meshio | vedo |
|---|---|---|---|---|---|
| Surface mesh | yes | yes | yes | yes | yes |
| Volumetric mesh | no | yes | no | partial | partial |
| File I/O | yes | yes | yes | yes | yes |
| Quality metrics | yes | yes | limited | no | limited |
| Topology checks | yes | yes | limited | no | no |
| Self-intersection | no | yes | no | no | no |
| Remeshing | yes | limited | yes | no | no |

### Severity Determination Rules

All operations use the same violation-rate thresholds from
`utils/metrics.determine_severity()`:

| Violation Rate | Severity |
|---|---|
| > 50% | CRITICAL |
| > 20% | ERROR |
| > 5% | WARNING |
| ≤ 5% | INFO |

Exceptions:
- `topology.detect_self_intersections`: Always CRITICAL.
- `topology.detect_duplicate_vertices`: CRITICAL capped to ERROR
  unless > 50%.
- `quality.compute_cell_quality` for `min_angle`: ERROR/CRITICAL
  capped to WARNING.
- `quality_3d.check_orientation_consistency`: Inverted cells always
  CRITICAL; inconsistent winding always ERROR.

### Supported File Formats

| Format | Extensions | Primary Library |
|---|---|---|
| STL | `.stl` | trimesh / pyvista |
| Wavefront OBJ | `.obj` | trimesh / pyvista |
| PLY | `.ply` | trimesh / pyvista |
| OFF | `.off` | trimesh |
| glTF / GLB | `.gltf`, `.glb` | trimesh / pyvista |
| VTK XML | `.vtp`, `.vtu` | pyvista |
| VTK Legacy | `.vtk` | pyvista |
| 3MF | `.3mf` | trimesh |
| NASTRAN | `.nas`, `.bdf`, `.fem` | custom parser / trimesh |
| Gmsh | `.msh` | meshio |
| Abaqus | `.inp` | meshio |
| ANSYS | `.ans` | meshio |
| Exodus | `.e`, `.exo` | meshio |
| XDMF | `.xdmf`, `.xmf` | meshio |
| MED/Salome | `.med` | meshio |
| CGNS | `.cgns` | meshio |
| FLAC3D | `.f3grid` | meshio |
| TetGen | `.node`, `.ele` | meshio |
| Medit | `.mesh`, `.meshb` | meshio |
| SU2 | `.su2` | meshio |
| UGRID | `.ugrid` | meshio |
| Tecplot | `.dat` | meshio |
| PERMAS | `.post`, `.dato` | meshio |
