# Adapter layer — deep inspection
Files read: all 10 files in `mesh_issue_examiner/adapters/` (`__init__.py`, `base_adapter.py`, `capabilities.py`, `exceptions.py`, `factory.py`, `trimesh_adapter.py`, `pyvista_adapter.py`, `pymeshlab_adapter.py`, `meshio_adapter.py`, `vedo_adapter.py`), plus `mesh_issue_examiner/utils/cell_types.py`, `mesh_issue_examiner/operations/volumetric/_face_tables.py`, `tests/test_adapters.py`, `tests/test_adapter_normalization.py`.

## Findings

### Base contract (`base_adapter.py`)
- 17 abstract members every adapter must implement: properties `capabilities`, `library_name`; `_check_import`; methods `load_mesh`, `create_mesh`, `get_vertices`, `get_faces`, `is_watertight`, `is_manifold`, `get_boundary_edges`, `get_nonmanifold_edges`, `get_duplicate_vertices(tolerance)`, `compute_face_areas`, `compute_face_normals`, `detect_self_intersections`, `compute_cell_quality` (base_adapter.py:59-335). Contract is "implement everything", not capability-gated, despite section comments saying "required if <capability>" (base_adapter.py:128-130, 186-188, 266-268).
- Concrete base behavior: `is_available` probes via `_check_import()` catching only `ImportError` (base_adapter.py:91-109); `get_volume_cells` default returns `{}` (base_adapter.py:341-359); default `to_canonical` builds MeshInput with `source_format=self.library_name` (base_adapter.py:380-386); default `from_canonical` is surface-only via `create_mesh` (base_adapter.py:388-410).
- `compute_all_quality_metrics` defaults to `['aspect_ratio', 'skewness', 'min_angle']` and swallows per-metric failures into `np.array([])` with only a debug log (base_adapter.py:435-445).
- Error reporting convention: library imports are lazy inside methods; every failure is re-raised as an adapter exception with `from e` chaining.

### Capabilities (`capabilities.py`)
- `AdapterCapability` is a `Flag` enum with 7 atomic flags (SURFACE_MESH, VOLUMETRIC_MESH, FILE_IO, QUALITY_METRICS, TOPOLOGY_CHECKS, SELF_INTERSECTION, REMESHING) plus composites SURFACE_FULL / VOLUMETRIC_FULL / ALL (capabilities.py:34-69).
- `require_capability` / `require_any_capability` decorators raise `UnsupportedOperationError` when the adapter lacks the flag (capabilities.py:97-101, 127-131). They are exported and tested, but no adapter method in the package uses them — the only production raise sites for `UnsupportedOperationError` are these decorators.
- `AdapterRegistry` is class-level global state: `_adapters`, `_capabilities`, `_instances` dicts (capabilities.py:157-159). All 5 adapters self-register at package import (adapters/__init__.py:114-118), registration order pyvista→trimesh→pymeshlab→meshio→vedo.
- `register()` snapshots capabilities by instantiating the class; on any exception stores `AdapterCapability(0)` (capabilities.py:176-182). Snapshots succeed even when the library is absent because adapter `__init__`s never import their library (e.g. pymeshlab_adapter.py:55-56, meshio_adapter.py:119-122) — `get_capability_matrix()` therefore works in any environment.
- `find_adapter_for_task` selection: preferred name first if it's a capability superset, else all supersets filtered by availability, then sorted by "specificity" = popcount of extra capability bits beyond the requirements (capabilities.py:299-351). Consequence: for `VOLUMETRIC_MESH|QUALITY_METRICS`, meshio (3 extra bits: SURFACE, FILE_IO, TOPOLOGY) beats pyvista (4 extra: +SELF_INTERSECTION) and vedo (4 extra: +REMESHING), so **meshio is the preferred volumetric-quality backend**, not pyvista. `sorted()` is stable, so ties fall back to registration order (pyvista first).
- No adapter has all capabilities; `ALL` is declared "rare in practice" (capabilities.py:68-69) and requesting all 7 flags raises `NoAdapterAvailableError` (test_adapter_normalization.py:240-252).

### Exceptions (`exceptions.py`)
- Hierarchy: `AdapterError(MeshExaminerError)` → `UnsupportedOperationError`, `MeshConversionError`, `QualityMetricError`, `FileIOError`, `NoAdapterAvailableError`, `AdapterNotAvailableError` (exceptions.py:11-88). All catchable via the package root `MeshExaminerError`.
- Raise sites: `FileIOError` in every adapter's `load_mesh` and meshio's `save_mesh` (e.g. trimesh_adapter.py:89, meshio_adapter.py:597); `MeshConversionError` in pyvista/trimesh/pymeshlab/vedo/meshio `from_canonical` paths (pyvista_adapter.py:225, 255; trimesh_adapter.py:403; pymeshlab_adapter.py:465; vedo_adapter.py:182; meshio_adapter.py:203); `QualityMetricError` only from trimesh/pymeshlab unsupported metrics and pyvista/meshio/vedo computation failures (trimesh_adapter.py:257, pymeshlab_adapter.py:372, pyvista_adapter.py:889, meshio_adapter.py:552, vedo_adapter.py:496).
- `NoAdapterAvailableError` vs `AdapterNotAvailableError`: the first means *no registered adapter meets capability requirements* (two distinct messages: no candidate at all vs candidates exist but libraries missing, capabilities.py:319-340); the second means *the named adapter's library is not installed* (capabilities.py:216-220).
- Quirk: `FileIOError` shadows the Python builtin (`os.FileIOError` == OSError alias); adapters import it locally inside methods to disambiguate.

### Per-adapter capability matrix
| Adapter | Caps (atomic) | Volume cells | from_canonical on volume mesh | Surface quality metrics | Save/write |
|---|---|---|---|---|---|
| pyvista | SURFACE, VOLUMETRIC, FILE_IO, QUALITY, TOPOLOGY, SELF_INTERSECTION (pyvista_adapter.py:109-116) | Yes — all VTK-mapped types | UnstructuredGrid; pyramid14 corner-reduced | 14 VTK metrics via `cell_quality` | none in adapter |
| trimesh | SURFACE, FILE_IO, QUALITY, TOPOLOGY (trimesh_adapter.py:54-59) | No | Surface boundary via PyVista (trimesh_adapter.py:346-353) | area, aspect_ratio, min_angle, max_angle (trimesh_adapter.py:225-260) | none |
| pymeshlab | SURFACE, FILE_IO, QUALITY, TOPOLOGY, SELF_INTERSECTION, REMESHING (pymeshlab_adapter.py:62-69) | No | Surface boundary via PyVista (pymeshlab_adapter.py:402-409) | area, aspect_ratio only (pymeshlab_adapter.py:336-375) | none |
| meshio | SURFACE, VOLUMETRIC, FILE_IO, QUALITY, TOPOLOGY (meshio_adapter.py:128-134) | Yes — all 11 canonical types | meshio.Mesh, quadratic with corner fallback | area, aspect_ratio surface; volumetric delegated to pyvista (meshio_adapter.py:499-504, 494-496) | `save_mesh` (meshio_adapter.py:579) |
| vedo | SURFACE, VOLUMETRIC, FILE_IO, QUALITY, TOPOLOGY, REMESHING (vedo_adapter.py:67-74) | Yes — VTK-mapped types | vedo.UnstructuredGrid; unknown types skipped w/ warning (vedo_adapter.py:159-171) | area, aspect_ratio surface; volumetric via pyvista (vedo_adapter.py:453-462) | `write` passthrough (vedo_adapter.py:719-721) |

- **Material extraction: absent.** No file in `adapters/` references materials or per-cell data arrays; volume-cell dicts carry connectivity only. Element-type support universe = the 11 names in `ELEMENT_NODE_COUNT` (cell_types.py:14-29); VTK mapping covers 10 — `pyramid14` has no VTK cell (VTK's triquadratic pyramid has 19 nodes) and is connectivity-analysis-only (cell_types.py:66-71).
- Only meshio and vedo adapters expose write support; pyvista/trimesh/pymeshlab adapters have no save/write method at all (verified across the full files).

### to_canonical specifics
- **pyvista**: PolyData → faces via `get_faces`; UnstructuredGrid → `extract_surface(algorithm='dataset_surface')` then faces; volume cells via `get_volume_cells` keyed by VTK-type→name reverse map, per-cell `get_cell(i).point_ids` (pyvista_adapter.py:282-342). `source_format` hardcoded `'vtk'` (pyvista_adapter.py:340). Mixed-size extracted surfaces (ragged `dtype=object` arrays) are fan-triangulated from vertex 0 into (M,3) — regression-tested with a wedge+tet grid yielding 10 triangles (pyvista_adapter.py:21-37, 329-330; test_adapters.py:181-204). Uniform quad surfaces stay (M,4) untouched (test_adapters.py:206-223).
- **pyvista get_faces parsing**: walks the packed `[n, v0..vk, ...]` array and *stops silently* (warning log, partial result) on n<0, n==0, n>1000, or truncation (pyvista_adapter.py:371-399). Uniform face size → 2D array; mixed → 1D `dtype=object` array of index arrays (pyvista_adapter.py:404-414). Bug #4 regression suite pins this (test_adapters.py:126-167).
- **meshio**: merges repeated CellBlocks of the same type by `vstack` in file order (meshio_adapter.py:264-267, 298-307); surface blocks triangle/triangle6 (corners `[:, :3]`), quad/quad8/quad9 split into two triangles `[0,1,2]`+`[0,2,3]` — quadratic surface nodes dropped (meshio_adapter.py:244-261). Type map: meshio `wedge`→prism6, `wedge15`→prism15, `wedge18`→wedge18, `pyramid13`/`pyramid14` pass through (meshio_adapter.py:22-35).
- **vedo**: reads `.vertices`/`.faces`; volume cells parsed by walking vedo's flat `[n, v0..]` cells array against `cell_types` (vedo_adapter.py:239-266).
- **trimesh/pymeshlab**: use the base `to_canonical` (no override) — vertices + faces only; since their `get_volume_cells` is the base `{}` they never produce volume cells from native loads.
- **Normalization is minimal**: no adapter welds/dedups vertices, reorders faces, or fixes winding in `to_canonical`. The only reshaping is triangulation (pyvista fan; meshio quad split). Winding/face order passes through unchanged.

### from_canonical specifics
- **pyvista**: builds `pv.UnstructuredGrid` from `volume_cells` insertion order; quadratic types with no VTK cell fall back to their linear base element truncated to corner nodes (VTK orders corners first) and record provenance in `grid.user_dict['corner_reduced_types']` (best-effort, never fails) (pyvista_adapter.py:209-250); unknown type → `MeshConversionError` (pyvista_adapter.py:224-228). Surface with 0 faces → points-only PolyData (pyvista_adapter.py:260-262).
- **meshio**: emits volume blocks in `volume_cells` insertion order (keeps per-cell arrays aligned), surface `triangle` block last (meshio_adapter.py:184-226). Unknown canonical type → `MeshConversionError` (meshio_adapter.py:201-205). Quadratic types the installed meshio can't construct (e.g. wedge15/pyramid13 in meshio 5.3.x) are probed once per type by building a 1-cell mesh, cached in module-global `_MESHIO_TYPE_SUPPORT`, and degraded to the linear corner block per `MESHIO_VOLUME_FALLBACK` (meshio_adapter.py:44-77, 207-217). Round-trip fidelity for tet10/hex20/hex27/wedge18/pyramid14 through .vtu is test-pinned (test_adapter_normalization.py:426-582).
- **vedo**: volumetric → `vedo.UnstructuredGrid([vertices, cells, celltypes])`; unknown element types are *skipped with a warning*, and `MeshConversionError` only if nothing valid remains (vedo_adapter.py:159-184). Surface → `vedo.Mesh([verts, faces])`; no faces → `vedo.Points` (vedo_adapter.py:131-136).
- **trimesh / pymeshlab** (surface-only): if `has_volume_cells`, delegate to `mesh_input.get_native('pyvista')` + `extract_surface`, then hand-parse the VTK face array; failure raises `MeshConversionError` advising pyvista install (trimesh_adapter.py:355-407; pymeshlab_adapter.py:411-468). Both duplicate the VTK `[n, v0..]` unpack loop inline.

### Notable per-adapter quirks
- **pymeshlab is stateful and mutating**: stores `_current_meshset` on load/create (pymeshlab_adapter.py:55-56, 130, 156); `is_manifold` runs selection filters (unselects after) (pymeshlab_adapter.py:174-205); `get_duplicate_vertices` *mutates the mesh* via `merge_close_vertices`, converting absolute tolerance to % of bbox diagonal, and returns an index *range* `[cleaned_count, original_count)` — approximate, not real indices (pymeshlab_adapter.py:268-284). Same range-approximation trick in pyvista (`mesh.clean`) (pyvista_adapter.py:544-551). Documented as not thread-safe (pymeshlab_adapter.py:40-42).
- **trimesh `detect_self_intersections` is a stub**: logs a warning and returns `[]` (trimesh_adapter.py:201-216). meshio/vedo "self-intersection" actually return trimesh *non-convex face-adjacency* pairs, not true intersections (meshio_adapter.py:452-465; vedo_adapter.py:413-437). Only pyvista implements real detection: manual Möller–Trumbore + KD-tree (`utils/triangle_intersection`), default `tolerance=1e-10`, with pymeshfix as legacy alternate returning `(face_idx, -1)` pairs (pyvista_adapter.py:655-794).
- **meshio `is_watertight` assumes True** for meshes with no triangle faces (pure volumetric) (meshio_adapter.py:316-319); vedo does the same for point clouds (vedo_adapter.py:273-275).
- **meshio/vedo cache pyvista grids keyed by `id(mesh)`**, capped at 10 with FIFO eviction (meshio_adapter.py:122, 528-541; vedo_adapter.py:61, 474-486) — a distinct cache from `MeshInput.get_native`.
- **Silent-empty quality fallbacks**: meshio and vedo `compute_cell_quality` return `np.array([])` for unsupported surface metrics without raising (meshio_adapter.py:499-504; vedo_adapter.py:457-462), unlike trimesh/pymeshlab which raise `QualityMetricError`.
- pymeshlab warns (UserWarning) when loading volumetric extensions `.vtu/.vtk/.cgns/.exo/.e/.msh` that only the surface boundary is processed (pymeshlab_adapter.py:89-91, 115-122).
- Extra non-contract methods: pymeshlab `smooth`/`decimate`/`repair` (pymeshlab_adapter.py:474-536); vedo `visualize`/`visualize_quality` (blocking `vedo.show`), `slice_mesh`, `clip_mesh`, `decimate`, `smooth`, `compute_curvature`, `extract_largest_region`, `clean`, `write`, `get_bounds`/`get_volume`/`get_surface_area` (vedo_adapter.py:524-745).
- Stale test comment: test_adapter_normalization.py:383-384 claims pyvista `to_canonical` "doesn't extract volume_cells" — contradicted by the override at pyvista_adapter.py:304-342.

### factory.py
- Contains **no per-format / extension mapping at all**; format preference lives in the loader, not here. `SUPPORTED_LIBRARIES = ['trimesh','pyvista','pymeshlab','meshio','vedo']` is "kept for reference and validation" and is not consulted by any factory function (factory.py:40-41) — note its order differs from registration order.
- All 8 functions are thin delegates to `AdapterRegistry` (factory.py:44-218): `get_adapter`, `find_adapter_for_task`, `list_supported_libraries` (sorted), `get_available_libraries` (sorted, import-probes each), `get_capability_matrix`, `clear_adapter_cache`, `get_adapter_capabilities`, `has_capability` (returns False instead of raising on unknown name, factory.py:214-218).
- Instance semantics: registry caches one adapter instance per name (singleton per process); `clear_adapter_cache()` / `AdapterRegistry.clear_cache()` resets (capabilities.py:204-224, 362-369).
- Error behavior: unknown name → `ValueError("Unsupported library: ...")` (capabilities.py:206-209); library missing → `AdapterNotAvailableError` (capabilities.py:216-220); no capability match → `NoAdapterAvailableError` with two message variants (capabilities.py:319-340).

## Numbers & limits
- Registry size: 5 adapters, registered at import (adapters/__init__.py:114-118); tests pin `len == 5` (test_adapter_normalization.py:592-600).
- Element node counts: tet4=4, hex8=8, prism6=6, pyramid5=5, tet10=10, hex20=20, prism15=15, pyramid13=13, hex27=27, wedge18=18, pyramid14=14 (cell_types.py:14-29); VTK map covers 10 of 11 (cell_types.py:52-72).
- pyvista `_MAX_VERTS_PER_FACE = 1000` parse guard (pyvista_adapter.py:345).
- pyvista self-intersection default tolerance `1e-10` (pyvista_adapter.py:659).
- trimesh duplicate-vertex heuristic threshold: `unique_ratio < 0.99` (trimesh_adapter.py:134).
- meshio/vedo pyvista-grid cache cap: 10 entries, FIFO eviction (meshio_adapter.py:538-541; vedo_adapter.py:484-486).
- `compute_all_quality_metrics` default metrics: `['aspect_ratio', 'skewness', 'min_angle']` (base_adapter.py:436).
- meshio `get_supported_formats`: 29 extensions listed (meshio_adapter.py:603-610).
- pymeshlab volumetric-extension warning set: `{'.vtu', '.vtk', '.cgns', '.exo', '.e', '.msh'}` (pymeshlab_adapter.py:89); decimation default `target_reduction=0.5`, smoothing default 3 iterations (pymeshlab_adapter.py:477, 501); vedo smoothing default 15 iterations / relax 0.1, decimate fraction 0.5 (vedo_adapter.py:622, 648-649).
- pyvista quality-name remaps: `'skewness'→'skew'`, `'Jacobian'→'jacobian'`; 14 mapped metric names (pyvista_adapter.py:835-851).
- meshio `MESHIO_VOLUME_FALLBACK`: 7 quadratic types → (linear type, corner count): tetra10→(tetra,4), hexahedron20/27→(hexahedron,8), wedge15/18→(wedge,6), pyramid13/14→(pyramid,5) (meshio_adapter.py:44-52).

## Candidate additions for the HTML note
- Capability-driven backend choice: `find_adapter_for_task(VOLUMETRIC|QUALITY)` picks **meshio** over pyvista because the specificity heuristic (fewest extra capability bits) penalizes pyvista's SELF_INTERSECTION and vedo's REMESHING flags (capabilities.py:344-351).
- `pyramid14` has no VTK cell: pyvista `from_canonical` corner-reduces it to pyramid5 and stamps `grid.user_dict['corner_reduced_types']`; vedo *skips* unknown types with a warning; meshio probes constructibility once per type and degrades to the linear corner block — three different degradation policies for the same problem.
- Only meshio and vedo adapters implement save (`save_mesh` / `write`); pyvista, trimesh, pymeshlab adapters are load-only.
- No adapter extracts materials or per-cell data; conversion is geometry+connectivity only.
- `to_canonical` normalization is limited to triangulation (pyvista fan-triangulation of ragged mixed surfaces; meshio quad→2-triangle split `[0,1,2]+[0,2,3]`); no vertex welding, dedup, face reordering, or winding repair anywhere in the adapter layer.
- pymeshlab adapter mutates input meshes (`merge_close_vertices` in duplicate detection; selection filters in manifold checks) and is documented not thread-safe; duplicate-vertex "indices" from pyvista/pymeshlab are synthetic ranges, not real indices.
- Self-intersection support is only real on pyvista (Möller–Trumbore + KD-tree, tol 1e-10); trimesh returns `[]`, meshio/vedo return trimesh non-convex adjacency pairs.
- Registry mechanics: capability snapshots work without libraries installed (lazy imports); `has_capability` swallows unknown names → False; `NoAdapterAvailableError` distinguishes "no candidate" from "libraries missing".

## Corrections
none
