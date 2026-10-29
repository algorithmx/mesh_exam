# Core models + utils/parsers — deep inspection

Files read: `mesh_issue_examiner/core/{models,exceptions,__init__}.py`; all of `mesh_issue_examiner/utils/{__init__,nas_parser,gmsh_parser,material_reader,mesh_loader,metrics,topology,triangle_intersection,cell_types,dimensionality}.py`; consumers `operations/base.py`, `visualization/renderer.py`, `mcp/server.py` (456-654), `adapters/exceptions.py`, `adapters/pyvista_adapter.py` (613-811).

## Findings

### Exception taxonomy (core/exceptions.py + parallel trees)
- 6 classes, all direct children of `MeshExaminerError` (core/exceptions.py:6-38). All are empty `pass` bodies — pure markers.
| Exception | Def | Actual raise sites |
|---|---|---|
| `MeshLoadError` | exceptions.py:11 | mesh_loader.py:24,26,29,31,36 (validation), :188 (parse_nas), :259 (no lib installed), :270 (lib missing), :300 (empty scene), :326/:332/:337 (import/SystemExit/other) |
| `UnsupportedFormatError` | exceptions.py:16 | mesh_loader.py:178 (unknown suffix), :265 (library lacks format) |
| `OperationError` | exceptions.py:21 | **none — shadowed**: operations/base.py:214 defines a *second* `OperationError(MeshExaminerError)`; the one actually raised is at base.py:127 inside `BaseOperation.run()` |
| `InvalidMeshError` | exceptions.py:26 | none anywhere (dead) |
| `LibraryNotAvailableError` | exceptions.py:31 | none; loader deliberately raises `MeshLoadError` instead (mesh_loader.py:269-273) |
| `VisualizationError` | exceptions.py:36 | renderer.py:295 (no GL context), :394 (screenshot produced no image) |
- Separate adapter subtree `AdapterError(MeshExaminerError)` (adapters/exceptions.py:11): `UnsupportedOperationError` :21, `MeshConversionError` :36, `QualityMetricError` :48, `FileIOError` :60, `NoAdapterAvailableError` :72, `AdapterNotAvailableError` :81. Both trees are caught by the single `MeshExaminerError` handler.
- MCP boundary maps: `ValueError`/`FileNotFoundError` → `business_error` (server.py:537-547); `MeshExaminerError` → `business_error` (server.py:548-558); anything else → `internal_error` with internals hidden (server.py:559-570).

### nas_parser.py — cards, mechanics, limits
- Supported cards: `GRID` (:91), `CTRIA3` (:94), `CQUAD4` split to 2 tris `(g1,g2,g3)+(g1,g3,g4)` (:99,:336), `CTETRA` (:104), `CHEXA`/`CPENTA` with continuation cards (:111-134). `PSOLID` is NOT parsed here (materials live in material_reader).
- Bulk-data flow: skip until `BEGIN BULK` (:66-69); stop at `ENDATA`/`ENDDATA` (:73); `$` comments skipped (:77); free-field if line contains `,` else short-field 8-char columns (:81-84, :283-297); continuations detected by first field starting `+` and merged (:119-128). Long-field (16-char) format unsupported.
- Quadratic retention: `len(nodes) >= 20 → hex20`, `>= 15 → prism15`, else `hex8`/`prism6` (:422-427); NASTRAN midside ordering asserted equal to VTK, no reordering (:14-18). **CTETRA always truncated to 4 nodes** — 10-node CTETRA midside nodes silently dropped (:364-372); no `tet10` output.
- Boundary extraction: `Counter` over `tuple(sorted(face))` keys; faces seen exactly once survive, first oriented occurrence kept (:448-467).
- Node IDs remapped: union of used nodes, sorted → contiguous 0-based (:148-155). GRID `CP` field skipped → coordinates in non-basic systems misinterpreted (:309-313).
- Raises `ValueError` if no GRID (:136-137) or no element cards (:139-140); loader wraps these into `MeshLoadError` (mesh_loader.py:187-190).
- `write_nas` (:194-276): free-field; GRID numbered 1..N using `repr(float)` for lossless round-trip (:221-226); EID counter starts at **100** (:228); writes volume cards OR CTRIA3 surface cards — never both, to avoid double-counting boundary faces on re-parse (:199-206); continuation chunks of 8 node fields with `+C{eid}k` markers (:254-266); unknown type → `ValueError` listing `_NAS_WRITABLE` = tet4/hex8/hex20/prism6/prism15 (:185-191, :233-237). Unreferenced vertices survive write but vanish on re-parse (:197-199).

### gmsh_parser.py — sections, types, fallback triggers
- Parses `$MeshFormat`, `$PhysicalNames`, `$Nodes`, `$Elements`; any other `$Section` skipped by the header loop (:129-206).
- ValueError (→ meshio fallback) raised for: version not starting `2.` (:139-143), `file-type != 0` i.e. binary (:143-144), truncated sections (:135,148,153,170,175,188,193), malformed node/element lines (:182-184, :202-204), unknown node id referenced by an element (:277-279), no `$Nodes` (:208-209), no usable elements (:210-211). Only `ValueError` triggers fallback (mesh_loader.py:225); `UnicodeDecodeError` (binary files read in text mode) is a `ValueError` subclass so still falls back.
- 16 element codes supported (GMSH_ELEMENT_TYPES :45-62): 2 tri3, 3 quad4, 4 tet4, 5 hex8, 6 prism6, 7 pyramid5, 9 tri6, 10 quad9, 11 tet10, 12 hex27, 13 wedge18, 14 pyramid14, 16 quad8, 17 hex20, 18 prism15, 19 pyramid13. All other codes (1/8 lines, 15 points) silently skipped (:269-271).
- Gmsh→VTK node permutations applied for tet10/hex20/hex27/prism15/pyramid13 only (:67-80), documented as mirroring meshio's `_gmsh_to_meshio_order`. Quad surfaces split to 2 triangles (:290-292). Pyramid corner decomposition local (:84-86, :325-328); tet/hex/prism tables imported from nas_parser (:35-40).
- Materials: first element tag = physical-group id (:298-299); any element with `num_tags < 1` sets `all_tagged=False` via private `_UntaggedElements` signal → `cell_materials=None` (:200-201, :240-247); names from `$PhysicalNames` (:249); array ordered per `volume_cells` insertion order (:242-247).

### material_reader.py — routes and defects
- PSOLID route (`extract_materials_from_nas` :16-89): scans raw lines; `PSOLID PID MID` → `Material_{mid}` names (:50-62); `CTETRA/CHEXA/CPENTA/CPYRAM` EID→PID (:66-78). `cell_materials` built in **sorted-EID order** with unmapped PID → 0 (:84-87) — this order matches `volume_cells` enumeration only when deck element IDs are already file-ordered and all types interleave consistently; alignment is not enforced. Note `CPYRAM` is read here but parse_nas drops CPYRAM entirely.
- Operator-precedence defect at :89: `return cell_materials, material_names if len(...) > 0 else (None, None)` parses as `cell_materials, (cond ? material_names : (None, None))` — the empty case returns `([], (None, None))`, not `(None, None)`.
- Gmsh route (`extract_materials_from_gmsh` :92-134): meshio `cell_data['gmsh:physical']` concatenation (:120-126) + own `$PhysicalNames` re-reader (:137-162). **No caller anywhere** (git grep: definition only) — dead code; the live Gmsh material route is gmsh_parser itself.
- The `from .nas_parser import parse_nas` import at :31-35 is vestigial (imported, never used).
- mesh_loader calls only `extract_materials_from_nas`, and only when `material_array_name` is truthy — the *value* of the name is ignored for NASTRAN (mesh_loader.py:203-213).

### utils/topology.py — algorithms & consumers
- Exposes 6 functions (utils/__init__.py:23-30). Production consumers: `compute_vertex_valence` ← operations/topological/vertex_valence.py:20; `build_edge_face_map` ← edge_dihedral.py:20; `extract_unique_edges`+`compute_edge_lengths` ← edge_length_ratio.py:20. `build_vertex_faces_map` and `compute_face_areas` have **no production consumer** (export-only).
- `compute_vertex_valence`: `np.add.at` per column for 2D arrays; Python loop fallback for ragged 1-D object arrays (:45-56); docstring codifies valence norms (tri interior ideal 6, problematic <4/>8) (:22-27).
- `build_edge_face_map`: edges keyed as `tuple(sorted((v1,v2)))` with `(i+1)%n` ring closure; values = incident face indices (:103-114). No loops/components algorithms here — boundary/disconnected ops implement their own.

### utils/metrics.py — native vs delegated
- **No geometry is computed here.** Metric values are always delegated to `adapter.compute_cell_quality` (:98-103); per-metric failures silently skipped (:119-121). Native content = thresholds, statistics, severity mapping.
- `DEFAULT_THRESHOLDS` 9 surface entries (:26-36); `VOLUMETRIC_THRESHOLDS` 12 entries (:39-52); `get_threshold` unknown-metric fallback **0.1** (:65-70). Direction convention documented in-module (:16-19) and hard-coded a second time as `HIGHER_IS_WORSE={'max_angle','condition','aspect_ratio'}` in models.py:152-156 (with `-1` VTK-undefined filtering :149).
- `compute_percentile_stats` (:159-199): worst-tail percentiles, defaults `[1,5,10,20,50]`; for higher-is-worse metrics returns `(100-p)`-th percentile; consumer quality/cell_quality.py:21.
- `determine_severity` (:202-219): >0.5 CRITICAL, >0.2 ERROR, >0.05 WARNING, else INFO; duplicated verbatim as `QualityOperation.get_recommended_severity` (operations/base.py:194-211); 8+ operation files import it.
- `compute_all_quality_metrics` default metric set `['aspect_ratio','min_angle','max_angle','scaled_jacobian']` (:96).

### utils/cell_types.py — table & consumers
- `ELEMENT_NODE_COUNT` 11 types (:14-29): linear tet4/hex8/prism6/pyramid5; serendipity quadratic tet10/hex20/prism15/pyramid13; complete quadratic hex27/wedge18/pyramid14. Reverse map by node count (:32-34).
- `get_cell_type_map` lazy pyvista CellType map (:41-78); **pyramid14 deliberately excluded** — no VTK equivalent (TRIQUADRATIC_PYRAMID=19 nodes), registered for connectivity-only analysis; `get_native('pyvista')` raises `MeshConversionError` (:66-71).
- Consumers: pyvista_adapter.py:199,285; vedo_adapter.py:123,152,231; volumetric/nonmanifold_faces.py:21; mesh_loader.py:510 (material reorder). `detect_element_type` (:93-116) and `validate_volume_cells` (:119-136) used only by tests in-package.
- No face-definition tables here — corner-face tables live in nas_parser.py:378-385 / gmsh_parser.py:84-86.

### utils/dimensionality.py — detection algorithm
- Decision order (:43-55): volume+surface → `MIXED_2D_3D`; volume only → `VOLUMETRIC_3D`; else SVD coplanarity → `PURE_2D` else `SURFACE_2D`.
- Coplanarity: bbox-extent shortcut (`min(ptp) < threshold` → planar, :118-120), n<3 → True (:114-116), else centered SVD, `σ3/σ1 < 1e-10` (default `_COPLANARITY_THRESHOLD` :14, :122-126).
- `detect_mesh_dimensionality` uses only the **first** volume-cell array (:75-78) — other arrays ignored. No production caller (tests only), matching the note's claim the registry doesn't filter by it.

### utils/triangle_intersection.py — approach & guards
- Pipeline `detect_self_intersections_manual` (:624-703): (1) convex early-out via `scipy ConvexHull` — convex ⇒ zero intersections (:602-621, :654-660); (2) KD-tree over centroids, radius `2 × max edge length` (:665-679); (3) subtract vertex-sharing adjacent pairs (:684-686, :39-62); (4) pairwise `triangles_intersect`.
- `triangles_intersect` (:342-391): bbox rejection → coplanar case via 2D projection + edge-intersection/barycentric containment (:242-285) → Möller-Trumbore edge-vs-triangle in both directions with segment-`t` clamped to `[tol, 1+tol]` (:288-339, :394-436).
- Distinctive guard `_is_surface_intersection` (:439-517) rejects chord-through-interior artifacts on tessellated curved surfaces: primary edge-depth test (both endpoints farther from a 2-triangle center estimate than the hit point ⇒ chord) (:477-488), fallback normal-orientation test (both centers on the interior side) (:490-517).
- Sole consumer: pyvista_adapter `_detect_self_intersections_manual` (:709-725), the default `method='manual'` (:655-688); `pymeshfix` path retained as legacy with known false positives (:734-794).

### mesh_loader.py — beyond dispatch
- `_validate_mesh_arrays` (:17-39): non-empty vertices, exactly Nx3, non-empty faces, K≥3, `max(faces) < n_vertices`. Applied on native-parser and surface fast paths (:192,:230,:314,:364) but **not** on the adapter `to_canonical` path, which instead uses a weaker emptiness check that accepts face-less pure-volume meshes (:297-302).
- Error taxonomy per stage: `ValueError` allowed-dirs violation (:163-169); `FileNotFoundError` (:171-172); `UnsupportedFormatError` (:176-180, :264-268); `MeshLoadError` everything else including translated `SystemExit` from meshio (:329-335) and empty trimesh scenes (:293-302).
- Adapter-route material extraction `_extract_material_data` (:443-494): pyvista `cell_data` first; reordered from grid-cell order into `volume_cells` block order via `celltypes` + `get_vtk_type_to_name` (:497-542), falling back to grid order on any count/type mismatch; then meshio `cell_data_dict` concatenation (:485-491).
- `**kwargs` forwarded only to `adapter.load_mesh` (:281) and save (:427,:430) — never to native parsers.

## Numbers & limits
- `MeshInput._MAX_CACHE_SIZE = 8` native objects (models.py:285); eviction is insertion-order FIFO (:502-504).
- `MeshIssue.to_dict` serializes at most **10** locations (models.py:106).
- FORMAT_SUPPORT covers **32** extensions (mesh_loader.py:44-112); `.fem` = meshio only (:64); `.msh` preference list `['meshio','pyvista']` (:67).
- nas_parser: short-field = 8-char columns (:288-297); `write_nas` EID base 100 (:228); continuation chunks of 8 fields (:257); `_NAS_WRITABLE` = 5 types (:185-191).
- gmsh_parser: 16 element codes; version must start `2.`; ASCII (`file-type 0`) only.
- metrics: DEFAULT surface thresholds — aspect_ratio 10.0, min_angle 15.0°, max_angle 120.0°, scaled_jacobian 0.5, condition 100.0, shape 0.25, area 1e-10, distortion 0.5, radius_ratio 2.0 (:26-36); volumetric — scaled_jacobian 0.1, volume 1e-12, aspect_ratio 0.1, condition 100.0, skewness 0.8, orthogonality 0.15, shape 0.3, distortion 0.5, radius_ratio 3.0, collapse_ratio 0.1, degenerate_volume 1e-12, sliver_ratio 1e-6 (:39-52); unknown-metric fallback 0.1 (:65-70); severity cutoffs 0.5/0.2/0.05 (:212-218); tail percentiles [1,5,10,20,50] (:189).
- triangle_intersection: default tolerance 1e-10 throughout; KD radius = 2×max edge (:674); plane-normal degeneracy 1e-15 (:93); barycentric denominator 1e-15 (:175); segment-degeneracy epsilon 1e-20 (:556-585).
- dimensionality: coplanarity σ-ratio threshold 1e-10 (dimensionality.py:14).
- `QualityMetrics` statistics: min/max/mean/median/std/p5/p95 (:138-146).

## Candidate additions for the HTML note
- Exception reality vs. definition: `InvalidMeshError` and `LibraryNotAvailableError` are dead; `OperationError` exists twice (core/exceptions.py:21 vs operations/base.py:214, the latter is the raised one); adapter errors form a second subtree under `AdapterError`/`MeshExaminerError`; MCP maps the whole tree to `business_error` vs `internal_error` for others.
- NASTRAN asymmetry: CHEXA/CPENTA keep quadratic forms (hex20/prism15) but CTETRA midside nodes are silently truncated to tet4; GRID `CP` ignored; long-field format unsupported; write_nas never writes both surface faces and volume cells (double-count guard) and starts EIDs at 100.
- Gmsh native parser accepts exactly 16 element codes and permutes 5 quadratic types to VTK order (meshio-mirroring); physical tags become materials only when **every** volume element is tagged (`_UntaggedElements` sentinel).
- PSOLID material route orders `cell_materials` by sorted element ID, not `volume_cells` enumeration order — alignment with parse_nas output is coincidental; `extract_materials_from_gmsh` is uncalled dead code; material_reader.py:89 has an operator-precedence bug returning `([], (None, None))` in the empty case.
- metrics.py computes no geometry (all values from adapters) and is the threshold/severity authority; `HIGHER_IS_WORSE` convention is duplicated in models.py:152.
- Self-intersection detector has two anti-false-positive devices not obvious from its name: full-mesh convex-hull early-out and chord-through-interior rejection, plus vertex-adjacency filtering and KD-tree radius 2×max-edge.
- pyramid14 is connectivity-only: excluded from the VTK CellType map, so pyvista conversion raises while topology_3d ops still work.
- `_validate_mesh_arrays` (strict, index-bounds-checking) is bypassed on the adapter volumetric path, which uses a weaker non-empty check to admit face-less volume meshes.

## Corrections
none
