# Operations — deep inspection

Files read: `operations/{__init__,base,_registry,operation_registry}.py`; all 13 `topological/` modules; `quality/{__init__,cell_quality}.py`; all 11 `volumetric/` modules; `geometric/__init__.py`; `core/models.py` (OperationConfig); `utils/metrics.py`; `utils/topology.py`.

## Findings

### Complete operation catalog (24 registered, insertion order = import order)
| ID | Cat | Detects | Severity rule | Dependency |
|---|---|---|---|---|
| topology.detect_nonmanifold_edges | TOPO | edges with >2 faces | determine_severity(rate vs faces) `nonmanifold_edges.py:84-85` | pure |
| topology.detect_nonmanifold_vertices | TOPO | vertices whose one-ring faces split into >1 component (Jiao&He AHF; tri fast path `nonmanifold_vertices.py:137-234`, general path `:236-309`) | determine_severity `nonmanifold_vertices.py:113-114` | pure |
| topology.detect_self_intersections | TOPO | intersecting non-adjacent triangles | always CRITICAL `self_intersection.py:117` | pyvista adapter `:70-92` |
| topology.detect_duplicate_vertices | TOPO | coincident vertices | determine_severity, CRITICAL downgraded to ERROR if rate<0.5 `duplicate_vertices.py:167-172` | adapter `get_duplicate_vertices`; numpy rounded-key fallback on MeshConversionError `:90-116,146-149` |
| topology.detect_boundary_edges | TOPO | 1-face edges grouped into loops | local: >20% ERROR, >5% WARNING else INFO `boundary_edges.py:101-106` | pure |
| topology.detect_inconsistent_orientation | ORIENT | faces traversing shared 2-face edges in same direction | determine_severity(inconsistent/total faces) `surface_orientation.py:200-201` | pure |
| topology.detect_disconnected_components | TOPO | >1 union-find component (path halving + rank `disconnected_components.py:74-108`) | always WARNING `:162` | pure |
| topology.detect_isolated_vertices | TOPO | vertices in no face/cell | >=5% WARNING else INFO `isolated_vertices.py:97-103` | pure |
| topology.detect_duplicate_faces | TOPO | same sorted vertex set (winding-agnostic) | >=5% WARNING else INFO `duplicate_faces.py:102-107` | pure |
| topology.detect_vertex_valence | TOPO | face-valence outside 4-8 (tri) / 3-6 (quad) | local: >20% ERROR, >5% WARNING `vertex_valence.py:201-208` | pure (`utils/topology.compute_vertex_valence` counts face occurrences, not distinct edges `utils/topology.py:45-58`) |
| topology.detect_edge_dihedral_angle | GEO | internal edges with dihedral <20° or >160° (180° − normal angle) `edge_dihedral.py:122-130` | local: >20% ERROR, >5% WARNING `:196-203` | pyvista adapter for face normals (hardcoded `'pyvista'` `:99`) |
| topology.detect_edge_length_ratio | GEO | edge len / min neighbor-edge len > 10 | local: >10% ERROR, >5% WARNING `edge_length_ratio.py:201-208` | pure |
| quality.compute_cell_quality | QUAL | 2D element shape metrics | standard 50/20/5 via `_determine_severity` + min_angle ERROR/CRITICAL→WARNING downgrade `cell_quality.py:158-165,225-234` | pyvista/trimesh adapter |
| topology_3d.detect_nonmanifold_faces | TOPO | faces shared by >2 cells | determine_severity `nonmanifold_faces.py:103,200-201` | pure (`_face_tables.build_face_to_cell_map`) |
| quality_3d.check_orientation_consistency | ORIENT | inverted cells + inconsistent winding | inverted CRITICAL `orientation_consistency.py:174`; winding ERROR `:237` | inverted needs `mesh.get_native('pyvista')` (skipped on MeshConversionError `:103-110`); winding pure |
| topology_3d.detect_duplicate_cells | TOPO | cells with identical sorted vertex set | determine_severity `duplicate_cells.py:90-91` | pure |
| topology_3d.detect_orphaned_cells | TOPO | cells referencing index <0 or >= n_vertices | always CRITICAL `orphaned_cells.py:104` | pure |
| topology_3d.check_boundary_closure | TOPO | free(=1)/non-manifold(>2) edges on the 1-cell boundary face set | determine_severity, INFO promoted to WARNING `boundary_closure.py:118-121` | pure |
| quality_3d.compute_volume_cell_quality | QUAL | per-metric threshold violations | standard 50/20/5 `volume_cell_quality.py:510-519` | pyvista adapter / linearized grid |
| quality_3d.assess_geometric_validity | GEO | degenerate (|vol|<1e-12) CRITICAL; sliver (|vol|/max_edge³<1e-6) WARNING `geometric_validity.py:132,221` | fixed per defect class | pyvista adapter (created in `__init__` `:53`); pure-numpy sliver detectors |
| quality_3d.check_boundary_layer | QUAL | BL growth/AR/orthogonality/coverage/count | per check: count WARNING `boundary_layer_quality.py:201-220`; growth/AR/ortho standard `:563-572`; coverage >=0.9 WARNING / >=0.5 ERROR / else CRITICAL `:540-544`; height INFO `:236-240` | pure (own face map, linear types only `:589-594`) |
| quality_3d.detect_material_interfaces | QUAL | 2-cell faces between different materials | always INFO `multi_material.py:298` | pure; needs `cell_materials` |
| quality_3d.check_interface_quality | QUAL | interface cells: scaled Jacobian<0.1 (tet4), vol<1e-12 | Jacobian standard `:578-586`; volume CRITICAL if >10% else ERROR `:532` | pure; needs `cell_materials` |
| quality_3d.per_material_statistics | QUAL | per-material stats report (not defects) | always INFO `multi_material.py:646` | pure; needs `cell_materials` |

Category vs subpackage mismatch: `edge_dihedral`, `edge_length_ratio`, `geometric_validity` are GEOMETRIC; `surface_orientation`, `orientation_consistency` are ORIENTATION — category is metadata-declared, not subpackage-derived.

### base.py contract
- `OperationMetadata` (base.py:24-49): required operation_id, display_name, description, category, supported_libraries, recommended_library; defaults requires_adjacency=False, complexity="O(n)", tags=[].
- `BaseOperation.__init__` (base.py:77-85): config default `OperationConfig()`; `_execution_count`, `_total_time`, `last_quality_metrics=[]`.
- `run()` beyond the documented timing/error wrap: temporarily sets `mesh._caching_enabled = active_config.enable_caching` and restores in `finally` (base.py:114-115,131); resets `last_quality_metrics=[]` before each execute (`:118`); accepts per-call config override (`:113`). `statistics` property exposes count/total/avg (`:133-141`).
- `QualityOperation` (base.py:147-211): separate ABC (metric_name/compute/get_threshold, no run()); `get_recommended_severity` duplicates the 50/20/5 thresholds; no concrete subclass exists in-tree.
- `register_operation()` (base.py:222-235) registers what it is given; its docstring claims class-decorator use, but decorating a class would register the class object (no instantiation) — `_register_decorator` (`:238-251`, aliased `register_operation_decorator` `:255`) is the variant that instantiates.

### _registry.py vs operation_registry.py
- `operation_registry.py:284` creates the real singleton `registry = OperationRegistry()`; every concrete module imports it and registers at import time.
- `_registry.py:16-31` is a lazy accessor returning that same instance, existing only to break the cycle base→_registry→operation_registry→base (`_registry.py:2-14`).
- Registry internals beyond selection: `unregister(operation_id)` removes from both maps (operation_registry.py:109-124); categories pre-initialized for every `IssueCategory` (`:74-80`); `get_statistics()` aggregates per-op `_execution_count`/`_total_time` and per-category counts (`:252-269`); `__len__/__contains__/__iter__` (`:271-278`); `create_workflow` collects `last_quality_metrics` after each op and measures whole-loop `processing_time` (`:190-205`).

### volumetric/_face_tables.py
- `ELEMENT_FACES` (`:38-67`): outward right-hand-rule corner face tuples for tet4/hex8/prism6/pyramid5; recalibrated 2026-10-28 — the previous table had inward-wound hex8 bottom, prism6 bottom cap and pyramid5 base, which made `check_orientation_consistency` report winding errors on regular grids (`:27-37`).
- `ELEMENT_CORNER_COUNT` (`:72-86`, 11 types) and `LINEAR_BASE_TYPE` (`:90-98`, 7 quadratic→4 linear) let quadratic cells reuse linear tables because VTK puts corners first.
- `QUADRATIC_EDGE_MIDSIDES` (`:114-150`): (corner_a, corner_b, midside_slot) triples extracted from VTK 9.7.1 `cell.GetEdge(k)`; face/body-centre nodes (hex27 20-26, wedge18 15-17, pyramid14 13) deliberately excluded.
- `linearized_volume_cells` (`:170-209`) truncates quadratic rows to corners and vstacks same-base blocks preserving global cell numbering (merged index i == original cell i).
- `build_face_to_cell_map` (`:212-259`): sorted face key → (global_cell_idx, oriented face); unknown element types are skipped but still advance the index.
- Consumers: shared map → nonmanifold_faces.py:23, orientation_consistency.py:26, boundary_closure.py:30. Local (linear-only, dict-descriptor) copies of the map in boundary_layer_quality.py:579-615 and multi_material.py:55-91 — these two ops silently ignore quadratic cells. Linearization helpers → volume_cell_quality.py:36-39, geometric_validity.py:28.

### Non-trivial mechanics
- **self_intersection**: no in-module algorithm; delegates to `adapter.detect_self_intersections(method='manual', tolerance=1e-10)` (self_intersection.py:86-90); the KD-tree/Möller-Trumbore/coplanar logic lives in the adapter. MeshConversionError → `[]` (`:81-82`), any other exception → `[]` (`:91-92`): failures are silent skips, not issues. One issue per face via `reported_faces` dedup (`:99-131`). `min_area_threshold=1e-10` (`:50`) is set but never read (dead attribute).
- **surface_orientation**: only edges with exactly 2 faces are checked; non-manifold (>2) and boundary (1) edges skipped (surface_orientation.py:150-155); both faces of a bad edge are flagged (`:162-165`); face adjacency built from 2-face edges only (`:325-331`); BFS components = Borodin-style patches (`:335-378`); locations capped 100 (`:181`), inconsistent_edges sample capped 20 (`:226`); single aggregated issue (`:231`).
- **boundary_closure**: boundary faces = exactly-1-cell entries; counts undirected edge incidences over that face set; closed 2-manifold boundary → no issue; single-cell, filled grid, interior cavity all pass (boundary_closure.py:33-50,93-113).
- **orientation_consistency inverted path**: prefers `grid.cell_data['Volume']` then `cell_quality('volume')`, filters the VTK −1 sentinel (`volumes != -1.0`), then falls back to pure-numpy signed volumes: tet4 determinant, hex8→5 tets, prism6→3, pyramid5→2 (`_HEX8_TETS` etc. `orientation_consistency.py:354-378,381-441`); inverted = volume < −config.tolerance (`:156-157`).
- **orphaned_cells** is a misnomer: it detects out-of-range vertex references (would crash native libs), not unattached cells (orphaned_cells.py:5-7,78).
- **duplicate_cells**: single map for mixed meshes because keys of different node counts never collide (duplicate_cells.py:29-33,77-81); reports `duplicate_of_cell` first-seen index (`:114`).
- **nonmanifold_faces**: violation rate over unique faces; per-location `sharing_cell_indices` capped 20 (`nonmanifold_faces.py:181`); metadata includes affected_cell_types and max_sharing_count (`:123-124`).
- **geometric_validity**: per-type sliver detectors use exact edge sets (tet 6, hex 12, prism 9, pyramid 8 edges, `geometric_validity.py:297-312,368-372,420-424,472-475`); volume from VTK on the (possibly linearized) grid; sliver indices re-attributed to original element types (`:172-197`); sliver issue tagged `metadata['linearized']=True` when judged on corners (`:229-233`).
- **boundary_layer_quality**: seeds = boundary faces that are *base* faces (BASE_FACE_INDICES prism6/hex8 [0,1], pyramid5 [0], `:51-55`) of BL-type cells (prism6/hex8/pyramid5, `:45`), filtered by two gates: thin-cell (in-plane/height ≥ 2.0) and growth-away (BL neighbour through opposite base face must not be < 0.9× seed height — rejects the outer side of the stack) (`:60-70,663-713`). Layers traced by BFS through non-boundary faces into BL-type cells only, max 100 layers (`:762-829`). Height of layer 0 = mean cell height (centroid distance base↔top; scipy pdist fallback `:928-949`); later layers = distance between layer-mean centroids (`:905-921`). Growth ratio = h(i+1)/h(i) > 1.2 flags transition (`:269-279`). Coverage excludes traced-stack faces and adjacent transition (core) cells from wall candidates (`:495-534`).
- **multi_material**: interfaces = exactly-2-cell faces with differing materials, grouped by sorted (mat1,mat2); face area fan-triangulated from vertex 0; material names from `MeshInput._material_names` else `Material_<id>` (multi_material.py:275-317,356-372,727-731). InterfaceQuality scaled Jacobian computed for tet4 only, as det(J)/Π‖edge‖ at node 0 (`:193-216`); volumes via abs-valued tet decompositions (hex 5-tet, prism 3-tet, pyramid 2-tet, bounding-box fallback `:115-190`). PerMaterialStatistics reports avg/min/max/std quality (tet4 only) + volume stats per material (`:660-720`).
- **edge_dihedral**: internal-edge only; severity denominator is *all* unique edges while `total_internal_edges` is reported separately (edge_dihedral.py:160,175) — rate is diluted by boundary/non-manifold edges.
- **edge_length_ratio**: neighborhood = edges sharing either endpoint (edge_length_ratio.py:100); min over neighbors; python loop over edges.
- **vertex_valence**: mesh type test = all-3-column array or mean verts/face < 3.5 (`:184-199`); volume meshes: vertices referenced only by cells (absent from faces) are excluded from irregular set (`:121-138`).
- Volume-mesh guards in 2D ops: disconnected_components unions volume-cell vertices (`disconnected_components.py:114-119`); isolated_vertices counts cell-referenced vertices as used (`:73-75`).

## Numbers & limits
- OperationConfig (frozen, models.py:233-241): tolerance 1e-6, quality_threshold 0.1, enable_caching True, max_iterations 1000, parallel_enabled True, library_preference None.
- determine_severity (utils/metrics.py:212-219): >0.5 CRITICAL, >0.2 ERROR, >0.05 WARNING, else INFO.
- `_MAX_LOCATIONS = 100` in edge_dihedral.py:58, edge_length_ratio.py:48, vertex_valence.py:60, boundary_closure.py:53, duplicate_cells.py:36, nonmanifold_faces.py:205, orientation_consistency.py:54, orphaned_cells.py:36, boundary_layer_quality.py:58, multi_material.py:48; slices `[:100]` surface_orientation.py:181, geometric_validity.py:136,203, volume_cell_quality.py:351.
- Metadata caps: inconsistent_edges 20 (surface_orientation.py:226); extreme_edges/high_ratio/irregular/violating_transitions 50 (edge_dihedral.py:189, edge_length_ratio.py:172, vertex_valence.py:177-178, boundary_layer_quality.py:324); sharing_cell_indices 20 (nonmanifold_faces.py:181).
- Thresholds: duplicate-vertex tolerance 1e-6 (duplicate_vertices.py:61; fallback rounding floor 1e-12 `:107`); self-intersection tolerance 1e-10 (`:89`); dihedral 20–160°; length ratio 10×; valence 4–8/3–6, mesh-type cut 3.5; degenerate 1e-12 / sliver 1e-6 (geometric_validity.py:51-52); BL growth 1.2, AR 1000, orthogonality 20°, min layers 3, max trace 100 layers, thin ratio 2.0, growth-away 0.9, coverage 0.9/0.5; interface SJ 0.1 / vol 1e-12, volume-CRITICAL cut 0.1.
- volume_cell_quality DEFAULT_THRESHOLDS (volume_cell_quality.py:98-119): scaled_jacobian 0.1, volume 1e-12, aspect_ratio 0.1, condition 100.0, shape 0.3, skewness/skew 0.8, orthogonality 0.15, warpage 0.5, equiangle_skew 0.25, stretch 0.25, taper 0.5, radius_ratio 10.0, collapse_ratio 0.1, midside_deviation 0.1; `_higher_is_worse` = {condition, warpage, equiangle_skew, taper, radius_ratio, midside_deviation} (`:176-179`); metric skipped when all values are the −1 sentinel (`:498-502`); issue metadata adds p5/p95 (`:374-375`).
- MeshInput._MAX_CACHE_SIZE = 8 (models.py:285). IssueCategory buckets pre-created for every enum member (operation_registry.py:74-80).

## Candidate additions for the HTML note
- 24 operations total; catalog mapping ID → category (which does not follow subpackage layout) → pure-canonical vs adapter dependency; 7 ops touch an adapter (self_intersection, duplicate_vertices, edge_dihedral, cell_quality, orientation_consistency, volume_cell_quality, geometric_validity), of which 4 hardcode pyvista (edge_dihedral, orientation_consistency, volume_cell_quality, geometric_validity).
- `run()` also toggles `mesh._caching_enabled` from config and resets `last_quality_metrics` per call (base.py:114-118).
- `_registry.py` is a lazy same-instance accessor breaking the base→operation_registry import cycle; `unregister()` and `get_statistics()` exist on the registry.
- `_face_tables.py` is the single source of volumetric face topology, including the 2026-10-28 outward-winding recalibration that fixed false winding errors on regular hex grids, and the VTK-derived midside slot table.
- Fixed-severity ops: self_intersections CRITICAL, disconnected_components WARNING, orphaned_cells CRITICAL, material interfaces/per-material stats INFO; duplicate-vertices CRITICAL→ERROR downgrade; boundary-closure INFO→WARNING promotion.
- Silent-skip behavior on conversion failure: self_intersection and duplicate-vertices (numpy fallback) return no issues rather than erroring.
- boundary_layer seeding gates (thin-cell ≥2×, growth-away 0.9×) exist to stop regular hex grids from being misread as boundary layers.
- Report-size caps are pervasive: 100 locations per issue, 20–50-entry metadata samples.

## Corrections
none
