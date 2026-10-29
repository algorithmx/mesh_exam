# Visualization + tests + e2e infrastructure — deep inspection

Files read in full or in structured part: `mesh_issue_examiner/visualization/__init__.py`, `renderer.py`; `tests/conftest.py`, `test_visualization.py`, `test_sample_meshes_e2e.py`, `test_thingi10k_tier1.py`, `test_thingi10k_tier2.py`, `test_large_mesh_smoke.py`, docstrings/builders of the other 25 test files; `scripts/generate_sample_meshes.py`, `scripts/fetch_thingi10k.py`; `e2e/thingi10k/selection.json` (head, tier2 sample, calibration tail; 5943 lines); `docs/thingi10k-cross-validation.md`; `pytest.ini`; `mcp/server.py` (render tool + handler); `git ls-tree` inventories of `e2e/`, `tests/`, `scripts/`.

## Findings

### Visualization module (`mesh_issue_examiner/visualization/`)
- Public API is exactly 5 names re-exported in `__init__.py:8-22`: `render_mesh`, `RenderResult`, `VisualizationError` (re-imported from `core.exceptions`), `SEVERITY_COLORS`, `VIEW_ANGLES`. Single function-style API; no classes to instantiate except the `RenderResult` dataclass (renderer.py:62-70).
- Backend is PyVista/VTK only (`import pyvista as pv` inside `render_mesh`, renderer.py:272). Offscreen GL capability is probed once and cached in module global `_offscreen_status` via a 32×32 throwaway plotter (renderer.py:73-90). All rendering is serialized behind a module-level `threading.Lock` because VTK render windows are not thread-safe (renderer.py:34-35, 317).
- Renders all three canonical mesh kinds: surface, volumetric (surface-extracted via `extract_surface(algorithm="dataset_surface")`, renderer.py:335-342), and point clouds (`add_points`, renderer.py:324-331). Scalar coloring by `"material"` (cell_materials) or by any `QualityMetrics.metric_name` from an `InspectionResult`, with the VTK `-1` undefined sentinel converted to NaN (renderer.py:181-213, 207).
- Issue highlighting: markers from `IssueLocation.coordinates`, or resolved from vertex coords / face centroid / volume-cell centroid by element type (renderer.py:126-178); colored per severity (red/orange/yellow/gray, renderer.py:49-54); default highlight = critical+error only (renderer.py:235).
- Sole non-test caller is the MCP server: `mesh_render` tool schema at `mcp/server.py:326-421`, handler `_handle_render` at `mcp/server.py:731-796` which lazy-imports `render_mesh` (server.py:767), defaults output to `<tempdir>/<mesh_id>_render.png` (server.py:757-759), and requires a prior `mesh_inspect` for metric scalars or issue highlighting (server.py:742-748). FastMCP layer also exposes it (server.py:1115; test_mcp_save_tool.py:78).
- `test_visualization.py` skips wholesale when pyvista is missing or offscreen is unavailable (`pytest.importorskip` + module-level `pytest.skip`, test_visualization.py:16, 32-36); validity check is PNG magic bytes + >1000 bytes (test_visualization.py:38-43). It also pins the MCP `mesh_render` surface (7 tool tests, test_visualization.py:395-517).

### Test suite layout, conftest, markers
- `mesh_issue_examiner/tests/` = 33 files: `__init__.py`, `conftest.py`, 31 `test_*.py`.
- Single conftest fixture: session-scoped **autouse** `sample_meshes` regenerates `e2e/sample_meshes/` by exec'ing `scripts/generate_sample_meshes.py` when the sentinel `isolated_vertices_surface.vtk` is missing; skips existing files so manual edits survive (conftest.py:15-32).
- `pytest.ini:8` deselects by default: `addopts = -m "not slow and not thingi10k_remote"`. `slow` = 1M-cell smoke (test_large_mesh_smoke.py:25); `thingi10k_remote` = tier-2 network suite (test_thingi10k_tier2.py:33). Optional-dep skips: `h5py`/`netCDF4` for CGNS/Exodus (test_sample_meshes_e2e.py:629, 652), `pyvista`, `meshio` elsewhere.

### Test coverage map (area → files → pinned behavior)
| Area | File(s) | Pins down |
|---|---|---|
| Core models/enums | test_core_models.py | severity ordering, IssueLocation/MeshIssue/QualityMetrics/InspectionResult/MeshInput contracts |
| Registry/operation base | test_operations_registry.py, test_operations.py | metadata, run() counting, error paths; basic nonmanifold/boundary/quality ops on synthetic meshes |
| Adapter layer | test_adapters.py, test_adapter_normalization.py | get_faces packed-array parsing incl. Bug #4 malformed input; capability matrix for 5 adapters; mixed wedge+tet fan-triangulation regression |
| Parsers | test_gmsh_parser.py, test_nas_parser.py, test_material_loading.py | Gmsh 2.2 native parse, tet10 Gmsh→VTK permutation, meshio-equivalence; CHEXA/CPENTA continuation cards, hex20/prism15 retention; material-array→volume_cells block-order alignment |
| Surface ops | test_bad_meshes.py, test_three_new_operations.py, test_surface_orientation.py, test_triangle_intersection.py | umbrella nonmanifold edge, touching-tet nonmanifold vertex; disconnected/isolated/duplicate-face detectors; winding consistency (Borodin et al. algorithm); Möller–Trumbore + manual SI detector (uses `icosphere_l1.obj` fixture with a face-order contract, generate_sample_meshes.py:182-190) |
| Volumetric ops | test_volumetric_connectivity.py, test_volumetric_operations.py, test_boundary_layer_quality.py, test_higher_order_elements.py, test_multi_material.py, test_midside_deviation.py, test_extended_quality_metrics.py | duplicate/orphaned/boundary-closure incl. T-junction; sliver/degenerate/inverted tets; BL tracing/growth; quadratic registration (hex27/wedge18/pyramid14) + corner face tables; material interface/quality/per-material stats; midside_deviation curvature metric; warpage/skew/stretch/taper/radius_ratio thresholds |
| Quality-metric surfacing | test_quality_metric_surfacing.py | metrics reach InspectionResult + `mesh_get_quality_metrics` (the gap-fix regression); 7-stat blocks (min/max/mean/median/std/p5/p95) |
| MCP server | test_mcp_modules.py, test_mcp_save_tool.py, test_visualization.py | request/response models, tool schemas, routing; mesh_save round-trips + `allowed_save_dirs` guard + 6-tool default surface; mesh_render tool |
| Utils | test_utils_and_metrics.py, test_dimensionality.py, test_imports.py, test_granular_reporting.py | DEFAULT_THRESHOLDS values, severity boundaries (0.05/0.2/0.5), dimensionality detection, import-chain/circular-import guards, valence/dihedral/edge-ratio ops |
| e2e real files | test_sample_meshes_e2e.py | full MCP pipeline over 30+ fixtures (below) |
| External corpus | test_thingi10k_tier1.py (default), test_thingi10k_tier2.py (opt-in) | frozen detector contracts; heavy-mesh robustness |
| Perf | test_large_mesh_smoke.py (`slow`) | 100³=1,000,000 hex8 / 1,030,301 vertices full inspect passes with passed=True (test_large_mesh_smoke.py:9-14, 102-114) |

### e2e sample-mesh fixture inventory (37 tracked files under `e2e/sample_meshes/`)
| Fixture | Defect scenario embodied | Format |
|---|---|---|
| perfect_cube_surface, perfect_hexahedron, pyramid, wedge, torus | zero-issue deltahedra chosen so AR/min/max-angle, dihedral∈(20°,160°), valence 4–8 never fire (generate_sample_meshes.py:7-34) | stl |
| icosphere_l1 / l2 | clean topology, guaranteed quality violations via 1.5× z-stretch | obj / stl |
| non_manifold_edge | cube + duplicated square face (3 faces/edge) | stl |
| non_manifold_vertex | two regular octahedra touching at one apex (WARNING-only → still passes) | stl |
| degenerate_triangle | tet + two 20:1 slivers | stl |
| self_intersecting | two X-crossing triangles | stl |
| varying_quality | icosahedron with vertex 0 scaled 30× (AR 10–16.5 on 5 faces) | stl |
| flipped_normals / perfect_tetrahedron | one face reversed → inconsistent orientation | stl |
| disconnected_components, duplicate_vertices, isolated_vertices | 2 disjoint tris; duplicated tet face; cube + 3 stray points | stl, stl, vtk |
| open_boundary, holey_plane, multi_hole_plane, subdivided_plane, cylinder, perfect_triangle, perfect_square | boundary-edge contracts (open surfaces) | stl |
| shell_structure | thin-walled double cube, actually watertight (boundary=False) | stl |
| closed_volume / duplicate_cell / leaky_volume | 2×2×2 hex8 clean; duplicated hex; T-junction (boundary_closure fires sub-failing → passes) | vtu |
| tet10_volume / quadratic_sliver | 2 clean tet10; single squashed tet10 (sliver via corner linearisation, ratio 4.2e-8) | vtu |
| nas_mixed_volume | CTETRA + 20-node CHEXA + CPENTA → tet4/hex20/prism6, 30 verts, 3 disjoint cells | nas |
| gmsh_mixed_volume | tet10 (Gmsh node order) + hex8, physical groups 1 "fluid"/2 "solid" | msh |
| two_material_volume | 2×2×1 hex, materials [1,2,1,2] split along x | vtu |
| boundary_layer / boundary_layer_bad_growth | prism6 stack (5 layers) + tet4 cap; growth 1.15 clean vs 3× jump at layer 3 (max_growth_ratio ≈2.14 > 1.2, ERROR) | vtu |
| cgns_volume / exo_volume | two-unit-tet (meshio CGNS is tetra-only) / clean 2×2×2 hex via meshio delegation; skipped without h5py / netCDF4 | cgns / exo |

- `test_sample_meshes_e2e.py` ties fixtures to expectations via per-group tables: `PERFECT_MESHES` (5, assert passed & total_count==0, lines 72-109), `TESSELLATED_MESHES` (2, assert issues>0), `DEFECTIVE_CASES` (10-entry dict with `expect_passed` + `expect_keywords` matched as substrings of operation_ids, lines 156-247), `VOLUME_CASES` (5-entry dict adding `forbid_keywords`, lines 254-334), plus dedicated classes for NASTRAN/Gmsh/CGNS/Exodus/materials/BL (lines 431-842), open/closed boundary contracts (849-920), category-scoped inspect (928-956), severity/category filter + limit/offset pagination (964-1025), and cache clearing (1033-1068). A parallel machine-readable contract table `CONTRACTS` (≈37 entries incl. `zero`, `boundary`, `min_issues`, `min_warnings`, `requires`, `material_array`) drives `generate_sample_meshes.py --verify` (generate_sample_meshes.py:370-507, 911-991).

### Thingi10K cross-validation design
- Two tiers from the HuggingFace mirror (`https://huggingface.co/datasets/Thingi10K/Thingi10K/resolve/main`, fetch_thingi10k.py:55), stdlib urllib only, deterministic selection (no RNG) from two metadata CSVs (geometry_data.csv labels + input_summary.csv license/ext), eligibility ≥20 vertices, ≥40 faces, ext ∈ {stl,obj,off,ply}, excluding 5 known-corrupt file_ids (fetch_thingi10k.py:61, 70, 266-272).
- Tier 1 (offline, default suite): 41 meshes ≤5000 facets committed under `e2e/thingi10k/tier1/` + `selection.json`; quotas 7 clean_canary/9 si_otherwise_clean/6 non_edge_manifold/4 non_vertex_manifold/5 open/3 non_oriented/3 multi_component/2 degenerate_faces/2 duplicate_faces; 4 historical IDs pinned (102625, 1224388, 1038433, 131372) (fetch_thingi10k.py:82-92, 68).
- Per-mesh frozen contract in selection.json: `contract.ops` (probe list), `expect` (label-mapped ops that fired), `allow` (other ops that fired), `forbid` (probed ops that stayed silent), `timings_s`; label→op mapping from `CATEGORY_EXPECT_OPS` (fetch_thingi10k.py:112-124). Tests assert all three directions via probe-granularity prefix matching (test_thingi10k_tier1.py:111-163); sha256 per file is verified (126-134); corpus hygiene: ≤60 MB, ≤50 meshes, no orphan files (169-196).
- SI detector `topology.detect_self_intersections` is O(n²) (~130 s at 5k faces) so it is probed only for `si_otherwise_clean` meshes and small canaries, facet-capped at 2500 (fetch_thingi10k.py:102-105; docs/thingi10k-cross-validation.md:44-47); tier 2 probes every op except SI (test_thingi10k_tier2.py:79-83).
- Label-backed misses (FN) and clean-canary firings (FP) are recorded, never asserted, in `selection.json → calibration.known_gaps` (fetch_thingi10k.py:641-740; test_thingi10k_tier1.py:180-187).
- Tier 2 (opt-in `thingi10k_remote`): 38 manifest entries (12 multi_defect, 10 extreme_stats, 8 large_clean, 6 large_self_intersecting, 2 corrupt_robustness), downloaded on the fly into gitignored `e2e/thingi10k/tier2/` by a module fixture that shells out to the fetch script (resumable, sha256-pinned, 429 backoff) and skips cleanly offline (test_thingi10k_tier2.py:52-70). Assertions are smoke-level: exact face-count match vs dataset ground truth (vertex count only bounded 0<v≤3f), full non-SI inspect completes, multi_defect must fire ≥1 op, corrupt files must raise `MeshLoadError` (test_thingi10k_tier2.py:93-141).
- Selection entry fields (selection.json:6-37): `file_id, category, ext, filename, path, url, license, labels{closed, self_intersecting, edge_manifold, vertex_manifold, oriented, pwn, solid, degenerate_faces, duplicate_faces, num_*}, stats{num_vertices, num_faces, max_aspect_ratio}, sha256, size_bytes` (+ `contract` for tier 1). Top level adds `dataset, base_url, generated` (2026-10-28) and `calibration{date, operations[24], projected_suite_seconds=154.2, known_gaps}` (selection.json:5890-5941).

### Scripts
- `generate_sample_meshes.py`: fully deterministic (no RNG) builders for all 37 fixtures above, incl. 18 hand-optimized torus vertices embedded verbatim (lines 73-92), `--force`/`--verify` modes; `verify()` runs the real MCP load→inspect→get_issues pipeline against `CONTRACTS` (911-991). Written via trimesh (surfaces), pyvista (vtk/vtu), embedded ASCII decks (nas/msh), meshio (cgns/exo).
- `fetch_thingi10k.py`: metadata download → deterministic select → tier fetch (atomic `.part` writes, 6 attempts, Retry-After-aware 429 backoff ≥15 s, 1.5 s pacing) → `--calibrate` (per-op probes with timings, freezes expect/allow/forbid + known_gaps, warns if projected suite >180 s) → `--verify` sha256 audit (131-172, 578-638, 641-740).

## Numbers & limits
- Renderer: default 1200×800; width/height clamped [64, 8192] px (renderer.py:224-225, 284-286); opacity ∈ (0,1] (282); zoom must be >0 (120-123); camera at 2.5× bounding-box diagonal from center (108-114); elevation clamped ±89.°9 (99); up-vector fallback when |dir_z|≥0.99 (117); marker cap `_MAX_MARKERS=5000` (56-57); marker size 14, point-cloud size 6 (329, 380); 7 named views, 4 severity colors (38-54); suffixless output gets `.png` (291-292).
- MCP get_issues: limit 1–200, default 50 (server.py:279-290, 697).
- pytest: `slow` and `thingi10k_remote` deselected by default (pytest.ini:8).
- Corpus caps: tier1 ≤50 meshes / ≤5000 facets / ≤2 MiB per file / ≤60 MB total / SI-probed ≤2500 facets; tier2 ≤50 meshes / ≤300k facets / ≤40 MiB per file; eligibility ≥20 verts, ≥40 faces (fetch_thingi10k.py:73-79, 105, 266-272); budget re-asserted in tests (test_thingi10k_tier1.py:169-178).
- Actual corpus: tier1 = 41 files; tier2 manifest = 38 entries; calibration over 24 operations, projected 154.2 s; known_gaps = 21 (15 fp, 6 fn) (selection.json:5920-5940).
- Large-mesh smoke: N=100 → 1,000,000 hex8 cells, 1,030,301 vertices, 6·N²·2 boundary triangles (test_large_mesh_smoke.py:27, 95-114).
- Fixture geometry constants: GYRO_APEX_Z=1.2, GYRO_BAND_Z=0.5, PENTA_RADIUS=0.85, ICOSPHERE_STRETCH=1.5, TORUS nu=6×nv=3 with 20°/ring twist; BL heights [0.2,0.23,0.2645,0.3042,0.3498] (r=1.15) vs bad [0.2,0.23,0.69,0.7935,0.9125] (3× jump); quadratic_sliver ratio ≈4.2e-8 < 1e-6 sliver threshold, > 1e-12 degenerate tolerance; varying_quality spike ×30 (generate_sample_meshes.py:61-92, 542-547, 654-673, 707-708).
- DEFAULT_THRESHOLDS pinned in tests: aspect_ratio 10.0, min_angle 15.0, max_angle 120.0, scaled_jacobian 0.5, condition 100.0, shape 0.25, area 1e-10, distortion 0.5, radius_ratio 2.0; severity boundaries 0.05/0.2/0.5 (test_utils_and_metrics.py:36-46, 100-125).

## Candidate additions for the HTML note
- A `mesh_issue_examiner.visualization` module exists: PyVista-only, offscreen-first, thread-lock-serialized `render_mesh()` producing PNGs with named views, scalar coloring (material/metric), and severity-colored issue markers capped at 5000; its only consumer is the MCP `mesh_render` tool, which lazy-imports it and defaults output to `<tempdir>/<mesh_id>_render.png`.
- Test architecture: 31 test files + session-autouse conftest that self-heals missing e2e fixtures by exec'ing the generator script; two non-default markers (`slow`, `thingi10k_remote`) deselected via `pytest.ini` addopts; env-dependent skips (pyvista/offscreen, h5py, netCDF4) keep the suite portable.
- The e2e fixture corpus: 37 deterministic fixtures across 8 formats (23 stl, 8 vtu, obj/vtk/nas/msh/cgns/exo) with a dual contract system — Python dicts in `test_sample_meshes_e2e.py` and the parallel `CONTRACTS` table verified by `generate_sample_meshes.py --verify`.
- The "perfect" fixtures are mathematically engineered (gyroelongated bipyramid, octahedron, pentagonal bipyramid, icosahedron, hill-climbed triangular-tube torus) because flat-faced polyhedra can never be zero-issue under the current dihedral/valence thresholds.
- Thingi10K two-tier cross-validation: tier 1 = 41 committed meshes with frozen per-mesh expect/allow/forbid contracts + sha256 pins + 21 documented known gaps (15 FP / 6 FN); tier 2 = 38 opt-in remote meshes (50k–300k facets + 2 corrupt files asserting graceful `MeshLoadError`), SI detector excluded for complexity.
- Calibration workflow (`--calibrate`) freezes *observed* detector behavior with timings and a projected-suite-seconds budget (154.2 s vs 180 s warning threshold), making detector drift a reviewable `selection.json` diff.
- Scale headroom pinned by the `slow` 1M-hex8 smoke test and tier-2 face-count-exact assertions against dataset ground truth.

## Corrections
- None against the existing note (this area is absent from it). Two intra-repo doc/code drifts found: (1) `conftest.py:5-6` and `test_sample_meshes_e2e.py:5` state fixtures are "NOT committed (e2e/ is gitignored)", but `git ls-tree HEAD` shows all 37 sample meshes and the 41 tier-1 STLs are tracked; (2) `docs/thingi10k-cross-validation.md:53` says 21 gaps = "14 FP, 7 FN" while the manifest records 15 FP + 6 FN, and the doc's "38 heavier meshes" + "two corrupt files" double-counts — the manifest's tier2 array is 38 entries *including* the 2 corrupt ones (quotas sum to 36; the script comment at fetch_thingi10k.py:94 "sum 38 + 2 corrupt = 40" misstates the sum).
