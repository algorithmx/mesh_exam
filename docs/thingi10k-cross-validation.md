# Thingi10K Two-Tier Cross-Validation Harness

Cross-validation of the mesh-issue detectors against
[Thingi10K](https://github.com/Thingi10K/Thingi10K) — 10,000 real-world
meshes with per-file ground-truth defect labels (self-intersections,
non-manifold edges/vertices, open boundaries, orientation, duplicate and
degenerate faces) — without keeping the ~70 GB dataset on disk.

Data source: the HuggingFace mirror, addressed per-file with plain HTTPS
(no `thingi10k`/`datasets` dependencies, stdlib `urllib` only):

* `…/metadata/geometry_data.csv` — per-file_id ground-truth labels
* `…/metadata/input_summary.csv` — license + original link (extension)
* `…/raw_meshes/{file_id}.{ext}` — individual raw mesh files

## Tier 1 — offline, default suite (`test_thingi10k_tier1.py`)

41 small (≤5,000 faces), deliberately nasty meshes under
`e2e/thingi10k/tier1/` (~19 MB, **tracked in git**) plus the frozen
manifest `e2e/thingi10k/selection.json`. Every clone runs tier 1 offline
as part of the default suite (~2.5 min added runtime).

Category quotas: 7 clean canaries, 9 self-intersecting-but-otherwise-clean,
6 non-edge-manifold, 4 non-vertex-manifold, 5 open, 3 non-oriented,
3 multi-component, 2 degenerate-faces, 2 duplicate-faces; the four
historical cross-validation IDs (102625, 1224388, 1038433, 131372) are
pinned into their natural categories.

### Frozen contracts

For every tier-1 mesh the manifest stores a **probe set** (which
operations to run) and a three-way partition frozen by `--calibrate`:

* `expect` — label-mapped operations that fired at calibration time;
* `allow` — other probed operations that fired;
* `forbid` — probed operations that stayed silent.

The tests load each mesh through the real `mesh_load` → `mesh_inspect`
MCP path with the mesh's `operations=` probe list and assert the frozen
partition. Any detector change flips an operation between fired/silent
and fails the suite — re-run `--calibrate` and review the
`selection.json` diff to accept the new behaviour.

`detect_self_intersections` is O(n²) (measured ~130 s at 5k faces vs
<0.2 s for all other operations combined), so it is probed only for
`si_otherwise_clean` meshes and small canaries — meshes that probe SI
are facet-capped at 2,500 (`TIER1_SI_MAX_FACETS`).

### Known gaps (documented, not asserted)

Label-backed detector misses (false negatives) and defect-op firings on
clean canaries (false positives) are recorded in
`selection.json → calibration.known_gaps` — currently 21 (14 FP, 7 FN),
including `detect_inconsistent_orientation` missing on all non-oriented
meshes and `detect_nonmanifold_edges` firing on every clean canary.
Contracts freeze *observed* behaviour; the gaps are the cross-validation
findings and never frozen into `expect`.

## Tier 2 — remote, opt-in (`test_thingi10k_tier2.py`)

38 heavier meshes (multi-defect combinations, extreme-statistic
outliers, 50k–300k-facet large meshes) downloaded on the fly into the
gitignored `e2e/thingi10k/tier2/` cache:

```powershell
& .\.venv\Scripts\python.exe -m pytest mesh_issue_examiner -m thingi10k_remote
```

The marker is deselected by default (`pytest.ini`). The session fixture
delegates to the fetch script (resumable, sha256-pinned, HTTP 429
backoff) and **skips cleanly when offline**. Inspections run every
operation except the SI detector (intractable at tier-2 sizes); face
counts are asserted against the dataset ground truth. Two dataset-errata
corrupt files (77942, 49911) assert a graceful `MeshLoadError` instead
of a crash or a silently-empty mesh.

## Maintaining the corpus

```powershell
& .\.venv\Scripts\python.exe scripts\fetch_thingi10k.py --metadata     # CSVs
& .\.venv\Scripts\python.exe scripts\fetch_thingi10k.py --select       # re-pick quotas
& .\.venv\Scripts\python.exe scripts\fetch_thingi10k.py --fetch-tier1  # download/verify
& .\.venv\Scripts\python.exe scripts\fetch_thingi10k.py --calibrate    # freeze contracts
& .\.venv\Scripts\python.exe scripts\fetch_thingi10k.py --verify       # sha256 audit
```

The selection is deterministic (ranking rules + quotas, no RNG), so the
same CSV snapshot always yields the same corpus. Refreshing is a
deliberate, reviewable change: commit the resulting `selection.json`
diff together with any tier-1 file additions/removals (budget: ≤50
meshes, ≤60 MB total).

## License hygiene

Per-mesh licenses from `input_summary.csv` are recorded in
`selection.json` so attribution travels with the manifest; the selection
prefers permissive licenses (CC0/public-domain/CC-BY/BSD) where quotas
allow.
