#!/usr/bin/env python
"""Thingi10K two-tier test corpus builder.

Selectively exploits the Thingi10K dataset (10,000 real-world meshes with
ground-truth defect labels) without keeping the ~70 GB dataset locally:

* Tier 1 (<= 50 small, most-challenging meshes) is fetched once and kept on
  disk under ``e2e/thingi10k/tier1`` (tracked in git together with
  ``e2e/thingi10k/selection.json``) so every clone can run the tier-1 tests
  offline.
* Tier 2 (<= 50 heavier/nastier meshes) is downloaded on the fly by the
  opt-in ``-m thingi10k_remote`` tests into the gitignored
  ``e2e/thingi10k/tier2`` cache.

Data source: the HuggingFace mirror of Thingi10K, addressed per-file with
plain HTTPS (stdlib ``urllib`` only, no ``thingi10k``/``datasets`` deps):

* ``{BASE}/metadata/geometry_data.csv``   per-file_id ground-truth labels
* ``{BASE}/metadata/input_summary.csv``   License + original Link (extension)
* ``{BASE}/raw_meshes/{file_id}.{ext}``   individual raw mesh files (LFS)

Usage:
    python scripts/fetch_thingi10k.py --metadata          # download CSVs
    python scripts/fetch_thingi10k.py --select            # print the pick
    python scripts/fetch_thingi10k.py --fetch-tier1       # download tier 1
    python scripts/fetch_thingi10k.py --fetch-tier2       # download tier 2
    python scripts/fetch_thingi10k.py --calibrate         # probe ops, freeze
                                                          # contracts + gaps
    python scripts/fetch_thingi10k.py --verify            # sha256 check

The selection is fully deterministic (no RNG): ranking rules + quotas pick
the same meshes from the same CSV snapshot.  ``selection.json`` is frozen in
git so tests never depend on upstream metadata edits.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
import urllib.request
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
T10K_DIR = REPO_ROOT / "e2e" / "thingi10k"
METADATA_DIR = T10K_DIR / "metadata"
TIER1_DIR = T10K_DIR / "tier1"
TIER2_DIR = T10K_DIR / "tier2"
SELECTION_JSON = T10K_DIR / "selection.json"

BASE = "https://huggingface.co/datasets/Thingi10K/Thingi10K/resolve/main"
GEOMETRY_CSV_URL = f"{BASE}/metadata/geometry_data.csv"
SUMMARY_CSV_URL = f"{BASE}/metadata/input_summary.csv"
RAW_URL_TEMPLATE = f"{BASE}/raw_meshes/{{file_id}}.{{ext}}"

# Dataset errata (https://github.com/Thingi10K/Thingi10K): corrupt models.
CORRUPT_FILE_IDS = frozenset([49911, 74463, 286163, 77942, 81313])

# Tier-2 robustness cases: assert the loader fails *gracefully* on these.
CORRUPT_ROBUSTNESS_IDS = [77942, 49911]

# Historical cross-validation IDs (docs/legacy/feature_gaps.md): pin as
# tier-1 regressions.  They participate in their natural category quota.
PINNED_TIER1 = {102625, 1224388, 1038433, 131372}

SUPPORTED_EXTS = {".stl", ".obj", ".off", ".ply"}

# Hard caps (plan): <= 50 meshes per tier; tier-1 budget <= 60 MB total.
TIER1_MAX_MESHES = 50
TIER2_MAX_MESHES = 50
TIER1_MAX_FACETS = 5000
TIER1_MAX_BYTES_PER_FILE = 2 * 1024 * 1024
TIER1_MAX_TOTAL_BYTES = 60 * 1024 * 1024
TIER2_MAX_FACETS = 300_000
TIER2_MAX_BYTES_PER_FILE = 40 * 1024 * 1024

# Tier-1 quotas per category (sum 41; pinned IDs join their category).
TIER1_QUOTAS = {
    "clean_canary": 7,        # all-clean but highest max_aspect_ratio
    "si_otherwise_clean": 9,  # self-intersecting, closed+manifold+oriented
    "non_edge_manifold": 6,
    "non_vertex_manifold": 4,
    "open": 5,                # num_boundary_edges > 0
    "non_oriented": 3,
    "multi_component": 3,
    "degenerate_faces": 2,
    "duplicate_faces": 2,
}

# Tier-2 quotas (sum 38 + 2 corrupt = 40).
TIER2_QUOTAS = {
    "multi_defect": 12,        # >= 2 defect labels simultaneously
    "extreme_stats": 10,       # aspect-ratio / valence / dihedral outliers
    "large_clean": 8,          # 50k-300k facets, manifold+closed
    "large_self_intersecting": 6,
}

# detect_self_intersections is O(n^2): ~2.8 s at ~2k faces but ~130 s at
# ~5k faces (measured).  Meshes whose *contract* probes the SI detector are
# therefore capped at this facet count so the tier-1 suite stays fast.
TIER1_SI_MAX_FACETS = 2500
SI_OP_ID = "topology.detect_self_intersections"

# Ground-truth (Thingi10K label) -> detector mapping per tier-1 category.
# ``--calibrate`` freezes ``expect`` from the observed intersection with
# these; label-backed ops that did NOT fire are reported as known gaps
# (potential false negatives), never silently frozen into ``expect``.
CATEGORY_EXPECT_OPS = {
    "clean_canary": [],
    "si_otherwise_clean": [SI_OP_ID],
    "non_edge_manifold": ["topology.detect_nonmanifold_edges"],
    "non_vertex_manifold": ["topology.detect_nonmanifold_vertices"],
    "open": ["topology.detect_boundary_edges"],
    "non_oriented": ["topology.detect_inconsistent_orientation"],
    "multi_component": ["topology.detect_disconnected_components"],
    # No dedicated zero-area-face detector exists; degeneracy shows up in
    # cell-quality metrics (both calibrations fired compute_cell_quality).
    "degenerate_faces": ["quality.compute_cell_quality"],
    "duplicate_faces": ["topology.detect_duplicate_faces"],
}


# ---------------------------------------------------------------------------
# Download helpers (stdlib only; honours HTTP(S)_PROXY automatically)
# ---------------------------------------------------------------------------

def _download(url: str, dest: Path, *, timeout: float = 120.0,
              attempts: int = 6) -> bytes:
    """Download *url* to *dest* atomically; returns the file bytes.

    HuggingFace rate-limits rapid successive requests (HTTP 429); those
    get long exponential backoff honouring any Retry-After header.
    """
    last_exc: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "mesh-issue-examiner/1.0"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = resp.read()
            tmp = dest.with_suffix(dest.suffix + ".part")
            tmp.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_bytes(data)
            tmp.replace(dest)
            return data
        except urllib.error.HTTPError as exc:
            last_exc = exc
            if attempt >= attempts:
                break
            if exc.code == 429:
                retry_after = exc.headers.get("Retry-After") if exc.headers else None
                wait = (float(retry_after)
                        if retry_after and retry_after.isdigit()
                        else max(15.0, 12.0 * attempt))
            else:
                wait = 2.0 * attempt
            print(f"    attempt {attempt}/{attempts} failed (HTTP {exc.code}); "
                  f"retrying in {wait:.0f}s")
            time.sleep(wait)
        except Exception as exc:  # noqa: BLE001 - report and retry
            last_exc = exc
            if attempt < attempts:
                wait = 2.0 * attempt
                print(f"    attempt {attempt}/{attempts} failed ({exc}); "
                      f"retrying in {wait:.0f}s")
                time.sleep(wait)
    raise RuntimeError(f"download failed after {attempts} attempts: {url} "
                       f"({last_exc})")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

def fetch_metadata(force: bool = False) -> None:
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    for url, name in ((GEOMETRY_CSV_URL, "geometry_data.csv"),
                      (SUMMARY_CSV_URL, "input_summary.csv")):
        dest = METADATA_DIR / name
        if dest.exists() and not force:
            print(f"  {name}: already cached ({dest.stat().st_size:,} bytes)")
            continue
        data = _download(url, dest)
        print(f"  {name}: downloaded {len(data):,} bytes")


def load_metadata() -> dict[int, dict]:
    """Merge geometry_data.csv + input_summary.csv into one entry per file_id."""
    geom_path = METADATA_DIR / "geometry_data.csv"
    sum_path = METADATA_DIR / "input_summary.csv"
    if not geom_path.exists() or not sum_path.exists():
        raise SystemExit("metadata missing - run with --metadata first")

    def _to_int(value, default=0):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return default

    def _to_float(value, default=0.0):
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    entries: dict[int, dict] = {}
    with open(geom_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            fid = _to_int(row.get("file_id"), -1)
            if fid < 0:
                continue
            entries[fid] = {
                "file_id": fid,
                "num_vertices": _to_int(row.get("num_vertices")),
                "num_faces": _to_int(row.get("num_faces")),
                "num_components": _to_int(row.get("num_connected_components")),
                "num_boundary_edges": _to_int(row.get("num_boundary_edges")),
                "num_self_intersections": _to_int(
                    row.get("num_self_intersections")),
                "num_degenerate_faces": _to_int(
                    row.get("num_geometrical_degenerated_faces")),
                "num_duplicated_faces": _to_int(
                    row.get("num_duplicated_faces")),
                "edge_manifold": _to_int(row.get("edge_manifold")) == 1,
                "vertex_manifold": _to_int(row.get("vertex_manifold")) == 1,
                "oriented": _to_int(row.get("oriented")) == 1,
                "pwn": _to_int(row.get("PWN")) == 1,
                "solid": _to_int(row.get("solid")) == 1,
                "max_aspect_ratio": _to_float(row.get("max_aspect_ratio")),
                "max_valance": _to_int(row.get("max_valance")),
                "max_dihedral_angle": _to_float(
                    row.get("max_dihedral_angle")),
                "ext": None,
                "license": "unknown",
            }

    with open(sum_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            fid = _to_int(row.get("ID"), -1)
            entry = entries.get(fid)
            if entry is None:
                continue
            link = row.get("Link") or ""
            ext = Path(link.split("?")[0]).suffix.lower()
            entry["ext"] = ext if ext else None
            entry["license"] = (row.get("License") or "unknown").strip()
    return entries


# ---------------------------------------------------------------------------
# Selection (deterministic: ranking rules + quotas, no RNG)
# ---------------------------------------------------------------------------

def _eligible(entry: dict) -> bool:
    return (
        entry["file_id"] not in CORRUPT_FILE_IDS
        and entry["ext"] in SUPPORTED_EXTS
        and entry["num_vertices"] >= 20
        and entry["num_faces"] >= 40
    )


def _labels(entry: dict) -> dict:
    return {
        "closed": entry["num_boundary_edges"] == 0,
        "self_intersecting": entry["num_self_intersections"] > 0,
        "edge_manifold": entry["edge_manifold"],
        "vertex_manifold": entry["vertex_manifold"],
        "oriented": entry["oriented"],
        "pwn": entry["pwn"],
        "solid": entry["solid"],
        "degenerate_faces": entry["num_degenerate_faces"] > 0,
        "duplicate_faces": entry["num_duplicated_faces"] > 0,
        "num_components": entry["num_components"],
        "num_boundary_edges": entry["num_boundary_edges"],
        "num_self_intersections": entry["num_self_intersections"],
        "num_vertices": entry["num_vertices"],
        "num_faces": entry["num_faces"],
    }


def _permissive_license(entry: dict) -> bool:
    lic = entry["license"].lower()
    return any(k in lic for k in (
        "public domain", "attribution - share alike", "bsd",
        "attribution\n", "creative commons - attribution",
        "creative commons - public domain",
    )) and "non-commercial" not in lic and "no derivatives" not in lic


def _take(pool: list[dict], quota: int, taken: set[int],
          key=lambda e: (-e["num_faces"], e["file_id"]),
          predicate=lambda e: True) -> list[dict]:
    """Deterministically take up to *quota* entries from *pool*."""
    picked = []
    for entry in sorted((e for e in pool if predicate(e)
                         and e["file_id"] not in taken), key=key):
        if len(picked) >= quota:
            break
        picked.append(entry)
        taken.add(entry["file_id"])
    return picked


def select_tier1(entries: dict[int, dict]) -> list[dict]:
    pool = [e for e in entries.values() if _eligible(e)]
    small = [e for e in pool
             if e["num_faces"] <= TIER1_MAX_FACETS
             and _labels(e)["num_vertices"] <= 2500]

    taken: set[int] = set()
    selection: list[dict] = []

    def add(entry: dict, category: str) -> None:
        selection.append((entry, category))  # type: ignore[arg-type]

    # Pinned historical regressions first (they claim their category slot).
    pinned_used = set()
    for fid in sorted(PINNED_TIER1):
        entry = entries.get(fid)
        if entry is None or fid not in {e["file_id"] for e in pool}:
            print(f"  WARNING: pinned file_id {fid} not eligible; skipping")
            continue
        lab = _labels(entry)
        if lab["self_intersecting"] and lab["closed"] and lab["edge_manifold"]:
            cat = "si_otherwise_clean"
        elif not lab["edge_manifold"]:
            cat = "non_edge_manifold"
        elif not lab["closed"]:
            cat = "open"
        else:
            cat = "clean_canary"
        pinned_used.add(cat)
        taken.add(fid)
        add(entry, cat)

    # Category definitions on the label snapshot.
    def is_clean(e):
        lab = _labels(e)
        return (lab["closed"] and lab["edge_manifold"]
                and lab["vertex_manifold"] and lab["oriented"] and lab["pwn"]
                and not lab["self_intersecting"]
                and not lab["degenerate_faces"]
                and not lab["duplicate_faces"]
                and lab["num_components"] == 1)

    quotas = dict(TIER1_QUOTAS)
    for cat in pinned_used:
        quotas[cat] = max(0, quotas[cat] - 1)

    # Clean canaries: hardest clean meshes = highest max aspect ratio.
    for e in _take(small, quotas["clean_canary"], taken,
                   key=lambda e: (-e["max_aspect_ratio"], e["file_id"]),
                   predicate=is_clean):
        add(e, "clean_canary")

    # Self-intersecting but otherwise pristine: hardest precision case.
    # Capped at TIER1_SI_MAX_FACETS because the SI probe is O(n^2).
    for e in _take(small, quotas["si_otherwise_clean"], taken,
                   key=lambda e: (-e["num_self_intersections"], e["file_id"]),
                   predicate=lambda e: (
                       _labels(e)["self_intersecting"]
                       and _labels(e)["closed"]
                       and _labels(e)["edge_manifold"]
                       and _labels(e)["oriented"]
                       and e["num_faces"] <= TIER1_SI_MAX_FACETS)):
        add(e, "si_otherwise_clean")

    for e in _take(small, quotas["non_edge_manifold"], taken,
                   predicate=lambda e: not _labels(e)["edge_manifold"]):
        add(e, "non_edge_manifold")

    for e in _take(small, quotas["non_vertex_manifold"], taken,
                   predicate=lambda e: (
                       not _labels(e)["vertex_manifold"]
                       and _labels(e)["edge_manifold"])):
        add(e, "non_vertex_manifold")

    for e in _take(small, quotas["open"], taken,
                   key=lambda e: (-e["num_boundary_edges"], e["file_id"]),
                   predicate=lambda e: not _labels(e)["closed"]):
        add(e, "open")

    for e in _take(small, quotas["non_oriented"], taken,
                   predicate=lambda e: not _labels(e)["oriented"]):
        add(e, "non_oriented")

    for e in _take(small, quotas["multi_component"], taken,
                   key=lambda e: (-e["num_components"], e["file_id"]),
                   predicate=lambda e: _labels(e)["num_components"] > 1):
        add(e, "multi_component")

    for e in _take(small, quotas["degenerate_faces"], taken,
                   key=lambda e: (-e["num_degenerate_faces"], e["file_id"]),
                   predicate=lambda e: _labels(e)["degenerate_faces"]):
        add(e, "degenerate_faces")

    for e in _take(small, quotas["duplicate_faces"], taken,
                   key=lambda e: (-e["num_duplicated_faces"], e["file_id"]),
                   predicate=lambda e: _labels(e)["duplicate_faces"]):
        add(e, "duplicate_faces")

    assert len(selection) <= TIER1_MAX_MESHES, len(selection)
    return selection


def _defect_count(e: dict) -> int:
    lab = _labels(e)
    return sum((
        not lab["closed"], lab["self_intersecting"],
        not lab["edge_manifold"], not lab["vertex_manifold"],
        not lab["oriented"], lab["degenerate_faces"],
        lab["duplicate_faces"], lab["num_components"] > 1,
    ))


def select_tier2(entries: dict[int, dict],
                 tier1_ids: set[int]) -> list[dict]:
    pool = [e for e in entries.values()
            if _eligible(e) and e["file_id"] not in tier1_ids]
    for e in pool:
        if e["num_faces"] > TIER2_MAX_FACETS:
            e["approx_bytes"] = TIER2_MAX_BYTES_PER_FILE + 1
        else:
            # STL dominates the dataset: 50 bytes per triangle + header.
            e["approx_bytes"] = 84 + 50 * e["num_faces"]

    heavy = [e for e in pool if e["approx_bytes"] <= TIER2_MAX_BYTES_PER_FILE]
    taken: set[int] = set()
    selection: list[dict] = []

    def add(entry: dict, category: str) -> None:
        selection.append((entry, category))  # type: ignore[arg-type]

    # Simultaneous defects, worst combos first.
    for e in _take(heavy, TIER2_QUOTAS["multi_defect"], taken,
                   key=lambda e: (-_defect_count(e),
                                  -e["num_self_intersections"], e["file_id"]),
                   predicate=lambda e: _defect_count(e) >= 2):
        add(e, "multi_defect")

    # Statistical outliers (exclude meshes already taken).
    for e in _take(heavy, 4, taken,
                   key=lambda e: (-e["max_aspect_ratio"], e["file_id"])):
        add(e, "extreme_stats")
    for e in _take(heavy, 3, taken,
                   key=lambda e: (-e["max_valance"], e["file_id"])):
        add(e, "extreme_stats")
    for e in _take(heavy, 3, taken,
                   key=lambda e: (-abs(e["max_dihedral_angle"]), e["file_id"])):
        add(e, "extreme_stats")

    # Large clean meshes: performance stress, manifold+closed preferred.
    for e in _take(heavy, TIER2_QUOTAS["large_clean"], taken,
                   key=lambda e: (-e["num_faces"], e["file_id"]),
                   predicate=lambda e: (
                       50_000 <= e["num_faces"] <= TIER2_MAX_FACETS
                       and e["edge_manifold"] and e["num_boundary_edges"] == 0
                       and not e["num_self_intersections"])):
        add(e, "large_clean")

    # Large self-intersecting meshes: SI detector stress.
    for e in _take(heavy, TIER2_QUOTAS["large_self_intersecting"], taken,
                   key=lambda e: (-e["num_faces"], e["file_id"]),
                   predicate=lambda e: (
                       20_000 <= e["num_faces"] <= 100_000
                       and e["num_self_intersections"] > 0)):
        add(e, "large_self_intersecting")

    # Corrupt-file robustness cases (graceful loader failure expected).
    # The metadata CSV excludes corrupt models (9,997 of 10,000 rows), so
    # fall back to a synthetic entry: both errata files are STL and their
    # labels are irrelevant (the test asserts graceful load failure).
    for fid in CORRUPT_ROBUSTNESS_IDS:
        entry = entries.get(fid)
        if entry is None:
            entry = {
                "file_id": fid, "num_vertices": 0, "num_faces": 0,
                "num_components": 0, "num_boundary_edges": 0,
                "num_self_intersections": 0, "num_degenerate_faces": 0,
                "num_duplicated_faces": 0,
                "edge_manifold": False, "vertex_manifold": False,
                "oriented": False, "pwn": False, "solid": False,
                "max_aspect_ratio": 0.0, "max_valance": 0,
                "max_dihedral_angle": 0.0,
                "ext": ".stl", "license": "unknown",
            }
        else:
            entry = dict(entry, ext=entry["ext"] or ".stl")
        taken.add(fid)
        add(entry, "corrupt_robustness")

    assert len(selection) <= TIER2_MAX_MESHES, len(selection)
    return selection


# ---------------------------------------------------------------------------
# selection.json
# ---------------------------------------------------------------------------

def _entry_json(entry: dict, category: str, tier_dir: Path) -> dict:
    fid = entry["file_id"]
    ext = entry["ext"]
    return {
        "file_id": fid,
        "category": category,
        "ext": ext,
        "filename": f"{fid}{ext}",
        "path": str(tier_dir.relative_to(REPO_ROOT) / f"{fid}{ext}"),
        "url": RAW_URL_TEMPLATE.format(file_id=fid, ext=ext.lstrip(".")),
        "license": entry["license"],
        "labels": _labels(entry),
        "stats": {
            "num_vertices": entry["num_vertices"],
            "num_faces": entry["num_faces"],
            "max_aspect_ratio": entry["max_aspect_ratio"],
        },
        "sha256": None,
        "size_bytes": None,
    }


def write_selection(entries: dict[int, dict]) -> dict:
    tier1 = select_tier1(entries)
    tier2 = select_tier2(entries, {e["file_id"] for e, _ in tier1})
    doc = {
        "dataset": "Thingi10K via HuggingFace mirror",
        "base_url": BASE,
        "generated": date.today().isoformat(),
        "tier1": [_entry_json(e, cat, TIER1_DIR) for e, cat in tier1],
        "tier2": [_entry_json(e, cat, TIER2_DIR) for e, cat in tier2],
    }
    T10K_DIR.mkdir(parents=True, exist_ok=True)
    SELECTION_JSON.write_text(
        json.dumps(doc, indent=1, sort_keys=False) + "\n", encoding="utf-8")
    return doc


def load_selection() -> dict:
    if not SELECTION_JSON.exists():
        raise SystemExit("selection.json missing - run --select first")
    return json.loads(SELECTION_JSON.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------

def action_select() -> None:
    entries = load_metadata()
    doc = write_selection(entries)
    print(f"Selection written to {SELECTION_JSON}")
    for tier in ("tier1", "tier2"):
        cats: dict[str, int] = {}
        for item in doc[tier]:
            cats[item["category"]] = cats.get(item["category"], 0) + 1
        total = len(doc[tier])
        print(f"\n{tier}: {total} meshes")
        for cat, n in sorted(cats.items()):
            print(f"  {cat:24s} {n}")
    verts = [item["stats"]["num_faces"] for item in doc["tier1"]]
    if verts:
        print(f"\ntier1 faces: min={min(verts)} max={max(verts)}")


def _fetch_tier(doc: dict, tier: str, force: bool) -> None:
    tier_dir = TIER1_DIR if tier == "tier1" else TIER2_DIR
    tier_dir.mkdir(parents=True, exist_ok=True)
    total_bytes = 0
    for item in doc[tier]:
        dest = REPO_ROOT / item["path"]
        if dest.exists() and not force:
            data_size = dest.stat().st_size
            digest = None if item["sha256"] is None else sha256_file(dest)
            if item["sha256"] in (None, digest):
                print(f"  {item['filename']}: cached "
                      f"({data_size:,} bytes)")
                item["size_bytes"] = data_size
                if item["sha256"] is None:
                    item["sha256"] = sha256_file(dest)
                total_bytes += data_size
                continue
            print(f"  {item['filename']}: sha mismatch, re-fetching")
        data = _download(item["url"], dest)
        if tier == "tier1" and len(data) > TIER1_MAX_BYTES_PER_FILE:
            raise SystemExit(
                f"{item['filename']}: {len(data):,} bytes exceeds the "
                f"tier-1 per-file cap ({TIER1_MAX_BYTES_PER_FILE:,}); "
                "tighten the selection and re-run --select")
        item["sha256"] = hashlib.sha256(data).hexdigest()
        item["size_bytes"] = len(data)
        total_bytes += len(data)
        print(f"  {item['filename']}: {len(data):,} bytes "
              f"[{item['category']}]")
        time.sleep(1.5)  # pace requests; HF returns 429 on bursts
    SELECTION_JSON.write_text(
        json.dumps(doc, indent=1, sort_keys=False) + "\n", encoding="utf-8")
    n = len(doc[tier])
    print(f"\n{tier}: {n} meshes, {total_bytes / 1e6:.1f} MB total")
    if tier == "tier1":
        if n > TIER1_MAX_MESHES:
            raise SystemExit(f"tier1 exceeds {TIER1_MAX_MESHES} meshes")
        if total_bytes > TIER1_MAX_TOTAL_BYTES:
            raise SystemExit(
                f"tier1 budget exceeded: {total_bytes / 1e6:.1f} MB > 60 MB")


def action_verify() -> None:
    doc = load_selection()
    missing = mismatched = 0
    for tier in ("tier1", "tier2"):
        for item in doc[tier]:
            path = REPO_ROOT / item["path"]
            if not path.exists():
                print(f"  MISSING  {item['path']}")
                missing += 1
                continue
            if item["sha256"] and sha256_file(path) != item["sha256"]:
                print(f"  MISMATCH {item['path']}")
                mismatched += 1
    if missing == 0 and mismatched == 0:
        total = sum(len(doc[t]) for t in ("tier1", "tier2"))
        print(f"OK: all {total} cached files match selection.json")
    else:
        raise SystemExit(f"verify failed: {missing} missing, "
                         f"{mismatched} mismatched")


def action_calibrate() -> None:
    """Run targeted probes over tier 1 and freeze detector contracts.

    For every tier-1 mesh each registry operation is executed individually
    (a "probe") and timed.  The three-way partition of the probe set is
    frozen into ``selection.json``:

    * ``expect`` - label-mapped ops (CATEGORY_EXPECT_OPS) that fired;
    * ``allow``  - other probed ops that fired;
    * ``forbid`` - probed ops that did not fire.

    Label-backed ops that did not fire, and defect ops that fired on clean
    canaries, are reported as *known gaps* (cross-validation findings) but
    never frozen into ``expect``: the frozen contract reflects observed
    behaviour so the tier-1 suite pins the current detector output.
    """
    sys.path.insert(0, str(REPO_ROOT))

    from mesh_issue_examiner.operations.operation_registry import registry
    from mesh_issue_examiner.mcp.server import (  # noqa: E402
        MeshIssueExaminerServer)
    from mesh_issue_examiner.utils.mesh_loader import (  # noqa: E402
        load_mesh_from_file)

    doc = load_selection()
    server = MeshIssueExaminerServer(auto_register_operations=True)
    all_ops = sorted(op.operation_id for op in registry.list_all())
    defect_ops = {op for ops in CATEGORY_EXPECT_OPS.values() for op in ops}

    gaps: list[str] = []
    projected = 0.0
    t0 = time.time()
    for item in doc["tier1"]:
        path = REPO_ROOT / item["path"]
        if not path.exists():
            raise SystemExit(f"{item['filename']}: missing (run --fetch-tier1)")
        probes = [op for op in all_ops if op != SI_OP_ID]
        si_probe = (
            item["category"] == "si_otherwise_clean"
            or (item["category"] == "clean_canary"
                and item["stats"]["num_faces"] <= TIER1_SI_MAX_FACETS))
        if si_probe:
            probes.append(SI_OP_ID)
        try:
            mesh = load_mesh_from_file(path)
        except Exception as exc:  # noqa: BLE001
            raise SystemExit(
                f"{item['filename']} [{item['category']}]: loader failed on "
                f"a verified file: {exc}") from exc
        fired: set[str] = set()
        timings: dict[str, float] = {}
        for op_id in sorted(probes):
            workflow = registry.create_workflow([op_id], config=server.config)
            w0 = time.perf_counter()
            result = workflow(mesh)
            timings[op_id] = round(time.perf_counter() - w0, 3)
            if result.issues:
                fired.add(op_id)
        projected += sum(timings.values())

        expect_ops = set(CATEGORY_EXPECT_OPS.get(item["category"], []))
        expect = sorted(expect_ops & fired)
        allow = sorted(fired - expect_ops)
        forbid = sorted(set(probes) - fired)
        gaps.extend(
            f"fn {item['filename']} [{item['category']}]: {op} did not fire "
            "(label-backed)"
            for op in sorted(expect_ops - fired))
        if item["category"] == "clean_canary":
            gaps.extend(
                f"fp {item['filename']} [clean_canary]: {op} fired on a "
                "fully-clean mesh"
                for op in sorted(fired & defect_ops))
        item["contract"] = {
            "ops": sorted(probes),
            "expect": expect,
            "allow": allow,
            "forbid": forbid,
            "timings_s": timings,
        }
        print(f"  {item['filename']} [{item['category']}] "
              f"{sum(timings.values()):.1f}s expect={len(expect)} "
              f"allow={len(allow)} forbid={len(forbid)}"
              + ("  [SI probed]" if si_probe else ""))

    doc["calibration"] = {
        "date": date.today().isoformat(),
        "operations": all_ops,
        "projected_suite_seconds": round(projected, 1),
        "known_gaps": gaps,
    }
    SELECTION_JSON.write_text(
        json.dumps(doc, indent=1, sort_keys=False) + "\n", encoding="utf-8")
    print(f"\ncalibration wall time: {time.time() - t0:.1f}s; "
          f"projected tier-1 suite runtime: {projected:.1f}s")
    if projected > 180.0:
        print("  WARNING: projected suite runtime exceeds the 3-minute budget")
    print(f"\nknown gaps ({len(gaps)}):")
    for gap in gaps:
        print(f"  {gap}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--metadata", action="store_true",
                        help="download/refresh metadata CSVs")
    parser.add_argument("--select", action="store_true",
                        help="run selection and write selection.json")
    parser.add_argument("--fetch-tier1", action="store_true",
                        help="download tier-1 meshes")
    parser.add_argument("--fetch-tier2", action="store_true",
                        help="download tier-2 meshes")
    parser.add_argument("--calibrate", action="store_true",
                        help="probe tier-1 meshes per op and freeze "
                             "expect/allow/forbid contracts")
    parser.add_argument("--verify", action="store_true",
                        help="verify cached files against selection.json")
    parser.add_argument("--force", action="store_true",
                        help="force re-download / re-selection")
    args = parser.parse_args()

    if not any(vars(args).values()):
        parser.print_help()
        return 1

    if args.metadata:
        print("[metadata]")
        fetch_metadata(force=args.force)
    if args.select:
        print("[select]")
        action_select()
    if args.fetch_tier1 or args.fetch_tier2:
        doc = load_selection()
        if args.fetch_tier1:
            print("[fetch-tier1]")
            _fetch_tier(doc, "tier1", force=args.force)
        if args.fetch_tier2:
            print("[fetch-tier2]")
            _fetch_tier(doc, "tier2", force=args.force)
    if args.calibrate:
        print("[calibrate]")
        action_calibrate()
    if args.verify:
        print("[verify]")
        action_verify()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
