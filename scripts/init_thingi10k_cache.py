#!/usr/bin/env python
"""
Fully offline cache preparation for Thingi10K.

Uses locally downloaded tarballs and metadata CSVs to build the dataset
directly, bypassing HuggingFace download entirely.  After running this
script, thingi10k.init() works without network.

Usage:
    python scripts/init_thingi10k_cache.py \
        --metadata-dir ~/Downloads/thingi10k_metadata \
        --npz-tarball ~/Downloads/Thingi10K_npz.tar.gz \
        --raw-tarball ~/Downloads/Thingi10K.tar.gz

    # Only npz variant (what the tests use):
    python scripts/init_thingi10k_cache.py \
        --metadata-dir ~/Downloads/thingi10k_metadata \
        --npz-tarball ~/Downloads/Thingi10K_npz.tar.gz \
        --skip-raw

Required metadata CSVs in --metadata-dir:
    - geometry_data.csv
    - contextual_data.csv
    - input_summary.csv
    - tag_data.csv

Download everything from:
    https://huggingface.co/datasets/Thingi10K/Thingi10K/tree/main

After this, run the e2e tests (no network needed):
    /home/dabajabaza/anaconda3/envs/PEEC/bin/python -m pytest \
        mesh_issue_examiner/tests/test_thingi10k_e2e.py -v
"""

import argparse
import os
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants from thingi10k source (_builder.py)
# ---------------------------------------------------------------------------
REPO_URL = "https://huggingface.co/datasets/Thingi10K/Thingi10K/resolve/main"
METADATA_URL = f"{REPO_URL}/metadata"
NPZ_URL = f"{REPO_URL}/Thingi10K_npz.tar.gz"
RAW_URL = f"{REPO_URL}/Thingi10K.tar.gz"

CORRUPT_FILE_IDS = frozenset([49911, 74463, 286163, 77942])

# ---------------------------------------------------------------------------
# HF datasets cache layout
# ---------------------------------------------------------------------------
HF_CACHE = Path.home() / ".cache" / "huggingface"
DOWNLOADS_DIR = HF_CACHE / "datasets" / "downloads"
EXTRACT_DIR = DOWNLOADS_DIR / "extracted"
BUILDER_CACHE = HF_CACHE / "datasets" / "_builder"


def _hash_url(url):
    from datasets.utils.file_utils import hash_url_to_filename
    return hash_url_to_filename(url)


def _extract_dir_for(download_path):
    from datasets.utils.file_utils import hash_url_to_filename
    return EXTRACT_DIR / hash_url_to_filename(str(Path(download_path).resolve()))


# ---------------------------------------------------------------------------
# Cache cleanup
# ---------------------------------------------------------------------------

def clean_stale():
    """Remove stale locks and incomplete files from previous failed runs."""
    cleaned = 0
    for pattern in ["**/*.lock", "**/*.incomplete"]:
        for f in (HF_CACHE / "datasets").glob(pattern):
            print(f"  Removing stale: {f}")
            if f.is_dir():
                shutil.rmtree(f)
            else:
                f.unlink()
            cleaned += 1
    hub_blobs = HF_CACHE / "hub" / "datasets--Thingi10K--Thingi10K" / "blobs"
    if hub_blobs.exists():
        for f in hub_blobs.glob("*.incomplete"):
            print(f"  Removing stale blob: {f}")
            f.unlink()
            cleaned += 1
    print(f"  Cleaned {cleaned} stale artifacts")


# ---------------------------------------------------------------------------
# Step 1: Place metadata CSVs into HF downloads cache
# ---------------------------------------------------------------------------

def place_metadata(metadata_dir, variant):
    """Place local CSVs at the exact paths dl_manager.download() expects."""
    csv_map = {
        "contextual_data": f"{METADATA_URL}/contextual_data.csv",
        "input_summary": f"{METADATA_URL}/input_summary.csv",
        "tag_data": f"{METADATA_URL}/tag_data.csv",
    }
    if variant in ("npz", "raw"):
        csv_map["geometry_data"] = f"{METADATA_URL}/geometry_data.csv"
    else:
        csv_map["geometry_data"] = f"{METADATA_URL}/tetwild_geometry_data.csv"

    for name, url in csv_map.items():
        src = Path(metadata_dir) / f"{name}.csv"
        if not src.exists():
            print(f"  WARNING: {src} not found, will need network for {name}")
            continue

        dst = DOWNLOADS_DIR / _hash_url(url)
        if dst.exists():
            print(f"  {name}: already cached")
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(src, dst)
            print(f"  {name}: hardlinked → {dst.name}")
        except OSError:
            shutil.copy2(src, dst)
            print(f"  {name}: copied → {dst.name}")


# ---------------------------------------------------------------------------
# Step 2: Place + extract tarball into HF downloads/extracted cache
# ---------------------------------------------------------------------------

def place_and_extract_tarball(tarball_path, url, label, expected_subdir):
    """Hardlink tarball into downloads cache, extract to expected directory."""
    tarball = Path(tarball_path).resolve()
    download_hash = _hash_url(url)
    download_path = DOWNLOADS_DIR / download_hash
    extract_dir = _extract_dir_for(download_path)

    print(f"\n[{label}]")
    print(f"  Tarball:      {tarball.name} ({tarball.stat().st_size / 1e9:.2f} GB)")
    print(f"  Download to:  {download_path.name}")
    print(f"  Extract to:   {extract_dir.name}")

    # Place tarball in downloads cache
    if download_path.exists():
        print(f"  Already in downloads cache")
    else:
        download_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(tarball, download_path)
            print(f"  Hardlinked into downloads cache")
        except OSError:
            shutil.copy2(tarball, download_path)
            print(f"  Copied into downloads cache")

    # Extract
    marker = extract_dir / ".thingi10k_extracted"
    if marker.exists():
        print(f"  Already extracted")
    else:
        extract_dir.mkdir(parents=True, exist_ok=True)
        print(f"  Extracting...")
        t0 = time.time()
        with tarfile.open(download_path, "r:gz") as tf:
            tf.extractall(extract_dir)
        marker.touch()
        print(f"  Done in {time.time() - t0:.1f}s")

    # Verify
    check = extract_dir / expected_subdir
    if check.exists():
        count = sum(1 for _ in check.iterdir())
        print(f"  Verified: {expected_subdir} ({count} files)")
    else:
        print(f"  WARNING: {check} not found!")

    return extract_dir


# ---------------------------------------------------------------------------
# Step 3: Build Arrow dataset offline (the real init replacement)
# ---------------------------------------------------------------------------

def build_dataset_offline(metadata_dir, npz_extract_dir, variant):
    """Build the Arrow dataset from local files and inject into thingi10k.

    This replicates what Thingi10KBuilder._generate_examples does, but
    using only local files. Sets thingi10k._dataset so the library works
    without calling init().
    """
    import polars as pl
    import datasets as ds

    print(f"\n[Build {variant}] Building Arrow dataset from local files...")
    t0 = time.time()

    # Load CSVs (same as _load_and_process_csv_files)
    geom = pl.read_csv(
        Path(metadata_dir) / "geometry_data.csv",
        schema_overrides=_geometry_schema(),
        ignore_errors=True,
    )
    ctx = pl.read_csv(
        Path(metadata_dir) / "contextual_data.csv",
        schema_overrides={
            "Thing ID": pl.Int32, "Date": pl.Datetime,
            "Category": pl.String, "Sub-category": pl.String,
            "Name": pl.String, "Author": pl.String, "License": pl.String,
        },
        ignore_errors=True,
    )
    summary = pl.read_csv(
        Path(metadata_dir) / "input_summary.csv",
        schema_overrides={"ID": pl.Int32, "Thing ID": pl.Int32},
        ignore_errors=True,
    )
    tags = pl.read_csv(
        Path(metadata_dir) / "tag_data.csv",
        schema_overrides={"Thing ID": pl.Int32, "Tag": pl.String},
        ignore_errors=True,
    )

    # Add self_intersecting column if missing
    if "self_intersecting" not in geom.columns:
        geom = geom.with_columns(
            (pl.col("num_self_intersections").fill_null(0) > 0)
            .cast(pl.Boolean)
            .alias("self_intersecting")
        )
    else:
        geom = geom.with_columns(
            pl.col("self_intersecting").fill_null(False)
        )

    # Join dataframes (same as _prepare_dataframe)
    df = geom
    df = df.join(summary, left_on="file_id", right_on="ID", how="left")
    df = df.join(ctx, on="Thing ID", how="left")
    tags_agg = tags.group_by("Thing ID").agg(pl.col("Tag").alias("Tags"))
    df = df.join(tags_agg, on="Thing ID", how="left")
    for col, default in [
        ("License", "unknown"), ("Author", "unknown"),
        ("Category", "unknown"), ("Sub-category", "unknown"),
        ("Name", "unknown"),
    ]:
        if col in df.columns:
            df = df.with_columns(pl.col(col).fill_null(default))
    if "Tags" in df.columns:
        df = df.with_columns(pl.col("Tags").fill_null(pl.lit([])))

    # Build file paths (same as _prepare_dataset_files)
    file_ids = df["file_id"].to_list()
    if variant == "npz":
        base = Path(npz_extract_dir) / "npz"
        file_paths = [base / f"{fid}.npz" for fid in file_ids
                      if fid not in CORRUPT_FILE_IDS]
    else:
        raw_data = summary.select(["ID", "Link"])
        raw_rows = {row[0]: row[1] for row in raw_data.iter_rows()}
        file_paths = [
            npz_extract_dir / "Thingi10K" / "raw_meshes"
            / f"{fid}.{raw_rows.get(fid, '').split('.')[-1].lower()}"
            for fid in file_ids
            if fid not in CORRUPT_FILE_IDS
        ]

    # Build metadata lookup
    meta = {row["file_id"]: row for row in df.iter_rows(named=True)}

    # Generate entries (same as _generate_examples)
    import datetime
    entries = []
    for fpath in file_paths:
        fpath = Path(fpath)
        if not fpath.exists() or fpath.stat().st_size == 0:
            continue
        fid = int(fpath.stem)
        if fid not in meta:
            continue
        m = meta[fid]
        entries.append({
            "file_id": int(fid),
            "thing_id": int(m.get("Thing ID", 0)),
            "file_path": str(fpath),
            "author": str(m.get("Author", "unknown")),
            "date": m.get("Date", datetime.datetime(1900, 1, 1)),
            "license": str(m.get("License", "unknown")),
            "category": str(m.get("Category", "unknown")),
            "subcategory": str(m.get("Sub-category", "unknown")),
            "name": str(m.get("Name", "unknown")),
            "tags": list(m.get("Tags", [])),
            "num_vertices": int(m["num_vertices"]),
            "num_facets": int(m["num_faces"]),
            "num_components": int(m["num_connected_components"]),
            "num_boundary_edges": int(m["num_boundary_edges"]),
            "closed": int(m["num_boundary_edges"]) == 0,
            "self_intersecting": bool(m.get("self_intersecting", False)),
            "vertex_manifold": int(m.get("vertex_manifold") or 0) == 1,
            "edge_manifold": int(m.get("edge_manifold") or 0) == 1,
            "oriented": int(m.get("oriented") or 0) == 1,
            "PWN": int(m.get("PWN") or 0) == 1,
            "solid": int(m.get("solid") or 0) == 1,
            "euler": int(m["euler_characteristic"]),
        })

    print(f"  {len(entries)} mesh entries built in {time.time() - t0:.1f}s")

    # Create HuggingFace Dataset
    features = ds.Features({
        "file_id": ds.Value("int32"),
        "thing_id": ds.Value("int32"),
        "file_path": ds.Value("string"),
        "author": ds.Value("string"),
        "date": ds.Value("date64"),
        "license": ds.Value("string"),
        "category": ds.Value("string"),
        "subcategory": ds.Value("string"),
        "name": ds.Value("string"),
        "tags": ds.Sequence(ds.Value("string")),
        "num_vertices": ds.Value("int32"),
        "num_facets": ds.Value("int32"),
        "num_components": ds.Value("int32"),
        "num_boundary_edges": ds.Value("int32"),
        "closed": ds.Value("bool"),
        "self_intersecting": ds.Value("bool"),
        "vertex_manifold": ds.Value("bool"),
        "edge_manifold": ds.Value("bool"),
        "oriented": ds.Value("bool"),
        "PWN": ds.Value("bool"),
        "solid": ds.Value("bool"),
        "euler": ds.Value("int32"),
    })
    dataset = ds.Dataset.from_list(entries, features=features)
    dataset_dict = ds.DatasetDict({"train": dataset})

    # Inject into thingi10k._utils module (where _dataset actually lives)
    import thingi10k._utils
    thingi10k._utils._dataset = dataset_dict
    print(f"  Injected {len(entries)} entries into thingi10k._dataset")

    # Also save to the builder cache so future thingi10k.init() finds it
    cache_path = BUILDER_CACHE / variant / "1.0.0"
    if not cache_path.exists():
        cache_path.mkdir(parents=True, exist_ok=True)
        dataset_dict.save_to_disk(str(cache_path))
        print(f"  Saved Arrow dataset to {cache_path}")
    else:
        print(f"  Builder cache already exists: {cache_path}")

    return dataset_dict


def _geometry_schema():
    import polars as pl
    return {
        "file_id": pl.Int32,
        "num_vertices": pl.Int32,
        "num_faces": pl.Int32,
        "num_geometrical_degenerated_faces": pl.Int32,
        "num_combinatorial_degenerated_faces": pl.Int32,
        "num_connected_components": pl.Int32,
        "num_boundary_edges": pl.Int32,
        "num_duplicated_faces": pl.Int32,
        "euler_characteristic": pl.Int32,
        "num_self_intersections": pl.Int32,
        "num_coplanar_intersecting_faces": pl.Int32,
        "vertex_manifold": pl.Int32,
        "edge_manifold": pl.Int32,
        "oriented": pl.Int32,
        "total_area": pl.Float64,
        "min_area": pl.Float64,
        "p25_area": pl.Float64,
        "median_area": pl.Float64,
        "p75_area": pl.Float64,
        "p90_area": pl.Float64,
        "p95_area": pl.Float64,
        "max_area": pl.Float64,
        "min_valance": pl.Int32,
        "p25_valance": pl.Int32,
        "median_valance": pl.Int32,
        "p75_valance": pl.Int32,
        "p90_valance": pl.Int32,
        "p95_valance": pl.Int32,
        "max_valance": pl.Int32,
        "min_dihedral_angle": pl.Float64,
        "p25_dihedral_angle": pl.Float64,
        "median_dihedral_angle": pl.Float64,
        "p75_dihedral_angle": pl.Float64,
        "p90_dihedral_angle": pl.Float64,
        "p95_dihedral_angle": pl.Float64,
        "max_dihedral_angle": pl.Float64,
        "min_aspect_ratio": pl.Float64,
        "p25_aspect_ratio": pl.Float64,
        "median_aspect_ratio": pl.Float64,
        "p75_aspect_ratio": pl.Float64,
        "p90_aspect_ratio": pl.Float64,
        "p95_aspect_ratio": pl.Float64,
        "max_aspect_ratio": pl.Float64,
        "PWN": pl.Int32,
        "solid": pl.Int32,
        "ave_area": pl.Float64,
        "ave_valance": pl.Float64,
        "ave_dihedral_angle": pl.Float64,
        "ave_aspect_ratio": pl.Float64,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Fully offline Thingi10K cache preparation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--metadata-dir", type=str, required=True,
                        help="Directory with geometry_data.csv, contextual_data.csv, "
                             "input_summary.csv, tag_data.csv")
    parser.add_argument("--npz-tarball", type=str, default=None,
                        help="Path to Thingi10K_npz.tar.gz")
    parser.add_argument("--raw-tarball", type=str, default=None,
                        help="Path to Thingi10K.tar.gz")
    parser.add_argument("--skip-npz", action="store_true",
                        help="Skip npz variant")
    parser.add_argument("--skip-raw", action="store_true",
                        help="Skip raw variant")
    parser.add_argument("--no-prefetch", action="store_true",
                        help="Skip pre-fetching sample meshes")
    args = parser.parse_args()

    metadata_dir = Path(args.metadata_dir).resolve()
    assert metadata_dir.is_dir(), f"--metadata-dir not found: {metadata_dir}"
    for csv in ["geometry_data.csv", "contextual_data.csv",
                "input_summary.csv", "tag_data.csv"]:
        assert (metadata_dir / csv).exists(), f"Missing: {metadata_dir / csv}"

    print(f"Metadata: {metadata_dir}")
    print(f"Cache:    {HF_CACHE}")

    # ------------------------------------------------------------------
    # 0. Clean
    # ------------------------------------------------------------------
    print("\n[0] Cleaning stale cache artifacts...")
    clean_stale()

    # ------------------------------------------------------------------
    # 1. Place metadata CSVs into HF downloads cache
    #     (so thingi10k.init() finds them if called later)
    # ------------------------------------------------------------------
    print("\n[1] Placing metadata CSVs into downloads cache...")
    if not args.skip_npz:
        place_metadata(metadata_dir, "npz")
    if not args.skip_raw:
        place_metadata(metadata_dir, "raw")

    # ------------------------------------------------------------------
    # 2. Place + extract tarballs into HF cache
    # ------------------------------------------------------------------
    npz_extract = None
    raw_extract = None

    if not args.skip_npz and args.npz_tarball:
        npz_extract = place_and_extract_tarball(
            args.npz_tarball, NPZ_URL, "NPZ tarball", "npz",
        )

    if not args.skip_raw and args.raw_tarball:
        raw_extract = place_and_extract_tarball(
            args.raw_tarball, RAW_URL, "Raw tarball",
            "Thingi10K/raw_meshes",
        )

    # ------------------------------------------------------------------
    # 3. Build Arrow dataset from local files (bypasses HF download)
    # ------------------------------------------------------------------
    import thingi10k

    if not args.skip_npz and npz_extract:
        build_dataset_offline(metadata_dir, npz_extract, "npz")

        # Verify — _dataset is already set by build_dataset_offline
        ds = thingi10k.dataset(num_vertices=(10, 500), closed=True, manifold=True)
        print(f"  Verify: small closed manifold = {len(ds)} meshes")
        assert len(ds) > 0, "Dataset is empty!"

    if not args.skip_raw and raw_extract:
        build_dataset_offline(metadata_dir, raw_extract, "raw")

        ds = thingi10k.dataset(num_vertices=(10, 500), closed=True, manifold=True)
        print(f"  Verify: small closed manifold (raw) = {len(ds)} meshes")

    # ------------------------------------------------------------------
    # 4. Pre-fetch sample meshes
    # ------------------------------------------------------------------
    if not args.no_prefetch and not args.skip_npz:
        print("\n[4] Pre-fetching sample meshes...")
        queries = {
            "manifold=True, closed": dict(
                manifold=True, closed=True,
                num_vertices=(20, 500), num_facets=(None, 1000)),
            "manifold=False": dict(
                manifold=False,
                num_vertices=(20, 1000), num_facets=(None, 2000)),
            "self_intersecting=True": dict(
                self_intersecting=True,
                num_vertices=(20, 2000), num_facets=(None, 4000)),
            "self_intersecting=False": dict(
                self_intersecting=False, closed=True, manifold=True,
                num_vertices=(20, 2000), num_facets=(None, 4000)),
            "quality sanity": dict(
                closed=True, manifold=True,
                num_vertices=(50, 5000), num_facets=(None, 10000)),
            "open meshes": dict(
                closed=False,
                num_vertices=(20, 2000), num_facets=(None, 4000)),
            "large meshes": dict(
                closed=True, manifold=True,
                num_vertices=(5000, 20000), num_facets=(None, 40000)),
        }

        total = 0
        for label, filters in queries.items():
            dset = thingi10k.dataset(**filters)
            n = min(len(dset), 10)
            print(f"  {label:40s} → {len(dset):6d} total, loading {n}...")
            for i in range(n):
                try:
                    thingi10k.load_file(dset[i]["file_path"])
                    total += 1
                except Exception as e:
                    print(f"    WARN: file_id={dset[i]['file_id']}: {e}")
            if n > 0:
                print(f"    OK")
        print(f"  Total pre-fetched: {total}")

    # ------------------------------------------------------------------
    # 5. Summary
    # ------------------------------------------------------------------
    print("\n[5] Cache summary:")
    if HF_CACHE.exists():
        r = subprocess.run(["du", "-sh", str(HF_CACHE)],
                           capture_output=True, text=True)
        print(f"  {r.stdout.strip()}")

    print("\n" + "=" * 60)
    print("DONE — cache fully populated. Run tests OFFLINE with:")
    print("  /home/dabajabaza/anaconda3/envs/PEEC/bin/python -m pytest \\")
    print("      mesh_issue_examiner/tests/test_thingi10k_e2e.py -v")
    print("=" * 60)


if __name__ == "__main__":
    main()
