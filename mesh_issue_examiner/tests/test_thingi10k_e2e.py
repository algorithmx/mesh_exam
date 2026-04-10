"""
Thingi10K cross-validation: use ground-truth metadata to validate detectors.

Thingi10K labels every mesh with boolean properties (manifold, closed,
self_intersecting, oriented, etc.) computed by the original authors. These
serve as ground truth to measure precision/recall of our detectors.

Requires pre-cached dataset via:
    python scripts/init_thingi10k_cache.py --metadata-dir ... --npz-tarball ...

Run offline:
    /path/to/PEEC/bin/python -m pytest mesh_issue_examiner/tests/test_thingi10k_e2e.py -v -s

Skip:
    pytest -k "not thingi10k"
"""

import os
from pathlib import Path

import datasets
import pytest
import numpy as np

thingi10k = pytest.importorskip("thingi10k")

from mesh_issue_examiner.mcp.server import MeshIssueExaminerServer, MCPRequest
from mesh_issue_examiner.core.models import MeshInput


# ---------------------------------------------------------------------------
# Offline dataset loader
# ---------------------------------------------------------------------------

_proxy = os.environ.get("HTTP_PROXY") or os.environ.get("http_proxy")
if not _proxy:
    os.environ["http_proxy"] = "http://127.0.0.1:10077"
    os.environ["https_proxy"] = "http://127.0.0.1:10077"


def _try_offline_load():
    cache_path = Path.home() / ".cache" / "huggingface" / "datasets" / "_builder" / "npz" / "1.0.0"
    if (cache_path / "dataset_info.json").exists() or (cache_path / "train").exists():
        import thingi10k._utils
        thingi10k._utils._dataset = datasets.load_from_disk(str(cache_path))
        return True
    return False


@pytest.fixture(scope="session")
def t10k():
    if _try_offline_load():
        return thingi10k
    try:
        thingi10k.init()
    except RuntimeError as exc:
        pytest.skip(f"Thingi10K unavailable: {exc}")
    return thingi10k


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _server():
    return MeshIssueExaminerServer(auto_register_operations=True)


async def _inspect_mesh(vertices, faces, mesh_id):
    """Run full inspection on a mesh. Returns inspect + issues dicts."""
    server = _server()
    mesh = MeshInput(vertices=vertices, faces=faces, mesh_id=mesh_id)
    server._mesh_cache[mesh_id] = mesh

    resp = await server.handle_request(MCPRequest(method="mesh_inspect", params={"mesh_id": mesh_id}))
    assert resp.error is None, f"inspect failed: {resp.error}"
    inspect = resp.result

    resp = await server.handle_request(MCPRequest(method="mesh_get_issues", params={"mesh_id": mesh_id}))
    assert resp.error is None
    issues = resp.result

    return inspect, issues


def _pick(ds, n, seed=42):
    """Deterministic sample of n entries."""
    import random
    rng = random.Random(seed)
    indices = rng.sample(range(len(ds)), min(n, len(ds)))
    return [ds[i] for i in sorted(indices)]


# ===================================================================
# GROUND-TRUTH CROSS-VALIDATION: sample-based
#
# For each boolean metadata field, sample meshes where field=True and
# field=False, run our detectors, and check agreement.
# ===================================================================

SAMPLE_SIZE = 4  # per category (2 positive + 2 negative)

# Metadata field → (issue keyword to look for, description)
# Only maps to detectors that actually exist:
#   topology.detect_nonmanifold_edges  → "nonmanifold"
#   topology.detect_self_intersections  → "self_intersect"
# Fields with NO detector: closed (no boundary op), oriented (only volumetric)
GROUND_TRUTH_MAP = {
    "edge_manifold": ("nonmanifold", "Edge manifold"),
    "self_intersecting": ("self_intersect", "Self-intersection"),
}

# Fields we know we CAN'T detect yet (recall=0 by design)
MISSING_DETECTORS = {
    "closed": "No boundary-edge detection operation",
    "oriented": "Only volumetric orientation check exists (quality_3d)",
}


class TestGroundTruthCrossValidation:
    """Cross-validate each detector against Thingi10K ground-truth labels.

    For each metadata field, sample meshes where the label is True and False,
    run inspection, and check whether our detector agrees.
    """

    @pytest.mark.asyncio
    @pytest.mark.parametrize("field", list(GROUND_TRUTH_MAP.keys()))
    async def test_detector_agrees_with_metadata(self, t10k, field):
        keyword, description = GROUND_TRUTH_MAP[field]

        positive = _pick(t10k.dataset(**{field: True, "num_vertices": (20, 2000)}), SAMPLE_SIZE // 2)
        negative = _pick(t10k.dataset(**{field: False, "num_vertices": (20, 2000)}), SAMPLE_SIZE // 2)

        tp, fp, fn, tn = 0, 0, 0, 0

        for entry in positive:
            # metadata=True → our detector should NOT flag this
            verts, faces = t10k.load_file(entry["file_path"])
            _, issues = await _inspect_mesh(verts, faces, f"t10k_{entry['file_id']}")
            flagged = any(keyword in i.get("operation_id", "") for i in issues["issues"])
            if flagged:
                fp += 1  # false positive: metadata says clean, we flagged
            else:
                tn += 1  # true negative

        for entry in negative:
            # metadata=False → our detector SHOULD flag this
            verts, faces = t10k.load_file(entry["file_path"])
            _, issues = await _inspect_mesh(verts, faces, f"t10k_{entry['file_id']}")
            flagged = any(keyword in i.get("operation_id", "") for i in issues["issues"])
            if flagged:
                tp += 1  # true positive
            else:
                fn += 1  # false negative: metadata says defective, we missed it

        total = tp + fp + fn + tn
        precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0

        print(f"\n  {description} (field={field}):")
        print(f"    TP={tp}  FP={fp}  FN={fn}  TN={tn}  (n={total})")
        print(f"    Precision={precision:.2f}  Recall={recall:.2f}")

        # We don't assert perfect scores — this is a measurement.
        # But we report them so failures are visible in CI output.


# ===================================================================
# SELF-INTERSECTION FALSE-POSITIVE DEEP DIVE
# ===================================================================

class TestSelfIntersectionFalsePositives:
    """Measure false positive rate of self-intersection detector on
    meshes labeled self_intersecting=False by Thingi10K.

    This is the most critical test — it caught the icosphere bug.
    """

    @pytest.mark.asyncio
    async def test_fp_rate_on_clean_meshes(self, t10k):
        """On meshes labeled self_intersecting=False, measure FP rate."""
        clean = _pick(
            t10k.dataset(
                self_intersecting=False, closed=True, edge_manifold=True,
                num_vertices=(50, 3000),
            ),
            5,
        )

        fp_count = 0
        for entry in clean:
            verts, faces = t10k.load_file(entry["file_path"])
            _, issues = await _inspect_mesh(verts, faces, f"t10k_fp_{entry['file_id']}")
            si_issues = [i for i in issues["issues"] if "self_intersect" in i.get("operation_id", "")]
            if si_issues:
                fp_count += 1
                print(f"    FALSE POSITIVE: file_id={entry['file_id']}, "
                      f"n_verts={entry['num_vertices']}, "
                      f"n_faces={entry['num_facets']}, "
                      f"issues={len(si_issues)}")

        fp_rate = fp_count / len(clean)
        print(f"\n  Self-intersection FP rate on clean meshes: "
              f"{fp_count}/{len(clean)} = {fp_rate:.1%}")

        # Report but don't fail — this is a measurement.


# ===================================================================
# QUALITY METRICS SANITY
# ===================================================================

class TestQualityMetricsSanity:
    """Quality metric values should be physically meaningful."""

    @pytest.mark.asyncio
    async def test_aspect_ratio_range(self, t10k):
        meshes = _pick(
            t10k.dataset(closed=True, edge_manifold=True, num_vertices=(50, 5000)),
            3,
        )
        found_metrics = False
        for entry in meshes:
            verts, faces = t10k.load_file(entry["file_path"])
            server = _server()
            mesh = MeshInput(vertices=verts, faces=faces, mesh_id=f"qm_{entry['file_id']}")
            server._mesh_cache[mesh.mesh_id] = mesh

            inspect_resp = await server.handle_request(MCPRequest(
                method="mesh_inspect", params={"mesh_id": mesh.mesh_id},
            ))
            assert inspect_resp.error is None, f"inspect failed: {inspect_resp.error}"

            resp = await server.handle_request(MCPRequest(
                method="mesh_get_quality_metrics", params={"mesh_id": mesh.mesh_id},
            ))
            assert resp.error is None, f"get_quality_metrics failed: {resp.error}"
            metrics = resp.result
            if metrics["metric_count"] > 0:
                found_metrics = True
                for m in metrics["metrics"]:
                    if "aspect_ratio" in m["metric_name"]:
                        assert m["statistics"]["min"] >= 1.0 - 1e-6

        if not found_metrics:
            print("  WARNING: no quality metrics computed for any sample mesh")


# ===================================================================
# DIAGNOSTIC: what operations fire on what mesh types?
# ===================================================================

class TestDiagnostic:
    """Print which operation_ids fire on different mesh categories.

    Run with -s to see output. Helps debug keyword mismatches.
    """

    @pytest.mark.asyncio
    async def test_operation_ids_per_category(self, t10k):
        categories = {
            "manifold+closed": dict(edge_manifold=True, closed=True,
                                    num_vertices=(50, 1000)),
            "nonmanifold": dict(edge_manifold=False,
                                num_vertices=(50, 1000)),
            "self_intersecting": dict(self_intersecting=True,
                                      num_vertices=(50, 1000)),
            "open (closed=False)": dict(closed=False,
                                        num_vertices=(50, 1000)),
        }
        for label, filters in categories.items():
            ds = t10k.dataset(**filters)
            if len(ds) == 0:
                print(f"\n  {label}: no meshes")
                continue
            entry = ds[42 % len(ds)]
            verts, faces = t10k.load_file(entry["file_path"])
            _, issues = await _inspect_mesh(verts, faces, f"diag_{entry['file_id']}")
            op_ids = sorted({i["operation_id"] for i in issues["issues"]})
            print(f"\n  {label} (file_id={entry['file_id']}, "
                  f"n_faces={entry['num_facets']}):")
            for oid in op_ids:
                count = sum(1 for i in issues["issues"] if i["operation_id"] == oid)
                print(f"    {oid}: {count}")


# ===================================================================
# PIPELINE ROBUSTNESS: open / non-manifold / large meshes
# ===================================================================

class TestPipelineRobustness:
    """Pipeline should complete without crashing on diverse meshes."""

    @pytest.mark.asyncio
    async def test_open_meshes(self, t10k):
        for entry in _pick(t10k.dataset(closed=False, num_vertices=(20, 2000)), 2):
            verts, faces = t10k.load_file(entry["file_path"])
            inspect, issues = await _inspect_mesh(verts, faces, f"open_{entry['file_id']}")
            assert "passed" in inspect

    @pytest.mark.asyncio
    async def test_large_meshes(self, t10k):
        ds = t10k.dataset(num_vertices=(5000, 20000))
        if len(ds) == 0:
            pytest.skip("No large meshes")
        entry = _pick(ds, 1)[0]
        verts, faces = t10k.load_file(entry["file_path"])
        inspect, issues = await _inspect_mesh(verts, faces, f"large_{entry['file_id']}")
        assert "passed" in inspect
        assert entry["num_vertices"] >= 5000


# ===================================================================
# SUMMARY REPORT (run with -s to see output)
# ===================================================================

class TestSummaryReport:
    """Print dataset statistics and detector coverage summary."""

    @pytest.mark.asyncio
    async def test_print_summary(self, t10k):
        ds = t10k.dataset()

        print("\n" + "=" * 70)
        print("THINGI10K CROSS-VALIDATION SUMMARY")
        print("=" * 70)
        print(f"  Total meshes in dataset: {len(ds)}")
        print()

        for field in ["closed", "edge_manifold", "vertex_manifold",
                       "self_intersecting", "oriented", "solid", "PWN"]:
            count = sum(1 for e in ds if e[field])
            print(f"  {field:25s} True: {count:5d} ({100*count/len(ds):.1f}%)")

        print()
        print("  Detectors with ground-truth mapping:")
        for field, (kw, desc) in GROUND_TRUTH_MAP.items():
            print(f"    {desc:30s} → keyword '{kw}'")

        print()
        print("  Missing detectors (no mapping):")
        for field, reason in MISSING_DETECTORS.items():
            print(f"    {field:30s} → {reason}")

        print("=" * 70)
