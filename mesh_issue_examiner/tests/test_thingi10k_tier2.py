"""Tier-2 Thingi10K remote tests (opt-in: ``pytest -m thingi10k_remote``).

Tier 2 is the heavy artillery of the cross-validation corpus: multi-defect
combinations, extreme-statistic outliers, and 50k-300k-facet meshes,
downloaded on the fly into the gitignored ``e2e/thingi10k/tier2`` cache by
``scripts/fetch_thingi10k.py --fetch-tier2`` (resumable, sha256-pinned).

The module is deselected from the default run (see ``pytest.ini``) because
it needs network + a few hundred MB of downloads; it **skips cleanly**
when the fetch cannot reach the network.

Probe set: every registered operation except ``detect_self_intersections``
— the SI detector is O(n^2) (measured ~130 s at only 5k faces), which is
intractable at tier-2 sizes; SI cross-validation is covered by tier 1.
"""

import asyncio
import json
import subprocess
import sys
from pathlib import Path

import pytest

from mesh_issue_examiner.core.exceptions import MeshLoadError
from mesh_issue_examiner.mcp.server import (
    MCPRequest,
    MeshIssueExaminerServer,
)
from mesh_issue_examiner.operations.operation_registry import registry
from mesh_issue_examiner.utils.mesh_loader import load_mesh_from_file

pytestmark = pytest.mark.thingi10k_remote

REPO_ROOT = Path(__file__).resolve().parents[2]
T10K_DIR = REPO_ROOT / "e2e" / "thingi10k"
SELECTION_JSON = T10K_DIR / "selection.json"
FETCH_SCRIPT = REPO_ROOT / "scripts" / "fetch_thingi10k.py"
SI_OP_ID = "topology.detect_self_intersections"


def _load_selection() -> dict:
    if not SELECTION_JSON.exists():
        pytest.skip(
            "selection.json missing - run "
            "`python scripts/fetch_thingi10k.py --select`",
            allow_module_level=True,
        )
    return json.loads(SELECTION_JSON.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def tier2(server) -> list[dict]:
    """Download/verify the tier-2 corpus once, then yield its entries.

    Delegates to the fetch script (rate-limit handling, sha256 pinning,
    resume).  Any fetch failure - offline, throttled - skips the module
    rather than failing it.
    """
    proc = subprocess.run(
        [sys.executable, str(FETCH_SCRIPT), "--fetch-tier2"],
        capture_output=True, text=True, timeout=3600,
    )
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-3:]
        pytest.skip(f"tier-2 fetch unavailable (offline?): {tail}")
    doc = _load_selection()
    tier2_entries = doc["tier2"]
    assert tier2_entries, "selection.json has no tier-2 entries"
    return tier2_entries


@pytest.fixture(scope="module")
def server() -> MeshIssueExaminerServer:
    return MeshIssueExaminerServer(auto_register_operations=True)


@pytest.fixture(scope="module")
def probe_ops() -> list[str]:
    """All registered ops except the intractable SI detector."""
    return sorted(op.operation_id for op in registry.list_all()
                  if op.operation_id != SI_OP_ID)


_TIER2 = _load_selection()["tier2"]
_INSPECT_ITEMS = [e for e in _TIER2 if e["category"] != "corrupt_robustness"]
_CORRUPT_ITEMS = [e for e in _TIER2 if e["category"] == "corrupt_robustness"]


@pytest.mark.parametrize(
    "item", _INSPECT_ITEMS,
    ids=[f"{e['file_id']}-{e['category']}" for e in _INSPECT_ITEMS])
def test_tier2_mesh_loads_and_inspects(tier2, server, probe_ops, item):
    """Every heavy tier-2 mesh loads through the MCP path and survives a
    full non-SI inspection without errors."""
    path = REPO_ROOT / item["path"]
    assert path.is_file(), f"tier-2 mesh not cached: {path}"

    load_resp = asyncio.run(server.handle_request(MCPRequest(
        method="mesh_load",
        params={"file_path": str(path),
                "mesh_id": f"tier2-{item['file_id']}"})))
    assert load_resp.error is None, load_resp.error
    mesh_id = load_resp.result["mesh_id"]
    n_faces = load_resp.result["n_faces"]
    n_vertices = load_resp.result["n_vertices"]

    # Face counts must match the dataset ground truth exactly; vertex
    # counts may differ (STL vertex merging is loader-dependent).
    expected_faces = item["stats"]["num_faces"]
    assert n_faces == expected_faces, (
        f"{item['filename']}: loaded {n_faces} faces, "
        f"Thingi10K reports {expected_faces}"
    )
    assert 0 < n_vertices <= 3 * n_faces

    inspect_resp = asyncio.run(server.handle_request(MCPRequest(
        method="mesh_inspect",
        params={"mesh_id": mesh_id, "operations": probe_ops})))
    assert inspect_resp.error is None, inspect_resp.error

    result = server._result_cache[mesh_id]
    issue_ids = {issue.operation_id for issue in result.issues}
    if item["category"] == "multi_defect":
        # Smoke-level cross-check on un-calibrated meshes: a mesh carrying
        # >= 2 defect labels must not pass through completely silently.
        assert issue_ids, (
            f"{item['filename']} [multi_defect]: no detector fired at all"
        )


@pytest.mark.parametrize(
    "item", _CORRUPT_ITEMS,
    ids=[f"{e['file_id']}-corrupt" for e in _CORRUPT_ITEMS])
def test_corrupt_file_fails_gracefully(tier2, item):
    """Dataset-errata corrupt files must raise MeshLoadError (never crash,
    never silently load garbage)."""
    path = REPO_ROOT / item["path"]
    assert path.is_file(), f"corrupt sample not cached: {path}"
    with pytest.raises(MeshLoadError):
        load_mesh_from_file(path)
