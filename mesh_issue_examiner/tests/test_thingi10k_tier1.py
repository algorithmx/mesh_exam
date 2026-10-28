"""Tier-1 Thingi10K cross-validation tests (offline, default suite).

Every mesh in the frozen tier-1 corpus (``e2e/thingi10k/tier1``, tracked in
git alongside the manifest ``e2e/thingi10k/selection.json``) is loaded
through the real ``mesh_load`` MCP path and inspected via ``mesh_inspect``
with the per-mesh frozen ``operations=`` probe list.  The frozen contract
partitions the probe set:

* ``expect``  - label-mapped operations that must fire;
* ``allow``   - other operations that fired at calibration time;
* ``forbid``  - probed operations that must stay silent.

Contracts freeze *observed* detector behaviour measured against the
Thingi10K ground-truth labels; label-backed misses and canary
false-positives are recorded in ``selection.json`` under
``calibration.known_gaps`` instead of being asserted, so this suite pins
the current detector output without going chronically red.  If a detector
improves or regresses, re-run ``scripts/fetch_thingi10k.py --calibrate``
and review the diff of ``selection.json``.

The whole module skips when the manifest or corpus is missing (fresh clone
before the corpus lands, or a deliberately trimmed checkout).
"""

import asyncio
import hashlib
import json
from pathlib import Path

import pytest

from mesh_issue_examiner.mcp.server import (
    MCPRequest,
    MeshIssueExaminerServer,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
T10K_DIR = REPO_ROOT / "e2e" / "thingi10k"
TIER1_DIR = T10K_DIR / "tier1"
SELECTION_JSON = T10K_DIR / "selection.json"

FETCH_HINT = (
    "tier-1 corpus unavailable - rebuild it with "
    "`python scripts/fetch_thingi10k.py --select --fetch-tier1 --calibrate`"
)


def _load_selection() -> dict:
    if not SELECTION_JSON.exists():
        pytest.skip(FETCH_HINT, allow_module_level=True)
    doc = json.loads(SELECTION_JSON.read_text(encoding="utf-8"))
    tier1 = doc.get("tier1") or []
    if not tier1 or not all("contract" in item for item in tier1):
        pytest.skip(
            "selection.json has no calibrated tier-1 contracts - "
            "run `python scripts/fetch_thingi10k.py --calibrate`",
            allow_module_level=True,
        )
    if not TIER1_DIR.is_dir() or not any(TIER1_DIR.iterdir()):
        pytest.skip(FETCH_HINT, allow_module_level=True)
    return doc


SELECTION = _load_selection()
TIER1 = SELECTION["tier1"]


@pytest.fixture(scope="module")
def server() -> MeshIssueExaminerServer:
    """One shared server instance for the whole module.

    The operation registry is a process-wide singleton, so a single server
    both avoids redundant re-registration and mirrors a real session.
    """
    return MeshIssueExaminerServer(auto_register_operations=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@pytest.mark.parametrize(
    "item", TIER1, ids=[f"{i['file_id']}-{i['category']}" for i in TIER1])
class TestTier1Contract:
    """Each tier-1 mesh must reproduce its frozen detector contract."""

    def _load_and_inspect(self, server: MeshIssueExaminerServer,
                          item: dict, path: Path) -> set[str]:
        load_resp = asyncio.run(server.handle_request(MCPRequest(
            method="mesh_load",
            params={"file_path": str(path),
                    "mesh_id": f"tier1-{item['file_id']}"})))
        assert load_resp.error is None, load_resp.error
        mesh_id = load_resp.result["mesh_id"]

        inspect_resp = asyncio.run(server.handle_request(MCPRequest(
            method="mesh_inspect",
            params={"mesh_id": mesh_id,
                    "operations": item["contract"]["ops"]})))
        assert inspect_resp.error is None, inspect_resp.error

        # mesh_inspect returns a summary; the full result (with per-issue
        # operation_id) is cached server-side, exactly as during calibration.
        result = server._result_cache[mesh_id]
        return {issue.operation_id for issue in result.issues}

    @staticmethod
    def _fired_probes(issue_ids: set[str], probes: list[str]) -> set[str]:
        """Map issue operation_ids back to probe granularity.

        Some operations emit issues with expanded ids (e.g.
        ``quality.compute_cell_quality.aspect_ratio`` for the probe
        ``quality.compute_cell_quality``); contracts are frozen at probe
        granularity, matching ``--calibrate``'s fired-probe semantics.
        """
        return {
            probe for probe in probes
            if any(oid == probe or oid.startswith(probe + ".")
                   for oid in issue_ids)
        }

    def test_file_matches_frozen_sha256(self, item):
        path = REPO_ROOT / item["path"]
        assert path.is_file(), f"missing tier-1 mesh: {path}"
        actual = _sha256(path)
        assert actual == item["sha256"], (
            f"{item['filename']}: sha256 mismatch - tier-1 corpus file "
            "differs from the frozen manifest; re-fetch with "
            "`python scripts/fetch_thingi10k.py --fetch-tier1`"
        )

    def test_frozen_contract(self, server, item):
        path = REPO_ROOT / item["path"]
        assert path.is_file(), f"missing tier-1 mesh: {path}"
        issue_ids = self._load_and_inspect(server, item, path)
        contract = item["contract"]
        fired = self._fired_probes(issue_ids, contract["ops"])
        expect = set(contract["expect"])
        allow = set(contract["allow"])
        forbid = set(contract["forbid"])

        missing = expect - fired
        assert not missing, (
            f"{item['filename']} [{item['category']}]: expected "
            f"{sorted(missing)} no longer fire (detector regression or "
            "contract drift - re-run --calibrate after review)"
        )
        forbidden_fired = fired & forbid
        assert not forbidden_fired, (
            f"{item['filename']} [{item['category']}]: forbidden "
            f"{sorted(forbidden_fired)} fired (behaviour drift - re-run "
            "--calibrate after review)"
        )
        unlisted = fired - expect - allow
        assert not unlisted, (
            f"{item['filename']} [{item['category']}]: newly firing "
            f"{sorted(unlisted)} not in the frozen allow-list (re-run "
            "--calibrate after review)"
        )


class TestTier1Manifest:
    """Manifest-level sanity checks for the frozen corpus."""

    def test_corpus_size_within_budget(self):
        total = sum(f.stat().st_size for f in TIER1_DIR.iterdir()
                    if f.is_file())
        budget = 60 * 1024 * 1024
        assert total <= budget, (
            f"tier-1 corpus is {total / 1e6:.1f} MB, budget is 60 MB"
        )

    def test_mesh_count_within_cap(self):
        assert len(TIER1) <= 50

    def test_calibration_records_known_gaps(self):
        """The cross-validation findings must ship with the manifest."""
        calibration = SELECTION.get("calibration", {})
        gaps = calibration.get("known_gaps")
        assert isinstance(gaps, list), "calibration.known_gaps missing"
        # Every gap line is classified as a false positive or false negative.
        for gap in gaps:
            assert gap.startswith(("fp ", "fn ")), gap

    def test_no_stale_files_in_corpus(self):
        """Every file in tier1/ belongs to the manifest (no orphans)."""
        listed = {item["filename"] for item in TIER1}
        orphans = {
            f.name for f in TIER1_DIR.iterdir()
            if f.is_file() and f.name not in listed
        }
        assert not orphans, f"orphaned tier-1 files: {sorted(orphans)}"
