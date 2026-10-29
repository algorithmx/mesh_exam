"""Offline tuning + validation sweeps for the sample-mesh constants.

Deterministic (no RNG) parameter search used to pick:

* the parametric triangular-tube ``torus`` constants (major radius, tube
  radius, per-ring twist) such that the tube is demonstrably embedded
  (zero proper non-adjacent triangle intersections on the float32 STL
  round-trip) AND the full inspection pipeline reports zero issues;
* the ``ICOSPHERE_STRETCH`` z-stretch such that both icosphere fixtures
  genuinely violate the min-angle cell-quality threshold through the real
  pipeline (not merely the edge-dihedral op).

It also validates the rebuilt fixtures (self_intersecting crossing /
point contact, multi_hole_plane, shell_structure, open_boundary) against
the pipeline before regeneration.

The chosen values are embedded in scripts/generate_sample_meshes.py.

Usage:
    python scripts/tune_sample_meshes.py            # full sweeps
    python scripts/tune_sample_meshes.py --quick    # validation only
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import distance as sdist

REPO_ROOT = Path(__file__).resolve().parents[1]

_spec = importlib.util.spec_from_file_location(
    "generate_sample_meshes", REPO_ROOT / "scripts" / "generate_sample_meshes.py")
gen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gen)

sys.path.insert(0, str(REPO_ROOT))

from mesh_issue_examiner.mcp.server import (  # noqa: E402
    MeshIssueExaminerServer, MCPRequest,
)
from mesh_issue_examiner.utils.triangle_intersection import (  # noqa: E402
    are_coplanar, bbox_overlap, detect_self_intersections_manual,
)

_TMP = Path(tempfile.mkdtemp(prefix="tune_meshes_"))


def export(mesh_v, mesh_f, name: str, ext: str = "stl") -> Path:
    p = _TMP / f"{name}.{ext}"
    m = trimesh.Trimesh(vertices=np.asarray(mesh_v, float),
                        faces=np.asarray(mesh_f), process=False)
    m.export(str(p))
    return p


async def inspect(server, path: Path):
    r = await server.handle_request(MCPRequest(
        method="mesh_load", params={"file_path": str(path)}))
    assert r.error is None, f"load failed {path}: {r.error}"
    ri = await server.handle_request(MCPRequest(
        method="mesh_inspect", params={"mesh_id": r.result["mesh_id"]}))
    assert ri.error is None, f"inspect failed {path}: {ri.error}"
    r2 = await server.handle_request(MCPRequest(
        method="mesh_get_issues", params={"mesh_id": r.result["mesh_id"]}))
    assert r2.error is None, f"get_issues failed {path}: {r2.error}"
    return ri.result, r2.result


# ---------------------------------------------------------------------------
# Geometry metrics (replicate the operation formulas for pre-filtering)
# ---------------------------------------------------------------------------

def face_angles(V, F):
    angs = []
    for a, b, c in F:
        P, Q, R = V[a], V[b], V[c]
        row = []
        for X, Y, Z in ((P, Q, R), (Q, R, P), (R, P, Q)):
            e1, e2 = Y - X, Z - X
            row.append(np.degrees(np.arccos(np.clip(
                np.dot(e1, e2) / np.linalg.norm(e1) / np.linalg.norm(e2),
                -1, 1))))
        angs.append(row)
    return np.array(angs)


def internal_dihedrals(V, F):
    """180 - angle-between-normals for edges shared by exactly 2 faces
    (same formula as topology.detect_edge_dihedral_angle)."""
    normals = []
    for a, b, c in F:
        n = np.cross(V[b] - V[a], V[c] - V[a])
        normals.append(n / np.linalg.norm(n))
    edge_faces = defaultdict(list)
    for fi, (a, b, c) in enumerate(F):
        for u, v in ((a, b), (b, c), (c, a)):
            edge_faces[(min(u, v), max(u, v))].append(fi)
    out = []
    for fl in edge_faces.values():
        if len(fl) == 2:
            n1, n2 = normals[fl[0]], normals[fl[1]]
            ang = np.degrees(np.arccos(np.clip(np.dot(n1, n2), -1, 1)))
            out.append(180.0 - ang)
    return np.array(out)


def vertex_valences(F):
    val = defaultdict(int)
    for a, b, c in F:
        val[a] += 1
        val[b] += 1
        val[c] += 1
    return np.array(list(val.values()))


# Pre-filter gates: slightly tighter than the pipeline thresholds
# (min angle 15, max angle 120, folds in [20, 160]) so anything that
# passes here still has margin when the real pipeline judges it.
GATE_MIN_ANGLE = 16.0
GATE_MAX_ANGLE = 114.0
GATE_MIN_DIHEDRAL = 22.0
GATE_MAX_DIHEDRAL = 158.0


def torus_metrics(V, F):
    """None if a hard geometric constraint fails, else margin dict.

    Works on in-memory arrays (no file round-trip): watertightness, single
    component, Euler characteristic 0, valence 4-8, angle and dihedral
    gates.  The on-disk round-trip is checked later for the survivors.
    """
    m = trimesh.Trimesh(vertices=V, faces=F, process=False)
    m.merge_vertices()
    if not m.is_watertight:
        return None
    if len(m.split(only_watertight=False)) != 1:
        return None
    V2, F2 = np.asarray(m.vertices, float), np.asarray(m.faces)
    angs = face_angles(V2, F2)
    dih = internal_dihedrals(V2, F2)
    val = vertex_valences(F2)
    edges = set()
    for a, b, c in F2:
        for u, v in ((a, b), (b, c), (c, a)):
            edges.add((min(u, v), max(u, v)))
    if len(V2) - len(edges) + len(F2) != 0:   # Euler characteristic must be 0
        return None
    if val.min() < 4 or val.max() > 8:
        return None
    if angs.min() < GATE_MIN_ANGLE or angs.max() > GATE_MAX_ANGLE:
        return None
    if dih.min() < GATE_MIN_DIHEDRAL or dih.max() > GATE_MAX_DIHEDRAL:
        return None
    return {
        "min_angle": float(angs.min()),
        "max_angle": float(angs.max()),
        "min_dihedral": float(dih.min()),
        "max_dihedral": float(dih.max()),
        "score": float(min(angs.min() - 15, 120 - angs.max(),
                           dih.min() - 20, 160 - dih.max())),
    }


def torus_margins(path: Path):
    """Re-check a written file (float32 round-trip) against the gates."""
    m = trimesh.load(str(path))
    m.merge_vertices()
    mg = torus_metrics(np.asarray(m.vertices, float), np.asarray(m.faces))
    if mg is None:
        return None
    if gen.proper_intersection_pairs(path):
        return None
    return mg


async def sweep_torus(server):
    print("\n=== torus sweep (seam-compatible twists only) ===")
    combos = []
    for nv in (3, 4):
        for nu in (6, 7, 8, 9, 10, 11, 12, 14):
            step = 360.0 / nv / nu          # seam closes: nu*step*k % (360/nv) == 0
            for k in range(1, int(round(60.0 / step)) + 1):
                twist = round(step * k, 4)
                for R in np.arange(2.0, 8.001, 0.25):
                    for r in np.arange(0.2, 1.601, 0.05):
                        combos.append((nu, nv, round(float(R), 2),
                                       round(float(r), 2), twist))
    survivors = []
    for nu, nv, R, r, twist in combos:
        V, F = gen.torus(R, r, twist, nu=nu, nv=nv)
        mg = torus_metrics(V, F)
        if mg is not None:
            survivors.append(((nu, nv, R, r, twist), mg))
    print(f"{len(survivors)} geometric survivors of {len(combos)} candidates")
    survivors.sort(key=lambda s: (-s[1]["score"], s[0][0]))
    checked = 0
    for params, mg in survivors:
        if checked >= 40:
            break
        nu, nv, R, r, twist = params
        p = export(*gen.torus(R, r, twist, nu=nu, nv=nv), f"torus_{checked}")
        mg2 = torus_margins(p)             # float32 round-trip + intersections
        if mg2 is None:
            print(f"  round-trip rejected {params}")
            checked += 1
            continue
        insp, iss = await inspect(server, p)
        checked += 1
        if iss["total_count"] == 0:
            print(f"RECOMMENDED torus: nu={nu} nv={nv} R={R} r={r} "
                  f"twist={twist}deg")
            print(f"  margins: {mg2}")
            return params
        print(f"  pipeline rejected {params}: "
              f"{sorted({i['operation_id'] for i in iss['issues']})}")
    print("!! no candidate satisfied the full pipeline")
    return None


def chord_pair_report(path: Path):
    """Validate the (40, 77) chord-through-interior regression pair."""
    m = trimesh.load(str(path))
    V, F = np.asarray(m.vertices, float), np.asarray(m.faces)
    t1, t2 = V[F[40]], V[F[77]]
    checks = {}
    checks["non_adjacent"] = len(set(F[40]) & set(F[77])) == 0
    checks["bbox_overlap"] = bool(bbox_overlap(t1, t2))
    checks["not_coplanar"] = are_coplanar(t1, t2) is False

    min_dist = float("inf")
    ts = np.linspace(0, 1, 500)
    edges1 = [(t1[0], t1[1]), (t1[1], t1[2]), (t1[2], t1[0])]
    edges2 = [(t2[0], t2[1]), (t2[1], t2[2]), (t2[2], t2[0])]
    for a, b in edges1:
        p1 = a[None, :] * (1 - ts[:, None]) + b[None, :] * ts[:, None]
        for c, d in edges2:
            p2 = c[None, :] * (1 - ts[:, None]) + d[None, :] * ts[:, None]
            min_dist = min(min_dist, float(sdist.cdist(p1, p2).min()))
    checks["min_edge_distance>0.01"] = min_dist > 0.01

    edge_start, edge_end = t1[2], t1[0]
    ray_dir = edge_end - edge_start
    v0, v1, v2 = t2[0], t2[1], t2[2]
    e1, e2 = v1 - v0, v2 - v0
    pvec = np.cross(ray_dir, e2)
    det = np.dot(e1, pvec)
    inv_det = 1.0 / det
    tvec = edge_start - v0
    u = np.dot(tvec, pvec) * inv_det
    qvec = np.cross(tvec, e1)
    v = np.dot(ray_dir, qvec) * inv_det
    t = np.dot(e2, qvec) * inv_det
    P = edge_start + t * ray_dir
    max_r = np.linalg.norm(V, axis=1).max()
    checks["chord_point_inside"] = np.linalg.norm(P) < max_r * 0.95

    checks["detector_silent"] = detect_self_intersections_manual(V, F) == []
    return checks, float(min_dist)


async def sweep_icosphere(server):
    print("\n=== icosphere stretch sweep ===")
    for s in (1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0, 8.0):
        v1, f1 = gen.icosphere(1, stretch=s, chord_pair=(18, 12))
        p1 = export(v1, f1, f"ico1_{s}", ext="obj")
        _, iss1 = await inspect(server, p1)
        ops1 = {i["operation_id"] for i in iss1["issues"]}
        minang1 = "quality.compute_cell_quality.min_angle" in ops1

        v2, f2 = gen.icosphere(2, stretch=s)
        p2 = export(v2, f2, f"ico2_{s}", ext="stl")
        _, iss2 = await inspect(server, p2)
        ops2 = {i["operation_id"] for i in iss2["issues"]}
        minang2 = "quality.compute_cell_quality.min_angle" in ops2

        chords, min_dist = chord_pair_report(p1)
        ok_chords = all(chords.values())
        print(f"stretch {s:4.1f}: l1 min_angle_op={minang1} "
              f"({len(iss1['issues'])} issues), l2 min_angle_op={minang2} "
              f"({len(iss2['issues'])} issues), chord_ok={ok_chords}")
        if minang1 and minang2 and ok_chords:
            print(f"RECOMMENDED ICOSPHERE_STRETCH = {s}")
            print(f"  l1 ops: {sorted(ops1)}")
            print(f"  l2 ops: {sorted(ops2)}")
            print(f"  chord checks: {chords} (min edge dist {min_dist:.4f})")
            return s
    print("!! no stretch satisfied all requirements")
    return None


async def validate_rebuilt(server):
    print("\n=== rebuilt-fixture validation (temp files) ===")
    fixtures = gen.build_fixtures()

    for name in ("self_intersecting", "self_intersecting_point_contact"):
        ext, v, f = fixtures[name]
        p = export(v, f, f"val_{name}", ext)
        insp, iss = await inspect(server, p)
        ops = {i["operation_id"] for i in iss["issues"]}
        crossings = gen.proper_intersection_pairs(p)
        ok = ("topology.detect_self_intersections" in ops
              and insp["passed"] is False)
        print(f"{name}: detect_op={'topology.detect_self_intersections' in ops} "
              f"passed={insp['passed']} proper_crossings={len(crossings)} "
              f"{'OK' if ok else 'FAIL'}")

    ext, v, f = fixtures["multi_hole_plane"]
    p = export(v, f, "val_multi_hole", ext)
    insp, iss = await inspect(server, p)
    ops = {i["operation_id"] for i in iss["issues"]}
    nb = [i for i in iss["issues"] if "boundary" in i["operation_id"]]
    be = nb[0]["metrics"].get("boundary_edge_count") if nb else None
    print(f"multi_hole_plane: boundary_edges={be} "
          f"nonmanifold_ops={sorted(o for o in ops if 'nonmanifold' in o)} "
          f"{'OK' if be == 28 and not any('nonmanifold' in o for o in ops) else 'FAIL'}")

    ext, v, f = fixtures["shell_structure"]
    p = export(v, f, "val_shell", ext)
    m = trimesh.load(str(p))
    insp, iss = await inspect(server, p)
    ops = {i["operation_id"] for i in iss["issues"]}
    print(f"shell_structure: signed_volume={m.volume:.6f} "
          f"watertight={m.is_watertight} "
          f"boundary_ops={[o for o in ops if 'boundary' in o]} "
          f"{'OK' if abs(m.volume - 0.875) < 1e-6 and m.is_watertight else 'FAIL'}")

    ext, v, f = fixtures["open_boundary"]
    p = export(v, f, "val_open", ext)
    insp, iss = await inspect(server, p)
    nb = [i for i in iss["issues"] if "boundary" in i["operation_id"]]
    be = nb[0]["metrics"].get("boundary_edge_count") if nb else None
    print(f"open_boundary: verts={len(v)} faces={len(f)} "
          f"boundary_edges={be} "
          f"{'OK' if be == 6 and len(f) == 4 else 'FAIL'}")

    # Default-constant torus: embedded constants must reproduce the
    # sweep's pipeline-verified zero-issue result on the float32 round-trip.
    p = export(*gen.torus(), "val_torus_default", "stl")
    insp, iss = await inspect(server, p)
    crossings = gen.proper_intersection_pairs(p)
    ok = (iss["total_count"] == 0 and not crossings)
    print(f"torus(default): issues={iss['total_count']} "
          f"proper_crossings={len(crossings)} "
          f"{'OK' if ok else 'FAIL'}")

    # Default-constant icospheres.
    p1 = export(*gen.icosphere(1, chord_pair=(18, 12)), "val_ico1", "obj")
    _, iss1 = await inspect(server, p1)
    p2 = export(*gen.icosphere(2), "val_ico2", "stl")
    _, iss2 = await inspect(server, p2)
    ok1 = any("compute_cell_quality" in i["operation_id"]
              for i in iss1["issues"])
    ok2 = any("compute_cell_quality" in i["operation_id"]
              for i in iss2["issues"])
    print(f"icosphere_l1(default): quality_op={ok1} "
          f"{'OK' if ok1 else 'FAIL'}")
    print(f"icosphere_l2(default): quality_op={ok2} "
          f"{'OK' if ok2 else 'FAIL'}")


async def main(quick: bool):
    server = MeshIssueExaminerServer(auto_register_operations=True)
    await validate_rebuilt(server)
    if quick:
        return
    torus_params = await sweep_torus(server)
    stretch = await sweep_icosphere(server)
    print("\n=== summary ===")
    print(f"torus params (nu, R, r, twist): {torus_params}")
    print(f"ICOSPHERE_STRETCH: {stretch}")
    print(f"temp dir (safe to delete): {_TMP}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="validation only, no parameter sweeps")
    args = ap.parse_args()
    asyncio.run(main(quick=args.quick))
