"""
Benchmark: edge-face adjacency construction (loop vs vectorized).

Times mesh_issue_examiner.utils.topology.build_edge_face_map (vectorized
triangle fast path) against the module-private loop reference
``_build_edge_face_map_loop`` on deterministic synthetic triangle grid
meshes with ~1e4 / 1e5 / 1e6 faces. Correctness (vectorized result ==
loop reference) is asserted for every size before timing.

Dependency-free: requires only numpy and the package itself (no mesh
libraries such as trimesh/pyvista).

Usage (repo venv python, from the repo root or anywhere else -- the
script adds the repo root to sys.path itself):

    python benchmarks/bench_edge_face_map.py
"""

import sys
import timeit
from pathlib import Path

import numpy as np

# Make the repo root importable regardless of the current directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mesh_issue_examiner.utils.topology import (
    build_edge_face_map,
    _build_edge_face_map_loop,
)

FACE_COUNTS = (10_000, 100_000, 1_000_000)
REPEAT = 3


def make_grid_triangles(n_faces_target: int):
    """Deterministically build a regular grid triangle mesh.

    Vertices lie on an (m+1) x (m+1) integer grid with
    ``m = ceil(sqrt(n_faces_target / 2))``; each of the m*m cells is split
    into two triangles, giving approximately ``n_faces_target`` faces.
    No randomness is involved, so runs are reproducible.

    Args:
        n_faces_target: approximate number of triangle faces requested.

    Returns:
        (vertices, faces) tuple of numpy arrays.
    """
    m = int(np.ceil(np.sqrt(n_faces_target / 2.0)))
    n = m + 1  # vertices per grid row

    # Vertex id v(i, j) = i * n + j
    i, j = np.meshgrid(np.arange(m), np.arange(m), indexing='ij')
    v00 = (i * n + j).ravel()
    v01 = (i * n + j + 1).ravel()
    v10 = ((i + 1) * n + j).ravel()
    v11 = ((i + 1) * n + j + 1).ravel()

    faces = np.concatenate([
        np.stack([v00, v01, v10], axis=1),
        np.stack([v01, v11, v10], axis=1),
    ]).astype(np.int64)

    n_vertices = n * n
    ids = np.arange(n_vertices)
    vertices = np.column_stack([ids % n, ids // n, np.zeros(n_vertices)])

    return vertices, faces


def best_of(fn, *args, repeat: int = REPEAT) -> float:
    """Return the best wall time of ``repeat`` single calls, in seconds."""
    timer = timeit.Timer(lambda: fn(*args))
    return min(timer.repeat(repeat=repeat, number=1))


def main() -> None:
    """Run the loop vs vectorized benchmark and print a results table."""
    header = (f"{'faces':>10} | {'edges':>10} | {'loop [s]':>10} | "
              f"{'vectorized [s]':>14} | {'speedup':>8}")
    print(header)
    print('-' * len(header))

    for target in FACE_COUNTS:
        _vertices, faces = make_grid_triangles(target)

        # Verify equivalence before timing anything
        reference = _build_edge_face_map_loop(faces)
        vectorized = build_edge_face_map(faces)
        if vectorized != reference:
            raise SystemExit(
                f"MISMATCH at {len(faces)} faces: vectorized result differs "
                "from loop reference"
            )

        t_loop = best_of(_build_edge_face_map_loop, faces)
        t_vec = best_of(build_edge_face_map, faces)
        print(f"{len(faces):>10,} | {len(reference):>10,} | {t_loop:>10.4f} | "
              f"{t_vec:>14.4f} | {t_loop / t_vec:>7.1f}x")

    print("\nVerified for all sizes: vectorized == loop reference.")


if __name__ == '__main__':
    main()
