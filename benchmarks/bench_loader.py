"""
Benchmark: adapter access and mesh file loading (cold vs warm).

Measures, via mesh_issue_examiner.adapters.factory.get_adapter:

1. Cold adapter access -- get_adapter() with the adapter instance cache
   cleared beforehand (clear_adapter_cache). Note: adapters lazy-import
   their heavy libraries, so cold access typically measures only adapter
   construction; the library import cost appears in the first file load
   instead (see pyvista in the file-load numbers). Interpreter-level
   imports are only paid once per process.
2. Warm adapter access -- repeated get_adapter() calls hitting the
   instance cache.
3. Optional file-load timing -- loads a small generated OBJ file through
   each available adapter. Degrades gracefully to a skip message when no
   mesh library is installed or an adapter cannot read the file.

Usage (repo venv python, from the repo root or anywhere else -- the
script adds the repo root to sys.path itself):

    python benchmarks/bench_loader.py
"""

import sys
import tempfile
import time
import timeit
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mesh_issue_examiner.adapters.factory import (
    clear_adapter_cache,
    get_adapter,
)

CANDIDATE_LIBRARIES = ('trimesh', 'pyvista', 'meshio', 'pymeshlab')
WARM_REPEATS = 100

# Small tetrahedron surface written as plain OBJ text (no writer needed)
OBJ_TEXT = (
    "v 0 0 0\n"
    "v 1 0 0\n"
    "v 0 1 0\n"
    "v 0 0 1\n"
    "f 1 2 3\n"
    "f 1 2 4\n"
    "f 2 3 4\n"
    "f 1 3 4\n"
)


def bench_adapter_access():
    """Time cold (cache-cleared) vs warm get_adapter for each library.

    Returns:
        List of library names that were available and benchmarked.
    """
    print("Adapter access via get_adapter():")
    header = (f"{'library':>10} | {'cold [s]':>10} | {'warm [us]':>10} | "
              f"{'speedup':>9}")
    print(header)
    print('-' * len(header))

    available = []
    for library in CANDIDATE_LIBRARIES:
        try:
            clear_adapter_cache()
            start = time.perf_counter()
            get_adapter(library)
            cold = time.perf_counter() - start
        except Exception as exc:  # library not installed / not supported
            print(f"{library:>10} | skipped ({type(exc).__name__}: {exc})")
            continue

        timer = timeit.Timer(lambda: get_adapter(library))
        warm = min(timer.repeat(repeat=3, number=WARM_REPEATS)) / WARM_REPEATS
        print(f"{library:>10} | {cold:>10.4f} | {warm * 1e6:>10.3f} | "
              f"{cold / warm:>8.2e}")
        available.append(library)

    if not available:
        print("  -> no mesh adapter libraries available")
    return available


def bench_file_load(available):
    """Time loading a small OBJ file through each available adapter.

    Args:
        available: Library names returned by bench_adapter_access.
    """
    print("\nFile loading (small generated tetrahedron OBJ):")
    if not available:
        print("  skipped (no mesh library available)")
        return

    with tempfile.TemporaryDirectory() as tmp_dir:
        obj_path = Path(tmp_dir) / 'bench_tetra.obj'
        obj_path.write_text(OBJ_TEXT)

        any_loaded = False
        for library in available:
            try:
                adapter = get_adapter(library)
                start = time.perf_counter()
                native = adapter.load_mesh(str(obj_path))
                elapsed = time.perf_counter() - start

                # Prefer n_faces (pyvista flattens .faces into a
                # connectivity array, so len(.faces) overcounts there)
                n_faces = 'unknown'
                for attr in ('n_faces', 'faces'):
                    value = getattr(native, attr, None)
                    if value is None:
                        continue
                    try:
                        n_faces = int(value)
                        break
                    except (TypeError, ValueError):
                        try:  # array-like .faces (e.g. trimesh)
                            n_faces = int(len(value))
                            break
                        except TypeError:
                            pass

                print(f"  {library:>10}: {elapsed:.4f} s "
                      f"({n_faces} faces reported)")
                any_loaded = True
            except Exception as exc:
                print(f"  {library:>10}: skipped ({type(exc).__name__}: {exc})")

        if not any_loaded:
            print("  -> no adapter could load the file; nothing measured")


def main() -> None:
    """Run the adapter access and file loading benchmarks."""
    available = bench_adapter_access()
    bench_file_load(available)


if __name__ == '__main__':
    main()
