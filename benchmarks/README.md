# Benchmarks

Micro-benchmarks for `mesh_issue_examiner` hot paths. These are manual
scripts (not part of the pytest suite) and are safe to run at any time.

## Usage

Run with the repository virtual-environment Python. The scripts add the
repo root to `sys.path` themselves, so they work from any directory:

```powershell
& Z:\Workspace\mesh_exam\.venv\Scripts\python.exe benchmarks\bench_edge_face_map.py
& Z:\Workspace\mesh_exam\.venv\Scripts\python.exe benchmarks\bench_loader.py
```

(When working in a worktree/clone, run the copy inside that checkout so
its code is the code being measured.)

## bench_edge_face_map.py

Dependency-free (numpy only, no mesh libraries). Times the vectorized
triangle fast path of `mesh_issue_examiner.utils.topology.build_edge_face_map`
against the module-private loop reference `_build_edge_face_map_loop` on
deterministic synthetic grid meshes with ~1e4 / 1e5 / 1e6 triangles.
Correctness (vectorized == loop reference) is asserted for every size
before timing, and the best-of-3 time of each implementation is printed
as a table.

## bench_loader.py

Times cold vs warm adapter access via
`mesh_issue_examiner.adapters.factory.get_adapter`:

- **cold**: `clear_adapter_cache()` first, then a single `get_adapter()`
  call. Interpreter-level library imports are only paid once per process,
  so only the first candidate library's cold number includes its one-time
  module import.
- **warm**: repeated `get_adapter()` calls hitting the instance cache
  (min-of-mean over 100-call batches).

If at least one mesh library is installed, it also writes a small
tetrahedron OBJ to a temp directory and loads it through each available
adapter; adapters that are missing or cannot read the file are skipped
with a message, so the script degrades gracefully.
