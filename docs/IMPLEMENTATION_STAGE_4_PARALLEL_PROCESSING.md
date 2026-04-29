# Stage 4 Implementation Plan: Gap #12 - Parallel Processing for Large Meshes

**Stage:** 4 of 5  
**Gap:** #12 - No Parallel Processing for Large Meshes  
**Priority:** MEDIUM  
**Duration:** 5-7 days  
**Prerequisites:** Stage 1 (stable metric implementations)  

---

## Overview

This stage implements parallel processing capabilities for handling large meshes (>10M cells). Currently all operations are single-threaded, which becomes a performance bottleneck for industrial-scale meshes.

---

## Background: Parallelization Options

### Option Comparison

| Method | Mechanism | Best For | Speedup | Complexity |
|--------|-----------|----------|---------|------------|
| Numba JIT | Compiled parallel loops | Hot numerical loops | 10-1000x | Low |
| multiprocessing | Process pools | Independent tasks | 2-8x | Medium |
| concurrent.futures | High-level pools | Mixed workloads | 2-8x | Low |
| NumPy vectorization | BLAS parallelism | Array operations | 2-4x | Lowest |

### Research Findings
- Numba JIT + parallel: Up to 1,000,000x speedup (best case)
- multiprocessing: Good for CPU-bound, high memory overhead
- NumPy: Always apply first (baseline optimization)

---

## Implementation Objectives

### Primary Goals
1. Implement Numba JIT optimizations for hot loops
2. Add multiprocessing for independent quality metrics
3. Benchmark and tune performance
4. Maintain backward compatibility

### Performance Targets

| Mesh Size | Current | Target | Speedup |
|-----------|---------|--------|---------|
| 100K cells | 2s | 1s | 2x |
| 1M cells | 20s | 5s | 4x |
| 10M cells | 200s | 30s | 6.7x |

---

## Day-by-Day Implementation Plan

### Day 1: Numba Integration Setup

#### Morning (4 hours): Environment & Dependencies

**Task 1.1: Add Numba dependency**
```
# requirements.txt
numba>=0.56.0
```

**Task 1.2: Create parallel utilities module**
```python
# mesh_issue_examiner/utils/parallel.py

"""Parallel processing utilities for mesh operations."""

import logging
from typing import Optional, Callable
import numpy as np

logger = logging.getLogger(__name__)

# Try to import numba, but don't fail if not available
try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    logger.info("Numba not available. Parallel JIT compilation disabled.")
    
    # Create dummy decorators
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    
    def prange(*args):
        return range(*args)


def maybe_jit(*args, **kwargs):
    """
    JIT decorator that falls back to plain function if numba unavailable.
    
    Usage:
        @maybe_jit(nopython=True, parallel=True)
        def my_function(arr):
            ...
    """
    if NUMBA_AVAILABLE:
        from numba import jit
        return jit(*args, **kwargs)
    else:
        def decorator(func):
            return func
        return decorator


def get_num_threads() -> int:
    """Get optimal number of threads."""
    import os
    return int(os.environ.get('MESH_EXAMINER_THREADS', 
                               os.cpu_count() or 1))
```

#### Afternoon (4 hours): Identify Hot Loops

**Task 1.3: Profile current operations**
```python
# profiling script to identify bottlenecks
import cProfile
import pstats

from mesh_issue_examiner.operations.volumetric.geometric_validity import (
    GeometricValidityAssessment
)

def profile_sliver_detection():
    """Profile sliver detection on test mesh."""
    # Load test mesh
    mesh = load_large_test_mesh()
    
    # Profile
    profiler = cProfile.Profile()
    profiler.enable()
    
    operation = GeometricValidityAssessment()
    issues = operation.execute(mesh)
    
    profiler.disable()
    
    # Print stats
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(20)
```

**Task 1.4: Document hot functions**
- `_detect_sliver_tets()` - Edge length computation
- `_detect_sliver_hexes()` - Edge length computation  
- `compute_edge_lengths()` - Common bottleneck
- Face adjacency building - O(n) with nested loops

---

### Day 2: Numba JIT for Hot Loops

#### Morning (4 hours): Edge Length Computation

**Task 2.1: JIT-compiled edge length function**
```python
# In mesh_issue_examiner/utils/parallel.py

@maybe_jit(nopython=True, parallel=True, cache=True)
def compute_edge_lengths_parallel(vertices, edges):
    """
    Compute edge lengths in parallel.
    
    Parameters
    ----------
    vertices : (n, 3) array
    edges : (m, 2) array of vertex indices
    
    Returns
    -------
    lengths : (m,) array of edge lengths
    """
    n_edges = edges.shape[0]
    lengths = np.empty(n_edges)
    
    for i in prange(n_edges):
        v1_idx = edges[i, 0]
        v2_idx = edges[i, 1]
        
        dx = vertices[v2_idx, 0] - vertices[v1_idx, 0]
        dy = vertices[v2_idx, 1] - vertices[v1_idx, 1]
        dz = vertices[v2_idx, 2] - vertices[v1_idx, 2]
        
        lengths[i] = np.sqrt(dx*dx + dy*dy + dz*dz)
    
    return lengths
```

**Task 2.2: Update sliver detection to use JIT**
```python
def _detect_sliver_tets(self, tet4_conn, vertices, volumes, 
                        degenerate_mask, tet_offset):
    """Updated to use parallel edge length computation."""
    
    n_tets = len(tet4_conn)
    tet_volumes = volumes[tet_offset:tet_offset + n_tets]
    tet_degenerate = degenerate_mask[tet_offset:tet_offset + n_tets]
    
    # Build edge list for all tets
    edges = self._build_tet_edges(tet4_conn)
    
    # Compute edge lengths in parallel
    edge_lengths = compute_edge_lengths_parallel(vertices, edges)
    
    # Reshape and find max per tet
    edge_lengths_per_tet = edge_lengths.reshape(n_tets, 6)
    max_edge = np.max(edge_lengths_per_tet, axis=1)
    
    # Compute sliver ratio (rest of logic unchanged)
    ...
```

#### Afternoon (4 hours): Face Normal Computation

**Task 2.3: JIT-compiled face normal function**
```python
@maybe_jit(nopython=True, parallel=True, cache=True)
def compute_face_normals_parallel(vertices, faces):
    """
    Compute face normals in parallel.
    
    Parameters
    ----------
    vertices : (n, 3) array
    faces : (m, 3) array (triangular faces)
    
    Returns
    -------
    normals : (m, 3) array of normalized face normals
    """
    n_faces = faces.shape[0]
    normals = np.empty((n_faces, 3))
    
    for i in prange(n_faces):
        v0 = faces[i, 0]
        v1 = faces[i, 1]
        v2 = faces[i, 2]
        
        # Edge vectors
        e1x = vertices[v1, 0] - vertices[v0, 0]
        e1y = vertices[v1, 1] - vertices[v0, 1]
        e1z = vertices[v1, 2] - vertices[v0, 2]
        
        e2x = vertices[v2, 0] - vertices[v0, 0]
        e2y = vertices[v2, 1] - vertices[v0, 1]
        e2z = vertices[v2, 2] - vertices[v0, 2]
        
        # Cross product
        nx = e1y * e2z - e1z * e2y
        ny = e1z * e2x - e1x * e2z
        nz = e1x * e2y - e1y * e2x
        
        # Normalize
        norm = np.sqrt(nx*nx + ny*ny + nz*nz)
        if norm > 0:
            normals[i, 0] = nx / norm
            normals[i, 1] = ny / norm
            normals[i, 2] = nz / norm
        else:
            normals[i, 0] = 0.0
            normals[i, 1] = 0.0
            normals[i, 2] = 0.0
    
    return normals
```

---

### Day 3: Multiprocessing for Quality Metrics

#### Morning (4 hours): Parallel Metric Computation

**Task 3.1: Implement parallel quality computation**
```python
# In mesh_issue_examiner/operations/volumetric/volume_cell_quality.py

from concurrent.futures import ProcessPoolExecutor, as_completed
import os

def compute_quality_metrics_parallel(adapter, grid, quality_types, 
                                      max_workers=None):
    """
    Compute multiple quality metrics in parallel.
    
    Each metric is computed independently, so we can parallelize
    across metrics (coarse-grained parallelism).
    """
    if max_workers is None:
        max_workers = min(len(quality_types), os.cpu_count() or 1)
    
    # Use single worker for small meshes
    if grid.n_cells < 100000 or max_workers == 1:
        return compute_quality_metrics_sequential(adapter, grid, quality_types)
    
    results = {}
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all metric computations
        futures = {
            executor.submit(
                compute_single_metric, adapter, grid, qt
            ): qt for qt in quality_types
        }
        
        # Collect results as they complete
        for future in as_completed(futures):
            qt = futures[future]
            try:
                results[qt] = future.result()
            except Exception as e:
                logger.warning(f"Failed to compute {qt}: {e}")
                results[qt] = None
    
    return results


def compute_single_metric(adapter, grid, quality_type):
    """Compute a single quality metric (for multiprocessing)."""
    try:
        return adapter.compute_cell_quality(grid, quality_type)
    except Exception as e:
        logger.debug(f"Metric {quality_type} failed: {e}")
        return None
```

#### Afternoon (4 hours): Batch Processing

**Task 3.2: Implement batch processing for large meshes**
```python
def process_large_mesh_in_batches(mesh, operation, batch_size=100000):
    """
    Process large mesh in batches to control memory usage.
    
    Useful for operations that don't fit in memory.
    """
    total_cells = sum(len(c) for c in mesh.volume_cells.values())
    
    if total_cells <= batch_size:
        return operation.execute(mesh)
    
    all_issues = []
    
    # Process each element type separately
    for element_type, connectivity in mesh.volume_cells.items():
        n_cells = len(connectivity)
        n_batches = (n_cells + batch_size - 1) // batch_size
        
        for batch_idx in range(n_batches):
            start = batch_idx * batch_size
            end = min((batch_idx + 1) * batch_size, n_cells)
            
            # Create batch mesh
            batch_mesh = MeshInput(
                vertices=mesh.vertices,
                volume_cells={element_type: connectivity[start:end]}
            )
            
            # Process batch
            batch_issues = operation.execute(batch_mesh)
            
            # Adjust indices
            for issue in batch_issues:
                for loc in issue.locations:
                    if loc.element_index >= 0:
                        loc.element_index += start
            
            all_issues.extend(batch_issues)
    
    return all_issues
```

---

### Day 4: Integration & Configuration

#### Morning (4 hours): Operation Updates

**Task 4.1: Add parallel configuration to operations**
```python
class VolumeCellQualityComputation(BaseOperation):
    """Updated with parallel support."""
    
    def __init__(
        self,
        config: Optional[OperationConfig] = None,
        quality_types: Optional[List[str]] = None,
        parallel: bool = True,
        max_workers: Optional[int] = None
    ):
        super().__init__(config)
        self.quality_types = quality_types or [...]
        self.parallel = parallel
        self.max_workers = max_workers
    
    def execute(self, mesh: MeshInput) -> List[MeshIssue]:
        # ... existing setup ...
        
        if self.parallel and len(self.quality_types) > 1:
            # Use parallel metric computation
            metric_results = compute_quality_metrics_parallel(
                adapter, grid, self.quality_types, self.max_workers
            )
        else:
            # Sequential computation
            metric_results = {
                qt: adapter.compute_cell_quality(grid, qt)
                for qt in self.quality_types
            }
        
        # Process results...
```

**Task 4.2: Add global configuration**
```python
# mesh_issue_examiner/config.py

class ParallelConfig:
    """Global parallel processing configuration."""
    
    ENABLE_PARALLEL = True
    DEFAULT_MAX_WORKERS = None  # None = auto (all cores)
    BATCH_SIZE = 100000  # Cells per batch for large meshes
    USE_NUMBA_JIT = True
    NUMBA_PARALLEL = True
    
    @classmethod
    def get_max_workers(cls):
        if cls.DEFAULT_MAX_WORKERS is None:
            return os.cpu_count() or 1
        return cls.DEFAULT_MAX_WORKERS
```

#### Afternoon (4 hours): Testing

**Task 4.3: Create performance tests**
```python
def test_parallel_speedup():
    """Verify parallel implementation is faster."""
    mesh = create_1m_cell_mesh()
    
    # Sequential
    op_seq = VolumeCellQualityComputation(parallel=False)
    t1 = time.time()
    op_seq.execute(mesh)
    t_seq = time.time() - t1
    
    # Parallel
    op_par = VolumeCellQualityComputation(parallel=True)
    t1 = time.time()
    op_par.execute(mesh)
    t_par = time.time() - t1
    
    speedup = t_seq / t_par
    assert speedup > 1.5, f"Parallel speedup too low: {speedup:.2f}x"


def test_numba_jit_compiles():
    """Verify Numba JIT compilation works."""
    from mesh_issue_examiner.utils.parallel import NUMBA_AVAILABLE
    
    if NUMBA_AVAILABLE:
        # Test JIT function
        vertices = np.random.rand(1000, 3)
        edges = np.random.randint(0, 1000, size=(500, 2))
        
        result = compute_edge_lengths_parallel(vertices, edges)
        assert len(result) == 500
```

---

### Day 5-7: Benchmarking & Optimization

#### Day 5: Benchmarking
- Create benchmark suite
- Test with various mesh sizes
- Profile memory usage
- Document results

#### Day 6: Optimization
- Optimize based on benchmark results
- Tune batch sizes
- Optimize Numba functions
- Reduce memory allocations

#### Day 7: Documentation
- Update operation documentation
- Add parallel configuration guide
- Document performance expectations
- Add troubleshooting section

---

## Success Criteria

- [ ] Numba JIT functions implemented for hot loops
- [ ] Multiprocessing for quality metrics working
- [ ] 4x speedup on 1M cell mesh
- [ ] 6x speedup on 10M cell mesh
- [ ] Memory usage < 2x sequential
- [ ] Backward compatibility maintained
- [ ] Configuration options documented
- [ ] Benchmark results documented

---

## Rollback Plan

If parallel implementation causes issues:
1. Set `ENABLE_PARALLEL = False` by default
2. Keep parallel code as optional feature
3. Document manual activation

---

*Stage 4 Planning Complete*
