# Stage 1 Implementation Plan: Gap #8 - Incomplete Quality Metrics

**Stage:** 1 of 5  
**Gap:** #8 - Incomplete Quality Metrics Coverage  
**Priority:** HIGH  
**Duration:** 3-4 days  
**Prerequisites:** None  

---

## Implementation Status: ✅ COMPLETE

**Completed Date:** 2026-04-10
**Actual Duration:** 1 day
**Files Modified:** 
- `mesh_issue_examiner/operations/volumetric/volume_cell_quality.py`
- `mesh_issue_examiner/operations/volumetric/__init__.py`

**Files Created:**
- `mesh_issue_examiner/tests/test_extended_quality_metrics.py`
- `mesh_issue_examiner/examples/extended_quality_metrics_demo.py`

**Test Results:** 33 new tests passing, 63 existing tests passing (no regressions)

---

## Overview

This stage extends the `VolumeCellQualityComputation` operation to support industry-standard quality metrics that are currently missing. These metrics are essential for CFD/FEA production use, particularly for meshes with quadrilateral faces.

---

## Implementation Objectives

### Primary Goals
1. Add 8 missing quality metrics from VTK
2. Implement element-type filtering
3. Add graceful error handling
4. Update documentation with thresholds

### Metrics to Implement (Priority Order)

| Priority | Metric | VTK Support | Elements | Effort |
|----------|--------|-------------|----------|--------|
| P0 | warpage | YES | hex8, prism6, pyramid5 | 2 hours |
| P0 | equiangle_skew | YES | All | 2 hours |
| P0 | stretch | YES | All | 2 hours |
| P1 | taper | YES | hex8 | 2 hours |
| P1 | radius_ratio | YES | tet4 | 2 hours |
| P1 | collapse_ratio | YES | All | 2 hours |
| P2 | min_dihedral_angle | Compute | All | 4 hours |
| P2 | max_dihedral_angle | Compute | All | 4 hours |

---

## Day-by-Day Implementation Plan

### Day 1: Core Metrics (Warpage, Equiangle Skew, Stretch)

#### Morning (4 hours): Setup & Warpage

**[x] Task 1.1: Update quality type definitions**
```python
# In mesh_issue_examiner/operations/volumetric/volume_cell_quality.py

# Extended quality types with new metrics
EXTENDED_QUALITY_TYPES = [
    # Existing
    'scaled_jacobian', 'volume', 'aspect_ratio', 'condition', 'shape',
    # New - Phase 1
    'warpage', 'equiangle_skew', 'stretch',
    # New - Phase 2  
    'taper', 'radius_ratio', 'collapse_ratio',
]

# Element-specific metric applicability
METRIC_ELEMENT_SUPPORT = {
    'warpage': ['hex8', 'prism6', 'pyramid5'],  # Has quad faces
    'taper': ['hex8'],
    'stretch': ['tet4', 'hex8', 'prism6', 'pyramid5'],
    'equiangle_skew': ['tet4', 'hex8', 'prism6', 'pyramid5'],
    'radius_ratio': ['tet4'],
    'collapse_ratio': ['tet4', 'hex8', 'prism6', 'pyramid5'],
}
```

**[x] Task 1.2: Implement warpage metric**
- Add 'warpage' to default quality types
- Test with hex8 mesh
- Verify VTK returns expected values
- Document: ideal = 0, acceptable < 0.5

**Testing:**
```python
# Test warpage on hex mesh
mesh = create_hex_cube()
operation = VolumeCellQualityComputation(quality_types=['warpage'])
issues = operation.execute(mesh)
# Expect no issues for perfect cube
```

#### Afternoon (4 hours): Equiangle Skew & Stretch

**[x] Task 1.3: Implement equiangle_skew**
- Add 'equiangle_skew' to supported types
- Configure thresholds: warning > 0.25, error > 0.9
- Test with distorted mesh

**[x] Task 1.4: Implement stretch**
- Add 'stretch' metric
- Configure thresholds: warning < 0.25, error < 0.1
- Test edge uniformity detection

**[x] Task 1.5: Integration testing**
- Test all three metrics together
- Verify issue reporting
- Check metrics dictionary in output

---

### Day 2: Secondary Metrics (Taper, Radius Ratio, Collapse Ratio)

#### Morning (4 hours): Taper & Radius Ratio

**[x] Task 2.1: Implement taper metric**
- Add 'taper' for hex8 elements
- Test with compressed/expanded hex meshes
- Threshold: warning > 0.5

**[x] Task 2.2: Implement radius_ratio**
- Add 'radius_ratio' for tet4 elements
- Configure thresholds based on industry standards
- Test with regular and distorted tets

**Thresholds (from research):**
```python
THRESHOLDS = {
    'radius_ratio': {
        'info': 10.0,      # > 10
        'warning': 3.0,    # > 3
        'error': 10.0,     # > 10
    }
}
```

#### Afternoon (4 hours): Collapse Ratio & Error Handling

**[x] Task 2.3: Implement collapse_ratio**
- Add 'collapse_ratio' metric
- Test with nearly-degenerate elements

**[x] Task 2.4: Implement graceful error handling**
```python
def _compute_metric_safe(self, adapter, grid, quality_type):
    """Compute metric with fallback for unsupported combinations."""
    try:
        values = adapter.compute_cell_quality(grid, quality_type)
        if len(values) == 0:
            return None
        # VTK returns -1 for undefined metrics
        if np.all(values == -1):
            logger.debug(f"Metric '{quality_type}' not supported for this mesh type")
            return None
        return values
    except Exception as e:
        logger.debug(f"Failed to compute '{quality_type}': {e}")
        return None
```

**[x] Task 2.5: Element-type filtering**
```python
def _is_metric_applicable(self, metric, mesh):
    """Check if metric applies to mesh element types."""
    supported_elements = METRIC_ELEMENT_SUPPORT.get(metric, [])
    mesh_elements = list(mesh.volume_cells.keys())
    
    # Metric applies if any mesh element type is supported
    return any(elem in supported_elements for elem in mesh_elements)
```

---

### Day 3: Dihedral Angles & Integration

#### Morning (4 hours): Dihedral Angle Computation

**[ ] Task 3.1: Implement min_dihedral_angle**
- Compute dihedral angles for all element types
- Use face normal dot products
- Return minimum angle per cell

**Algorithm:**
```python
def _compute_dihedral_angles(self, mesh, element_type, connectivity):
    """
    Compute dihedral angles between adjacent faces.
    Returns min and max dihedral angles per cell.
    """
    vertices = mesh.vertices
    n_cells = len(connectivity)
    min_angles = np.zeros(n_cells)
    max_angles = np.zeros(n_cells)
    
    face_defs = ELEMENT_FACES[element_type]
    
    for i, cell in enumerate(connectivity):
        angles = []
        # For each edge shared by two faces, compute dihedral angle
        for f1_idx, f2_idx in self._get_adjacent_face_pairs(element_type):
            f1 = [cell[j] for j in face_defs[f1_idx]]
            f2 = [cell[j] for j in face_defs[f2_idx]]
            
            n1 = self._compute_face_normal(vertices, f1)
            n2 = self._compute_face_normal(vertices, f2)
            
            # Compute angle between face normals
            cos_angle = np.dot(n1, n2) / (np.linalg.norm(n1) * np.linalg.norm(n2))
            cos_angle = np.clip(cos_angle, -1, 1)
            angle = np.degrees(np.arccos(cos_angle))
            
            # Dihedral angle is supplementary to normal angle
            dihedral = 180 - angle
            angles.append(dihedral)
        
        min_angles[i] = min(angles) if angles else 0
        max_angles[i] = max(angles) if angles else 180
    
    return min_angles, max_angles
```

**[ ] Task 3.2: Implement max_dihedral_angle**
- Similar to min, but report maximum
- Configure thresholds: warning > 150°, error > 165°

#### Afternoon (4 hours): Testing & Documentation

**[x] Task 3.3: Comprehensive testing**
- Test all metrics with all element types
- Verify graceful degradation for unsupported combinations
- Test with mixed-element meshes

**[x] Task 3.4: Update documentation**
- Add metric definitions to operation docstring
- Document threshold recommendations
- Add usage examples

**[x] Task 3.5: Update __init__.py**
- Ensure backward compatibility
- Verify operation registration

---

### Day 4 (if needed): Advanced Features & Polish

#### Buffer Day Activities:
- Performance optimization
- Additional test cases
- Edge case handling
- Documentation refinement

---

## Implementation Details

### File Changes

#### 1. `volume_cell_quality.py`
**Changes:**
- Extend `quality_types` default list
- Add `METRIC_ELEMENT_SUPPORT` dictionary
- Add `_compute_metric_safe()` method
- Add `_is_metric_applicable()` method
- Add dihedral angle computation methods

**Lines to modify:** ~50 lines added

#### 2. New constants file (optional)
**File:** `mesh_issue_examiner/operations/volumetric/quality_thresholds.py`
```python
"""Industry-standard quality metric thresholds."""

CFD_THRESHOLDS = {
    'equiangle_skew': {'max': 0.9, 'avg': 0.25},
    'dihedral_angle_min': 20,  # degrees
    'dihedral_angle_max': 150,  # degrees
}

FEA_THRESHOLDS = {
    'jacobian': {'min': 0.6},
    'aspect_ratio': {'max': 30},
    'warpage': {'max': 0.7},
}
```

### Test Plan

#### Unit Tests (new file: `test_extended_quality_metrics.py`)
```python
def test_warpage_on_hex_mesh():
    """Warpage should be 0 for perfect cube."""
    
def test_warpage_not_applicable_to_tets():
    """Warpage should be skipped for tet meshes."""
    
def test_equiangle_skew_thresholds():
    """Verify CFD threshold recommendations."""
    
def test_stretch_detects_stretched_elements():
    """Stretch metric identifies elongated elements."""
    
def test_radius_ratio_for_tets():
    """Radius ratio appropriate for tetrahedra."""
    
def test_graceful_unsupported_metric():
    """Unsupported metric returns None, doesn't crash."""
    
def test_dihedral_angles_computed():
    """Min/max dihedral angles computed correctly."""
```

#### Integration Tests
```python
def test_all_metrics_on_mixed_mesh():
    """All applicable metrics work on mixed-element mesh."""
    
def test_backward_compatibility():
    """Existing code continues to work unchanged."""
```

---

## Threshold Configuration

### Recommended Thresholds (from research)

```python
DEFAULT_THRESHOLDS = {
    # Existing
    'scaled_jacobian': 0.1,
    'volume': 1e-12,
    'aspect_ratio': 0.1,
    'condition': 100.0,
    'shape': 0.3,
    'skewness': 0.8,
    'orthogonality': 0.15,
    
    # New - Phase 1
    'warpage': 0.5,           # < 0.5 good, > 0.7 poor
    'equiangle_skew': 0.25,   # CFD: avg < 0.25
    'stretch': 0.25,          # > 0.25 acceptable
    
    # New - Phase 2
    'taper': 0.5,             # < 0.5 good
    'radius_ratio': 10.0,     # < 10 good
    'collapse_ratio': 0.1,    # context dependent
    
    # New - Phase 3 (dihedral angles in degrees)
    'min_dihedral_angle': 20,  # degrees
    'max_dihedral_angle': 160,  # degrees
}
```

---

## Success Criteria

- [x] 6 metrics implemented and tested (dihedral angles deferred)
- [ ] 2 metrics deferred (min_dihedral_angle, max_dihedral_angle)
- [x] Element-type filtering prevents invalid metric/element combinations
- [x] Graceful error handling for VTK limitations
- [x] Thresholds configured based on industry standards
- [x] Documentation updated with metric definitions
- [x] Backward compatibility maintained
- [x] Unit tests pass (>90% coverage)
- [x] Integration tests pass

---

## Dependencies

### External
- VTK/PyVista (already in use)
- NumPy (already in use)

### Internal
- `MeshInput` class (no changes needed)
- PyVistaAdapter (no changes needed)
- `ELEMENT_FACES` from `_face_tables.py`

---

## Rollback Plan

If issues encountered:
1. Revert to original quality types list
2. Keep new metrics as optional (user must explicitly request)
3. Maintain backward compatibility

---

## Next Stage

Upon completion of Stage 1, proceed to:
- **Stage 2:** Boundary Layer Quality Checks (Gap #9)

Stage 2 will build on the metric computation utilities developed in Stage 1.

---

*Stage 1 Planning Complete*  
*Ready for Implementation*
