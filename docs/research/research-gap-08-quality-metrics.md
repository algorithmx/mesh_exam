# Research Report: Gap #8 - Incomplete Quality Metrics Coverage

**Research Date:** 2026-04-10  
**Gap Status:** OPEN - Critical for Production Use  
**Effort Estimate:** Medium (3-4 days)

---

## Executive Summary

The current volumetric quality metrics implementation is limited to those directly supported by VTK's `cell_quality` filter. While basic metrics (scaled_jacobian, volume, aspect_ratio, condition, skew) are available, several critical quality metrics for CFD/FEA applications are missing, particularly for quadrilateral faces and specialized element quality assessment.

---

## Current Implementation Status

### Currently Supported Metrics (via VTK)
- `scaled_jacobian` - Scaled Jacobian determinant
- `volume` - Cell volume
- `aspect_ratio` - Aspect ratio
- `condition` - Condition number
- `skew` / `skewness` - Deviation from ideal shape
- `orthogonality` - Face normal orthogonality

**Source:** `mesh_issue_examiner/operations/volumetric/volume_cell_quality.py`

---

## Missing Critical Metrics

### 1. Warpage
**Description:** Deviation of quadrilateral faces from planarity  
**Applicable Elements:** hex8, prism6, pyramid5 (elements with quad faces)  
**Use Case:** Critical for quad face quality in CFD meshes  
**Threshold:** Ideal = 0, Acceptable < 0.5-0.7  
**VTK Support:** YES - `vtkMeshQuality::QuadWarpage()`  

**Mathematical Definition:**
```
Warpage = cosine of minimum dihedral angle formed by planes intersecting in diagonals
        = cos(θ_dihedral_min)
```

**Implementation Notes:**
- VTK provides `SetQuadQualityMeasureToWarpage()`
- Warpage measures non-planarity of quadrilateral faces
- Zero warpage indicates perfectly planar quad face
- High warpage indicates twisted/distorted quad faces

**References:**
- VTK Documentation: `vtkMeshQuality::QuadWarpage()` 
- ANSYS Mesh Metrics: "Warping Factor" - ideal value 0

---

### 2. Taper
**Description:** Ratio of face areas on opposite sides of a hexahedron  
**Applicable Elements:** hex8 (primarily)  
**Use Case:** Hex distortion detection  
**Threshold:** Ideal = 0, Acceptable < 0.5  
**VTK Support:** YES - `vtkMeshQuality::QuadTaper()`  

**Mathematical Definition:**
```
Taper = |A1 - A2| / (A1 + A2) for opposite faces
```

---

### 3. Stretch
**Description:** Ratio of minimum edge length to maximum diagonal length  
**Applicable Elements:** All element types  
**Use Case:** Edge uniformity assessment  
**Threshold:** Ideal = 1, Acceptable > 0.25-0.5  
**VTK Support:** YES - `vtkMeshQuality::QuadStretch()`  

---

### 4. Equiangle Skew
**Description:** Angular deviation from ideal element angles  
**Applicable Elements:** All element types  
**Use Case:** Shape quality assessment  
**Threshold:** Ideal = 0, Acceptable < 0.25-0.9  
**VTK Support:** YES - `vtkMeshQuality::EQUIANGLE_SKEW`  

**Mathematical Definition:**
```
Q_eas = max((θ_max - θ_e)/(180 - θ_e), (θ_e - θ_min)/θ_e)
```

**CFD Best Practices:**
- Maximum threshold: 0.9
- Average threshold: 0.25

---

### 5. Dihedral Angle
**Description:** Angle between adjacent faces  
**Applicable Elements:** All element types  
**Use Case:** Mesh smoothness assessment  
**Threshold:** Typically 20°-150° for tets  
**VTK Support:** Partial (can compute but not as standard metric)  

---

### 6. Radius Ratio
**Description:** Ratio of circumradius to inradius  
**Applicable Elements:** tet4 (primary)  
**Use Case:** Tet quality assessment  
**Threshold:** Ideal = 1, Acceptable < 3-10  
**VTK Support:** YES - `vtkMeshQuality::RADIUS_RATIO`  

---

## VTK Implementation Guide

PyVista exposes VTK's `cell_quality` method which supports:
```python
grid.cell_quality(quality_measure='warpage')
grid.cell_quality(quality_measure='taper')
grid.cell_quality(quality_measure='stretch')
grid.cell_quality(quality_measure='equiangle_skew')
```

---

## Implementation Plan

1. **Phase 1 (Days 1-2):** Add warpage, equiangle_skew, stretch
2. **Phase 2 (Days 3-4):** Add taper, radius_ratio, collapse_ratio
3. **Add element-type filtering** - Not all metrics apply to all elements
4. **Add graceful error handling** - VTK may not support all metrics for all elements

---

## References

1. VTK Documentation: `vtkMeshQuality` class
2. ANSYS Mesh Metrics Explained
3. SimScale Mesh Quality Documentation
4. FEA Academy: Fundamentals of Mesh Quality
