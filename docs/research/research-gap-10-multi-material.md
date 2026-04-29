# Research Report: Gap #10 - Multi-Material Interface Checks

**Research Date:** 2026-04-10  
**Gap Status:** OPEN - Medium Priority for Multi-Physics  
**Effort Estimate:** Medium-High (3-5 days)

---

## Executive Summary

Multi-material meshes are essential for simulations involving:
- Composite materials
- Fluid-structure interaction
- Thermal interface analysis
- Bi-material systems

Currently, the framework has no support for detecting or validating interfaces between different materials or regions.

---

## Background: Multi-Material Meshes

### Definition
A multi-material mesh contains distinct regions (materials) that may have:
- Different material properties
- Independent mesh resolution
- Shared interface boundaries

### Common Use Cases

1. **Composite Materials**
   - Fiber-reinforced polymers
   - Layered structures
   - Sandwich panels

2. **Fluid-Structure Interaction (FSI)**
   - Blood flow in arteries
   - Airflow over wings
   - Thermal-fluid systems

3. **Thermal Analysis**
   - Heat sinks with different materials
   - Electronic component assemblies
   - Thermal protection systems

4. **Geological/Medical**
   - Rock layers
   - Bone/tissue interfaces
   - Brain segmentation

---

## Missing Functionality - Research Findings

### 1. Material Assignment
**Description:** Associate cells with material IDs  
**Current State:** `MeshInput` has no material support  
**Required Changes:**
```python
class MeshInput:
    # Current
    volume_cells: Dict[str, np.ndarray]  # element_type -> connectivity
    
    # Needed
    cell_materials: Optional[np.ndarray] = None  # material_id per cell
    material_names: Optional[Dict[int, str]] = None  # id -> name mapping
```

**Reference:** Simpleware FE "Multipart Volumetric Meshing" - supports "part-wise mesh generation with smooth, conforming interfaces at boundaries"

---

### 2. Interface Detection
**Description:** Identify faces between materials  
**Algorithm:**
```python
def detect_material_interfaces(mesh):
    """
    1. Build face-to-cell adjacency
    2. For each internal face, check if adjacent cells have different materials
    3. Return list of interface faces grouped by material pair
    """
    face_to_cells = build_face_to_cell_map(mesh)
    
    interfaces = defaultdict(list)
    for face_key, cells in face_to_cells.items():
        if len(cells) == 2:
            mat1 = mesh.cell_materials[cells[0]]
            mat2 = mesh.cell_materials[cells[1]]
            if mat1 != mat2:
                pair = tuple(sorted([mat1, mat2]))
                interfaces[pair].append(face_key)
    
    return interfaces
```

---

### 3. Interface Quality
**Description:** Check quality of transition cells  
**Metrics:**
- Cell quality at interface
- Face quality of interface faces
- Aspect ratio of transition elements

**Reference:** Simpleware FE supports "Part-wise mesh generation with smooth, conforming interfaces at boundaries"

---

### 4. Zero-Thickness Detection
**Description:** Detect cells at material boundaries  
**Problem:** Elements with one face on interface may have poor quality  
**Metric:** Distance from cell centroid to interface face  

---

### 5. Region Statistics
**Description:** Per-material quality metrics  
**Output:**
```python
{
    'material_1': {
        'cell_count': 1000,
        'avg_quality': 0.85,
        'min_quality': 0.23,
        'volume': 1.5e-3
    },
    'material_2': { ... }
}
```

---

## Research: Interface Modeling Approaches

### Perfect Interface
- Continuity of displacement/temperature
- No special elements needed
- Standard FEA formulation

### Imperfect Interface
- Discontinuity allowed
- Special interface elements
- Cohesive zone models

**Reference:** "Modeling Imperfect Interfaces in the Material Point Method" (Nairn et al.)
- Extends multimaterial MPM to handle imperfect interfaces
- Displacement discontinuities at interfaces
- Interfacial traction laws

---

## File Format Support

### Formats Supporting Multi-Material

1. **NASTRAN (.nas/.bdf)**
   - PSOLID entries for material properties
   - MAT1, MAT8, MAT10 for material definitions

2. **Abaqus (.inp)**
   - *MATERIAL sections
   - *SOLID SECTION with material assignment

3. **Exodus (.exo)**
   - Element blocks with IDs
   - Side sets for interfaces

4. **CGNS (.cgns)**
   - Zone definitions for regions
   - Family trees for material groups

5. **Gmsh (.msh)**
   - Physical groups for materials
   - $PhysicalNames section

---

## Implementation Plan

### Phase 1: Data Model Extension (Day 1)
- Extend `MeshInput` with material support
- Update `cell_types.py` for material ID handling
- Add material reading to mesh loaders

### Phase 2: Interface Detection (Day 2-3)
- Implement face-to-cell adjacency building
- Material interface identification
- Interface face extraction

### Phase 3: Quality Assessment (Day 4)
- Interface cell quality metrics
- Per-material statistics
- Interface face quality checks

### Phase 4: Reporting (Day 5)
- Interface visualization support
- Per-material quality reports
- Export interface faces

---

## Configuration Schema

```python
MultiMaterialConfig = {
    'material_properties': {
        1: {'name': 'steel', 'density': 7850},
        2: {'name': 'aluminum', 'density': 2700}
    },
    'interface_checks': {
        'check_perfect_interfaces': True,
        'min_interface_quality': 0.1,
        'report_per_material_stats': True
    }
}
```

---

## References

1. Simpleware FE Datasheet - "Multipart Volumetric Meshing"
2. "Modeling Imperfect Interfaces in the Material Point Method" - Nairn et al.
3. FEA Academy - "Thermal Analysis" section on material interfaces
4. Marine FEA Guidelines - "Heterogeneous interfaces" refinement
5. COMSOL Multiphysics - Multi-material mesh documentation

---

## Summary

Multi-material support requires:
1. **Data model extension** for material IDs
2. **Interface detection** algorithms
3. **Quality metrics** for transition regions
4. **File format support** for material data

This is **medium-high effort** due to data model changes and multiple file format integrations.
