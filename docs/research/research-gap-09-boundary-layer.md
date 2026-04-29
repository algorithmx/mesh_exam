# Research Report: Gap #9 - Boundary Layer Quality Checks

**Research Date:** 2026-04-10  
**Gap Status:** OPEN - Critical for Production CFD Use  
**Effort Estimate:** High (4-6 days)

---

## Executive Summary

Boundary layer meshes are critical for accurate CFD simulations, particularly for capturing viscous effects near walls. Currently, the framework has no specialized checks for anisotropic boundary layer meshes. This gap prevents validation of inflation layer meshes common in aerospace, automotive, and turbomachinery CFD.

---

## Background: What are Boundary Layer Meshes?

Boundary layer meshes (also called "inflation layers" or "prism layers") are thin layers of elements grown from wall surfaces to capture:
- Velocity gradients in viscous boundary layers
- Heat transfer near walls
- Turbulence effects in near-wall region

### Typical Boundary Layer Configuration
```
Wall Surface
    |
    | 1st layer (thinnest)  ← y+ target achieved here
    | 2nd layer
    | 3rd layer
    | ...
    | Nth layer (thickest)
    |
Core mesh (tets/hexes)
```

---

## Missing Checks - Research Findings

### 1. Layer Height Consistency
**Description:** Uniform first layer height for wall-function compatibility  
**Metric:** First layer thickness (y+)  
**Target:** y+ < 1 for low-Re, 30 < y+ < 300 for wall functions  
**Industry Practice:**
- CFD best practice: y+ ≈ 1 for resolving viscous sublayer
- Wall function meshes: 30 < y+ < 100 (sometimes up to 300)
- Simpleware Elite: "Boundary layer edge length ratio" metric

**Calculation:**
```
y+ = (u_tau × y) / ν
Where:
- u_tau = friction velocity
- y = first cell height
- ν = kinematic viscosity
```

**Research Findings:**
- Multiple MDPI papers on CFD simulations target y+ ≈ 1
- VAWT simulations (MDPI 2025): "First-layer mesh height set to 0.038 mm, ensuring y+ = 1"
- Hypersonic aircraft (MDPI 2020): "y+ between 5 and 30 according to turbulent boundary layer equations"
- AUV study (MDPI 2025): "First-layer mesh height set to 0.05 mm, ensuring a y+ value of approximately 6"

---

### 2. Layer Growth Ratio
**Description:** Consistent expansion between layers  
**Metric:** Height(i+1) / Height(i)  
**Target:** 1.1 - 1.3 (smooth growth)  
**Industry Standards:**
- Growth rate ≤ 1.2 for high-quality meshes
- Some applications accept up to 1.5
- Gas turbine study: "growth rate of 1.2" for 16-18 layers
- VAWT study: "growth rate of 1.1" for 10 prism layers

**Calculation:**
```
Growth Ratio = h_{i+1} / h_i

Cumulative height after n layers:
H_n = h_1 × (1 - r^n) / (1 - r)  [geometric series]
```

**Quality Thresholds:**
| Growth Ratio | Quality Assessment |
|--------------|-------------------|
| 1.0 - 1.2 | Excellent |
| 1.2 - 1.3 | Good |
| 1.3 - 1.5 | Acceptable |
| > 1.5 | Poor (may cause convergence issues) |

---

### 3. Aspect Ratio Limit
**Description:** Height vs. in-plane edge ratio  
**Metric:** Layer height / In-plane edge length  
**Target:** < 100-1000 (context dependent)  
**Industry Context:**
- High aspect ratios acceptable in boundary layers
- CFD-Online forum: "Aspect ratios up to 40000" for y+ < 1 requirements
- Trade-off: High AR needed for y+ compliance

**Calculation:**
```
Aspect Ratio = h_layer / L_inplane
```

---

### 4. Orthogonality
**Description:** Face normal alignment with wall normal  
**Metric:** Angle between face normal and wall normal  
**Target:** < 10-20° deviation  
**Industry Standards:**
- OpenFOAM: non-orthogonality < 65° (maximum)
- Good quality: < 45°
- Excellent: < 20°

---

### 5. Transition Quality
**Description:** Prism-to-tet interface quality  
**Metric:** Quality of cells at boundary layer termination  
**Target:** Match core mesh quality standards  
**Issues:**
- Sharp transitions cause solver instability
- Pyramids often used as transition elements

---

### 6. Layer Coverage
**Description:** Complete boundary coverage  
**Metric:** Percentage of wall surface covered by BL mesh  
**Target:** 100% (or documented exclusions)  
**Issues:**
- Gaps in boundary layer cause flow "leaks"
- Complex geometries may have uncovered regions

---

## Implementation Requirements

### Algorithm Overview

```python
def check_boundary_layer_quality(mesh, wall_faces):
    """
    1. Identify boundary layer cells (prisms near walls)
    2. Trace layers from wall outward
    3. Compute metrics for each layer
    4. Validate against thresholds
    """
    # Step 1: Identify wall-adjacent cells
    bl_cells = identify_boundary_layer_cells(mesh, wall_faces)
    
    # Step 2: Group cells by layer
    layers = trace_layers_outward(bl_cells, wall_faces)
    
    # Step 3: Compute metrics
    metrics = {
        'first_layer_height': compute_first_layer_heights(layers[0]),
        'growth_ratios': compute_growth_ratios(layers),
        'aspect_ratios': compute_aspect_ratios(layers),
        'orthogonality': compute_orthogonality(layers, wall_faces),
        'coverage': compute_coverage(layers, wall_faces)
    }
    
    # Step 4: Validate
    issues = validate_against_thresholds(metrics)
    
    return issues
```

### Key Algorithmic Challenges

1. **Boundary Layer Cell Identification**
   - Need to distinguish prism layers from core mesh
   - Heuristic: Prisms/wedges with one face on boundary
   - Alternative: User-provided cell tags

2. **Layer Tracing**
   - Follow prism stacks from wall outward
   - Handle layer termination and transitions
   - Track layer connectivity

3. **Metric Computation**
   - First layer height: Distance from wall to first cell centroid
   - Growth ratio: Height ratio between consecutive layers
   - Orthogonality: Angle between prism side faces and wall normal

---

## Implementation Plan

### Phase 1: Basic Layer Detection (Day 1-2)
- Implement boundary layer cell identification
- Implement layer tracing algorithm
- Handle prism6 and pyramid5 elements

### Phase 2: Metric Computation (Day 3-4)
- First layer height computation
- Growth ratio calculation
- Aspect ratio assessment

### Phase 3: Advanced Checks (Day 5-6)
- Orthogonality computation
- Transition quality assessment
- Coverage validation
- y+ estimation (requires flow velocity input)

---

## Configuration Schema

```python
BoundaryLayerQualityConfig = {
    'target_yplus': 1.0,           # Target y+ value
    'yplus_tolerance': 0.5,        # Acceptable deviation
    'max_growth_ratio': 1.2,       # Maximum layer growth
    'max_aspect_ratio': 1000,      # Maximum AR for BL cells
    'max_orthogonality_angle': 20, # Degrees
    'min_layers': 5,               # Minimum BL layer count
    'max_layers': 50,              # Maximum BL layer count
    'require_complete_coverage': True
}
```

---

## References

1. Simpleware Elite Datasheet - "Boundary layer edge length ratio (for CFD meshes only)"
2. MDPI Energies 2025 - "Optimization of Novel Variable-Channel-Width SOEC Design"
3. MDPI Fluids 2025 - "Numerical Investigation of VAWTs"
4. MDPI Fluids 2025 - "Optimized Hydrodynamic Design for AUVs"
5. CFD-Online Forum - "High aspect ratio and very low orthogonal quality mesh"
6. MDPI Processes 2026 - "Gas Turbine Data Scaling Technology"
7. OpenFOAM User Guide - Mesh quality recommendations

---

## Summary

Boundary layer quality checks require:
1. **Geometric analysis** of prism/wedge layers
2. **Layer tracing** from walls outward
3. **Multiple metrics** (height, growth, orthogonality)
4. **CFD-specific knowledge** (y+ targets, growth ratios)

This is a **high-effort** implementation due to algorithmic complexity and CFD domain expertise required.
