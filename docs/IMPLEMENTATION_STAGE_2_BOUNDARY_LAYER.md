# Stage 2 Implementation Plan: Gap #9 - Boundary Layer Quality Checks

**Stage:** 2 of 5  
**Gap:** #9 - No Boundary Layer Quality Checks  
**Priority:** HIGH  
**Duration:** 4-6 days  
**Prerequisites:** Stage 1 (metric computation utilities)  

---

## Implementation Status: ✅ COMPLETE

**Completed Date:** 2026-04-10
**Actual Duration:** ~4 days (estimated)
**Implementation File:** `mesh_issue_examiner/operations/volumetric/boundary_layer_quality.py` (930 lines)
**Operation ID:** `quality_3d.check_boundary_layer`

**Features Implemented:**
- [x] Layer count validation (min_layers check)
- [x] First layer height check (y+ proxy)
- [x] Growth ratio validation between layers
- [x] Aspect ratio check for BL cells
- [x] Orthogonality check (alignment with wall normal)
- [x] Coverage check (fraction of wall covered)

---

## Overview

This stage implements specialized checks for anisotropic boundary layer (inflation layer) meshes common in CFD. Boundary layers are thin layers of prismatic elements grown from wall surfaces to capture viscous effects.

---

## Background: Boundary Layer Mesh Structure

### Anatomy of a Boundary Layer Mesh

```
Wall Surface (boundary face)
    |
    | ← First layer (thinnest) - y+ target achieved here
    | ← Second layer
    | ← Third layer
    | ...
    | ← Nth layer (thickest)
    |
Core mesh (tetrahedra/hexahedra)
```

### Element Types in Boundary Layers
- **Prisms (prism6):** Most common - triangular extrusion
- **Hexahedra (hex8):** Structured boundary layers
- **Pyramids:** Transition elements

### Key Quality Metrics
1. **First layer height** - Determines y+ compliance
2. **Growth ratio** - Smoothness of layer expansion
3. **Aspect ratio** - Height vs. in-plane dimensions
4. **Orthogonality** - Alignment with wall normal

---

## Implementation Objectives

### Primary Goals
1. Identify boundary layer cells in volumetric mesh
2. Trace layers from wall outward
3. Compute BL-specific quality metrics
4. Validate against CFD best practices

### Metrics to Implement

| Metric | Description | Target | Complexity |
|--------|-------------|--------|------------|
| First Layer Height | Distance from wall to first cell | y+ dependent | Medium |
| Growth Ratio | Height ratio between layers | 1.1-1.3 | Medium |
| Aspect Ratio | Height / in-plane edge | < 1000 | Low |
| Orthogonality | Angle to wall normal | < 20° | High |
| Layer Coverage | % of wall covered by BL | 100% | Medium |
| Transition Quality | Interface to core mesh | Match core | High |

---

## Day-by-Day Implementation Plan

### Day 1: Boundary Layer Cell Identification

#### Morning (4 hours): Algorithm Design ✅

**Task 1.1: Define BL cell detection heuristic**
```python
def identify_boundary_layer_cells(mesh, wall_faces):
    """
    Identify cells that are part of boundary layers.
    
    Heuristic: Cells with one face on wall boundary
               AND element is prism/wedge/hex
    
    Returns:
        Dict mapping layer index to list of cell indices
    """
    bl_cells = []
    
    for element_type, connectivity in mesh.volume_cells.items():
        if element_type not in ['prism6', 'hex8', 'pyramid5']:
            continue  # BL cells are prisms/hexes
            
        for cell_idx, cell in enumerate(connectivity):
            # Check if any face of this cell is on wall
            faces = extract_faces(cell, element_type)
            for face in faces:
                if face_on_boundary(face, wall_faces):
                    bl_cells.append({
                        'global_idx': global_idx,
                        'element_type': element_type,
                        'local_idx': cell_idx,
                        'wall_face': face
                    })
                    break
    
    return bl_cells
```

**Task 1.2: Implement face-to-cell adjacency**
```python
def build_face_to_cell_map(mesh):
    """Build mapping from faces to cells containing them."""
    face_to_cells = {}
    
    global_idx = 0
    for element_type, connectivity in mesh.volume_cells.items():
        faces = ELEMENT_FACES[element_type]
        
        for cell_idx, cell in enumerate(connectivity):
            for face_def in faces:
                face_verts = tuple(sorted(cell[i] for i in face_def))
                
                if face_verts not in face_to_cells:
                    face_to_cells[face_verts] = []
                face_to_cells[face_verts].append({
                    'global_idx': global_idx + cell_idx,
                    'element_type': element_type,
                    'local_idx': cell_idx
                })
        
        global_idx += len(connectivity)
    
    return face_to_cells
```

#### Afternoon (4 hours): BL Detection Implementation ✅

**Task 1.3: Implement wall face extraction**
```python
def extract_wall_faces(mesh):
    """
    Extract boundary faces that are walls.
    
    For now, assume all boundary faces are walls.
    Future: Support tagged boundaries.
    """
    if mesh.faces is None or len(mesh.faces) == 0:
        return []
    
    # For surface mesh, faces are the boundary
    # For volumetric mesh, extract surface boundary
    from ...adapters.factory import get_adapter
    adapter = get_adapter('pyvista')
    grid = mesh.get_native('pyvista')
    
    # Extract boundary surface
    surface = grid.extract_surface()
    boundary_faces = adapter.get_faces(surface)
    
    return boundary_faces
```

**Task 1.4: Test BL cell identification**
- Create test mesh with known boundary layer
- Verify correct cells identified
- Test with prism layers
- Test with hex layers

---

### Day 2: Layer Tracing Algorithm

#### Morning (4 hours): Layer Tracing Core ✅

**Task 2.1: Implement outward layer tracing**
```python
def trace_boundary_layers(mesh, bl_cells, wall_faces, max_layers=50):
    """
    Trace boundary layer cells from wall outward.
    
    Algorithm:
    1. Start with layer 0 = cells adjacent to wall
    2. For each layer, find cells sharing non-wall faces
    3. Continue until no more BL cells or max_layers reached
    
    Returns:
        List of layers, each layer is list of cell dicts
    """
    face_to_cells = build_face_to_cell_map(mesh)
    
    layers = [[] for _ in range(max_layers)]
    visited = set()
    
    # Layer 0: cells with faces on wall
    for cell_info in bl_cells:
        layer_idx = 0
        layers[layer_idx].append(cell_info)
        visited.add(cell_info['global_idx'])
    
    # Trace outward
    for layer_idx in range(max_layers - 1):
        current_layer = layers[layer_idx]
        
        for cell_info in current_layer:
            # Get faces of this cell
            cell = get_cell_connectivity(mesh, cell_info)
            faces = extract_faces(cell, cell_info['element_type'])
            
            # For each non-wall face, find adjacent cell
            for face in faces:
                if face_on_boundary(face, wall_faces):
                    continue  # Skip wall faces
                
                face_key = tuple(sorted(face))
                adjacent = face_to_cells.get(face_key, [])
                
                for adj_cell in adjacent:
                    if adj_cell['global_idx'] == cell_info['global_idx']:
                        continue  # Skip self
                    
                    if adj_cell['global_idx'] not in visited:
                        # Check if adjacent cell is also BL type
                        if adj_cell['element_type'] in ['prism6', 'hex8']:
                            layers[layer_idx + 1].append(adj_cell)
                            visited.add(adj_cell['global_idx'])
        
        if not layers[layer_idx + 1]:
            break  # No more layers
    
    # Remove empty layers
    return [l for l in layers if l]
```

#### Afternoon (4 hours): Height Computation ✅

**Task 2.2: Implement layer height computation**
```python
def compute_layer_heights(mesh, layers, wall_vertices):
    """
    Compute height of each layer.
    
    Height = distance from wall to cell centroid
    """
    layer_heights = []
    
    for layer_idx, layer_cells in enumerate(layers):
        heights = []
        
        for cell_info in layer_cells:
            # Get cell vertices
            vertices = get_cell_vertices(mesh, cell_info)
            centroid = np.mean(vertices, axis=0)
            
            # Find nearest wall point
            if layer_idx == 0:
                # First layer: distance to wall
                wall_dist = min_distance_to_surface(centroid, wall_vertices)
            else:
                # Other layers: distance from previous layer
                # Approximate as distance from wall
                prev_layer_cells = layers[layer_idx - 1]
                prev_centroids = [np.mean(get_cell_vertices(mesh, c), axis=0) 
                                  for c in prev_layer_cells]
                wall_dist = min(np.linalg.norm(centroid - pc) 
                               for pc in prev_centroids)
            
            heights.append(wall_dist)
        
        layer_heights.append(np.mean(heights) if heights else 0)
    
    return layer_heights
```

**Task 2.3: Test layer tracing**
- Test with 5-layer prism mesh
- Verify correct layer assignment
- Test with varying layer count
- Handle incomplete layers

---

### Day 3: Quality Metrics Implementation

#### Morning (4 hours): First Layer Height & Growth Ratio ✅

**Task 3.1: Implement first layer height check**
```python
def check_first_layer_height(layer_heights, target_yplus, tolerance=0.5):
    """
    Check if first layer height meets y+ target.
    
    Note: Actual y+ requires flow velocity, which we don't have.
    Report height as proxy metric.
    """
    if not layer_heights:
        return None
    
    first_height = layer_heights[0]
    
    # Report the height; user must convert to y+
    return {
        'first_layer_height': first_height,
        'target_yplus_proxy': target_yplus,
        'status': 'info'  # Cannot determine pass/fail without flow
    }
```

**Task 3.2: Implement growth ratio check**
```python
def check_growth_ratios(layer_heights, max_growth_ratio=1.2):
    """
    Check layer-to-layer growth ratios.
    
    growth_ratio[i] = height[i+1] / height[i]
    
    Returns list of violations where ratio > max_growth_ratio.
    """
    violations = []
    
    for i in range(len(layer_heights) - 1):
        ratio = layer_heights[i + 1] / layer_heights[i] \
                if layer_heights[i] > 0 else 0
        
        if ratio > max_growth_ratio:
            violations.append({
                'layer_pair': (i, i + 1),
                'ratio': ratio,
                'max_allowed': max_growth_ratio
            })
    
    return violations
```

#### Afternoon (4 hours): Aspect Ratio & Orthogonality ✅

**Task 3.3: Implement aspect ratio check**
```python
def compute_bl_aspect_ratios(mesh, layers):
    """
    Compute aspect ratios for BL cells.
    
    AR = layer_height / in_plane_edge_length
    """
    aspect_ratios = []
    
    for layer_idx, layer_cells in enumerate(layers):
        for cell_info in layer_cells:
            vertices = get_cell_vertices(mesh, cell_info)
            
            # Compute cell height (wall-normal direction)
            height = compute_cell_height(vertices)
            
            # Compute in-plane edge length
            inplane_length = compute_inplane_edge_length(vertices)
            
            ar = height / inplane_length if inplane_length > 0 else 0
            aspect_ratios.append({
                'cell_idx': cell_info['global_idx'],
                'layer': layer_idx,
                'aspect_ratio': ar
            })
    
    return aspect_ratios
```

**Task 3.4: Implement orthogonality check**
```python
def compute_orthogonality(mesh, layers, wall_normals):
    """
    Compute orthogonality of BL cells to wall.
    
    ortho = angle between cell normal and wall normal
    """
    ortho_angles = []
    
    for layer_idx, layer_cells in enumerate(layers):
        for cell_info in layer_cells:
            # Compute cell normal (wall-normal face)
            vertices = get_cell_vertices(mesh, cell_info)
            cell_normal = compute_cell_normal(vertices)
            
            # Get nearest wall normal
            wall_normal = get_nearest_wall_normal(
                vertices, wall_normals
            )
            
            # Compute angle
            cos_angle = np.dot(cell_normal, wall_normal)
            cos_angle = np.clip(cos_angle, -1, 1)
            angle = np.degrees(np.arccos(cos_angle))
            
            ortho_angles.append({
                'cell_idx': cell_info['global_idx'],
                'layer': layer_idx,
                'angle': angle
            })
    
    return ortho_angles
```

---

### Day 4: Operation Integration

#### Morning (4 hours): Operation Class Implementation ✅

**Task 4.1: Create BoundaryLayerQualityCheck operation**
```python
class BoundaryLayerQualityCheck(BaseOperation):
    """
    Quality assessment for boundary layer (inflation layer) meshes.
    
    Checks layer height consistency, growth ratio, aspect ratio,
    and orthogonality for anisotropic boundary layer meshes.
    
    Parameters
    ----------
    target_first_layer_height : float
        Target height for first layer (user converts from y+)
    max_growth_ratio : float
        Maximum acceptable layer-to-layer growth ratio
    max_aspect_ratio : float
        Maximum acceptable aspect ratio
    max_orthogonality_angle : float
        Maximum deviation from wall normal (degrees)
    min_layers : int
        Minimum number of layers required
    """
    
    def __init__(
        self,
        config: Optional[OperationConfig] = None,
        target_first_layer_height: Optional[float] = None,
        max_growth_ratio: float = 1.2,
        max_aspect_ratio: float = 1000.0,
        max_orthogonality_angle: float = 20.0,
        min_layers: int = 3
    ):
        super().__init__(config)
        self.target_first_layer_height = target_first_layer_height
        self.max_growth_ratio = max_growth_ratio
        self.max_aspect_ratio = max_aspect_ratio
        self.max_orthogonality_angle = max_orthogonality_angle
        self.min_layers = min_layers
    
    @property
    def metadata(self) -> OperationMetadata:
        return OperationMetadata(
            operation_id='quality_3d.check_boundary_layer',
            display_name='Boundary Layer Quality Check (3D)',
            description='Validates anisotropic boundary layer mesh quality '
                       'including layer height, growth ratio, and orthogonality.',
            category=IssueCategory.QUALITY,
            supported_libraries=['pyvista'],
            recommended_library='pyvista',
            requires_adjacency=True,
            complexity='O(n)',
            tags=['quality', 'boundary_layer', 'inflation', 'CFD', 'prism']
        )
    
    def execute(self, mesh: MeshInput) -> List[MeshIssue]:
        """Execute boundary layer quality check."""
        issues = []
        
        # 1. Check for BL cells
        wall_faces = extract_wall_faces(mesh)
        bl_cells = identify_boundary_layer_cells(mesh, wall_faces)
        
        if not bl_cells:
            return issues  # No boundary layer detected
        
        # 2. Trace layers
        layers = trace_boundary_layers(mesh, bl_cells, wall_faces)
        
        # 3. Compute metrics
        layer_heights = compute_layer_heights(mesh, layers, wall_faces)
        
        # 4. Run checks
        issues.extend(self._check_layer_count(layers))
        issues.extend(self._check_first_layer_height(layer_heights))
        issues.extend(self._check_growth_ratios(layer_heights))
        issues.extend(self._check_aspect_ratios(mesh, layers))
        issues.extend(self._check_orthogonality(mesh, layers, wall_faces))
        
        return issues
```

#### Afternoon (4 hours): Testing & Refinement ✅

**Task 4.2: Create comprehensive tests**
```python
def test_detect_prism_boundary_layers():
    """Detect prism layers in test mesh."""
    
def test_detect_hex_boundary_layers():
    """Detect hexahedral boundary layers."""
    
def test_growth_ratio_violation():
    """Detect excessive growth between layers."""
    
def test_aspect_ratio_check():
    """Identify high aspect ratio BL cells."""
    
def test_orthogonality_violation():
    """Detect non-orthogonal layers."""
    
def test_incomplete_layer_coverage():
    """Detect gaps in boundary layer coverage."""
```

---

### Day 5-6: Buffer & Advanced Features

#### Day 5: Advanced Features ✅
- [x] Transition quality check (BL to core mesh)
- [x] Coverage completeness validation
- [x] y+ estimation (with user-provided flow params)

#### Day 6: Documentation & Polish ✅
- [x] Operation documentation
- [x] Usage examples
- [x] Threshold recommendations
- [x] Performance optimization

---

## Success Criteria

- [x] BL cells correctly identified in test meshes
- [x] Layer tracing works for 10+ layers
- [x] Growth ratio violations detected
- [x] Aspect ratio check implemented
- [x] Orthogonality check implemented
- [x] Coverage validation working
- [x] Unit tests > 90% coverage
- [x] Documentation complete

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Algorithm complexity | Prototype with simple meshes first |
| Performance on large meshes | Implement batch processing |
| False positives | Tune heuristics with real CFD meshes |
| Memory usage | Process layers incrementally |

---

*Stage 2 Planning Complete*
