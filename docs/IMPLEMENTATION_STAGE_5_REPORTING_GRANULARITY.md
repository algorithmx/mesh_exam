# Stage 5 Implementation Plan: Gap #13 - Limited Reporting Granularity

**Stage:** 5 of 5  
**Gap:** #13 - Limited Reporting Granularity  
**Priority:** LOW  
**Duration:** 2-3 days  
**Prerequisites:** None (can parallelize with any stage)  

---

## Implementation Status: ✅ COMPLETE

**Completed Date:** 2026-04-10
**Actual Duration:** ~2 days
**Implementation Files:**
- `mesh_issue_examiner/operations/topological/vertex_valence.py`
- `mesh_issue_examiner/operations/topological/edge_dihedral.py`
- `mesh_issue_examiner/operations/topological/edge_length_ratio.py`
- `mesh_issue_examiner/utils/topology.py`

**Operations Implemented:**
- [x] VertexValenceDetection (`topology.detect_vertex_valence`)
- [x] EdgeDihedralAngleDetection (`topology.detect_edge_dihedral_angle`)
- [x] EdgeLengthRatioDetection (`topology.detect_edge_length_ratio`)

**Utilities Added:**
- [x] compute_vertex_valence
- [x] build_edge_face_map
- [x] extract_unique_edges
- [x] compute_edge_lengths
- [x] compute_face_areas

---

## Overview

This stage extends the framework's reporting capabilities from cell-level to include vertex-level and edge-level diagnostics. This enables more precise identification of mesh problems and targeted repair.

---

## Background: Current Reporting Limitations

### Current IssueLocation Structure
```python
IssueLocation(
    element_type='cell',      # Only 'cell' supported
    element_index=int(idx),   # Cell index
    coordinates=None,         # Optional centroid
    adjacent_elements={}      # Limited adjacency info
)
```

### Missing Diagnostics
- **Vertex valence** - Number of connected edges/cells
- **Edge dihedral angles** - Angle between adjacent faces
- **Vertex smoothness** - Normal variation around vertex
- **Edge length ratios** - Local uniformity

---

## Implementation Objectives

### Primary Goals
1. Implement vertex valence computation
2. Implement edge dihedral angle detection
3. Implement vertex smoothness metric
4. Create new operations for granular diagnostics

### New Operations

| Operation | Level | Complexity |
|-----------|-------|------------|
| VertexValenceDetection | Vertex | Low |
| EdgeDihedralAngleDetection | Edge | Medium |
| VertexSmoothnessDetection | Vertex | Medium |
| EdgeLengthRatioDetection | Edge | Low |

---

## Day-by-Day Implementation Plan

### Day 1: Vertex-Level Diagnostics ✅ COMPLETE

#### Morning (4 hours): Vertex Valence ✅

**Task 1.1: Implement vertex valence computation**
```python
# In mesh_issue_examiner/utils/topology.py

def compute_vertex_valence(faces, n_vertices):
    """
    Compute valence (number of incident edges) for each vertex.
    
    Algorithm: Count face occurrences for each vertex
    
    Time: O(F * k) where k = vertices per face
    Space: O(V)
    
    Parameters
    ----------
    faces : (n_faces, k) array
    n_vertices : int
    
    Returns
    -------
    valence : (n_vertices,) array
    """
    valence = np.zeros(n_vertices, dtype=int)
    
    # Handle both uniform and ragged faces
    if faces.ndim == 2:
        # Uniform faces (all triangles or all quads)
        for i in range(faces.shape[1]):
            np.add.at(valence, faces[:, i], 1)
    else:
        # Ragged faces (mixed types)
        for face in faces:
            for vertex in face:
                valence[vertex] += 1
    
    return valence
```

**Task 1.2: Create VertexValenceDetection operation**
```python
class VertexValenceDetection(BaseOperation):
    """
    Detects vertices with abnormal valence.
    
    Valence is the number of edges/faces incident to a vertex.
    Abnormal valence indicates mesh irregularities.
    
    For triangle meshes:
    - Interior vertices: ideal = 6, problematic < 4 or > 8
    - Boundary vertices: ideal = 4, problematic < 2 or > 6
    
    For quad meshes:
    - Interior vertices: ideal = 4, problematic < 3 or > 6
    - Boundary vertices: ideal = 2, problematic < 2 or > 4
    """
    
    def __init__(
        self,
        config: Optional[OperationConfig] = None,
        min_valence_triangle: int = 4,
        max_valence_triangle: int = 8,
        min_valence_quad: int = 3,
        max_valence_quad: int = 6
    ):
        super().__init__(config)
        self.min_valence_triangle = min_valence_triangle
        self.max_valence_triangle = max_valence_triangle
        self.min_valence_quad = min_valence_quad
        self.max_valence_quad = max_valence_quad
    
    @property
    def metadata(self) -> OperationMetadata:
        return OperationMetadata(
            operation_id='topology.detect_vertex_valence',
            display_name='Vertex Valence Detection',
            description='Identifies vertices with abnormal valence (connected edge count).',
            category=IssueCategory.TOPOLOGICAL,
            supported_libraries=['trimesh', 'pyvista'],
            recommended_library='pyvista',
            requires_adjacency=True,
            complexity='O(F)',
            tags=['topology', 'vertex', 'valence', 'irregular']
        )
    
    def execute(self, mesh: MeshInput) -> List[MeshIssue]:
        """Execute vertex valence detection."""
        if len(mesh.faces) == 0:
            return []
        
        # Compute valence for all vertices
        valence = compute_vertex_valence(mesh.faces, len(mesh.vertices))
        
        # Determine mesh type (triangle vs quad)
        avg_verts_per_face = np.mean([len(f) for f in mesh.faces])
        is_triangle_mesh = avg_verts_per_face < 3.5
        
        # Set thresholds based on mesh type
        if is_triangle_mesh:
            min_val = self.min_valence_triangle
            max_val = self.max_valence_triangle
        else:
            min_val = self.min_valence_quad
            max_val = self.max_valence_quad
        
        # Find irregular vertices
        irregular_mask = (valence < min_val) | (valence > max_val)
        irregular_indices = np.where(irregular_mask)[0]
        
        if len(irregular_indices) == 0:
            return []
        
        # Create locations for irregular vertices
        locations = []
        for v_idx in irregular_indices[:self._MAX_LOCATIONS]:
            locations.append(IssueLocation(
                element_type='vertex',
                element_index=int(v_idx),
                coordinates=tuple(mesh.vertices[v_idx]),
                adjacent_elements={'valence': int(valence[v_idx])}
            ))
        
        # Create issue
        return [MeshIssue(
            operation_id=self.metadata.operation_id,
            category=IssueCategory.TOPOLOGICAL,
            severity=self._determine_severity(len(irregular_indices), len(mesh.vertices)),
            message=f"Found {len(irregular_indices)} vertices with irregular valence",
            locations=locations,
            metrics={
                'irregular_count': int(len(irregular_indices)),
                'total_vertices': int(len(mesh.vertices)),
                'valence_distribution': self._compute_distribution(valence)
            }
        )]
```

#### Afternoon (4 hours): Vertex Smoothness ✅（取消 - 优先实施边级检测）

**Task 1.3: Implement vertex smoothness computation**
```python
def compute_vertex_smoothness(vertices, faces, face_normals):
    """
    Compute smoothness (normal variation) at each vertex.
    
    Returns maximum angle between adjacent face normals.
    High values indicate sharp features or noise.
    """
    n_vertices = len(vertices)
    smoothness = np.zeros(n_vertices)
    
    # Build vertex-to-faces map
    vertex_faces = [[] for _ in range(n_vertices)]
    for f_idx, face in enumerate(faces):
        for v_idx in face:
            vertex_faces[v_idx].append(f_idx)
    
    # Compute max normal angle for each vertex
    for v_idx in range(n_vertices):
        adjacent_faces = vertex_faces[v_idx]
        
        if len(adjacent_faces) < 2:
            smoothness[v_idx] = 0.0
            continue
        
        # Compute max angle between adjacent face normals
        max_angle = 0.0
        normals = face_normals[adjacent_faces]
        
        for i in range(len(normals)):
            for j in range(i + 1, len(normals)):
                n1 = normals[i]
                n2 = normals[j]
                
                cos_angle = np.dot(n1, n2)
                cos_angle = np.clip(cos_angle, -1, 1)
                angle = np.degrees(np.arccos(cos_angle))
                
                max_angle = max(max_angle, angle)
        
        smoothness[v_idx] = max_angle
    
    return smoothness
```

---

### Day 2: Edge-Level Diagnostics ✅ COMPLETE

#### Morning (4 hours): Edge Dihedral Angles ✅

**Task 2.1: Build edge-face adjacency**
```python
def build_edge_face_map(faces):
    """
    Build mapping from edges to adjacent faces.
    
    Returns:
        Dict mapping (v1, v2) edge to list of face indices
    """
    edge_faces = {}
    
    for f_idx, face in enumerate(faces):
        n_verts = len(face)
        for i in range(n_verts):
            v1 = face[i]
            v2 = face[(i + 1) % n_verts]
            
            # Canonical edge ordering
            edge = tuple(sorted([v1, v2]))
            
            if edge not in edge_faces:
                edge_faces[edge] = []
            edge_faces[edge].append(f_idx)
    
    return edge_faces
```

**Task 2.2: Implement EdgeDihedralAngleDetection**
```python
class EdgeDihedralAngleDetection(BaseOperation):
    """
    Detects edges with extreme dihedral angles.
    
    Dihedral angle is the angle between two adjacent faces.
    - Smooth surface: dihedral ~ 0°
    - Sharp edge: dihedral ~ 90°
    - Inverted faces: dihedral > 180°
    
    Parameters
    ----------
    min_angle : float
        Minimum acceptable dihedral angle (degrees)
    max_angle : float
        Maximum acceptable dihedral angle (degrees)
    """
    
    def __init__(
        self,
        config: Optional[OperationConfig] = None,
        min_angle: float = 20.0,
        max_angle: float = 160.0
    ):
        super().__init__(config)
        self.min_angle = min_angle
        self.max_angle = max_angle
    
    def execute(self, mesh: MeshInput) -> List[MeshIssue]:
        """Execute dihedral angle detection."""
        if len(mesh.faces) < 2:
            return []
        
        # Compute face normals
        from ...adapters.factory import get_adapter
        adapter = get_adapter('pyvista')
        native_mesh = adapter.from_canonical(mesh)
        face_normals = adapter.compute_face_normals(native_mesh)
        
        # Build edge-face map
        edge_faces = build_edge_face_map(mesh.faces)
        
        # Find edges with extreme dihedral angles
        extreme_edges = []
        
        for edge, adjacent_faces in edge_faces.items():
            if len(adjacent_faces) != 2:
                continue  # Boundary or non-manifold
            
            f1, f2 = adjacent_faces
            n1 = face_normals[f1]
            n2 = face_normals[f2]
            
            # Compute dihedral angle
            cos_angle = np.dot(n1, n2)
            cos_angle = np.clip(cos_angle, -1, 1)
            angle = np.degrees(np.arccos(cos_angle))
            
            # Dihedral is supplementary to normal angle
            dihedral = 180 - angle
            
            if dihedral < self.min_angle or dihedral > self.max_angle:
                extreme_edges.append({
                    'edge': edge,
                    'dihedral': dihedral,
                    'faces': adjacent_faces
                })
        
        if not extreme_edges:
            return []
        
        # Create locations for extreme edges
        locations = []
        for edge_info in extreme_edges[:self._MAX_LOCATIONS]:
            v1, v2 = edge_info['edge']
            midpoint = (mesh.vertices[v1] + mesh.vertices[v2]) / 2
            
            locations.append(IssueLocation(
                element_type='edge',
                element_index=-1,  # Edges don't have global index
                coordinates=tuple(midpoint),
                adjacent_elements={
                    'vertices': [int(v1), int(v2)],
                    'dihedral_angle': float(edge_info['dihedral']),
                    'faces': [int(f) for f in edge_info['faces']]
                }
            ))
        
        return [MeshIssue(
            operation_id=self.metadata.operation_id,
            category=IssueCategory.GEOMETRIC,
            severity=self._determine_severity(len(extreme_edges)),
            message=f"Found {len(extreme_edges)} edges with extreme dihedral angles",
            locations=locations,
            metrics={
                'extreme_edge_count': len(extreme_edges),
                'min_dihedral': float(min(e['dihedral'] for e in extreme_edges)),
                'max_dihedral': float(max(e['dihedral'] for e in extreme_edges))
            }
        )]
```

#### Afternoon (4 hours): Edge Length Ratio ✅

**Task 2.3: Implement EdgeLengthRatioDetection**
```python
class EdgeLengthRatioDetection(BaseOperation):
    """
    Detects edges with abnormal length ratios in neighborhood.
    
    High length ratios indicate non-uniform mesh refinement.
    """
    
    def __init__(
        self,
        config: Optional[OperationConfig] = None,
        max_ratio: float = 10.0
    ):
        super().__init__(config)
        self.max_ratio = max_ratio
    
    def execute(self, mesh: MeshInput) -> List[MeshIssue]:
        """Execute edge length ratio detection."""
        if len(mesh.faces) == 0:
            return []
        
        # Build edges and compute lengths
        edges = self._extract_unique_edges(mesh.faces)
        lengths = self._compute_edge_lengths(mesh.vertices, edges)
        
        # Build vertex-to-edges map
        vertex_edges = [[] for _ in range(len(mesh.vertices))]
        for e_idx, (v1, v2) in enumerate(edges):
            vertex_edges[v1].append(e_idx)
            vertex_edges[v2].append(e_idx)
        
        # Find edges with high length ratio in neighborhood
        high_ratio_edges = []
        
        for e_idx, (v1, v2) in enumerate(edges):
            # Get neighborhood edges
            neighbor_edges = set(vertex_edges[v1] + vertex_edges[v2])
            neighbor_edges.discard(e_idx)
            
            if not neighbor_edges:
                continue
            
            neighbor_lengths = lengths[list(neighbor_edges)]
            min_neighbor = np.min(neighbor_lengths)
            
            if min_neighbor > 0:
                ratio = lengths[e_idx] / min_neighbor
                if ratio > self.max_ratio:
                    high_ratio_edges.append({
                        'edge': (v1, v2),
                        'ratio': ratio,
                        'length': lengths[e_idx]
                    })
        
        # Create issue if found
        if not high_ratio_edges:
            return []
        
        # Create locations...
        return [MeshIssue(...)]
```

---

### Day 3: Integration & Testing ✅ COMPLETE

#### Morning (4 hours): Integration Tests ✅

**Task 3.1: Create comprehensive tests**
```python
def test_vertex_valence_regular_mesh():
    """Regular mesh should have all regular valence."""
    # Create regular grid
    # Verify no issues reported

def test_vertex_valence_detects_irregular():
    """Irregular vertices should be detected."""
    # Create mesh with high-valence vertex
    # Verify detection

def test_edge_dihedral_smooth_surface():
    """Smooth surface has low dihedral angles."""
    # Create smooth sphere
    # Verify no issues

def test_edge_dihedral_detects_sharp_edges():
    """Sharp edges detected by high dihedral."""
    # Create cube (sharp edges)
    # Verify detection of 90° edges

def test_edge_length_ratio_uniform():
    """Uniform mesh has good length ratios."""
    # Create uniform mesh
    # Verify no issues

def test_edge_length_ratio_detects_refinement():
    """Detects non-uniform refinement."""
    # Create mesh with localized refinement
    # Verify detection
```

#### Afternoon (4 hours): Documentation ✅

**Task 3.2: Update operation documentation**
- Add new operations to volumetric operations index
- Document vertex/edge level reporting
- Add usage examples

**Task 3.3: Update __init__.py files**
```python
# In mesh_issue_examiner/operations/topological/__init__.py
from .vertex_valence import VertexValenceDetection
from .edge_dihedral import EdgeDihedralAngleDetection
from .edge_length_ratio import EdgeLengthRatioDetection

__all__ = [
    # ... existing ...
    "VertexValenceDetection",
    "EdgeDihedralAngleDetection",
    "EdgeLengthRatioDetection",
]
```

---

## Success Criteria

- [x] Vertex valence computation working
- [x] Edge dihedral angle detection working
- [ ] Vertex smoothness metric implemented (deferred)
- [x] Edge length ratio detection working
- [x] Unit tests > 90% coverage
- [x] Integration with existing operations
- [x] Documentation complete

---

*Stage 5 Planning Complete*
