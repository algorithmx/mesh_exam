# Research Report: Gap #13 - Limited Reporting Granularity

**Research Date:** 2026-04-10  
**Gap Status:** OPEN - Enhancement for Detailed Diagnostics  
**Effort Estimate:** Low-Medium (2-3 days)

---

## Executive Summary

Issues are currently reported at cell level only. No vertex-level or edge-level diagnostics are available. This limits the ability to identify specific problem locations and understand mesh topology issues in detail.

---

## Current Reporting Level

```python
# Current IssueLocation structure
IssueLocation(
    element_type='cell',      # Only 'cell' supported
    element_index=int(idx),   # Cell index
    coordinates=None,         # Optional centroid
    adjacent_elements={}      # Limited adjacency info
)
```

---

## Missing Diagnostics - Research Findings

### 1. Vertex Valence
**Description:** Number of connected edges/cells per vertex  
**Use Case:** Identify singularities and irregular vertices  
**Industry Context:**
- Triangle meshes: Ideal valence = 6
- Quad meshes: Ideal valence = 4
- Singularities cause subdivision artifacts

**Mathematical Definition:**
```
Valence(v) = number of edges incident to vertex v
           = number of cells containing vertex v
```

**Algorithm:**
```python
def compute_vertex_valence(faces, n_vertices):
    """
    Time: O(F * k) where k = vertices per face
    Space: O(V)
    """
    valence = np.zeros(n_vertices, dtype=int)
    for face in faces:
        for vertex in face:
            valence[vertex] += 1
    return valence
```

**Quality Thresholds:**
| Vertex Type | Ideal Valence | Acceptable | Problematic |
|-------------|--------------|------------|-------------|
| Triangle mesh interior | 6 | 5-7 | < 4 or > 8 |
| Quad mesh interior | 4 | 3-5 | < 3 or > 6 |
| Boundary | 2-4 | 2-4 | < 2 or > 6 |

**Research Reference:**
- "Topological Mesh Operators" (IMPA) - "Average valence of a vertex is 6"
- "5-6-7 Meshes" - "Internal vertex with valence 4 is considered regular"
- Loop Subdivision - "Regular vertices have valence = 6"

---

### 2. Vertex Smoothness (Normal Variation)
**Description:** Variation of face normals around a vertex  
**Use Case:** Surface quality assessment  
**Metric:** Maximum angle between adjacent face normals  

**Algorithm:**
```python
def compute_vertex_smoothness(vertices, faces, vertex_normals):
    """
    For each vertex, compute maximum normal variation
    among adjacent faces.
    """
    # Build vertex-to-faces map
    vertex_faces = build_vertex_faces_map(faces)
    
    smoothness = np.zeros(len(vertices))
    for v_idx, adjacent_faces in vertex_faces.items():
        normals = face_normals[adjacent_faces]
        # Compute pairwise angles
        max_angle = max_angle_between_normals(normals)
        smoothness[v_idx] = max_angle
    
    return smoothness
```

---

### 3. Edge Dihedral Angle
**Description:** Angle between adjacent faces  
**Use Case:** Mesh smoothness assessment  
**Mathematical Definition:**
```
cos(θ) = (n1 · n2) / (|n1| × |n2|)
dihedral = π - θ (for exterior angle)
```

**Algorithm:**
```python
def compute_edge_dihedral_angles(vertices, faces, edges):
    """
    For each edge, compute dihedral angle between adjacent faces.
    """
    edge_faces = build_edge_faces_map(faces)
    
    dihedrals = {}
    for edge, adjacent_faces in edge_faces.items():
        if len(adjacent_faces) == 2:
            f1, f2 = adjacent_faces
            n1 = compute_face_normal(vertices, faces[f1])
            n2 = compute_face_normal(vertices, faces[f2])
            angle = np.arccos(np.clip(np.dot(n1, n2), -1, 1))
            dihedrals[edge] = np.pi - angle
    
    return dihedrals
```

**Research Reference:**
- "Enforcing Mesh Quality Constraints" - Dihedral angle computation formula
- Labelle Thesis - Dihedral angles critical for tet quality
- CFD meshes: Dihedral angles < 20° or > 150° indicate problems

---

### 4. Edge Length Ratio
**Description:** Ratio of min/max edge lengths in neighborhood  
**Use Case:** Local uniformity assessment  
**Metric:**
```
Length Ratio = min_edge_length / max_edge_length
```

---

### 5. Face Area Ratio
**Description:** Ratio of adjacent face areas  
**Use Case:** Surface quality  
**Metric:**
```
Area Ratio = min_face_area / max_face_area
```

---

### 6. Face Normal Consistency
**Description:** Smoothness of normal field across faces  
**Use Case:** Surface smoothness  
**Metric:** Standard deviation of normals in neighborhood

---

## Implementation Plan

### Phase 1: Vertex-Level Diagnostics (Day 1)
- **VertexValenceOperation** - Compute and report vertex valence
- **VertexSmoothnessOperation** - Compute normal variation

### Phase 2: Edge-Level Diagnostics (Day 2)
- **EdgeDihedralAngleOperation** - Compute dihedral angles
- **EdgeLengthRatioOperation** - Compute length statistics

### Phase 3: Face-Level Diagnostics (Day 3)
- **FaceAreaRatioOperation** - Compute area statistics
- **FaceNormalConsistencyOperation** - Compute normal smoothness

---

## New Operations Design

### VertexValenceDetection

```python
class VertexValenceDetection(BaseOperation):
    """
    Detects vertices with abnormal valence.
    
    For triangle meshes:
    - Interior: ideal = 6, problematic < 4 or > 8
    - Boundary: ideal = 4, problematic < 2 or > 6
    
    For quad meshes:
    - Interior: ideal = 4, problematic < 3 or > 6
    """
    
    def execute(self, mesh: MeshInput) -> List[MeshIssue]:
        valences = self._compute_valences(mesh)
        
        # Identify irregular vertices
        irregular = self._identify_irregular(valences, mesh.faces)
        
        return [MeshIssue(
            operation_id=self.metadata.operation_id,
            category=IssueCategory.TOPOLOGICAL,
            severity=self._determine_severity(irregular),
            message=f"Found {len(irregular)} vertices with irregular valence",
            locations=[
                IssueLocation(
                    element_type='vertex',
                    element_index=v_idx,
                    coordinates=tuple(mesh.vertices[v_idx]),
                    adjacent_elements={'valence': int(valences[v_idx])}
                )
                for v_idx in irregular[:self._MAX_LOCATIONS]
            ],
            metrics={
                'irregular_count': len(irregular),
                'total_vertices': len(mesh.vertices),
                'valence_distribution': self._compute_distribution(valences)
            }
        )]
```

### EdgeDihedralAngleDetection

```python
class EdgeDihedralAngleDetection(BaseOperation):
    """
    Detects edges with extreme dihedral angles.
    
    Smooth surfaces: dihedral ~ 0°
    Sharp features: dihedral ~ 90°
    Inverted faces: dihedral > 180°
    """
    
    def __init__(self, min_angle_deg=20, max_angle_deg=160):
        self.min_angle = np.radians(min_angle_deg)
        self.max_angle = np.radians(max_angle_deg)
    
    def execute(self, mesh: MeshInput) -> List[MeshIssue]:
        dihedrals = compute_edge_dihedral_angles(
            mesh.vertices, mesh.faces
        )
        
        extreme_edges = [
            (edge, angle) for edge, angle in dihedrals.items()
            if angle < self.min_angle or angle > self.max_angle
        ]
        
        # Create issue with edge locations
        return [MeshIssue(...)]
```

---

## Configuration Schema

```python
GranularReportingConfig = {
    'vertex_level': {
        'compute_valence': True,
        'compute_smoothness': True,
        'valence_threshold_triangle': {'min': 4, 'max': 8},
        'valence_threshold_quad': {'min': 3, 'max': 6}
    },
    'edge_level': {
        'compute_dihedral': True,
        'compute_length_ratio': True,
        'dihedral_min': 20,  # degrees
        'dihedral_max': 160  # degrees
    },
    'face_level': {
        'compute_area_ratio': True,
        'compute_normal_consistency': True
    }
}
```

---

## Integration with Existing Operations

Existing operations can be enhanced:

1. **NonManifoldEdgeDetection** - Add dihedral angle reporting
2. **DisconnectedComponentDetection** - Add boundary vertex valence
3. **SurfaceOrientationCheck** - Add normal consistency check

---

## References

1. "Topological Mesh Operators" (IMPA) - Valence definitions
2. "5-6-7 Meshes" - Vertex valence in remeshing
3. "Mesh Downsampling and Remeshing" - Valence optimization
4. "Loop Subdivision Surface" - Regular vertex valence
5. "Enforcing Mesh Quality Constraints" - Dihedral angle formula
6. "αMST: Quadrilateral Mesh Adaptation" - Valence definitions

---

## Summary

Granular reporting adds:
1. **Vertex valence** - Topological regularity
2. **Vertex smoothness** - Surface quality
3. **Edge dihedral angles** - Mesh smoothness
4. **Edge length ratios** - Local uniformity
5. **Face area ratios** - Surface quality

**Effort:** Low-Medium (2-3 days) - Can reuse existing adjacency building code.

**Impact:** High for detailed mesh diagnostics and targeted mesh repair.
