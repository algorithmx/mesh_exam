# Mesh Issue Operations Catalog

**Generated**: 2026-04-11  
**Last Updated**: 2026-04-11  
**Purpose**: Comprehensive reference of all implemented mesh issue examination operations

---

## Overview

This document catalogs all 21 mesh inspection operations implemented in the `mesh_issue_examiner` framework. Operations are organized by their **actual** `IssueCategory` as defined in operation metadata, providing comprehensive coverage of topological, geometric, orientation, and quality defects in both surface and volumetric meshes.

**Key Distinctions:**
- **Surface Mesh Operations**: Work on 2D manifold surfaces (triangles, polygons)
- **Volumetric Mesh Operations**: Work on 3D cells (tetrahedra, hexahedra, prisms, pyramids)
- **Mixed Operations**: Some operations support both surface and volumetric meshes

---

## Category Summary

| Category | Count | Description |
|----------|-------|-------------|
| TOPOLOGICAL | 9 | Connectivity, manifold properties, valence, and structural integrity |
| ORIENTATION | 2 | Face/cell winding order and normal consistency |
| GEOMETRIC | 3 | Shape irregularities, dihedral angles, and geometric defects |
| QUALITY | 7 | Metric-based quality assessment, statistics, and material interfaces |

**Note on Category Assignment**: Operations are categorized by their `IssueCategory` metadata as defined in code, not by semantic grouping. For example, `detect_vertex_valence` is TOPOLOGICAL (per metadata) despite being conceptually geometric.

---

## 1. TOPOLOGICAL Operations

### Surface Mesh Topology

#### `topology.detect_nonmanifold_edges`
- **Display Name**: Non-Manifold Edge Detection
- **Description**: Identifies edges shared by more than two faces, violating the 2-manifold property required for solid models
- **Complexity**: O(n)
- **Supported Libraries**: trimesh, pyvista, pymeshlab
- **Recommended**: pyvista
- **Tags**: topology, manifold, connectivity, basic
- **Algorithm**: Edge-to-face adjacency mapping with face count verification. Boundary edges (1 face) are excluded as they are normal for open surfaces.
- **Parameters**: None (uses default tolerance for vertex comparison)

#### `topology.detect_nonmanifold_vertices`
- **Display Name**: Non-Manifold Vertex Detection
- **Description**: Identifies vertices whose incident faces form disconnected groups, violating the 2-manifold property
- **Complexity**: O(V + F)
- **Supported Libraries**: trimesh, pyvista, pymeshlab
- **Recommended**: pyvista
- **Tags**: topology, manifold, connectivity, vertices
- **Algorithm**: One-ring connected component analysis per Jiao & He (2014). Uses Union-Find to detect disconnected face groups around each vertex.

#### `topology.detect_self_intersections`
- **Display Name**: Self-Intersection Detection
- **Description**: Identifies triangles that improperly intersect other non-adjacent triangles in the same mesh
- **Complexity**: O(n log n)
- **Supported Libraries**: pyvista
- **Recommended**: pyvista
- **Tags**: topology, self-intersection, geometric-defect, critical
- **Algorithm**: Möller-Trumbore triangle-triangle intersection with KD-tree acceleration. Only checks non-adjacent triangles (triangles sharing an edge or vertex are excluded from intersection tests).

#### `topology.detect_duplicate_vertices`
- **Display Name**: Duplicate Vertex Detection
- **Description**: Identifies vertices with identical or nearly identical coordinates
- **Complexity**: O(n log n)
- **Supported Libraries**: trimesh, pyvista, pymeshlab, meshio, vedo
- **Recommended**: pyvista
- **Tags**: topology, cleanup, vertices, duplicate, weld
- **Parameters**: tolerance (default: 1e-6) - distance threshold for considering vertices as duplicates
- **Algorithm**: Spatial hashing or KD-tree based nearest neighbor search with configurable tolerance

#### `topology.detect_boundary_edges`
- **Display Name**: Boundary Edge Detection
- **Description**: Identifies edges incident to exactly one face, indicating open surfaces or holes
- **Complexity**: O(n)
- **Supported Libraries**: trimesh, pyvista, pymeshlab
- **Recommended**: pyvista
- **Tags**: topology, boundary, open, watertight, holes
- **Output**: Groups boundary edges into continuous loops using adjacency traversal
- **Algorithm**: Edge-to-face adjacency mapping, filtering edges with exactly one incident face, then grouping into connected boundary loops

#### `topology.detect_disconnected_components`
- **Display Name**: Disconnected Component Detection
- **Description**: Identifies separate connected components by analyzing vertex connectivity through shared edges
- **Complexity**: O(V + F)
- **Supported Libraries**: trimesh, pyvista, pymeshlab
- **Recommended**: pyvista
- **Tags**: topology, connectivity, components, disconnected
- **Algorithm**: Union-Find (Disjoint Set Union) with path compression. Each component is assigned a unique ID and statistics (vertex count, face count, bounding box) are computed.

#### `topology.detect_isolated_vertices`
- **Display Name**: Isolated Vertex Detection
- **Description**: Identifies vertices not referenced by any face
- **Complexity**: O(V + F)
- **Supported Libraries**: trimesh, pyvista, pymeshlab
- **Recommended**: pyvista
- **Tags**: topology, vertices, isolated, unreferenced
- **Algorithm**: Set-difference between all vertex indices and the set of vertices referenced by faces. Returns indices of unreferenced vertices.

#### `topology.detect_duplicate_faces`
- **Display Name**: Duplicate Face Detection
- **Description**: Identifies faces that share the same set of vertex indices
- **Complexity**: O(F * k log k) where k is vertices per face
- **Supported Libraries**: trimesh, pyvista, pymeshlab
- **Recommended**: pyvista
- **Tags**: topology, faces, duplicate, degenerate
- **Algorithm**: Canonical sorted-key hash for face vertex sets. Creates a hash key by sorting vertex indices within each face, then identifies faces with identical keys.

#### `topology.detect_vertex_valence`
- **Display Name**: Vertex Valence Detection
- **Description**: Identifies vertices with abnormal valence (number of incident edges/faces). Abnormal valence indicates mesh irregularities and singularities that can affect subdivision surfaces and simulation accuracy.
- **Complexity**: O(F)
- **Supported Libraries**: pyvista, trimesh
- **Recommended**: pyvista
- **Tags**: topology, vertex, valence, irregular, singularity
- **Category**: TOPOLOGICAL (per metadata; conceptually geometric but classified by connectivity analysis)
- **Thresholds**:
  - Triangle meshes: ideal=6, acceptable=4-8 (interior), 2-6 (boundary)
  - Quad meshes: ideal=4, acceptable=3-6 (interior), 2-4 (boundary)
- **Parameters**: 
  - min_valence_triangle (default: 4)
  - max_valence_triangle (default: 8)
  - min_valence_quad (default: 3)
  - max_valence_quad (default: 6)
- **Algorithm**: 
  1. Computes valence per vertex using `compute_vertex_valence()` from topology utils
  2. Auto-detects mesh type: `faces.shape[1] == 3` for uniform, or `avg_verts < 3.5` for mixed
  3. Applies mesh-type-specific thresholds
  4. Returns valence distribution statistics (mean, min, max, histogram)
- **Implementation Notes**:
  - Locations capped at 100 (`_MAX_LOCATIONS`)
  - Returns irregular indices and valences in metadata (first 50)
  - Severity: >20% irregular = ERROR, >5% = WARNING

### Volumetric Mesh Topology

#### `topology_3d.detect_nonmanifold_faces`
- **Display Name**: Non-Manifold Face Detection (3D)
- **Description**: Identifies faces shared by more than two volume cells
- **Complexity**: O(n)
- **Supported Libraries**: pyvista
- **Recommended**: pyvista
- **Tags**: topology, manifold, connectivity, volumetric, 3d
- **Applicable Elements**: tet4, hex8, prism6, pyramid5
- **Algorithm**: Face-to-cell adjacency mapping with cell count verification. Internal faces should be shared by exactly 2 cells in a valid manifold volumetric mesh.

---

## 2. ORIENTATION Operations

#### `topology.detect_inconsistent_orientation`
- **Display Name**: Surface Orientation Consistency Check
- **Description**: Detects faces with inconsistent winding order by verifying adjacent faces traverse shared edges in opposite directions
- **Complexity**: O(n)
- **Supported Libraries**: trimesh, pyvista, pymeshlab
- **Recommended**: pyvista
- **Tags**: orientation, normals, winding, flipped, consistency, surface
- **Algorithm**: Edge-direction comparison per StackOverflow algorithm: for each internal edge shared by two faces, verify that the edge is traversed in opposite directions (e.g., face A has edge (a,b), face B must have (b,a)). Uses BFS to identify connected components (patches) per Borodin et al. (2004).
- **References**: Borodin et al. (CGI 2004), StackOverflow orientation algorithm

#### `quality_3d.check_orientation_consistency`
- **Display Name**: Orientation and Consistency Check (3D)
- **Description**: Detects inverted elements (negative volume) and inconsistent face winding in volumetric meshes
- **Complexity**: O(n log n)
- **Supported Libraries**: pyvista
- **Recommended**: pyvista
- **Tags**: orientation, volumetric, 3d, inverted, winding, consistency
- **Algorithm**:
  - **Inverted cells**: Uses PyVista's cell_quality('volume') to compute signed volumes. Cells with volume < -tolerance are flagged as CRITICAL.
  - **Inconsistent winding**: For each internal face shared by two cells, uses directed-edge method: every edge (u→v) in face A must appear as (v→u) in face B. Inconsistent winding is flagged as ERROR.
- **Checks**:
  - Inverted cells: signed volume < -tolerance → CRITICAL
  - Inconsistent winding: faces with same vertex order in adjacent cells → ERROR
- **Applicable Elements**: tet4, hex8, prism6, pyramid5

---

## 3. GEOMETRIC Operations

**Category Note**: Operations in this section have `category=IssueCategory.GEOMETRIC` in their metadata. They detect shape irregularities, angles, and geometric defects that don't involve topological connectivity.

#### `topology.detect_edge_dihedral_angle`
- **Display Name**: Edge Dihedral Angle Detection
- **Description**: Identifies edges with extreme dihedral angles between adjacent faces. Useful for detecting sharp features, creases, and mesh irregularities.
- **Complexity**: O(E)
- **Supported Libraries**: pyvista, trimesh
- **Recommended**: pyvista
- **Tags**: topology, edge, dihedral, angle, smoothness, sharp
- **Category**: GEOMETRIC (per metadata, despite `topology.` prefix in operation_id)
- **Parameters**:
  - min_angle (default: 20.0°) - minimum acceptable dihedral angle
  - max_angle (default: 160.0°) - maximum acceptable dihedral angle
- **Interpretation**:
  - Smooth surface: dihedral ~ 0-20°
  - Sharp edge: dihedral ~ 90°
  - Crease: dihedral > 90°
  - Inverted faces: dihedral approaching 180°
- **Algorithm**: 
  1. Computes face normals using PyVista adapter (`compute_face_normals`)
  2. Builds edge-to-face map via `build_edge_face_map()` from topology utils
  3. For internal edges (exactly 2 adjacent faces): computes angle between normals
  4. Dihedral = 180° - angle_between_normals
  5. Flags edges outside [min_angle, max_angle] range
- **Implementation Notes**:
  - Only checks internal edges (2 adjacent faces); boundary and non-manifold edges skipped
  - Returns edge midpoint coordinates for visualization
  - Severity based on fraction of extreme edges: >20% = ERROR, >5% = WARNING

#### `topology.detect_edge_length_ratio`
- **Display Name**: Edge Length Ratio Detection
- **Description**: Identifies edges with abnormal length ratios in their neighborhood. High ratios indicate non-uniform mesh refinement or stretched elements.
- **Complexity**: O(E)
- **Supported Libraries**: pyvista, trimesh
- **Recommended**: pyvista
- **Tags**: topology, edge, length, ratio, uniformity, refinement
- **Category**: GEOMETRIC (per metadata, despite `topology.` prefix in operation_id)
- **Parameters**:
  - max_ratio (default: 10.0) - maximum acceptable ratio
- **Formula**: ratio = edge_length / min_neighbor_length
- **Algorithm**:
  1. Extracts unique edges via `extract_unique_edges()` from topology utils
  2. Computes edge lengths via `compute_edge_lengths()`
  3. Builds vertex-to-edges incidence map
  4. For each edge, finds neighbors (edges sharing either endpoint)
  5. Computes ratio to minimum neighbor length
  6. Flags edges where ratio > max_ratio
- **Implementation Notes**:
  - Neighborhood defined as edges incident to either endpoint vertex
  - Severity thresholds: >10% edges = ERROR, >5% = WARNING, else INFO

#### `quality_3d.assess_geometric_validity`
- **Display Name**: Geometric Validity Assessment (3D)
- **Description**: Detects degenerate cells (near-zero volume) and sliver elements (poor volume-to-edge-length ratio) in volumetric meshes.
- **Complexity**: O(n)
- **Supported Libraries**: pyvista
- **Recommended**: pyvista
- **Tags**: quality, volumetric, 3d, degenerate, sliver, validity
- **Category**: GEOMETRIC
- **Parameters**:
  - degenerate_tolerance (default: 1e-12) - absolute volume threshold for degenerate cells
  - sliver_threshold (default: 1e-6) - volume/max_edge³ ratio threshold
- **Checks**:
  - **Degenerate cells**: |volume| < tolerance → CRITICAL severity
  - **Sliver elements**: volume/max_edge³ < threshold → WARNING severity
- **Applicable Elements**: tet4, hex8, prism6, pyramid5
- **Algorithm**:
  1. Retrieves PyVista UnstructuredGrid via `mesh.get_native('pyvista')`
  2. Computes cell volumes via adapter's `compute_cell_quality(grid, 'volume')`
  3. Degenerate check: absolute volume below tolerance
  4. Sliver check: element-specific edge enumeration and ratio computation
- **Sliver Detection per Element Type**:
  | Element | Edge Count | Edge Definition |
  |---------|------------|-----------------|
  | tet4 | 6 | ab, ac, ad, bc, bd, cd |
  | hex8 | 12 | bottom×4, top×4, vertical×4 |
  | prism6 | 9 | bottom△×3, top△×3, vertical×3 |
  | pyramid5 | 8 | base□×4, apex×4 |
- **Implementation Notes**:
  - Sliver detection excludes already-flagged degenerate cells
  - Cell index offset computed per element type for global indexing
  - Locations capped at 100 per issue type

---

## 4. QUALITY Operations

**Category Note**: Operations in this section have `category=IssueCategory.QUALITY` in their metadata. They compute metric-based quality assessments and statistical analyses.

### Surface Mesh Quality

#### `quality.compute_cell_quality`
- **Display Name**: Cell Quality Computation
- **Description**: Computes element shape quality metrics for surface meshes (triangles/polygons)
- **Complexity**: O(n)
- **Supported Libraries**: pyvista, trimesh
- **Recommended**: pyvista
- **Tags**: quality, shape, aspect_ratio, skewness, angles
- **Parameters**:
  - quality_types (default: ['aspect_ratio', 'min_angle', 'max_angle']) - list of metrics to compute
- **Supported Metrics**:
  - aspect_ratio: aspect ratio (higher is worse, threshold: 10.0)
  - min_angle: minimum angle in degrees (lower is worse, threshold: 15.0)
  - max_angle: maximum angle in degrees (higher is worse, threshold: 120.0)
  - scaled_jacobian: scaled Jacobian determinant (threshold: 0.5)
  - condition: condition number (threshold: 100.0)
  - shape: shape metric 0-1 (threshold: 0.25)
  - area: face area (threshold: 1e-10)
  - distortion: distortion metric
  - radius_ratio: circumradius/inradius ratio
- **Note**: 'skewness' is not valid for triangular meshes (VTK returns -1). Use 'scaled_jacobian' or 'shape' instead for triangle quality assessment.
- **Algorithm**: Uses PyVista/VTK cell_quality() function with configurable metric types and thresholds.

### Volumetric Mesh Quality

#### `quality_3d.compute_volume_cell_quality`
- **Display Name**: Volume Cell Quality Computation (3D)
- **Description**: Computes quality metrics for volumetric cells (tet/hex/prism/pyramid)
- **Complexity**: O(n)
- **Supported Libraries**: pyvista
- **Recommended**: pyvista
- **Tags**: quality, shape, volumetric, 3d, jacobian, volume, warpage, skew, taper, stretch
- **Parameters**:
  - quality_types (default: ['scaled_jacobian', 'volume', 'aspect_ratio', 'condition', 'shape']) - list of metrics to compute
  - skip_unsupported_metrics (default: True) - skip metrics not applicable to mesh element types
- **Supported Metrics**:
  - scaled_jacobian: scaled Jacobian determinant (threshold: 0.1, higher is worse)
  - volume: cell volume (threshold: 1e-12, lower is worse)
  - aspect_ratio: aspect ratio (threshold: 0.1, lower is worse)
  - condition: condition number (threshold: 100.0, higher is worse)
  - shape: shape metric (threshold: 0.3, lower is worse)
  - skewness/skew: deviation from ideal (threshold: 0.8, lower is worse)
  - orthogonality: face normal orthogonality (threshold: 0.15, lower is worse)
  - warpage: quadrilateral face planarity deviation (threshold: 0.5, higher is worse)
  - equiangle_skew: angular deviation from ideal (threshold: 0.25, higher is worse)
  - stretch: min edge / max diagonal (threshold: 0.25, lower is worse)
  - taper: opposite face area ratio for hex (threshold: 0.5, higher is worse)
  - radius_ratio: circumradius/inradius for tets (threshold: 10.0, higher is worse)
  - collapse_ratio: min altitude / max edge (threshold: 0.1, lower is worse)
- **Element-Specific Support**:
  - warpage: hex8, prism6, pyramid5 (elements with quad faces)
  - taper: hex8 (elements with opposite faces)
  - radius_ratio: tet4 (primary), extensible to others
  - Other metrics: all element types (tet4, hex8, prism6, pyramid5)
- **Algorithm**: Uses PyVista/VTK cell quality computation with configurable thresholds. Metrics not applicable to mesh element types are automatically skipped.

### CFD-Specific Quality

#### `quality_3d.check_boundary_layer`
- **Display Name**: Boundary Layer Quality Check (3D)
- **Description**: Validates anisotropic boundary layer meshes for CFD applications
- **Complexity**: O(n)
- **Supported Libraries**: pyvista
- **Recommended**: pyvista
- **Tags**: quality, boundary_layer, inflation, CFD, prism
- **Parameters**:
  - target_first_layer_height (optional): Target height for first layer (for y+ comparison)
  - max_growth_ratio (default: 1.2): Maximum layer-to-layer height ratio
  - max_aspect_ratio (default: 1000.0): Maximum height vs in-plane edge length ratio
  - max_orthogonality_angle (default: 20.0°): Maximum deviation from wall normal
  - min_layers (default: 3): Minimum number of boundary layer elements required
- **Checks**:
  - Layer count: minimum 3 layers recommended
  - First layer height: comparison against target (y+ proxy)
  - Growth ratio: layer-to-layer height ratio (threshold: 1.2)
  - Aspect ratio: height vs in-plane edge length (threshold: 1000)
  - Orthogonality: deviation from wall normal (threshold: 20°)
  - Coverage: fraction of wall surface covered by BL cells
- **Research Thresholds**:
  - Growth ratio: 1.0-1.2 excellent, 1.2-1.3 good, 1.3-1.5 acceptable, >1.5 poor
  - Orthogonality: <20° excellent, <45° good, <65° maximum (OpenFOAM)
- **Algorithm**: Identifies boundary layer cells adjacent to wall surfaces, traces prism/hex stacks outward, computes layer heights, growth ratios, aspect ratios, and orthogonality angles.

### Multi-Material Meshes

#### `quality_3d.detect_material_interfaces`
- **Display Name**: Material Interface Detection (3D)
- **Description**: Detects and reports faces at material boundaries in multi-material volumetric meshes
- **Complexity**: O(n)
- **Supported Libraries**: pyvista
- **Recommended**: pyvista
- **Tags**: quality, multi-material, interface, volumetric, 3d
- **Requirements**: MeshInput.cell_materials must be set
- **Output**: Per material-pair statistics (face count, cell count, total area)
- **Algorithm**: Identifies internal faces shared by cells with different material IDs, groups by material pair, computes interface statistics.

#### `quality_3d.check_interface_quality`
- **Display Name**: Interface Quality Check (3D)
- **Description**: Assesses quality of cells at material interfaces for poor shape metrics
- **Complexity**: O(n)
- **Supported Libraries**: pyvista
- **Recommended**: pyvista
- **Tags**: quality, multi-material, interface, volumetric, 3d
- **Parameters**:
  - jacobian_threshold (default: 0.1): Scaled Jacobian threshold for interface cells
  - volume_threshold (default: 1e-12): Volume threshold for interface cells
- **Checks**:
  - Scaled Jacobian below threshold (default: 0.1)
  - Volume below threshold (default: 1e-12)
- **Requirements**: MeshInput.cell_materials must be set
- **Algorithm**: Identifies cells adjacent to material interfaces and checks their quality metrics against thresholds.

#### `quality_3d.per_material_statistics`
- **Display Name**: Per-Material Statistics (3D)
- **Description**: Computes per-material-region quality statistics
- **Complexity**: O(n)
- **Supported Libraries**: pyvista
- **Recommended**: pyvista
- **Tags**: quality, multi-material, statistics, volumetric, 3d
- **Metrics per Material**:
  - Cell count
  - Total volume
  - Average/min/max volume
  - Average/min/max scaled Jacobian (tet4)
- **Requirements**: MeshInput.cell_materials must be set
- **Algorithm**: Groups cells by material ID and computes aggregate statistics for each material region.

---

## Operation Metadata Schema

All operations expose the following metadata via the `metadata` property:

| Field | Type | Description |
|-------|------|-------------|
| operation_id | str | Unique identifier (e.g., 'topology.detect_nonmanifold_edges') |
| display_name | str | Human-readable name |
| description | str | Detailed description of what the operation does |
| category | IssueCategory | TOPOLOGICAL, GEOMETRIC, QUALITY, or ORIENTATION |
| supported_libraries | List[str] | Libraries that can execute this operation |
| recommended_library | str | Preferred library for best results |
| requires_adjacency | bool | Whether the operation needs adjacency information (edge/face connectivity) |
| complexity | str | Computational complexity (e.g., 'O(n)', 'O(n log n)', 'O(V+F)') |
| tags | List[str] | Classification tags for filtering and grouping |

**Note on Complexity Notation:**
- O(n): Linear in number of elements (faces/cells)
- O(V+F): Linear in vertices + faces (surface meshes)
- O(E): Linear in number of edges
- O(n log n): Log-linear due to spatial data structures (KD-tree, sorting)
- O(F * k log k): Face count × log(vertices per face) for face hashing

---

## Severity Classification

Operations use the following severity levels based on violation rates:

| Severity | Color | Threshold | Interpretation |
|----------|-------|-----------|----------------|
| INFO | Blue | < 5% | Minor issue, informational only |
| WARNING | Yellow | 5-20% | Moderate issue, may need attention |
| ERROR | Orange | 20-50% | Significant issue, likely needs repair |
| CRITICAL | Red | > 50% | Severe issue, definitely needs repair |

**Specialized Severity Rules:**
- **Inverted cells** (volumetric): Always CRITICAL regardless of count
- **Non-manifold edges**: Boundary edges (1 face) are normal for open surfaces
- **Self-intersections**: Always flagged as ERROR or CRITICAL (geometric defect)
- **Degenerate cells**: CRITICAL when |volume| < tolerance

Note: Some operations use specialized thresholds based on domain-specific knowledge (e.g., boundary layer growth ratios, dihedral angle ranges).

---

## Supported Element Types (Volumetric)

| Element | Nodes | Description |
|---------|-------|-------------|
| tet4 | 4 | Tetrahedron (linear) |
| hex8 | 8 | Hexahedron (linear) |
| prism6 | 6 | Triangular prism (wedge) |
| pyramid5 | 5 | Pyramid with quadrilateral base |

---

## Usage Example

### Basic Usage: Single Operation

```python
from mesh_issue_examiner.operations import registry
from mesh_issue_examiner.core.models import MeshInput
import numpy as np

# Create a mesh
mesh = MeshInput(
    vertices=np.array([[0,0,0], [1,0,0], [0,1,0], [0,0,1]]),
    faces=np.array([[0,1,2], [0,1,3], [1,2,3], [0,2,3]]),
    mesh_id="example"
)

# Get an operation and execute
op = registry.get('topology.detect_nonmanifold_edges')
issues = op.execute(mesh)

# Check results
for issue in issues:
    print(f"{issue.severity.value}: {issue.message}")
```

### Workflow Usage: Multiple Operations

```python
from mesh_issue_examiner.operations import registry
from mesh_issue_examiner.core.models import MeshInput, OperationConfig

# Create workflow for all topological operations
workflow = registry.create_category_workflow(
    category=IssueCategory.TOPOLOGICAL,
    config=OperationConfig(enable_caching=True)
)

# Execute workflow
result = workflow(mesh)
print(f"Found {len(result.issues)} issues")
print(f"Processing time: {result.processing_time:.3f}s")

# Get summary
for issue in result.issues:
    print(f"{issue.severity.value}: {issue.message}")
```

### Volumetric Mesh Example

```python
# For volumetric meshes with cells
mesh = MeshInput(
    vertices=np.array([[0,0,0], [1,0,0], [0,1,0], [0,0,1]]),
    volume_cells={'tet4': np.array([[0,1,2,3]])},
    mesh_id="tet_example"
)

# Check volumetric quality
op = registry.get('quality_3d.compute_volume_cell_quality')
issues = op.execute(mesh)
```

### Listing Available Operations

```python
# List all operations
all_ops = registry.list_all()
print(f"Total operations: {len(all_ops)}")

# List by category
topo_ops = registry.list_by_category(IssueCategory.TOPOLOGICAL)
print(f"Topological operations: {len(topo_ops)}")

# List by library
pyvista_ops = registry.list_by_library('pyvista')
print(f"PyVista-compatible operations: {len(pyvista_ops)}")

# List by tag
quality_ops = registry.list_by_tag('quality')
print(f"Quality operations: {len(quality_ops)}")
```

---

## Quick Reference: All 21 Operations

| # | Operation ID | Category | Libraries | Requires Adjacency |
|---|--------------|----------|-----------|-------------------|
| 1 | `topology.detect_nonmanifold_edges` | TOPOLOGICAL | trimesh, pyvista, pymeshlab | Yes |
| 2 | `topology.detect_nonmanifold_vertices` | TOPOLOGICAL | trimesh, pyvista, pymeshlab | Yes |
| 3 | `topology.detect_self_intersections` | TOPOLOGICAL | pyvista | Yes |
| 4 | `topology.detect_duplicate_vertices` | TOPOLOGICAL | trimesh, pyvista, pymeshlab, meshio, vedo | No |
| 5 | `topology.detect_boundary_edges` | TOPOLOGICAL | trimesh, pyvista, pymeshlab | Yes |
| 6 | `topology.detect_disconnected_components` | TOPOLOGICAL | trimesh, pyvista, pymeshlab | Yes |
| 7 | `topology.detect_isolated_vertices` | TOPOLOGICAL | trimesh, pyvista, pymeshlab | Yes |
| 8 | `topology.detect_duplicate_faces` | TOPOLOGICAL | trimesh, pyvista, pymeshlab | No |
| 9 | `topology.detect_vertex_valence` | TOPOLOGICAL | pyvista, trimesh | Yes |
| 10 | `topology_3d.detect_nonmanifold_faces` | TOPOLOGICAL | pyvista | Yes |
| 11 | `topology.detect_inconsistent_orientation` | ORIENTATION | trimesh, pyvista, pymeshlab | Yes |
| 12 | `quality_3d.check_orientation_consistency` | ORIENTATION | pyvista | Yes |
| 13 | `topology.detect_edge_dihedral_angle` | GEOMETRIC | pyvista, trimesh | Yes |
| 14 | `topology.detect_edge_length_ratio` | GEOMETRIC | pyvista, trimesh | No |
| 15 | `quality_3d.assess_geometric_validity` | GEOMETRIC | pyvista | No |
| 16 | `quality.compute_cell_quality` | QUALITY | pyvista, trimesh | No |
| 17 | `quality_3d.compute_volume_cell_quality` | QUALITY | pyvista | No |
| 18 | `quality_3d.check_boundary_layer` | QUALITY | pyvista | Yes |
| 19 | `quality_3d.detect_material_interfaces` | QUALITY | pyvista | Yes |
| 20 | `quality_3d.check_interface_quality` | QUALITY | pyvista | Yes |
| 21 | `quality_3d.per_material_statistics` | QUALITY | pyvista | No |

**Note on ID Prefixes**: Some operations have prefixes that differ from their category:
- `topology.detect_edge_dihedral_angle` and `topology.detect_edge_length_ratio` are GEOMETRIC
- `quality_3d.assess_geometric_validity` is GEOMETRIC (not QUALITY)
- `topology.detect_vertex_valence` is TOPOLOGICAL (not GEOMETRIC)

---

## References

1. Jiao & He, "AHF: Array-based Half-Facet Data Structure for Mixed-Dimensional and Non-manifold Meshes", 2014
2. Borodin, Zachmann, & Klein, "Consistent Normal Orientation for Polygonal Meshes", CGI 2004
3. Möller & Trumbore, "Fast, Minimum Storage Ray-Triangle Intersection", 1997
4. OpenFOAM User Guide - Mesh quality recommendations
5. ANSYS Fluent User's Guide - Mesh quality criteria
6. FEA Academy - Material interface guidelines

---

## Document History

| Date | Author | Changes |
|------|--------|---------|
| 2026-04-11 | Initial | Initial catalog creation |
| 2026-04-11 | Review | Updated to reflect 21 operations (was 20), corrected category counts, added algorithm descriptions |
| 2026-04-11 | Review | Enhanced precision with: detailed algorithm descriptions, parameter specifications, complexity notation explanations, specialized severity rules, expanded usage examples |
| 2026-04-11 | **Precision** | **Multi-stage precision revision:**<br>- Corrected category counts: TOPOLOGICAL (9), GEOMETRIC (3), ORIENTATION (2), QUALITY (7)<br>- Added missing `topology_3d.detect_nonmanifold_faces` operation with full details<br>- Added `topology.detect_vertex_valence` to TOPOLOGICAL section (was missing)<br>- Reorganized GEOMETRIC section: clarified `topology.detect_edge_dihedral_angle` and `topology.detect_edge_length_ratio` are GEOMETRIC per metadata<br>- Added **Category Note** headers to explain metadata-based organization<br>- Added **Implementation Notes** subsections with caps (e.g., `_MAX_LOCATIONS = 100`)<br>- Added detailed **Algorithm** step-by-step breakdowns for all operations<br>- Added element-specific edge tables for sliver detection<br>- Documented severity threshold variations per operation |
