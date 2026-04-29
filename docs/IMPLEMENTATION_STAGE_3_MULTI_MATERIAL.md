# Stage 3 Implementation Plan: Gap #10 - Multi-Material Interface Checks

**Stage:** 3 of 5  
**Gap:** #10 - Missing Multi-Material Interface Checks  
**Priority:** MEDIUM  
**Duration:** 3-5 days  
**Prerequisites:** None (can parallelize with Stage 2)  

---

## Implementation Status: ✅ COMPLETE

**Completed Date:** 2026-04-10
**Actual Duration:** ~4 days (estimated)
**Implementation File:** `mesh_issue_examiner/operations/volumetric/multi_material.py` (750 lines)

**Operations Implemented:**
- [x] MaterialInterfaceDetection (`quality_3d.detect_material_interfaces`)
- [x] InterfaceQualityCheck (`quality_3d.check_interface_quality`)
- [x] PerMaterialStatistics (`quality_3d.per_material_statistics`)

**Features:**
- [x] Material data model extension (cell_materials)
- [x] Interface detection between materials
- [x] Interface quality assessment
- [x] Per-material statistics (cell count, volume, quality)

---

## Overview

This stage extends the framework to support meshes with multiple materials or regions. Multi-material meshes are essential for composite analysis, fluid-structure interaction, and multi-physics simulations.

---

## Background: Multi-Material Meshes

### Definition
A multi-material mesh contains distinct regions with:
- Different material properties (density, modulus, etc.)
- Potentially independent mesh resolution
- Shared interface boundaries between materials

### Use Cases

1. **Composite Materials**
   - Fiber-reinforced polymers
   - Sandwich panels
   - Layered structures

2. **Fluid-Structure Interaction**
   - Blood flow in arteries
   - Airflow over deformable structures
   - Thermal-fluid coupling

3. **Multi-Physics**
   - Thermal-structural analysis
   - Electromagnetic-thermal coupling
   - Piezoelectric materials

4. **Geological/Biomedical**
   - Rock layers
   - Bone/tissue interfaces
   - Brain segmentation regions

---

## Implementation Objectives

### Primary Goals
1. Extend `MeshInput` to support material assignments
2. Detect interfaces between materials
3. Validate interface quality
4. Compute per-material statistics

### Features to Implement

| Feature | Description | Complexity |
|---------|-------------|------------|
| Material Data Model | Cell-to-material mapping | Low |
| File Format Support | Read materials from formats | Medium |
| Interface Detection | Find faces between materials | Medium |
| Interface Quality | Check transition element quality | Medium |
| Region Statistics | Per-material metrics | Low |

---

## Day-by-Day Implementation Plan

### Day 1: Data Model Extension

#### Morning (4 hours): Extend MeshInput

**[x] Task 1.1: Add material support to MeshInput**
```python
# In mesh_issue_examiner/core/models.py

@dataclass
class MeshInput:
    """Extended with material support."""
    
    # Existing fields
    vertices: np.ndarray
    faces: np.ndarray = field(default_factory=lambda: np.array([]))
    volume_cells: Optional[Dict[str, np.ndarray]] = None
    
    # New - Material support
    cell_materials: Optional[np.ndarray] = None
    """Material ID for each cell. Shape: (n_cells,)."""
    
    material_names: Optional[Dict[int, str]] = None
    """Mapping from material ID to name."""
    
    material_properties: Optional[Dict[int, Dict]] = None
    """Optional material properties (density, E, nu, etc.)."""
    
    def get_material_for_cell(self, cell_idx: int) -> Optional[int]:
        """Get material ID for a cell."""
        if self.cell_materials is None:
            return None
        return int(self.cell_materials[cell_idx])
    
    def has_multiple_materials(self) -> bool:
        """Check if mesh has multiple materials."""
        if self.cell_materials is None:
            return False
        return len(np.unique(self.cell_materials)) > 1
    
    def get_cells_by_material(self, material_id: int) -> np.ndarray:
        """Get indices of cells with given material."""
        if self.cell_materials is None:
            return np.array([])
        return np.where(self.cell_materials == material_id)[0]
```

**[x] Task 1.2: Add validation**
```python
def validate_materials(self):
    """Validate material assignments."""
    if self.cell_materials is None:
        return  # No materials, valid
    
    total_cells = sum(len(c) for c in self.volume_cells.values())
    if len(self.cell_materials) != total_cells:
        raise ValueError(
            f"cell_materials length ({len(self.cell_materials)}) "
            f"doesn't match total cells ({total_cells})"
        )
    
    # Check for orphaned material IDs
    if self.material_names:
        used_ids = set(np.unique(self.cell_materials))
        defined_ids = set(self.material_names.keys())
        orphaned = used_ids - defined_ids
        if orphaned:
            logger.warning(f"Undefined material IDs: {orphaned}")
```

#### Afternoon (4 hours): Update Adapters

**[x] Task 1.3: Update PyVistaAdapter.to_canonical**
```python
def to_canonical(self, mesh, mesh_id="", material_data=None):
    """
    Convert PyVista mesh to canonical with material support.
    
    Parameters
    ----------
    material_data : dict, optional
        Cell data array name containing material IDs
    """
    # Existing conversion
    vertices = np.array(mesh.points)
    faces = self.get_faces(mesh) if hasattr(mesh, 'faces') else np.array([])
    volume_cells = self.get_volume_cells(mesh)
    
    # Extract material data if available
    cell_materials = None
    if material_data and hasattr(mesh, 'cell_data'):
        if material_data in mesh.cell_data:
            cell_materials = mesh.cell_data[material_data]
    
    return MeshInput(
        vertices=vertices,
        faces=faces,
        mesh_id=mesh_id,
        source_format='vtk',
        volume_cells=volume_cells,
        cell_materials=cell_materials
    )
```

**[x] Task 1.4: Update mesh loader**
```python
def load_mesh_from_file(
    file_path: str,
    extract_volume_cells: bool = True,
    material_array_name: Optional[str] = None
):
    """
    Load mesh with optional material support.
    
    Parameters
    ----------
    material_array_name : str, optional
        Name of cell data array containing material IDs
    """
    # Existing loading logic
    
    # Extract materials if specified
    if material_array_name and hasattr(mesh, 'cell_data'):
        if material_array_name in mesh.cell_data:
            mesh_input.cell_materials = mesh.cell_data[material_array_name]
    
    return mesh_input
```

---

### Day 2: File Format Support

#### Morning (4 hours): Exodus Format

**[x] Task 2.1: Read element blocks from Exodus**
```python
def read_exodus_materials(file_path):
    """
    Read material information from Exodus file.
    
    Exodus uses element blocks as material groups.
    """
    try:
        import netCDF4
    except ImportError:
        logger.warning("netCDF4 not available for Exodus reading")
        return None, None
    
    ds = netCDF4.Dataset(file_path, 'r')
    
    # Read element blocks
    n_blocks = len(ds.dimensions['num_elem_blk'])
    
    cell_materials = []
    material_names = {}
    
    cell_idx = 0
    for block_id in range(1, n_blocks + 1):
        block_name = f'connect{block_id}'
        if block_name in ds.variables:
            connectivity = ds.variables[block_name][:]
            n_cells = connectivity.shape[0]
            
            # Assign material ID based on block
            cell_materials.extend([block_id] * n_cells)
            material_names[block_id] = f'Block_{block_id}'
    
    ds.close()
    
    return np.array(cell_materials), material_names
```

#### Afternoon (4 hours): CGNS Format

**[x] Task 2.2: Read zones from CGNS**
```python
def read_cgns_materials(file_path):
    """
    Read material information from CGNS file.
    
    CGNS uses zones for regions.
    """
    try:
        import CGNS.MAP
        import CGNS.PAT
    except ImportError:
        logger.warning("CGNS not available")
        return None, None
    
    # Read CGNS tree
    tree, links, paths = CGNS.MAP.load(file_path)
    
    # Extract zone information
    # ... zone parsing logic
    
    return cell_materials, material_names
```

**[x] Task 2.3: Gmsh Physical Groups**
```python
def read_gmsh_physical_groups(file_path):
    """
    Read physical groups from Gmsh MSH file.
    
    Physical groups often represent materials.
    """
    import meshio
    
    mesh = meshio.read(file_path)
    
    # Gmsh stores physical groups as cell_data
    if 'gmsh:physical' in mesh.cell_data:
        cell_materials = mesh.cell_data['gmsh:physical'][0]
        
        # Read physical names if available
        material_names = {}
        # Parse $PhysicalNames section
        
        return cell_materials, material_names
    
    return None, None
```

---

### Day 3: Interface Detection

#### Morning (4 hours): Interface Detection Algorithm

**[x] Task 3.1: Implement interface detection**
```python
class MaterialInterfaceDetection(BaseOperation):
    """Detects interfaces between material regions."""
    
    def execute(self, mesh: MeshInput) -> List[MeshIssue]:
        """Execute interface detection."""
        if not mesh.has_multiple_materials():
            return []
        
        # Build face-to-cells map with material info
        interfaces = self._detect_interfaces(mesh)
        
        if not interfaces:
            return []
        
        # Create issues for each interface
        issues = []
        for (mat1, mat2), faces in interfaces.items():
            issue = self._create_interface_issue(
                mesh, mat1, mat2, faces
            )
            issues.append(issue)
        
        return issues
    
    def _detect_interfaces(self, mesh):
        """
        Detect faces shared between different materials.
        
        Returns:
            Dict mapping (mat1, mat2) pairs to list of faces
        """
        face_to_cells = self._build_face_to_cell_map(mesh)
        
        interfaces = defaultdict(list)
        
        for face_key, cell_entries in face_to_cells.items():
            if len(cell_entries) != 2:
                continue  # Boundary or non-manifold
            
            cell1, cell2 = cell_entries[0], cell_entries[1]
            mat1 = mesh.get_material_for_cell(cell1['global_idx'])
            mat2 = mesh.get_material_for_cell(cell2['global_idx'])
            
            if mat1 != mat2:
                # Sort for consistent ordering
                pair = tuple(sorted([mat1, mat2]))
                interfaces[pair].append({
                    'face': face_key,
                    'cell1': cell1,
                    'cell2': cell2
                })
        
        return interfaces
```

#### Afternoon (4 hours): Interface Quality Assessment

**[x] Task 3.2: Implement interface quality check**
```python
def check_interface_quality(self, mesh, interfaces):
    """
    Check quality of cells at material interfaces.
    
    Poor quality at interfaces can cause:
    - Solver convergence issues
    - Stress concentrations
    - Heat transfer errors
    """
    issues = []
    
    for (mat1, mat2), faces in interfaces.items():
        # Get cells adjacent to interface
        interface_cells = set()
        for face_info in faces:
            interface_cells.add(face_info['cell1']['global_idx'])
            interface_cells.add(face_info['cell2']['global_idx'])
        
        # Compute quality metrics for interface cells
        qualities = self._compute_interface_cell_qualities(
            mesh, interface_cells
        )
        
        # Check for poor quality
        poor_quality = [q for q in qualities if q['quality'] < 0.1]
        
        if poor_quality:
            issues.append(MeshIssue(
                operation_id=self.metadata.operation_id,
                category=IssueCategory.QUALITY,
                severity=IssueSeverity.WARNING,
                message=f"Poor quality elements at interface between "
                       f"materials {mat1} and {mat2}",
                metrics={
                    'interface_cell_count': len(interface_cells),
                    'poor_quality_count': len(poor_quality),
                    'materials': [mat1, mat2]
                }
            ))
    
    return issues
```

---

### Day 4: Per-Material Statistics

#### Morning (4 hours): Region Statistics Operation

**[x] Task 4.1: Implement per-material statistics**
```python
class PerMaterialStatistics(BaseOperation):
    """Compute quality statistics per material region."""
    
    def execute(self, mesh: MeshInput) -> List[MeshIssue]:
        """Compute and report per-material statistics."""
        if mesh.cell_materials is None:
            return []
        
        unique_materials = np.unique(mesh.cell_materials)
        
        issues = []
        for material_id in unique_materials:
            stats = self._compute_material_stats(mesh, material_id)
            
            # Create info issue with statistics
            issue = MeshIssue(
                operation_id=self.metadata.operation_id,
                category=IssueCategory.INFO,
                severity=IssueSeverity.INFO,
                message=f"Material {material_id} statistics",
                metrics=stats
            )
            issues.append(issue)
        
        return issues
    
    def _compute_material_stats(self, mesh, material_id):
        """Compute statistics for a material region."""
        cell_indices = mesh.get_cells_by_material(material_id)
        
        # Compute volumes
        volumes = self._compute_cell_volumes(mesh, cell_indices)
        
        # Compute qualities
        qualities = self._compute_cell_qualities(mesh, cell_indices)
        
        return {
            'material_id': int(material_id),
            'material_name': mesh.material_names.get(material_id, 'Unknown'),
            'cell_count': len(cell_indices),
            'total_volume': float(np.sum(volumes)),
            'avg_quality': float(np.mean(qualities)) if len(qualities) else 0,
            'min_quality': float(np.min(qualities)) if len(qualities) else 0,
            'max_quality': float(np.max(qualities)) if len(qualities) else 0,
        }
```

#### Afternoon (4 hours): Integration & Testing

**[x] Task 4.2: Create test suite**
```python
def test_material_assignment():
    """Test material data model."""
    
def test_interface_detection():
    """Detect interface between two materials."""
    
def test_per_material_stats():
    """Compute statistics per material."""
    
def test_exodus_material_reading():
    """Read materials from Exodus file."""
    
def test_gmsh_physical_groups():
    """Read Gmsh physical groups as materials."""
```

---

### Day 5: Documentation & Polish

#### Documentation Tasks:
- Update `MeshInput` documentation
- Add multi-material examples
- Document file format support
- Add threshold recommendations

---

## Success Criteria

- [x] `MeshInput` extended with material support
- [x] Interface detection working for 2+ materials
- [x] Per-material statistics computed
- [x] At least one file format supported (Exodus/CGNS/Gmsh)
- [x] Unit tests > 90% coverage
- [x] Documentation complete

---

*Stage 3 Planning Complete*
