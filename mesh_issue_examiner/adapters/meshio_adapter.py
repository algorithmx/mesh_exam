"""
Meshio library adapter implementation.

Meshio provides I/O for many mesh formats. This adapter uses meshio for
file loading/saving and delegates processing to PyVista when needed.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np

from .base_adapter import MeshAdapter
from .exceptions import QualityMetricError, FileIOError, MeshConversionError

logger = logging.getLogger(__name__)


class MeshioAdapter(MeshAdapter):
    """
    Adapter for the meshio library.

    Meshio supports reading and writing over 30 mesh file formats including:
    - Abaqus (.inp)
    - ANSYS (.msh)
    - CGNS (.cgns)
    - DOLFIN XML (.xml)
    - Exodus (.e, .exo)
    - FLAC3D (.f3grid)
    - Gmsh (.msh, versions 2.2, 4.0, 4.1)
    - MED/Salome (.med)
    - NASTRAN (.bdf, .fem, .nas)
    - TetGen (.node/.ele)
    - VTK (.vtk)
    - VTU (.vtu)
    - XDMF (.xdmf, .xmf)

    **Capabilities:**
    - Surface meshes: Full support
    - Volumetric meshes: Full support (tet4, hex8, prism6, pyramid5)
    - File I/O: 30+ formats
    - Quality metrics: Via PyVista delegation for volumetric
    
    **Notes:**
    - For volumetric quality metrics, delegates to PyVista
    - Preserves all element types through to_canonical/from_canonical
    - Caches PyVista conversions for multiple quality metric calls
    
    **Element Type Mapping:**
    - tet4 -> tetra
    - hex8 -> hexahedron
    - prism6 -> wedge
    - pyramid5 -> pyramid
    - tet10 -> tetra10 (quadratic)
    - hex20 -> hexahedron20 (quadratic)
    """

    def __init__(self):
        # Cache for PyVista conversions to avoid repeated conversions
        # when computing multiple quality metrics
        self._pv_grid_cache: Dict[int, Any] = {}

    @property
    def capabilities(self) -> 'AdapterCapability':
        """Return capabilities supported by meshio."""
        from .capabilities import AdapterCapability
        return (
            AdapterCapability.SURFACE_MESH |
            AdapterCapability.VOLUMETRIC_MESH |
            AdapterCapability.FILE_IO |
            AdapterCapability.QUALITY_METRICS |
            AdapterCapability.TOPOLOGY_CHECKS
        )

    @property
    def library_name(self) -> str:
        return 'meshio'

    def _check_import(self):
        """Check that meshio is importable."""
        import meshio
        return meshio

    def load_mesh(self, file_path: str, **kwargs) -> 'meshio.Mesh':
        """Load mesh from file using meshio.

        Args:
            file_path: Path to mesh file
            **kwargs: Additional meshio-specific arguments

        Returns:
            meshio.Mesh object
            
        Raises:
            FileIOError: If file cannot be loaded
        """
        import meshio
        from .exceptions import FileIOError
        
        path = Path(file_path)
        if not path.exists():
            raise FileIOError(f"Mesh file not found: {file_path}")
        
        try:
            return meshio.read(file_path, **kwargs)
        except Exception as e:
            raise FileIOError(
                f"Failed to load mesh from {file_path} using meshio: {e}"
            ) from e

    def create_mesh(self, vertices: np.ndarray, faces: np.ndarray) -> 'meshio.Mesh':
        """Create meshio mesh from arrays."""
        import meshio

        # meshio uses cells dictionary format
        cells = [("triangle", np.array(faces))] if len(faces) > 0 else []

        return meshio.Mesh(
            points=np.array(vertices),
            cells=cells
        )

    def from_canonical(self, mesh_input) -> 'meshio.Mesh':
        """
        Create meshio mesh from canonical format.

        Supports both surface and volumetric meshes.
        """
        import meshio

        cells = []

        # Add volume cells if present
        if mesh_input.has_volume_cells:
            etype_map = {
                'tet4': 'tetra',
                'hex8': 'hexahedron',
                'prism6': 'wedge',
                'pyramid5': 'pyramid',
            }
            for etype, connectivity in mesh_input._volume_cells.items():
                meshio_type = etype_map.get(etype)
                if meshio_type:
                    cells.append((meshio_type, connectivity))

        # Add surface faces if present
        if len(mesh_input.faces) > 0:
            cells.append(("triangle", mesh_input.faces))

        return meshio.Mesh(
            points=mesh_input._vertices,
            cells=cells
        )

    def to_canonical(self, mesh: 'meshio.Mesh', mesh_id: str = "") -> 'MeshInput':
        """
        Convert meshio mesh to canonical MeshInput format.

        Automatically extracts volumetric cells and surface faces.
        """
        from ..core.models import MeshInput

        vertices = mesh.points

        # Extract cells by type
        volume_cells = {}
        faces = np.array([])

        # meshio cell type names to our internal names
        volume_type_map = {
            'tetra': 'tet4',
            'hexahedron': 'hex8',
            'wedge': 'prism6',
            'pyramid': 'pyramid5',
            # Quadratic variants
            'tetra10': 'tet10',
            'hexahedron20': 'hex20',
            'wedge15': 'wedge15',
            'pyramid13': 'pyramid13',
        }

        surface_types = {'triangle', 'quad', 'triangle6', 'quad9'}

        for cell_block in mesh.cells:
            cell_type = cell_block.type
            connectivity = cell_block.data

            if cell_type in volume_type_map:
                internal_type = volume_type_map[cell_type]
                volume_cells[internal_type] = connectivity
            elif cell_type in surface_types:
                # For surface meshes, use triangle data
                if cell_type in ('triangle', 'triangle6'):
                    faces = connectivity

        return MeshInput(
            vertices=vertices,
            faces=faces,
            mesh_id=mesh_id,
            source_format='meshio',
            volume_cells=volume_cells if volume_cells else None
        )

    def get_vertices(self, mesh: 'meshio.Mesh') -> np.ndarray:
        """Extract vertex array."""
        return np.array(mesh.points)

    def get_faces(self, mesh: 'meshio.Mesh') -> np.ndarray:
        """
        Extract face array.

        Returns triangular faces if present, otherwise empty array.
        """
        for cell_block in mesh.cells:
            if cell_block.type in ('triangle', 'triangle6'):
                return np.array(cell_block.data)
        return np.array([])

    def get_volume_cells(self, mesh: 'meshio.Mesh') -> Dict[str, np.ndarray]:
        """Extract volumetric cells as dictionary."""
        volume_type_map = {
            'tetra': 'tet4',
            'hexahedron': 'hex8',
            'wedge': 'prism6',
            'pyramid': 'pyramid5',
            'tetra10': 'tet10',
            'hexahedron20': 'hex20',
        }

        volume_cells = {}
        for cell_block in mesh.cells:
            internal_type = volume_type_map.get(cell_block.type)
            if internal_type:
                volume_cells[internal_type] = np.array(cell_block.data)
        return volume_cells

    def is_watertight(self, mesh: 'meshio.Mesh') -> bool:
        """
        Check if mesh is watertight.

        For surface meshes, checks if boundary edges form closed loops.
        For volumetric meshes, assumes watertight.
        """
        faces = self.get_faces(mesh)
        if len(faces) == 0:
            # Volumetric mesh - check if it has volume
            return True  # Assume watertight for pure volumetric

        # Build edge to face count
        edges = {}
        for face in faces:
            for i in range(3):
                v1, v2 = int(face[i]), int(face[(i + 1) % 3])
                edge = tuple(sorted([v1, v2]))
                edges[edge] = edges.get(edge, 0) + 1

        # Check for boundary edges (count == 1)
        boundary_edges = sum(1 for count in edges.values() if count == 1)
        return boundary_edges == 0

    def is_manifold(self, mesh: 'meshio.Mesh') -> Tuple[bool, List[Dict[str, Any]]]:
        """Check if mesh is manifold."""
        issues = []
        faces = self.get_faces(mesh)

        if len(faces) == 0:
            return True, issues

        # Build edge to face count
        edges = {}
        for face_idx, face in enumerate(faces):
            for i in range(3):
                v1, v2 = int(face[i]), int(face[(i + 1) % 3])
                edge = tuple(sorted([v1, v2]))
                if edge not in edges:
                    edges[edge] = []
                edges[edge].append(face_idx)

        # Find non-manifold edges (shared by != 2 faces)
        nonmanifold_edges = [edge for edge, face_list in edges.items() if len(face_list) != 2]
        if nonmanifold_edges:
            issues.append({
                'type': 'non_manifold_edges',
                'count': len(nonmanifold_edges)
            })

        return len(issues) == 0, issues

    def get_boundary_edges(self, mesh: 'meshio.Mesh') -> np.ndarray:
        """Get boundary edges."""
        faces = self.get_faces(mesh)
        if len(faces) == 0:
            return np.array([])

        edges = {}
        for face in faces:
            for i in range(3):
                v1, v2 = int(face[i]), int(face[(i + 1) % 3])
                edge = tuple(sorted([v1, v2]))
                edges[edge] = edges.get(edge, 0) + 1

        boundary = [edge for edge, count in edges.items() if count == 1]
        return np.array(boundary)

    def get_nonmanifold_edges(self, mesh: 'meshio.Mesh') -> List[Tuple[int, int, int]]:
        """Detect non-manifold edges."""
        faces = self.get_faces(mesh)
        if len(faces) == 0:
            return []

        edges = {}
        for face_idx, face in enumerate(faces):
            for i in range(3):
                v1, v2 = int(face[i]), int(face[(i + 1) % 3])
                edge = tuple(sorted([v1, v2]))
                if edge not in edges:
                    edges[edge] = []
                edges[edge].append(face_idx)

        return [(edge[0], edge[1], len(face_list)) for edge, face_list in edges.items() if len(face_list) != 2]

    def get_duplicate_vertices(
        self,
        mesh: 'meshio.Mesh',
        tolerance: float
    ) -> List[int]:
        """Find duplicate vertices."""
        from scipy.spatial import cKDTree

        vertices = self.get_vertices(mesh)
        tree = cKDTree(vertices)
        duplicates = set()

        for i, vertex in enumerate(vertices):
            neighbors = tree.query_ball_point(vertex, tolerance)
            for j in neighbors:
                if j > i:
                    duplicates.add(j)

        return list(duplicates)

    def compute_face_areas(self, mesh: 'meshio.Mesh') -> np.ndarray:
        """Compute area of each face."""
        faces = self.get_faces(mesh)
        vertices = self.get_vertices(mesh)

        if len(faces) == 0:
            return np.array([])

        areas = np.zeros(len(faces))
        for i, face in enumerate(faces):
            v = vertices[face]
            edge1 = v[1] - v[0]
            edge2 = v[2] - v[0]
            cross = np.cross(edge1, edge2)
            areas[i] = np.linalg.norm(cross) / 2

        return areas

    def compute_face_normals(self, mesh: 'meshio.Mesh') -> np.ndarray:
        """Compute face normals."""
        faces = self.get_faces(mesh)
        vertices = self.get_vertices(mesh)

        if len(faces) == 0:
            return np.array([])

        normals = np.zeros((len(faces), 3))
        for i, face in enumerate(faces):
            v = vertices[face]
            edge1 = v[1] - v[0]
            edge2 = v[2] - v[0]
            cross = np.cross(edge1, edge2)
            norm = np.linalg.norm(cross)
            if norm > 0:
                normals[i] = cross / norm

        return normals

    def detect_self_intersections(self, mesh: 'meshio.Mesh') -> List[Tuple[int, int]]:
        """Detect self-intersections."""
        # meshio doesn't have built-in self-intersection detection
        # Delegate to trimesh if available
        try:
            import trimesh
            faces = self.get_faces(mesh)
            vertices = self.get_vertices(mesh)
            if len(faces) == 0:
                return []
            tm = trimesh.Trimesh(vertices=vertices, faces=faces)
            return tm.face_adjacency[tm.face_adjacency_convex == False].tolist()
        except Exception:
            return []

    def compute_cell_quality(
        self,
        mesh: 'meshio.Mesh',
        quality_type: str
    ) -> np.ndarray:
        """
        Compute cell quality metrics.

        For volumetric meshes, converts to PyVista and uses VTK.
        For surface meshes, computes manually.
        
        Uses caching to avoid repeated conversions when computing
        multiple quality metrics on the same mesh.

        Args:
            mesh: meshio.Mesh object
            quality_type: Type of quality metric to compute
            
        Returns:
            Array of quality values
            
        Raises:
            QualityMetricError: If quality computation fails
        """
        # Check if we have volumetric cells
        volume_cells = self.get_volume_cells(mesh)
        
        if volume_cells:
            # Use PyVista for volumetric quality
            return self._compute_volumetric_quality(mesh, quality_type)

        # Surface mesh - compute manually
        if quality_type == 'area':
            return self.compute_face_areas(mesh)
        elif quality_type == 'aspect_ratio':
            return self._compute_aspect_ratio(mesh)

        return np.array([])

    def _compute_volumetric_quality(
        self,
        mesh: 'meshio.Mesh',
        quality_type: str
    ) -> np.ndarray:
        """Compute volumetric cell quality via PyVista with caching.
        
        Args:
            mesh: meshio.Mesh with volumetric cells
            quality_type: Type of quality metric
            
        Returns:
            Array of quality values
            
        Raises:
            QualityMetricError: If computation fails
        """
        # Use mesh id for caching (hash of mesh data)
        mesh_id = id(mesh)
        
        try:
            # Check cache for existing PyVista grid
            if mesh_id in self._pv_grid_cache:
                grid = self._pv_grid_cache[mesh_id]
            else:
                # Convert to MeshInput and get PyVista representation
                mesh_input = self.to_canonical(mesh)
                grid = mesh_input.get_native('pyvista')
                # Cache the grid for future calls
                self._pv_grid_cache[mesh_id] = grid
                
                # Limit cache size to prevent memory issues
                if len(self._pv_grid_cache) > 10:
                    # Remove oldest entry
                    oldest_key = next(iter(self._pv_grid_cache))
                    del self._pv_grid_cache[oldest_key]
            
            # Use PyVista's cell quality
            import pyvista as pv
            quality_mesh = grid.cell_quality(quality_measure=quality_type)
            values = quality_mesh.cell_data.get(quality_type, np.array([]))
            return values
            
        except QualityMetricError:
            raise
        except Exception as e:
            raise QualityMetricError(
                f"Failed to compute volumetric quality metric '{quality_type}': {e}. "
                f"Ensure pyvista is installed for volumetric mesh support."
            ) from e

    def _compute_aspect_ratio(self, mesh: 'meshio.Mesh') -> np.ndarray:
        """Compute aspect ratio for triangular faces."""
        faces = self.get_faces(mesh)
        vertices = self.get_vertices(mesh)

        if len(faces) == 0:
            return np.array([])

        aspect_ratios = []
        for face in faces:
            v = vertices[face]
            edges = [
                np.linalg.norm(v[1] - v[0]),
                np.linalg.norm(v[2] - v[1]),
                np.linalg.norm(v[0] - v[2])
            ]
            min_edge = min(edges)
            max_edge = max(edges)
            aspect_ratios.append(max_edge / min_edge if min_edge > 0 else 0)

        return np.array(aspect_ratios)

    def save_mesh(self, mesh: 'meshio.Mesh', file_path: str, **kwargs) -> None:
        """
        Save mesh to file.

        Args:
            mesh: meshio.Mesh object
            file_path: Output file path
            **kwargs: Additional meshio.write arguments
            
        Raises:
            FileIOError: If save fails
        """
        import meshio
        from .exceptions import FileIOError
        
        try:
            meshio.write(file_path, mesh, **kwargs)
        except Exception as e:
            raise FileIOError(
                f"meshio failed to save mesh to {file_path}: {e}"
            ) from e

    def get_supported_formats(self) -> List[str]:
        """Return list of supported file extensions."""
        return [
            '.inp', '.msh', '.avs', '.cgns', '.xml',
            '.e', '.exo', '.f3grid', '.h5m', '.mdpa',
            '.mesh', '.meshb', '.med', '.bdf', '.fem',
            '.nas', '.node', '.ele', '.obj', '.off',
            '.post', '.dato', '.ply', '.stl', '.dat',
            '.su2', '.ugrid', '.vtk', '.vtu', '.xdmf', '.xmf'
        ]

    def clear_cache(self):
        """Clear the PyVista grid cache.
        
        This frees memory used by cached PyVista conversions.
        """
        self._pv_grid_cache.clear()
        logger.debug("MeshioAdapter cache cleared")
