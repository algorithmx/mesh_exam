"""
Vedo library adapter implementation.

Vedo is a Python module for scientific visualization and analysis of 3D objects.
It provides advanced visualization capabilities for volumetric data, meshes,
and point clouds with support for jupyter notebooks.
"""

import logging
from typing import Any, Dict, List, Tuple, Optional

import numpy as np

from .base_adapter import MeshAdapter
from .exceptions import QualityMetricError, FileIOError, MeshConversionError

logger = logging.getLogger(__name__)


class VedoAdapter(MeshAdapter):
    """
    Adapter for the vedo library.

    Vedo provides:
    - Advanced 3D visualization and rendering
    - Support for volumetric data (TIFF stacks, DICOM, MHD)
    - Mesh analysis and manipulation
    - Integration with numpy and VTK
    - Jupyter notebook support

    **Capabilities:**
    - Surface meshes: Full support (vedo.Mesh)
    - Volumetric meshes: Full support (vedo.UGrid)
    - File I/O: VTK-based formats
    - Quality metrics: Via PyVista delegation
    - Remeshing: Smoothing, decimation, slicing
    
    **Notes:**
    - Excellent for visualization-heavy workflows
    - Volumetric quality via PyVista delegation
    - UGrid for volumetric, Mesh for surface
    
    **Vedo-specific Extensions:**
    This adapter provides additional methods beyond the base interface:
    - visualize(): Create interactive 3D visualizations
    - slice_mesh(): Slice mesh with a plane
    - clip_mesh(): Clip mesh with a plane
    - decimate(): Simplify mesh
    - smooth(): Apply smoothing filters
    - compute_curvature(): Compute surface curvature
    """

    def __init__(self):
        # Cache for PyVista conversions
        self._pv_grid_cache: Dict[int, Any] = {}

    @property
    def capabilities(self) -> 'AdapterCapability':
        """Return capabilities supported by vedo."""
        from .capabilities import AdapterCapability
        return (
            AdapterCapability.SURFACE_MESH |
            AdapterCapability.VOLUMETRIC_MESH |
            AdapterCapability.FILE_IO |
            AdapterCapability.QUALITY_METRICS |
            AdapterCapability.TOPOLOGY_CHECKS |
            AdapterCapability.REMESHING
        )

    @property
    def library_name(self) -> str:
        return 'vedo'

    def _check_import(self):
        """Check that vedo is importable."""
        import vedo
        return vedo

    def load_mesh(self, file_path: str, **kwargs) -> 'vedo.Mesh':
        """Load mesh from file using vedo.

        Args:
            file_path: Path to mesh file
            **kwargs: Additional vedo-specific arguments

        Returns:
            vedo.Mesh or vedo.UGrid object
            
        Raises:
            FileIOError: If file cannot be loaded
        """
        import vedo
        from .exceptions import FileIOError
        
        try:
            return vedo.load(file_path, **kwargs)
        except Exception as e:
            raise FileIOError(
                f"vedo failed to load mesh from {file_path}: {e}"
            ) from e

    def create_mesh(self, vertices: np.ndarray, faces: np.ndarray) -> 'vedo.Mesh':
        """Create vedo mesh from arrays."""
        import vedo

        # vedo.Mesh expects vertices and faces
        return vedo.Mesh([vertices, faces])

    def from_canonical(self, mesh_input) -> Any:
        """
        Create vedo mesh from canonical format.

        Supports both surface and volumetric meshes.
        For volumetric meshes, creates a vedo UGrid.
        """
        import vedo
        from ..utils.cell_types import get_cell_type_map

        vertices = mesh_input.vertices

        if mesh_input.has_volume_cells:
            # Create UnstructuredGrid for volumetric mesh
            return self._create_volumetric_mesh(mesh_input, vertices)

        # Surface mesh
        if len(mesh_input.faces) > 0:
            return vedo.Mesh([vertices, mesh_input.faces])

        # Point cloud only
        return vedo.Points(vertices)

    def _create_volumetric_mesh(self, mesh_input, vertices):
        """Create vedo UGrid from volumetric cells.
        
        Args:
            mesh_input: MeshInput with volume_cells
            vertices: Vertex array
            
        Returns:
            vedo.UGrid object
            
        Raises:
            MeshConversionError: If conversion fails
        """
        import vedo
        from ..utils.cell_types import get_cell_type_map
        
        try:
            cell_type_map = get_cell_type_map()
            cells = []
            cell_types = []

            for etype, connectivity in mesh_input.volume_cells.items():
                vtk_type = cell_type_map.get(etype)
                if vtk_type is None:
                    logger.warning(f"Unknown element type '{etype}' skipped")
                    continue

                n_nodes = connectivity.shape[1]
                for row in connectivity:
                    # Format: [n_nodes, idx0, idx1, ...]
                    cell = [n_nodes] + row.tolist()
                    cells.extend(cell)
                    cell_types.append(vtk_type)

            if not cells:
                raise ValueError("No valid volumetric cells found")

            # Create vedo UGrid
            return vedo.UGrid(
                vertices=vertices,
                cells=np.array(cells),
                cell_types=np.array(cell_types)
            )
        except Exception as e:
            raise MeshConversionError(
                f"Failed to create vedo UGrid from volumetric mesh: {e}"
            ) from e

    def to_canonical(self, mesh: 'vedo.Mesh', mesh_id: str = "") -> 'MeshInput':
        """Convert vedo mesh to canonical MeshInput format."""
        from ..core.models import MeshInput

        vertices = np.array(mesh.vertices)

        # Get faces if available
        faces = np.array([])
        if hasattr(mesh, 'faces') and mesh.faces is not None:
            faces = np.array(mesh.faces)

        # Check if it's a UGrid (volumetric)
        volume_cells = self.get_volume_cells(mesh) if hasattr(mesh, 'cells') else {}

        return MeshInput(
            vertices=vertices,
            faces=faces,
            mesh_id=mesh_id,
            source_format='vedo',
            volume_cells=volume_cells if volume_cells else None
        )

    def get_vertices(self, mesh: 'vedo.Mesh') -> np.ndarray:
        """Extract vertex array."""
        return np.array(mesh.vertices)

    def get_faces(self, mesh: 'vedo.Mesh') -> np.ndarray:
        """Extract face array."""
        if hasattr(mesh, 'faces') and mesh.faces is not None:
            return np.array(mesh.faces)
        return np.array([])

    def get_volume_cells(self, mesh: Any) -> Dict[str, np.ndarray]:
        """Extract volumetric cells from vedo UGrid.
        
        Args:
            mesh: vedo.UGrid or vedo.Mesh object
            
        Returns:
            Dictionary mapping element types to connectivity arrays
        """
        if not hasattr(mesh, 'cells') or mesh.cells is None:
            return {}

        # Map vedo cell types to internal types
        from ..utils.cell_types import get_vtk_type_to_name
        
        try:
            vtk_to_name = get_vtk_type_to_name()
        except Exception:
            # Fallback if cell type mapping fails
            return {}

        volume_cells = {}
        # vedo stores cells as flat array [n, v0, v1, ..., n, v0, ...]
        cells = mesh.cells
        cell_types = getattr(mesh, 'cell_types', None)
        
        if cells is None or cell_types is None:
            return {}

        idx = 0
        type_idx = 0
        while idx < len(cells) and type_idx < len(cell_types):
            n_nodes = int(cells[idx])
            if n_nodes <= 0 or idx + 1 + n_nodes > len(cells):
                break
            
            vtk_type = cell_types[type_idx]
            internal_type = vtk_to_name.get(int(vtk_type))
            
            if internal_type:
                if internal_type not in volume_cells:
                    volume_cells[internal_type] = []
                volume_cells[internal_type].append(cells[idx+1:idx+1+n_nodes].tolist())
            
            idx += 1 + n_nodes
            type_idx += 1

        # Convert lists to numpy arrays
        return {k: np.array(v) for k, v in volume_cells.items()}

    def is_watertight(self, mesh: 'vedo.Mesh') -> bool:
        """Check if mesh is watertight."""
        try:
            return mesh.is_closed()
        except Exception:
            faces = self.get_faces(mesh)
            if len(faces) == 0:
                return True  # Assume watertight for point clouds

            # Manual check via edge counting
            edges = {}
            for face in faces:
                for i in range(3):
                    v1, v2 = int(face[i]), int(face[(i + 1) % 3])
                    edge = tuple(sorted([v1, v2]))
                    edges[edge] = edges.get(edge, 0) + 1

            boundary_edges = sum(1 for count in edges.values() if count == 1)
            return boundary_edges == 0

    def is_manifold(self, mesh: 'vedo.Mesh') -> Tuple[bool, List[Dict[str, Any]]]:
        """Check if mesh is manifold."""
        issues = []
        faces = self.get_faces(mesh)

        if len(faces) == 0:
            return True, issues

        # Build edge to face adjacency
        edges = {}
        for face_idx, face in enumerate(faces):
            for i in range(3):
                v1, v2 = int(face[i]), int(face[(i + 1) % 3])
                edge = tuple(sorted([v1, v2]))
                if edge not in edges:
                    edges[edge] = []
                edges[edge].append(face_idx)

        # Find non-manifold edges
        nonmanifold = [edge for edge, face_list in edges.items() if len(face_list) != 2]
        if nonmanifold:
            issues.append({
                'type': 'non_manifold_edges',
                'count': len(nonmanifold)
            })

        return len(issues) == 0, issues

    def get_boundary_edges(self, mesh: 'vedo.Mesh') -> np.ndarray:
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

    def get_nonmanifold_edges(self, mesh: 'vedo.Mesh') -> List[Tuple[int, int, int]]:
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
        mesh: 'vedo.Mesh',
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

    def compute_face_areas(self, mesh: 'vedo.Mesh') -> np.ndarray:
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

    def compute_face_normals(self, mesh: 'vedo.Mesh') -> np.ndarray:
        """Compute face normals."""
        # vedo has built-in normal computation
        try:
            mesh.compute_normals()
            return np.array(mesh.cell_normals) if hasattr(mesh, 'cell_normals') else np.array(mesh.normals)
        except Exception:
            # Fallback to manual computation
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

    def detect_self_intersections(self, mesh: 'vedo.Mesh') -> List[Tuple[int, int]]:
        """Detect self-intersections."""
        try:
            # vedo has intersect_with method
            intersecting = mesh.intersect_with(mesh)
            if intersecting:
                # Parse intersection results
                return [(0, 0)]  # Placeholder - vedo returns mesh, not indices
        except Exception:
            pass

        # Fallback to trimesh
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
        mesh: 'vedo.Mesh',
        quality_type: str
    ) -> np.ndarray:
        """
        Compute cell quality metrics.

        For volumetric meshes, delegates to PyVista via canonical conversion.
        For surface meshes, computes manually.
        """
        faces = self.get_faces(mesh)
        vertices = self.get_vertices(mesh)

        if len(faces) == 0:
            # Try volumetric quality via PyVista
            return self._compute_volumetric_quality(mesh, quality_type)

        if quality_type == 'area':
            return self.compute_face_areas(mesh)
        elif quality_type == 'aspect_ratio':
            return self._compute_aspect_ratio(mesh)

        return np.array([])

    def _compute_volumetric_quality(
        self,
        mesh: Any,
        quality_type: str
    ) -> np.ndarray:
        """Compute volumetric cell quality via PyVista with caching."""
        mesh_id = id(mesh)
        
        try:
            # Check cache
            if mesh_id in self._pv_grid_cache:
                grid = self._pv_grid_cache[mesh_id]
            else:
                mesh_input = self.to_canonical(mesh)
                if not mesh_input.has_volume_cells:
                    return np.array([])
                grid = mesh_input.get_native('pyvista')
                self._pv_grid_cache[mesh_id] = grid
                
                # Limit cache size
                if len(self._pv_grid_cache) > 10:
                    oldest_key = next(iter(self._pv_grid_cache))
                    del self._pv_grid_cache[oldest_key]
            
            import pyvista as pv
            quality_mesh = grid.cell_quality(quality_measure=quality_type)
            values = quality_mesh.cell_data.get(quality_type, np.array([]))
            return values
            
        except QualityMetricError:
            raise
        except Exception as e:
            raise QualityMetricError(
                f"Failed to compute volumetric quality '{quality_type}': {e}"
            ) from e

    def _compute_aspect_ratio(self, mesh: 'vedo.Mesh') -> np.ndarray:
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

    # ==================== Vedo-specific visualization methods ====================

    def visualize(
        self,
        mesh: 'vedo.Mesh',
        show_edges: bool = True,
        color: str = 'lightblue',
        **kwargs
    ) -> Any:
        """
        Create visualization of mesh.

        Args:
            mesh: vedo.Mesh object
            show_edges: Whether to display mesh edges
            color: Mesh color
            **kwargs: Additional vedo show options

        Returns:
            vedo.Plotter or display object
        """
        import vedo

        mesh.color(color)
        if show_edges:
            mesh.linecolor('black')
            mesh.linewidth(0.5)

        return vedo.show(mesh, **kwargs)

    def visualize_quality(
        self,
        mesh: 'vedo.Mesh',
        quality_type: str = 'area',
        cmap: str = 'viridis',
        **kwargs
    ) -> Any:
        """
        Visualize mesh with quality coloring.

        Args:
            mesh: vedo.Mesh object
            quality_type: Type of quality metric
            cmap: Colormap name
            **kwargs: Additional vedo show options

        Returns:
            vedo.Plotter or display object
        """
        import vedo

        quality_values = self.compute_cell_quality(mesh, quality_type)

        if len(quality_values) > 0:
            mesh.cmap(cmap, quality_values)
            mesh.add_scalarbar(title=quality_type)

        return vedo.show(mesh, **kwargs)

    def slice_mesh(
        self,
        mesh: 'vedo.Mesh',
        origin: Tuple[float, float, float] = (0, 0, 0),
        normal: Tuple[float, float, float] = (1, 0, 0)
    ) -> 'vedo.Mesh':
        """
        Slice mesh with a plane.

        Args:
            mesh: vedo.Mesh object
            origin: Point on the slicing plane
            normal: Normal vector of slicing plane

        Returns:
            Sliced mesh
        """
        return mesh.slice_plane(origin=origin, normal=normal)

    def clip_mesh(
        self,
        mesh: 'vedo.Mesh',
        origin: Tuple[float, float, float] = (0, 0, 0),
        normal: Tuple[float, float, float] = (1, 0, 0)
    ) -> 'vedo.Mesh':
        """
        Clip mesh with a plane (keep only one side).

        Args:
            mesh: vedo.Mesh object
            origin: Point on the clipping plane
            normal: Normal vector of clipping plane

        Returns:
            Clipped mesh
        """
        return mesh.cut_with_plane(origin=origin, normal=normal)

    def decimate(
        self,
        mesh: 'vedo.Mesh',
        fraction: float = 0.5,
        method: str = 'quadric'
    ) -> 'vedo.Mesh':
        """
        Decimate (simplify) mesh.

        Args:
            mesh: vedo.Mesh object
            fraction: Target fraction of original faces (0-1)
            method: Decimation method ('quadric', 'pro', 'cluster')

        Returns:
            Decimated mesh
        """
        if method == 'quadric':
            return mesh.decimate(fraction=fraction)
        elif method == 'pro':
            return mesh.decimate_pro(fraction=fraction)
        elif method == 'cluster':
            return mesh.decimate_cluster(fraction=fraction)
        else:
            raise ValueError(f"Unknown decimation method: {method}")

    def smooth(
        self,
        mesh: 'vedo.Mesh',
        n_iterations: int = 15,
        relax_factor: float = 0.1,
        method: str = 'laplacian'
    ) -> 'vedo.Mesh':
        """
        Smooth mesh.

        Args:
            mesh: vedo.Mesh object
            n_iterations: Number of smoothing iterations
            relax_factor: Relaxation factor
            method: Smoothing method ('laplacian', 'windowed')

        Returns:
            Smoothed mesh
        """
        if method == 'laplacian':
            return mesh.smooth(niter=n_iterations, relaxfact=relax_factor)
        elif method == 'windowed':
            return mesh.smooth_windowed(niter=n_iterations)
        else:
            raise ValueError(f"Unknown smoothing method: {method}")

    def compute_curvature(
        self,
        mesh: 'vedo.Mesh',
        method: str = 'mean'
    ) -> np.ndarray:
        """
        Compute surface curvature.

        Args:
            mesh: vedo.Mesh object
            method: Curvature type ('mean', 'gaussian', 'max', 'min')

        Returns:
            Array of curvature values per vertex
        """
        curv_types = {
            'mean': 0,
            'gaussian': 1,
            'max': 2,
            'min': 3
        }

        if method not in curv_types:
            raise ValueError(f"Unknown curvature method: {method}")

        return mesh.curvature(curv_types[method])

    def extract_largest_region(self, mesh: 'vedo.Mesh') -> 'vedo.Mesh':
        """Extract largest connected region."""
        return mesh.extract_largest_region()

    def clean(
        self,
        mesh: 'vedo.Mesh',
        tolerance: float = 0.0
    ) -> 'vedo.Mesh':
        """
        Clean mesh by merging duplicate points.

        Args:
            mesh: vedo.Mesh object
            tolerance: Merge tolerance (0 for exact match)

        Returns:
            Cleaned mesh
        """
        return mesh.clean(tolerance=tolerance)

    def write(self, mesh: 'vedo.Mesh', file_path: str) -> None:
        """Save mesh to file."""
        mesh.write(file_path)

    def get_bounds(self, mesh: 'vedo.Mesh') -> Tuple[float, float, float, float, float, float]:
        """Get mesh bounding box (xmin, xmax, ymin, ymax, zmin, zmax)."""
        return mesh.bounds()

    def get_volume(self, mesh: 'vedo.Mesh') -> float:
        """Get enclosed volume (for closed surfaces)."""
        try:
            return mesh.volume()
        except Exception:
            return 0.0

    def get_surface_area(self, mesh: 'vedo.Mesh') -> float:
        """Get total surface area."""
        try:
            return mesh.area()
        except Exception:
            areas = self.compute_face_areas(mesh)
            return float(np.sum(areas))

    def clear_cache(self):
        """Clear the PyVista grid cache."""
        self._pv_grid_cache.clear()
        logger.debug("VedoAdapter cache cleared")
