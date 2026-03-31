"""
Abstract base adapter for library-specific implementations.

This module defines the base interface that all mesh adapters must implement.
It provides a unified interface to library-specific mesh representations
and operations for both 2D surface meshes and 3D volumetric meshes.

New adapters should inherit from MeshAdapter and implement all abstract methods.
Adapters must also declare their capabilities via the `capabilities` property.

Example:
    >>> from mesh_issue_examiner.adapters.base_adapter import MeshAdapter
    >>> from mesh_issue_examiner.adapters.capabilities import AdapterCapability
    >>> 
    >>> class MyAdapter(MeshAdapter):
    ...     @property
    ...     def capabilities(self) -> AdapterCapability:
    ...         return AdapterCapability.SURFACE_MESH | AdapterCapability.FILE_IO
    ...     
    ...     @property
    ...     def library_name(self) -> str:
    ...         return 'my_library'
    ...     
    ...     # Implement all abstract methods...
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..core.models import MeshInput


class MeshAdapter(ABC):
    """Abstract base class for library-specific mesh adapters.
    
    Adapters provide a unified interface to library-specific mesh
    representations and operations. Each adapter bridges between the
    canonical MeshInput format and a specific mesh library's format.
    
    Adapters support two types of meshes:
    - **Surface meshes (2D)**: Triangular or polygonal face meshes
    - **Volumetric meshes (3D)**: Tetrahedral, hexahedral, prism, or pyramid cells
    
    Subclasses must implement:
    1. `capabilities` property - declare supported features
    2. `library_name` property - return library identifier
    3. All abstract methods for supported operations
    
    Attributes:
        capabilities: AdapterCapability flags indicating supported features
    """

    @property
    @abstractmethod
    def capabilities(self) -> 'AdapterCapability':
        """Return the capabilities supported by this adapter.
        
        Subclasses must override this to declare which operations they support.
        Use bitwise OR to combine multiple capabilities.
        
        Returns:
            AdapterCapability flags
            
        Example:
            >>> @property
            ... def capabilities(self) -> AdapterCapability:
            ...     return (
            ...         AdapterCapability.SURFACE_MESH |
            ...         AdapterCapability.FILE_IO |
            ...         AdapterCapability.QUALITY_METRICS
            ...     )
        """
        pass

    @property
    @abstractmethod
    def library_name(self) -> str:
        """Return the name of the wrapped library.
        
        Returns:
            Library identifier string (e.g., 'pyvista', 'trimesh')
        """
        pass

    @property
    def is_available(self) -> bool:
        """Check if the underlying library is installed.
        
        This property attempts to import the library to verify it's
        available in the current environment.
        
        Returns:
            True if the library is installed and importable
            
        Note:
            Subclasses typically don't need to override this - the
            default implementation uses `_check_import()`.
        """
        try:
            self._check_import()
            return True
        except ImportError:
            return False

    @abstractmethod
    def _check_import(self):
        """Check that the library is importable. Raise ImportError if not.
        
        This method should simply import the library to verify it's
        available. It doesn't need to return anything.
        
        Raises:
            ImportError: If the library is not installed
            
        Example:
            >>> def _check_import(self):
            ...     import pyvista  # Will raise ImportError if not installed
            ...     return pyvista
        """
        pass

    # ======================================================================
    # Surface Mesh Operations (required if SURFACE_MESH capability)
    # ======================================================================

    @abstractmethod
    def load_mesh(self, file_path: str, **kwargs) -> Any:
        """Load mesh from file.
        
        Args:
            file_path: Path to mesh file
            **kwargs: Additional library-specific arguments
            
        Returns:
            Library-specific mesh object
            
        Raises:
            FileIOError: If file cannot be loaded
        """
        pass

    @abstractmethod
    def create_mesh(self, vertices: np.ndarray, faces: np.ndarray) -> Any:
        """Create mesh from vertex and face arrays.
        
        Args:
            vertices: Nx3 array of vertex coordinates
            faces: Mx3 array of face indices (triangles)
            
        Returns:
            Library-specific mesh object
        """
        pass

    @abstractmethod
    def get_vertices(self, mesh: Any) -> np.ndarray:
        """Extract vertex array from mesh.
        
        Args:
            mesh: Library-specific mesh object
            
        Returns:
            Nx3 array of vertex coordinates
        """
        pass

    @abstractmethod
    def get_faces(self, mesh: Any) -> np.ndarray:
        """Extract face array from mesh.
        
        Args:
            mesh: Library-specific mesh object
            
        Returns:
            Mx3 array of face vertex indices
        """
        pass

    # ======================================================================
    # Topology Operations (required if TOPOLOGY_CHECKS capability)
    # ======================================================================

    @abstractmethod
    def is_watertight(self, mesh: Any) -> bool:
        """Check if mesh is watertight (closed manifold).
        
        A mesh is watertight if it has no boundary edges - every edge
        is shared by exactly two faces.
        
        Args:
            mesh: Library-specific mesh object
            
        Returns:
            True if mesh has no boundary edges
        """
        pass

    @abstractmethod
    def is_manifold(self, mesh: Any) -> Tuple[bool, List[Dict[str, Any]]]:
        """Check if mesh is manifold.
        
        A manifold mesh has consistent topology where:
        - Each edge is shared by at most 2 faces
        - Faces meet only at edges (not in the middle)
        
        Args:
            mesh: Library-specific mesh object
            
        Returns:
            Tuple of (is_manifold, list of issues)
            Each issue is a dict with 'type', 'message', and optional details
        """
        pass

    @abstractmethod
    def get_boundary_edges(self, mesh: Any) -> np.ndarray:
        """Get edges that are on the mesh boundary.
        
        Boundary edges are edges that belong to only one face.
        
        Args:
            mesh: Library-specific mesh object
            
        Returns:
            Array of boundary edge vertex pairs
        """
        pass

    @abstractmethod
    def get_nonmanifold_edges(self, mesh: Any) -> List[Tuple[int, int, int]]:
        """Get non-manifold edges.
        
        Non-manifold edges are edges shared by more than 2 faces.
        
        Args:
            mesh: Library-specific mesh object
            
        Returns:
            List of (vertex1, vertex2, incident_face_count) tuples
        """
        pass

    @abstractmethod
    def get_duplicate_vertices(
        self,
        mesh: Any,
        tolerance: float
    ) -> List[int]:
        """Find duplicate vertices.
        
        Args:
            mesh: Library-specific mesh object
            tolerance: Distance threshold for considering vertices duplicate
            
        Returns:
            List of duplicate vertex indices
        """
        pass

    # ======================================================================
    # Geometric Operations (required if QUALITY_METRICS capability)
    # ======================================================================

    @abstractmethod
    def compute_face_areas(self, mesh: Any) -> np.ndarray:
        """Compute area of each face.
        
        Args:
            mesh: Library-specific mesh object
            
        Returns:
            Array of face areas
        """
        pass

    @abstractmethod
    def compute_face_normals(self, mesh: Any) -> np.ndarray:
        """Compute normal vector for each face.
        
        Args:
            mesh: Library-specific mesh object
            
        Returns:
            Mx3 array of face normal vectors (unit length)
        """
        pass

    @abstractmethod
    def detect_self_intersections(self, mesh: Any) -> List[Tuple[int, int]]:
        """Detect self-intersecting face pairs.
        
        Args:
            mesh: Library-specific mesh object
            
        Returns:
            List of (face1_index, face2_index) tuples
        """
        pass

    @abstractmethod
    def compute_cell_quality(
        self,
        mesh: Any,
        quality_type: str
    ) -> np.ndarray:
        """Compute cell quality metric.
        
        Supported quality types vary by adapter:
        - 'aspect_ratio': Longest edge / shortest edge
        - 'skewness': Angular deviation from ideal
        - 'scaled_jacobian': Scaled Jacobian determinant
        - 'max_angle': Maximum face angle
        - 'min_angle': Minimum face angle
        - 'condition': Condition number
        - 'Jacobian': Jacobian determinant
        - 'volume': Cell volume (for volumetric cells)
        - 'area': Face area (for surface faces)
        
        Args:
            mesh: Library-specific mesh object
            quality_type: Name of quality metric to compute
            
        Returns:
            Array of quality values for each cell
            
        Raises:
            QualityMetricError: If quality type is not supported
        """
        pass

    # ======================================================================
    # Conversion Methods
    # ======================================================================

    def to_canonical(self, mesh: Any, mesh_id: str = "") -> 'MeshInput':
        """Convert library mesh to canonical format.
        
        This default implementation creates a MeshInput from vertices
        and faces. Adapters that support volumetric meshes should
        override this to also extract volume_cells.
        
        Args:
            mesh: Library-specific mesh object
            mesh_id: Optional mesh identifier
            
        Returns:
            MeshInput in canonical format
        """
        from ..core.models import MeshInput

        return MeshInput(
            vertices=self.get_vertices(mesh),
            faces=self.get_faces(mesh),
            mesh_id=mesh_id,
            source_format=self.library_name
        )

    def from_canonical(self, mesh_input: 'MeshInput') -> Any:
        """Create library mesh from canonical format.
        
        This default implementation handles surface meshes. Adapters
        that support volumetric meshes should override this to also
        handle volume_cells.
        
        Args:
            mesh_input: MeshInput in canonical format
            
        Returns:
            Library-specific mesh object
            
        Note:
            If the adapter doesn't support volumetric meshes but
            mesh_input.has_volume_cells is True, this method should
            either:
            1. Extract and return the surface boundary only
            2. Raise UnsupportedOperationError with clear message
        """
        # Default: create surface mesh
        # Subclasses override for volumetric support
        return self.create_mesh(mesh_input.vertices, mesh_input.faces)

    # ======================================================================
    # Convenience Methods
    # ======================================================================

    def compute_all_quality_metrics(
        self,
        mesh: Any,
        metrics: Optional[List[str]] = None
    ) -> Dict[str, np.ndarray]:
        """Compute multiple quality metrics at once.
        
        This is a convenience method that calls compute_cell_quality
        for multiple metrics and returns the results in a dictionary.
        Errors for individual metrics are silently ignored.
        
        Args:
            mesh: Library-specific mesh object
            metrics: List of metric names to compute. If None, computes
                a default set: ['aspect_ratio', 'skewness', 'min_angle']
                
        Returns:
            Dictionary mapping metric names to value arrays
        """
        if metrics is None:
            metrics = ['aspect_ratio', 'skewness', 'min_angle']

        results = {}
        for metric in metrics:
            try:
                results[metric] = self.compute_cell_quality(mesh, metric)
            except Exception:
                # Silently skip metrics that fail
                results[metric] = np.array([])

        return results

    def supports(self, capability: 'AdapterCapability') -> bool:
        """Check if adapter supports a specific capability.
        
        This is a convenience method for checking capabilities.
        
        Args:
            capability: Capability flag to check
            
        Returns:
            True if adapter has the capability
            
        Example:
            >>> if adapter.supports(AdapterCapability.VOLUMETRIC_MESH):
            ...     # Do volumetric operation
        """
        return bool(self.capabilities & capability)

    def __repr__(self) -> str:
        """String representation of adapter."""
        available = "available" if self.is_available else "not available"
        return f"<{self.__class__.__name__}: {self.library_name} ({available})>"
