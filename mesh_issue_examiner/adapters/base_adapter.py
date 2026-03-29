"""
Abstract base adapter for library-specific implementations.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from core.models import MeshInput


class MeshAdapter(ABC):
    """
    Abstract base for library-specific mesh adapters.

    Adapters provide a unified interface to library-specific
    mesh representations and operations.

    Each adapter must implement methods for:
    - Mesh loading and creation
    - Topology inspection (manifold, boundary, etc.)
    - Geometry computation (areas, normals, etc.)
    - Quality metrics
    """

    @property
    @abstractmethod
    def library_name(self) -> str:
        """Return the name of the wrapped library."""
        pass

    @property
    def is_available(self) -> bool:
        """Check if the library is installed."""
        try:
            self._check_import()
            return True
        except ImportError:
            return False

    @abstractmethod
    def _check_import(self):
        """Check that the library is importable. Raise ImportError if not."""
        pass

    @abstractmethod
    def load_mesh(self, file_path: str, **kwargs) -> Any:
        """
        Load mesh from file.

        Args:
            file_path: Path to mesh file
            **kwargs: Additional library-specific arguments

        Returns:
            Library-specific mesh object
        """
        pass

    @abstractmethod
    def create_mesh(self, vertices: np.ndarray, faces: np.ndarray) -> Any:
        """
        Create mesh from arrays.

        Args:
            vertices: Nx3 array of vertex coordinates
            faces: Mx3 array of face indices

        Returns:
            Library-specific mesh object
        """
        pass

    @abstractmethod
    def get_vertices(self, mesh: Any) -> np.ndarray:
        """Extract vertex array from mesh."""
        pass

    @abstractmethod
    def get_faces(self, mesh: Any) -> np.ndarray:
        """Extract face array from mesh."""
        pass

    @abstractmethod
    def is_watertight(self, mesh: Any) -> bool:
        """
        Check if mesh is watertight (closed manifold).

        Args:
            mesh: Library-specific mesh object

        Returns:
            True if mesh has no boundary edges
        """
        pass

    @abstractmethod
    def is_manifold(self, mesh: Any) -> Tuple[bool, List[Dict[str, Any]]]:
        """
        Check if mesh is manifold.

        Args:
            mesh: Library-specific mesh object

        Returns:
            Tuple of (is_manifold, list of issues)
        """
        pass

    @abstractmethod
    def get_boundary_edges(self, mesh: Any) -> np.ndarray:
        """
        Get edges that are on the mesh boundary.

        Args:
            mesh: Library-specific mesh object

        Returns:
            Array of boundary edge indices
        """
        pass

    @abstractmethod
    def get_nonmanifold_edges(self, mesh: Any) -> List[Tuple[int, int, int]]:
        """
        Get non-manifold edges.

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
        """
        Find duplicate vertices.

        Args:
            mesh: Library-specific mesh object
            tolerance: Distance threshold for considering vertices duplicate

        Returns:
            List of duplicate vertex indices
        """
        pass

    @abstractmethod
    def compute_face_areas(self, mesh: Any) -> np.ndarray:
        """
        Compute area of each face.

        Args:
            mesh: Library-specific mesh object

        Returns:
            Array of face areas
        """
        pass

    @abstractmethod
    def compute_face_normals(self, mesh: Any) -> np.ndarray:
        """
        Compute normal vector for each face.

        Args:
            mesh: Library-specific mesh object

        Returns:
            Nx3 array of face normals
        """
        pass

    @abstractmethod
    def detect_self_intersections(self, mesh: Any) -> List[Tuple[int, int]]:
        """
        Detect self-intersecting face pairs.

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
        """
        Compute cell quality metric.

        Args:
            mesh: Library-specific mesh object
            quality_type: One of 'aspect_ratio', 'skewness', 'scaled_jacobian',
                         'max_angle', 'min_angle', 'condition', 'Jacobian', 'volume'

        Returns:
            Array of quality values for each cell
        """
        pass

    def to_canonical(self, mesh: Any, mesh_id: str = "") -> 'MeshInput':
        """
        Convert library mesh to canonical format.

        Args:
            mesh: Library-specific mesh object
            mesh_id: Optional mesh identifier

        Returns:
            MeshInput in canonical format
        """
        from core.models import MeshInput

        return MeshInput(
            vertices=self.get_vertices(mesh),
            faces=self.get_faces(mesh),
            mesh_id=mesh_id,
            source_format=self.library_name
        )

    def from_canonical(self, mesh_input: 'MeshInput') -> Any:
        """
        Create library mesh from canonical format.

        Args:
            mesh_input: MeshInput in canonical format

        Returns:
            Library-specific mesh object
        """
        return self.create_mesh(mesh_input.vertices, mesh_input.faces)

    def compute_all_quality_metrics(
        self,
        mesh: Any,
        metrics: Optional[List[str]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Compute multiple quality metrics.

        Args:
            mesh: Library-specific mesh object
            metrics: List of metric types to compute

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
                results[metric] = np.array([])

        return results
