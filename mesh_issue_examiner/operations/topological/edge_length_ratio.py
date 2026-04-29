"""
Edge length ratio detection operation.

Detects edges with abnormal length ratios in their neighborhood.
High length ratios indicate non-uniform mesh refinement or stretched elements.
"""

from typing import List, Optional, Set
import numpy as np
import logging

from ...core.models import (
    MeshInput,
    MeshIssue,
    IssueLocation,
    IssueCategory,
    IssueSeverity,
    OperationConfig,
)
from ...utils.topology import extract_unique_edges, compute_edge_lengths
from ..base import BaseOperation, OperationMetadata
from ..operation_registry import registry

logger = logging.getLogger(__name__)


class EdgeLengthRatioDetection(BaseOperation):
    """
    Detects edges with abnormal length ratios in neighborhood.
    
    High length ratios indicate:
    - Non-uniform mesh refinement
    - Stretched/elongated elements
    - Poor mesh transitions
    
    Length ratio is computed as:
        ratio = edge_length / min_neighbor_length
    
    High ratios (> 10) indicate potential quality issues.
    
    Parameters
    ----------
    max_ratio : float
        Maximum acceptable length ratio (default: 10.0)
    """
    
    # Maximum number of IssueLocation objects per issue
    _MAX_LOCATIONS = 100

    def __init__(
        self,
        config: Optional[OperationConfig] = None,
        max_ratio: float = 10.0
    ):
        super().__init__(config)
        self.max_ratio = max_ratio

    @property
    def metadata(self) -> OperationMetadata:
        return OperationMetadata(
            operation_id='topology.detect_edge_length_ratio',
            display_name='Edge Length Ratio Detection',
            description='Identifies edges with abnormal length ratios in neighborhood. '
                       'High ratios indicate non-uniform refinement or stretched elements. '
                       f'Default threshold: {self.max_ratio}x.',
            category=IssueCategory.GEOMETRIC,
            supported_libraries=['pyvista', 'trimesh'],
            recommended_library='pyvista',
            requires_adjacency=False,
            complexity='O(E)',
            tags=['topology', 'edge', 'length', 'ratio', 'uniformity', 'refinement']
        )

    def execute(self, mesh: MeshInput) -> List[MeshIssue]:
        """Execute edge length ratio detection.
        
        Returns:
            List of MeshIssue objects for edges with high length ratios.
            Empty list if no high ratios found or mesh has no faces.
        """
        if len(mesh.faces) == 0:
            return []
        
        # Extract unique edges and compute lengths
        edges = extract_unique_edges(mesh.faces)
        
        if len(edges) == 0:
            return []
        
        lengths = compute_edge_lengths(mesh.vertices, edges)
        
        # Build vertex-to-edges map
        vertex_edges = self._build_vertex_edges_map(len(mesh.vertices), edges)
        
        # Find edges with high length ratio in neighborhood
        high_ratio_edges = []
        
        for e_idx, (v1, v2) in enumerate(edges):
            # Get neighborhood edges (edges sharing either vertex)
            neighbor_edges: Set[int] = set(vertex_edges[v1]) | set(vertex_edges[v2])
            neighbor_edges.discard(e_idx)  # Remove self
            
            if not neighbor_edges:
                continue
            
            # Compute min neighbor length
            neighbor_lengths = lengths[list(neighbor_edges)]
            min_neighbor = np.min(neighbor_lengths)
            
            if min_neighbor > 0:
                ratio = lengths[e_idx] / min_neighbor
                if ratio > self.max_ratio:
                    high_ratio_edges.append({
                        'edge': (v1, v2),
                        'edge_idx': e_idx,
                        'ratio': float(ratio),
                        'length': float(lengths[e_idx]),
                        'min_neighbor': float(min_neighbor),
                    })
        
        if not high_ratio_edges:
            return []
        
        # Create locations for high ratio edges
        locations = []
        for edge_info in high_ratio_edges[:self._MAX_LOCATIONS]:
            v1, v2 = edge_info['edge']
            midpoint = (mesh.vertices[v1] + mesh.vertices[v2]) / 2.0
            
            locations.append(IssueLocation(
                element_type='edge',
                element_index=-1,
                coordinates=tuple(float(x) for x in midpoint),
                adjacent_elements={
                    'vertices': [int(v1), int(v2)],
                    'length_ratio': edge_info['ratio'],
                    'edge_length': edge_info['length'],
                    'min_neighbor_length': edge_info['min_neighbor'],
                }
            ))
        
        # Determine severity
        high_ratio_rate = len(high_ratio_edges) / len(edges)
        severity = self._determine_severity(high_ratio_rate)
        
        # Compute statistics
        ratios = [e['ratio'] for e in high_ratio_edges]
        
        issue = MeshIssue(
            operation_id=self.metadata.operation_id,
            category=IssueCategory.GEOMETRIC,
            severity=severity,
            message=f"Found {len(high_ratio_edges)} edges with high length ratios "
                   f"(>{self.max_ratio}x threshold)",
            locations=locations,
            metrics={
                'high_ratio_count': int(len(high_ratio_edges)),
                'total_edges': int(len(edges)),
                'high_ratio_rate': float(high_ratio_rate),
                'min_ratio': float(np.min(ratios)),
                'max_ratio': float(np.max(ratios)),
                'mean_ratio': float(np.mean(ratios)),
                'threshold': self.max_ratio,
            },
            metadata={
                'high_ratio_edges': [
                    {
                        'vertices': list(e['edge']),
                        'ratio': e['ratio'],
                        'length': e['length'],
                    }
                    for e in high_ratio_edges[:50]
                ],
            }
        )
        
        return [issue]
    
    def _build_vertex_edges_map(
        self, 
        n_vertices: int, 
        edges: np.ndarray
    ) -> List[List[int]]:
        """Build mapping from vertices to incident edge indices.
        
        Args:
            n_vertices: Total number of vertices
            edges: (n_edges, 2) array of edge vertex indices
            
        Returns:
            List where vertex_edges[v] = list of edge indices incident to vertex v
        """
        vertex_edges: List[List[int]] = [[] for _ in range(n_vertices)]
        
        for e_idx, (v1, v2) in enumerate(edges):
            vertex_edges[int(v1)].append(e_idx)
            vertex_edges[int(v2)].append(e_idx)
        
        return vertex_edges
    
    def _determine_severity(self, high_ratio_rate: float) -> IssueSeverity:
        """Determine severity based on percentage of high ratio edges."""
        if high_ratio_rate > 0.1:
            return IssueSeverity.ERROR
        elif high_ratio_rate > 0.05:
            return IssueSeverity.WARNING
        else:
            return IssueSeverity.INFO


# Register operation
registry.register(EdgeLengthRatioDetection())
