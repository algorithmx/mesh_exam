"""
Surface orientation consistency check operation.

Detects faces with inconsistent winding order (flipped normals) in surface meshes
by verifying that adjacent faces traverse shared edges in opposite directions.

Algorithm References:
--------------------
1. StackOverflow - "how to check if all the faces face outward"
   https://stackoverflow.com/questions/77927993/how-to-check-if-all-the-faces-face-outward
   
   Core principle (quoted from answer):
   "If a face has an edge with vertices (in counterclockwise order) `a,b`, then 
   the other face (and it's manifold, so there should be only one at most) with 
   an edge with those vertices should have them in the order `b,a`. If the other 
   face has them in the order `a,b`, flip its winding."
   
   This operation uses this edge-direction comparison to detect (not fix) 
   orientation inconsistencies.

2. Borodin, P., Zachmann, G., & Klein, R. (2004).
   "Consistent Normal Orientation for Polygonal Meshes"
   Computer Graphics International (CGI), June 16-19, 2004, Crete, Greece.
   IEEE Computer Society Press, pp. 18-25.
   https://user.informatik.uni-bremen.de/zach/papers/orientation_print.pdf
   
   Key concepts from Section 3.1:
   - "Since the orientation of each polygon's normal can be encoded in its 
     vertex order, the problem of orientation of normals is equivalent to 
     the problem of ordering vertices in polygons consistently."
   
   From Section 3.2 (Detection of Patches):
   - Mesh divided into manifold surface patches at boundary/non-manifold edges
   - "For each patch we consistently orient all polygons belonging to it"
   
   This operation implements the patch-based detection approach to identify
   inconsistencies between connected components.

3. CGAL Polygon Mesh Processing - Orientation
   https://doc.cgal.org/latest/Polygon_mesh_processing/index.html
   
   Provides the theoretical foundation for polygon soup orientation checking.

Usage Context:
--------------
This operation implements the "detection" part of orientation checking per
Borodin et al. It identifies inconsistencies but does not repair them. 
For mesh repair algorithms, refer to:
- Borodin et al. (2004) for global consistent orientation
- Takayama et al. (2014) "A simple method for correcting face orientations 
  in polygon meshes based on ray casting"

See Also:
---------
- quality_3d.check_orientation_consistency : For volumetric cells
- CGAL::Polygon_mesh_processing::orient_polygon_soup : For repair
"""

from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict
import numpy as np

from ...core.models import (
    MeshInput,
    MeshIssue,
    IssueLocation,
    IssueCategory,
    IssueSeverity,
    OperationConfig,
)
from ..base import BaseOperation, OperationMetadata
from ..operation_registry import registry
from ...utils.metrics import determine_severity


class SurfaceOrientationCheck(BaseOperation):
    """
    Detects inconsistent face orientation (flipped normals) in surface meshes.
    
    For each internal edge shared by two faces, verifies that the edge is traversed
    in opposite directions in the two faces. If both faces traverse the edge in
    the same direction, one of them has inconsistent orientation.
    
    The algorithm follows the approach described in:
    - StackOverflow answer on face orientation checking (edge direction comparison)
    - Borodin et al. (2004) "Consistent Normal Orientation for Polygonal Meshes"
      for patch-based component handling
    
    Algorithm Steps:
    1. Build edge-to-faces adjacency map with edge direction information
    2. For each edge with exactly 2 incident faces, check winding consistency
       per StackOverflow algorithm (opposite edge direction required)
    3. BFS traversal to identify connected components (patches per Borodin et al.)
    4. Report faces with inconsistent orientation within each component
    
    Edge Cases Handled:
    - Non-manifold edges (>2 faces): Skipped (cannot determine orientation)
    - Boundary edges (1 face): Skipped (no adjacent face to compare)
    - Single face meshes: No shared edges to validate
    - Disconnected components: Each analyzed independently (Borodin patch concept)
    """

    @property
    def metadata(self) -> OperationMetadata:
        return OperationMetadata(
            operation_id='topology.detect_inconsistent_orientation',
            display_name='Surface Orientation Consistency Check',
            description='Detects faces with inconsistent winding order (flipped normals) '
                       'by verifying adjacent faces traverse shared edges in opposite directions. '
                       'Algorithm based on Borodin et al. (CGI 2004) and StackOverflow community algorithm.',
            category=IssueCategory.ORIENTATION,
            supported_libraries=['trimesh', 'pyvista', 'pymeshlab'],
            recommended_library='pyvista',
            requires_adjacency=True,
            complexity='O(n)',
            tags=['orientation', 'normals', 'winding', 'flipped', 'consistency', 'surface']
        )

    def execute(self, mesh: MeshInput) -> List[MeshIssue]:
        """
        Execute surface orientation consistency check.
        
        Implementation follows the edge-direction comparison algorithm from
        StackOverflow (reference 1) and patch-based component analysis from
        Borodin et al. (reference 2).
        
        Args:
            mesh: Input surface mesh with faces
            
        Returns:
            List of MeshIssue objects describing orientation inconsistencies
        """
        faces = mesh.faces
        if len(faces) == 0:
            return []
        
        # Step 1: Build edge-to-faces map with edge direction
        # This maps each edge (as sorted tuple) to list of (face_idx, edge_direction)
        # where edge_direction is the vertex order as it appears in the face
        edge_face_map = self._build_edge_face_map(mesh)
        
        if not edge_face_map:
            return []
        
        # Step 2: Find edges with inconsistent winding
        # Per StackOverflow algorithm: adjacent faces must have opposite edge directions
        inconsistent_faces: Set[int] = set()
        inconsistent_edges: List[Tuple[Tuple[int, int], int, int]] = []
        
        for edge_key, face_entries in edge_face_map.items():
            # Only check manifold internal edges (exactly 2 faces)
            # Per Borodin et al.: non-manifold edges (>2) and boundary edges (1) 
            # are handled separately
            if len(face_entries) != 2:
                continue
            
            (face_a, dir_a), (face_b, dir_b) = face_entries[0], face_entries[1]
            
            # Check if edge directions are opposite
            # Per StackOverflow: if face A has (a,b), face B must have (b,a)
            # Same direction means one face has flipped orientation
            if not self._are_directions_opposite(dir_a, dir_b):
                inconsistent_faces.add(face_a)
                inconsistent_faces.add(face_b)
                inconsistent_edges.append((edge_key, face_a, face_b))
        
        if not inconsistent_faces:
            return []
        
        # Step 3: Analyze connected components
        # Per Borodin et al. Section 3.2: "divide the model into a set of 
        # manifold surface patches"
        adjacency = self._build_face_adjacency(edge_face_map)
        components = self._find_connected_components(len(faces), adjacency)
        
        # Count components with orientation issues
        affected_components = self._count_affected_components(components, inconsistent_faces)
        
        # Step 4: Create issue locations
        locations = []
        for face_idx in sorted(inconsistent_faces)[:100]:  # Limit locations
            face = faces[face_idx]
            # Use first vertex as representative location
            if len(face) > 0 and face[0] < len(mesh.vertices):
                coords = tuple(mesh.vertices[face[0]])
            else:
                coords = None
            
            locations.append(IssueLocation(
                element_type='face',
                element_index=int(face_idx),
                coordinates=coords,
                adjacent_elements={
                    'vertices': [int(v) for v in face],
                    'in_component': self._get_face_component(face_idx, components)
                }
            ))
        
        # Determine severity based on violation rate
        violation_rate = len(inconsistent_faces) / max(len(faces), 1)
        severity = determine_severity(violation_rate)
        
        # Build informative message
        component_msg = f"across {affected_components} component(s)" if affected_components > 1 else ""
        
        issue = MeshIssue(
            operation_id=self.metadata.operation_id,
            category=IssueCategory.ORIENTATION,
            severity=severity,
            message=f'Detected {len(inconsistent_faces)} face(s) with inconsistent orientation '
                   f'({violation_rate * 100:.2f}% of faces) {component_msg}. '
                   f'Adjacent faces share edges with same winding direction.',
            locations=locations,
            metrics={
                'inconsistent_face_count': len(inconsistent_faces),
                'total_faces': len(faces),
                'violation_rate': violation_rate,
                'inconsistent_edge_count': len(inconsistent_edges),
                'affected_component_count': affected_components,
                'total_component_count': len(components)
            },
            metadata={
                'reference': 'Borodin et al. CGI 2004, StackOverflow orientation algorithm',
                'inconsistent_edges': [
                    {'edge': list(e), 'face_a': fa, 'face_b': fb}
                    for e, fa, fb in inconsistent_edges[:20]
                ]
            }
        )
        
        return [issue]

    # Directed variant: unlike the shared undirected map in
    # utils.topology.build_edge_face_map, it records each edge's traversal
    # direction within its face, which winding-consistency checks require.
    def _build_edge_face_map(
        self, 
        mesh: MeshInput
    ) -> Dict[Tuple[int, int], List[Tuple[int, Tuple[int, int]]]]:
        """
        Build mapping from edge to incident faces with edge direction.
        
        Per Borodin et al. Section 3.2: Build neighbourhood information for the mesh.
        
        Args:
            mesh: Input mesh
            
        Returns:
            Dictionary mapping edge (sorted vertex tuple) to list of 
            (face_idx, edge_direction) tuples, where edge_direction is 
            (v1, v2) as they appear in the face.
        """
        faces = mesh.faces
        n_faces = len(faces)
        if n_faces == 0:
            return {}
        
        edge_face_map: Dict[Tuple[int, int], List[Tuple[int, Tuple[int, int]]]] = {}
        
        # Handle triangular meshes (vectorized fast path)
        if faces.ndim == 2 and faces.shape[1] == 3:
            for face_idx in range(n_faces):
                face = faces[face_idx]
                # Three edges per triangle: (0,1), (1,2), (2,0)
                edges = [(face[0], face[1]), (face[1], face[2]), (face[2], face[0])]
                
                for v1, v2 in edges:
                    # Edge key: sorted vertices for canonical lookup
                    edge_key = (min(v1, v2), max(v1, v2))
                    # Edge direction: as they appear in face
                    edge_dir = (int(v1), int(v2))
                    
                    if edge_key not in edge_face_map:
                        edge_face_map[edge_key] = []
                    edge_face_map[edge_key].append((face_idx, edge_dir))
        else:
            # Fallback for polygonal faces
            for face_idx, face in enumerate(faces):
                n = len(face)
                for i in range(n):
                    v1, v2 = int(face[i]), int(face[(i + 1) % n])
                    edge_key = (min(v1, v2), max(v1, v2))
                    edge_dir = (v1, v2)
                    
                    if edge_key not in edge_face_map:
                        edge_face_map[edge_key] = []
                    edge_face_map[edge_key].append((face_idx, edge_dir))
        
        return edge_face_map
    
    def _are_directions_opposite(
        self, 
        dir_a: Tuple[int, int], 
        dir_b: Tuple[int, int]
    ) -> bool:
        """
        Check if two edge directions are opposite.
        
        Per StackOverflow algorithm: if face A traverses edge as (a→b),
        adjacent face B must traverse it as (b→a) for consistent orientation.
        
        Args:
            dir_a: Edge direction in face A as (v1, v2)
            dir_b: Edge direction in face B as (v1, v2)
            
        Returns:
            True if directions are opposite (consistent), False if same (inconsistent)
        """
        return dir_a[0] == dir_b[1] and dir_a[1] == dir_b[0]
    
    def _build_face_adjacency(
        self, 
        edge_face_map: Dict[Tuple[int, int], List[Tuple[int, Tuple[int, int]]]]
    ) -> Dict[int, Set[int]]:
        """
        Build face adjacency graph from edge-face map.
        
        Two faces are adjacent if they share an edge.
        
        Args:
            edge_face_map: Mapping from edge to incident faces
            
        Returns:
            Dictionary mapping face index to set of adjacent face indices
        """
        adjacency: Dict[int, Set[int]] = defaultdict(set)
        
        for edge_key, face_entries in edge_face_map.items():
            # Only consider edges shared by exactly 2 faces for adjacency
            if len(face_entries) == 2:
                face_a = face_entries[0][0]
                face_b = face_entries[1][0]
                adjacency[face_a].add(face_b)
                adjacency[face_b].add(face_a)
        
        return adjacency
    
    def _find_connected_components(
        self, 
        n_faces: int, 
        adjacency: Dict[int, Set[int]]
    ) -> List[Set[int]]:
        """
        Find connected components using BFS.
        
        Per Borodin et al. Section 3.2: "divide the model into a set of 
        manifold surface patches, which either are not connected with each 
        other or connected only at vertices or non-manifold edges."
        
        Args:
            n_faces: Total number of faces
            adjacency: Face adjacency graph
            
        Returns:
            List of sets, each containing face indices of one component
        """
        visited: Set[int] = set()
        components: List[Set[int]] = []
        
        for face_idx in range(n_faces):
            if face_idx in visited:
                continue
            
            # BFS to find all faces in this component
            component: Set[int] = set()
            queue = [face_idx]
            visited.add(face_idx)
            
            while queue:
                current = queue.pop(0)
                component.add(current)
                
                for neighbor in adjacency.get(current, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            
            if component:
                components.append(component)
        
        return components
    
    def _count_affected_components(
        self, 
        components: List[Set[int]], 
        inconsistent_faces: Set[int]
    ) -> int:
        """
        Count how many components contain inconsistent faces.
        
        Args:
            components: List of component face sets
            inconsistent_faces: Set of faces with orientation issues
            
        Returns:
            Number of components containing at least one inconsistent face
        """
        count = 0
        for component in components:
            if any(face in inconsistent_faces for face in component):
                count += 1
        return count
    
    def _get_face_component(
        self, 
        face_idx: int, 
        components: List[Set[int]]
    ) -> int:
        """
        Get the component index for a given face.
        
        Args:
            face_idx: Face index to look up
            components: List of component face sets
            
        Returns:
            Component index, or -1 if not found
        """
        for i, component in enumerate(components):
            if face_idx in component:
                return i
        return -1


# Register operation
registry.register(SurfaceOrientationCheck())
