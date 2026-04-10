"""
Duplicate face detection operation.

Detects faces that share the same set of vertex indices (regardless of
winding order). Duplicate faces waste memory, can cause rendering artifacts,
and indicate data quality issues in the mesh pipeline.

Algorithm — Canonical sorted-key hash:
    For each face, sort the vertex indices to produce a canonical key.
    Track seen keys; any face whose key was already seen is a duplicate.
    Group duplicates by their canonical key for clear reporting.

    Complexity: O(F * k log k) where k = vertices per face.
"""

from collections import defaultdict
from typing import List, Tuple

import numpy as np

from ...core.models import (
    MeshInput,
    MeshIssue,
    IssueLocation,
    IssueCategory,
    IssueSeverity,
)
from ..base import BaseOperation, OperationMetadata
from ..operation_registry import registry


class DuplicateFaceDetection(BaseOperation):
    """
    Detects duplicate faces sharing the same vertex indices.

    Two faces are duplicates if they reference the same set of vertex
    indices, regardless of winding order. This catches both exact copies
    and faces with reversed winding but identical vertex sets.
    """

    @property
    def metadata(self) -> OperationMetadata:
        return OperationMetadata(
            operation_id='topology.detect_duplicate_faces',
            display_name='Duplicate Face Detection',
            description='Identifies faces that share the same set of vertex '
                       'indices, indicating duplicate geometry.',
            category=IssueCategory.TOPOLOGICAL,
            supported_libraries=['trimesh', 'pyvista', 'pymeshlab'],
            recommended_library='pyvista',
            requires_adjacency=False,
            complexity='O(F * k log k)',
            tags=['topology', 'faces', 'duplicate', 'degenerate']
        )

    def execute(self, mesh: MeshInput) -> List[MeshIssue]:
        """Execute duplicate face detection."""
        faces = mesh.faces
        n_faces = len(faces)

        if n_faces == 0:
            return []

        # Build canonical key -> list of face indices
        key_to_faces: defaultdict = defaultdict(list)
        for fi in range(n_faces):
            face = faces[fi]
            key = tuple(sorted(face.tolist()))
            key_to_faces[key].append(fi)

        # Find groups with duplicates (>1 face sharing same key)
        duplicate_groups: List[Tuple[tuple, List[int]]] = []
        for key, face_indices in key_to_faces.items():
            if len(face_indices) > 1:
                duplicate_groups.append((key, face_indices))

        if not duplicate_groups:
            return []

        # Count total duplicate faces (all copies beyond the first per group)
        total_duplicates = sum(len(fis) - 1 for _, fis in duplicate_groups)
        total_face_indices = [fi for _, fis in duplicate_groups for fi in fis]

        # Build locations — one per duplicate group
        vertices = mesh.vertices
        locations = []
        for key, face_indices in duplicate_groups:
            # Use first vertex of the face for coordinates
            sample_vi = int(key[0])
            coords = tuple(vertices[sample_vi]) if sample_vi < len(vertices) else None
            locations.append(IssueLocation(
                element_type='face',
                element_index=face_indices[0],
                coordinates=coords,
                adjacent_elements={
                    'duplicate_face_indices': face_indices,
                    'vertex_set': list(key),
                    'duplicate_count': len(face_indices) - 1,
                }
            ))

        violation_rate = total_duplicates / n_faces

        if violation_rate >= 0.05:
            severity = IssueSeverity.WARNING
        else:
            severity = IssueSeverity.INFO

        issue = MeshIssue(
            operation_id=self.metadata.operation_id,
            category=IssueCategory.TOPOLOGICAL,
            severity=severity,
            message=f'Detected {total_duplicates} duplicate faces in '
                   f'{len(duplicate_groups)} groups '
                   f'({violation_rate * 100:.2f}% of {n_faces} faces). '
                   f'Duplicate faces share identical vertex index sets.',
            locations=locations,
            metrics={
                'duplicate_face_count': total_duplicates,
                'unique_face_groups': len(duplicate_groups),
                'violation_rate': violation_rate,
                'total_faces': n_faces,
            },
            metadata={
                'groups': [
                    {'vertex_set': list(key), 'face_indices': fis}
                    for key, fis in duplicate_groups
                ]
            }
        )

        return [issue]


# Register operation
registry.register(DuplicateFaceDetection())
