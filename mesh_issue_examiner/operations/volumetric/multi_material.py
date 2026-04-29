"""
Multi-material interface checks for volumetric (3D) meshes.

Implements three operations for meshes with multiple materials or regions:

1. **MaterialInterfaceDetection** — Detects faces shared between cells of
   different materials, groups them by material pair, and reports interface
   statistics.

2. **InterfaceQualityCheck** — Assesses the quality of transition elements
   at material interfaces.  Poor quality at interfaces can cause solver
   convergence issues, stress concentrations, and heat-transfer errors.

3. **PerMaterialStatistics** — Computes per-material-region quality
   statistics (cell count, volume, average quality, etc.).

All operations require ``cell_materials`` to be set on the ``MeshInput``
object.  If no materials are assigned, the operations return empty results.

Research references:
    - Simpleware FE Datasheet — "Multipart Volumetric Meshing"
    - Nairn et al. — "Modeling Imperfect Interfaces in the Material Point Method"
    - FEA Academy — Thermal Analysis material interface guidelines
    - Marine FEA Guidelines — "Heterogeneous interfaces" refinement
"""

import logging
from collections import defaultdict
from typing import List, Dict, Optional, Tuple, Set

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
from ._face_tables import ELEMENT_FACES

logger = logging.getLogger(__name__)

# Maximum number of IssueLocation objects per issue (prevents huge reports)
_MAX_LOCATIONS = 100


# ======================================================================
# Shared helper: face-to-cell map with material info
# ======================================================================

def _build_face_to_cell_map(
    mesh: MeshInput,
) -> Dict[Tuple[int, ...], List[dict]]:
    """Build mapping from sorted face tuples to list of cell descriptors.

    Each entry is ``{face_key: [{'global_idx', 'element_type', 'face_verts'}, ...]}``.
    """
    face_map: Dict[Tuple[int, ...], List[dict]] = {}
    global_idx = 0

    for element_type, connectivity in mesh.volume_cells.items():
        if element_type not in ELEMENT_FACES:
            global_idx += len(connectivity)
            continue

        face_defs = ELEMENT_FACES[element_type]

        for local_idx in range(len(connectivity)):
            cell_verts = connectivity[local_idx]

            for face_def in face_defs:
                face_verts = tuple(int(cell_verts[i]) for i in face_def)
                face_key = tuple(sorted(face_verts))

                entry = {
                    'global_idx': global_idx,
                    'element_type': element_type,
                    'face_verts': face_verts,
                }

                if face_key not in face_map:
                    face_map[face_key] = []
                face_map[face_key].append(entry)

            global_idx += 1

    return face_map


def _get_cell_connectivity(
    mesh: MeshInput, element_type: str, global_idx: int
) -> Optional[np.ndarray]:
    """Look up the vertex indices of a cell by global index."""
    offset = 0
    for etype, conn in mesh.volume_cells.items():
        n = len(conn)
        if etype == element_type and offset <= global_idx < offset + n:
            return conn[global_idx - offset]
        offset += n
    return None


def _compute_tet_volume(vertices: np.ndarray, cell_conn: np.ndarray) -> float:
    """Compute volume of a tetrahedron from its 4 vertex indices."""
    v = vertices[cell_conn[:4]]
    # Volume = |det([v1-v0, v2-v0, v3-v0])| / 6
    m = np.column_stack([v[1] - v[0], v[2] - v[0], v[3] - v[0]])
    return abs(float(np.linalg.det(m))) / 6.0


def _compute_cell_volume(vertices: np.ndarray, element_type: str, cell_conn: np.ndarray) -> float:
    """Compute volume of a volumetric cell."""
    if element_type == 'tet4':
        return _compute_tet_volume(vertices, cell_conn)
    elif element_type == 'hex8':
        return _compute_hex_volume(vertices, cell_conn)
    elif element_type == 'prism6':
        return _compute_prism_volume(vertices, cell_conn)
    elif element_type == 'pyramid5':
        return _compute_pyramid_volume(vertices, cell_conn)
    else:
        # Fallback: bounding box volume
        v = vertices[cell_conn]
        bb = np.max(v, axis=0) - np.min(v, axis=0)
        return float(np.prod(bb))


def _compute_hex_volume(vertices: np.ndarray, cell_conn: np.ndarray) -> float:
    """Compute approximate volume of a hexahedron by splitting into 5 tets."""
    v = vertices[cell_conn[:8]]
    # Decompose hex into 5 tetrahedra (standard decomposition)
    tets = [
        [0, 1, 3, 4],
        [1, 2, 3, 6],
        [1, 4, 5, 6],
        [3, 4, 6, 7],
        [1, 3, 4, 6],
    ]
    vol = 0.0
    for tet in tets:
        m = np.column_stack([
            v[tet[1]] - v[tet[0]],
            v[tet[2]] - v[tet[0]],
            v[tet[3]] - v[tet[0]],
        ])
        vol += abs(float(np.linalg.det(m))) / 6.0
    return vol


def _compute_prism_volume(vertices: np.ndarray, cell_conn: np.ndarray) -> float:
    """Compute volume of a triangular prism by splitting into 3 tets."""
    v = vertices[cell_conn[:6]]
    # Split prism into 3 tetrahedra
    tets = [
        [0, 1, 2, 3],
        [1, 2, 3, 4],
        [2, 3, 4, 5],
    ]
    vol = 0.0
    for tet in tets:
        m = np.column_stack([
            v[tet[1]] - v[tet[0]],
            v[tet[2]] - v[tet[0]],
            v[tet[3]] - v[tet[0]],
        ])
        vol += abs(float(np.linalg.det(m))) / 6.0
    return vol


def _compute_pyramid_volume(vertices: np.ndarray, cell_conn: np.ndarray) -> float:
    """Compute volume of a pyramid by splitting into 2 tets."""
    v = vertices[cell_conn[:5]]
    # Split quad base into 2 triangles, each forming a tet with apex
    tets = [
        [0, 1, 2, 4],
        [0, 2, 3, 4],
    ]
    vol = 0.0
    for tet in tets:
        m = np.column_stack([
            v[tet[1]] - v[tet[0]],
            v[tet[2]] - v[tet[0]],
            v[tet[3]] - v[tet[0]],
        ])
        vol += abs(float(np.linalg.det(m))) / 6.0
    return vol


def _compute_scaled_jacobian_tet(vertices: np.ndarray, cell_conn: np.ndarray) -> float:
    """Compute scaled Jacobian for a tet4 element."""
    v = vertices[cell_conn[:4]]

    # Edge vectors from node 0
    edges = [v[1] - v[0], v[2] - v[0], v[3] - v[0]]

    # Jacobian at node 0
    J = np.column_stack(edges)
    det_J = np.linalg.det(J)
    norm_J = np.linalg.norm(J, 'fro')

    if norm_J < 1e-15:
        return 0.0

    # Scaled Jacobian = det(J) / (||J||_F ^ (n/2) for n=3... simplified
    # Standard formula: det(J) / product of edge norms at each node
    edge_norms = [np.linalg.norm(e) for e in edges]
    product = edge_norms[0] * edge_norms[1] * edge_norms[2]

    if product < 1e-15:
        return 0.0

    return float(det_J / product)


# ======================================================================
# Operation 1: MaterialInterfaceDetection
# ======================================================================

class MaterialInterfaceDetection(BaseOperation):
    """Detects interfaces between material regions in multi-material meshes.

    Builds a face-to-cell adjacency map and identifies faces shared by cells
    belonging to different materials.  Reports each unique material pair as a
    separate issue with statistics (interface face count, total area, etc.).

    Requires ``MeshInput.cell_materials`` to be set; returns an empty list
    otherwise.
    """

    def __init__(self, config: Optional[OperationConfig] = None):
        super().__init__(config)

    @property
    def metadata(self) -> OperationMetadata:
        return OperationMetadata(
            operation_id='quality_3d.detect_material_interfaces',
            display_name='Material Interface Detection (3D)',
            description='Detects and reports faces at material boundaries in '
                       'multi-material volumetric meshes.',
            category=IssueCategory.QUALITY,
            supported_libraries=['pyvista'],
            recommended_library='pyvista',
            requires_adjacency=True,
            complexity='O(n)',
            tags=['quality', 'multi-material', 'interface', 'volumetric', '3d'],
        )

    def execute(self, mesh: MeshInput) -> List[MeshIssue]:
        """Detect material interfaces.

        Returns an empty list if the mesh has no volume cells, no material
        assignments, or only a single material.
        """
        if not mesh.has_volume_cells:
            return []
        if not mesh.has_multiple_materials():
            return []

        face_to_cells = _build_face_to_cell_map(mesh)
        if not face_to_cells:
            return []

        # Find interface faces (shared by cells of different materials)
        interfaces = self._detect_interfaces(mesh, face_to_cells)
        if not interfaces:
            return []

        issues = []
        vertices = mesh._vertices

        for (mat1, mat2), face_entries in interfaces.items():
            # Compute interface face areas
            total_area = 0.0
            interface_cells: Set[int] = set()
            for entry in face_entries:
                fv = entry['face_verts']
                face_verts = vertices[list(fv)]
                area = self._face_area(face_verts)
                total_area += area
                interface_cells.add(entry['cell1_idx'])
                interface_cells.add(entry['cell2_idx'])

            mat1_name = _material_label(mesh, mat1)
            mat2_name = _material_label(mesh, mat2)

            locations = [
                IssueLocation(element_type='cell', element_index=int(idx))
                for idx in sorted(interface_cells)[:_MAX_LOCATIONS]
            ]

            issues.append(MeshIssue(
                operation_id=self.metadata.operation_id,
                category=IssueCategory.QUALITY,
                severity=IssueSeverity.INFO,
                message=(
                    f'Material interface between {mat1_name} and {mat2_name}: '
                    f'{len(face_entries)} interface face(s), '
                    f'{len(interface_cells)} adjacent cell(s), '
                    f'total area {total_area:.6g}.'
                ),
                metrics={
                    'interface_face_count': len(face_entries),
                    'interface_cell_count': len(interface_cells),
                    'total_interface_area': total_area,
                    'material_1': int(mat1),
                    'material_2': int(mat2),
                },
                metadata={
                    'material_1_name': mat1_name,
                    'material_2_name': mat2_name,
                },
                locations=locations,
            ))

        return issues

    def _detect_interfaces(
        self,
        mesh: MeshInput,
        face_to_cells: Dict[Tuple[int, ...], List[dict]],
    ) -> Dict[Tuple[int, int], List[dict]]:
        """Detect faces between different materials.

        Returns dict mapping (mat1, mat2) pairs to interface face entries.
        """
        interfaces: Dict[Tuple[int, int], List[dict]] = defaultdict(list)

        for face_key, cell_entries in face_to_cells.items():
            if len(cell_entries) != 2:
                continue  # Boundary or non-manifold face

            c1 = cell_entries[0]
            c2 = cell_entries[1]

            mat1 = mesh.get_material_for_cell(c1['global_idx'])
            mat2 = mesh.get_material_for_cell(c2['global_idx'])

            if mat1 is None or mat2 is None:
                continue

            if mat1 != mat2:
                pair = tuple(sorted([mat1, mat2]))
                interfaces[pair].append({
                    'face_key': face_key,
                    'face_verts': c1['face_verts'],
                    'cell1_idx': c1['global_idx'],
                    'cell2_idx': c2['global_idx'],
                })

        return interfaces

    @staticmethod
    def _face_area(face_verts: np.ndarray) -> float:
        """Compute area of a polygon face from its vertices."""
        n = len(face_verts)
        if n < 3:
            return 0.0
        if n == 3:
            e1 = face_verts[1] - face_verts[0]
            e2 = face_verts[2] - face_verts[0]
            return 0.5 * float(np.linalg.norm(np.cross(e1, e2)))
        # Triangulate from first vertex
        area = 0.0
        for i in range(1, n - 1):
            e1 = face_verts[i] - face_verts[0]
            e2 = face_verts[i + 1] - face_verts[0]
            area += 0.5 * float(np.linalg.norm(np.cross(e1, e2)))
        return area


# ======================================================================
# Operation 2: InterfaceQualityCheck
# ======================================================================

class InterfaceQualityCheck(BaseOperation):
    """Assesses quality of transition elements at material interfaces.

    Checks cells adjacent to material interfaces for poor quality metrics
    (scaled Jacobian, volume).  Poor quality at interfaces can cause:
    - Solver convergence issues
    - Stress concentrations
    - Heat transfer errors

    Requires ``MeshInput.cell_materials`` to be set; returns an empty list
    otherwise.
    """

    def __init__(
        self,
        config: Optional[OperationConfig] = None,
        min_scaled_jacobian: float = 0.1,
        min_volume: float = 1e-12,
    ):
        super().__init__(config)
        self.min_scaled_jacobian = min_scaled_jacobian
        self.min_volume = min_volume

    @property
    def metadata(self) -> OperationMetadata:
        return OperationMetadata(
            operation_id='quality_3d.check_interface_quality',
            display_name='Interface Quality Check (3D)',
            description='Assesses quality of cells at material interfaces '
                       'for poor shape metrics that may cause solver issues.',
            category=IssueCategory.QUALITY,
            supported_libraries=['pyvista'],
            recommended_library='pyvista',
            requires_adjacency=True,
            complexity='O(n)',
            tags=['quality', 'multi-material', 'interface', 'volumetric', '3d'],
        )

    def execute(self, mesh: MeshInput) -> List[MeshIssue]:
        """Check quality of cells at material interfaces."""
        if not mesh.has_volume_cells:
            return []
        if not mesh.has_multiple_materials():
            return []

        face_to_cells = _build_face_to_cell_map(mesh)
        if not face_to_cells:
            return []

        # Find interface cells
        interface_cells_by_pair: Dict[Tuple[int, int], Set[int]] = defaultdict(set)

        for face_key, cell_entries in face_to_cells.items():
            if len(cell_entries) != 2:
                continue
            c1, c2 = cell_entries[0], cell_entries[1]
            mat1 = mesh.get_material_for_cell(c1['global_idx'])
            mat2 = mesh.get_material_for_cell(c2['global_idx'])
            if mat1 is None or mat2 is None or mat1 == mat2:
                continue
            pair = tuple(sorted([mat1, mat2]))
            interface_cells_by_pair[pair].add(c1['global_idx'])
            interface_cells_by_pair[pair].add(c2['global_idx'])

        if not interface_cells_by_pair:
            return []

        issues: List[MeshIssue] = []
        vertices = mesh._vertices

        for (mat1, mat2), cell_indices in interface_cells_by_pair.items():
            pair_issues = self._check_interface_cells(
                mesh, vertices, mat1, mat2, cell_indices
            )
            issues.extend(pair_issues)

        return issues

    def _check_interface_cells(
        self,
        mesh: MeshInput,
        vertices: np.ndarray,
        mat1: int,
        mat2: int,
        cell_indices: Set[int],
    ) -> List[MeshIssue]:
        """Check quality metrics for cells at a specific material interface."""
        poor_jacobian = []
        poor_volume = []
        jacobian_values = []
        volume_values = []

        for gidx in cell_indices:
            conn, etype = self._get_cell_data(mesh, gidx)
            if conn is None:
                continue

            # Compute quality
            if etype == 'tet4':
                sj = _compute_scaled_jacobian_tet(vertices, conn)
                jacobian_values.append(sj)
                if sj < self.min_scaled_jacobian:
                    poor_jacobian.append((gidx, sj))

            vol = _compute_cell_volume(vertices, etype, conn)
            volume_values.append(vol)
            if vol < self.min_volume:
                poor_volume.append((gidx, vol))

        issues: List[MeshIssue] = []
        mat1_name = _material_label(mesh, mat1)
        mat2_name = _material_label(mesh, mat2)

        # Report poor Jacobian
        if poor_jacobian:
            violation_rate = len(poor_jacobian) / max(len(jacobian_values), 1)
            severity = self._determine_severity(violation_rate)

            locations = [
                IssueLocation(element_type='cell', element_index=int(idx))
                for idx, _ in poor_jacobian[:_MAX_LOCATIONS]
            ]

            issues.append(MeshIssue(
                operation_id=self.metadata.operation_id,
                category=IssueCategory.QUALITY,
                severity=severity,
                message=(
                    f'{len(poor_jacobian)} cell(s) at interface '
                    f'{mat1_name}-{mat2_name} have scaled Jacobian below '
                    f'{self.min_scaled_jacobian} '
                    f'(min: {min(v for _, v in poor_jacobian):.4f}). '
                    f'Poor quality at interfaces may cause convergence issues.'
                ),
                metrics={
                    'violation_count': len(poor_jacobian),
                    'total_interface_cells': len(jacobian_values) or len(cell_indices),
                    'violation_rate': violation_rate,
                    'min_scaled_jacobian': min(v for _, v in poor_jacobian) if poor_jacobian else 0,
                    'mean_scaled_jacobian': float(np.mean(jacobian_values)) if jacobian_values else 0,
                    'threshold': self.min_scaled_jacobian,
                },
                metadata={
                    'check': 'scaled_jacobian',
                    'material_1': int(mat1),
                    'material_2': int(mat2),
                },
                locations=locations,
            ))

        # Report zero/negative volume
        if poor_volume:
            violation_rate = len(poor_volume) / max(len(volume_values), 1)
            severity = IssueSeverity.CRITICAL if violation_rate > 0.1 else IssueSeverity.ERROR

            locations = [
                IssueLocation(element_type='cell', element_index=int(idx))
                for idx, _ in poor_volume[:_MAX_LOCATIONS]
            ]

            issues.append(MeshIssue(
                operation_id=self.metadata.operation_id,
                category=IssueCategory.QUALITY,
                severity=severity,
                message=(
                    f'{len(poor_volume)} cell(s) at interface '
                    f'{mat1_name}-{mat2_name} have near-zero or negative volume '
                    f'(min: {min(v for _, v in poor_volume):.4e}). '
                    f'Zero-volume cells at interfaces are invalid.'
                ),
                metrics={
                    'violation_count': len(poor_volume),
                    'total_interface_cells': len(volume_values) or len(cell_indices),
                    'violation_rate': violation_rate,
                    'min_volume': min(v for _, v in poor_volume) if poor_volume else 0,
                    'threshold': self.min_volume,
                },
                metadata={
                    'check': 'volume',
                    'material_1': int(mat1),
                    'material_2': int(mat2),
                },
                locations=locations,
            ))

        return issues

    @staticmethod
    def _get_cell_data(mesh: MeshInput, global_idx: int) -> Tuple[Optional[np.ndarray], str]:
        """Get cell connectivity and element type for a global index."""
        offset = 0
        for etype, conn in mesh.volume_cells.items():
            n = len(conn)
            if offset <= global_idx < offset + n:
                return conn[global_idx - offset], etype
            offset += n
        return None, ''

    @staticmethod
    def _determine_severity(violation_rate: float) -> IssueSeverity:
        if violation_rate > 0.5:
            return IssueSeverity.CRITICAL
        elif violation_rate > 0.2:
            return IssueSeverity.ERROR
        elif violation_rate > 0.05:
            return IssueSeverity.WARNING
        else:
            return IssueSeverity.INFO


# ======================================================================
# Operation 3: PerMaterialStatistics
# ======================================================================

class PerMaterialStatistics(BaseOperation):
    """Computes quality statistics per material region.

    For each material in the mesh, computes:
    - Cell count and total volume
    - Average/min/max scaled Jacobian (for tet4 cells)
    - Volume distribution statistics

    Results are reported as INFO-level issues so they are always included
    in the inspection report.

    Requires ``MeshInput.cell_materials`` to be set; returns an empty list
    otherwise.
    """

    def __init__(self, config: Optional[OperationConfig] = None):
        super().__init__(config)

    @property
    def metadata(self) -> OperationMetadata:
        return OperationMetadata(
            operation_id='quality_3d.per_material_statistics',
            display_name='Per-Material Statistics (3D)',
            description='Computes per-material-region quality statistics '
                       'including cell count, volume, and quality distributions.',
            category=IssueCategory.QUALITY,
            supported_libraries=['pyvista'],
            recommended_library='pyvista',
            requires_adjacency=False,
            complexity='O(n)',
            tags=['quality', 'multi-material', 'statistics', 'volumetric', '3d'],
        )

    def execute(self, mesh: MeshInput) -> List[MeshIssue]:
        """Compute and report per-material statistics."""
        if not mesh.has_volume_cells:
            return []
        if mesh._cell_materials is None:
            return []

        unique_materials = np.unique(mesh._cell_materials)
        if len(unique_materials) == 0:
            return []

        issues: List[MeshIssue] = []
        vertices = mesh._vertices

        for material_id in unique_materials:
            stats = self._compute_material_stats(mesh, vertices, int(material_id))

            issues.append(MeshIssue(
                operation_id=self.metadata.operation_id,
                category=IssueCategory.QUALITY,
                severity=IssueSeverity.INFO,
                message=(
                    f'Material {stats["material_name"]} '
                    f'(ID {int(material_id)}): '
                    f'{stats["cell_count"]} cell(s), '
                    f'volume {stats["total_volume"]:.6g}, '
                    f'avg quality {stats["avg_quality"]:.4f}.'
                ),
                metrics=stats,
                metadata={'material_id': int(material_id)},
            ))

        return issues

    def _compute_material_stats(
        self,
        mesh: MeshInput,
        vertices: np.ndarray,
        material_id: int,
    ) -> Dict[str, float]:
        """Compute statistics for a single material region."""
        cell_indices = mesh.get_cells_by_material(material_id)
        n_cells = len(cell_indices)

        if n_cells == 0:
            return {
                'material_id': material_id,
                'material_name': _material_label(mesh, material_id),
                'cell_count': 0,
                'total_volume': 0.0,
                'avg_quality': 0.0,
                'min_quality': 0.0,
                'max_quality': 0.0,
            }

        volumes = []
        qualities = []

        for gidx in cell_indices:
            conn, etype = _get_cell_by_global_idx(mesh, int(gidx))
            if conn is None:
                continue

            vol = _compute_cell_volume(vertices, etype, conn)
            volumes.append(vol)

            if etype == 'tet4':
                sj = _compute_scaled_jacobian_tet(vertices, conn)
                qualities.append(sj)

        stats: Dict[str, float] = {
            'material_id': float(material_id),
            'material_name': _material_label(mesh, material_id),
            'cell_count': float(n_cells),
            'total_volume': float(np.sum(volumes)) if volumes else 0.0,
            'avg_volume': float(np.mean(volumes)) if volumes else 0.0,
            'min_volume': float(np.min(volumes)) if volumes else 0.0,
            'max_volume': float(np.max(volumes)) if volumes else 0.0,
        }

        if qualities:
            stats.update({
                'avg_quality': float(np.mean(qualities)),
                'min_quality': float(np.min(qualities)),
                'max_quality': float(np.max(qualities)),
                'std_quality': float(np.std(qualities)),
            })
        else:
            stats.update({
                'avg_quality': 0.0,
                'min_quality': 0.0,
                'max_quality': 0.0,
            })

        return stats


# ======================================================================
# Module-level helpers
# ======================================================================

def _material_label(mesh: MeshInput, material_id: int) -> str:
    """Get human-readable name for a material ID."""
    if mesh._material_names and material_id in mesh._material_names:
        return mesh._material_names[material_id]
    return f'Material_{material_id}'


def _get_cell_by_global_idx(
    mesh: MeshInput, global_idx: int
) -> Tuple[Optional[np.ndarray], str]:
    """Look up cell connectivity and type by global index."""
    offset = 0
    for etype, conn in mesh.volume_cells.items():
        n = len(conn)
        if offset <= global_idx < offset + n:
            return conn[global_idx - offset], etype
        offset += n
    return None, ''


# Register all three operations
registry.register(MaterialInterfaceDetection())
registry.register(InterfaceQualityCheck())
registry.register(PerMaterialStatistics())
