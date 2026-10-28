"""
Boundary layer quality check operation for volumetric (3D) meshes.

Validates anisotropic boundary layer (inflation layer) meshes common in CFD.
Boundary layers are thin layers of prismatic/hexahedral elements grown from
wall surfaces to capture viscous effects.

Checks implemented:
    - Layer count: minimum number of layers for adequate resolution
    - First layer height: distance from wall to first cell centroid
    - Growth ratio: height(i+1)/height(i) between consecutive layers
    - Aspect ratio: layer height vs. in-plane edge length
    - Orthogonality: alignment of cell with wall normal direction
    - Layer coverage: fraction of wall surface covered by BL cells

Research-based thresholds (Gap #9):
    - Growth ratio: 1.0-1.2 excellent, 1.2-1.3 good, 1.3-1.5 acceptable,
      >1.5 poor (may cause convergence issues)
    - Orthogonality: <20 deg excellent, <45 deg good, <65 deg maximum
      (per OpenFOAM recommendations)
    - Aspect ratio: context-dependent; up to 1000 acceptable for y+ targets
    - Minimum layers: 3 for basic, 5+ recommended for production CFD
"""

import logging
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

# Element types that typically form boundary layers
BL_ELEMENT_TYPES = {'prism6', 'hex8', 'pyramid5'}

# Base face indices per element type: these are the faces that form the
# wall-adjacent surface of a BL cell.  For prisms these are the two
# triangular faces; for hexes the bottom and top quad faces; for pyramids
# the base quad face.
BASE_FACE_INDICES: Dict[str, List[int]] = {
    'prism6': [0, 1],   # bottom triangle, top triangle
    'hex8': [0, 1],     # bottom quad, top quad
    'pyramid5': [0],    # base quad
}

# Maximum number of IssueLocation objects per issue (prevents huge reports)
_MAX_LOCATIONS = 100

# Wall-seed calibration (2026-10-28):
#   1. Thin-cell gate — a BL seed cell must be thin: its in-plane extent
#      must exceed its wall-normal height by at least this factor.  Regular
#      hex grids have near-cubic cells and must NOT be treated as boundary
#      layers (they previously seeded on every volume fixture).
#   2. Growth-away gate — inflation layers grow away from the wall, so the
#      neighbour through the opposite base face must not be systematically
#      thinner than the seed.  This rejects seeds taken from the BL's outer
#      surface (the wrong side of the stack).
_MIN_SEED_THIN_RATIO = 2.0
_GROWTH_AWAY_TOLERANCE = 0.9


class BoundaryLayerQualityCheck(BaseOperation):
    """
    Quality assessment for boundary layer (inflation layer) meshes.

    Identifies boundary layer cells adjacent to wall surfaces, traces layers
    outward, and computes CFD-relevant quality metrics including layer heights,
    growth ratios, aspect ratios, and orthogonality.

    The operation detects wall faces (base faces of BL elements that lie on the
    mesh boundary) and traces prism/hex stacks from those faces outward.

    Parameters
    ----------
    target_first_layer_height : float or None
        Target height for the first layer.  When provided, the first layer
        height check reports deviations from this target.  The user is
        responsible for converting y+ requirements to an actual height using
        their flow conditions (y+ = u_tau * y / nu).
    max_growth_ratio : float
        Maximum acceptable layer-to-layer growth ratio (default 1.2,
        per CFD best practices).
    max_aspect_ratio : float
        Maximum acceptable aspect ratio (default 1000).
    max_orthogonality_angle : float
        Maximum deviation from wall normal in degrees (default 20.0,
        per research findings).
    min_layers : int
        Minimum number of boundary layer elements required (default 3).
    """

    def __init__(
        self,
        config: Optional[OperationConfig] = None,
        target_first_layer_height: Optional[float] = None,
        max_growth_ratio: float = 1.2,
        max_aspect_ratio: float = 1000.0,
        max_orthogonality_angle: float = 20.0,
        min_layers: int = 3,
    ):
        super().__init__(config)
        self.target_first_layer_height = target_first_layer_height
        self.max_growth_ratio = max_growth_ratio
        self.max_aspect_ratio = max_aspect_ratio
        self.max_orthogonality_angle = max_orthogonality_angle
        self.min_layers = min_layers

    @property
    def metadata(self) -> OperationMetadata:
        return OperationMetadata(
            operation_id='quality_3d.check_boundary_layer',
            display_name='Boundary Layer Quality Check (3D)',
            description='Validates anisotropic boundary layer mesh quality '
                       'including layer height, growth ratio, aspect ratio, '
                       'orthogonality, and coverage.',
            category=IssueCategory.QUALITY,
            supported_libraries=['pyvista'],
            recommended_library='pyvista',
            requires_adjacency=True,
            complexity='O(n)',
            tags=['quality', 'boundary_layer', 'inflation', 'CFD', 'prism']
        )

    def execute(self, mesh: MeshInput) -> List[MeshIssue]:
        """Execute boundary layer quality check.

        Returns an empty list if no boundary layer cells are detected.
        """
        if not mesh.has_volume_cells:
            return []

        # Build adjacency data
        face_to_cells = _build_face_to_cell_map(mesh)
        if not face_to_cells:
            return []

        # Identify boundary faces (faces shared by exactly one cell)
        boundary_face_keys = [
            key for key, cells in face_to_cells.items()
            if len(cells) == 1
        ]
        boundary_set = set(boundary_face_keys)
        if not boundary_face_keys:
            return []

        # Identify wall seed cells: BL-type cells whose *base face* is on the
        # boundary.  This distinguishes the wall-adjacent face (the face the
        # BL grows from) from side faces that merely happen to be on the
        # exterior of the mesh.
        seed_cells = _identify_wall_seed_cells(
            mesh, face_to_cells, boundary_set
        )
        if not seed_cells:
            return []

        # Trace layers outward from wall
        layers = _trace_boundary_layers(
            mesh, seed_cells, face_to_cells, boundary_set
        )
        if not layers:
            return []

        # Compute metrics and run checks
        issues: List[MeshIssue] = []
        vertices = mesh._vertices

        issues.extend(self._check_layer_count(layers))

        layer_heights = _compute_layer_heights(vertices, mesh, layers)
        if layer_heights:
            issues.extend(self._check_first_layer_height(layer_heights))
            issues.extend(self._check_growth_ratios(layer_heights, layers))

        issues.extend(self._check_aspect_ratios(vertices, mesh, layers))
        issues.extend(
            self._check_orthogonality(vertices, mesh, layers)
        )
        issues.extend(
            self._check_coverage(
                seed_cells, boundary_face_keys, mesh, layers, face_to_cells
            )
        )

        return issues

    # ------------------------------------------------------------------
    # Individual check methods
    # ------------------------------------------------------------------

    def _check_layer_count(self, layers: List[List[dict]]) -> List[MeshIssue]:
        """Check if enough boundary layer elements are present."""
        n_layers = len(layers)
        if n_layers < self.min_layers:
            return [MeshIssue(
                operation_id=self.metadata.operation_id,
                category=IssueCategory.QUALITY,
                severity=IssueSeverity.WARNING,
                message=(
                    f'Boundary layer has {n_layers} layer(s), below the '
                    f'recommended minimum of {self.min_layers}. '
                    f'Insufficient layers may not resolve viscous gradients.'
                ),
                metrics={
                    'layer_count': n_layers,
                    'min_required': self.min_layers,
                },
                metadata={'check': 'layer_count'},
            )]
        return []

    def _check_first_layer_height(
        self, layer_heights: List[float]
    ) -> List[MeshIssue]:
        """Check first layer height against target (y+ proxy)."""
        if not layer_heights or self.target_first_layer_height is None:
            return []

        first_height = layer_heights[0]
        if first_height <= 0:
            return []

        deviation = abs(first_height - self.target_first_layer_height)
        rel_deviation = deviation / self.target_first_layer_height

        # Report as INFO: actual y+ requires flow parameters we don't have
        return [MeshIssue(
            operation_id=self.metadata.operation_id,
            category=IssueCategory.QUALITY,
            severity=IssueSeverity.INFO,
            message=(
                f'First layer height is {first_height:.6f} vs. target '
                f'{self.target_first_layer_height:.6f} '
                f'(deviation {rel_deviation * 100:.1f}%). '
                f'Convert to y+ using flow conditions: '
                f'y+ = u_tau * y / nu.'
            ),
            metrics={
                'first_layer_height': first_height,
                'target_height': self.target_first_layer_height,
                'relative_deviation': rel_deviation,
            },
            metadata={'check': 'first_layer_height'},
        )]

    def _check_growth_ratios(
        self,
        layer_heights: List[float],
        layers: List[List[dict]],
    ) -> List[MeshIssue]:
        """Check layer-to-layer growth ratios.

        Research thresholds:
            1.0-1.2 excellent, 1.2-1.3 good, 1.3-1.5 acceptable, >1.5 poor.
        """
        violations = []
        ratios = []

        for i in range(len(layer_heights) - 1):
            h_curr = layer_heights[i]
            h_next = layer_heights[i + 1]
            if h_curr > 0:
                ratio = h_next / h_curr
            else:
                ratio = float('inf')

            ratios.append(ratio)
            if ratio > self.max_growth_ratio:
                violations.append((i, i + 1, ratio))

        if not violations:
            return []

        max_ratio = max(v[2] for v in violations)
        violation_rate = len(violations) / max(len(ratios), 1)

        severity = self._determine_severity(violation_rate)

        # Build locations from the violating layers
        locations = []
        for layer_idx, _, _ in violations:
            for cell_info in layers[layer_idx]:
                locations.append(IssueLocation(
                    element_type='cell',
                    element_index=cell_info['global_idx'],
                ))
                if len(locations) >= _MAX_LOCATIONS:
                    break
            if len(locations) >= _MAX_LOCATIONS:
                break

        return [MeshIssue(
            operation_id=self.metadata.operation_id,
            category=IssueCategory.QUALITY,
            severity=severity,
            message=(
                f'{len(violations)} layer transition(s) exceed max growth '
                f'ratio of {self.max_growth_ratio:.2f} '
                f'(max observed: {max_ratio:.2f}). '
                f'Excessive growth ratio may cause CFD convergence issues.'
            ),
            metrics={
                'violation_count': len(violations),
                'total_transitions': len(ratios),
                'violation_rate': violation_rate,
                'max_growth_ratio': max_ratio,
                'mean_growth_ratio': float(np.mean(ratios)),
                'threshold': self.max_growth_ratio,
            },
            metadata={
                'check': 'growth_ratio',
                'violating_transitions': [
                    {'from_layer': v[0], 'to_layer': v[1], 'ratio': v[2]}
                    for v in violations[:50]
                ],
            },
            locations=locations,
        )]

    def _check_aspect_ratios(
        self,
        vertices: np.ndarray,
        mesh: MeshInput,
        layers: List[List[dict]],
    ) -> List[MeshIssue]:
        """Check aspect ratio of BL cells (height / in-plane edge length).

        Aspect ratios up to 1000 are generally acceptable for boundary layers.
        Very high AR (>10000) may indicate mesh quality problems.
        """
        violations = []
        ar_values = []

        for layer_idx, layer_cells in enumerate(layers):
            for cell_info in layer_cells:
                cell_verts = _get_cell_vertices(vertices, mesh, cell_info)
                ar = _compute_cell_aspect_ratio(cell_verts, cell_info['element_type'])
                ar_values.append(ar)

                if ar > self.max_aspect_ratio:
                    violations.append({
                        'global_idx': cell_info['global_idx'],
                        'layer': layer_idx,
                        'aspect_ratio': ar,
                    })

        if not ar_values or not violations:
            return []

        violation_rate = len(violations) / len(ar_values)
        severity = self._determine_severity(violation_rate)

        locations = [
            IssueLocation(
                element_type='cell',
                element_index=v['global_idx'],
            )
            for v in violations[:_MAX_LOCATIONS]
        ]

        return [MeshIssue(
            operation_id=self.metadata.operation_id,
            category=IssueCategory.QUALITY,
            severity=severity,
            message=(
                f'{len(violations)}/{len(ar_values)} BL cells '
                f'({violation_rate * 100:.1f}%) exceed max aspect ratio '
                f'of {self.max_aspect_ratio:.0f} '
                f'(max observed: {max(v["aspect_ratio"] for v in violations):.1f}).'
            ),
            metrics={
                'violation_count': len(violations),
                'total_bl_cells': len(ar_values),
                'violation_rate': violation_rate,
                'mean_ar': float(np.mean(ar_values)),
                'max_ar': float(np.max(ar_values)),
                'threshold': self.max_aspect_ratio,
            },
            metadata={'check': 'aspect_ratio'},
            locations=locations,
        )]

    def _check_orthogonality(
        self,
        vertices: np.ndarray,
        mesh: MeshInput,
        layers: List[List[dict]],
    ) -> List[MeshIssue]:
        """Check orthogonality of BL cells relative to wall normal.

        Research thresholds:
            <20 deg excellent, <45 deg good, <65 deg maximum (OpenFOAM).
        """
        if not layers:
            return []

        # Compute wall normals from first-layer cells
        wall_normals = _compute_wall_normals(vertices, mesh, layers[0])
        if not wall_normals:
            return []

        angles = []
        violations = []

        for layer_idx, layer_cells in enumerate(layers):
            for cell_info in layer_cells:
                cell_verts = _get_cell_vertices(vertices, mesh, cell_info)
                cell_normal = _compute_cell_wall_normal(
                    cell_verts, cell_info['element_type']
                )
                if cell_normal is None:
                    continue

                # Find nearest wall normal
                centroid = np.mean(cell_verts, axis=0)
                wall_n = _nearest_wall_normal(centroid, wall_normals)

                cos_angle = np.clip(np.dot(cell_normal, wall_n), -1.0, 1.0)
                angle_deg = float(np.degrees(np.arccos(cos_angle)))
                angles.append(angle_deg)

                if angle_deg > self.max_orthogonality_angle:
                    violations.append({
                        'global_idx': cell_info['global_idx'],
                        'layer': layer_idx,
                        'angle': angle_deg,
                    })

        if not angles or not violations:
            return []

        violation_rate = len(violations) / len(angles)
        severity = self._determine_severity(violation_rate)

        locations = [
            IssueLocation(
                element_type='cell',
                element_index=v['global_idx'],
            )
            for v in violations[:_MAX_LOCATIONS]
        ]

        return [MeshIssue(
            operation_id=self.metadata.operation_id,
            category=IssueCategory.QUALITY,
            severity=severity,
            message=(
                f'{len(violations)}/{len(angles)} BL cells '
                f'({violation_rate * 100:.1f}%) exceed max orthogonality '
                f'angle of {self.max_orthogonality_angle:.0f} deg '
                f'(max observed: {max(v["angle"] for v in violations):.1f} deg).'
            ),
            metrics={
                'violation_count': len(violations),
                'total_checked': len(angles),
                'violation_rate': violation_rate,
                'mean_angle': float(np.mean(angles)),
                'max_angle': float(np.max(angles)),
                'threshold_deg': self.max_orthogonality_angle,
            },
            metadata={'check': 'orthogonality'},
            locations=locations,
        )]

    def _check_coverage(
        self,
        seed_cells: List[dict],
        boundary_face_keys: List[Tuple[int, ...]],
        mesh: MeshInput,
        layers: List[List[dict]],
        face_to_cells: Dict[Tuple[int, ...], List[dict]],
    ) -> List[MeshIssue]:
        """Check how much of the wall surface is covered by BL cells.

        Wall candidates are boundary faces that lie neither on the traced
        BL cells themselves (their own caps and sides are the BL exterior,
        not wall) nor on transition cells — non-BL cells sharing a face
        with the traced stack (e.g. the tetrahedral core the BL grows
        into).  Uncovered candidates indicate the BL wraps only part of
        the wall.
        """
        if not seed_cells or not boundary_face_keys:
            return []

        # Collect wall boundary faces covered by seed cells
        covered_faces: Set[Tuple[int, ...]] = set()
        for cell_info in seed_cells:
            for fkey in cell_info.get('wall_face_keys', []):
                covered_faces.add(fkey)

        # Faces of the traced BL stack
        traced_gidx = {
            cell['global_idx'] for layer in layers for cell in layer
        }
        traced_faces: Set[Tuple[int, ...]] = set()
        transition_cells: Dict[int, str] = {}  # global_idx -> element_type
        for layer in layers:
            for cell_info in layer:
                for fkey in _get_cell_faces(mesh, cell_info):
                    traced_faces.add(fkey)
                    for adj in face_to_cells.get(fkey, []):
                        if adj['global_idx'] not in traced_gidx:
                            transition_cells[adj['global_idx']] = (
                                adj['element_type']
                            )

        # Faces belonging to transition (core) cells next to the BL
        excluded = set(traced_faces)
        for tc_idx, tc_type in transition_cells.items():
            excluded.update(
                _get_cell_faces(
                    mesh, {'global_idx': tc_idx, 'element_type': tc_type}
                )
            )

        wall_candidate_faces = [
            k for k in boundary_face_keys if k not in excluded
        ]

        if not wall_candidate_faces and not covered_faces:
            return []

        covered = len(covered_faces)
        total = covered + len(wall_candidate_faces)
        coverage = covered / max(total, 1)

        if coverage >= 1.0:
            return []

        severity = (
            IssueSeverity.WARNING if coverage >= 0.9
            else IssueSeverity.ERROR if coverage >= 0.5
            else IssueSeverity.CRITICAL
        )

        return [MeshIssue(
            operation_id=self.metadata.operation_id,
            category=IssueCategory.QUALITY,
            severity=severity,
            message=(
                f'Boundary layer covers {coverage * 100:.1f}% of wall '
                f'surface ({covered}/{total} faces). '
                f'Incomplete coverage may cause flow "leaks" at walls.'
            ),
            metrics={
                'covered_faces': covered,
                'total_wall_faces': total,
                'coverage_ratio': coverage,
            },
            metadata={'check': 'layer_coverage'},
        )]

    def _determine_severity(self, violation_rate: float) -> IssueSeverity:
        """Map violation rate to severity (standard thresholds)."""
        if violation_rate > 0.5:
            return IssueSeverity.CRITICAL
        elif violation_rate > 0.2:
            return IssueSeverity.ERROR
        elif violation_rate > 0.05:
            return IssueSeverity.WARNING
        else:
            return IssueSeverity.INFO


# ======================================================================
# Module-level helper functions
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


def _identify_wall_seed_cells(
    mesh: MeshInput,
    face_to_cells: Dict[Tuple[int, ...], List[dict]],
    boundary_set: Set[Tuple[int, ...]],
) -> List[dict]:
    """Identify BL cells whose *base face* is on the boundary.

    Only cells whose base face (triangular face for prisms, bottom/top face
    for hexes, base quad for pyramids) touches the boundary are seeded as
    layer 0.  This prevents all BL cells with any boundary side face from
    being lumped into layer 0.

    Returns list of ``{'global_idx', 'element_type', 'wall_face_keys'}``.
    """
    seed_cells: Dict[int, dict] = {}  # global_idx -> info

    vertices = mesh.vertices

    for bfkey in boundary_set:
        entries = face_to_cells.get(bfkey, [])
        for entry in entries:
            etype = entry['element_type']
            if etype not in BL_ELEMENT_TYPES:
                continue

            gidx = entry['global_idx']

            # Check if this boundary face is a base face of the cell
            if _is_base_face_of_cell(mesh, gidx, etype, bfkey):
                if not _passes_seed_gates(
                    mesh, vertices, face_to_cells, gidx, etype, bfkey
                ):
                    continue
                if gidx not in seed_cells:
                    seed_cells[gidx] = {
                        'global_idx': gidx,
                        'element_type': etype,
                        'wall_face_keys': [bfkey],
                    }
                else:
                    seed_cells[gidx]['wall_face_keys'].append(bfkey)

    return list(seed_cells.values())


def _passes_seed_gates(
    mesh: MeshInput,
    vertices: np.ndarray,
    face_to_cells: Dict[Tuple[int, ...], List[dict]],
    gidx: int,
    etype: str,
    seed_face_key: Tuple[int, ...],
) -> bool:
    """Apply the BL-typical calibration gates to a seed candidate.

    Gate 1 (thin cell): in-plane extent / height >= ``_MIN_SEED_THIN_RATIO``.
    Gate 2 (growth away from wall): when the cell opposite the seed face
    through the stack is itself a BL-type cell, it must not be >10% thinner
    than the seed (layers grow away from walls).  A missing neighbour
    (last layer against the core, or a single-cell shell) is accepted.
    """
    conn = _get_cell_connectivity(mesh, etype, gidx)
    if conn is None:
        return False
    cell_verts = vertices[conn]

    height = _estimate_cell_height(cell_verts, etype)
    inplane = _inplane_edge_length(cell_verts, etype)
    if height <= 0.0 or inplane <= 0.0:
        return False

    # Gate 1: thin cell
    if inplane / height < _MIN_SEED_THIN_RATIO:
        return False

    # Gate 2: growth away from the seeded face
    opposite_key = _opposite_base_face_key(mesh, conn, etype, seed_face_key)
    if opposite_key is not None:
        for adj in face_to_cells.get(opposite_key, []):
            if adj['global_idx'] == gidx:
                continue
            if adj['element_type'] not in BL_ELEMENT_TYPES:
                break  # transition into the core mesh — acceptable
            adj_conn = _get_cell_connectivity(
                mesh, adj['element_type'], adj['global_idx']
            )
            if adj_conn is None:
                break
            adj_height = _estimate_cell_height(
                vertices[adj_conn], adj['element_type']
            )
            if adj_height > 0 and adj_height < _GROWTH_AWAY_TOLERANCE * height:
                return False  # stack thins toward the seed face — wrong side
            break

    return True


def _opposite_base_face_key(
    mesh: MeshInput,
    conn: np.ndarray,
    element_type: str,
    seed_face_key: Tuple[int, ...],
) -> Optional[Tuple[int, ...]]:
    """Return the base face key on the opposite side of the cell."""
    base_indices = BASE_FACE_INDICES.get(element_type) or []
    face_defs = ELEMENT_FACES[element_type]
    for idx in base_indices:
        if idx >= len(face_defs):
            continue
        face_verts = tuple(int(conn[i]) for i in face_defs[idx])
        key = tuple(sorted(face_verts))
        if key != seed_face_key:
            return key
    return None


def _is_base_face_of_cell(
    mesh: MeshInput,
    global_idx: int,
    element_type: str,
    face_key: Tuple[int, ...],
) -> bool:
    """Return True if *face_key* corresponds to a base face of the cell."""
    base_indices = BASE_FACE_INDICES.get(element_type)
    if not base_indices:
        return False

    conn = _get_cell_connectivity(mesh, element_type, global_idx)
    if conn is None:
        return False

    face_defs = ELEMENT_FACES[element_type]
    for idx in base_indices:
        if idx >= len(face_defs):
            continue
        face_def = face_defs[idx]
        face_verts = tuple(int(conn[i]) for i in face_def)
        if tuple(sorted(face_verts)) == face_key:
            return True

    return False


def _trace_boundary_layers(
    mesh: MeshInput,
    seed_cells: List[dict],
    face_to_cells: Dict[Tuple[int, ...], List[dict]],
    boundary_set: Set[Tuple[int, ...]],
    max_layers: int = 100,
) -> List[List[dict]]:
    """Trace boundary layer cells from wall outward.

    Algorithm:
        1. Layer 0 = seed cells (BL-type cells with base face on boundary).
        2. For each cell in current layer, find neighbors through non-boundary
           faces that are also BL-type cells.
        3. Stop when no new BL-type neighbors are found.

    Returns list of layers, each layer a list of cell descriptors.
    """
    visited: Set[int] = set()

    # Layer 0: seed cells
    layer0 = []
    for cell_info in seed_cells:
        visited.add(cell_info['global_idx'])
        layer0.append(cell_info)
    layers = [layer0]

    # Build global_idx -> element_type quick lookup
    element_type_map: Dict[int, str] = {}
    for entry_list in face_to_cells.values():
        for entry in entry_list:
            element_type_map[entry['global_idx']] = entry['element_type']

    for _ in range(max_layers - 1):
        current_layer = layers[-1]
        next_layer = []

        for cell_info in current_layer:
            # Find all faces of this cell
            cell_faces = _get_cell_faces(mesh, cell_info)

            for fkey in cell_faces:
                if fkey in boundary_set:
                    continue  # skip boundary faces

                adjacent_entries = face_to_cells.get(fkey, [])
                for adj in adjacent_entries:
                    adj_idx = adj['global_idx']
                    if adj_idx == cell_info['global_idx']:
                        continue
                    if adj_idx in visited:
                        continue
                    # Only continue tracing through BL-type elements
                    adj_type = element_type_map.get(adj_idx, '')
                    if adj_type not in BL_ELEMENT_TYPES:
                        continue

                    visited.add(adj_idx)
                    next_layer.append({
                        'global_idx': adj_idx,
                        'element_type': adj_type,
                        'wall_face_keys': [],
                    })

        if not next_layer:
            break
        layers.append(next_layer)

    return layers


def _get_cell_faces(mesh: MeshInput, cell_info: dict) -> List[Tuple[int, ...]]:
    """Return sorted face keys for a cell."""
    etype = cell_info['element_type']
    gidx = cell_info['global_idx']

    connectivity = _get_cell_connectivity(mesh, etype, gidx)
    if connectivity is None:
        return []

    face_defs = ELEMENT_FACES.get(etype, [])
    faces = []
    for face_def in face_defs:
        face_verts = tuple(int(connectivity[i]) for i in face_def)
        faces.append(tuple(sorted(face_verts)))
    return faces


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


def _get_cell_vertices(
    vertices: np.ndarray, mesh: MeshInput, cell_info: dict
) -> np.ndarray:
    """Return Nx3 array of vertex coordinates for a cell."""
    conn = _get_cell_connectivity(
        mesh, cell_info['element_type'], cell_info['global_idx']
    )
    if conn is None:
        return np.zeros((0, 3))
    return vertices[conn]


def _compute_layer_heights(
    vertices: np.ndarray,
    mesh: MeshInput,
    layers: List[List[dict]],
) -> List[float]:
    """Compute average height for each layer.

    For layer 0, height is estimated from the cell geometry (distance between
    base and top faces).  For subsequent layers, height is the distance between
    consecutive layer centroids.
    """
    if not layers:
        return []

    heights = []

    for i, layer_cells in enumerate(layers):
        if i == 0:
            # Estimate first layer height from the cells themselves
            cell_heights = []
            for cell_info in layer_cells:
                cell_verts = _get_cell_vertices(vertices, mesh, cell_info)
                if len(cell_verts) > 0:
                    h = _estimate_cell_height(
                        cell_verts, cell_info['element_type']
                    )
                    if h > 0:
                        cell_heights.append(h)
            heights.append(
                float(np.mean(cell_heights)) if cell_heights else 0.0
            )
        else:
            # Distance between consecutive layer centroids
            prev_centroids = []
            curr_centroids = []
            for cell_info in layers[i - 1]:
                v = _get_cell_vertices(vertices, mesh, cell_info)
                if len(v) > 0:
                    prev_centroids.append(np.mean(v, axis=0))
            for cell_info in layer_cells:
                v = _get_cell_vertices(vertices, mesh, cell_info)
                if len(v) > 0:
                    curr_centroids.append(np.mean(v, axis=0))

            if prev_centroids and curr_centroids:
                prev_c = np.mean(prev_centroids, axis=0)
                curr_c = np.mean(curr_centroids, axis=0)
                heights.append(float(np.linalg.norm(curr_c - prev_c)))
            else:
                heights.append(0.0)

    return heights


def _estimate_cell_height(
    cell_verts: np.ndarray, element_type: str
) -> float:
    """Estimate the wall-normal height of a BL cell.

    For prism6: distance between the two triangular faces.
    For hex8: distance between the two largest parallel faces.
    """
    if element_type == 'prism6' and len(cell_verts) == 6:
        bottom_c = np.mean(cell_verts[:3], axis=0)
        top_c = np.mean(cell_verts[3:], axis=0)
        return float(np.linalg.norm(top_c - bottom_c))
    elif element_type == 'hex8' and len(cell_verts) == 8:
        bottom_c = np.mean(cell_verts[:4], axis=0)
        top_c = np.mean(cell_verts[4:], axis=0)
        return float(np.linalg.norm(top_c - bottom_c))
    else:
        # Fallback: max distance between any two vertices
        from scipy.spatial.distance import pdist
        if len(cell_verts) < 2:
            return 0.0
        return float(np.max(pdist(cell_verts)))


def _inplane_edge_length(
    cell_verts: np.ndarray, element_type: str
) -> float:
    """Estimate the mean in-plane (wall-parallel) edge length of a BL cell.

    For prism6: mean bottom-triangle edge length (verts 0-2).
    For hex8: mean bottom-quad edge length (verts 0-3).
    Other types: median pairwise vertex distance (fallback).
    """
    if element_type == 'prism6' and len(cell_verts) == 6:
        bottom = cell_verts[:3]
        edges = [
            np.linalg.norm(bottom[1] - bottom[0]),
            np.linalg.norm(bottom[2] - bottom[1]),
            np.linalg.norm(bottom[0] - bottom[2]),
        ]
        return float(np.mean(edges))
    elif element_type == 'hex8' and len(cell_verts) == 8:
        bottom = cell_verts[:4]
        edges = [
            np.linalg.norm(bottom[1] - bottom[0]),
            np.linalg.norm(bottom[2] - bottom[1]),
            np.linalg.norm(bottom[3] - bottom[2]),
            np.linalg.norm(bottom[0] - bottom[3]),
        ]
        return float(np.mean(edges))
    else:
        from scipy.spatial.distance import pdist
        if len(cell_verts) < 2:
            return 0.0
        return float(np.median(pdist(cell_verts)))


def _compute_cell_aspect_ratio(
    cell_verts: np.ndarray, element_type: str
) -> float:
    """Compute aspect ratio = height / in-plane characteristic length."""
    if len(cell_verts) < 3:
        return 0.0

    height = _estimate_cell_height(cell_verts, element_type)
    if height <= 0:
        return 0.0

    inplane = _inplane_edge_length(cell_verts, element_type)

    return height / inplane if inplane > 0 else 0.0


def _compute_wall_normals(
    vertices: np.ndarray,
    mesh: MeshInput,
    first_layer_cells: List[dict],
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Compute wall normal directions from first-layer cells.

    Returns list of (centroid, normal) tuples for each BL cell.
    """
    results = []
    for cell_info in first_layer_cells:
        cell_verts = _get_cell_vertices(vertices, mesh, cell_info)
        normal = _compute_cell_wall_normal(
            cell_verts, cell_info['element_type']
        )
        if normal is not None:
            centroid = np.mean(cell_verts, axis=0)
            results.append((centroid, normal))
    return results


def _compute_cell_wall_normal(
    cell_verts: np.ndarray, element_type: str
) -> Optional[np.ndarray]:
    """Compute the wall-normal direction of a BL cell.

    For prism6: normal to the bottom triangular face (verts 0,1,2).
    For hex8: normal to the bottom quad face (verts 0,1,2,3).
    """
    if element_type == 'prism6' and len(cell_verts) == 6:
        v = cell_verts[:3]
        edge1 = v[1] - v[0]
        edge2 = v[2] - v[0]
        n = np.cross(edge1, edge2)
        norm = np.linalg.norm(n)
        if norm > 1e-15:
            return n / norm
    elif element_type == 'hex8' and len(cell_verts) == 8:
        v = cell_verts[:4]
        edge1 = v[1] - v[0]
        edge2 = v[3] - v[0]
        n = np.cross(edge1, edge2)
        norm = np.linalg.norm(n)
        if norm > 1e-15:
            return n / norm
    return None


def _nearest_wall_normal(
    centroid: np.ndarray,
    wall_normals: List[Tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    """Find the wall normal closest to the given centroid."""
    best_n = wall_normals[0][1]
    best_d = float('inf')
    for w_centroid, w_normal in wall_normals:
        d = np.linalg.norm(centroid - w_centroid)
        if d < best_d:
            best_d = d
            best_n = w_normal
    return best_n


# Register operation
registry.register(BoundaryLayerQualityCheck())
