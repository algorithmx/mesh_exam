"""
Geometric validity assessment operation for volumetric (3D) meshes.

Detects degenerate cells (near-zero volume) and sliver elements
(poor volume-to-edge-length ratio) that indicate geometric defects
in volumetric meshes.
"""

import logging
from typing import List, Optional, Dict

import numpy as np

logger = logging.getLogger(__name__)

from ...core.models import (
    MeshInput,
    MeshIssue,
    IssueCategory,
    IssueSeverity,
    IssueLocation,
    OperationConfig,
)
from ..base import BaseOperation, OperationMetadata
from ..operation_registry import registry
from ...adapters.factory import get_adapter
from ...adapters.exceptions import MeshConversionError
from ._face_tables import linearized_volume_cells


class GeometricValidityAssessment(BaseOperation):
    """Detects geometric validity defects in volumetric mesh cells.

    Checks for two categories of geometric defects:

    1. **Degenerate cells**: Cells with near-zero absolute volume
       (|volume| < tolerance), indicating collapsed or inverted elements.

    2. **Sliver elements**: Cells whose volume is above the degenerate
       threshold but whose volume / max_edge_length^3 ratio is very small,
       indicating flat, needle-like, or otherwise poorly shaped elements.
    """

    def __init__(
        self,
        config: Optional[OperationConfig] = None,
        degenerate_tolerance: Optional[float] = None,
        sliver_threshold: Optional[float] = None,
    ):
        super().__init__(config)
        self.degenerate_tolerance = degenerate_tolerance or 1e-12
        self.sliver_threshold = sliver_threshold or 1e-6  # volume / max_edge^3 ratio
        # NOTE: the pyvista adapter is resolved lazily in execute(), not
        # here -- this __init__ runs at import time via module-level
        # registration, and resolving it eagerly would make the whole
        # package unimportable on machines without pyvista.

    @property
    def metadata(self) -> OperationMetadata:
        return OperationMetadata(
            operation_id='quality_3d.assess_geometric_validity',
            display_name='Geometric Validity Assessment (3D)',
            description='Detects degenerate and sliver volumetric elements.',
            category=IssueCategory.GEOMETRIC,
            supported_libraries=['pyvista'],
            recommended_library='pyvista',
            requires_adjacency=False,
            complexity='O(n)',
            tags=['quality', 'volumetric', '3d', 'degenerate', 'sliver', 'validity'],
        )

    def execute(self, mesh: MeshInput) -> List[MeshIssue]:
        """Execute geometric validity assessment on a volumetric mesh.

        Args:
            mesh: Input mesh with volumetric cell data.

        Returns:
            List of MeshIssue objects for degenerate and sliver defects.
        """
        issues: List[MeshIssue] = []

        # 1. Check that the mesh has volumetric cells
        if not mesh.has_volume_cells:
            return issues

        # 1b. Resolve the pyvista adapter lazily (import-safe; raises only
        # when the operation actually runs on a volumetric mesh).
        adapter = get_adapter('pyvista')

        # 2. Corner-linearised cell table: quadratic element types keep
        #    their corner connectivity (VTK node ordering places corners
        #    first), re-keyed by linear base type.  vtkCellQuality cannot
        #    evaluate quadratic cells (it returns -1 for them), so volumes
        #    are computed on the linearised grid whenever quadratic types
        #    are present.  This also unblocks pyramid14 (no VTK cell),
        #    whose pyramid5 corners carry the geometry.
        linearized_cells = linearized_volume_cells(mesh)

        grid = None
        try:
            grid = mesh.get_native('pyvista')
        except MeshConversionError as exc:
            if linearized_cells is None:
                # No VTK equivalent and no corner fallback: the check is
                # skipped entirely.
                logger.warning(
                    'Skipping geometric validity check (no VTK equivalent '
                    'for some element types): %s', exc,
                )
                return issues

        linearized = False
        if linearized_cells is not None:
            try:
                lin_mesh = MeshInput(
                    vertices=mesh.vertices,
                    faces=mesh.faces,
                    mesh_id=f'{mesh.mesh_id}_corner_linearized',
                    volume_cells=linearized_cells,
                )
                grid = lin_mesh.get_native('pyvista')
                linearized = True
            except MeshConversionError as exc:
                if grid is None:
                    logger.warning(
                        'Skipping geometric validity check (native and '
                        'corner-linearised conversion both failed): %s', exc,
                    )
                    return issues
                # Native grid remains usable; keep the original mesh.

        # 3. Compute cell volumes via the adapter
        volumes = adapter.compute_cell_quality(grid, 'volume')
        if len(volumes) == 0:
            return issues

        # 4. Find degenerate cells: abs(volume) < tolerance -> CRITICAL
        degenerate_mask = np.abs(volumes) < self.degenerate_tolerance
        degenerate_indices = np.where(degenerate_mask)[0]

        if len(degenerate_indices) > 0:
            capped_degenerate = degenerate_indices[:100]
            locations = [
                IssueLocation(element_type='cell', element_index=int(idx))
                for idx in capped_degenerate
            ]
            issues.append(MeshIssue(
                operation_id=self.metadata.operation_id,
                category=IssueCategory.GEOMETRIC,
                severity=IssueSeverity.CRITICAL,
                message=(
                    f"Found {len(degenerate_indices)} degenerate volumetric cell(s) "
                    f"with |volume| < {self.degenerate_tolerance:.2e}"
                ),
                locations=locations,
                metrics={
                    'degenerate_count': float(len(degenerate_indices)),
                    'degenerate_tolerance': self.degenerate_tolerance,
                    'total_cells': float(len(volumes)),
                },
            ))

        # 5. Find sliver cells: volume > degenerate_tol but
        #    volume / (max_edge_length^3) < sliver_threshold
        #
        #    Detection runs over the merged (linearised) cell table: when
        #    quadratic types are present, quadratic rows enter their base
        #    type's block with corner-only connectivity.  The merged
        #    enumeration preserves the mesh's global cell numbering, so
        #    detector offsets and reported indices line up with the
        #    original mesh, and each index can be attributed back to its
        #    original element type.
        vertices = mesh.vertices
        detector_cells = linearized_cells if linearized else mesh.volume_cells
        sliver_results: Dict[str, List[int]] = {}

        # Original element type per global cell index (insertion order).
        cell_types: List[str] = []
        for etype, conn in mesh.volume_cells.items():
            cell_types.extend([etype] * len(conn))

        # Map base element types to their detector methods
        detector_map = {
            'tet4': self._detect_sliver_tets,
            'hex8': self._detect_sliver_hexes,
            'prism6': self._detect_sliver_prisms,
            'pyramid5': self._detect_sliver_pyramids,
        }

        for element_type, detector in detector_map.items():
            conn = detector_cells.get(element_type, np.array([]))
            if len(conn) == 0:
                continue

            offset = self._compute_element_offset(detector_cells, element_type)
            sliver_indices = detector(conn, vertices, volumes, degenerate_mask, offset)

            for idx in sliver_indices:
                orig_type = (
                    cell_types[int(idx)] if int(idx) < len(cell_types)
                    else element_type
                )
                sliver_results.setdefault(orig_type, []).append(int(idx))

        if sliver_results:
            all_sliver_indices = sorted(
                idx for indices in sliver_results.values() for idx in indices
            )
            capped_sliver = all_sliver_indices[:100]
            locations = [
                IssueLocation(element_type='cell', element_index=int(idx))
                for idx in capped_sliver
            ]

            total_slivers = len(all_sliver_indices)
            metrics: Dict[str, float] = {
                'sliver_count': float(total_slivers),
                'sliver_threshold': self.sliver_threshold,
                'total_cells': float(len(volumes)),
            }
            for element_type, indices in sorted(sliver_results.items()):
                metrics[f'sliver_{element_type}_count'] = float(len(indices))

            issue = MeshIssue(
                operation_id=self.metadata.operation_id,
                category=IssueCategory.GEOMETRIC,
                severity=IssueSeverity.WARNING,
                message=(
                    f"Found {total_slivers} sliver volumetric cell(s) "
                    f"with volume/max_edge^3 ratio < {self.sliver_threshold:.2e}"
                ),
                locations=locations,
                metrics=metrics,
            )
            if linearized:
                # Slivers were judged on corner-only geometry; midside
                # curvature is ignored (same provenance tag as the
                # linearised quality metrics).
                issue.metadata['linearized'] = True
            issues.append(issue)

        return issues

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_element_offset(volume_cells: Dict[str, np.ndarray],
                                element_type: str) -> int:
        """Return the global cell index where *element_type* cells begin.

        Works on the (possibly linearised/merged) cell table the detectors
        run over: it iterates the dict in insertion order, summing cell
        counts for every element type that appears before *element_type*.
        """
        offset = 0
        for etype, conn in volume_cells.items():
            if etype == element_type:
                break
            offset += len(conn)
        return offset

    def _detect_sliver_tets(
        self,
        tet4_conn: np.ndarray,
        vertices: np.ndarray,
        volumes: np.ndarray,
        degenerate_mask: np.ndarray,
        tet_offset: int,
    ) -> np.ndarray:
        """Detect sliver tetrahedra based on volume-to-edge-length ratio.

        For each tetrahedron with vertices *a, b, c, d* the six edge
        lengths are:

            |a-b|, |a-c|, |a-d|, |b-c|, |b-d|, |c-d|

        A tet is classified as a sliver when its volume is above the
        degenerate threshold but the ratio
        ``volume / max_edge_length**3`` is below *sliver_threshold*.

        Args:
            tet4_conn: (N, 4) integer array of tet vertex indices.
            vertices: (V, 3) float array of vertex coordinates.
            volumes: (C,) float array of cell volumes (one per cell).
            degenerate_mask: Boolean mask over *volumes* marking degenerate cells.
            tet_offset: Global cell-index offset for tet4 cells.

        Returns:
            Array of integer indices (into the full cell list) of sliver tets.
        """
        n_tets = len(tet4_conn)

        # Slice volumes and degenerate mask to the tet4 portion
        tet_volumes = volumes[tet_offset:tet_offset + n_tets]
        tet_degenerate = degenerate_mask[tet_offset:tet_offset + n_tets]

        # Gather vertex coordinates for all tets: shape (n_tets, 4, 3)
        tet_verts = vertices[tet4_conn]  # [a, b, c, d] per tet

        # Compute the 6 edge vectors for each tet
        ab = tet_verts[:, 1] - tet_verts[:, 0]
        ac = tet_verts[:, 2] - tet_verts[:, 0]
        ad = tet_verts[:, 3] - tet_verts[:, 0]
        bc = tet_verts[:, 2] - tet_verts[:, 1]
        bd = tet_verts[:, 3] - tet_verts[:, 1]
        cd = tet_verts[:, 3] - tet_verts[:, 2]

        # Compute edge lengths: shape (n_tets, 6)
        edge_lengths = np.stack([
            np.linalg.norm(ab, axis=1),
            np.linalg.norm(ac, axis=1),
            np.linalg.norm(ad, axis=1),
            np.linalg.norm(bc, axis=1),
            np.linalg.norm(bd, axis=1),
            np.linalg.norm(cd, axis=1),
        ], axis=1)

        max_edge = np.max(edge_lengths, axis=1)  # (n_tets,)

        # Avoid division by zero for max_edge == 0 (those are degenerate anyway)
        zero_edge_mask = max_edge == 0
        if np.any(zero_edge_mask):
            logger.debug(
                f"Found {np.sum(zero_edge_mask)} tetrahedra with zero-length edges"
            )
        safe_max_edge = np.where(max_edge > 0, max_edge, 1.0)

        sliver_ratio = np.abs(tet_volumes) / (safe_max_edge ** 3)

        sliver_mask = (
            (~tet_degenerate) &
            (sliver_ratio < self.sliver_threshold)
        )

        # Map back to global cell indices
        sliver_global_indices = np.where(sliver_mask)[0] + tet_offset

        return sliver_global_indices

    def _detect_sliver_hexes(
        self,
        hex8_conn: np.ndarray,
        vertices: np.ndarray,
        volumes: np.ndarray,
        degenerate_mask: np.ndarray,
        hex_offset: int,
    ) -> np.ndarray:
        """Detect sliver hexahedra based on volume-to-edge-length ratio.

        A hex is classified as a sliver when its volume is above the
        degenerate threshold but the ratio
        ``volume / max_edge_length**3`` is below *sliver_threshold*.

        Args:
            hex8_conn: (N, 8) integer array of hex vertex indices.
            vertices: (V, 3) float array of vertex coordinates.
            volumes: (C,) float array of cell volumes (one per cell).
            degenerate_mask: Boolean mask over *volumes* marking degenerate cells.
            hex_offset: Global cell-index offset for hex8 cells.

        Returns:
            Array of integer indices (into the full cell list) of sliver hexes.
        """
        n_hexes = len(hex8_conn)

        hex_volumes = volumes[hex_offset:hex_offset + n_hexes]
        hex_degenerate = degenerate_mask[hex_offset:hex_offset + n_hexes]

        hex_verts = vertices[hex8_conn]

        # 12 edges of a hexahedron
        edge_pairs = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # bottom face
            (4, 5), (5, 6), (6, 7), (7, 4),  # top face
            (0, 4), (1, 5), (2, 6), (3, 7),  # vertical edges
        ]

        edge_lengths = np.stack([
            np.linalg.norm(hex_verts[:, j] - hex_verts[:, i], axis=1)
            for i, j in edge_pairs
        ], axis=1)

        max_edge = np.max(edge_lengths, axis=1)
        safe_max_edge = np.where(max_edge > 0, max_edge, 1.0)

        sliver_ratio = np.abs(hex_volumes) / (safe_max_edge ** 3)

        sliver_mask = (
            (~hex_degenerate) &
            (np.abs(hex_volumes) > self.degenerate_tolerance) &
            (sliver_ratio < self.sliver_threshold)
        )

        return np.where(sliver_mask)[0] + hex_offset

    def _detect_sliver_prisms(
        self,
        prism6_conn: np.ndarray,
        vertices: np.ndarray,
        volumes: np.ndarray,
        degenerate_mask: np.ndarray,
        prism_offset: int,
    ) -> np.ndarray:
        """Detect sliver triangular prisms based on volume-to-edge-length ratio.

        Args:
            prism6_conn: (N, 6) integer array of prism vertex indices.
            vertices: (V, 3) float array of vertex coordinates.
            volumes: (C,) float array of cell volumes (one per cell).
            degenerate_mask: Boolean mask over *volumes* marking degenerate cells.
            prism_offset: Global cell-index offset for prism6 cells.

        Returns:
            Array of integer indices (into the full cell list) of sliver prisms.
        """
        n_prisms = len(prism6_conn)

        prism_volumes = volumes[prism_offset:prism_offset + n_prisms]
        prism_degenerate = degenerate_mask[prism_offset:prism_offset + n_prisms]

        prism_verts = vertices[prism6_conn]

        # 9 edges of a triangular prism
        edge_pairs = [
            (0, 1), (1, 2), (2, 0),  # bottom triangle
            (3, 4), (4, 5), (5, 3),  # top triangle
            (0, 3), (1, 4), (2, 5),  # vertical edges
        ]

        edge_lengths = np.stack([
            np.linalg.norm(prism_verts[:, j] - prism_verts[:, i], axis=1)
            for i, j in edge_pairs
        ], axis=1)

        max_edge = np.max(edge_lengths, axis=1)
        safe_max_edge = np.where(max_edge > 0, max_edge, 1.0)

        sliver_ratio = np.abs(prism_volumes) / (safe_max_edge ** 3)

        sliver_mask = (
            (~prism_degenerate) &
            (np.abs(prism_volumes) > self.degenerate_tolerance) &
            (sliver_ratio < self.sliver_threshold)
        )

        return np.where(sliver_mask)[0] + prism_offset

    def _detect_sliver_pyramids(
        self,
        pyramid5_conn: np.ndarray,
        vertices: np.ndarray,
        volumes: np.ndarray,
        degenerate_mask: np.ndarray,
        pyramid_offset: int,
    ) -> np.ndarray:
        """Detect sliver pyramids based on volume-to-edge-length ratio.

        Args:
            pyramid5_conn: (N, 5) integer array of pyramid vertex indices.
            vertices: (V, 3) float array of vertex coordinates.
            volumes: (C,) float array of cell volumes (one per cell).
            degenerate_mask: Boolean mask over *volumes* marking degenerate cells.
            pyramid_offset: Global cell-index offset for pyramid5 cells.

        Returns:
            Array of integer indices (into the full cell list) of sliver pyramids.
        """
        n_pyramids = len(pyramid5_conn)

        pyramid_volumes = volumes[pyramid_offset:pyramid_offset + n_pyramids]
        pyramid_degenerate = degenerate_mask[pyramid_offset:pyramid_offset + n_pyramids]

        pyramid_verts = vertices[pyramid5_conn]

        # 8 edges of a pyramid
        edge_pairs = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # base quad
            (0, 4), (1, 4), (2, 4), (3, 4),  # apex edges
        ]

        edge_lengths = np.stack([
            np.linalg.norm(pyramid_verts[:, j] - pyramid_verts[:, i], axis=1)
            for i, j in edge_pairs
        ], axis=1)

        max_edge = np.max(edge_lengths, axis=1)
        safe_max_edge = np.where(max_edge > 0, max_edge, 1.0)

        sliver_ratio = np.abs(pyramid_volumes) / (safe_max_edge ** 3)

        sliver_mask = (
            (~pyramid_degenerate) &
            (np.abs(pyramid_volumes) > self.degenerate_tolerance) &
            (sliver_ratio < self.sliver_threshold)
        )

        return np.where(sliver_mask)[0] + pyramid_offset


# Register operation
registry.register(GeometricValidityAssessment())
