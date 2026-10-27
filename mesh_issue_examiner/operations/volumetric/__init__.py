"""
Volumetric (3D) mesh inspection operations.
"""

from .nonmanifold_faces import NonManifoldFaceDetection
from .orientation_consistency import OrientationConsistencyCheck
from .duplicate_cells import DuplicateCellDetection
from .orphaned_cells import OrphanedCellDetection
from .boundary_closure import BoundaryClosureCheck
from .volume_cell_quality import (
    VolumeCellQualityComputation,
    METRIC_ELEMENT_SUPPORT,
    DEFAULT_THRESHOLDS,
    EXTENDED_QUALITY_TYPES,
)
from .geometric_validity import GeometricValidityAssessment
from .boundary_layer_quality import BoundaryLayerQualityCheck
from .multi_material import (
    MaterialInterfaceDetection,
    InterfaceQualityCheck,
    PerMaterialStatistics,
)

__all__ = [
    "NonManifoldFaceDetection",
    "OrientationConsistencyCheck",
    "DuplicateCellDetection",
    "OrphanedCellDetection",
    "BoundaryClosureCheck",
    "VolumeCellQualityComputation",
    "GeometricValidityAssessment",
    "BoundaryLayerQualityCheck",
    "MaterialInterfaceDetection",
    "InterfaceQualityCheck",
    "PerMaterialStatistics",
    "METRIC_ELEMENT_SUPPORT",
    "DEFAULT_THRESHOLDS",
    "EXTENDED_QUALITY_TYPES",
]
