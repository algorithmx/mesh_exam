# Mesh Issue Examiner

[![Version](https://img.shields.io/badge/version-0.1.0--alpha-orange.svg)](https://github.com/matrix/mesh-issue-examiner)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-alpha-yellow.svg)]()

A Python framework for **automated 3D mesh quality assessment and defect detection**. Load triangle meshes from common CAD and graphics formats, run composable inspection operations, and get structured reports about topological, geometric, and quality issues. Designed as an MCP (Model Context Protocol) server so AI agents can inspect meshes directly.

---

## Table of Contents

- [Why Mesh Issue Examiner?](#why-mesh-issue-examiner)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Core Models](#core-models)
- [Operations and the Registry](#operations-and-the-registry)
- [Adapters](#adapters)
- [MCP Server](#mcp-server)
- [Utility Functions](#utility-functions)
- [Configuration](#configuration)
- [Supported File Formats](#supported-file-formats)
- [Library Comparison](#library-comparison)
- [Development Guide](#development-guide)
- [Contributing](#contributing)
- [License](#license)

---

## Why Mesh Issue Examiner?

3D meshes used in simulation, 3D printing, and game engines often contain subtle defects -- non-manifold edges, degenerate faces, poor element quality -- that cause silent failures downstream. Existing tools are either commercial, tied to a single library, or lack a programmatic API. Mesh Issue Examiner provides:

- A **unified, library-agnostic** inspection pipeline that works across trimesh, PyVista, and PyMeshLab.
- **Composable operations** that can be combined into custom workflows.
- **Structured output** (severity levels, locations, metrics) suitable for automated CI/CD pipelines.
- An **MCP server** interface so LLM-based agents can load and inspect meshes through tool calls.

---

## Features

| Category | Description |
|----------|-------------|
| **Multi-format I/O** | STL, OBJ, PLY, OFF, GLTF, GLB, VTP, VTK, 3MF |
| **Topology checks** | Non-manifold edge detection, manifold validation |
| **Quality metrics** | Aspect ratio, skewness, scaled Jacobian, min/max angles, condition number |
| **Library abstraction** | Common adapter interface for trimesh, PyVista (VTK), PyMeshLab |
| **Composable workflows** | Combine individual operations into full or category-scoped pipelines |
| **Severity classification** | INFO, WARNING, ERROR, CRITICAL with automatic determination |
| **MCP integration** | Built-in server with 5 tools for AI agent interoperability |
| **Extensible** | Add new operations or adapters with minimal boilerplate |

---

## Installation

### Basic install (numpy only)

```bash
pip install mesh-issue-examiner
```

### With specific libraries

```bash
pip install mesh-issue-examiner[trimesh]
pip install mesh-issue-examiner[pyvista]
pip install mesh-issue-examiner[pymeshlab]
```

### With all supported libraries

```bash
pip install mesh-issue-examiner[all]
```

### From source

```bash
git clone https://github.com/matrix/mesh-issue-examiner.git
cd mesh-issue-examiner
pip install -e ".[dev]"
```

### Development dependencies

```bash
pip install mesh-issue-examiner[dev]
```

This pulls in `pytest`, `pytest-asyncio`, `black`, and `flake8`.

---

## Quick Start

### Load a mesh and run all inspections

```python
from mesh_issue_examiner import MeshInput, registry
from mesh_issue_examiner.utils import load_mesh_from_file

# Load a mesh from file (auto-selects the best available library)
mesh = load_mesh_from_file("example.stl")

# Run every registered operation
workflow = registry.create_full_workflow()
result = workflow(mesh)

# Inspect the result
print(f"Total issues:  {result.summary['total_issues']}")
print(f"Critical:      {result.summary['critical_count']}")
print(f"Errors:        {result.summary['error_count']}")
print(f"Warnings:      {result.summary['warning_count']}")
print(f"Passed:        {result.passed}")
```

### Run specific operations

```python
from mesh_issue_examiner import registry

# Run only topological checks
workflow = registry.create_category_workflow(
    registry.get("topology.detect_nonmanifold_edges").metadata.category
)
result = workflow(mesh)

# Run a hand-picked set of operations
workflow = registry.create_workflow([
    "topology.detect_nonmanifold_edges",
    "quality.compute_cell_quality",
])
result = workflow(mesh)
```

### Create a mesh from raw arrays

```python
import numpy as np
from mesh_issue_examiner import MeshInput
from mesh_issue_examiner.utils import create_mesh_from_arrays

vertices = np.array([
    [0, 0, 0],
    [1, 0, 0],
    [0.5, 0.866, 0],
])
faces = np.array([[0, 1, 2]])

mesh = create_mesh_from_arrays(vertices, faces, mesh_id="my_tri")
```

### Use the MCP server programmatically

```python
import asyncio
from mcp.server import MeshIssueExaminerServer, MCPRequest

async def main():
    server = MeshIssueExaminerServer()

    # Load
    load_resp = await server.handle_request(MCPRequest(
        method="mesh_load",
        params={"file_path": "example.stl"},
    ))
    mesh_id = load_resp.result["mesh_id"]

    # Inspect
    inspect_resp = await server.handle_request(MCPRequest(
        method="mesh_inspect",
        params={"mesh_id": mesh_id, "operations": ["all"]},
    ))
    print(inspect_resp.result["summary"])

asyncio.run(main())
```

---

## Architecture

```
mesh_issue_examiner/
|
|-- core/                          # Data models and exceptions
|   |-- models.py                  #   MeshInput, MeshIssue, QualityMetrics,
|   |                              #   InspectionResult, OperationConfig,
|   |                              #   IssueSeverity, IssueCategory, Vertex,
|   |                              #   Face, Edge, IssueLocation
|   |-- exceptions.py              #   MeshExaminerError hierarchy
|
|-- adapters/                      # Library abstraction layer
|   |-- base_adapter.py            #   MeshAdapter ABC
|   |-- trimesh_adapter.py         #   TrimeshAdapter
|   |-- pyvista_adapter.py         #   PyVistaAdapter
|   |-- pymeshlab_adapter.py       #   PyMeshLabAdapter
|
|-- operations/                    # Inspection operations
|   |-- base.py                    #   BaseOperation, QualityOperation,
|   |                              #   OperationMetadata
|   |-- operation_registry.py      #   OperationRegistry (singleton)
|   |-- topological/
|   |   |-- nonmanifold_edges.py   #   NonManifoldEdgeDetection
|   |-- quality/
|   |   |-- cell_quality.py        #   CellQualityComputation
|   |-- geometric/                 #   (planned, not yet implemented)
|
|-- mcp/                           # MCP server
|   |-- server.py                  #   MeshIssueExaminerServer
|   |-- handlers.py                #   RequestHandler, AsyncRequestHandler
|   |-- tools.py                   #   ToolDefinition, schema helpers
|
|-- utils/                         # Utilities
|   |-- mesh_loader.py             #   load/save/create mesh functions
|   |-- metrics.py                 #   Thresholds, violation stats, severity
|
|-- tests/                         # Unit tests
    |-- test_operations.py
```

**Data flow:**

```
                     +-----------+
    Mesh file  ----> | mesh_loader| ----> MeshInput (canonical)
                     +-----------+                |
                                                    |  get_native("pyvista")
                      +-----------+                v
                      |  Adapter  | <---- pyvista.PolyData / trimesh.Trimesh
                      +-----------+
                            |
                            v
                      +------------+
                      | Operation  | ----> List[MeshIssue]
                      +------------+
                            |
                            v
                      +-----------------+
                      | InspectionResult| (summary, issues, metrics)
                      +-----------------+
```

---

## Core Models

All models live in `core.models` and are re-exported from the top-level package.

### MeshInput

The canonical mesh representation. Wraps vertices and faces as numpy arrays and lazily constructs library-specific mesh objects on demand.

```python
from mesh_issue_examiner import MeshInput
import numpy as np

mesh = MeshInput(
    vertices=np.array([[0,0,0], [1,0,0], [0,1,0]]),
    faces=np.array([[0, 1, 2]]),
    mesh_id="my_mesh",
    source_format="arrays",
    file_path=None,
)

# Properties
mesh.n_vertices          # 3
mesh.n_faces             # 1
mesh.bounding_box        # (array([0,0,0]), array([1,1,0]))
mesh.center              # array([0.5, 0.5, 0.0])
mesh.extent              # array([1.0, 1.0, 0.0])

# Get a library-specific mesh (lazy construction, cached)
pv_mesh = mesh.get_native("pyvista")   # pyvista.PolyData
tm_mesh = mesh.get_native("trimesh")   # trimesh.Trimesh

# Serialization
mesh.to_dict()           # dict with mesh_id, n_vertices, n_faces, bounding_box
mesh.copy()              # deep copy with "_copy" suffix on mesh_id
```

### IssueSeverity

```python
from mesh_issue_examiner import IssueSeverity

IssueSeverity.INFO       # "info"
IssueSeverity.WARNING    # "warning"
IssueSeverity.ERROR      # "error"
IssueSeverity.CRITICAL   # "critical"
```

Severity is determined automatically from violation rates:

| Violation Rate | Severity |
|---------------|----------|
| > 50% | CRITICAL |
| > 20% | ERROR |
| > 5% | WARNING |
| <= 5% | INFO |

### IssueCategory

```python
from mesh_issue_examiner import IssueCategory

IssueCategory.TOPOLOGICAL    # "topological"
IssueCategory.GEOMETRIC      # "geometric"
IssueCategory.QUALITY        # "quality"
IssueCategory.ORIENTATION    # "orientation"
```

### MeshIssue

Represents a single detected defect.

```python
from mesh_issue_examiner import MeshIssue, IssueCategory, IssueSeverity

issue = MeshIssue(
    operation_id="topology.detect_nonmanifold_edges",
    category=IssueCategory.TOPOLOGICAL,
    severity=IssueSeverity.ERROR,
    message="Detected 12 non-manifold edges (2.4% of faces).",
    locations=[],           # List[IssueLocation]
    metrics={"count": 12, "violation_rate": 0.024},
    metadata={},
)

issue.is_critical         # False
issue.to_dict()           # Serialized form (locations truncated to 10)
```

### QualityMetrics

Stores computed metric values along with automatic statistics and threshold violation counting.

```python
from mesh_issue_examiner import QualityMetrics
import numpy as np

qm = QualityMetrics(
    operation_id="quality.aspect_ratio",
    metric_name="aspect_ratio",
    values=np.array([0.9, 0.8, 0.3, 0.95]),
    threshold=0.5,
)

qm.statistics       # {'min': 0.3, 'max': 0.95, 'mean': 0.7375, 'median': 0.85,
                    #  'std': ..., 'p5': ..., 'p95': ...}
qm.violation_count  # 1  (one value below threshold)
qm.to_dict()        # Serialized form (excludes raw values)
```

### InspectionResult

Aggregates all issues and quality metrics from one or more operations.

```python
from mesh_issue_examiner import InspectionResult

result = InspectionResult(mesh_id="my_mesh", file_path="mesh.stl")
result.add_issue(issue)
result.compute_summary()

result.passed                # True if no CRITICAL or ERROR issues
result.summary               # Dict with total_issues, severity_breakdown, etc.
result.to_dict()             # Full serialized form
```

### OperationConfig

Configuration shared across operations.

```python
from mesh_issue_examiner import OperationConfig

config = OperationConfig(
    tolerance=1e-6,              # Numeric tolerance for geometric comparisons
    quality_threshold=0.1,       # Default quality threshold
    enable_caching=True,         # Cache native mesh objects
    max_iterations=1000,         # Iteration cap for iterative algorithms
    parallel_enabled=True,       # Enable parallel processing (future)
    library_preference=None,     # Force a specific library: "trimesh", "pyvista", "pymeshlab"
)
```

---

## Operations and the Registry

### Operation Registry

The `OperationRegistry` is a singleton that manages all registered operations. It is imported as `registry` from the top-level package.

```python
from mesh_issue_examiner import registry

# List all registered operations
registry.list_all()                   # List[OperationMetadata]

# List by category or library
from mesh_issue_examiner import IssueCategory
registry.list_by_category(IssueCategory.TOPOLOGICAL)
registry.list_by_library("pyvista")

# List by tag
registry.list_by_tag("basic")

# Get a specific operation
op = registry.get("topology.detect_nonmanifold_edges")

# Check membership
"topology.detect_nonmanifold_edges" in registry   # True

# Statistics
registry.get_statistics()              # total_operations, execution counts, timing
```

### Creating Workflows

```python
# All registered operations
workflow = registry.create_full_workflow()

# All operations in a category
from mesh_issue_examiner import IssueCategory
workflow = registry.create_category_workflow(IssueCategory.QUALITY)

# Custom subset
workflow = registry.create_workflow([
    "topology.detect_nonmanifold_edges",
    "quality.compute_cell_quality",
])

# With a config override
from mesh_issue_examiner import OperationConfig
config = OperationConfig(quality_threshold=0.3)
workflow = registry.create_full_workflow(config=config)

# Execute
result = workflow(mesh)   # Returns InspectionResult
```

### Registered Operations (v0.1.0)

| Operation ID | Category | Description | Recommended Library | Complexity |
|---|---|---|---|---|
| `topology.detect_nonmanifold_edges` | TOPOLOGICAL | Detects edges shared by more than two faces | pyvista | O(n) |
| `quality.compute_cell_quality` | QUALITY | Computes aspect ratio, skewness, min/max angles, scaled Jacobian | pyvista | O(n) |

### Adding a Custom Operation

Subclass `BaseOperation`, implement `metadata` and `execute`, then register:

```python
from mesh_issue_examiner import MeshInput, MeshIssue, IssueCategory, IssueSeverity
from mesh_issue_examiner.operations.base import BaseOperation, OperationMetadata
from mesh_issue_examiner import registry

class MyCustomCheck(BaseOperation):
    @property
    def metadata(self) -> OperationMetadata:
        return OperationMetadata(
            operation_id="custom.my_check",
            display_name="My Custom Check",
            description="Checks something specific",
            category=IssueCategory.GEOMETRIC,
            supported_libraries=["trimesh", "pyvista"],
            recommended_library="trimesh",
            tags=["custom"],
        )

    def execute(self, mesh: MeshInput) -> list:
        issues = []
        # ... your logic here ...
        return issues

# Register the operation
registry.register(MyCustomCheck())
```

You can also use the decorator pattern:

```python
from mesh_issue_examiner.operations.base import register_operation

@register_operation
class AnotherCheck(BaseOperation):
    @property
    def metadata(self):
        return OperationMetadata(
            operation_id="custom.another",
            display_name="Another Check",
            description="...",
            category=IssueCategory.GEOMETRIC,
            supported_libraries=["trimesh"],
            recommended_library="trimesh",
        )

    def execute(self, mesh):
        return []
```

### QualityOperation (advanced)

For operations that primarily produce `QualityMetrics` rather than `MeshIssue` objects, subclass `QualityOperation` instead:

```python
from mesh_issue_examiner.operations.base import QualityOperation
from mesh_issue_examiner import QualityMetrics

class MyMetric(QualityOperation):
    @property
    def metric_name(self) -> str:
        return "my_metric"

    def compute(self, mesh) -> QualityMetrics:
        values = ...  # compute per-cell values
        return QualityMetrics(
            operation_id="custom.my_metric",
            metric_name=self.metric_name,
            values=values,
            threshold=self.get_threshold(),
        )

    def get_threshold(self) -> float:
        return 0.5
```

---

## Adapters

Adapters translate between the canonical `MeshInput` format and library-specific mesh objects. All adapters implement the `MeshAdapter` abstract base class.

### MeshAdapter Interface

| Method | Description |
|--------|-------------|
| `load_mesh(path)` | Load mesh from file |
| `create_mesh(verts, faces)` | Create mesh from numpy arrays |
| `get_vertices(mesh)` | Extract Nx3 vertex array |
| `get_faces(mesh)` | Extract Mx3 face array |
| `is_watertight(mesh)` | Check for no boundary edges |
| `is_manifold(mesh)` | Returns `(bool, list_of_issues)` |
| `get_boundary_edges(mesh)` | Boundary edge indices |
| `get_nonmanifold_edges(mesh)` | List of `(v1, v2, face_count)` |
| `get_duplicate_vertices(mesh, tol)` | Duplicate vertex indices |
| `compute_face_areas(mesh)` | Per-face area array |
| `compute_face_normals(mesh)` | Nx3 normal array |
| `compute_cell_quality(mesh, type)` | Quality metric array |
| `detect_self_intersections(mesh)` | Intersecting face pairs |
| `to_canonical(mesh)` | Convert to `MeshInput` |
| `from_canonical(mesh_input)` | Convert from `MeshInput` |
| `compute_all_quality_metrics(mesh)` | Compute multiple metrics at once |

### Using Adapters Directly

```python
from adapters.pyvista_adapter import PyVistaAdapter

adapter = PyVistaAdapter()

# Check availability
adapter.is_available          # True if pyvista is installed

# Load
native = adapter.load_mesh("model.stl")

# Inspect
watertight = adapter.is_watertight(native)
is_manifold, issues = adapter.is_manifold(native)
areas = adapter.compute_face_areas(native)

# Convert to/from canonical format
from mesh_issue_examiner import MeshInput
canonical = adapter.to_canonical(native, mesh_id="model")
native_back = adapter.from_canonical(canonical)

# Quality metrics
aspect_ratios = adapter.compute_cell_quality(native, "aspect_ratio")
all_metrics = adapter.compute_all_quality_metrics(native, ["aspect_ratio", "skewness"])
```

### Adapter Implementations

| Adapter | Class | Native Type | Quality Metrics |
|---------|-------|-------------|-----------------|
| **TrimeshAdapter** | `trimesh.Trimesh` | Pure Python triangle mesh | area, aspect_ratio, min_angle, max_angle |
| **PyVistaAdapter** | `pyvista.PolyData` | VTK-backed surface mesh | area, aspect_ratio, skewness, scaled_jacobian, max_angle, min_angle, condition, Jacobian, volume |
| **PyMeshLabAdapter** | `pymeshlab.MeshSet` | MeshLab C++ engine | area, aspect_ratio (others via filters) |

---

## MCP Server

The MCP server exposes mesh inspection as tools that an AI agent can call through the Model Context Protocol.

### Tools

| Tool | Description | Required Params |
|------|-------------|-----------------|
| `mesh_load` | Load a mesh file, return mesh ID | `file_path` |
| `mesh_inspect` | Run inspection operations on a loaded mesh | `mesh_id` |
| `mesh_get_issues` | Retrieve filtered/paginated issues from an inspection | `mesh_id` |
| `mesh_get_quality_metrics` | Retrieve quality metrics from an inspection | `mesh_id` |
| `mesh_render` | Render a loaded mesh to a PNG (camera/appearance control, metric or material coloring, issue highlight markers) | `mesh_id` |
| `mesh_clear_cache` | Clear mesh and result caches | _(none)_ |

### Visualization

Controlled, offscreen rendering lives in `mesh_issue_examiner.visualization`:

```python
from mesh_issue_examiner.visualization import render_mesh

# Basic offscreen render
rr = render_mesh(mesh, "out.png", view="iso", width=1600, height=900)

# Camera fine-control (any named view + degrees)
rr = render_mesh(mesh, "angled.png", view="front", azimuth=25.0, elevation=10.0, zoom=1.4)

# Color cells by a cached quality metric (from an InspectionResult)
rr = render_mesh(mesh, "quality.png", scalars="scaled_jacobian", cmap="coolwarm", result=result)

# Color volume cells by material ID (requires load with material_array_name)
rr = render_mesh(mesh, "materials.png", scalars="material")

# Overlay severity-colored markers at detected issue locations
rr = render_mesh(mesh, "issues.png", issues=result.issues,
                 highlight_severities=("critical", "error"))
```

The same capability is exposed over MCP via the `mesh_render` tool (path,
view, camera, size, color, scalars, scalar bar, and issue highlighting
parameters). Rendering requires an OpenGL-capable environment; when no
offscreen GL context can be created, a `VisualizationError` with guidance
is raised.

### Request / Response

Requests use `MCPRequest` objects; responses are `MCPResponse`.

```python
from mcp.server import MeshIssueExaminerServer, MCPRequest, MCPResponse

server = MeshIssueExaminerServer()

# Load
resp = await server.handle_request(MCPRequest(
    method="mesh_load",
    params={"file_path": "bracket.stl", "mesh_id": "bracket"},
))
# resp.result => {"mesh_id": "bracket", "n_vertices": 5000, "n_faces": 10000, ...}

# Inspect (all operations)
resp = await server.handle_request(MCPRequest(
    method="mesh_inspect",
    params={"mesh_id": "bracket", "operations": ["all"]},
))
# resp.result => InspectionResult.to_dict()

# Inspect (specific operations)
resp = await server.handle_request(MCPRequest(
    method="mesh_inspect",
    params={"mesh_id": "bracket", "operations": ["topology.detect_nonmanifold_edges"]},
))

# Get cached result
resp = await server.handle_request(MCPRequest(
    method="mesh_get_result",
    params={"mesh_id": "bracket", "include_details": True},
))

# List operations
resp = await server.handle_request(MCPRequest(
    method="list_operations",
    params={"category": "topological"},
))

# Clear cache
resp = await server.handle_request(MCPRequest(
    method="mesh_clear_cache",
    params={},  # clears all; pass mesh_id to clear one
))
```

### AsyncRequestHandler

For finer-grained control, use the `AsyncRequestHandler` from `mcp.handlers`:

```python
from mcp.handlers import AsyncRequestHandler

handler = AsyncRequestHandler()
handler.register("custom_method", my_async_handler_func)
```

### Tool Definitions

To get machine-readable tool schemas for MCP integration:

```python
from mcp.tools import get_tool_definitions, ToolDefinition

tools = get_tool_definitions()    # List[ToolDefinition]
for t in tools:
    print(t.name, t.input_schema)
```

---

## Utility Functions

### Mesh Loading (`utils.mesh_loader`)

| Function | Description |
|----------|-------------|
| `load_mesh_from_file(path, mesh_id, library)` | Load mesh from file into `MeshInput` |
| `create_mesh_from_arrays(vertices, faces, mesh_id)` | Create `MeshInput` from numpy arrays |
| `save_mesh_to_file(mesh, path, library)` | Save `MeshInput` to disk |
| `get_supported_formats()` | Returns `{suffix: [libraries]}` dict |
| `is_format_supported(suffix)` | Boolean check for a file extension |

### Metrics and Thresholds (`utils.metrics`)

| Function | Description |
|----------|-------------|
| `get_default_thresholds()` | Returns dict of metric name to threshold value |
| `get_threshold(metric_name)` | Threshold for a single metric |
| `compute_all_quality_metrics(mesh, adapter)` | Compute multiple metrics, return `List[QualityMetrics]` |
| `compute_violation_stats(metrics_list)` | Aggregate violation statistics |
| `determine_severity(violation_rate)` | Map violation rate to `IssueSeverity` |

**Default thresholds:**

| Metric | Threshold |
|--------|-----------|
| aspect_ratio | 0.3 |
| skewness | 0.5 |
| min_angle | 20.0 degrees |
| max_angle | 140.0 degrees |
| scaled_jacobian | 0.1 |
| condition | 10.0 |
| Jacobian | 0.0 |
| volume | 1e-10 |
| area | 1e-10 |

---

## Configuration

Pass an `OperationConfig` to customize behavior:

```python
from mesh_issue_examiner import OperationConfig, registry

config = OperationConfig(
    tolerance=1e-8,
    quality_threshold=0.2,
    enable_caching=True,
    max_iterations=500,
    parallel_enabled=False,
    library_preference="pyvista",
)

# Use with a workflow
workflow = registry.create_full_workflow(config=config)
result = workflow(mesh)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tolerance` | `float` | `1e-6` | Numeric tolerance for geometric comparisons |
| `quality_threshold` | `float` | `0.1` | Default threshold for quality metrics |
| `enable_caching` | `bool` | `True` | Cache native mesh objects on `MeshInput` |
| `max_iterations` | `int` | `1000` | Cap for iterative algorithms |
| `parallel_enabled` | `bool` | `True` | Enable parallel execution (reserved) |
| `library_preference` | `str` or `None` | `None` | Force a specific adapter library |

---

## Supported File Formats

| Format | Extension | trimesh | PyVista | PyMeshLab |
|--------|-----------|:-------:|:-------:|:---------:|
| STL | `.stl` | yes | yes | yes |
| Wavefront OBJ | `.obj` | yes | yes | yes |
| Stanford PLY | `.ply` | yes | yes | yes |
| Object File Format | `.off` | yes | -- | yes |
| glTF | `.gltf` | yes | yes | -- |
| Binary glTF | `.glb` | yes | yes | -- |
| VTK PolyData | `.vtp` | -- | yes | -- |
| VTK Legacy | `.vtk` | -- | yes | -- |
| 3D Manufacturing | `.3mf` | yes | -- | -- |

---

## Library Comparison

| Feature | trimesh | PyVista (VTK) | PyMeshLab |
|---------|---------|---------------|-----------|
| **Pure Python** | Yes | No (C++ VTK) | No (C++ MeshLab) |
| **File formats** | 7 | 6 | 5 |
| **Cell quality metrics** | area, aspect_ratio, angles | area, aspect_ratio, skewness, scaled_jacobian, angles, condition, Jacobian, volume | area, aspect_ratio (via filters) |
| **Topology queries** | edges, face adjacency, boundary | manifold check, open edges | selection-based filters |
| **Self-intersection** | heuristic (volume ratio) | VTK filter | native filter |
| **Duplicate vertex removal** | scipy cKDTree | `mesh.clean()` | `merge_close_vertices` |
| **Recommended for** | Lightweight scripts, rapid prototyping | Quality-heavy workflows, VTK ecosystem | Mesh repair, advanced filtering |

---

## Development Guide

### Running Tests

```bash
# All tests
pytest tests/ -v

# Specific test class
pytest tests/test_operations.py::TestNonManifoldEdgeDetection -v

# With coverage
pytest tests/ -v --cov=mesh_issue_examiner
```

### Project Structure Conventions

- **Operations** go in `operations/<category>/<name>.py` and self-register with the module-level `registry` at import time.
- **Adapters** go in `adapters/<library>_adapter.py` and implement `MeshAdapter`.
- **Tests** go in `tests/` and use `pytest`.

### Adding a New Operation

1. Create `operations/<category>/my_operation.py`.
2. Subclass `BaseOperation`.
3. Implement the `metadata` property (return `OperationMetadata`).
4. Implement the `execute(self, mesh: MeshInput) -> List[MeshIssue]` method.
5. Call `registry.register(MyOperation())` at module level.
6. Add tests in `tests/test_operations.py`.

### Adding a New Adapter

1. Create `adapters/mylib_adapter.py`.
2. Subclass `MeshAdapter` from `adapters.base_adapter`.
3. Implement all abstract methods.
4. Update `adapters/__init__.py` to export the new class.
5. Update `utils/mesh_loader.py` to include format support mappings.
6. Add the library to `setup.py` extras.

### Code Style

```bash
# Format
black mesh_issue_examiner/

# Lint
flake8 mesh_issue_examiner/
```

---

## Contributing

Contributions are welcome. To contribute:

1. Fork the repository.
2. Create a feature branch: `git checkout -b feature/my-feature`.
3. Write tests for any new functionality.
4. Ensure all tests pass: `pytest tests/ -v`.
5. Run linting: `flake8 mesh_issue_examiner/` and `black --check mesh_issue_examiner/`.
6. Commit with a descriptive message.
7. Open a pull request.

**Areas that especially need contributions:**

- Geometric operations (the `operations/geometric/` package is empty)
- Orientation checks (face normal consistency, winding order)
- Self-intersection detection improvements
- MCP transport layer (stdio, SSE)
- Additional test coverage
- Documentation and examples

---

## License

MIT License. See [LICENSE](LICENSE) for details.
