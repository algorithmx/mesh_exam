# Mesh Inspection MCP Architecture: Distributed Privacy-Preserving System Design Report

## Executive Summary

This report presents a comprehensive design analysis for a distributed mesh inspection system that leverages WebAssembly (WASM) running on the user's browser as the compute engine, with a cloud-based AI agent orchestrating inspection workflows. The architecture fundamentally addresses the privacy concern of keeping mesh data localized to user devices while providing sophisticated AI-driven inspection capabilities. However, this design introduces several non-trivial architectural challenges that require careful resolution, particularly around role definitions, communication patterns, and data ownership boundaries.

The proposed system implements an unconventional but architecturally sound role reversal where the WASM module functions as an MCP server (exposing mesh processing tools) and the cloud service functions as an MCP client (orchestrating inspection workflows). This document clarifies the roles and responsibilities of each component, identifies gaps in the original design specification, and proposes concrete solutions for realistic implementation.

---

## 1. System Overview and Design Philosophy

### 1.1 Core Design Principles

The architecture embodies three fundamental design principles that guide all architectural decisions. First, privacy by computation placement ensures that raw mesh geometry data never traverses the network boundary, with all computationally intensive operations executing within the user's browser. Second, intelligence via cloud synthesis leverages the cloud service's access to large language models for natural language understanding, report generation, and inspection workflow orchestration. Third, user sovereignty over resources places the computational burden entirely on the user's device, eliminating the operator's infrastructure costs for mesh processing.

These principles address a genuine market need in intellectual property-sensitive industries including aerospace, automotive, manufacturing, and architectural sectors. Companies in these industries often possess valuable CAD designs that cannot be uploaded to third-party services for inspection due to confidentiality requirements. The proposed architecture enables these companies to leverage sophisticated AI inspection capabilities without compromising their design data.

### 1.2 High-Level System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              USER'S DEVICE                                       │
│                                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────────┐ │
│  │                            Web Browser                                       │ │
│  │  ┌───────────────────────────────────────────────────────────────────────┐   │ │
│  │  │                     Web Application Container                           │   │ │
│  │  │                                                                        │   │ │
│  │  │  ┌──────────────────────┐  ┌──────────────────────┐                    │   │ │
│  │  │  │   UI/UX Layer        │  │   WebSocket/HTTP     │                    │   │ │
│  │  │  │   - Upload Control   │  │   Client             │                    │   │ │
│  │  │  │   - Description Input│  │   (Outbound Only)    │                    │   │ │
│  │  │  │   - Result Display   │  │                      │                    │   │ │
│  │  │  └──────────┬───────────┘  └──────────┬───────────┘                    │   │ │
│  │  │             │                         │                                 │   │ │
│  │  │             └────────────┬────────────┘                                 │   │ │
│  │  │                          │                                              │   │ │
│  │  │             ┌────────────┴────────────┐                                 │   │ │
│  │  │             │   JavaScript Bridge     │                                 │   │ │
│  │  │             │   - WASM Lifecycle     │                                 │   │ │
│  │  │             │   - Memory Management  │                                 │   │ │
│  │  │             │   - Callbacks         │                                 │   │ │
│  │  │             └────────────┬────────────┘                                 │   │ │
│  │  │                          │                                              │   │ │
│  │  │  ┌───────────────────────┼───────────────────────────────────────┐   │   │ │
│  │  │  │                       ▼                                        │   │   │ │
│  │  │  │  ┌─────────────────────────────────────────────────────────┐    │   │ │ │
│  │  │  │  │              WASM Mesh Processor                          │    │   │ │ │
│  │  │  │  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐   │    │   │ │ │
│  │  │  │  │  │ Mesh Loader │  │  Topology   │  │  Quality        │   │    │   │ │ │
│  │  │  │  │  │ & Validator │  │  Analyzer   │  │  Metrics        │   │    │   │ │ │
│  │  │  │  │  └─────────────┘  └─────────────┘  └─────────────────┘   │    │   │ │ │
│  │  │  │  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐   │    │   │ │ │
│  │  │  │  │  │ Geometric   │  │  Defect     │  │  Abstraction    │   │    │   │ │ │
│  │  │  │  │  │ Operations  │  │  Detector   │  │  Generator      │   │    │   │ │ │
│  │  │  │  │  └─────────────┘  └─────────────┘  └─────────────────┘   │    │   │ │ │
│  │  │  │  └─────────────────────────────────────────────────────────┘    │   │ │ │
│  │  │  │                                                                        │   │ │
│  │  │  │  ┌─────────────────────────────────────────────────────────┐    │   │ │ │
│  │  │  │  │              IndexedDB Local Storage                     │    │   │ │ │
│  │  │  │  │  - Mesh Data (encrypted at rest)                         │    │   │ │ │
│  │  │  │  │  - Session Checkpoints                                    │    │   │ │ │
│  │  │  │  │  - Inspection History                                     │    │   │ │ │
│  │  │  │  └─────────────────────────────────────────────────────────┘    │   │ │ │
│  │  │  └───────────────────────────────────────────────────────────────────────┘   │ │
│  │  └─────────────────────────────────────────────────────────────────────────────┘   │ │
│  │                                                                                  │
│  │  ┌─────────────────────────────────────────────────────────────────────────────┐ │ │
│  │  │                     User-Owned Resources (All Compute Here)                  │ │ │
│  │  │  CPU/GPU/Memory: User's device    Storage: User's IndexedDB                  │ │ │
│  │  └─────────────────────────────────────────────────────────────────────────────┘ │ │
│  └─────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
                                      │
                    ══════════════════╪═════════════════
                           HTTPS/WSS (Outbound Only)
                    ══════════════════╪═════════════════
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              CLOUD INFRASTRUCTURE                               │
│                           (Operator's Responsibility)                            │
│                                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────────┐ │
│  │                        Public Internet Endpoint                              │ │
│  │  ┌───────────────────────────────────────────────────────────────────────┐   │ │
│  │  │                    Load Balancer / API Gateway                          │   │ │
│  │  │                    - TLS Termination                                    │   │ │
│  │  │                    - Rate Limiting                                       │   │ │
│  │  │                    - Authentication                                     │   │ │
│  │  └────────────────────────────┬──────────────────────────────────────────┘   │ │
│  │                               │                                              │ │
│  │         ┌─────────────────────┼─────────────────────┐                       │ │
│  │         │                     │                     │                       │ │
│  │         ▼                     ▼                     ▼                       │ │
│  │  ┌─────────────┐       ┌─────────────┐       ┌─────────────┐                │ │
│  │  │ MCP Client  │       │ MCP Client  │  ...  │ MCP Client  │                │ │
│  │  │ Session 1   │       │ Session 2   │       │ Session N   │                │ │
│  │  └──────┬──────┘       └──────┬──────┘       └──────┬──────┘                │ │
│  │         │                     │                     │                       │ │
│  │         └─────────────────────┼─────────────────────┘                       │ │
│  │                               │                                              │ │
│  │                               ▼                                              │ │
│  │  ┌───────────────────────────────────────────────────────────────────────┐   │ │
│  │  │                     LLM Orchestration Layer                              │   │ │
│  │  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐          │   │ │
│  │  │  │ Natural Language│  │ Inspection      │  │ Report          │          │   │ │
│  │  │  │ Processor       │  │ Workflow Engine │  │ Generator       │          │   │ │
│  │  │  └─────────────────┘  └─────────────────┘  └─────────────────┘          │   │ │
│  │  └───────────────────────────────────────────────────────────────────────┘   │ │
│  │                               │                                              │ │
│  │                               ▼                                              │ │
│  │  ┌───────────────────────────────────────────────────────────────────────┐   │ │
│  │  │                         LLM Service (External)                          │   │ │
│  │  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐          │   │ │
│  │  │  │ Intent          │  │ Context         │  │ Report          │          │   │ │
│  │  │  │ Classification  │  │ Synthesis       │  │ Composition     │          │   │ │
│  │  │  └─────────────────┘  └─────────────────┘  └─────────────────┘          │   │ │
│  │  └───────────────────────────────────────────────────────────────────────┘   │ │
│  │                                                                              │ │
│  │  ┌───────────────────────────────────────────────────────────────────────┐   │ │
│  │  │                     Session State Store (Redis/DB)                      │   │ │
│  │  │  - Session metadata only (no mesh data)                                 │   │ │
│  │  │  - LLM conversation history                                             │   │ │
│  │  │  - Inspection workflow state                                            │   │ │
│  │  └───────────────────────────────────────────────────────────────────────┘   │ │
│  │                                                                              │ │
│  └─────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────────┐ │
│  │                   Operator-Owned Resources (All Compute Here)                 │ │
│  │  CPU/GPU: Cloud instances    Storage: Operator's database                    │ │
│  │  Network: Operator's servers    LLM: Third-party API                        │ │
│  └─────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Role Definitions and Responsibilities Matrix

### 2.1 Component Role Clarification

The architecture involves three primary actors with distinct responsibilities that must be clearly delineated to avoid confusion and ensure accountability.

**The User** is the human operator who owns the mesh data and initiates inspection requests. The user possesses exclusive ownership of the mesh file, which remains on their local device throughout the entire inspection workflow. The user provides natural language descriptions of inspection requirements and receives final inspection reports. The user provides the computational resources necessary for mesh processing, including CPU cycles, memory, and local storage through their browser environment.

**The WASM Mesh Processor (MCP Server)** is the computational engine executing within the user's browser. Despite being labeled as a server in the MCP protocol sense, this component is entirely under the user's control and runs on the user's device. The WASM processor exposes mesh inspection tools as MCP resources and handles mesh loading, geometric analysis, defect detection, and quality metric computation. It never transmits raw mesh data over the network and generates geometric abstractions for LLM context.

**The Cloud Service (MCP Client)** is the operator's infrastructure that provides AI orchestration capabilities. The cloud service functions as the MCP client, connecting to WASM instances running in user browsers. It receives natural language inspection requests from users, translates them into structured inspection commands via LLM processing, sends commands to WASM processors for execution, receives results, and generates final inspection reports. The cloud service never receives raw mesh geometry data.

### 2.2 Detailed Responsibility Matrix

| Responsibility Domain | User | WASM Processor | Cloud Service |
|----------------------|------|-----------------|---------------|
| **Mesh Data Ownership** | Full ownership; can delete anytime | Transient access during processing only | Never receives; only receives abstractions |
| **Computational Resources** | Provides all CPU/memory/GPU via browser | Consumes user's resources; bounded by browser limits | Provides LLM API resources only |
| **Network Communication** | Initiates connection; all traffic originates here | No direct network access; communicates via JS bridge | Receives connections; processes requests |
| **Inspection Logic** | Defines requirements via NL description | Executes inspection algorithms locally | Orchestrates workflow; generates reports |
| **Result Interpretation** | Receives final reports | Generates raw measurements and classifications | Synthesizes explanations and recommendations |
| **Session Persistence** | Stores mesh locally in IndexedDB | Writes checkpoints to IndexedDB | Stores session metadata only |
| **Authentication** | Provides credentials to cloud | No authentication; runs on user's device | Validates credentials; enforces policies |
| **Error Handling** | Receives user-facing error messages | Reports errors to JS bridge | Logs server-side errors; generates user messages |
| **Cost Attribution** | Pays for own device usage | Zero cost (runs locally) | Operator pays for infrastructure |

### 2.3 Data Ownership Boundaries

The following data ownership rules are fundamental to the architecture's privacy guarantees and must be enforced through implementation.

**Data That Belongs to the User**: Raw mesh geometry including vertex positions, face indices, and mesh topology. Textured mesh assets including texture coordinates and material definitions. Inspection results containing location-specific findings that could reveal geometry. Session state including mesh processing checkpoints and intermediate results. User preferences and inspection history stored locally.

**Data That Belongs to the Operator**: Natural language inspection descriptions (though these may contain sensitive context). Generated inspection reports (though these may describe sensitive findings). LLM conversation context including the conversation history maintained by the cloud service. Aggregated anonymized usage statistics about inspection patterns.

**Data That Must Never Leave the User's Device**: Raw mesh file content in any form. Vertex coordinate arrays or any data enabling geometry reconstruction. Face adjacency information that could reveal topology. Any data that would enable an adversary to reconstruct the original design.

---

## 3. Communication Protocol Design

### 3.1 Why Traditional MCP Roles Are Reversed

The Model Context Protocol defines a symmetric communication pattern where participants can assume server or client roles based on capability exposure rather than network topology. In standard MCP deployments, an MCP server runs on a stable host with a known network endpoint, and MCP clients connect to discover and invoke tools. This architecture reverses those roles because the capability provider (WASM mesh processor) runs on an ephemeral browser endpoint that cannot accept inbound connections.

The WASM module exposes mesh processing tools, qualifying it as an MCP server in the protocol sense. However, it cannot listen on a TCP port or accept incoming connections due to browser security restrictions. Therefore, the browser initiates an outbound connection to the cloud service, which assumes the MCP client role. This creates an unconventional but technically valid configuration where the cloud service connects to nothing; instead, it receives inbound connections from browser clients and responds to requests received over those connections.

### 3.2 Recommended Transport: Streamable HTTP

The MCP specification's Streamable HTTP transport, standardized in early 2025, provides the most natural mechanism for browser-to-cloud communication. This transport uses standard HTTP POST requests from the browser client to the cloud server endpoint, with responses delivered in the same HTTP connection using chunked transfer encoding for streaming.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         Communication Flow (Streamable HTTP)                     │
│                                                                                  │
│  1. Browser establishes HTTPS connection to cloud endpoint                       │
│                                                                                  │
│  2. Browser (MCP Client role) sends POST request with:                          │
│     - User's NL description                                                     │
│     - Session identifier                                                        │
│     - Authentication token                                                      │
│                                                                                  │
│     POST /api/v1/mcp/sessions/{session_id}/invoke                               │
│     Content-Type: application/json                                              │
│     Authorization: Bearer {token}                                               │
│                                                                                  │
│     {                                                                           │
│       "jsonrpc": "2.0",                                                         │
│       "method": "tools/call",                                                   │
│       "params": {                                                                │
│         "name": "mesh_topology.analyze",                                        │
│         "arguments": {"query": "identify all non-manifold edges"}              │
│       },                                                                        │
│       "id": "req-001"                                                           │
│     }                                                                           │
│                                                                                  │
│  3. Cloud server (MCP Server role) receives request                              │
│     - Validates session                                                         │
│     - Routes to appropriate session handler                                     │
│     - Processes via LLM orchestration                                           │
│                                                                                  │
│  4. Cloud sends response via chunked transfer encoding                          │
│                                                                                  │
│     HTTP/1.1 200 OK                                                             │
│     Content-Type: application/json                                             │
│     Transfer-Encoding: chunked                                                 │
│                                                                                  │
│     {"jsonrpc": "2.0", "result": {...}, "id": "req-001"}                       │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

This approach offers several advantages. First, browsers can initiate HTTPS connections trivially without requiring inbound connection capability. Second, standard HTTP infrastructure including load balancers, API gateways, and CDN providers handle this pattern natively. Third, TLS encryption protects all communication without requiring special WebSocket configuration. Fourth, the request-response pattern aligns with how browsers naturally communicate with servers.

### 3.3 Alternative: WebSocket with Relay (For Real-Time Interactions)

If the user experience requires real-time bidirectional communication, such as streaming inspection progress updates or enabling conversational inspection where the AI can ask follow-up questions, WebSocket communication becomes necessary. However, browsers cannot accept inbound WebSocket connections, necessitating a relay architecture.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         WebSocket Communication with Relay                        │
│                                                                                  │
│  ┌─────────────────┐                              ┌─────────────────┐           │
│  │   User Browser  │                              │  Relay Server   │           │
│  │                 │                              │  (Cloud)        │           │
│  │  ┌───────────┐  │    WebSocket Connection 1     │                 │           │
│  │  │ WASM MCP  │◄─┼──────────────────────────────►│                 │           │
│  │  │  Client   │  │     (Outbound from browser)   │                 │           │
│  │  └───────────┘  │                              │                 │           │
│  │        │        │                              │        │        │           │
│  │        │        │                              │        │        │           │
│  │        ▼        │                              │        ▼        │           │
│  │  ┌───────────┐  │    Internal Communication     │  ┌───────────┐  │           │
│  │  │   JS      │  │                              │  │  Message  │  │           │
│  │  │  Bridge   │  │                              │  │  Router   │  │           │
│  │  └───────────┘  │                              │  └─────┬─────┘  │           │
│  └─────────────────┘                              │        │        │           │
│                                                     │        │        │           │
│                                                     │        ▼        │           │
│                                                     │  ┌───────────┐  │           │
│                                                     │  │ MCP Client│  │           │
│                                                     │  │  Pool     │  │           │
│                                                     │  └───────────┘  │           │
│                                                     └─────────────────┘           │
│                                                                                  │
│  Message Flow:                                                                  │
│  1. Browser opens WebSocket to relay: wss://relay.operator.com/ws/{session_id} │
│  2. Cloud connects internal MCP client to same relay session                     │
│  3. Browser sends message → Relay → Cloud MCP Client                             │
│  4. Cloud sends message → Relay → Browser                                       │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

The relay server maintains mappings between browser WebSocket connections and cloud MCP client connections. Messages from either side are forwarded to the other based on session identifiers. This adds latency and requires the relay to maintain persistent state, but it enables true bidirectional communication necessary for interactive inspection sessions.

### 3.4 MCP Protocol Message Flows

The architecture implements a modified MCP protocol adapted for browser constraints. The following sequence diagrams illustrate the complete interaction lifecycle.

**Session Initialization Flow**:

```
User          Browser          WASM           Cloud           LLM
 │              │               │              │              │
 │  Upload Mesh │               │              │              │
 ├─────────────►│               │              │              │
 │              │               │              │              │
 │              │  Load Mesh    │              │              │
 │              ├──────────────►│              │              │
 │              │               │              │              │
 │              │◄──────────────┤              │              │
 │              │  Mesh Loaded  │              │              │
 │              │               │              │              │
 │              │  Generate Abstraction         │              │
 │              ├──────────────►│              │              │
 │              │               │              │              │
 │              │◄──────────────┤              │              │
 │              │  Abstraction  │              │              │
 │              │               │              │              │
 │  Enter NL    │               │              │              │
 │  Description │               │              │              │
 ├─────────────►│               │              │              │
 │              │               │              │              │
 │              │  HTTP POST (Abstraction + NL Description)    │
 │              ├──────────────────────────────────────────────┤
 │              │               │              │              │
 │              │               │   Validate & Route Request    │
 │              │               ├─────────────►│              │
 │              │               │              │              │
 │              │               │   Invoke LLM Processing       │
 │              │               │              ├─────────────►│
 │              │               │              │              │
 │              │               │◄────────────────────────────┤
 │              │               │  Structured Inspection Plan  │
 │              │               │              │              │
```

**Inspection Execution Flow**:

```
User          Browser          WASM           Cloud           LLM
 │              │               │              │              │
 │  Hit Start   │               │              │              │
 ├─────────────►│               │              │              │
 │              │               │              │              │
 │              │  HTTP POST (Inspection Command)              │
 │              ├───────────────►│              │              │
 │              │               │              │              │
 │              │               │  Execute Locally              │
 │              │               │  ┌─────────────────────┐      │
 │              │               │  │ 1. Non-manifold Edges      │
 │              │               │  │ 2. Aspect Ratio Check     │
 │              │               │  │ 3. Self-Intersection      │
 │              │               │  │ 4. Thickness Analysis     │
 │              │               │  │ ...                       │
 │              │               │  └─────────────────────┘      │
 │              │               │              │              │
 │              │  HTTP POST (Raw Results)      │              │
 │              │◄───────────────┤              │              │
 │              │               │              │              │
 │              │  Forward Results to LLM       │              │
 │              │               ├─────────────►│              │
 │              │               │              │              │
 │              │               │◄─────────────┤              │
 │              │               │  Synthesized Analysis        │
 │              │               │              │              │
 │              │  Generate Final Report       │              │
 │              │◄─────────────────────────────┤              │
 │              │               │              │              │
 │  View Report │               │              │              │
 ├◄─────────────┤               │              │              │
 │              │               │              │              │
```

---

## 4. Security and Privacy Architecture

### 4.1 Threat Model

The architecture must defend against several distinct threat categories, each requiring specific countermeasures.

**Threat Category 1: Mesh Data Exfiltration** encompasses scenarios where an adversary attempts to obtain the raw mesh geometry. This includes both technical attacks such as network sniffing and social engineering approaches such as compelling the operator to reveal data. The primary defense is the architectural guarantee that raw mesh data never reaches the operator's infrastructure. Even if the operator's servers are compromised, no mesh data is present to steal.

**Threat Category 2: Inspection Result Inference** involves deriving mesh geometry from inspection results. While raw coordinates never leave the browser, inspection results contain location-specific information that could theoretically enable geometry reconstruction. A determined adversary with access to both the geometric abstraction and detailed inspection results might reconstruct significant portions of the original geometry. The geometric abstraction layer provides a defense by omitting precise coordinate information.

**Threat Category 3: Description-Based Information Leakage** recognizes that user descriptions reveal context about the mesh being inspected. Descriptions like "analyze the turbine blade cooling channels" convey significant design information. The operator's LLM infrastructure necessarily processes these descriptions, creating an information disclosure risk. Local description preprocessing can generalize descriptions before transmission.

**Threat Category 4: Infrastructure Compromise** addresses scenarios where the operator's cloud infrastructure is breached. The defense is minimizing what data resides on operator infrastructure to only data essential for service provision. Session metadata, LLM conversation history, and generated reports must be protected through standard cloud security measures.

### 4.2 Defense Layers

**Layer 1: Architecture Enforcement** guarantees that raw mesh data never enters the operator's network. This is enforced by the WASM implementation, which provides no mechanism for exporting raw geometry. The JavaScript bridge between WASM and the network layer is explicitly designed to prevent geometry transmission.

**Layer 2: Geometric Abstraction** generates structured summaries of mesh properties that provide LLM context without revealing sensitive geometry. The abstraction includes only aggregate statistics, topological features, and general characteristics that cannot be used to reconstruct the mesh.

**Layer 3: Description Sanitization** preprocesses user descriptions to remove or generalize potentially sensitive terminology before transmission to the cloud. Keyword replacement, phrase abstraction, and terminology generalization reduce information leakage.

**Layer 4: Transport Security** encrypts all communication between browser and cloud using TLS 1.3. This prevents network-level eavesdropping and man-in-the-middle attacks.

**Layer 5: Infrastructure Hardening** applies standard cloud security practices including network segmentation, access controls, audit logging, and intrusion detection to protect operator infrastructure.

### 4.3 Geometric Abstraction Specification

The geometric abstraction layer generates a structured description of mesh properties without exposing raw geometry. The abstraction must balance providing sufficient context for meaningful LLM assistance against minimizing geometry disclosure.

**Required Abstraction Fields**:

```json
{
  "mesh_statistics": {
    "vertex_count_range": "string",      // e.g., "100K-500K", "1M-5M"
    "face_count_range": "string",         // e.g., "200K-1M"
    "surface_area_range": "string",      // e.g., "10-100 cm²"
    "volume_range": "string",             // e.g., "1-10 cm³" or "N/A (open surface)"
  },
  "topology_classification": {
    "is_watertight": "boolean",
    "has_boundary": "boolean",
    "component_count": "integer",
    "euler_characteristic": "integer",
    "genus": "integer",
  },
  "feature_summary": {
    "hole_count": "integer | unknown",
    "channel_count": "integer | unknown",
    "cavity_count": "integer | unknown",
    "sharp_edge_count_range": "string",
    "high_curvature_region_count": "integer | unknown",
  },
  "shape_characteristics": {
    "bounding_box_aspect_ratios": [0.0-1.0, 0.0-1.0, 0.0-1.0],
    "is_symmetric": "boolean | unknown",
    "symmetry_type": "none | bilateral | radial | reflective | unknown",
    "primary_axis_alignment": "x_aligned | y_aligned | z_aligned | arbitrary | unknown",
  },
  "quality_summary": {
    "average_aspect_ratio_range": "string",
    "minimum_angle_distribution": "good | moderate | poor",
    "degenerate_element_percentage": "string",
    "self_intersection_detected": "boolean | unknown",
  },
  "semantic_hints": {
    "inferred_type": "string | null",    // e.g., "mechanical_part", "architectural_element"
    "inferred_manufacturing_method": "string | null",
    "keywords_from_description": ["string"],
  }
}
```

The abstraction omits precise coordinates, exact measurements at specific locations, detailed curvature values at points, and any data enabling surface reconstruction.

### 4.4 Privacy-Preserving Description Processing

User descriptions undergo local preprocessing before transmission to the cloud service. This preprocessing operates within the WASM module or JavaScript layer and never transmits raw descriptions.

**Preprocessing Pipeline**:

1. **Terminology Detection**: Identify industry-specific terms that reveal sensitive information
2. **Abstraction Mapping**: Replace specific terms with abstract equivalents
3. **Structure Preservation**: Maintain grammatical structure for LLM comprehension
4. **Intent Preservation**: Ensure inspection intent remains clear after transformation

**Example Transformation**:

```
Original: "Check the turbine blade leading edges for erosion depth exceeding 0.5mm"

Transformed: "Check leading edges of aerodynamic surfaces for erosion exceeding threshold"

Original: "Inspect conformal cooling channels in the injection mold for blockages"

Transformed: "Inspect internal channel structures for obstructions"
```

The transformation preserves the general inspection intent while removing specific terminology that could reveal design details. The LLM receives enough context to formulate appropriate inspection strategies without understanding the specific application domain.

---

## 5. Session Management and State Architecture

### 5.1 Session Lifecycle

Sessions in this architecture differ from traditional server-side sessions because significant state resides on the client device. The session lifecycle must account for the distributed nature of state.

**Session States**:

| State | Description | Where Data Lives |
|-------|-------------|------------------|
| **Initial** | User opens application | None |
| **Mesh Loaded** | User uploads mesh file | User's IndexedDB + WASM memory |
| **Session Established** | Connection to cloud created | User's device + Cloud's session registry |
| **Inspection Active** | User submits inspection request | User's device + Cloud's LLM context |
| **Awaiting Results** | WASM processing in progress | User's device |
| **Report Ready** | Results delivered to user | User's device + Cloud's session store |
| **Session Expired** | Connection closed, data cleaned | User's IndexedDB (optional retention) |

### 5.2 State Partitioning

State is partitioned between client and server based on data ownership principles and access patterns.

**Client-Side State (User-Owned)**:

- Raw mesh geometry data (in IndexedDB, encrypted)
- WASM module memory (transient, lost on tab close)
- Session checkpoint data (in IndexedDB)
- Local inspection history (in IndexedDB)
- User preferences (in IndexedDB)

**Server-Side State (Operator-Owned)**:

- Session metadata (session ID, creation time, user ID)
- LLM conversation context (for multi-turn inspection dialogues)
- Workflow state (which inspection steps completed)
- Report cache (final reports for retrieval)
- Usage metrics and analytics

### 5.3 Checkpoint and Recovery Protocol

Long-running inspections may be interrupted by connection drops, browser crashes, or user tab closes. The checkpoint protocol enables recovery and resumption.

**Checkpoint Data Structure**:

```json
{
  "checkpoint_id": "uuid",
  "session_id": "uuid",
  "timestamp": "ISO8601",
  "workflow_state": {
    "current_step": "integer",
    "completed_steps": ["integer", "integer", ...],
    "pending_commands": ["command_id", ...],
    "results_received": ["result_id", ...],
  },
  "abstraction_version": "string",
  "hash_of_mesh": "sha256",  // To verify mesh unchanged
  "resume_token": "encrypted_string",
}
```

**Recovery Flow**:

1. User reopens application or reconnects after interruption
2. Application checks IndexedDB for existing checkpoints
3. If checkpoint found, prompt user: "Resume previous inspection?"
4. User confirms, application sends resume token to cloud
5. Cloud retrieves session state and issues remaining commands
6. WASM resumes from checkpoint, avoiding reprocessing completed steps

### 5.4 Connection Management

**WebSocket Connection Handling**:

The architecture supports WebSocket connections for real-time inspection when required. Connection management follows these principles:

1. **Outbound Only**: Browser always initiates connections; cloud never initiates
2. **Heartbeat**: Client sends ping every 30 seconds to maintain connection
3. **Reconnection**: Exponential backoff starting at 1 second, max 60 seconds
4. **Session Binding**: Connection bound to session ID; messages routed accordingly

**HTTP Polling Fallback**:

If WebSocket connections are unavailable or unreliable, the architecture falls back to HTTP long-polling. In this mode, the browser periodically sends POST requests with inspection commands, and the cloud responds when results are available or a timeout occurs.

```
Browser                              Cloud
 │                                     │
 │  POST /poll?session={id}&since={seq}               │
 ├────────────────────────────────────►
 │                                     │
 │                    [Cloud processes request]
 │                                     │
 │◄────────────────────────────────────┤
 │  Response (results or timeout)       │
 │                                     │
 │  [Browser waits 5 seconds, repeats]  │
 │                                     │
```

Long-polling increases latency and server load compared to WebSocket but provides reliable communication without requiring special network configuration.

---

## 6. Gap Analysis and Proposed Solutions

### 6.1 Gap: Connection Establishment Architecture

**Identified Gap**: The original design assumes direct cloud-to-browser communication for MCP server invocation, which is impossible due to browser network restrictions.

**Proposed Solution**: Implement a relay-based architecture where the browser maintains persistent connections to a relay endpoint, and the cloud MCP client connects to the same relay session. The relay forwards messages bidirectionally without either party requiring inbound connection capability.

**Implementation Specifics**: Deploy a WebSocket relay service on the cloud infrastructure. When a user session begins, the browser opens a WebSocket to the relay with a session-specific path. The cloud MCP client connects to the same relay path using an internal connection. The relay maintains a session table mapping session IDs to connection pairs and forwards messages accordingly. This adds approximately 10-20ms latency per message hop but enables full bidirectional communication.

### 6.2 Gap: LLM Context Without Geometry

**Identified Gap**: The LLM cannot effectively assist with inspection planning or result interpretation without understanding the mesh geometry, but providing raw geometry violates the privacy requirement.

**Proposed Solution**: Implement a structured geometric abstraction layer that generates comprehensive mesh summaries without exposing coordinates or enabling reconstruction.

**Implementation Specifics**: The WASM module generates an abstraction immediately after mesh loading. This abstraction includes topological invariants (face counts, Euler characteristic, genus), statistical distributions (aspect ratio ranges, angle distributions), feature presence indicators (holes detected: yes, channels detected: yes), and general shape characteristics (symmetric: bilateral, primary axis: z-aligned). The abstraction undergoes privacy review to ensure it provides LLM context without enabling geometry reconstruction. The abstraction is versioned so changes to the abstraction algorithm can be tracked.

### 6.3 Gap: WASM Resource Limits

**Identified Gap**: Browser-based WASM has strict memory limits (typically 2-4GB) and cannot process arbitrarily large meshes that server-side processing could handle.

**Proposed Solution**: Implement a tiered processing architecture with graceful degradation.

**Implementation Specifics**: Tier 1 (Local WASM) handles meshes up to 50M polygons with full functionality. Tier 2 (Chunked Processing) handles meshes up to 500M polygons by dividing into spatial regions processed sequentially. Tier 3 (Service Warning) advises users with larger meshes to use desktop tools, as browser-based processing is unsuitable. The WASM module automatically detects mesh size and routes to appropriate tier, providing clear user feedback when mesh exceeds local processing capability.

### 6.4 Gap: Offline Capability

**Identified Gap**: The original design requires cloud connectivity for any inspection, preventing offline usage.

**Proposed Solution**: Implement a basic offline mode with limited functionality.

**Implementation Specifics**: When offline, the WASM module provides basic inspection capabilities without LLM assistance. Users receive structured measurement results but must interpret them without AI-generated explanations. Inspection history syncs to cloud when connectivity returns. The offline mode is clearly indicated in the UI to manage user expectations. Not all operations are available offline; complex multi-step workflows require cloud coordination.

### 6.5 Gap: Cross-Session Continuity

**Identified Gap**: Users cannot resume previous inspection sessions across different devices or after clearing browser data.

**Proposed Solution**: Implement optional cloud backup for session metadata and inspection history.

**Implementation Specifics**: Users can optionally enable session backup to their cloud account. Backed up data includes abstraction versions, inspection history, report summaries, and user preferences. Raw mesh data is never backed up; users must reload meshes on new devices. The backup feature requires authentication and provides clear user control over what data is stored. This enables cross-device continuity while maintaining the privacy guarantee for mesh data.

---

## 7. Issues for Realistic Implementation

### 7.1 Technical Issues

**Issue 1: WASM Module Loading Time**: Large WASM modules may take significant time to download and compile on first load. Solution: Implement progressive loading with essential functions available immediately and advanced functions loading on demand. Cache compiled WASM in browser cache to avoid recompilation on return visits.

**Issue 2: Memory Fragmentation**: Repeated mesh loading and unloading can fragment WASM linear memory, eventually causing allocation failures. Solution: Implement a memory pooling strategy and periodic compaction. Provide clear memory usage indicators to users. Implement graceful degradation when memory pressure increases.

**Issue 3: Browser Tab Suspension**: When browser tabs are suspended (common on mobile), WASM processing pauses and connections may timeout. Solution: Implement aggressive checkpointing with checkpoints saved every 30 seconds. Detect tab suspension via visibility API and save state immediately. Implement longer connection timeouts when suspended tabs are detected.

**Issue 4: Coordinate System Ambiguity**: Different mesh formats use different coordinate systems and units, which may not be preserved in abstractions. Solution: Normalize all coordinates internally and track original units. Include unit information in abstraction. Provide user-selectable unit systems in results display.

**Issue 5: Large Result Payloads**: Detailed inspection results can be large, especially for complex meshes with many findings. Solution: Implement result compression before transmission. Use incremental result streaming for progress updates. Implement result pagination for display.

### 7.2 Operational Issues

**Issue 6: LLM Rate Limits and Costs**: LLM API calls are expensive and rate-limited, potentially creating bottlenecks for concurrent users. Solution: Implement request batching and caching of common inspection patterns. Use smaller models for initial request classification and routing. Implement usage-based pricing that accounts for LLM costs.

**Issue 7: Multi-Tenant Isolation**: Multiple users share cloud infrastructure, requiring strict isolation of session data. Solution: Implement session-scoped data isolation at the database level. Use encryption keys per session where sensitive data must be stored. Implement comprehensive audit logging for compliance.

**Issue 8: Browser Compatibility**: WASM and WebSocket support vary across browsers and versions. Solution: Implement feature detection on application load. Provide clear error messages when features are unavailable. Maintain a browser compatibility matrix and update as necessary.

**Issue 9: Error Propagation**: Errors in WASM processing must be communicated clearly to users without exposing technical details. Solution: Implement error classification (recoverable, retryable, fatal). Generate user-friendly error messages from technical error codes. Provide actionable guidance for error recovery.

### 7.3 Business Issues

**Issue 10: Pricing Model Uncertainty**: The architecture's cost structure differs from traditional SaaS, requiring careful pricing model design. Solution: Model costs per inspection based on mesh complexity and LLM usage. Consider tiered pricing based on feature access. Implement usage tracking and cost transparency for users.

**Issue 11: Liability for Inspection Accuracy**: If the inspection system misses defects that cause downstream problems, liability questions arise. Solution: Clearly document inspection limitations and intended use cases. Implement confidence scores in results. Provide audit trails for inspection decisions.

**Issue 12: Competitive Differentiation**: The architecture could be replicated by competitors, requiring continuous innovation. Solution: Focus on inspection algorithm quality and coverage as core differentiators. Build domain-specific expertise for target industries. Invest in user experience and workflow integration.

---

## 8. Implementation Roadmap

### 8.1 Phase 1: Foundation (Months 1-3)

**Objectives**: Demonstrate core WASM mesh processing capability and basic browser-cloud communication.

**Deliverables**:
- WASM mesh processor with STL/OBJ loading and basic topology analysis
- Simple web application with mesh upload and basic result display
- Cloud service with session management and MCP message routing
- Streamable HTTP transport implementation

**Success Criteria**:
- Users can upload meshes up to 10M polygons
- Basic topology analysis completes within 60 seconds
- Cloud session establishes reliably on first attempt
- No mesh data leaves the browser

### 8.2 Phase 2: Intelligence (Months 4-6)

**Objectives**: Add LLM integration for natural language inspection and report generation.

**Deliverables**:
- Natural language description processing
- Geometric abstraction layer implementation
- LLM-based inspection workflow orchestration
- Report generation with visualizations

**Success Criteria**:
- Users can request inspections using natural language
- System interprets ambiguous descriptions correctly
- Generated reports are accurate and actionable
- Abstraction provides sufficient context without geometry disclosure

### 8.3 Phase 3: Resilience (Months 7-9)

**Objectives**: Add reliability features for production deployment.

**Deliverables**:
- Checkpoint and recovery system
- Connection resilience with automatic reconnection
- Offline basic functionality mode
- Comprehensive error handling and user feedback

**Success Criteria**:
- Inspections survive browser tab close/reopen
- Connection failures handled gracefully with user notification
- Basic inspection available without cloud connectivity
- Error messages are clear and actionable

### 8.4 Phase 4: Scale (Months 10-12)

**Objectives**: Optimize for production scale and performance.

**Deliverables**:
- Horizontal scaling architecture for cloud infrastructure
- Performance optimization for large meshes
- Advanced inspection algorithms
- Enterprise features (SSO, audit logging, API access)

**Success Criteria**:
- System handles 100+ concurrent users without degradation
- Meshes up to 50M polygons process smoothly
- Enterprise customers can integrate via API
- Compliance requirements met for target industries

---

## 9. Conclusion

### 9.1 Summary of Key Findings

The distributed mesh inspection architecture represents a viable approach to privacy-preserving mesh analysis, enabling sophisticated AI-driven inspection without compromising design confidentiality. The core innovation of running mesh processing as an MCP server in the browser while the cloud service acts as an MCP client achieves the stated goals of keeping mesh data local while providing intelligent inspection capabilities.

However, successful implementation requires addressing several significant technical challenges. The connection establishment architecture must use relay-based communication or HTTP transport rather than direct WebSocket connections. The LLM context challenge requires a well-designed geometric abstraction layer that provides sufficient information for meaningful assistance without enabling geometry reconstruction. Session management must account for the distributed nature of state across client and server. The WASM resource constraints require tiered processing strategies for meshes exceeding browser memory limits.

### 9.2 Recommendations

The following recommendations should guide implementation efforts:

First, adopt Streamable HTTP as the primary transport mechanism, with WebSocket relay as an optional enhancement for real-time use cases. Second, invest heavily in the geometric abstraction layer, as this component determines both the utility of LLM assistance and the strength of the privacy guarantee. Third, design for failure from the outset, implementing comprehensive checkpointing, graceful degradation, and clear error communication. Fourth, validate privacy properties formally through threat modeling and penetration testing before production deployment. Fifth, engage with target industry users early to validate that the abstraction provides sufficient context for their inspection needs.

### 9.3 Open Questions

The following questions require further clarification before implementation can proceed:

First, what specific mesh formats must be supported, and what is the priority order for format coverage? Second, what inspection operations constitute the minimum viable feature set for initial deployment? Third, how should the system handle meshes that exceed local processing capability while maintaining the privacy guarantee? Fourth, what regulatory requirements apply to the target industries, and how do these requirements affect data handling and retention? Fifth, what is the expected usage pattern (single inspections versus complex workflows) that should drive session management design?

These questions should be addressed through user research, technical investigation, and business analysis before proceeding to detailed implementation.

---

## Appendix A: Glossary

| Term | Definition |
|------|------------|
| **MCP Server** | In this architecture, the WASM module exposing mesh processing tools via MCP protocol |
| **MCP Client** | In this architecture, the cloud service orchestrating inspection workflows via MCP |
| **Geometric Abstraction** | Structured summary of mesh properties excluding raw coordinates |
| **WASM** | WebAssembly, a binary instruction format for browser-based computation |
| **IndexedDB** | Browser's native document-oriented database for local storage |
| **Streamable HTTP** | MCP transport using HTTP POST with chunked transfer encoding for streaming |
| **Relay Server** | Cloud component forwarding WebSocket messages between browser and cloud |

## Appendix B: Reference Architecture Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **WASM Language** | Rust | Memory safety, performance, mature ecosystem |
| **Primary Transport** | Streamable HTTP | Browser-compatible, standard infrastructure |
| **Real-Time Transport** | WebSocket with Relay | When low latency required |
| **Session Storage** | Redis (cloud), IndexedDB (client) | Appropriate for each tier's characteristics |
| **LLM Integration** | API-based external service | Flexibility, cost management |
| **Mesh Formats** | STL, OBJ (Phase 1) | Widest support, simple parsing |

---

*Document Version: 1.0*
*Classification: Internal Design Document*
*Last Updated: March 2026*
