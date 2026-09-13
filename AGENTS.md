```markdown
# AGENTS.md — Hydraulic Sandbox Simulation Platform

## Project Mission & Context
The goal is to build an extensible, physically grounded hydraulic simulation platform focused on granular-fluid interactions (water carving, damming, pooling, and sediment transport) inspired by beach stream management. 

The primary target is capturing the physical dynamics of carving channels, holding water in reservoirs, managing overflow spills, and routing flow. 

The platform must prioritize **accurate continuum mechanics**, **clean domain abstractions**, and **hardware scalability** before building specific game mechanics.

---

## Architecture Principles

### 1. Decoupled Core vs. Presentation (Renderer Agnostic)
* **Simulation Core (`sim-core`):** Pure physics, mathematical models, state containers, and step integrators. Must compile headless with **zero** graphics API, windowing, or game engine dependencies.
* **Compute Backend (`sim-backend`):** Platform-agnostic execution abstraction. Backends implement a unified simulation interface:
  * Reference CPU backend (for deterministic CI validation and unit tests).
  * GPU compute backend (WGSL/HLSL/SPIR-V for APIs like `wgpu`, Metal, or Vulkan).
* **Render Adapters (`render-adapter`):** The simulation state serves as a passive data producer. Any renderer (2D orthographic debug view, 2.5D isometric view, or a full modern 3D deferred/forward+ engine) consumes simulation textures as read-only inputs.
* **Client / Application Layer:** User input, tool brushes (digging, banking, dams), UI, and future game loops.

### 2. Multi-Scale Fidelity & Scalability Target
The architecture must scale across hardware tiers without altering core simulation math:
* **Mobile / Integrated Tier:** Fixed $512 \times 512$ to $1024 \times 1024$ grid, single-pass approximations, lower temporal sub-steps.
* **Desktop High-End Tier:** $2048 \times 2048$ to $4096 \times 4096+$ grid, adaptive sub-stepping, multi-layer soil saturation, full hydrodynamic advection.
* **Decoupled Grid Geometry:** Physical world dimensions ($L_x, L_y$ in meters) are decoupled from discretization resolution ($\Delta x, \Delta y$).

### 3. Extensible Material & Device Model
Do not hard-code "sand" and "water." Design the state fields so additional materials and active hydraulic devices can be injected cleanly:
* **Soil/Bed Stratigraphy:** Support layered material properties (e.g., bedrock, compacted clay, loose sand, gravel) defined by friction coefficients, cohesion, and critical shear stress ($\tau_c$).
* **Active Devices (Pumps, Gates, Siphons):** Represented as boundary source/sink terms or external flux constraints in the PDE solver passes.

---

## Technology Stack Choices

* **Language:** Rust (for memory safety, C-ABI FFI capabilities on mobile, and `no_std` purity).
* **Workspace Structure:**
  * `sim-core`: Pure math, hardware-agnostic library.
  * `sim-client-2d`: A lightweight PC debug visualizer built with `macroquad`.
* **Compute / Rendering Backend (Phase 2):** `wgpu` with `WGSL` compute shaders.
* **Mobile Strategy:** The decoupled `sim-core` can be compiled via NDK for Android (`.so` + JNI/uniffi) and as a static library for iOS, allowing integration with native Kotlin/Swift clients or Rust game engines like Bevy later.

---

## Core Simulation Specifications

The platform is standardized around a **2.5D Multilayer Hydrodynamic Formulation**:

1. **Hydrodynamics (2D Shallow Water Equations / SWE):**
   * Conservative variables: Water depth $h$ and discharge/momentum fluxes $uh, vh$.
   * Surface elevation: $\eta = z_{bed} + h$.
   * Governing equations: Saint-Venant equations with bed slope source terms and Manning friction drag.
2. **Sediment Dynamics (Exner Equation & Sediment Transport):**
   * Capacity calculation: Formulations based on local shear velocity $u_*$ (e.g., Grass, Shields, or Engelund-Hansen).
   * Exchange terms: Non-equilibrium pickup ($E$) and deposition ($D$) rates:
     $$\frac{\partial z_{bed}}{\partial t} = \frac{D - E}{1 - p}$$
     where $p$ is sediment porosity.
   * Advection of suspended sediment concentration ($C$) with fluid velocity field.
3. **Geotechnical Stability (Angle of Repose & Talus Transport):**
   * Dynamic repose limit: Differentiates dry sand ($\sim 34^\circ$), saturated sand (low cohesion/liquefaction risk), and damp capillary sand (higher temporary cohesion).
   * Multi-directional iterative slope-collapse filter to eliminate non-physical vertical shears.

---

## 3D Rendering Contract & Future-Proofing Requirements

To prevent early 2D architectural decisions from creating tech debt for future 3D renderers, all agents must adhere to the following contracts:

### 1. Standard Physical Coordinate & World Metadata
All spatial data in `sim-core` must map directly to SI units (meters, seconds, kilograms). Do not use unitless "pixel space" for physics state. Every simulation instance exposes a fixed world bounds descriptor:

```rust
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct SimDomainDescriptor {
    pub extent_x: f32,       // Domain width in real-world meters
    pub extent_y: f32,       // Domain length in real-world meters
    pub max_elevation: f32,  // Maximum allowable z_bed (meters)
    pub grid_res_x: u32,     // Discrete simulation grid width
    pub grid_res_y: u32,     // Discrete simulation grid height
    pub world_origin: [f32; 3], // 3D world space anchor [X, Y, Z]
}

```

### 2. Exportable GPU State Buffer Layout

The simulation state must be stored in standard GPU texture formats suitable for direct sampling by vertex displacement and fragment shaders. Keep states separated into predictable 2D texture formats:

| Texture Channel | Recommended Format | Purpose in 3D Renderer |
| --- | --- | --- |
| **`ElevationMap`** (`R32_FLOAT`) | $z_{bed}$ | Displaces terrain mesh vertices in world $Z$ (or $Y$). |
| **`WaterMap`** (`RG32_FLOAT` or `RGBA16_FLOAT`) | $(h, \eta)$ | Displaces water surface mesh vertices; computes light absorption. |
| **`VelocityMap`** (`RG16_FLOAT`) | $(u, v)$ | Drives flow-map UV advection, foam generation, and wave normal distortion. |
| **`SedimentWetnessMap`** (`RG16_FLOAT`) | $(C, W_{sat})$ | Albedo tinting (mud/sand), subsurface scattering, and PBR roughness maps. |

### 3. Spatial Continuity for Normal Reconstruction

* **Never introduce cell-indexing artifacts:** Solvers must maintain $C^0$ continuity in heightfield data. Sudden 1-cell discontinuous spikes break 3D lighting normals and cause mesh tearing.
* **Central Difference Normals:** Ensure outer grid boundaries are padded (1-cell ghost/halo ring) so a downstream 3D vertex/fragment shader can cleanly evaluate finite difference gradients without bounds-checking branches:

$$\vec{N} = \text{normalize}\left( -\frac{\partial z}{\partial x}, -\frac{\partial z}{\partial y}, 1 \right)$$



### 4. Temporal Decoupling (Tick vs. Frame)

* **Never tie simulation sub-steps to the display refresh rate.**
* The simulation runs at a deterministic, fixed frequency (e.g., 60 Hz or 120 Hz fixed $\Delta t$).
* `sim-core` must preserve `State_{current}` and `State_{previous}` (double-buffered) so a 3D renderer can perform hardware frame interpolation ($\alpha \in [0.0, 1.0]$) to render at 144 Hz+ without stutter.

### 5. Boundary for 3D Hybrid VFX

A 2.5D heightfield cannot natively represent overhangs, droplet sprays, or aeration. The simulation core must output diagnostic scalar triggers for external 3D visual effects:

* **Wave Breaking / Spray Metric:** Expose kinetic divergence $\nabla \cdot \vec{v}$ and temporal rise rate $\frac{\partial h}{\partial t}$ in a secondary channel so a 3D particle system (e.g., GPU Niagara/VFX) can spawn mist/foam particles without mutating physical mass.

---

## Agent Guidelines & Coding Standards

When generating, refactoring, or reviewing code for this repository, follow these rules:

### Conservation & Stability Constraints

* **Strict Mass Conservation:** Mass loss or gain is a bug. Water volume ($\sum h$) and total solid volume ($\sum z_{bed} + \sum \frac{C \cdot h}{1-p}$) must remain invariant across closed boundaries.
* **CFL Condition Adherence:** Integrators must verify or enforce the Courant-Friedrichs-Lewy condition:

$$\Delta t \le \text{CFL} \cdot \frac{\Delta x}{\sqrt{g \cdot h} + |\vec{v}|}$$



Support adaptive sub-stepping to prevent solver explosions when flow velocities spike during dam breaches.
* **Positivity Preserving:** Water depth $h \ge 0$ must be guaranteed everywhere. Implement wet/dry front reconstruction thresholds (e.g., $h_{dry} \approx 10^{-4}\,\text{m}$) to prevent division-by-zero errors in velocity computations ($u = \frac{uh}{h}$).

### Codebase Organization & Purity

* **Keep Core Math Stateless & Pure:** Fluid solvers must receive state buffers (in/out) and step parameters ($\Delta t, \Delta x, g, \nu$), avoiding hidden global state.
* **Double Buffering:** All iterative stencil operations must be ping-ponged or double-buffered to guarantee execution order independence and race-condition immunity on parallel backends.
* **No Engine Coupling in `sim-core`:** Avoid embedding graphics engine runtime types, platform window handles, or input event systems inside simulation modules.

### Verification & Testing Expectations

* Every new simulation kernel or solver component must be accompanied by an automated, headless test:
* **Lake at Rest Test:** Flat water over an uneven bathymetry must remain static with zero artificial velocity generation.
* **Dam-Break Benchmark:** Classical 1D/2D dam-break shock front comparison against analytical Ritter or Stoker solutions.
* **Sediment Conservation Test:** Total system sediment mass must be strictly conserved before and after erosion/deposition cycles.



---

## Roadmap

* [x] **Phase 1: Mathematical Scaffolding & CPU Reference Integrator**
* [x] Flat 2D grid representation with double-buffered memory layout.
* [x] 2D Shallow Water Equation solver (Kurganov-Petrova or central-upwind scheme).
* [x] Unit tests validating mass conservation and "Lake at Rest."


* [x] **Phase 2: GPU Compute Pipeline Integration**
* [x] Translate CPU PDE kernel into compute shaders (WGSL / HLSL).
* [x] Benchmark throughput across different grid densities ($512^2$ to $2048^2$).
* [x] Implement standard texture binding contract (`ElevationMap`, `WaterMap`, `VelocityMap`).


* [x] **Phase 3: Sediment Transport & Dynamic Bed Mechanics**
* [x] Implement Exner mass conservation and capacity formulations.
* [x] Add multi-neighbor talus/angle-of-repose relaxation pass.
* [x] Saturation tracking ($W_{sat}$) for soil stability and roughness shading.
* [x] Non-erodible bedrock constraint ($z_{bed} \ge z_{bedrock}$) preventing infinite scour.
* [x] GPU compute shaders for coupled hydro-sediment dynamics (`sediment_exner.wgsl`, `sediment_sat.wgsl`, `sediment_talus.wgsl`, `sediment_boundary.wgsl`).

* [x] **Optimization & Hardware Scalability Milestone**
* [x] Selective host-to-device bus synchronization (`upload_water_depth` reducing PCIe bandwidth by 85%).
* [x] Single-encoder subdivided compute pass batching eliminating driver submission bubbles.
* [x] Parallel row-level image blitting via Rayon with cached finite-difference hillshading.
* [x] Throttled HUD telemetry to eliminate 1.3M redundant cell iterations per frame.
* [x] Validated sustained 110–120 FPS throughput on mobile/integrated GPU hardware tiers.

* [x] **Phase 4: Tooling, Boundary Conditions & Coastal Interaction**
* [x] Formal boundary conduit abstraction in `sim-core` & `sim-backend` (reflective walls, non-reflecting radiation outflows, constant inflows, and Stokes swell/wave generators with tidal oscillations).
* [x] GPU-accelerated boundary pass in `swe.wgsl` supporting time-dependent Stokes wave crests and backwash absorption.
* [x] Non-erodible structures & stone breakwaters (`bedrock_z == z_bed`) preventing scour and talus slump while deflecting wave fronts.
* [x] Coastal beach & sandcastle scenario (`Scenarios::beach_sandcastle_waves`) featuring a sloping swash zone, erodible sandcastle with curtain ramparts, corner towers, and surrounding moat, plus a stone jetty.
* [x] Interactive stone masonry placement brush ([Ctrl+LMB]) and stone demolition brush ([Ctrl+RMB]), complementing sand dams ([Shift+LMB]), digging ([RMB]), and fluid addition ([LMB]).

* [ ] **Phase 5: Render Adapter Prototypes**
* [x] 2D orthographic debug canvas (`sim-client-2d`) with depth gradients, turbidity, foam, directional hillshade, velocity vectors, Lagrangian tracer particles, and real-time coastal wave/tide telemetry.
* [ ] 3D vertex-displaced mesh proof-of-concept consuming exported simulation textures (`ElevationMap`, `WaterMap`, `VelocityMap`).
