# DeepBlue

A physically grounded hydraulic sandbox simulation platform focused on granular-fluid interactions (water carving, pooling, damming, and sediment transport) inspired by beach stream management.

## Architecture

DeepBlue is designed around decoupled layers to support multiple hardware tiers and renderer frontends:

- **`sim-core`**: Pure physics and mathematical kernels (2D Shallow Water Equations with well-balanced hydrostatic reconstruction, conservative Exner equation scaffolding). Zero graphics API dependencies.
- **`sim-backend`**: High-performance compute execution layer powered by `wgpu` and WGSL compute shaders using a double-buffered ping-pong storage architecture.
- **`sim-client-2d`**: Lightweight 2D debug visualizer built with `macroquad` for inspecting fluid height and flow dynamics in real time.

## Getting Started

### Prerequisites

- Rust toolchain (2021 edition)
- Vulkan runtime / graphics drivers (e.g. `mesa-vulkan-drivers`) and standard X11 development libraries if running on Linux

### Running the Visualizer

To run the real-time 2D GPU-accelerated simulation client:

```bash
cargo run --release -p sim-client-2d
```

### Running Tests

To verify physical conservation properties (such as Lake at Rest and strict mass conservation) in the headless core engine:

```bash
cargo test -p sim-core
```

## Roadmap

Detailed architectural specs, continuum mechanics formulation, and roadmap milestones are documented in [`AGENTS.md`](./AGENTS.md).
