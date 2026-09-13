use crate::domain::SimDomainDescriptor;
use crate::state::{DoubleBufferedGrid, GridState};
use crate::solver::swe::{step_swe_with_params, compute_max_stable_dt, SweParams};

/// Platform-agnostic execution abstraction for hydraulic simulation engines.
///
/// Both headless CPU reference integrators and hardware-accelerated GPU compute pipelines
/// implement this interface, enabling plug-and-play swapping across hardware tiers,
/// deterministic CI testing, and renderer decoupling.
pub trait SimulationBackend {
    /// Human-readable identifier of the compute backend (e.g., "CPU Reference", "GPU (wgpu/WGSL)").
    fn backend_name(&self) -> &'static str;

    /// Returns a reference to the spatial domain descriptor.
    fn descriptor(&self) -> &SimDomainDescriptor;

    /// Advances the simulation forward by a single time step `dt` (in seconds).
    fn step(&mut self, dt: f32);

    /// Advances the simulation over a duration of `total_dt` by subdividing into
    /// stable sub-steps of at most `max_sub_dt`.
    fn step_subdivided(&mut self, total_dt: f32, max_sub_dt: f32) {
        let mut remaining = total_dt;
        while remaining > 1e-6 {
            let dt = remaining.min(max_sub_dt);
            self.step(dt);
            remaining -= dt;
        }
    }

    /// Synchronizes the latest simulation state from the execution device (e.g. GPU VRAM)
    /// to host CPU memory. For CPU-based backends, this is a no-op.
    fn sync_to_cpu(&mut self);

    /// Returns an immutable reference to the current CPU grid state.
    ///
    /// For GPU backends, call `sync_to_cpu()` first to ensure host memory is up to date.
    fn current_state(&self) -> &GridState;

    /// Returns a mutable reference to the current CPU grid state.
    fn current_state_mut(&mut self) -> &mut GridState;

    /// Returns an immutable reference to the previous time step's grid state.
    /// Useful for temporal sub-frame interpolation: state(alpha) = lerp(previous, current, alpha).
    fn previous_state(&self) -> &GridState;

    /// Uploads modified host CPU grid state to device compute buffers.
    /// For CPU-based backends, this reapplies boundary conditions.
    fn upload_state(&mut self);

    /// Calculates the total fluid mass in the active physical domain.
    fn total_fluid_mass(&self) -> f32 {
        self.current_state().interior_mass()
    }

    /// Evaluates the Courant-Friedrichs-Lewy (CFL) condition across the active domain
    /// and returns the maximum allowable stable time step.
    fn compute_max_stable_dt(&self, cfl: f32) -> f32;
}

/// Headless reference CPU simulation backend.
///
/// Designed for deterministic CI validation, numerical verification, unit testing,
/// and low-spec/no-GPU fallback environments.
pub struct CpuSimulator {
    pub grid: DoubleBufferedGrid,
    pub params: SweParams,
}

impl CpuSimulator {
    /// Creates a new CPU simulator with default SWE physical parameters.
    pub fn new(mut grid: DoubleBufferedGrid) -> Self {
        grid.apply_reflective_boundaries();
        Self {
            grid,
            params: SweParams::default(),
        }
    }

    /// Creates a new CPU simulator with custom SWE physical parameters.
    pub fn with_params(mut grid: DoubleBufferedGrid, params: SweParams) -> Self {
        grid.apply_reflective_boundaries();
        Self {
            grid,
            params,
        }
    }
}

impl SimulationBackend for CpuSimulator {
    fn backend_name(&self) -> &'static str {
        "CPU Reference (sim-core)"
    }

    fn descriptor(&self) -> &SimDomainDescriptor {
        &self.grid.descriptor
    }

    fn step(&mut self, dt: f32) {
        step_swe_with_params(&mut self.grid, dt, &self.params);
    }

    fn sync_to_cpu(&mut self) {
        // No-op: CPU simulator memory is already on host
    }

    fn current_state(&self) -> &GridState {
        &self.grid.current
    }

    fn current_state_mut(&mut self) -> &mut GridState {
        &mut self.grid.current
    }

    fn previous_state(&self) -> &GridState {
        &self.grid.next
    }

    fn upload_state(&mut self) {
        self.grid.apply_reflective_boundaries();
    }

    fn compute_max_stable_dt(&self, cfl: f32) -> f32 {
        let dx = self.grid.descriptor.extent_x / self.grid.descriptor.grid_res_x as f32;
        let dy = self.grid.descriptor.extent_y / self.grid.descriptor.grid_res_y as f32;
        compute_max_stable_dt(&self.grid.current, dx, dy, cfl, self.params.gravity)
    }
}
