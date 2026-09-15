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
        if total_dt <= 1e-6 {
            return;
        }
        let steps = (total_dt / max_sub_dt).ceil().max(1.0) as usize;
        let dt = total_dt / steps as f32;
        for _ in 0..steps {
            self.step(dt);
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

    /// Uploads only fluid depth modifications (`h`) to device compute buffers.
    ///
    /// For continuous sources, rain, or tidal boundaries where bed elevation and velocity
    /// are not altered by the host, this avoids re-uploading all storage buffers across the bus.
    fn upload_water_depth(&mut self) {
        self.upload_state();
    }

    /// Calculates the total fluid mass in the active physical domain.
    fn total_fluid_mass(&self) -> f32 {
        self.current_state().interior_mass()
    }

    /// Calculates the total solid sediment mass (bed + suspended) in the active physical domain.
    fn total_sediment_mass(&self) -> f32 {
        self.current_state().interior_sediment_mass(0.40)
    }

    /// Evaluates the Courant-Friedrichs-Lewy (CFL) condition across the active domain
    /// and returns the maximum allowable stable time step.
    fn compute_max_stable_dt(&self, cfl: f32) -> f32;

    /// Returns the current boundary configuration.
    fn boundaries(&self) -> &crate::boundary::DomainBoundaryConfig;

    /// Updates the boundary configuration.
    fn set_boundaries(&mut self, boundaries: crate::boundary::DomainBoundaryConfig);

    /// Returns the total elapsed physical simulation time (in seconds).
    fn sim_time(&self) -> f32;

    /// Enables or disables continuous stream inflow injection at the northern source zone.
    fn set_stream_inflow(&mut self, _active: bool) {}

    /// Enables or disables the southern coastal absorption sink.
    fn set_coastal_sink(&mut self, _active: bool) {}
}

/// Headless reference CPU simulation backend.
///
/// Designed for deterministic CI validation, numerical verification, unit testing,
/// and low-spec/no-GPU fallback environments.
pub struct CpuSimulator {
    pub grid: DoubleBufferedGrid,
    pub swe_params: SweParams,
    pub sediment_params: crate::solver::sediment::SedimentParams,
    pub enable_sediment: bool,
    pub stream_inflow_active: bool,
    pub coastal_sink_active: bool,
}

impl CpuSimulator {
    /// Creates a new CPU simulator with default physical parameters.
    pub fn new(mut grid: DoubleBufferedGrid) -> Self {
        grid.apply_boundaries();
        Self {
            grid,
            swe_params: SweParams::default(),
            sediment_params: crate::solver::sediment::SedimentParams::default(),
            enable_sediment: true,
            stream_inflow_active: false,
            coastal_sink_active: false,
        }
    }

    /// Creates a new CPU simulator with custom SWE physical parameters.
    pub fn with_params(mut grid: DoubleBufferedGrid, swe_params: SweParams) -> Self {
        grid.apply_boundaries();
        Self {
            grid,
            swe_params,
            sediment_params: crate::solver::sediment::SedimentParams::default(),
            enable_sediment: true,
            stream_inflow_active: false,
            coastal_sink_active: false,
        }
    }

    /// Creates a new CPU simulator with custom SWE and sediment physical parameters.
    pub fn with_sediment_params(
        mut grid: DoubleBufferedGrid,
        swe_params: SweParams,
        sediment_params: crate::solver::sediment::SedimentParams,
    ) -> Self {
        grid.apply_boundaries();
        Self {
            grid,
            swe_params,
            sediment_params,
            enable_sediment: true,
            stream_inflow_active: false,
            coastal_sink_active: false,
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
        step_swe_with_params(&mut self.grid, dt, &self.swe_params);
        if self.enable_sediment {
            crate::solver::sediment::step_sediment(&mut self.grid, dt, &self.sediment_params);
        }

        if self.stream_inflow_active {
            let width = self.grid.descriptor.grid_res_x as i32;
            let height = self.grid.descriptor.grid_res_y as i32;
            let inflow_x_center = (width as f32 * 0.54) as i32;
            let inflow_radius = (width as f32 * 0.05).max(4.0) as i32;
            let inflow_y_end = (height as f32 * 0.05).max(6.0) as i32;
            for y in 1..=inflow_y_end {
                for x in (inflow_x_center - inflow_radius)..=(inflow_x_center + inflow_radius) {
                    let idx = self.grid.current.idx(x as u32, y as u32);
                    let z = self.grid.current.z_bed[idx];
                    let target_h = (2.6 - z).max(0.65);
                    if self.grid.current.h[idx] < target_h {
                        self.grid.current.h[idx] = target_h;
                    }
                }
            }
        }

        if self.coastal_sink_active {
            let width = self.grid.descriptor.grid_res_x as i32;
            let height = self.grid.descriptor.grid_res_y as i32;
            let h_start = height - 12;
            for y in h_start..(height - 1) {
                for x in 1..(width - 1) {
                    let idx = self.grid.current.idx(x as u32, y as u32);
                    self.grid.current.h[idx] *= 0.82;
                }
            }
        }
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
        self.grid.apply_boundaries();
    }

    fn upload_water_depth(&mut self) {
        // No-op for CPU backend: memory is already updated directly in current.h
    }

    fn compute_max_stable_dt(&self, cfl: f32) -> f32 {
        let dx = self.grid.descriptor.extent_x / self.grid.descriptor.grid_res_x as f32;
        let dy = self.grid.descriptor.extent_y / self.grid.descriptor.grid_res_y as f32;
        compute_max_stable_dt(&self.grid.current, dx, dy, cfl, self.swe_params.gravity)
    }

    fn boundaries(&self) -> &crate::boundary::DomainBoundaryConfig {
        &self.grid.boundaries
    }

    fn set_boundaries(&mut self, boundaries: crate::boundary::DomainBoundaryConfig) {
        self.grid.boundaries = boundaries;
    }

    fn sim_time(&self) -> f32 {
        self.grid.time
    }

    fn set_stream_inflow(&mut self, active: bool) {
        self.stream_inflow_active = active;
    }

    fn set_coastal_sink(&mut self, active: bool) {
        self.coastal_sink_active = active;
    }
}
