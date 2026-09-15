use sim_core::backend::SimulationBackend;
use sim_core::domain::SimDomainDescriptor;
use sim_core::state::GridState;
use crate::simulator::WgpuSimulator;

impl SimulationBackend for WgpuSimulator {
    fn backend_name(&self) -> &'static str {
        "GPU (wgpu/WGSL)"
    }

    fn descriptor(&self) -> &SimDomainDescriptor {
        &self.cpu_grid.descriptor
    }

    fn step(&mut self, dt: f32) {
        self.step(dt);
    }

    fn step_subdivided(&mut self, total_dt: f32, max_sub_dt: f32) {
        self.step_subdivided(total_dt, max_sub_dt);
    }

    fn sync_to_cpu(&mut self) {
        self.sync_to_cpu();
    }

    fn current_state(&self) -> &GridState {
        &self.cpu_grid.current
    }

    fn current_state_mut(&mut self) -> &mut GridState {
        &mut self.cpu_grid.current
    }

    fn previous_state(&self) -> &GridState {
        &self.cpu_grid.next
    }

    fn upload_state(&mut self) {
        self.upload_state();
    }

    fn upload_water_depth(&mut self) {
        self.upload_water_depth();
    }

    fn compute_max_stable_dt(&self, cfl: f32) -> f32 {
        let dx = self.cpu_grid.descriptor.extent_x / self.cpu_grid.descriptor.grid_res_x as f32;
        let dy = self.cpu_grid.descriptor.extent_y / self.cpu_grid.descriptor.grid_res_y as f32;
        sim_core::solver::swe::compute_max_stable_dt(&self.cpu_grid.current, dx, dy, cfl, 9.81)
    }

    fn boundaries(&self) -> &sim_core::boundary::DomainBoundaryConfig {
        &self.cpu_grid.boundaries
    }

    fn set_boundaries(&mut self, boundaries: sim_core::boundary::DomainBoundaryConfig) {
        self.cpu_grid.boundaries = boundaries;
    }

    fn sim_time(&self) -> f32 {
        self.cpu_grid.time
    }

    fn set_stream_inflow(&mut self, active: bool) {
        self.stream_inflow_active = active;
    }

    fn set_coastal_sink(&mut self, active: bool) {
        self.coastal_sink_active = active;
    }
}
