pub mod domain;
pub mod state;
pub mod solver;
pub mod backend;
pub mod scenario;

pub use backend::{SimulationBackend, CpuSimulator};
pub use scenario::Scenarios;

#[cfg(test)]
mod tests {
    use crate::domain::SimDomainDescriptor;
    use crate::state::DoubleBufferedGrid;
    use crate::solver::swe::{step_swe, step_swe_with_params, SweParams};

    #[test]
    fn test_lake_at_rest() {
        let mut desc = SimDomainDescriptor::default();
        desc.grid_res_x = 16;
        desc.grid_res_y = 16;
        
        let mut grid = DoubleBufferedGrid::new(desc);
        
        // Setup uneven bathymetry (z_bed) with flat water surface (eta = 10.0)
        let target_eta = 10.0;
        for y in 0..desc.grid_res_y {
            for x in 0..desc.grid_res_x {
                let idx = grid.current.idx(x, y);
                let z = (x as f32 * 0.5).sin() + (y as f32 * 0.5).cos();
                grid.current.z_bed[idx] = z;
                grid.current.h[idx] = target_eta - z;
            }
        }
        
        // Run simulation for 20 steps
        let dt = 0.01;
        for _ in 0..20 {
            step_swe(&mut grid, dt);
        }
        
        // Assert velocities remain close to zero across the entire domain
        for y in 1..(desc.grid_res_y - 1) {
            for x in 1..(desc.grid_res_x - 1) {
                let idx = grid.current.idx(x, y);
                assert!(grid.current.u[idx].abs() < 1e-3, "Velocity u generated at ({},{}): {}", x, y, grid.current.u[idx]);
                assert!(grid.current.v[idx].abs() < 1e-3, "Velocity v generated at ({},{}): {}", x, y, grid.current.v[idx]);
            }
        }
    }

    #[test]
    fn test_mass_conservation_strict_sloshing() {
        let mut desc = SimDomainDescriptor::default();
        desc.grid_res_x = 24;
        desc.grid_res_y = 24;
        desc.extent_x = 24.0;
        desc.extent_y = 24.0;
        
        let mut grid = DoubleBufferedGrid::new(desc);
        
        // Place water column in an off-center position to trigger high-velocity sloshing into walls
        for y in 2..8 {
            for x in 2..8 {
                let idx = grid.current.idx(x, y);
                grid.current.h[idx] = 4.0;
            }
        }
        
        grid.apply_reflective_boundaries();
        let initial_mass = grid.current.interior_mass();
        assert!(initial_mass > 0.0);
        
        // Step the simulation for 100 ticks, allowing waves to reflect multiple times off the boundary
        let dt = 0.005;
        for _ in 0..100 {
            step_swe(&mut grid, dt);
        }
        
        let final_mass = grid.current.interior_mass();
        let diff = (initial_mass - final_mass).abs();
        
        // Strict conservation check: difference must be near floating-point roundoff
        assert!(
            diff < 1e-4,
            "Mass not strictly conserved! Initial: {}, Final: {}, Diff: {}",
            initial_mass,
            final_mass,
            diff
        );
    }

    #[test]
    fn test_dam_break_ritter_benchmark() {
        let mut desc = SimDomainDescriptor::default();
        desc.grid_res_x = 100;
        desc.grid_res_y = 8;
        desc.extent_x = 100.0; // 1 meter per cell
        desc.extent_y = 8.0;
        
        let mut grid = DoubleBufferedGrid::new(desc);
        let h0 = 4.0f32;
        let dam_x = 50; // Dam located at index 50
        
        // Upstream reservoir: h = h0 for x <= 50; Downstream: dry (h = 0)
        for y in 0..desc.grid_res_y {
            for x in 0..desc.grid_res_x {
                let idx = grid.current.idx(x, y);
                if x <= dam_x {
                    grid.current.h[idx] = h0;
                } else {
                    grid.current.h[idx] = 0.0;
                }
            }
        }
        
        grid.apply_reflective_boundaries();
        let initial_mass = grid.current.interior_mass();
        
        // Run frictionless (Manning n = 0.0) dam break simulation for t = 1.0s
        let params = SweParams {
            manning_n: 0.0,
            h_dry: 1e-4,
            gravity: 9.81,
        };
        
        let dt = 0.01;
        let steps = 100; // 1.0 second total
        for _ in 0..steps {
            step_swe_with_params(&mut grid, dt, &params);
        }
        
        // 1. Strict mass conservation during dam break
        let final_mass = grid.current.interior_mass();
        let diff = (initial_mass - final_mass).abs();
        assert!(diff < 1e-3, "Mass leaked during dam break! Diff: {}", diff);
        
        // 2. Analytical Ritter solution comparison at breach location (interface between dam_x and dam_x + 1):
        // Analytical Ritter depth at breach interface: h(x0, t) = 4/9 * h0 = ~1.7778 m
        let mid_y = desc.grid_res_y / 2;
        let breach_left_idx = grid.current.idx(dam_x, mid_y);
        let breach_right_idx = grid.current.idx(dam_x + 1, mid_y);
        let breach_interface_h = (grid.current.h[breach_left_idx] + grid.current.h[breach_right_idx]) * 0.5;
        let theoretical_h = (4.0 / 9.0) * h0;
        
        // First-order Rusanov numerical schemes exhibit diffusion on sharp fronts; check within 15% of analytical value
        let relative_error = (breach_interface_h - theoretical_h).abs() / theoretical_h;
        assert!(
            relative_error < 0.15,
            "Dam break breach depth diverged from Ritter analytical solution! Computed: {}, Expected: {}, Rel Error: {}",
            breach_interface_h,
            theoretical_h,
            relative_error
        );
        
        // 3. Shock front propagation: water has advanced past dam_x with positive forward velocity
        let downstream_idx = grid.current.idx(dam_x + 5, mid_y);
        assert!(grid.current.h[downstream_idx] > 0.1, "Wave front did not advance downstream!");
        assert!(grid.current.u[downstream_idx] > 0.0, "Downstream velocity should be directed forward!");
    }

    #[test]
    fn test_cpu_simulator_backend_trait() {
        use crate::backend::{CpuSimulator, SimulationBackend};

        let mut desc = SimDomainDescriptor::default();
        desc.grid_res_x = 20;
        desc.grid_res_y = 20;
        let mut grid = DoubleBufferedGrid::new(desc);

        for y in 5..15 {
            for x in 5..15 {
                let idx = grid.current.idx(x, y);
                grid.current.h[idx] = 2.0;
            }
        }

        // Test through trait object
        let mut sim: Box<dyn SimulationBackend> = Box::new(CpuSimulator::new(grid));
        assert_eq!(sim.backend_name(), "CPU Reference (sim-core)");

        let initial_mass = sim.total_fluid_mass();
        assert!(initial_mass > 0.0);

        // Subdivided step
        sim.step_subdivided(0.1, 0.01);
        sim.sync_to_cpu();

        let final_mass = sim.total_fluid_mass();
        let diff = (initial_mass - final_mass).abs();
        assert!(diff < 1e-3, "Mass not conserved through trait object! Diff: {}", diff);
    }
}

