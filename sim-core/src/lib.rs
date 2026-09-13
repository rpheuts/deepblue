pub mod domain;
pub mod state;
pub mod solver;

#[cfg(test)]
mod tests {
    use crate::domain::SimDomainDescriptor;
    use crate::state::DoubleBufferedGrid;
    use crate::solver::swe::step_swe;

    #[test]
    fn test_lake_at_rest() {
        let mut desc = SimDomainDescriptor::default();
        desc.grid_res_x = 10;
        desc.grid_res_y = 10;
        
        let mut grid = DoubleBufferedGrid::new(desc);
        
        // Setup uneven bathymetry (z_bed) but flat water surface (eta)
        let target_eta = 10.0;
        for y in 0..desc.grid_res_y {
            for x in 0..desc.grid_res_x {
                let idx = grid.current.idx(x, y);
                // Bumpy bed
                let z = (x as f32).sin() + (y as f32).cos();
                grid.current.z_bed[idx] = z;
                // Water depth fills up to target_eta
                grid.current.h[idx] = target_eta - z;
            }
        }
        
        // Run simulation for a few steps
        let dt = 0.01;
        for _ in 0..10 {
            step_swe(&mut grid, dt);
        }
        
        // Assert velocities remain close to zero (allowing minor float drift)
        for y in 1..(desc.grid_res_y - 1) {
            for x in 1..(desc.grid_res_x - 1) {
                let idx = grid.current.idx(x, y);
                assert!(grid.current.u[idx].abs() < 1e-3, "Velocity u generated at ({},{}): {}", x, y, grid.current.u[idx]);
                assert!(grid.current.v[idx].abs() < 1e-3, "Velocity v generated at ({},{}): {}", x, y, grid.current.v[idx]);
            }
        }
    }

    #[test]
    fn test_mass_conservation() {
        let mut desc = SimDomainDescriptor::default();
        desc.grid_res_x = 20;
        desc.grid_res_y = 20;
        
        let mut grid = DoubleBufferedGrid::new(desc);
        
        // Initialize a block of water in the middle
        let mut initial_mass = 0.0;
        for y in 5..15 {
            for x in 5..15 {
                let idx = grid.current.idx(x, y);
                grid.current.h[idx] = 5.0;
                initial_mass += 5.0;
            }
        }
        
        let dt = 0.005;
        for _ in 0..50 {
            step_swe(&mut grid, dt);
        }
        
        // Calculate final mass
        let mut final_mass = 0.0;
        for y in 0..desc.grid_res_y {
            for x in 0..desc.grid_res_x {
                let idx = grid.current.idx(x, y);
                final_mass += grid.current.h[idx];
            }
        }
        
        // Assert total mass is strictly conserved (with f32 tolerance)
        let diff = (initial_mass - final_mass).abs();
        assert!(diff < 0.1, "Mass not conserved! Initial: {}, Final: {}, Diff: {}", initial_mass, final_mass, diff);
    }
}
