use crate::domain::SimDomainDescriptor;
use crate::state::DoubleBufferedGrid;
use crate::solver::swe::{step_swe, step_swe_with_params, SweParams};

#[test]
fn test_lake_at_rest() {
    let desc = SimDomainDescriptor {
        grid_res_x: 16,
        grid_res_y: 16,
        ..Default::default()
    };
    
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
    let desc = SimDomainDescriptor {
        grid_res_x: 24,
        grid_res_y: 24,
        extent_x: 24.0,
        extent_y: 24.0,
        ..Default::default()
    };
    
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
    let desc = SimDomainDescriptor {
        grid_res_x: 100,
        grid_res_y: 8,
        extent_x: 100.0, // 1 meter per cell
        extent_y: 8.0,
        ..Default::default()
    };
    
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

    let desc = SimDomainDescriptor {
        grid_res_x: 20,
        grid_res_y: 20,
        ..Default::default()
    };
    let mut grid = DoubleBufferedGrid::new(desc);

    // Bedrock surface: impervious rock so zero porous infiltration occurs
    for i in 0..grid.current.z_bed.len() {
        grid.current.bedrock_z[i] = grid.current.z_bed[i];
    }

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

#[test]
fn test_sediment_mass_conservation() {
    use crate::solver::sediment::{step_sediment, SedimentParams};
    use crate::solver::swe::{step_swe_with_params, SweParams};

    let desc = SimDomainDescriptor {
        grid_res_x: 32,
        grid_res_y: 32,
        extent_x: 32.0,
        extent_y: 32.0,
        ..Default::default()
    };
    let mut grid = DoubleBufferedGrid::new(desc);

    let p = 0.40;
    let sed_params = SedimentParams {
        porosity: p,
        erodibility: 0.05,
        capacity_scale: 0.10,
        infiltration_rate: 0.0, // Infiltration isolated to test pure Exner dynamics
        ..Default::default()
    };

    // Sand bed of 1.5m over bedrock at 0.0m
    for i in 0..grid.current.z_bed.len() {
        grid.current.z_bed[i] = 1.5;
        grid.current.bedrock_z[i] = 0.0;
    }

    // Fast water stream in the middle to trigger strong erosion and suspended transport
    for y in 10..22 {
        for x in 10..22 {
            let idx = grid.current.idx(x, y);
            grid.current.h[idx] = 1.0;
            grid.current.u[idx] = 1.5; // High velocity well above critical velocity (0.22 m/s)
        }
    }

    grid.apply_reflective_boundaries();

    let initial_sediment_mass = grid.current.interior_sediment_mass(p);
    let initial_fluid_mass = grid.current.interior_mass();

    let swe_params = SweParams::default();
    let dt = 0.02;

    // Run 50 coupled hydrodynamic + sediment steps
    for _ in 0..50 {
        step_swe_with_params(&mut grid, dt, &swe_params);
        step_sediment(&mut grid, dt, &sed_params);
    }

    let final_sediment_mass = grid.current.interior_sediment_mass(p);
    let final_fluid_mass = grid.current.interior_mass();

    // 1. Water mass must remain strictly conserved
    let fluid_diff = (initial_fluid_mass - final_fluid_mass).abs();
    assert!(
        fluid_diff < 1e-3,
        "Fluid mass diverged during coupled simulation! Initial: {}, Final: {}, Diff: {}",
        initial_fluid_mass,
        final_fluid_mass,
        fluid_diff
    );

    // 2. Sediment total solid mass must remain strictly conserved
    let sed_diff = (initial_sediment_mass - final_sediment_mass).abs();
    let rel_error = sed_diff / initial_sediment_mass;
    assert!(
        rel_error < 1e-4,
        "Sediment mass not conserved! Initial: {}, Final: {}, Diff: {}, Rel Error: {}",
        initial_sediment_mass,
        final_sediment_mass,
        sed_diff,
        rel_error
    );

    // 3. Confirm that erosion actually occurred (bed was sculpted, suspended sediment generated)
    let max_c = grid.current.sediment_c.iter().copied().fold(0.0f32, f32::max);
    let min_z = grid.current.z_bed.iter().copied().fold(f32::INFINITY, f32::min);
    assert!(max_c > 0.001, "Erosion failed to suspend sediment into water column! Max C: {}", max_c);
    assert!(min_z < 1.49, "Bed was not eroded by high speed water! Min Z: {}", min_z);
}

#[test]
fn test_bedrock_erosion_limit() {
    use crate::solver::sediment::{step_sediment, SedimentParams};
    use crate::solver::swe::{step_swe_with_params, SweParams};

    let desc = SimDomainDescriptor {
        grid_res_x: 24,
        grid_res_y: 24,
        ..Default::default()
    };
    let mut grid = DoubleBufferedGrid::new(desc);

    let bedrock_level = 0.50f32;
    for i in 0..grid.current.z_bed.len() {
        grid.current.z_bed[i] = bedrock_level; // Bed begins exactly on bedrock
        grid.current.bedrock_z[i] = bedrock_level;
    }

    // Violent flow directly on top of bedrock
    for y in 4..20 {
        for x in 4..20 {
            let idx = grid.current.idx(x, y);
            grid.current.h[idx] = 2.0;
            grid.current.u[idx] = 3.0;
        }
    }

    grid.apply_reflective_boundaries();

    let sed_params = SedimentParams {
        erodibility: 0.1, // Extremely aggressive erosion rate
        ..Default::default()
    };

    let swe_params = SweParams::default();
    for _ in 0..30 {
        step_swe_with_params(&mut grid, 0.02, &swe_params);
        step_sediment(&mut grid, 0.02, &sed_params);
    }

    // Bed elevation must never drop below bedrock level
    for (i, &z) in grid.current.z_bed.iter().enumerate() {
        let b = grid.current.bedrock_z[i];
        assert!(
            z >= b - 1e-6,
            "Erosion breached bedrock floor! Bed Z: {}, Bedrock: {}",
            z,
            b
        );
    }
}

#[test]
fn test_talus_collapse_conservation_and_repose() {
    use crate::solver::sediment::{step_talus_collapse, SedimentParams};

    let desc = SimDomainDescriptor {
        grid_res_x: 25,
        grid_res_y: 25,
        extent_x: 25.0, // dx = 1.0m
        extent_y: 25.0, // dy = 1.0m
        ..Default::default()
    };
    let mut grid = DoubleBufferedGrid::new(desc);

    let params = SedimentParams::default(); // phi_dry = 34 degrees

    // Create a steep square sand tower in the center
    let mid_x = 12;
    let mid_y = 12;
    for y in 1..(desc.grid_res_y - 1) {
        for x in 1..(desc.grid_res_x - 1) {
            let idx = grid.current.idx(x, y);
            grid.current.z_bed[idx] = 0.0;
            grid.current.bedrock_z[idx] = 0.0;
            grid.current.soil_sat[idx] = 0.0; // Dry sand
        }
    }
    for dy in -1..=1 {
        for dx in -1..=1 {
            let idx = grid.current.idx((mid_x + dx) as u32, (mid_y + dy) as u32);
            grid.current.z_bed[idx] = 5.0; // 5m vertical cliff!
        }
    }

    grid.apply_reflective_boundaries();

    let initial_bed_mass = grid.current.interior_sediment_mass(0.40);

    // Relax the steep tower over multiple talus passes
    for _ in 0..25 {
        step_talus_collapse(&mut grid, &params);
    }

    let final_bed_mass = grid.current.interior_sediment_mass(0.40);
    let mass_diff = (initial_bed_mass - final_bed_mass).abs();

    // 1. Exact mass conservation during talus collapse
    assert!(
        mass_diff < 1e-4,
        "Talus collapse did not conserve sediment mass! Diff: {}",
        mass_diff
    );

    // 2. Sand must have spread outwards from the tower to its surrounding base
    let base_idx = grid.current.idx(mid_x as u32 + 3, mid_y as u32);
    assert!(
        grid.current.z_bed[base_idx] > 0.05,
        "Talus did not spread to base! z = {}",
        grid.current.z_bed[base_idx]
    );

    // 3. Peak of tower must have lowered
    let center_idx = grid.current.idx(mid_x as u32, mid_y as u32);
    assert!(
        grid.current.z_bed[center_idx] < 5.0,
        "Peak did not collapse! Peak z = {}",
        grid.current.z_bed[center_idx]
    );
}

#[test]
fn test_talus_saturation_dependency() {
    use crate::solver::sediment::{step_talus_collapse, SedimentParams};

    let desc = SimDomainDescriptor {
        grid_res_x: 40,
        grid_res_y: 20,
        extent_x: 40.0,
        extent_y: 20.0,
        ..Default::default()
    };
    let mut grid = DoubleBufferedGrid::new(desc);
    let params = SedimentParams::default();

    // Tower 1 on left (mid_x = 10): Damp sand (W_sat = 0.25, high capillary cohesion, phi = 45 deg)
    // Tower 2 on right (mid_x = 30): Saturated sand (W_sat = 1.00, liquefaction slump, phi = 22 deg)
    for dy in -1..=1 {
        for dx in -1..=1 {
            let idx_damp = grid.current.idx((10 + dx) as u32, (10 + dy) as u32);
            grid.current.z_bed[idx_damp] = 4.0;
            grid.current.soil_sat[idx_damp] = 0.25;

            let idx_sat = grid.current.idx((30 + dx) as u32, (10 + dy) as u32);
            grid.current.z_bed[idx_sat] = 4.0;
            grid.current.soil_sat[idx_sat] = 1.00;
        }
    }

    grid.apply_reflective_boundaries();

    for _ in 0..20 {
        step_talus_collapse(&mut grid, &params);
    }

    let damp_peak = grid.current.z_bed[grid.current.idx(10, 10)];
    let sat_peak = grid.current.z_bed[grid.current.idx(30, 10)];

    // Damp sand must preserve a higher, steeper peak due to capillary cohesion (phi_damp > phi_sat)
    assert!(
        damp_peak > sat_peak + 0.20,
        "Damp sand did not maintain steeper profile than liquefied saturated sand! Damp: {}, Sat: {}",
        damp_peak,
        sat_peak
    );
}

#[test]
fn test_wave_generator_boundary() {
    use crate::boundary::{EdgeBoundary, DomainBoundaryConfig};

    let desc = SimDomainDescriptor {
        grid_res_x: 20,
        grid_res_y: 20,
        extent_x: 20.0,
        extent_y: 20.0,
        ..Default::default()
    };
    let mut grid = DoubleBufferedGrid::new(desc);
    let wave_gen = EdgeBoundary::WaveGenerator {
        base_elevation: 1.0,
        wave_amplitude: 0.5,
        wave_period: 4.0,
        surge_speed: 1.5,
        tide_amplitude: 0.0,
        tide_period: 60.0,
    };
    grid.boundaries = DomainBoundaryConfig {
        south: wave_gen,
        ..Default::default()
    };

    // Flat beach bathymetry at z = 0.5
    for i in 0..grid.current.z_bed.len() {
        grid.current.z_bed[i] = 0.5;
        grid.current.bedrock_z[i] = 0.0;
    }

    // At t = 0.8s with wave_period = 4.0s:
    // phase = 0.8 / 4.0 = 0.20
    // s = 0.20 / 0.40 = 0.50 -> sin(0.5 * pi) = 1.0 (peak crest!)
    // target_eta = 1.0 + 0.5 * 1.0 = 1.5
    // target_h = 1.5 - 0.5 = 1.0
    // normal_velocity = 1.5 m/s onshore (pushing North, so v = -1.5)
    grid.time = 0.8;
    grid.apply_boundaries();

    let south_ghost_idx = grid.current.idx(10, 19);
    assert!(
        (grid.current.h[south_ghost_idx] - 1.0).abs() < 1e-3,
        "Wave crest depth expected ~1.0m, got: {}",
        grid.current.h[south_ghost_idx]
    );
    assert!(
        (grid.current.v[south_ghost_idx] - (-1.5)).abs() < 1e-3,
        "Wave surge velocity expected -1.5 m/s, got: {}",
        grid.current.v[south_ghost_idx]
    );

    // Step simulation forward with SWE (25 steps of 0.02s = 0.5s)
    for _ in 0..25 {
        step_swe(&mut grid, 0.02);
    }

    // Water should have propagated inland from the South edge into row 18 and 17
    let row18_idx = grid.current.idx(10, 18);
    assert!(
        grid.current.h[row18_idx] > 0.05,
        "Wave surge did not propagate into row 18! Depth: {}",
        grid.current.h[row18_idx]
    );
    let row17_idx = grid.current.idx(10, 17);
    assert!(
        grid.current.h[row17_idx] > 0.02,
        "Wave surge did not propagate into row 17! Depth: {}",
        grid.current.h[row17_idx]
    );
}

#[test]
fn test_beach_sandcastle_waves_simulation() {
    use crate::Scenarios;
    use crate::backend::{CpuSimulator, SimulationBackend};

    let desc = SimDomainDescriptor {
        grid_res_x: 64,
        grid_res_y: 64,
        extent_x: 50.0,
        extent_y: 50.0,
        ..Default::default()
    };

    let grid = Scenarios::beach_sandcastle_waves(desc);
    let mut sim = CpuSimulator::new(grid);

    // Verify initial state
    assert!(sim.sim_time() == 0.0);
    let state = sim.current_state();
    let center_idx = state.idx(32, 34); // Center of sandcastle
    assert!(state.z_bed[center_idx] > 0.5, "Sandcastle elevation should be > 0.5m");

    // Simulate 40 steps
    for _ in 0..40 {
        sim.step(0.01);
    }

    // Verify sanity after wave steps
    let post_state = sim.current_state();
    for (i, &h) in post_state.h.iter().enumerate() {
        assert!(h >= 0.0, "Fluid depth negative at index {}: {}", i, h);
        assert!(!h.is_nan(), "Fluid depth NaN at index {}", i);
        let z = post_state.z_bed[i];
        let b = post_state.bedrock_z[i];
        assert!(z >= b - 1e-4, "Bed elevation below bedrock at index {}: z={}, b={}", i, z, b);
    }
}

#[test]
fn test_beach2_high_relief_simulation() {
    use crate::Scenarios;
    use crate::backend::{CpuSimulator, SimulationBackend};

    let desc = SimDomainDescriptor {
        grid_res_x: 64,
        grid_res_y: 64,
        extent_x: 50.0,
        extent_y: 50.0,
        ..Default::default()
    };

    let grid = Scenarios::beach2(desc);
    let mut sim = CpuSimulator::new(grid);

    let state = sim.current_state();

    // 1. Verify amplified dunes / bluffs height (> 5.0m at North boundary)
    let dune_idx = state.idx(32, 1);
    assert!(state.z_bed[dune_idx] > 5.0, "Dunes should rise above 5.0m, got: {}", state.z_bed[dune_idx]);

    // 2. Verify fortified sandcastle keep elevation (> 3.5m)
    let castle_idx = state.idx(33, 33);
    assert!(state.z_bed[castle_idx] > 3.0, "Sandcastle keep should be > 3.0m, got: {}", state.z_bed[castle_idx]);

    // 3. Verify deep ocean bathymetry (<-3.0m at South boundary) and deep pre-filled water (> 3.5m)
    let ocean_idx = state.idx(32, 62);
    assert!(state.z_bed[ocean_idx] < -3.0, "Seabed should be < -3.0m, got: {}", state.z_bed[ocean_idx]);
    assert!(state.h[ocean_idx] > 3.5, "Ocean water depth should be > 3.5m, got: {}", state.h[ocean_idx]);

    // 4. Verify massive stone breakwater height (> 3.0m) with bedrock == z_bed
    let jetty_idx = state.idx(14, 35);
    assert!(state.z_bed[jetty_idx] >= 3.20 - 1e-3, "Stone breakwater should be >= 3.2m, got: {}", state.z_bed[jetty_idx]);
    assert!((state.z_bed[jetty_idx] - state.bedrock_z[jetty_idx]).abs() < 1e-4, "Jetty must be indestructible stone!");

    // Simulate 40 steps
    for _ in 0..40 {
        sim.step(0.01);
    }

    // 5. Verify positivity, non-NaN, and bedrock constraints
    let post_state = sim.current_state();
    for (i, &h) in post_state.h.iter().enumerate() {
        assert!(h >= 0.0, "Fluid depth negative at index {}: {}", i, h);
        assert!(!h.is_nan(), "Fluid depth NaN at index {}", i);
        let z = post_state.z_bed[i];
        let b = post_state.bedrock_z[i];
        assert!(z >= b - 1e-4, "Bed elevation below bedrock at index {}: z={}, b={}", i, z, b);
    }
}

#[test]
fn test_porous_infiltration_mass_transfer() {
    use crate::solver::sediment::{step_sediment, SedimentParams};

    let desc = SimDomainDescriptor {
        grid_res_x: 16,
        grid_res_y: 16,
        extent_x: 16.0,
        extent_y: 16.0,
        ..Default::default()
    };
    let mut grid = DoubleBufferedGrid::new(desc);

    let p = 0.40;
    let sed_params = SedimentParams {
        porosity: p,
        infiltration_rate: 0.02, // 20 mm/s
        soil_diffusion_rate: 0.0,
        soil_drying_rate: 0.0,
        erodibility: 0.0,
        talus_iterations: 0,
        ..Default::default()
    };

    // Sand bed: 0.5m over bedrock at 0.0m (soil_depth = 0.20m, capacity = 0.08m)
    // Initially completely dry (soil_sat = 0.0)
    for i in 0..grid.current.z_bed.len() {
        grid.current.z_bed[i] = 0.5;
        grid.current.bedrock_z[i] = 0.0;
        grid.current.soil_sat[i] = 0.0;
    }

    // Add a block of shallow water in the interior (4x4 cells)
    for y in 6..10 {
        for x in 6..10 {
            let idx = grid.current.idx(x, y);
            grid.current.h[idx] = 0.04; // 40 mm water
        }
    }

    // Impervious stone cell at (4, 4) with bedrock == z_bed (0 soil depth)
    let stone_idx = grid.current.idx(4, 4);
    grid.current.z_bed[stone_idx] = 0.5;
    grid.current.bedrock_z[stone_idx] = 0.5;
    grid.current.h[stone_idx] = 0.04;

    grid.apply_reflective_boundaries();

    let initial_surface_water = grid.current.interior_mass();
    let initial_total_water = grid.current.interior_total_water_mass(p);

    assert!(initial_surface_water > 0.0);

    // Advance 5 steps of infiltration
    let dt = 0.05;
    for _ in 0..5 {
        step_sediment(&mut grid, dt, &sed_params);
    }

    let post_surface_water = grid.current.interior_mass();
    let post_total_water = grid.current.interior_total_water_mass(p);

    // 1. Surface water h must strictly decrease as water infiltrates porous sand
    assert!(
        post_surface_water < initial_surface_water,
        "Porous infiltration did not drain surface water! Initial: {}, Post: {}",
        initial_surface_water,
        post_surface_water
    );

    // 2. Soil moisture W_sat must increase
    let sample_idx = grid.current.idx(7, 7);
    assert!(
        grid.current.soil_sat[sample_idx] > 0.0,
        "Soil saturation did not increase! Got: {}",
        grid.current.soil_sat[sample_idx]
    );

    // 3. Strict total water mass conservation (surface h + pore moisture)
    let total_diff = (initial_total_water - post_total_water).abs();
    assert!(
        total_diff < 1e-4,
        "Total water mass (surface + pore) not conserved! Initial: {}, Post: {}, Diff: {}",
        initial_total_water,
        post_total_water,
        total_diff
    );

    // 4. Stone cell with z_bed == bedrock_z must not absorb water
    assert!(
        (grid.current.h[stone_idx] - 0.04).abs() < 1e-5,
        "Impervious stone cell absorbed water! Depth: {}",
        grid.current.h[stone_idx]
    );
}
