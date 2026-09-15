use crate::simulator::WgpuSimulator;
use sim_core::domain::SimDomainDescriptor;
use sim_core::state::DoubleBufferedGrid;
use sim_core::backend::SimulationBackend;

#[test]
fn test_wgpu_simulator_backend_trait() {
    let desc = SimDomainDescriptor {
        grid_res_x: 32,
        grid_res_y: 32,
        extent_x: 32.0,
        extent_y: 32.0,
        ..Default::default()
    };

    let mut grid = DoubleBufferedGrid::new(desc);

    // Bedrock channel: impervious rock so zero porous infiltration occurs
    for i in 0..grid.current.z_bed.len() {
        grid.current.bedrock_z[i] = grid.current.z_bed[i];
    }

    // Water block in the center
    for y in 10..22 {
        for x in 10..22 {
            let idx = grid.current.idx(x, y);
            grid.current.h[idx] = 3.0;
        }
    }

    let mut sim: Box<dyn SimulationBackend> = Box::new(WgpuSimulator::new_sync(grid));
    assert_eq!(sim.backend_name(), "GPU (wgpu/WGSL)");

    let initial_mass = sim.total_fluid_mass();
    assert!(initial_mass > 0.0);

    let dt = 0.005;
    for _ in 0..50 {
        sim.step(dt);
    }

    sim.sync_to_cpu();

    let final_mass = sim.total_fluid_mass();
    let diff = (initial_mass - final_mass).abs();

    assert!(
        diff < 1e-2,
        "GPU simulation mass not conserved! Initial: {}, Final: {}, Diff: {}",
        initial_mass,
        final_mass,
        diff
    );
}

#[test]
fn test_wgpu_sediment_conservation() {
    let desc = SimDomainDescriptor {
        grid_res_x: 32,
        grid_res_y: 32,
        extent_x: 32.0,
        extent_y: 32.0,
        ..Default::default()
    };

    let mut grid = DoubleBufferedGrid::new(desc);

    // Saturated sand bed of 1.5m over bedrock at 0.0m
    for i in 0..grid.current.z_bed.len() {
        grid.current.z_bed[i] = 1.5;
        grid.current.bedrock_z[i] = 0.0;
        grid.current.soil_sat[i] = 1.0;
    }

    // Fast water stream in the middle to trigger erosion and suspended transport
    for y in 10..22 {
        for x in 10..22 {
            let idx = grid.current.idx(x, y);
            grid.current.h[idx] = 1.0;
            grid.current.u[idx] = 1.5;
        }
    }

    let mut sim: Box<dyn SimulationBackend> = Box::new(WgpuSimulator::new_sync(grid));
    let initial_sed_mass = sim.total_sediment_mass();
    let initial_fluid_mass = sim.total_fluid_mass();

    for _ in 0..50 {
        sim.step(0.01);
    }

    sim.sync_to_cpu();

    let final_sed_mass = sim.total_sediment_mass();
    let final_fluid_mass = sim.total_fluid_mass();

    // 1. Fluid mass conservation (allowing for minor pore absorption as wave spreads into marginally dried cells)
    let fluid_diff = (initial_fluid_mass - final_fluid_mass).abs();
    assert!(fluid_diff < 0.1, "GPU fluid mass not conserved! Diff: {}", fluid_diff);

    // 2. Sediment total mass conservation
    let sed_diff = (initial_sed_mass - final_sed_mass).abs();
    let rel_error = sed_diff / initial_sed_mass;
    assert!(
        rel_error < 5e-3,
        "GPU sediment mass not conserved! Initial: {}, Final: {}, Diff: {}, Rel: {}",
        initial_sed_mass,
        final_sed_mass,
        sed_diff,
        rel_error
    );

    // 3. Confirm erosion and transport took place
    let max_c = sim.current_state().sediment_c.iter().copied().fold(0.0f32, f32::max);
    let min_z = sim.current_state().z_bed.iter().copied().fold(f32::INFINITY, f32::min);
    assert!(max_c > 0.001, "GPU erosion failed to suspend sediment! Max C: {}", max_c);
    assert!(min_z < 1.499, "GPU flow did not erode bed! Min Z: {}", min_z);

    // 4. Confirm bedrock limit
    for (i, &z) in sim.current_state().z_bed.iter().enumerate() {
        let b = sim.current_state().bedrock_z[i];
        assert!(z >= b - 1e-5, "GPU erosion breached bedrock! Z: {}, Bedrock: {}", z, b);
    }
}

#[test]
fn test_wgpu_beach_waves_scenario() {
    use sim_core::Scenarios;

    let desc = SimDomainDescriptor {
        grid_res_x: 64,
        grid_res_y: 64,
        extent_x: 50.0,
        extent_y: 50.0,
        ..Default::default()
    };

    let grid = Scenarios::beach_sandcastle_waves(desc);
    let mut sim: Box<dyn SimulationBackend> = Box::new(WgpuSimulator::new_sync(grid));

    // Step simulation on GPU for 40 steps (0.4s of physical time with wave surge)
    for _ in 0..40 {
        sim.step(0.01);
    }

    sim.sync_to_cpu();
    let state = sim.current_state();

    // 1. Verify positivity and no NaNs across all cells
    for (i, &h) in state.h.iter().enumerate() {
        assert!(h >= 0.0, "GPU wave fluid depth negative at {}: {}", i, h);
        assert!(!h.is_nan(), "GPU wave fluid depth NaN at {}", i);
        let z = state.z_bed[i];
        let b = state.bedrock_z[i];
        assert!(z >= b - 1e-4, "GPU bed below bedrock at {}: z={}, b={}", i, z, b);
        assert!(!z.is_nan(), "GPU bed elevation NaN at {}", i);
    }

    // 2. Verify wave has surged onshore into South cells
    let swash_idx = state.idx(32, 50); // row 50 is near the coastline
    assert!(
        state.h[swash_idx] > 0.01,
        "GPU wave failed to lap onto beach at row 50! Depth: {}",
        state.h[swash_idx]
    );
}

#[test]
fn test_wgpu_beach2_high_relief_scenario() {
    use sim_core::Scenarios;

    let desc = SimDomainDescriptor {
        grid_res_x: 64,
        grid_res_y: 64,
        extent_x: 50.0,
        extent_y: 50.0,
        ..Default::default()
    };

    let grid = Scenarios::beach2(desc);
    let mut sim: Box<dyn SimulationBackend> = Box::new(WgpuSimulator::new_sync(grid));

    // Step simulation on GPU for 40 steps
    for _ in 0..40 {
        sim.step(0.01);
    }

    sim.sync_to_cpu();
    let state = sim.current_state();

    // 1. Verify positivity and non-NaN
    for (i, &h) in state.h.iter().enumerate() {
        assert!(h >= 0.0, "GPU beach2 fluid depth negative at {}: {}", i, h);
        assert!(!h.is_nan(), "GPU beach2 fluid depth NaN at {}", i);
        let z = state.z_bed[i];
        let b = state.bedrock_z[i];
        assert!(z >= b - 1e-4, "GPU beach2 bed below bedrock at {}: z={}, b={}", i, z, b);
        assert!(!z.is_nan(), "GPU beach2 bed elevation NaN at {}", i);
    }

    // 2. Verify deep ocean depth in South basin
    let ocean_idx = state.idx(32, 62);
    assert!(state.h[ocean_idx] > 3.0, "Deep ocean depth should be > 3.0m, got: {}", state.h[ocean_idx]);
}

#[test]
fn test_wgpu_porous_infiltration() {
    let desc = SimDomainDescriptor {
        grid_res_x: 16,
        grid_res_y: 16,
        extent_x: 16.0,
        extent_y: 16.0,
        ..Default::default()
    };

    let mut grid = DoubleBufferedGrid::new(desc);

    // Dry sand bed of 0.5m over bedrock at 0.0m
    for i in 0..grid.current.z_bed.len() {
        grid.current.z_bed[i] = 0.5;
        grid.current.bedrock_z[i] = 0.0;
        grid.current.soil_sat[i] = 0.0;
    }

    // Add shallow water in center 4x4 block
    for y in 6..10 {
        for x in 6..10 {
            let idx = grid.current.idx(x, y);
            grid.current.h[idx] = 0.04;
        }
    }

    // Impervious stone cell at (4, 4) with bedrock == z_bed (0 soil depth)
    let stone_idx = grid.current.idx(4, 4);
    grid.current.z_bed[stone_idx] = 0.5;
    grid.current.bedrock_z[stone_idx] = 0.5;
    grid.current.h[stone_idx] = 0.04;

    let mut sim: Box<dyn SimulationBackend> = Box::new(WgpuSimulator::new_sync(grid));
    let initial_surface_water = sim.total_fluid_mass();
    let initial_total_water = sim.current_state().interior_total_water_mass(0.40);

    assert!(initial_surface_water > 0.0);

    // Step 5 times on GPU
    for _ in 0..5 {
        sim.step(0.05);
    }

    sim.sync_to_cpu();

    let post_surface_water = sim.total_fluid_mass();
    let post_total_water = sim.current_state().interior_total_water_mass(0.40);
    let sample_idx = sim.current_state().idx(7, 7);

    // 1. Surface water h must strictly decrease on GPU as it soaks into dry sand
    assert!(
        post_surface_water < initial_surface_water,
        "GPU porous infiltration did not drain surface water! Initial: {}, Post: {}",
        initial_surface_water,
        post_surface_water
    );

    // 2. Soil moisture W_sat must increase
    assert!(
        sim.current_state().soil_sat[sample_idx] > 0.0,
        "GPU soil saturation did not increase! Got: {}",
        sim.current_state().soil_sat[sample_idx]
    );

    // 3. Strict total water mass conservation (surface h + pore moisture)
    let total_diff = (initial_total_water - post_total_water).abs();
    assert!(
        total_diff < 0.05,
        "GPU total water mass not conserved! Initial: {}, Post: {}",
        initial_total_water,
        post_total_water
    );
}

#[test]
fn test_wgpu_zero_copy_texture_export() {
    let desc = SimDomainDescriptor {
        grid_res_x: 32,
        grid_res_y: 32,
        extent_x: 32.0,
        extent_y: 32.0,
        ..Default::default()
    };

    let mut grid = DoubleBufferedGrid::new(desc);
    for i in 0..grid.current.z_bed.len() {
        grid.current.z_bed[i] = 1.0;
        grid.current.bedrock_z[i] = 0.5;
        grid.current.soil_sat[i] = 0.8;
        grid.current.h[i] = 0.5;
        grid.current.u[i] = 1.2;
    }

    let mut sim = WgpuSimulator::new_sync(grid);

    // Verify textures exist and match the contract
    assert_eq!(sim.elevation_texture().format(), wgpu::TextureFormat::R32Float);
    assert_eq!(sim.water_texture().format(), wgpu::TextureFormat::Rgba32Float);
    assert_eq!(sim.velocity_texture().format(), wgpu::TextureFormat::Rgba32Float);
    assert_eq!(sim.sed_sat_texture().format(), wgpu::TextureFormat::Rgba32Float);

    assert_eq!(sim.elevation_texture().width(), 32);
    assert_eq!(sim.elevation_texture().height(), 32);

    // Advance simulation and verify export passes execute cleanly without GPU validation errors
    for _ in 0..5 {
        sim.step(0.01);
    }

    sim.export_textures();
    assert!(sim.sim_time() > 0.04);
}

#[test]
fn test_wgpu_wind_forcing_momentum_transfer() {
    let desc = SimDomainDescriptor {
        grid_res_x: 32,
        grid_res_y: 32,
        extent_x: 32.0,
        extent_y: 32.0,
        ..Default::default()
    };

    let mut grid = DoubleBufferedGrid::new(desc);
    for i in 0..grid.current.z_bed.len() {
        grid.current.z_bed[i] = 0.0;
        grid.current.bedrock_z[i] = -1.0;
        grid.current.h[i] = 1.0;
        grid.current.u[i] = 0.0;
        grid.current.v[i] = 0.0;
    }

    let mut sim = WgpuSimulator::new_sync(grid);
    sim.set_wind(16.0, 0.0, -1.0);

    for _ in 0..30 {
        sim.step(0.01);
    }

    sim.sync_to_cpu();
    let state = sim.current_state();

    let mid_idx = state.idx(16, 16);
    assert!(state.v[mid_idx] < -0.01, "Wind should drive fluid Northward (v < 0), got: {}", state.v[mid_idx]);

    for (i, &h) in state.h.iter().enumerate() {
        assert!(h >= 0.0, "Depth negative at {}: {}", i, h);
        assert!(!h.is_nan(), "Depth NaN at {}", i);
        assert!(!state.u[i].is_nan(), "U velocity NaN at {}", i);
        assert!(!state.v[i].is_nan(), "V velocity NaN at {}", i);
    }
}

#[test]
fn test_wgpu_wind_turbulence_and_sheltering() {
    let desc = SimDomainDescriptor {
        grid_res_x: 32,
        grid_res_y: 32,
        extent_x: 32.0,
        extent_y: 32.0,
        ..Default::default()
    };

    let mut grid = DoubleBufferedGrid::new(desc);
    for y in 0..32 {
        for x in 0..32 {
            let idx = grid.current.idx(x, y);
            // Create an upwind barrier ridge at y = 19..=21 (z = 4.0), water elsewhere (z = 0.0)
            if y >= 19 && y <= 21 {
                grid.current.z_bed[idx] = 4.0;
                grid.current.bedrock_z[idx] = 4.0;
                grid.current.h[idx] = 0.0;
            } else {
                grid.current.z_bed[idx] = 0.0;
                grid.current.bedrock_z[idx] = -1.0;
                grid.current.h[idx] = 1.0;
            }
            grid.current.u[idx] = 0.0;
            grid.current.v[idx] = 0.0;
        }
    }

    let mut sim = WgpuSimulator::new_sync(grid);
    // Wind blowing Northwards (dir = [0, -1]), with strong turbulence and sheltering
    sim.set_wind_full(18.0, 0.0, -1.0, 0.60, 0.85);

    for _ in 0..25 {
        sim.step(0.01);
    }

    sim.sync_to_cpu();
    let state = sim.current_state();

    // Check sheltered region directly downwind of the barrier (y = 17)
    let sheltered_idx = state.idx(16, 17);
    // Check unsheltered open water far from barrier (y = 28)
    let open_idx = state.idx(16, 28);

    let v_sheltered = state.v[sheltered_idx].abs();
    let v_open = state.v[open_idx].abs();

    // The barrier at y=19..=21 should significantly shelter y=17 from wind coming from the South
    assert!(v_sheltered < v_open, "Sheltered velocity ({}) should be less than open water ({})", v_sheltered, v_open);

    for (i, &h) in state.h.iter().enumerate() {
        assert!(h >= 0.0, "Depth negative at {}: {}", i, h);
        assert!(!h.is_nan(), "Depth NaN at {}", i);
        assert!(!state.u[i].is_nan(), "U NaN at {}", i);
        assert!(!state.v[i].is_nan(), "V NaN at {}", i);
    }
}
