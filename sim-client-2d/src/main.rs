use macroquad::prelude::*;
use sim_core::domain::SimDomainDescriptor;
use sim_core::state::DoubleBufferedGrid;
use sim_core::backend::{SimulationBackend, CpuSimulator};
use sim_core::Scenarios;
use sim_backend::WgpuSimulator;

mod types;
mod camera;
mod flow_vis;
mod renderer;
mod input;
mod hud;

use types::{ActivePreset, FlowVisMode, FastRng};
use camera::CameraState;
use flow_vis::{FlowParticle, respawn_particle, render_flow_vectors, update_and_render_particles};
use renderer::{compute_hillshade_parallel, render_terrain_and_water};
use input::{handle_camera_input, handle_tools};
use hud::render_hud;

#[macroquad::main("DeepBlue Hydraulic Sandbox")]
async fn main() {
    let initial_grid_res: u32 = std::env::var("GRID_RES")
        .ok()
        .and_then(|v| v.parse().ok())
        .or_else(|| {
            let args: Vec<String> = std::env::args().collect();
            for i in 1..args.len() {
                if args[i] == "--res" || args[i] == "-r" || args[i] == "--grid" {
                    if let Some(val) = args.get(i + 1) {
                        if let Ok(num) = val.parse() {
                            return Some(num);
                        }
                    }
                } else if let Ok(num) = args[i].parse() {
                    return Some(num);
                }
            }
            None
        })
        .unwrap_or(512)
        .clamp(64, 4096);

    let initial_grid_res = (initial_grid_res / 16) * 16;

    let mut desc = SimDomainDescriptor {
        grid_res_x: initial_grid_res,
        grid_res_y: initial_grid_res,
        extent_x: 100.0,
        extent_y: 100.0,
        ..Default::default()
    };

    // Initialize with dynamic beach stream / mountain reservoir scenario
    let initial_grid = Scenarios::beach_stream(desc);
    let mut sim: Box<dyn SimulationBackend> = Box::new(WgpuSimulator::new(initial_grid).await);

    let mut current_sub_dt = 0.004f32;
    let mut is_gpu = true;
    let mut is_inflow_active = true;
    let mut is_paused = false;
    let mut current_preset = ActivePreset::BeachStream;
    let mut flow_vis_mode = FlowVisMode::Particles;

    // Fast texture blitting buffer
    let mut img = Image::gen_image_color(desc.grid_res_x as u16, desc.grid_res_y as u16, BLACK);
    let mut texture = Texture2D::from_image(&img);
    texture.set_filter(FilterMode::Linear);

    let mut shade_cache = vec![1.0f32; (desc.grid_res_x * desc.grid_res_y) as usize];
    let mut bed_dirty = true;
    let mut frame_count: u64 = 0;
    let mut cached_fluid_mass = sim.total_fluid_mass();
    let mut cached_sed_mass = sim.total_sediment_mass();
    let mut cached_max_c = 0.0f32;
    let mut safe_dt = 0.004f32;
    let mut sim_accumulator = 0.0f32;

    const PARTICLE_COUNT: usize = 900;
    let mut rng = FastRng::new(0x9E37_79B9);
    let mut particles: Vec<FlowParticle> = (0..PARTICLE_COUNT)
        .map(|_| {
            let mut p = FlowParticle {
                x: 0.0,
                y: 0.0,
                prev_x: 0.0,
                prev_y: 0.0,
                life: 0.0,
                max_life: 1.0,
                speed: 0.0,
            };
            respawn_particle(&mut p, sim.current_state(), &mut rng, desc.grid_res_x, desc.grid_res_y);
            p
        })
        .collect();

    let mut camera = CameraState::new();
    let mut prev_mouse_pos = mouse_position();

    loop {
        // --- 1. Scenario Presets & Configuration Hotkeys ---
        let mut reload_scenario = None;
        if is_key_pressed(KeyCode::Key1) {
            reload_scenario = Some((Scenarios::beach_stream(desc), ActivePreset::BeachStream, true));
        } else if is_key_pressed(KeyCode::Key2) {
            reload_scenario = Some((Scenarios::beach_sandcastle_waves(desc), ActivePreset::BeachWaves, false));
        } else if is_key_pressed(KeyCode::Key3) {
            reload_scenario = Some((Scenarios::dam_break(desc), ActivePreset::DamBreak, false));
        } else if is_key_pressed(KeyCode::Key4) {
            reload_scenario = Some((Scenarios::lake_at_rest(desc), ActivePreset::LakeAtRest, false));
        } else if is_key_pressed(KeyCode::Key5) {
            reload_scenario = Some((Scenarios::beach2(desc), ActivePreset::BeachWaves2, false));
        } else if is_key_pressed(KeyCode::R) {
            let grid = match current_preset {
                ActivePreset::BeachStream => Scenarios::beach_stream(desc),
                ActivePreset::BeachWaves => Scenarios::beach_sandcastle_waves(desc),
                ActivePreset::BeachWaves2 => Scenarios::beach2(desc),
                ActivePreset::DamBreak => Scenarios::dam_break(desc),
                ActivePreset::LakeAtRest => Scenarios::lake_at_rest(desc),
            };
            reload_scenario = Some((grid, current_preset, is_inflow_active));
        }

        if let Some((grid, preset, inflow)) = reload_scenario {
            sim = if is_gpu {
                Box::new(WgpuSimulator::new(grid).await)
            } else {
                Box::new(CpuSimulator::new(grid))
            };
            for p in &mut particles {
                respawn_particle(p, sim.current_state(), &mut rng, desc.grid_res_x, desc.grid_res_y);
            }
            is_inflow_active = inflow;
            current_preset = preset;
            bed_dirty = true;
        }

        if is_key_pressed(KeyCode::Space) {
            is_gpu = !is_gpu;
            let current_grid = sim.current_state().clone();
            let mut new_grid = DoubleBufferedGrid::new(desc);
            new_grid.current = current_grid;
            new_grid.boundaries = *sim.boundaries();
            sim = if is_gpu {
                Box::new(WgpuSimulator::new(new_grid).await)
            } else {
                Box::new(CpuSimulator::new(new_grid))
            };
            bed_dirty = true;
        }

        if is_key_pressed(KeyCode::P) { is_paused = !is_paused; }
        if is_key_pressed(KeyCode::I) { is_inflow_active = !is_inflow_active; }
        if is_key_pressed(KeyCode::V) { flow_vis_mode = flow_vis_mode.next(); }

        if is_key_pressed(KeyCode::G) {
            let next_res = if desc.grid_res_x <= 256 { 512 } else if desc.grid_res_x <= 512 { 1024 } else { 256 };
            desc.grid_res_x = next_res;
            desc.grid_res_y = next_res;
            let grid = match current_preset {
                ActivePreset::BeachStream => Scenarios::beach_stream(desc),
                ActivePreset::BeachWaves => Scenarios::beach_sandcastle_waves(desc),
                ActivePreset::BeachWaves2 => Scenarios::beach2(desc),
                ActivePreset::DamBreak => Scenarios::dam_break(desc),
                ActivePreset::LakeAtRest => Scenarios::lake_at_rest(desc),
            };
            sim = if is_gpu {
                Box::new(WgpuSimulator::new(grid).await)
            } else {
                Box::new(CpuSimulator::new(grid))
            };
            img = Image::gen_image_color(desc.grid_res_x as u16, desc.grid_res_y as u16, BLACK);
            texture = Texture2D::from_image(&img);
            texture.set_filter(FilterMode::Linear);
            shade_cache = vec![1.0f32; (desc.grid_res_x * desc.grid_res_y) as usize];
            for p in &mut particles {
                respawn_particle(p, sim.current_state(), &mut rng, desc.grid_res_x, desc.grid_res_y);
            }
            bed_dirty = true;
        }

        // --- 2. Camera Navigation & Tool Dispatch ---
        handle_camera_input(&mut camera, &mut prev_mouse_pos);
        handle_tools(&mut sim, &camera, &desc, &mut bed_dirty);

        // Continuous Flow Sources & Sinks
        let stream_active = !is_paused && current_preset == ActivePreset::BeachStream && is_inflow_active;
        let coastal_active = !is_paused && current_preset == ActivePreset::BeachStream;
        sim.set_stream_inflow(stream_active);
        sim.set_coastal_sink(coastal_active);

        // --- 3. Simulation Stepping (Fixed-Timestep Accumulator for Temporal Decoupling) ---
        let frame_time = get_frame_time().min(0.05);
        if !is_paused {
            sim_accumulator += frame_time;
            let fixed_step = 1.0 / 60.0;
            let mut steps = 0;
            while sim_accumulator >= fixed_step && steps < 4 {
                let max_cfl_dt = sim.compute_max_stable_dt(0.45);
                safe_dt = max_cfl_dt;
                let max_sub_dt = 0.004 * (512.0 / desc.grid_res_x as f32);
                current_sub_dt = fixed_step.min(max_cfl_dt.min(max_sub_dt));
                sim.step_subdivided(fixed_step, current_sub_dt);
                sim_accumulator -= fixed_step;
                steps += 1;
            }
        }

        frame_count += 1;
        sim.sync_to_cpu();

        clear_background(BLACK);
        let state = sim.current_state();

        // --- 4. Topographic Hillshading & High-Fidelity Optics Rendering ---
        if bed_dirty || (!is_paused && frame_count.is_multiple_of(4)) {
            bed_dirty = false;
            compute_hillshade_parallel(
                &state.z_bed,
                &mut shade_cache,
                desc.grid_res_x as usize,
                desc.grid_res_y as usize,
            );
        }

        let is_coastal_waves = current_preset == ActivePreset::BeachWaves || current_preset == ActivePreset::BeachWaves2;
        render_terrain_and_water(&mut img, state, &shade_cache, &desc, is_coastal_waves);
        texture.update(&img);

        let (view_x, view_y, view_w, view_h) = camera.view_bounds(desc.grid_res_x as f32, desc.grid_res_y as f32);
        draw_texture_ex(
            &texture,
            0.0,
            0.0,
            WHITE,
            DrawTextureParams {
                dest_size: Some(vec2(screen_width(), screen_height())),
                source: Some(Rect::new(view_x, view_y, view_w, view_h)),
                ..Default::default()
            },
        );

        // --- 5. Flow Visualization (Streamlines & Tracer Particles) ---
        let screen_w = screen_width();
        let screen_h = screen_height();
        if flow_vis_mode == FlowVisMode::Both || flow_vis_mode == FlowVisMode::Vectors {
            render_flow_vectors(state, &camera, &desc, screen_w, screen_h);
        }
        if flow_vis_mode == FlowVisMode::Both || flow_vis_mode == FlowVisMode::Particles {
            update_and_render_particles(
                &mut particles,
                state,
                &camera,
                &desc,
                &mut rng,
                is_paused,
                frame_time,
                screen_w,
                screen_h,
            );
        }

        // --- 6. Telemetry & HUD Overlay ---
        if frame_count.is_multiple_of(12) {
            cached_fluid_mass = sim.total_fluid_mass();
            cached_sed_mass = sim.total_sediment_mass();
            cached_max_c = state.sediment_c.iter().copied().fold(0.0f32, f32::max);
        }

        render_hud(
            &sim,
            &desc,
            &camera,
            current_preset,
            flow_vis_mode,
            is_inflow_active,
            is_paused,
            cached_fluid_mass,
            cached_sed_mass,
            cached_max_c,
            safe_dt,
            current_sub_dt,
        );

        next_frame().await
    }
}
