use macroquad::prelude::*;
use sim_core::domain::SimDomainDescriptor;
use sim_core::state::DoubleBufferedGrid;
use sim_core::backend::{SimulationBackend, CpuSimulator};
use sim_core::Scenarios;
use sim_backend::WgpuSimulator;

#[derive(Copy, Clone, PartialEq, Eq)]
enum ActivePreset {
    BeachStream,
    DamBreak,
    LakeAtRest,
}

#[macroquad::main("DeepBlue Hydraulic Sandbox")]
async fn main() {
    let desc = SimDomainDescriptor {
        grid_res_x: 512,
        grid_res_y: 512,
        extent_x: 100.0,
        extent_y: 100.0,
        ..Default::default()
    };

    // Initialize with the dynamic beach stream / mountain reservoir scenario
    let initial_grid = Scenarios::beach_stream(desc);
    let mut sim: Box<dyn SimulationBackend> = Box::new(WgpuSimulator::new(initial_grid).await);

    let mut current_sub_dt = 0.004f32;
    let mut is_gpu = true;
    let mut is_inflow_active = true;
    let mut is_paused = false;
    let mut current_preset = ActivePreset::BeachStream;

    // Fast texture blitting buffer
    let mut img = Image::gen_image_color(desc.grid_res_x as u16, desc.grid_res_y as u16, BLACK);
    let texture = Texture2D::from_image(&img);
    texture.set_filter(FilterMode::Linear);

    loop {
        // --- 1. Scenario Presets ---
        if is_key_pressed(KeyCode::Key1) {
            let grid = Scenarios::beach_stream(desc);
            sim = if is_gpu {
                Box::new(WgpuSimulator::new(grid).await)
            } else {
                Box::new(CpuSimulator::new(grid))
            };
            is_inflow_active = true;
            current_preset = ActivePreset::BeachStream;
        } else if is_key_pressed(KeyCode::Key2) {
            let grid = Scenarios::dam_break(desc);
            sim = if is_gpu {
                Box::new(WgpuSimulator::new(grid).await)
            } else {
                Box::new(CpuSimulator::new(grid))
            };
            is_inflow_active = false;
            current_preset = ActivePreset::DamBreak;
        } else if is_key_pressed(KeyCode::Key3) {
            let grid = Scenarios::lake_at_rest(desc);
            sim = if is_gpu {
                Box::new(WgpuSimulator::new(grid).await)
            } else {
                Box::new(CpuSimulator::new(grid))
            };
            is_inflow_active = false;
            current_preset = ActivePreset::LakeAtRest;
        } else if is_key_pressed(KeyCode::R) {
            // Reset current preset
            let grid = match current_preset {
                ActivePreset::BeachStream => Scenarios::beach_stream(desc),
                ActivePreset::DamBreak => Scenarios::dam_break(desc),
                ActivePreset::LakeAtRest => Scenarios::lake_at_rest(desc),
            };
            sim = if is_gpu {
                Box::new(WgpuSimulator::new(grid).await)
            } else {
                Box::new(CpuSimulator::new(grid))
            };
        }

        // --- 2. Toggles ---
        if is_key_pressed(KeyCode::Space) {
            sim.sync_to_cpu();
            let current_grid = DoubleBufferedGrid {
                current: sim.current_state().clone(),
                next: sim.previous_state().clone(),
                descriptor: *sim.descriptor(),
            };

            if is_gpu {
                sim = Box::new(CpuSimulator::new(current_grid));
                is_gpu = false;
            } else {
                sim = Box::new(WgpuSimulator::new(current_grid).await);
                is_gpu = true;
            }
        }

        if is_key_pressed(KeyCode::I) {
            is_inflow_active = !is_inflow_active;
        }

        if is_key_pressed(KeyCode::P) {
            is_paused = !is_paused;
        }

        // --- 3. Interactive Mouse Tools ---
        let (mx, my) = mouse_position();
        let mouse_in_window = mx >= 0.0 && mx < screen_width() && my >= 0.0 && my < screen_height();

        if mouse_in_window {
            let gx = (mx / screen_width() * desc.grid_res_x as f32) as i32;
            let gy = (my / screen_height() * desc.grid_res_y as f32) as i32;
            let radius = 10;
            let r2 = radius * radius;

            let add_water = is_mouse_button_down(MouseButton::Left) && !is_key_down(KeyCode::LeftShift);
            let build_dam = (is_key_down(KeyCode::LeftShift) && is_mouse_button_down(MouseButton::Left))
                || is_mouse_button_down(MouseButton::Middle);
            let dig_trench = is_mouse_button_down(MouseButton::Right) && !is_key_down(KeyCode::LeftShift);
            let dump_sand = is_key_down(KeyCode::LeftShift) && is_mouse_button_down(MouseButton::Right);

            if add_water || build_dam || dig_trench || dump_sand {
                for dy in -radius..=radius {
                    for dx in -radius..=radius {
                        if dx * dx + dy * dy <= r2 {
                            let px = (gx + dx).clamp(1, desc.grid_res_x as i32 - 2) as u32;
                            let py = (gy + dy).clamp(1, desc.grid_res_y as i32 - 2) as u32;
                            let idx = sim.current_state().idx(px, py);

                            if add_water {
                                sim.current_state_mut().h[idx] += 0.3;
                            } else if build_dam {
                                sim.current_state_mut().z_bed[idx] += 0.12;
                            } else if dump_sand {
                                // Dump loose erodible sand
                                sim.current_state_mut().z_bed[idx] += 0.08;
                                sim.current_state_mut().soil_sat[idx] = 0.10;
                            } else if dig_trench {
                                let cur_z = sim.current_state().z_bed[idx];
                                let bedrock = sim.current_state().bedrock_z[idx];
                                sim.current_state_mut().z_bed[idx] = (cur_z - 0.12).max(bedrock);
                            }
                        }
                    }
                }
                sim.upload_state();
            }
        }

        // --- 4. Continuous Flow Sources & Sinks (Beach Stream Preset) ---
        if !is_paused && current_preset == ActivePreset::BeachStream {
            if is_inflow_active {
                // Mountain spring inlet feeding the reservoir at the top
                let w = desc.grid_res_x;
                let inflow_y_end = (desc.grid_res_y as f32 * 0.06) as u32;
                let inflow_x_center = (w as f32 * 0.5) as u32;
                let inflow_radius = (w as f32 * 0.05) as u32;

                for y in 1..=inflow_y_end {
                    for x in (inflow_x_center - inflow_radius)..=(inflow_x_center + inflow_radius) {
                        let idx = sim.current_state().idx(x, y);
                        let z = sim.current_state().z_bed[idx];
                        let target_h = (2.6 - z).max(0.6);
                        if sim.current_state().h[idx] < target_h {
                            sim.current_state_mut().h[idx] = target_h;
                        }
                    }
                }
            }

            // Coastal sink: absorb fluid entering the bottom boundary into the ocean
            let h_start = desc.grid_res_y - 12;
            for y in h_start..(desc.grid_res_y - 1) {
                for x in 1..(desc.grid_res_x - 1) {
                    let idx = sim.current_state().idx(x, y);
                    sim.current_state_mut().h[idx] *= 0.82;
                }
            }
            sim.upload_state();
        }

        // --- 5. Simulation Stepping ---
        if !is_paused {
            // Adaptive sub-stepping: compute maximum safe dt according to CFL (target CFL = 0.45)
            let max_cfl_dt = sim.compute_max_stable_dt(0.45);
            let frame_sim_time = 0.012f32;
            current_sub_dt = frame_sim_time.min(max_cfl_dt.min(0.004));
            sim.step_subdivided(frame_sim_time, current_sub_dt);
        }

        // Sync data back to CPU for rendering
        sim.sync_to_cpu();

        clear_background(BLACK);

        let state = sim.current_state();

        // --- 6. Topographic Hillshade, Soil Moisture & Suspended Sediment Shading ---
        for y in 0..desc.grid_res_y {
            for x in 0..desc.grid_res_x {
                let idx = state.idx(x, y);
                let z = state.z_bed[idx];
                let depth = state.h[idx];
                let sat = state.soil_sat[idx].clamp(0.0, 1.0);
                let byte_idx = idx * 4;

                // 3D Directional hillshading: finite differences to estimate slope
                let x_prev = if x > 0 { state.idx(x - 1, y) } else { idx };
                let x_next = if x < desc.grid_res_x - 1 { state.idx(x + 1, y) } else { idx };
                let y_prev = if y > 0 { state.idx(x, y - 1) } else { idx };
                let y_next = if y < desc.grid_res_y - 1 { state.idx(x, y + 1) } else { idx };

                let dz_x = (state.z_bed[x_next] - state.z_bed[x_prev]) * 0.5;
                let dz_y = (state.z_bed[y_next] - state.z_bed[y_prev]) * 0.5;
                let shade = (1.0 - 0.28 * (dz_x + dz_y)).clamp(0.60, 1.40);

                // Terrain color palette based on elevation
                let (tr, tg, tb) = if z < 0.9 {
                    // Moist gravel / wet sand
                    (155.0, 130.0, 95.0)
                } else if z < 2.2 {
                    // Golden beach sand dunes
                    (215.0, 185.0, 135.0)
                } else {
                    // Rocky canyon wall
                    (160.0, 145.0, 130.0)
                };

                // Soil moisture / saturation effect: wet sand darkens naturally
                let moisture_darkening = 1.0 - 0.32 * sat;
                let r_land = (tr * shade * moisture_darkening).clamp(0.0, 255.0);
                let g_land = (tg * shade * moisture_darkening).clamp(0.0, 255.0);
                let b_land = (tb * shade * moisture_darkening).clamp(0.0, 255.0);

                if depth > 0.005 {
                    // Water depth tint: shallow turquoise -> deep cobalt
                    let (wr, wg, wb) = if depth < 0.15 {
                        (65.0, 195.0, 220.0)
                    } else if depth < 0.9 {
                        (30.0, 135.0, 215.0)
                    } else {
                        (15.0, 60.0, 165.0)
                    };

                    // Suspended sediment tinting: muddy silty river brown where erosion occurs
                    let c = state.sediment_c[idx].clamp(0.0, 0.5);
                    let turbidity = (c / 0.10).clamp(0.0, 1.0);
                    let (mud_r, mud_g, mud_b) = (175.0, 125.0, 70.0);
                    let base_water_r = wr * (1.0 - turbidity) + mud_r * turbidity;
                    let base_water_g = wg * (1.0 - turbidity) + mud_g * turbidity;
                    let base_water_b = wb * (1.0 - turbidity) + mud_b * turbidity;

                    // Whitewater rapids foam based on fluid velocity
                    let u = state.u[idx];
                    let v = state.v[idx];
                    let speed = (u * u + v * v).sqrt();
                    let foam = (speed / 3.2).clamp(0.0, 1.0);

                    let water_alpha = (depth / 0.75).clamp(0.42, 0.92);
                    let final_wr = base_water_r * (1.0 - foam) + 245.0 * foam;
                    let final_wg = base_water_g * (1.0 - foam) + 250.0 * foam;
                    let final_wb = base_water_b * (1.0 - foam) + 255.0 * foam;

                    img.bytes[byte_idx] = (r_land * (1.0 - water_alpha) + final_wr * water_alpha) as u8;
                    img.bytes[byte_idx + 1] = (g_land * (1.0 - water_alpha) + final_wg * water_alpha) as u8;
                    img.bytes[byte_idx + 2] = (b_land * (1.0 - water_alpha) + final_wb * water_alpha) as u8;
                    img.bytes[byte_idx + 3] = 255;
                } else {
                    img.bytes[byte_idx] = r_land as u8;
                    img.bytes[byte_idx + 1] = g_land as u8;
                    img.bytes[byte_idx + 2] = b_land as u8;
                    img.bytes[byte_idx + 3] = 255;
                }
            }
        }

        texture.update(&img);

        draw_texture_ex(
            &texture,
            0.0,
            0.0,
            WHITE,
            DrawTextureParams {
                dest_size: Some(vec2(screen_width(), screen_height())),
                ..Default::default()
            },
        );

        // --- 7. HUD Telemetry & Control Overlays ---
        let safe_dt = sim.compute_max_stable_dt(0.5);
        let fluid_mass = sim.total_fluid_mass();
        let sed_mass = sim.total_sediment_mass();
        let max_c = state.sediment_c.iter().copied().fold(0.0f32, f32::max);

        // Top bar
        draw_rectangle(8.0, 8.0, 520.0, 118.0, Color::new(0.0, 0.0, 0.0, 0.78));

        draw_text(
            format!("FPS: {} | Backend: {}", get_fps(), sim.backend_name()),
            16.0,
            28.0,
            20.0,
            WHITE,
        );

        let preset_name = match current_preset {
            ActivePreset::BeachStream => "Beach Stream (Flowing & Carving)",
            ActivePreset::DamBreak => "Dam Break",
            ActivePreset::LakeAtRest => "Lake at Rest",
        };

        draw_text(
            format!(
                "Scene: {} | Inflow: {} | Status: {}",
                preset_name,
                if is_inflow_active { "ON" } else { "OFF" },
                if is_paused { "PAUSED" } else { "RUNNING" }
            ),
            16.0,
            48.0,
            16.0,
            YELLOW,
        );

        draw_text(
            "Tools: [LMB] Water | [RMB] Dig | [Shift+LMB] Dam | [Shift+RMB] Sand",
            16.0,
            68.0,
            15.0,
            GREEN,
        );

        draw_text(
            "Keys: [1..3] Presets | [R] Reset | [I] Inflow | [P] Pause | [Space] Swap",
            16.0,
            86.0,
            14.0,
            LIGHTGRAY,
        );

        draw_text(
            format!(
                "Fluid: {:.0} m³ | Solid Sed: {:.0} m³ | Max C: {:.1}%",
                fluid_mass,
                sed_mass,
                max_c * 100.0
            ),
            16.0,
            104.0,
            14.0,
            SKYBLUE,
        );

        draw_text(
            format!("CFL Max dt: {:.4}s | Sub-step dt: {:.4}s", safe_dt, current_sub_dt),
            16.0,
            120.0,
            13.0,
            DARKGRAY,
        );

        next_frame().await
    }
}
