use macroquad::prelude::*;
use rayon::prelude::*;
use sim_core::domain::SimDomainDescriptor;
use sim_core::state::{DoubleBufferedGrid, GridState};
use sim_core::backend::{SimulationBackend, CpuSimulator};
use sim_core::Scenarios;
use sim_backend::WgpuSimulator;

#[derive(Copy, Clone, PartialEq, Eq)]
enum ActivePreset {
    BeachStream,
    BeachWaves,
    DamBreak,
    LakeAtRest,
}

#[derive(Copy, Clone, PartialEq, Eq)]
enum FlowVisMode {
    Both,
    Vectors,
    Particles,
    Off,
}

impl FlowVisMode {
    fn next(self) -> Self {
        match self {
            Self::Both => Self::Vectors,
            Self::Vectors => Self::Particles,
            Self::Particles => Self::Off,
            Self::Off => Self::Both,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Both => "Flow Lines & Particles",
            Self::Vectors => "Flow Lines (Vectors)",
            Self::Particles => "Particles (Tracers)",
            Self::Off => "Off",
        }
    }
}

struct FlowParticle {
    x: f32,
    y: f32,
    prev_x: f32,
    prev_y: f32,
    life: f32,
    max_life: f32,
    speed: f32,
}

struct FastRng(u32);
impl FastRng {
    fn new(seed: u32) -> Self {
        Self(seed.max(1))
    }
    fn next_u32(&mut self) -> u32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 17;
        self.0 ^= self.0 << 5;
        self.0
    }
    fn next_f32(&mut self) -> f32 {
        (self.next_u32() & 0x00FF_FFFF) as f32 / 16777216.0
    }
    fn range_f32(&mut self, min: f32, max: f32) -> f32 {
        min + (max - min) * self.next_f32()
    }
}

fn respawn_particle(
    p: &mut FlowParticle,
    state: &GridState,
    rng: &mut FastRng,
    width: u32,
    height: u32,
) {
    p.life = rng.range_f32(1.0, 3.5);
    p.max_life = p.life;
    p.speed = 0.0;

    // Sample randomly to find active flowing water
    for _ in 0..20 {
        let gx = rng.range_f32(2.0, (width - 3) as f32) as u32;
        let gy = rng.range_f32(2.0, (height - 3) as f32) as u32;
        let idx = state.idx(gx, gy);
        if state.h[idx] > 0.03 {
            p.x = gx as f32 + rng.next_f32();
            p.y = gy as f32 + rng.next_f32();
            p.prev_x = p.x;
            p.prev_y = p.y;
            return;
        }
    }

    // Fallback: spawn in lower or middle domain
    p.x = rng.range_f32(10.0, (width - 10) as f32);
    p.y = rng.range_f32(height as f32 * 0.35, height as f32 * 0.85);
    p.prev_x = p.x;
    p.prev_y = p.y;
}

/// Directional 3D hillshading pass executed in parallel across grid rows.
fn compute_hillshade_parallel(z_bed: &[f32], shade_cache: &mut [f32], width: usize, height: usize) {
    shade_cache
        .par_chunks_exact_mut(width)
        .enumerate()
        .for_each(|(y, row)| {
            let y_prev_offset = if y > 0 { (y - 1) * width } else { y * width };
            let y_next_offset = if y < height - 1 { (y + 1) * width } else { y * width };
            let y_curr_offset = y * width;

            for x in 0..width {
                let x_prev = if x > 0 { x - 1 } else { x };
                let x_next = if x < width - 1 { x + 1 } else { x };

                let dz_x = (z_bed[y_curr_offset + x_next] - z_bed[y_curr_offset + x_prev]) * 0.5;
                let dz_y = (z_bed[y_next_offset + x] - z_bed[y_prev_offset + x]) * 0.5;
                row[x] = (1.0 - 0.28 * (dz_x + dz_y)).clamp(0.60, 1.40);
            }
        });
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
    let mut flow_vis_mode = FlowVisMode::Both;

    // Fast texture blitting buffer
    let mut img = Image::gen_image_color(desc.grid_res_x as u16, desc.grid_res_y as u16, BLACK);
    let texture = Texture2D::from_image(&img);
    texture.set_filter(FilterMode::Linear);

    // Cached hillshade and telemetry buffers to eliminate single-core CPU stalls
    let mut shade_cache = vec![1.0f32; (desc.grid_res_x * desc.grid_res_y) as usize];
    let mut bed_dirty = true;
    let mut frame_count: u64 = 0;
    let mut cached_fluid_mass = sim.total_fluid_mass();
    let mut cached_sed_mass = sim.total_sediment_mass();
    let mut cached_max_c = 0.0f32;
    let mut safe_dt = 0.004f32;

    // Flow tracer particles (tuned for smooth high-frame-rate rendering)
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

    loop {
        // --- 1. Scenario Presets ---
        if is_key_pressed(KeyCode::Key1) {
            let grid = Scenarios::beach_stream(desc);
            sim = if is_gpu {
                Box::new(WgpuSimulator::new(grid).await)
            } else {
                Box::new(CpuSimulator::new(grid))
            };
            for p in &mut particles {
                respawn_particle(p, sim.current_state(), &mut rng, desc.grid_res_x, desc.grid_res_y);
            }
            is_inflow_active = true;
            current_preset = ActivePreset::BeachStream;
            bed_dirty = true;
        } else if is_key_pressed(KeyCode::Key2) {
            let grid = Scenarios::beach_sandcastle_waves(desc);
            sim = if is_gpu {
                Box::new(WgpuSimulator::new(grid).await)
            } else {
                Box::new(CpuSimulator::new(grid))
            };
            for p in &mut particles {
                respawn_particle(p, sim.current_state(), &mut rng, desc.grid_res_x, desc.grid_res_y);
            }
            is_inflow_active = false;
            current_preset = ActivePreset::BeachWaves;
            bed_dirty = true;
        } else if is_key_pressed(KeyCode::Key3) {
            let grid = Scenarios::dam_break(desc);
            sim = if is_gpu {
                Box::new(WgpuSimulator::new(grid).await)
            } else {
                Box::new(CpuSimulator::new(grid))
            };
            for p in &mut particles {
                respawn_particle(p, sim.current_state(), &mut rng, desc.grid_res_x, desc.grid_res_y);
            }
            is_inflow_active = false;
            current_preset = ActivePreset::DamBreak;
            bed_dirty = true;
        } else if is_key_pressed(KeyCode::Key4) {
            let grid = Scenarios::lake_at_rest(desc);
            sim = if is_gpu {
                Box::new(WgpuSimulator::new(grid).await)
            } else {
                Box::new(CpuSimulator::new(grid))
            };
            for p in &mut particles {
                respawn_particle(p, sim.current_state(), &mut rng, desc.grid_res_x, desc.grid_res_y);
            }
            is_inflow_active = false;
            current_preset = ActivePreset::LakeAtRest;
            bed_dirty = true;
        } else if is_key_pressed(KeyCode::R) {
            // Reset current preset
            let grid = match current_preset {
                ActivePreset::BeachStream => Scenarios::beach_stream(desc),
                ActivePreset::BeachWaves => Scenarios::beach_sandcastle_waves(desc),
                ActivePreset::DamBreak => Scenarios::dam_break(desc),
                ActivePreset::LakeAtRest => Scenarios::lake_at_rest(desc),
            };
            sim = if is_gpu {
                Box::new(WgpuSimulator::new(grid).await)
            } else {
                Box::new(CpuSimulator::new(grid))
            };
            for p in &mut particles {
                respawn_particle(p, sim.current_state(), &mut rng, desc.grid_res_x, desc.grid_res_y);
            }
            bed_dirty = true;
        }

        // --- 2. Toggles ---
        if is_key_pressed(KeyCode::Space) {
            sim.sync_to_cpu();
            let current_grid = DoubleBufferedGrid {
                current: sim.current_state().clone(),
                next: sim.previous_state().clone(),
                descriptor: *sim.descriptor(),
                boundaries: *sim.boundaries(),
                time: sim.sim_time(),
            };

            if is_gpu {
                sim = Box::new(CpuSimulator::new(current_grid));
                is_gpu = false;
            } else {
                sim = Box::new(WgpuSimulator::new(current_grid).await);
                is_gpu = true;
            }
            bed_dirty = true;
        }

        if is_key_pressed(KeyCode::V) {
            flow_vis_mode = flow_vis_mode.next();
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

            let ctrl_down = is_key_down(KeyCode::LeftControl) || is_key_down(KeyCode::RightControl);
            let shift_down = is_key_down(KeyCode::LeftShift) || is_key_down(KeyCode::RightShift);

            let add_water = is_mouse_button_down(MouseButton::Left) && !shift_down && !ctrl_down;
            let build_sand_dam = (shift_down && is_mouse_button_down(MouseButton::Left))
                || is_mouse_button_down(MouseButton::Middle);
            let dig_trench = is_mouse_button_down(MouseButton::Right) && !shift_down && !ctrl_down;
            let dump_sand = shift_down && is_mouse_button_down(MouseButton::Right);
            let place_stone_wall = ctrl_down && is_mouse_button_down(MouseButton::Left);
            let demolish_stone = ctrl_down && is_mouse_button_down(MouseButton::Right);

            if add_water || build_sand_dam || dig_trench || dump_sand || place_stone_wall || demolish_stone {
                for dy in -radius..=radius {
                    for dx in -radius..=radius {
                        if dx * dx + dy * dy <= r2 {
                            let px = (gx + dx).clamp(1, desc.grid_res_x as i32 - 2) as u32;
                            let py = (gy + dy).clamp(1, desc.grid_res_y as i32 - 2) as u32;
                            let idx = sim.current_state().idx(px, py);

                            if add_water {
                                sim.current_state_mut().h[idx] += 0.3;
                            } else if build_sand_dam {
                                sim.current_state_mut().z_bed[idx] += 0.12;
                            } else if dump_sand {
                                // Dump loose erodible sand
                                sim.current_state_mut().z_bed[idx] += 0.08;
                                sim.current_state_mut().soil_sat[idx] = 0.10;
                            } else if dig_trench {
                                let cur_z = sim.current_state().z_bed[idx];
                                let bedrock = sim.current_state().bedrock_z[idx];
                                sim.current_state_mut().z_bed[idx] = (cur_z - 0.12).max(bedrock);
                            } else if place_stone_wall {
                                // Indestructible stone masonry / breakwater: raise both z_bed and bedrock_z
                                sim.current_state_mut().z_bed[idx] += 0.15;
                                sim.current_state_mut().bedrock_z[idx] = sim.current_state().z_bed[idx];
                            } else if demolish_stone {
                                // Demolish stone breakwater down
                                let cur_z = sim.current_state().z_bed[idx];
                                sim.current_state_mut().z_bed[idx] = (cur_z - 0.15).max(-1.0);
                                sim.current_state_mut().bedrock_z[idx] = (sim.current_state().z_bed[idx] - 0.45).max(-1.5);
                            }
                        }
                    }
                }
                if add_water && !build_sand_dam && !dig_trench && !dump_sand && !place_stone_wall && !demolish_stone {
                    sim.upload_water_depth();
                } else {
                    sim.upload_state();
                    bed_dirty = true;
                }
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
            sim.upload_water_depth();
        }

        // --- 5. Simulation Stepping ---
        if !is_paused {
            // Adaptive sub-stepping: compute maximum safe dt according to CFL (target CFL = 0.45)
            let max_cfl_dt = sim.compute_max_stable_dt(0.45);
            safe_dt = max_cfl_dt;
            // Frame simulation duration clamped to ensure true 1:1 real-time pacing across variable refresh rates
            let frame_sim_time = get_frame_time().clamp(0.003, 0.016);
            current_sub_dt = frame_sim_time.min(max_cfl_dt.min(0.004));
            sim.step_subdivided(frame_sim_time, current_sub_dt);
        }

        frame_count += 1;

        // Sync data back to CPU for rendering
        sim.sync_to_cpu();

        clear_background(BLACK);

        let state = sim.current_state();

        // --- 6. Topographic Hillshade, Soil Moisture & Suspended Sediment Shading ---
        if bed_dirty || (!is_paused && frame_count.is_multiple_of(4)) {
            bed_dirty = false;
            compute_hillshade_parallel(
                &state.z_bed,
                &mut shade_cache,
                desc.grid_res_x as usize,
                desc.grid_res_y as usize,
            );
        }

        let width = desc.grid_res_x as usize;
        let row_bytes = width * 4;

        img.bytes
            .par_chunks_exact_mut(row_bytes)
            .enumerate()
            .for_each(|(y, row_slice)| {
                let row_offset = y * width;
                for x in 0..width {
                    let idx = row_offset + x;
                    let byte_idx = x * 4;

                    let z = state.z_bed[idx];
                    let depth = state.h[idx];
                    let sat = state.soil_sat[idx].clamp(0.0, 1.0);
                    let shade = shade_cache[idx];

                    // Terrain color palette based on material & elevation
                    let is_stone_rock = (z - state.bedrock_z[idx]).abs() < 0.04 && state.bedrock_z[idx] > 0.25;
                    let (tr, tg, tb) = if is_stone_rock {
                        // Indestructible stone breakwater / masonry granite
                        (120.0, 125.0, 135.0)
                    } else if z < 0.9 {
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
                        // Distinct water depth gradient:
                        // Shallow: clear, translucent turquoise (depth of riverbed clearly visible through it!)
                        // Mid: rich tropical cyan/azure
                        // Deep: deep ocean navy
                        let (wr, wg, wb) = if depth < 0.12 {
                            (40.0, 175.0, 205.0)
                        } else if depth < 0.60 {
                            (25.0, 110.0, 195.0)
                        } else {
                            (10.0, 45.0, 140.0)
                        };

                        // Suspended sediment tinting: muddy silty river brown where erosion occurs
                        let c = state.sediment_c[idx].clamp(0.0, 0.5);
                        let turbidity = (c / 0.08).clamp(0.0, 1.0);
                        let (mud_r, mud_g, mud_b) = (165.0, 115.0, 65.0);
                        let base_water_r = wr * (1.0 - turbidity) + mud_r * turbidity;
                        let base_water_g = wg * (1.0 - turbidity) + mud_g * turbidity;
                        let base_water_b = wb * (1.0 - turbidity) + mud_b * turbidity;

                        // Rapids foam: ONLY appears in violent rapids or plunge pools (> 2.8 m/s),
                        // capped at 45% max opacity so it never completely washes out the water color.
                        let u = state.u[idx];
                        let v = state.v[idx];
                        let speed_sq = u * u + v * v;
                        let foam = if speed_sq > 7.84 {
                            let speed = speed_sq.sqrt();
                            ((speed - 2.8) / 3.0).clamp(0.0, 0.45)
                        } else {
                            0.0
                        };

                        // Water transparency: shallow water is translucent (alpha ~ 0.38) so the riverbed shows through cleanly
                        let water_alpha = (depth / 0.80).clamp(0.38, 0.92);
                        let final_wr = base_water_r * (1.0 - foam) + 245.0 * foam;
                        let final_wg = base_water_g * (1.0 - foam) + 250.0 * foam;
                        let final_wb = base_water_b * (1.0 - foam) + 255.0 * foam;

                        row_slice[byte_idx] = (r_land * (1.0 - water_alpha) + final_wr * water_alpha) as u8;
                        row_slice[byte_idx + 1] = (g_land * (1.0 - water_alpha) + final_wg * water_alpha) as u8;
                        row_slice[byte_idx + 2] = (b_land * (1.0 - water_alpha) + final_wb * water_alpha) as u8;
                        row_slice[byte_idx + 3] = 255;
                    } else {
                        row_slice[byte_idx] = r_land as u8;
                        row_slice[byte_idx + 1] = g_land as u8;
                        row_slice[byte_idx + 2] = b_land as u8;
                        row_slice[byte_idx + 3] = 255;
                    }
                }
            });

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

        // --- 7. Flow Visualization (Streamlines & Tracer Particles) ---
        let screen_w = screen_width();
        let screen_h = screen_height();
        let frame_dt = get_frame_time().min(0.05);

        // Mode: Flow Lines (Grid Vectors)
        if flow_vis_mode == FlowVisMode::Both || flow_vis_mode == FlowVisMode::Vectors {
            let step: usize = 20;
            for gy in (step / 2..desc.grid_res_y as usize).step_by(step) {
                for gx in (step / 2..desc.grid_res_x as usize).step_by(step) {
                    let idx = state.idx(gx as u32, gy as u32);
                    let depth = state.h[idx];
                    if depth > 0.02 {
                        let u = state.u[idx];
                        let v = state.v[idx];
                        let speed_sq = u * u + v * v;
                        if speed_sq > 0.0064 {
                            let speed = speed_sq.sqrt();
                            let sx = (gx as f32 + 0.5) / desc.grid_res_x as f32 * screen_w;
                            let sy = (gy as f32 + 0.5) / desc.grid_res_y as f32 * screen_h;
                            let dir_x = u / speed;
                            let dir_y = v / speed;
                            // Flow line length scales directly with fluid speed (longer for faster water!)
                            let len = (speed * 12.0).clamp(4.0, 24.0);
                            let ex = sx + dir_x * len;
                            let ey = sy + dir_y * len;
                            let alpha = (speed / 1.5).clamp(0.30, 0.85);

                            draw_line(sx, sy, ex, ey, 1.6, Color::new(0.70, 0.92, 1.0, alpha));
                            draw_circle(ex, ey, 1.3, Color::new(1.0, 1.0, 1.0, alpha * 0.95));
                        }
                    }
                }
            }
        }

        // Mode: Dynamic Tracer Particles
        if flow_vis_mode == FlowVisMode::Both || flow_vis_mode == FlowVisMode::Particles {
            for p in particles.iter_mut() {
                if !is_paused {
                    p.prev_x = p.x;
                    p.prev_y = p.y;

                    let gx = p.x.clamp(1.0, (desc.grid_res_x - 2) as f32) as u32;
                    let gy = p.y.clamp(1.0, (desc.grid_res_y - 2) as f32) as u32;
                    let idx = state.idx(gx, gy);
                    let depth = state.h[idx];
                    let u = state.u[idx];
                    let v = state.v[idx];
                    let speed = (u * u + v * v).sqrt();
                    p.speed = speed;

                    p.life -= frame_dt;
                    if p.life <= 0.0
                        || depth < 0.02
                        || p.x < 2.0
                        || p.x >= (desc.grid_res_x - 2) as f32
                        || p.y < 2.0
                        || p.y >= (desc.grid_res_y - 2) as f32
                    {
                        respawn_particle(p, state, &mut rng, desc.grid_res_x, desc.grid_res_y);
                    } else {
                        // Advect with velocity field
                        p.x += u * 32.0 * frame_dt;
                        p.y += v * 32.0 * frame_dt;
                    }
                }

                let sx0 = p.prev_x / desc.grid_res_x as f32 * screen_w;
                let sy0 = p.prev_y / desc.grid_res_y as f32 * screen_h;
                let sx1 = p.x / desc.grid_res_x as f32 * screen_w;
                let sy1 = p.y / desc.grid_res_y as f32 * screen_h;

                let dist_sq = (sx1 - sx0).powi(2) + (sy1 - sy0).powi(2);
                if dist_sq < 60.0 * 60.0 {
                    let life_alpha = (p.life / p.max_life).clamp(0.0, 1.0);
                    let speed_alpha = (p.speed / 1.0).clamp(0.25, 0.95);
                    let alpha = life_alpha * speed_alpha;
                    draw_line(sx0, sy0, sx1, sy1, 1.8, Color::new(0.88, 0.96, 1.0, alpha));
                    if speed_alpha > 0.40 {
                        draw_circle(sx1, sy1, 1.2, Color::new(1.0, 1.0, 1.0, alpha * 0.90));
                    }
                }
            }
        }

        // --- 8. HUD Telemetry & Control Overlays ---
        if frame_count.is_multiple_of(12) {
            cached_fluid_mass = sim.total_fluid_mass();
            cached_sed_mass = sim.total_sediment_mass();
            cached_max_c = state.sediment_c.iter().copied().fold(0.0f32, f32::max);
        }

        // Top bar
        let hud_height = if current_preset == ActivePreset::BeachWaves { 166.0 } else { 148.0 };
        draw_rectangle(8.0, 8.0, 630.0, hud_height, Color::new(0.0, 0.0, 0.0, 0.82));

        draw_text(
            format!("FPS: {} | Backend: {}", get_fps(), sim.backend_name()).as_str(),
            16.0,
            28.0,
            20.0,
            WHITE,
        );

        let preset_name = match current_preset {
            ActivePreset::BeachStream => "Beach Stream (Flowing & Carving)",
            ActivePreset::BeachWaves => "Coastal Beach & Sandcastle Waves",
            ActivePreset::DamBreak => "Dam Break",
            ActivePreset::LakeAtRest => "Lake at Rest",
        };

        draw_text(
            format!(
                "Scene: {} | Inflow: {} | Status: {}",
                preset_name,
                if is_inflow_active { "ON" } else { "OFF" },
                if is_paused { "PAUSED" } else { "RUNNING" }
            ).as_str(),
            16.0,
            48.0,
            16.0,
            YELLOW,
        );

        draw_text(
            "Tools: [LMB] Water | [RMB] Dig | [Shift+LMB] Sand | [Ctrl+LMB] Stone | [Ctrl+RMB] Break",
            16.0,
            66.0,
            14.0,
            GREEN,
        );

        draw_text(
            "Keys: [1..4] Presets ([1] Stream, [2] Waves, [3] Dam, [4] Lake) | [R] Reset | [Space] Swap",
            16.0,
            82.0,
            13.0,
            LIGHTGRAY,
        );

        draw_text(
            format!(
                "Flow Vis: {} ([V] cycle)",
                flow_vis_mode.label()
            ).as_str(),
            16.0,
            98.0,
            13.0,
            GOLD,
        );

        draw_text(
            format!(
                "Fluid: {:.0} m³ | Solid Sed: {:.0} m³ | Max C: {:.1}%",
                cached_fluid_mass,
                cached_sed_mass,
                cached_max_c * 100.0
            ).as_str(),
            16.0,
            114.0,
            13.0,
            SKYBLUE,
        );

        draw_text(
            format!("CFL Max dt: {:.4}s | Sub-step dt: {:.4}s", safe_dt, current_sub_dt).as_str(),
            16.0,
            130.0,
            12.0,
            DARKGRAY,
        );

        if current_preset == ActivePreset::BeachWaves {
            let t = sim.sim_time();
            let wave_period = 15.0f32;
            let phase = (t / wave_period).rem_euclid(1.0);
            let tide_phase = (2.0 * std::f32::consts::PI * t / 90.0).sin();

            let wave_state = if phase < 0.40 {
                let left = (0.40 - phase) * wave_period;
                format!("Surging Onshore >> ({:.1}s left)", left)
            } else {
                let r = (phase - 0.40) / 0.60;
                if r < 0.50 {
                    let left = (0.50 - r) * 0.60 * wave_period;
                    format!("<< Receding Backwash ({:.1}s)", left)
                } else {
                    let next_in = (1.0 - phase) * wave_period;
                    format!("Calm Inter-surge Rest (Next swell in {:.1}s)", next_in)
                }
            };
            let tide_state = if tide_phase > 0.0 { "High/Rising Tide" } else { "Low/Ebb Tide" };
            draw_text(
                format!("Coastal Dynamics: Wave [{}] | Tide [{}] | Sim Time: {:.1}s", wave_state, tide_state, t).as_str(),
                16.0,
                150.0,
                13.0,
                ORANGE,
            );
        }

        next_frame().await
    }
}
