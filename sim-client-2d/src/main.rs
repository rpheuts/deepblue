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
            Self::Particles => Self::Both,
            Self::Both => Self::Vectors,
            Self::Vectors => Self::Off,
            Self::Off => Self::Particles,
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

#[derive(Copy, Clone, Debug)]
struct CameraState {
    target_x: f32, // normalized center [0.0, 1.0]
    target_y: f32,
    zoom: f32,     // 1.0 = full view, up to 6.0x
}

impl CameraState {
    fn new() -> Self {
        Self {
            target_x: 0.5,
            target_y: 0.5,
            zoom: 1.0,
        }
    }

    fn reset(&mut self) {
        self.target_x = 0.5;
        self.target_y = 0.5;
        self.zoom = 1.0;
    }

    /// Returns active view bounds in grid cells: (view_x, view_y, view_w, view_h)
    fn view_bounds(&self, width: f32, height: f32) -> (f32, f32, f32, f32) {
        let view_w = width / self.zoom;
        let view_h = height / self.zoom;
        let view_x = (self.target_x * width - view_w * 0.5).clamp(0.0, width - view_w);
        let view_y = (self.target_y * height - view_h * 0.5).clamp(0.0, height - view_h);
        (view_x, view_y, view_w, view_h)
    }

    /// Converts screen pixel coordinates to simulation grid cell coordinates
    fn screen_to_grid(&self, sx: f32, sy: f32, screen_w: f32, screen_h: f32, grid_w: f32, grid_h: f32) -> (i32, i32) {
        let (view_x, view_y, view_w, view_h) = self.view_bounds(grid_w, grid_h);
        let gx = (view_x + (sx / screen_w) * view_w) as i32;
        let gy = (view_y + (sy / screen_h) * view_h) as i32;
        (gx, gy)
    }

    /// Converts grid coordinates to screen pixel coordinates
    fn grid_to_screen(&self, gx: f32, gy: f32, screen_w: f32, screen_h: f32, grid_w: f32, grid_h: f32) -> (f32, f32) {
        let (view_x, view_y, view_w, view_h) = self.view_bounds(grid_w, grid_h);
        let sx = (gx - view_x) / view_w * screen_w;
        let sy = (gy - view_y) / view_h * screen_h;
        (sx, sy)
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
    // Read command line argument or environment variable for initial grid resolution
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

    // Initialize with the dynamic beach stream / mountain reservoir scenario
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

    let mut camera = CameraState::new();
    let mut prev_mouse_pos = mouse_position();

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
        } else if is_key_pressed(KeyCode::G) {
            // Cycle grid density: 256 -> 512 -> 768 -> 1024 -> 2048
            let shift = is_key_down(KeyCode::LeftShift) || is_key_down(KeyCode::RightShift);
            let densities = [256, 512, 768, 1024, 2048];
            let cur_idx = densities
                .iter()
                .position(|&r| r == desc.grid_res_x)
                .unwrap_or(1);
            let next_idx = if shift {
                if cur_idx == 0 { densities.len() - 1 } else { cur_idx - 1 }
            } else {
                (cur_idx + 1) % densities.len()
            };
            let new_res = densities[next_idx];
            desc.grid_res_x = new_res;
            desc.grid_res_y = new_res;

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

            img = Image::gen_image_color(new_res as u16, new_res as u16, BLACK);
            texture = Texture2D::from_image(&img);
            texture.set_filter(FilterMode::Linear);
            shade_cache = vec![1.0f32; (new_res * new_res) as usize];
            bed_dirty = true;
            camera = CameraState::new();

            for p in &mut particles {
                respawn_particle(p, sim.current_state(), &mut rng, desc.grid_res_x, desc.grid_res_y);
            }
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

        // --- 3. Camera Zoom & Pan Controls ---
        let (mx, my) = mouse_position();
        let mouse_in_window = mx >= 0.0 && mx < screen_width() && my >= 0.0 && my < screen_height();

        let wheel = mouse_wheel().1;
        if wheel.abs() > 0.01 {
            let prev_zoom = camera.zoom;
            let factor = if wheel > 0.0 { 1.15 } else { 1.0 / 1.15 };
            camera.zoom = (camera.zoom * factor).clamp(1.0, 6.0);

            if (camera.zoom - prev_zoom).abs() > 1e-4 {
                let norm_x = (mx / screen_width()).clamp(0.0, 1.0);
                let norm_y = (my / screen_height()).clamp(0.0, 1.0);
                camera.target_x += (norm_x - 0.5) * (1.0 / prev_zoom - 1.0 / camera.zoom);
                camera.target_y += (norm_y - 0.5) * (1.0 / prev_zoom - 1.0 / camera.zoom);
                camera.target_x = camera.target_x.clamp(0.0, 1.0);
                camera.target_y = camera.target_y.clamp(0.0, 1.0);
            }
        }

        if is_mouse_button_down(MouseButton::Middle) {
            let dx = mx - prev_mouse_pos.0;
            let dy = my - prev_mouse_pos.1;
            camera.target_x -= (dx / screen_width()) / camera.zoom;
            camera.target_y -= (dy / screen_height()) / camera.zoom;
            camera.target_x = camera.target_x.clamp(0.0, 1.0);
            camera.target_y = camera.target_y.clamp(0.0, 1.0);
        }

        let pan_step = 0.012 / camera.zoom;
        if is_key_down(KeyCode::Left) { camera.target_x = (camera.target_x - pan_step).max(0.0); }
        if is_key_down(KeyCode::Right) { camera.target_x = (camera.target_x + pan_step).min(1.0); }
        if is_key_down(KeyCode::Up) { camera.target_y = (camera.target_y - pan_step).max(0.0); }
        if is_key_down(KeyCode::Down) { camera.target_y = (camera.target_y + pan_step).min(1.0); }

        if is_key_pressed(KeyCode::C) || is_key_pressed(KeyCode::Key0) {
            camera.reset();
        }

        prev_mouse_pos = (mx, my);

        // --- 4. Interactive Mouse Tools ---
        if mouse_in_window && !is_mouse_button_down(MouseButton::Middle) {
            let (gx, gy) = camera.screen_to_grid(mx, my, screen_width(), screen_height(), desc.grid_res_x as f32, desc.grid_res_y as f32);
            let base_radius = 10.0 * (desc.grid_res_x as f32 / 512.0);
            let radius = (base_radius / camera.zoom.sqrt()).max(3.0) as i32;
            let r2 = radius * radius;

            let ctrl_down = is_key_down(KeyCode::LeftControl) || is_key_down(KeyCode::RightControl);
            let shift_down = is_key_down(KeyCode::LeftShift) || is_key_down(KeyCode::RightShift);

            let add_water = is_mouse_button_down(MouseButton::Left) && !shift_down && !ctrl_down;
            let build_sand_dam = shift_down && is_mouse_button_down(MouseButton::Left);
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
        let stream_active = !is_paused && current_preset == ActivePreset::BeachStream && is_inflow_active;
        let coastal_active = !is_paused && current_preset == ActivePreset::BeachStream;
        sim.set_stream_inflow(stream_active);
        sim.set_coastal_sink(coastal_active);

        // --- 5. Simulation Stepping ---
        if !is_paused {
            // Adaptive sub-stepping: compute maximum safe dt according to CFL (target CFL = 0.45)
            let max_cfl_dt = sim.compute_max_stable_dt(0.45);
            safe_dt = max_cfl_dt;
            // Frame simulation duration clamped to ensure true 1:1 real-time pacing across variable refresh rates
            let frame_sim_time = get_frame_time().clamp(0.003, 0.016);
            let max_sub_dt = 0.004 * (512.0 / desc.grid_res_x as f32);
            current_sub_dt = frame_sim_time.min(max_cfl_dt.min(max_sub_dt));
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
        let height = desc.grid_res_y as usize;
        let row_bytes = width * 4;
        let is_coastal_waves = current_preset == ActivePreset::BeachWaves;

        img.bytes
            .par_chunks_exact_mut(row_bytes)
            .enumerate()
            .for_each(|(y, row_slice)| {
                let row_offset = y * width;
                let y_prev_offset = if y > 0 { (y - 1) * width } else { y * width };
                let y_next_offset = if y < height - 1 { (y + 1) * width } else { y * width };

                for x in 0..width {
                    let idx = row_offset + x;
                    let byte_idx = x * 4;

                    let z = state.z_bed[idx];
                    let depth = state.h[idx];
                    let sat = state.soil_sat[idx].clamp(0.0, 1.0);
                    let shade = shade_cache[idx];

                    let x_prev = if x > 0 { x - 1 } else { x };
                    let x_next = if x < width - 1 { x + 1 } else { x };

                    // 1. Terrain base material & elevation color
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

                    // Soil moisture: wet sand darkens naturally
                    let moisture_darkening = 1.0 - 0.32 * sat;
                    let r_land = (tr * shade * moisture_darkening).clamp(0.0, 255.0);
                    let g_land = (tg * shade * moisture_darkening).clamp(0.0, 255.0);
                    let b_land = (tb * shade * moisture_darkening).clamp(0.0, 255.0);

                    if depth > 0.005 {
                        // --- HIGH-FIDELITY WATER OPTICS & LIGHTING PASS ---
                        let u = state.u[idx];
                        let v = state.v[idx];
                        let speed_sq = u * u + v * v;
                        let speed = speed_sq.sqrt();

                        // 3D Water Surface Normal & Slopes
                        let eta_l = state.z_bed[row_offset + x_prev] + state.h[row_offset + x_prev];
                        let eta_r = state.z_bed[row_offset + x_next] + state.h[row_offset + x_next];
                        let eta_t = state.z_bed[y_prev_offset + x] + state.h[y_prev_offset + x];
                        let eta_b = state.z_bed[y_next_offset + x] + state.h[y_next_offset + x];

                        let deta_x = (eta_r - eta_l) * 0.5;
                        let deta_y = (eta_b - eta_t) * 0.5;

                        // A. Physical Beer-Lambert Optical Extinction
                        // Red absorbs rapidly, green moderately, blue penetrates deepest
                        let t_r = (-4.2 * depth).exp();
                        let t_g = (-1.25 * depth).exp();
                        let t_b = (-0.40 * depth).exp();

                        // Deep water body in-scattering (tropical azure to deep oceanic sapphire)
                        let deep_r = 10.0;
                        let deep_g = 58.0;
                        let deep_b = 148.0;
                        let inscatter_r = deep_r * (1.0 - t_r);
                        let inscatter_g = deep_g * (1.0 - t_g);
                        let inscatter_b = deep_b * (1.0 - t_b);

                        let mut water_r = r_land * t_r + inscatter_r;
                        let mut water_g = g_land * t_g + inscatter_g;
                        let mut water_b = b_land * t_b + inscatter_b;

                        // B. Suspended Sediment Turbidity
                        let c = state.sediment_c[idx].clamp(0.0, 0.5);
                        let turbidity = (c / 0.08).clamp(0.0, 1.0);
                        let (mud_r, mud_g, mud_b) = (165.0, 115.0, 65.0);
                        water_r = water_r * (1.0 - turbidity) + mud_r * turbidity;
                        water_g = water_g * (1.0 - turbidity) + mud_g * turbidity;
                        water_b = water_b * (1.0 - turbidity) + mud_b * turbidity;

                        // C. 3D Water Surface Normal & Specular Sun Glint
                        let nx = -deta_x * 2.8;
                        let ny = -deta_y * 2.8;
                        let n_len = (nx * nx + ny * ny + 1.0).sqrt();
                        let norm_x = nx / n_len;
                        let norm_y = ny / n_len;
                        let norm_z = 1.0 / n_len;

                        // Sun Half-Vector H = (-0.209, -0.247, 0.946)
                        let n_dot_h = (norm_x * (-0.209) + norm_y * (-0.247) + norm_z * 0.946).max(0.0);
                        // Flat water (n_dot_h = 0.946) has zero specular glint, keeping still water clear.
                        // Only wave facets tilted towards the sun sparkle with glints.
                        let spec_sun = if n_dot_h > 0.960 {
                            let t = (n_dot_h - 0.960) / (1.0 - 0.960);
                            t.powi(12) * 1.6
                        } else {
                            0.0
                        };

                        // Fresnel sky reflectance
                        let one_minus_cos = (1.0 - norm_z).max(0.0);
                        let fresnel = 0.04 + 0.96 * one_minus_cos.powi(4);
                        let sky_r = 180.0;
                        let sky_g = 215.0;
                        let sky_b = 248.0;

                        water_r = water_r * (1.0 - fresnel * 0.45) + sky_r * (fresnel * 0.45);
                        water_g = water_g * (1.0 - fresnel * 0.45) + sky_g * (fresnel * 0.45);
                        water_b = water_b * (1.0 - fresnel * 0.45) + sky_b * (fresnel * 0.45);

                        water_r += 255.0 * spec_sun;
                        water_g += 248.0 * spec_sun;
                        water_b += 220.0 * spec_sun;

                        // D. Multi-Source Sea Foam (Only enabled for coastal waves preset)
                        let foam = if is_coastal_waves {
                            // 1. Breaker / rapids foam in high velocity ocean surge
                            let rapids_foam = if speed > 2.2 {
                                ((speed - 2.2) / 2.0).clamp(0.0, 0.85)
                            } else {
                                0.0
                            };

                            // 2. Shoreline lapping wave edge (wetting swash front)
                            let shore_foam = if depth < 0.06 && v < -0.10 {
                                ((-v - 0.10) / 0.40).clamp(0.0, 0.70)
                            } else {
                                0.0
                            };

                            // 3. Obstacle collision foam (slamming into rocks/castle walls)
                            let dz_x = (state.z_bed[row_offset + x_next] - state.z_bed[row_offset + x_prev]) * 0.5;
                            let dz_y = (state.z_bed[y_next_offset + x] - state.z_bed[y_prev_offset + x]) * 0.5;
                            let obstacle_impact = -(u * dz_x + v * dz_y);
                            let obstacle_foam = if obstacle_impact > 0.15 {
                                ((obstacle_impact - 0.15) / 0.50).clamp(0.0, 0.75)
                            } else {
                                0.0
                            };

                            (rapids_foam + shore_foam + obstacle_foam).clamp(0.0, 0.92)
                        } else {
                            0.0
                        };

                        if foam > 0.001 {
                            let foam_r = 248.0;
                            let foam_g = 252.0;
                            let foam_b = 255.0;
                            row_slice[byte_idx] = (water_r * (1.0 - foam) + foam_r * foam).clamp(0.0, 255.0) as u8;
                            row_slice[byte_idx + 1] = (water_g * (1.0 - foam) + foam_g * foam).clamp(0.0, 255.0) as u8;
                            row_slice[byte_idx + 2] = (water_b * (1.0 - foam) + foam_b * foam).clamp(0.0, 255.0) as u8;
                        } else {
                            row_slice[byte_idx] = water_r.clamp(0.0, 255.0) as u8;
                            row_slice[byte_idx + 1] = water_g.clamp(0.0, 255.0) as u8;
                            row_slice[byte_idx + 2] = water_b.clamp(0.0, 255.0) as u8;
                        }
                        row_slice[byte_idx + 3] = 255;
                    } else {
                        // --- DRY / EXPOSED LAND PASS ---
                        let mut final_r = r_land;
                        let mut final_g = g_land;
                        let mut final_b = b_land;

                        // Wet Sand Specular Gloss ("Mirror Beach")
                        if sat > 0.40 {
                            let dz_x = (state.z_bed[row_offset + x_next] - state.z_bed[row_offset + x_prev]) * 0.5;
                            let dz_y = (state.z_bed[y_next_offset + x] - state.z_bed[y_prev_offset + x]) * 0.5;
                            let land_nx = -dz_x * 2.0;
                            let land_ny = -dz_y * 2.0;
                            let land_len = (land_nx * land_nx + land_ny * land_ny + 1.0).sqrt();
                            let land_dot_h = ((land_nx / land_len) * (-0.209) + (land_ny / land_len) * (-0.247) + (1.0 / land_len) * 0.946).max(0.0);
                            let wet_spec = land_dot_h.powi(22) * (sat - 0.40) * 2.2;

                            final_r = (final_r + 210.0 * wet_spec).min(255.0);
                            final_g = (final_g + 225.0 * wet_spec).min(255.0);
                            final_b = (final_b + 245.0 * wet_spec).min(255.0);
                        }

                        row_slice[byte_idx] = final_r as u8;
                        row_slice[byte_idx + 1] = final_g as u8;
                        row_slice[byte_idx + 2] = final_b as u8;
                        row_slice[byte_idx + 3] = 255;
                    }
                }
            });

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

        // --- 7. Flow Visualization (Streamlines & Tracer Particles) ---
        let screen_w = screen_width();
        let screen_h = screen_height();
        let frame_dt = get_frame_time().min(0.05);

        // Mode: Flow Lines (Grid Vectors)
        if flow_vis_mode == FlowVisMode::Both || flow_vis_mode == FlowVisMode::Vectors {
            let step: usize = (20.0 / camera.zoom.sqrt()).max(10.0) as usize;
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
                            let (sx, sy) = camera.grid_to_screen(
                                gx as f32 + 0.5,
                                gy as f32 + 0.5,
                                screen_w,
                                screen_h,
                                desc.grid_res_x as f32,
                                desc.grid_res_y as f32,
                            );
                            if sx >= -20.0 && sx <= screen_w + 20.0 && sy >= -20.0 && sy <= screen_h + 20.0 {
                                let dir_x = u / speed;
                                let dir_y = v / speed;
                                let len = (speed * 12.0 * camera.zoom.sqrt()).clamp(4.0, 36.0);
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

                let (sx0, sy0) = camera.grid_to_screen(
                    p.prev_x,
                    p.prev_y,
                    screen_w,
                    screen_h,
                    desc.grid_res_x as f32,
                    desc.grid_res_y as f32,
                );
                let (sx1, sy1) = camera.grid_to_screen(
                    p.x,
                    p.y,
                    screen_w,
                    screen_h,
                    desc.grid_res_x as f32,
                    desc.grid_res_y as f32,
                );

                if (sx0 >= -20.0 && sx0 <= screen_w + 20.0 && sy0 >= -20.0 && sy0 <= screen_h + 20.0)
                    || (sx1 >= -20.0 && sx1 <= screen_w + 20.0 && sy1 >= -20.0 && sy1 <= screen_h + 20.0)
                {
                    let dist_sq = (sx1 - sx0).powi(2) + (sy1 - sy0).powi(2);
                    if dist_sq < (60.0 * camera.zoom).powi(2) {
                        let life_alpha = (p.life / p.max_life).clamp(0.0, 1.0);
                        let speed_alpha = (p.speed / 1.0).clamp(0.25, 0.95);
                        let alpha = life_alpha * speed_alpha;
                        draw_line(sx0, sy0, sx1, sy1, (1.8 * camera.zoom.sqrt()).clamp(1.6, 4.0), Color::new(0.88, 0.96, 1.0, alpha));
                        if speed_alpha > 0.40 {
                            draw_circle(sx1, sy1, (1.2 * camera.zoom.sqrt()).clamp(1.2, 3.0), Color::new(1.0, 1.0, 1.0, alpha * 0.90));
                        }
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
        let hud_height = if current_preset == ActivePreset::BeachWaves { 186.0 } else { 166.0 };
        draw_rectangle(8.0, 8.0, 680.0, hud_height, Color::new(0.0, 0.0, 0.0, 0.84));

        draw_text(
            format!(
                "FPS: {} | Backend: {} | Grid: {}x{} ({:.2}M cells)",
                get_fps(),
                sim.backend_name(),
                desc.grid_res_x,
                desc.grid_res_y,
                (desc.grid_res_x * desc.grid_res_y) as f32 / 1_000_000.0
            ).as_str(),
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
            format!(
                "Camera: [Scroll] Zoom ({:.1}x) | [MMB Drag / Arrows] Pan | [C] Reset View",
                camera.zoom
            ).as_str(),
            16.0,
            82.0,
            13.0,
            Color::new(0.35, 0.88, 1.0, 1.0),
        );

        draw_text(
            "Keys: [1..4] Presets | [G] Density (256..2048) | [R] Reset | [Space] Swap Backend",
            16.0,
            98.0,
            13.0,
            LIGHTGRAY,
        );

        draw_text(
            format!(
                "Flow Vis: {} ([V] cycle)",
                flow_vis_mode.label()
            ).as_str(),
            16.0,
            114.0,
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
            130.0,
            13.0,
            SKYBLUE,
        );

        draw_text(
            format!("CFL Max dt: {:.4}s | Sub-step dt: {:.4}s", safe_dt, current_sub_dt).as_str(),
            16.0,
            146.0,
            12.0,
            DARKGRAY,
        );

        if current_preset == ActivePreset::BeachWaves {
            let t = sim.sim_time();
            let wave_period = match sim.boundaries().south {
                sim_core::boundary::EdgeBoundary::WaveGenerator { wave_period, .. } => wave_period,
                _ => 25.0f32,
            };
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
                166.0,
                13.0,
                ORANGE,
            );
        }

        next_frame().await
    }
}
