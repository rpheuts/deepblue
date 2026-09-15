use std::sync::Arc;
use std::time::Instant;
use glam::Vec2;
use winit::dpi::LogicalSize;
use winit::event::{Event, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::WindowBuilder;

use sim_core::domain::SimDomainDescriptor;
use sim_core::scenario::Scenarios;
use sim_backend::WgpuSimulator;

mod camera;
mod context;
mod input;
mod mesh;
mod passes;
mod raycast;

use camera::Camera;
use context::RenderContext;
use input::InputState;
use mesh::{GridMesh, SkirtMesh};
use passes::{ActiveTool, DecalPass, SelectedScenario, SkirtPass, TerrainPass, UiPass, UiState, WaterPass};
use raycast::HeightfieldRaycaster;

fn main() {
    let event_loop = EventLoop::new().expect("Failed to create event loop");
    let window = Arc::new(
        WindowBuilder::new()
            .with_title("🌊 DeepBlue 3D Hydraulic Engine")
            .with_inner_size(LogicalSize::new(1600, 1000))
            .with_min_inner_size(LogicalSize::new(800, 600))
            .build(&event_loop)
            .expect("Failed to build window"),
    );

    let mut context = pollster::block_on(RenderContext::new(window.clone()));

    // 1. Simulation Setup (Decoupled Grid Geometry: 100m x 100m at 1024x1024 resolution)
    let domain_res = 1024;
    let desc = SimDomainDescriptor {
        extent_x: 100.0,
        extent_y: 100.0,
        max_elevation: 12.0,
        grid_res_x: domain_res,
        grid_res_y: domain_res,
        world_origin: [0.0, 0.0, 0.0],
    };
    let initial_grid = Scenarios::beach2(desc);

    let mut sim = WgpuSimulator::from_device(context.device.clone(), context.queue.clone(), initial_grid);

    // 2. Meshes (Hardware Scalable: 256x256 regular grid patch displaced by heightfield)
    let grid_mesh = GridMesh::new(&context.device, 256, 256);
    let skirt_mesh = SkirtMesh::new(&context.device, 64);

    // 3. Render Passes
    let terrain_pass = TerrainPass::new(
        &context.device,
        context.config.format,
        &context.camera_bind_group_layout,
        context.float32_filterable,
    );
    let skirt_pass = SkirtPass::new(
        &context.device,
        context.config.format,
        &context.camera_bind_group_layout,
        context.float32_filterable,
    );
    let water_pass = WaterPass::new(
        &context.device,
        context.config.format,
        &context.camera_bind_group_layout,
        context.float32_filterable,
    );
    let decal_pass = DecalPass::new(
        &context.device,
        context.config.format,
        &context.camera_bind_group_layout,
        context.float32_filterable,
    );
    let mut ui_pass = UiPass::new(&window, &context.device, context.config.format);

    // 4. Zero-Copy Texture Bind Groups
    let mut terrain_bg = terrain_pass.create_bind_group(
        &context.device,
        sim.elevation_view(),
        sim.water_view(),
        sim.sed_sat_view(),
        &context.sampler_linear,
    );
    let mut skirt_bg = skirt_pass.create_bind_group(
        &context.device,
        sim.elevation_view(),
        &context.sampler_linear,
    );
    let mut water_bg = water_pass.create_bind_group(
        &context.device,
        sim.elevation_view(),
        sim.water_view(),
        sim.velocity_view(),
        sim.sed_sat_view(),
        &context.sampler_linear,
    );
    let mut decal_tex_bg = decal_pass.create_tex_bind_group(
        &context.device,
        sim.elevation_view(),
        &context.sampler_linear,
    );

    // 5. Camera, Input, & UI Controller State
    let mut camera = Camera::new(desc.extent_x, desc.extent_y);
    let mut input = InputState::new();
    let mut ui_state = UiState::default();
    ui_state.grid_res = (desc.grid_res_x, desc.grid_res_y);

    let start_time = Instant::now();
    let mut last_frame_time = Instant::now();
    let mut frame_count = 0u32;
    let mut fps_timer = Instant::now();
    let mut sim_accumulator = 0.0f32;

    event_loop.set_control_flow(ControlFlow::Poll);

    let _ = event_loop.run(move |event, target| {
        match event {
            Event::WindowEvent { event, window_id } if window_id == window.id() => {
                let consumed = ui_pass.handle_event(&window, &event);

                match event {
                    WindowEvent::CloseRequested => {
                        target.exit();
                    }

                    WindowEvent::Resized(physical_size) => {
                        context.resize(physical_size.width, physical_size.height);
                        camera.aspect = (physical_size.width as f32) / (physical_size.height as f32).max(1.0);
                    }

                    WindowEvent::RedrawRequested => {
                        let now = Instant::now();
                        let dt_frame = now.duration_since(last_frame_time).as_secs_f32().min(0.10);
                        last_frame_time = now;

                        // Telemetry update (2 Hz)
                        frame_count += 1;
                        if fps_timer.elapsed().as_secs_f32() >= 0.5 {
                            ui_state.fps = (frame_count as f32) / fps_timer.elapsed().as_secs_f32();
                            ui_state.frame_time_ms = dt_frame * 1000.0;
                            frame_count = 0;
                            fps_timer = Instant::now();
                        }

                        // Handle Scenario Reset / Switch
                        if ui_state.trigger_scenario_reset {
                            ui_state.trigger_scenario_reset = false;
                            let new_grid = match ui_state.selected_scenario {
                                SelectedScenario::Beach2 => Scenarios::beach2(desc),
                                SelectedScenario::BeachSandcastleWaves => Scenarios::beach_sandcastle_waves(desc),
                                SelectedScenario::DamBreak => Scenarios::dam_break(desc),
                                SelectedScenario::MeanderingRiver => Scenarios::beach_stream(desc),
                            };
                            sim = WgpuSimulator::from_device(context.device.clone(), context.queue.clone(), new_grid);
                            terrain_bg = terrain_pass.create_bind_group(
                                &context.device,
                                sim.elevation_view(),
                                sim.water_view(),
                                sim.sed_sat_view(),
                                &context.sampler_linear,
                            );
                            skirt_bg = skirt_pass.create_bind_group(
                                &context.device,
                                sim.elevation_view(),
                                &context.sampler_linear,
                            );
                            water_bg = water_pass.create_bind_group(
                                &context.device,
                                sim.elevation_view(),
                                sim.water_view(),
                                sim.velocity_view(),
                                sim.sed_sat_view(),
                                &context.sampler_linear,
                            );
                            decal_tex_bg = decal_pass.create_tex_bind_group(
                                &context.device,
                                sim.elevation_view(),
                                &context.sampler_linear,
                            );
                        }

                        // Simulation Tick (Fixed-Timestep Accumulator for Temporal Decoupling)
                        if !ui_state.is_paused {
                            let wind_dir = ui_state.wind_dir();
                            sim.set_wind_full(
                                ui_state.wind_speed,
                                wind_dir[0],
                                wind_dir[1],
                                ui_state.wind_turbulence,
                                ui_state.wind_shelter,
                            );

                            let sim_start = Instant::now();
                            sim_accumulator += dt_frame * ui_state.sim_speed;
                            let fixed_dt = 1.0 / 60.0;
                            let mut steps = 0;
                            while sim_accumulator >= fixed_dt && steps < 4 {
                                sim.step_subdivided(fixed_dt, 0.004);
                                sim_accumulator -= fixed_dt;
                                steps += 1;
                            }
                            ui_state.sim_tick_ms = sim_start.elapsed().as_secs_f32() * 1000.0;
                        }

                        // 3D Terrain Raycasting & Tool Interaction
                        let screen_size = Vec2::new(context.config.width as f32, context.config.height as f32);
                        let (ray_origin, ray_dir) = camera.screen_to_ray(input.cursor_screen_pos, screen_size);
                        let hit_pos = if input.cursor_in_window {
                            HeightfieldRaycaster::intersect(
                                ray_origin,
                                ray_dir,
                                desc.extent_x,
                                desc.extent_y,
                                desc.grid_res_x,
                                desc.grid_res_y,
                                &sim.cpu_grid.current.z_bed,
                            )
                        } else {
                            None
                        };

                        // Update Decal Brush Ring & Apply Active Tool
                        if let Some(hit) = hit_pos {
                            decal_pass.update(
                                &context.queue,
                                hit.x,
                                hit.y,
                                ui_state.brush_radius,
                                true,
                                ui_state.active_tool.color(),
                            );

                            if input.is_left_down {
                                let strength = ui_state.brush_strength;
                                let tool = if input.is_ctrl_down {
                                    ActiveTool::StoneBreakwater
                                } else if input.is_shift_down {
                                    ActiveTool::SandDam
                                } else {
                                    ui_state.active_tool
                                };

                                let tool_type = match tool {
                                    ActiveTool::Water => 1,
                                    ActiveTool::SandDam => 2,
                                    ActiveTool::StoneBreakwater => 3,
                                    ActiveTool::Dig => 4,
                                };

                                // Dispatch in-situ GPU compute brush with automatic host mirror sync
                                sim.apply_brush(
                                    hit.x,
                                    hit.y,
                                    ui_state.brush_radius,
                                    strength,
                                    tool_type,
                                );
                            }
                        } else {
                            decal_pass.update(&context.queue, 0.0, 0.0, 1.0, false, [0.0; 4]);
                        }

                        // Update Camera Uniforms
                        let total_time = start_time.elapsed().as_secs_f32();
                        let camera_uniforms = camera.create_uniforms(
                            total_time,
                            desc.extent_x,
                            desc.extent_y,
                            ui_state.wind_uniform(),
                            ui_state.wind_turb_uniform(),
                        );
                        context.update_camera(&camera_uniforms);

                        // Build egui UI
                        ui_pass.begin_frame(&window);
                        ui_pass.build_ui(&mut ui_state);

                        // Render Frame
                        let output = match context.surface.get_current_texture() {
                            Ok(frame) => frame,
                            Err(wgpu::SurfaceError::Lost) => {
                                context.resize(context.config.width, context.config.height);
                                return;
                            }
                            Err(wgpu::SurfaceError::OutOfMemory) => {
                                target.exit();
                                return;
                            }
                            Err(_) => return,
                        };

                        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());

                        let mut encoder = context.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("Primary Frame Encoder"),
                        });

                        {
                            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                                label: Some("3D World Render Pass"),
                                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                    view: &view,
                                    resolve_target: None,
                                    ops: wgpu::Operations {
                                        load: wgpu::LoadOp::Clear(wgpu::Color {
                                            r: 0.045,
                                            g: 0.055,
                                            b: 0.075,
                                            a: 1.0,
                                        }),
                                        store: wgpu::StoreOp::Store,
                                    },
                                })],
                                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                                    view: &context.depth_view,
                                    depth_ops: Some(wgpu::Operations {
                                        load: wgpu::LoadOp::Clear(1.0),
                                        store: wgpu::StoreOp::Store,
                                    }),
                                    stencil_ops: None,
                                }),
                                occlusion_query_set: None,
                                timestamp_writes: None,
                            });

                            // 1. Skirt / Diorama Perimeter Pass
                            skirt_pass.render(&mut rpass, &context.camera_bind_group, &skirt_bg, &skirt_mesh);

                            // 2. Terrain Mesh Pass
                            terrain_pass.render(&mut rpass, &context.camera_bind_group, &terrain_bg, &grid_mesh);

                            // 3. Transparent Dynamic Water Pass (Beer-Lambert optics, caustics, sun glitter)
                            water_pass.render(&mut rpass, &context.camera_bind_group, &water_bg, &grid_mesh);

                            // 4. Decal Brush Ring Cursor Pass
                            decal_pass.render(&mut rpass, &context.camera_bind_group, &decal_tex_bg, &grid_mesh);
                        }

                        // 5. egui Telemetry & Tool Overlay Pass
                        ui_pass.render(
                            &context.device,
                            &context.queue,
                            &mut encoder,
                            &window,
                            &view,
                            context.config.width,
                            context.config.height,
                        );

                        context.queue.submit(Some(encoder.finish()));
                        output.present();

                        window.request_redraw();
                    }

                    other_event => {
                        input.handle_event(&other_event, consumed, &mut camera, &mut ui_state, &desc);
                    }
                }
            }

            Event::AboutToWait => {
                window.request_redraw();
            }

            _ => {}
        }
    });
}
