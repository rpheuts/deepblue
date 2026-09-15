use std::sync::Arc;
use egui_wgpu::{Renderer, ScreenDescriptor};
use egui_winit::State;
use winit::window::Window;

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum ActiveTool {
    Water,
    SandDam,
    StoneBreakwater,
    Dig,
}

impl ActiveTool {
    pub fn color(&self) -> [f32; 4] {
        match self {
            ActiveTool::Water => [0.15, 0.55, 0.95, 0.85],           // Azure blue
            ActiveTool::SandDam => [0.88, 0.68, 0.38, 0.85],         // Warm sand orange
            ActiveTool::StoneBreakwater => [0.45, 0.52, 0.60, 0.90],  // Stone slate
            ActiveTool::Dig => [0.90, 0.25, 0.20, 0.85],             // Excavation red
        }
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum SelectedScenario {
    BeachSandcastleWaves,
    Beach2,
    DamBreak,
    MeanderingRiver,
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum FpsLimit {
    Limit30,
    Limit60,
    Limit120,
    Unlimited,
}

impl FpsLimit {
    pub fn target_fps(&self) -> Option<u32> {
        match self {
            FpsLimit::Limit30 => Some(30),
            FpsLimit::Limit60 => Some(60),
            FpsLimit::Limit120 => Some(120),
            FpsLimit::Unlimited => None,
        }
    }
}

pub struct UiState {
    pub active_tool: ActiveTool,
    pub brush_radius: f32,
    pub brush_strength: f32,
    pub is_paused: bool,
    pub sim_speed: f32,
    pub selected_scenario: SelectedScenario,
    pub trigger_scenario_reset: bool,

    // Atmospheric wind & ocean sea state
    pub wind_speed: f32,
    pub wind_angle_deg: f32,
    pub wave_chop_factor: f32,
    pub wind_turbulence: f32,
    pub wind_shelter: f32,
    pub caustics_intensity: f32,

    // Hydraulic conduits (Stream Inflow & Coastal Sink)
    pub stream_inflow_enabled: bool,
    pub coastal_sink_enabled: bool,

    // Frame pacing / FPS limit
    pub fps_limit: FpsLimit,

    // Telemetry
    pub fps: f32,
    pub frame_time_ms: f32,
    pub sim_tick_ms: f32,
    pub grid_res: (u32, u32),
    pub camera_mode_name: &'static str,
}

impl Default for UiState {
    fn default() -> Self {
        Self {
            active_tool: ActiveTool::Water,
            brush_radius: 1.8,
            brush_strength: 0.25,
            is_paused: false,
            sim_speed: 1.0,
            selected_scenario: SelectedScenario::Beach2,
            trigger_scenario_reset: false,
            wind_speed: 9.0,
            wind_angle_deg: 0.0,
            wave_chop_factor: 1.0,
            wind_turbulence: 0.45,
            wind_shelter: 0.70,
            caustics_intensity: 1.25,
            stream_inflow_enabled: true,
            coastal_sink_enabled: true,
            fps_limit: FpsLimit::Limit60,
            fps: 0.0,
            frame_time_ms: 0.0,
            sim_tick_ms: 0.0,
            grid_res: (1024, 1024),
            camera_mode_name: "Perspective (Orbit)",
        }
    }
}

impl UiState {
    pub fn wind_dir(&self) -> [f32; 2] {
        let rad = self.wind_angle_deg.to_radians();
        [rad.sin(), -rad.cos()]
    }

    pub fn wind_uniform(&self) -> [f32; 4] {
        let dir = self.wind_dir();
        [dir[0], dir[1], self.wind_speed, self.wave_chop_factor]
    }

    pub fn wind_turb_uniform(&self) -> [f32; 4] {
        [self.wind_turbulence, self.wind_shelter, self.caustics_intensity, 0.0]
    }
}

pub struct UiPass {
    pub egui_ctx: egui::Context,
    pub egui_winit_state: State,
    pub renderer: Renderer,
}

impl UiPass {
    pub fn new(
        window: &Arc<Window>,
        device: &wgpu::Device,
        output_format: wgpu::TextureFormat,
    ) -> Self {
        let egui_ctx = egui::Context::default();
        egui_ctx.set_visuals(egui::Visuals::dark());

        let viewport_id = egui_ctx.viewport_id();
        let egui_winit_state = State::new(
            egui_ctx.clone(),
            viewport_id,
            window,
            Some(window.scale_factor() as f32),
            None,
        );

        let renderer = Renderer::new(device, output_format, None, 1);

        Self {
            egui_ctx,
            egui_winit_state,
            renderer,
        }
    }

    pub fn handle_event(&mut self, window: &Window, event: &winit::event::WindowEvent) -> bool {
        let response = self.egui_winit_state.on_window_event(window, event);
        response.consumed
    }

    pub fn build_ui(&self, state: &mut UiState) {
        egui::Window::new("🌊 DeepBlue 3D Simulation Engine")
            .default_width(340.0)
            .resizable(false)
            .show(&self.egui_ctx, |ui| {
                ui.heading("Hydraulic Sandbox Platform");
                ui.separator();

                // Telemetry section
                ui.label(format!("Resolution: {}x{} cells", state.grid_res.0, state.grid_res.1));
                ui.horizontal(|ui| {
                    ui.label(format!("FPS: {:.1}", state.fps));
                    ui.label(format!("Frame: {:.2} ms", state.frame_time_ms));
                    ui.label(format!("Sim Tick: {:.2} ms", state.sim_tick_ms));
                });
                ui.label(format!("Camera Mode: {}", state.camera_mode_name));
                ui.horizontal(|ui| {
                    ui.label("FPS Limit:");
                    ui.selectable_value(&mut state.fps_limit, FpsLimit::Limit30, "30");
                    ui.selectable_value(&mut state.fps_limit, FpsLimit::Limit60, "60");
                    ui.selectable_value(&mut state.fps_limit, FpsLimit::Limit120, "120");
                    ui.selectable_value(&mut state.fps_limit, FpsLimit::Unlimited, "Unlimited");
                });
                ui.separator();

                // Scenario Selector
                ui.label("Scenario:");
                ui.horizontal(|ui| {
                    if ui.selectable_label(state.selected_scenario == SelectedScenario::Beach2, "🌊 Beach 2 (High Relief)").clicked() {
                        state.selected_scenario = SelectedScenario::Beach2;
                        state.trigger_scenario_reset = true;
                    }
                    if ui.selectable_label(state.selected_scenario == SelectedScenario::BeachSandcastleWaves, "🏖 Beach 1").clicked() {
                        state.selected_scenario = SelectedScenario::BeachSandcastleWaves;
                        state.trigger_scenario_reset = true;
                    }
                    if ui.selectable_label(state.selected_scenario == SelectedScenario::DamBreak, "💥 Dam Break").clicked() {
                        state.selected_scenario = SelectedScenario::DamBreak;
                        state.trigger_scenario_reset = true;
                    }
                    if ui.selectable_label(state.selected_scenario == SelectedScenario::MeanderingRiver, "🏞 River").clicked() {
                        state.selected_scenario = SelectedScenario::MeanderingRiver;
                        state.trigger_scenario_reset = true;
                    }
                });

                ui.separator();

                // Simulation Controls
                ui.horizontal(|ui| {
                    let pause_btn = if state.is_paused { "▶ Resume" } else { "⏸ Pause" };
                    if ui.button(pause_btn).clicked() {
                        state.is_paused = !state.is_paused;
                    }
                    if ui.button("🔄 Reset Scenario").clicked() {
                        state.trigger_scenario_reset = true;
                    }
                });
                ui.add(egui::Slider::new(&mut state.sim_speed, 0.25..=4.0).text("Sim Speed Multiplier"));
                if state.selected_scenario == SelectedScenario::MeanderingRiver {
                    ui.separator();
                    ui.label("🏞 River Hydraulics:");
                    ui.horizontal(|ui| {
                        ui.checkbox(&mut state.stream_inflow_enabled, "Continuous Inflow (North)");
                        ui.checkbox(&mut state.coastal_sink_enabled, "Ocean Sink (South)");
                    });
                }

                ui.separator();

                // Interactive Tools
                ui.heading("Sculpting & Hydraulic Tools");
                ui.horizontal(|ui| {
                    ui.selectable_value(&mut state.active_tool, ActiveTool::Water, "💧 Water [LMB]");
                    ui.selectable_value(&mut state.active_tool, ActiveTool::SandDam, "🏖 Dam [Shift+LMB]");
                });
                ui.horizontal(|ui| {
                    ui.selectable_value(&mut state.active_tool, ActiveTool::StoneBreakwater, "🧱 Stone [Ctrl+LMB]");
                    ui.selectable_value(&mut state.active_tool, ActiveTool::Dig, "⛏ Dig [RMB]");
                });

                ui.add(egui::Slider::new(&mut state.brush_radius, 0.2..=8.0).text("Brush Radius (m)"));
                ui.add(egui::Slider::new(&mut state.brush_strength, 0.01..=1.00).text("Strength / Flow"));

                ui.separator();

                // Atmospheric & Ocean Wind Controls
                ui.heading("🌬 Wind & Ocean Sea State");
                ui.add(egui::Slider::new(&mut state.wind_speed, 0.0..=25.0).text("Wind Speed (m/s)"));
                ui.add(egui::Slider::new(&mut state.wind_angle_deg, 0.0..=360.0).text("Wind Heading (deg)"));
                ui.add(egui::Slider::new(&mut state.wave_chop_factor, 0.0..=2.5).text("Wave Chop Scale"));
                ui.add(egui::Slider::new(&mut state.wind_turbulence, 0.0..=1.0).text("Turbulence / Gusts"));
                ui.add(egui::Slider::new(&mut state.wind_shelter, 0.0..=1.0).text("Terrain Sheltering (Lee)"));
                ui.add(egui::Slider::new(&mut state.caustics_intensity, 0.0..=2.5).text("Caustics Intensity"));

                ui.separator();
                ui.collapsing("Navigation Shortcuts", |ui| {
                    ui.label("• Left Drag: Orbit Camera");
                    ui.label("• Right Drag: Pan Camera");
                    ui.label("• Scroll Wheel: Zoom Camera");
                    ui.label("• [Tab]: Toggle 3D Perspective / 2.5D Isometric");
                    ui.label("• [R]: Reset Camera to Default Frame");
                    ui.label("• [Space]: Pause / Resume Simulation");
                });
            });
    }

    pub fn begin_frame(&mut self, window: &Window) {
        let raw_input = self.egui_winit_state.take_egui_input(window);
        self.egui_ctx.begin_frame(raw_input);
    }

    pub fn render(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        window: &Window,
        view: &wgpu::TextureView,
        screen_width: u32,
        screen_height: u32,
    ) {
        let full_output = self.egui_ctx.end_frame();

        self.egui_winit_state.handle_platform_output(window, full_output.platform_output);

        let pixels_per_point = self.egui_ctx.pixels_per_point();
        let clipped_primitives = self.egui_ctx.tessellate(full_output.shapes, pixels_per_point);

        let screen_descriptor = ScreenDescriptor {
            size_in_pixels: [screen_width, screen_height],
            pixels_per_point,
        };

        for (id, image_delta) in &full_output.textures_delta.set {
            self.renderer.update_texture(device, queue, *id, image_delta);
        }

        self.renderer.update_buffers(
            device,
            queue,
            encoder,
            &clipped_primitives,
            &screen_descriptor,
        );

        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("egui Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });

            self.renderer.render(&mut rpass, &clipped_primitives, &screen_descriptor);
        }

        for id in &full_output.textures_delta.free {
            self.renderer.free_texture(id);
        }
    }
}
