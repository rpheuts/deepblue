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
    DamBreak,
    MeanderingRiver,
}

pub struct UiState {
    pub active_tool: ActiveTool,
    pub brush_radius: f32,
    pub brush_strength: f32,
    pub is_paused: bool,
    pub sim_speed: f32,
    pub selected_scenario: SelectedScenario,
    pub trigger_scenario_reset: bool,

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
            brush_radius: 1.2,
            brush_strength: 0.15,
            is_paused: false,
            sim_speed: 1.0,
            selected_scenario: SelectedScenario::BeachSandcastleWaves,
            trigger_scenario_reset: false,
            fps: 0.0,
            frame_time_ms: 0.0,
            sim_tick_ms: 0.0,
            grid_res: (1024, 1024),
            camera_mode_name: "Perspective (3D Orbit)",
        }
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
                ui.separator();

                // Scenario Selector
                ui.label("Scenario:");
                ui.horizontal(|ui| {
                    if ui.selectable_label(state.selected_scenario == SelectedScenario::BeachSandcastleWaves, "🏖 Beach Waves").clicked() {
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

                ui.add(egui::Slider::new(&mut state.brush_radius, 0.2..=5.0).text("Brush Radius (m)"));
                ui.add(egui::Slider::new(&mut state.brush_strength, 0.01..=0.50).text("Strength / Flow"));

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
