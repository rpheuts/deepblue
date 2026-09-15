use macroquad::prelude::*;
use sim_core::domain::SimDomainDescriptor;
use sim_core::backend::SimulationBackend;
use crate::camera::CameraState;
use crate::types::{ActivePreset, FlowVisMode};

pub fn render_hud(
    sim: &Box<dyn SimulationBackend>,
    desc: &SimDomainDescriptor,
    camera: &CameraState,
    current_preset: ActivePreset,
    flow_vis_mode: FlowVisMode,
    is_inflow_active: bool,
    is_paused: bool,
    cached_fluid_mass: f32,
    cached_sed_mass: f32,
    cached_max_c: f32,
    safe_dt: f32,
    current_sub_dt: f32,
) {
    let hud_height = if current_preset == ActivePreset::BeachWaves || current_preset == ActivePreset::BeachWaves2 {
        186.0
    } else {
        166.0
    };
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
        ActivePreset::BeachWaves => "Coastal Beach 1 (Standard Waves)",
        ActivePreset::BeachWaves2 => "Beach 2 (High Relief Coastal Waves)",
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
        "Keys: [1..5] Presets ([2] Beach 1, [5] Beach 2) | [G] Density | [R] Reset | [Space] Swap Backend",
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

    if current_preset == ActivePreset::BeachWaves || current_preset == ActivePreset::BeachWaves2 {
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
}
