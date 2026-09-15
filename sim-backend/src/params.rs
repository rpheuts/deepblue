#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuStepParams {
    pub dt: f32,
    pub time: f32,
    pub south_type: u32,
    pub south_base_eta: f32,

    pub south_wave_amp: f32,
    pub south_wave_period: f32,
    pub south_surge_speed: f32,
    pub south_tide_amp: f32,

    pub south_tide_period: f32,
    pub south_outflow_rate: f32,
    pub north_type: u32,
    pub north_inflow_h: f32,

    pub north_inflow_v: f32,
    pub north_outflow_rate: f32,
    pub west_type: u32,
    pub east_type: u32,

    pub west_outflow_rate: f32,
    pub east_outflow_rate: f32,
    pub stream_inflow_active: f32,
    pub coastal_sink_active: f32,

    pub wind_speed: f32,
    pub wind_dir_x: f32,
    pub wind_dir_y: f32,
    pub wind_drag_coeff: f32,

    pub wind_turbulence: f32,
    pub wind_shelter: f32,
    pub _step_param_pad1: f32,
    pub _step_param_pad2: f32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BrushParams {
    pub center_x: f32,
    pub center_y: f32,
    pub radius: f32,
    pub strength: f32,
    pub tool_type: u32,
    pub grid_res_x: u32,
    pub grid_res_y: u32,
    pub extent_x: f32,
    pub extent_y: f32,
    pub pad0: f32,
    pub pad1: f32,
    pub pad2: f32,
}

impl GpuStepParams {
    pub fn new(
        dt: f32,
        time: f32,
        b: &sim_core::boundary::DomainBoundaryConfig,
        stream_inflow_active: bool,
        coastal_sink_active: bool,
    ) -> Self {
        let mut p = Self {
            dt,
            time,
            south_type: 0,
            south_base_eta: 0.0,
            south_wave_amp: 0.0,
            south_wave_period: 4.0,
            south_surge_speed: 0.0,
            south_tide_amp: 0.0,
            south_tide_period: 60.0,
            south_outflow_rate: 0.0,

            north_type: 0,
            north_inflow_h: 0.0,
            north_inflow_v: 0.0,
            north_outflow_rate: 0.0,

            west_type: 0,
            east_type: 0,
            west_outflow_rate: 0.0,
            east_outflow_rate: 0.0,
            stream_inflow_active: if stream_inflow_active { 1.0 } else { 0.0 },
            coastal_sink_active: if coastal_sink_active { 1.0 } else { 0.0 },
            wind_speed: 0.0,
            wind_dir_x: 0.0,
            wind_dir_y: -1.0,
            wind_drag_coeff: 0.00025,
            wind_turbulence: 0.45,
            wind_shelter: 0.70,
            _step_param_pad1: 0.0,
            _step_param_pad2: 0.0,
        };

        match b.south {
            sim_core::boundary::EdgeBoundary::SolidWall => p.south_type = 0,
            sim_core::boundary::EdgeBoundary::OpenOutflow { absorption_rate } => {
                p.south_type = 1;
                p.south_outflow_rate = absorption_rate;
            }
            sim_core::boundary::EdgeBoundary::ConstantInflow { target_depth, inflow_velocity } => {
                p.south_type = 2;
                p.south_base_eta = target_depth;
                p.south_surge_speed = inflow_velocity;
            }
            sim_core::boundary::EdgeBoundary::WaveGenerator {
                base_elevation,
                wave_amplitude,
                wave_period,
                surge_speed,
                tide_amplitude,
                tide_period,
            } => {
                p.south_type = 3;
                p.south_base_eta = base_elevation;
                p.south_wave_amp = wave_amplitude;
                p.south_wave_period = wave_period;
                p.south_surge_speed = surge_speed;
                p.south_tide_amp = tide_amplitude;
                p.south_tide_period = tide_period;
            }
        }

        match b.north {
            sim_core::boundary::EdgeBoundary::SolidWall => p.north_type = 0,
            sim_core::boundary::EdgeBoundary::OpenOutflow { absorption_rate } => {
                p.north_type = 1;
                p.north_outflow_rate = absorption_rate;
            }
            sim_core::boundary::EdgeBoundary::ConstantInflow { target_depth, inflow_velocity } => {
                p.north_type = 2;
                p.north_inflow_h = target_depth;
                p.north_inflow_v = inflow_velocity;
            }
            _ => p.north_type = 0,
        }

        match b.west {
            sim_core::boundary::EdgeBoundary::SolidWall => p.west_type = 0,
            sim_core::boundary::EdgeBoundary::OpenOutflow { absorption_rate } => {
                p.west_type = 1;
                p.west_outflow_rate = absorption_rate;
            }
            _ => p.west_type = 0,
        }

        match b.east {
            sim_core::boundary::EdgeBoundary::SolidWall => p.east_type = 0,
            sim_core::boundary::EdgeBoundary::OpenOutflow { absorption_rate } => {
                p.east_type = 1;
                p.east_outflow_rate = absorption_rate;
            }
            _ => p.east_type = 0,
        }

        p
    }
}
