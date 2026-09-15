use std::sync::Arc;
use sim_core::state::DoubleBufferedGrid;

pub struct WgpuSimulator {
    pub(crate) device: Arc<wgpu::Device>,
    pub(crate) queue: Arc<wgpu::Queue>,

    // Pipelines
    pub(crate) swe_pipeline: wgpu::ComputePipeline,
    pub(crate) swe_boundary_pipeline: wgpu::ComputePipeline,
    pub(crate) sat_pipeline: wgpu::ComputePipeline,
    pub(crate) exner_pipeline: wgpu::ComputePipeline,
    pub(crate) talus_pipeline: wgpu::ComputePipeline,
    pub(crate) sed_boundary_pipeline: wgpu::ComputePipeline,
    pub(crate) export_pipeline: wgpu::ComputePipeline,
    pub(crate) brush_pipeline: wgpu::ComputePipeline,

    // Bind groups (ping-pong [0] for read 0 -> write 1, [1] for read 1 -> write 0)
    pub(crate) bg_swe: [wgpu::BindGroup; 2],
    pub(crate) bg_sat: [wgpu::BindGroup; 2],
    pub(crate) bg_exner: [wgpu::BindGroup; 2],
    pub(crate) bg_talus: [wgpu::BindGroup; 2],
    pub(crate) bg_sed_boundary: [wgpu::BindGroup; 2],
    pub(crate) bg_export: [wgpu::BindGroup; 2],
    pub(crate) bg_brush: wgpu::BindGroup,

    // Exportable GPU 2D Textures (Zero-Copy VRAM - AGENTS.md contract)
    pub(crate) tex_elevation: wgpu::Texture,
    pub(crate) view_elevation: wgpu::TextureView,
    pub(crate) tex_water: wgpu::Texture,
    pub(crate) view_water: wgpu::TextureView,
    pub(crate) tex_velocity: wgpu::Texture,
    pub(crate) view_velocity: wgpu::TextureView,
    pub(crate) tex_sed_sat: wgpu::Texture,
    pub(crate) view_sed_sat: wgpu::TextureView,

    // Cached CPU state for readback and interactions
    pub cpu_grid: DoubleBufferedGrid,

    // Continuous stream and coastal sink settings
    pub stream_inflow_active: bool,
    pub coastal_sink_active: bool,

    // Atmospheric wind forcing
    pub wind_speed: f32,
    pub wind_dir: [f32; 2],
    pub wind_turbulence: f32,
    pub wind_shelter: f32,

    // GPU Uniform Buffers
    pub(crate) _buf_domain: wgpu::Buffer,
    pub(crate) buf_dt: wgpu::Buffer,
    pub(crate) buf_brush: wgpu::Buffer,

    // Ping-pong storage buffers
    pub(crate) buf_h: [wgpu::Buffer; 2],
    pub(crate) buf_u: [wgpu::Buffer; 2],
    pub(crate) buf_v: [wgpu::Buffer; 2],
    pub(crate) buf_z: [wgpu::Buffer; 2],
    pub(crate) buf_c: [wgpu::Buffer; 2],
    pub(crate) buf_sat: [wgpu::Buffer; 2],
    pub(crate) buf_bedrock: wgpu::Buffer,

    // Persistent staging buffers for readback
    pub(crate) staging_h: wgpu::Buffer,
    pub(crate) staging_u: wgpu::Buffer,
    pub(crate) staging_v: wgpu::Buffer,
    pub(crate) staging_z: wgpu::Buffer,
    pub(crate) staging_c: wgpu::Buffer,
    pub(crate) staging_sat: wgpu::Buffer,
    pub(crate) buffer_byte_size: u64,

    pub(crate) ping_pong: usize,
}

impl WgpuSimulator {
    /// Configures atmospheric wind speed (m/s) and horizontal vector (dir_x, dir_y).
    pub fn set_wind(&mut self, speed: f32, dir_x: f32, dir_y: f32) {
        self.set_wind_full(speed, dir_x, dir_y, self.wind_turbulence, self.wind_shelter);
    }

    /// Configures full atmospheric wind parameters including turbulence and terrain sheltering.
    pub fn set_wind_full(&mut self, speed: f32, dir_x: f32, dir_y: f32, turbulence: f32, shelter: f32) {
        self.wind_speed = speed.max(0.0);
        let len = (dir_x * dir_x + dir_y * dir_y).sqrt();
        if len > 1e-4 {
            self.wind_dir = [dir_x / len, dir_y / len];
        } else {
            self.wind_dir = [0.0, -1.0];
        }
        self.wind_turbulence = turbulence.clamp(0.0, 1.0);
        self.wind_shelter = shelter.clamp(0.0, 1.0);
    }

    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    pub fn elevation_view(&self) -> &wgpu::TextureView {
        &self.view_elevation
    }

    pub fn water_view(&self) -> &wgpu::TextureView {
        &self.view_water
    }

    pub fn velocity_view(&self) -> &wgpu::TextureView {
        &self.view_velocity
    }

    pub fn sed_sat_view(&self) -> &wgpu::TextureView {
        &self.view_sed_sat
    }

    pub fn active_buf_h(&self) -> &wgpu::Buffer {
        &self.buf_h[self.ping_pong]
    }

    pub fn active_buf_u(&self) -> &wgpu::Buffer {
        &self.buf_u[self.ping_pong]
    }

    pub fn active_buf_v(&self) -> &wgpu::Buffer {
        &self.buf_v[self.ping_pong]
    }

    pub fn active_buf_z(&self) -> &wgpu::Buffer {
        &self.buf_z[self.ping_pong]
    }

    pub fn active_buf_c(&self) -> &wgpu::Buffer {
        &self.buf_c[self.ping_pong]
    }

    pub fn active_buf_sat(&self) -> &wgpu::Buffer {
        &self.buf_sat[self.ping_pong]
    }

    pub fn buf_bedrock(&self) -> &wgpu::Buffer {
        &self.buf_bedrock
    }

    pub fn elevation_texture(&self) -> &wgpu::Texture {
        &self.tex_elevation
    }

    pub fn water_texture(&self) -> &wgpu::Texture {
        &self.tex_water
    }

    pub fn velocity_texture(&self) -> &wgpu::Texture {
        &self.tex_velocity
    }

    pub fn sed_sat_texture(&self) -> &wgpu::Texture {
        &self.tex_sed_sat
    }

    pub fn set_stream_inflow(&mut self, active: bool) {
        self.stream_inflow_active = active;
    }

    pub fn set_coastal_sink(&mut self, active: bool) {
        self.coastal_sink_active = active;
    }
}
