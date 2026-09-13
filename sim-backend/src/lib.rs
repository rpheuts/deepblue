use sim_core::state::DoubleBufferedGrid;
use sim_core::backend::SimulationBackend;
use sim_core::domain::SimDomainDescriptor;
use sim_core::state::GridState;
use wgpu::util::DeviceExt;
use std::borrow::Cow;

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
    pub pad0: f32,
    pub pad1: f32,
}

impl GpuStepParams {
    pub fn new(dt: f32, time: f32, b: &sim_core::boundary::DomainBoundaryConfig) -> Self {
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
            pad0: 0.0,
            pad1: 0.0,
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

pub struct WgpuSimulator {
    device: wgpu::Device,
    queue: wgpu::Queue,
    
    // Pipelines
    swe_pipeline: wgpu::ComputePipeline,
    swe_boundary_pipeline: wgpu::ComputePipeline,
    sat_pipeline: wgpu::ComputePipeline,
    exner_pipeline: wgpu::ComputePipeline,
    talus_pipeline: wgpu::ComputePipeline,
    sed_boundary_pipeline: wgpu::ComputePipeline,
    
    // Bind groups (ping-pong [0] for read 0 -> write 1, [1] for read 1 -> write 0)
    bg_swe: [wgpu::BindGroup; 2],
    bg_sat: [wgpu::BindGroup; 2],
    bg_exner: [wgpu::BindGroup; 2],
    bg_talus: [wgpu::BindGroup; 2],
    bg_sed_boundary: [wgpu::BindGroup; 2],
    
    // Cached CPU state for readback and interactions
    pub cpu_grid: DoubleBufferedGrid,
    
    // GPU Uniform Buffers
    _buf_domain: wgpu::Buffer,
    buf_dt: wgpu::Buffer,
    
    // Ping-pong storage buffers
    buf_h: [wgpu::Buffer; 2],
    buf_u: [wgpu::Buffer; 2],
    buf_v: [wgpu::Buffer; 2],
    buf_z: [wgpu::Buffer; 2],
    buf_c: [wgpu::Buffer; 2],
    buf_sat: [wgpu::Buffer; 2],
    buf_bedrock: wgpu::Buffer,

    // Persistent staging buffers for readback
    staging_h: wgpu::Buffer,
    staging_u: wgpu::Buffer,
    staging_v: wgpu::Buffer,
    staging_z: wgpu::Buffer,
    staging_c: wgpu::Buffer,
    staging_sat: wgpu::Buffer,
    buffer_byte_size: u64,
    
    ping_pong: usize,
}

impl WgpuSimulator {
    /// Asynchronously initializes a new GPU-accelerated hydraulic & sediment simulator.
    pub async fn new(mut cpu_grid: DoubleBufferedGrid) -> Self {
        cpu_grid.current.apply_reflective_boundaries();
        cpu_grid.next.apply_reflective_boundaries();

        // Initialize WGPU (Force Primary backends to avoid EGL/GL conflicts with Macroquad)
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            ..Default::default()
        });
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                ..Default::default()
            })
            .await
            .expect("Failed to find wgpu adapter");

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default(), None)
            .await
            .expect("Failed to create wgpu device");

        // Load shader modules
        let swe_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("SWE Shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("shaders/swe.wgsl"))),
        });
        let sat_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Saturation Shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("shaders/sediment_sat.wgsl"))),
        });
        let exner_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Exner Shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("shaders/sediment_exner.wgsl"))),
        });
        let talus_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Talus Shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("shaders/sediment_talus.wgsl"))),
        });
        let sed_boundary_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Sediment Boundary Shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("shaders/sediment_boundary.wgsl"))),
        });

        // Compute pipelines
        let swe_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("SWE Compute Pipeline"),
            layout: None,
            module: &swe_shader,
            entry_point: "main",
            compilation_options: Default::default(),
        });
        let swe_bg_layout = swe_pipeline.get_bind_group_layout(0);
        let swe_pipe_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("SWE Pipeline Layout"),
            bind_group_layouts: &[&swe_bg_layout],
            push_constant_ranges: &[],
        });
        let swe_boundary_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("SWE Boundary Pipeline"),
            layout: Some(&swe_pipe_layout),
            module: &swe_shader,
            entry_point: "boundary",
            compilation_options: Default::default(),
        });

        let sat_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Sat Pipeline"),
            layout: None,
            module: &sat_shader,
            entry_point: "main",
            compilation_options: Default::default(),
        });
        let sat_bg_layout = sat_pipeline.get_bind_group_layout(0);

        let exner_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Exner Pipeline"),
            layout: None,
            module: &exner_shader,
            entry_point: "main",
            compilation_options: Default::default(),
        });
        let exner_bg_layout = exner_pipeline.get_bind_group_layout(0);

        let talus_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Talus Pipeline"),
            layout: None,
            module: &talus_shader,
            entry_point: "main",
            compilation_options: Default::default(),
        });
        let talus_bg_layout = talus_pipeline.get_bind_group_layout(0);

        let sed_boundary_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Sediment Boundary Pipeline"),
            layout: None,
            module: &sed_boundary_shader,
            entry_point: "main",
            compilation_options: Default::default(),
        });
        let sed_bg_layout = sed_boundary_pipeline.get_bind_group_layout(0);

        // Uniform buffers
        let buf_domain = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Domain Buffer"),
            contents: bytemuck::bytes_of(&cpu_grid.descriptor),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let initial_params = GpuStepParams::new(0.016, cpu_grid.time, &cpu_grid.boundaries);
        let buf_dt = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Step Params Buffer"),
            contents: bytemuck::bytes_of(&initial_params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Storage buffers
        let cell_count = (cpu_grid.descriptor.grid_res_x * cpu_grid.descriptor.grid_res_y) as usize;
        let buffer_byte_size = (cell_count * std::mem::size_of::<f32>()) as u64;

        let create_storage = |label: &str, data: &[f32]| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::cast_slice(data),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            })
        };

        let buf_h = [create_storage("h_0", &cpu_grid.current.h), create_storage("h_1", &cpu_grid.next.h)];
        let buf_u = [create_storage("u_0", &cpu_grid.current.u), create_storage("u_1", &cpu_grid.next.u)];
        let buf_v = [create_storage("v_0", &cpu_grid.current.v), create_storage("v_1", &cpu_grid.next.v)];
        let buf_z = [create_storage("z_0", &cpu_grid.current.z_bed), create_storage("z_1", &cpu_grid.next.z_bed)];
        let buf_c = [create_storage("c_0", &cpu_grid.current.sediment_c), create_storage("c_1", &cpu_grid.next.sediment_c)];
        let buf_sat = [create_storage("sat_0", &cpu_grid.current.soil_sat), create_storage("sat_1", &cpu_grid.next.soil_sat)];
        let buf_bedrock = create_storage("bedrock", &cpu_grid.current.bedrock_z);

        // Bind Groups
        let bg_swe = [
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("SWE BG 0->1"),
                layout: &swe_bg_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: buf_domain.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: buf_dt.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: buf_h[0].as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: buf_u[0].as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: buf_v[0].as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 5, resource: buf_z[0].as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 6, resource: buf_h[1].as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 7, resource: buf_u[1].as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 8, resource: buf_v[1].as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 9, resource: buf_z[1].as_entire_binding() },
                ],
            }),
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("SWE BG 1->0"),
                layout: &swe_bg_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: buf_domain.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: buf_dt.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: buf_h[1].as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: buf_u[1].as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: buf_v[1].as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 5, resource: buf_z[1].as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 6, resource: buf_h[0].as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 7, resource: buf_u[0].as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 8, resource: buf_v[0].as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 9, resource: buf_z[0].as_entire_binding() },
                ],
            }),
        ];

        let bg_sat = [
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Sat BG 0->1"),
                layout: &sat_bg_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: buf_domain.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: buf_dt.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: buf_h[1].as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: buf_sat[0].as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: buf_sat[1].as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 5, resource: buf_z[1].as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 6, resource: buf_bedrock.as_entire_binding() },
                ],
            }),
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Sat BG 1->0"),
                layout: &sat_bg_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: buf_domain.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: buf_dt.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: buf_h[0].as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: buf_sat[1].as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: buf_sat[0].as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 5, resource: buf_z[0].as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 6, resource: buf_bedrock.as_entire_binding() },
                ],
            }),
        ];

        let bg_exner = [
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Exner BG 0->1"),
                layout: &exner_bg_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: buf_domain.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: buf_dt.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: buf_h[1].as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: buf_u[1].as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: buf_v[1].as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 5, resource: buf_z[1].as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 6, resource: buf_bedrock.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 7, resource: buf_c[0].as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 8, resource: buf_z[0].as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 9, resource: buf_c[1].as_entire_binding() },
                ],
            }),
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Exner BG 1->0"),
                layout: &exner_bg_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: buf_domain.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: buf_dt.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: buf_h[0].as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: buf_u[0].as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: buf_v[0].as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 5, resource: buf_z[0].as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 6, resource: buf_bedrock.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 7, resource: buf_c[1].as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 8, resource: buf_z[1].as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 9, resource: buf_c[0].as_entire_binding() },
                ],
            }),
        ];

        let bg_talus = [
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Talus BG 0->1"),
                layout: &talus_bg_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: buf_domain.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: buf_z[0].as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: buf_bedrock.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: buf_sat[1].as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: buf_z[1].as_entire_binding() },
                ],
            }),
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Talus BG 1->0"),
                layout: &talus_bg_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: buf_domain.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: buf_z[1].as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: buf_bedrock.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: buf_sat[0].as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: buf_z[0].as_entire_binding() },
                ],
            }),
        ];

        let bg_sed_boundary = [
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Sed Boundary BG 1"),
                layout: &sed_bg_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: buf_domain.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: buf_z[1].as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: buf_c[1].as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: buf_sat[1].as_entire_binding() },
                ],
            }),
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Sed Boundary BG 0"),
                layout: &sed_bg_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: buf_domain.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: buf_z[0].as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: buf_c[0].as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: buf_sat[0].as_entire_binding() },
                ],
            }),
        ];

        // Staging buffers for readback
        let make_staging = |label: &str| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: buffer_byte_size,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };

        let staging_h = make_staging("Staging H");
        let staging_u = make_staging("Staging U");
        let staging_v = make_staging("Staging V");
        let staging_z = make_staging("Staging Z");
        let staging_c = make_staging("Staging C");
        let staging_sat = make_staging("Staging Sat");

        Self {
            device,
            queue,
            swe_pipeline,
            swe_boundary_pipeline,
            sat_pipeline,
            exner_pipeline,
            talus_pipeline,
            sed_boundary_pipeline,
            bg_swe,
            bg_sat,
            bg_exner,
            bg_talus,
            bg_sed_boundary,
            cpu_grid,
            _buf_domain: buf_domain,
            buf_dt,
            buf_h,
            buf_u,
            buf_v,
            buf_z,
            buf_c,
            buf_sat,
            buf_bedrock,
            staging_h,
            staging_u,
            staging_v,
            staging_z,
            staging_c,
            staging_sat,
            buffer_byte_size,
            ping_pong: 0,
        }
    }

    /// Synchronously initializes the GPU simulator via blocking execution.
    pub fn new_sync(cpu_grid: DoubleBufferedGrid) -> Self {
        pollster::block_on(Self::new(cpu_grid))
    }

    /// Advances the GPU compute simulation forward by a single time step `dt`.
    pub fn step(&mut self, dt: f32) {
        self.cpu_grid.time += dt;
        let gpu_params = GpuStepParams::new(dt, self.cpu_grid.time, &self.cpu_grid.boundaries);
        self.queue.write_buffer(&self.buf_dt, 0, bytemuck::bytes_of(&gpu_params));

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        let workgroups_x = self.cpu_grid.descriptor.grid_res_x.div_ceil(16);
        let workgroups_y = self.cpu_grid.descriptor.grid_res_y.div_ceil(16);
        let pp = self.ping_pong;

        // Pass 1: SWE Interior Kernel
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("SWE Main Pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.swe_pipeline);
            cpass.set_bind_group(0, &self.bg_swe[pp], &[]);
            cpass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
        }

        // Pass 2: SWE Boundary Reflection
        {
            let mut bpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("SWE Boundary Pass"),
                timestamp_writes: None,
            });
            bpass.set_pipeline(&self.swe_boundary_pipeline);
            bpass.set_bind_group(0, &self.bg_swe[pp], &[]);
            bpass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
        }

        // Pass 3: Soil Saturation Tracking
        {
            let mut spass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Saturation Pass"),
                timestamp_writes: None,
            });
            spass.set_pipeline(&self.sat_pipeline);
            spass.set_bind_group(0, &self.bg_sat[pp], &[]);
            spass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
        }

        // Pass 4: Exner Sediment Exchange
        {
            let mut epass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Exner Pass"),
                timestamp_writes: None,
            });
            epass.set_pipeline(&self.exner_pipeline);
            epass.set_bind_group(0, &self.bg_exner[pp], &[]);
            epass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
        }

        // Pass 5: Talus Angle-of-Repose Relaxation
        {
            let mut tpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Talus Pass"),
                timestamp_writes: None,
            });
            tpass.set_pipeline(&self.talus_pipeline);
            tpass.set_bind_group(0, &self.bg_talus[pp], &[]);
            tpass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
        }

        // Pass 6: Sediment Fields Boundary Reflection
        {
            let mut sbpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Sed Boundary Pass"),
                timestamp_writes: None,
            });
            sbpass.set_pipeline(&self.sed_boundary_pipeline);
            sbpass.set_bind_group(0, &self.bg_sed_boundary[pp], &[]);
            sbpass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
        }

        self.queue.submit(Some(encoder.finish()));
        self.ping_pong = 1 - self.ping_pong;
    }

    /// Synchronously reads back the latest fluid depth, velocity, bed, and sediment fields from GPU.
    pub fn sync_to_cpu(&mut self) {
        let out_idx = self.ping_pong;

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Sync Encoder"),
        });

        encoder.copy_buffer_to_buffer(&self.buf_h[out_idx], 0, &self.staging_h, 0, self.buffer_byte_size);
        encoder.copy_buffer_to_buffer(&self.buf_u[out_idx], 0, &self.staging_u, 0, self.buffer_byte_size);
        encoder.copy_buffer_to_buffer(&self.buf_v[out_idx], 0, &self.staging_v, 0, self.buffer_byte_size);
        encoder.copy_buffer_to_buffer(&self.buf_z[out_idx], 0, &self.staging_z, 0, self.buffer_byte_size);
        encoder.copy_buffer_to_buffer(&self.buf_c[out_idx], 0, &self.staging_c, 0, self.buffer_byte_size);
        encoder.copy_buffer_to_buffer(&self.buf_sat[out_idx], 0, &self.staging_sat, 0, self.buffer_byte_size);
        self.queue.submit(Some(encoder.finish()));

        let slice_h = self.staging_h.slice(..);
        let slice_u = self.staging_u.slice(..);
        let slice_v = self.staging_v.slice(..);
        let slice_z = self.staging_z.slice(..);
        let slice_c = self.staging_c.slice(..);
        let slice_sat = self.staging_sat.slice(..);

        let (sender, receiver) = std::sync::mpsc::channel();
        let s_u = sender.clone();
        let s_v = sender.clone();
        let s_z = sender.clone();
        let s_c = sender.clone();
        let s_sat = sender.clone();

        slice_h.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());
        slice_u.map_async(wgpu::MapMode::Read, move |v| s_u.send(v).unwrap());
        slice_v.map_async(wgpu::MapMode::Read, move |v| s_v.send(v).unwrap());
        slice_z.map_async(wgpu::MapMode::Read, move |v| s_z.send(v).unwrap());
        slice_c.map_async(wgpu::MapMode::Read, move |v| s_c.send(v).unwrap());
        slice_sat.map_async(wgpu::MapMode::Read, move |v| s_sat.send(v).unwrap());

        self.device.poll(wgpu::Maintain::Wait);

        for _ in 0..6 {
            let _ = receiver.recv().unwrap();
        }

        {
            let data = slice_h.get_mapped_range();
            self.cpu_grid.current.h.copy_from_slice(bytemuck::cast_slice(&data));
        }
        self.staging_h.unmap();

        {
            let data = slice_u.get_mapped_range();
            self.cpu_grid.current.u.copy_from_slice(bytemuck::cast_slice(&data));
        }
        self.staging_u.unmap();

        {
            let data = slice_v.get_mapped_range();
            self.cpu_grid.current.v.copy_from_slice(bytemuck::cast_slice(&data));
        }
        self.staging_v.unmap();

        {
            let data = slice_z.get_mapped_range();
            self.cpu_grid.current.z_bed.copy_from_slice(bytemuck::cast_slice(&data));
        }
        self.staging_z.unmap();

        {
            let data = slice_c.get_mapped_range();
            self.cpu_grid.current.sediment_c.copy_from_slice(bytemuck::cast_slice(&data));
        }
        self.staging_c.unmap();

        {
            let data = slice_sat.get_mapped_range();
            self.cpu_grid.current.soil_sat.copy_from_slice(bytemuck::cast_slice(&data));
        }
        self.staging_sat.unmap();
    }

    /// Uploads host CPU state modifications to the active GPU compute storage buffers.
    pub fn upload_state(&mut self) {
        self.cpu_grid.apply_reflective_boundaries();
        let in_idx = self.ping_pong;
        self.queue.write_buffer(&self.buf_h[in_idx], 0, bytemuck::cast_slice(&self.cpu_grid.current.h));
        self.queue.write_buffer(&self.buf_u[in_idx], 0, bytemuck::cast_slice(&self.cpu_grid.current.u));
        self.queue.write_buffer(&self.buf_v[in_idx], 0, bytemuck::cast_slice(&self.cpu_grid.current.v));
        self.queue.write_buffer(&self.buf_z[in_idx], 0, bytemuck::cast_slice(&self.cpu_grid.current.z_bed));
        self.queue.write_buffer(&self.buf_c[in_idx], 0, bytemuck::cast_slice(&self.cpu_grid.current.sediment_c));
        self.queue.write_buffer(&self.buf_sat[in_idx], 0, bytemuck::cast_slice(&self.cpu_grid.current.soil_sat));
        self.queue.write_buffer(&self.buf_bedrock, 0, bytemuck::cast_slice(&self.cpu_grid.current.bedrock_z));
    }

    /// Uploads host CPU fluid depth modifications (`h`) to the active GPU compute storage buffer.
    pub fn upload_water_depth(&mut self) {
        let in_idx = self.ping_pong;
        self.queue.write_buffer(&self.buf_h[in_idx], 0, bytemuck::cast_slice(&self.cpu_grid.current.h));
    }

    /// Subdivides the total simulation duration `total_dt` into uniform stable sub-steps
    /// and encodes all passes into a single command buffer submission to eliminate GPU driver bubbles.
    pub fn step_subdivided(&mut self, total_dt: f32, max_sub_dt: f32) {
        if total_dt <= 1e-6 {
            return;
        }
        self.cpu_grid.time += total_dt;
        let steps = (total_dt / max_sub_dt).ceil().max(1.0) as usize;
        let dt = total_dt / steps as f32;

        let gpu_params = GpuStepParams::new(dt, self.cpu_grid.time, &self.cpu_grid.boundaries);
        self.queue.write_buffer(&self.buf_dt, 0, bytemuck::bytes_of(&gpu_params));

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Subdivided Compute Passes"),
        });
        let workgroups_x = self.cpu_grid.descriptor.grid_res_x.div_ceil(16);
        let workgroups_y = self.cpu_grid.descriptor.grid_res_y.div_ceil(16);

        for _ in 0..steps {
            let pp = self.ping_pong;

            // Pass 1: SWE Interior Kernel
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("SWE Main Pass"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&self.swe_pipeline);
                cpass.set_bind_group(0, &self.bg_swe[pp], &[]);
                cpass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
            }

            // Pass 2: SWE Boundary Reflection
            {
                let mut bpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("SWE Boundary Pass"),
                    timestamp_writes: None,
                });
                bpass.set_pipeline(&self.swe_boundary_pipeline);
                bpass.set_bind_group(0, &self.bg_swe[pp], &[]);
                bpass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
            }

            // Pass 3: Soil Saturation Tracking
            {
                let mut spass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Saturation Pass"),
                    timestamp_writes: None,
                });
                spass.set_pipeline(&self.sat_pipeline);
                spass.set_bind_group(0, &self.bg_sat[pp], &[]);
                spass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
            }

            // Pass 4: Exner Sediment Exchange
            {
                let mut epass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Exner Pass"),
                    timestamp_writes: None,
                });
                epass.set_pipeline(&self.exner_pipeline);
                epass.set_bind_group(0, &self.bg_exner[pp], &[]);
                epass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
            }

            // Pass 5: Talus Angle-of-Repose Relaxation
            {
                let mut tpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Talus Pass"),
                    timestamp_writes: None,
                });
                tpass.set_pipeline(&self.talus_pipeline);
                tpass.set_bind_group(0, &self.bg_talus[pp], &[]);
                tpass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
            }

            // Pass 6: Sediment Fields Boundary Reflection
            {
                let mut sbpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Sed Boundary Pass"),
                    timestamp_writes: None,
                });
                sbpass.set_pipeline(&self.sed_boundary_pipeline);
                sbpass.set_bind_group(0, &self.bg_sed_boundary[pp], &[]);
                sbpass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
            }

            self.ping_pong = 1 - self.ping_pong;
        }

        self.queue.submit(Some(encoder.finish()));
    }
}

impl SimulationBackend for WgpuSimulator {
    fn backend_name(&self) -> &'static str {
        "GPU (wgpu/WGSL)"
    }

    fn descriptor(&self) -> &SimDomainDescriptor {
        &self.cpu_grid.descriptor
    }

    fn step(&mut self, dt: f32) {
        self.step(dt);
    }

    fn step_subdivided(&mut self, total_dt: f32, max_sub_dt: f32) {
        self.step_subdivided(total_dt, max_sub_dt);
    }

    fn sync_to_cpu(&mut self) {
        self.sync_to_cpu();
    }

    fn current_state(&self) -> &GridState {
        &self.cpu_grid.current
    }

    fn current_state_mut(&mut self) -> &mut GridState {
        &mut self.cpu_grid.current
    }

    fn previous_state(&self) -> &GridState {
        &self.cpu_grid.next
    }

    fn upload_state(&mut self) {
        self.upload_state();
    }

    fn upload_water_depth(&mut self) {
        self.upload_water_depth();
    }

    fn compute_max_stable_dt(&self, cfl: f32) -> f32 {
        let dx = self.cpu_grid.descriptor.extent_x / self.cpu_grid.descriptor.grid_res_x as f32;
        let dy = self.cpu_grid.descriptor.extent_y / self.cpu_grid.descriptor.grid_res_y as f32;
        sim_core::solver::swe::compute_max_stable_dt(&self.cpu_grid.current, dx, dy, cfl, 9.81)
    }

    fn boundaries(&self) -> &sim_core::boundary::DomainBoundaryConfig {
        &self.cpu_grid.boundaries
    }

    fn set_boundaries(&mut self, boundaries: sim_core::boundary::DomainBoundaryConfig) {
        self.cpu_grid.boundaries = boundaries;
    }

    fn sim_time(&self) -> f32 {
        self.cpu_grid.time
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sim_core::domain::SimDomainDescriptor;

    #[test]
    fn test_wgpu_simulator_backend_trait() {
        let desc = SimDomainDescriptor {
            grid_res_x: 32,
            grid_res_y: 32,
            extent_x: 32.0,
            extent_y: 32.0,
            ..Default::default()
        };

        let mut grid = DoubleBufferedGrid::new(desc);

        // Bedrock channel: impervious rock so zero porous infiltration occurs
        for i in 0..grid.current.z_bed.len() {
            grid.current.bedrock_z[i] = grid.current.z_bed[i];
        }

        // Water block in the center
        for y in 10..22 {
            for x in 10..22 {
                let idx = grid.current.idx(x, y);
                grid.current.h[idx] = 3.0;
            }
        }

        let mut sim: Box<dyn SimulationBackend> = Box::new(WgpuSimulator::new_sync(grid));
        assert_eq!(sim.backend_name(), "GPU (wgpu/WGSL)");

        let initial_mass = sim.total_fluid_mass();
        assert!(initial_mass > 0.0);

        let dt = 0.005;
        for _ in 0..50 {
            sim.step(dt);
        }

        sim.sync_to_cpu();

        let final_mass = sim.total_fluid_mass();
        let diff = (initial_mass - final_mass).abs();

        assert!(
            diff < 1e-2,
            "GPU simulation mass not conserved! Initial: {}, Final: {}, Diff: {}",
            initial_mass,
            final_mass,
            diff
        );
    }

    #[test]
    fn test_wgpu_sediment_conservation() {
        let desc = SimDomainDescriptor {
            grid_res_x: 32,
            grid_res_y: 32,
            extent_x: 32.0,
            extent_y: 32.0,
            ..Default::default()
        };

        let mut grid = DoubleBufferedGrid::new(desc);

        // Saturated sand bed of 1.5m over bedrock at 0.0m
        for i in 0..grid.current.z_bed.len() {
            grid.current.z_bed[i] = 1.5;
            grid.current.bedrock_z[i] = 0.0;
            grid.current.soil_sat[i] = 1.0;
        }

        // Fast water stream in the middle to trigger erosion and suspended transport
        for y in 10..22 {
            for x in 10..22 {
                let idx = grid.current.idx(x, y);
                grid.current.h[idx] = 1.0;
                grid.current.u[idx] = 1.5;
            }
        }

        let mut sim: Box<dyn SimulationBackend> = Box::new(WgpuSimulator::new_sync(grid));
        let initial_sed_mass = sim.total_sediment_mass();
        let initial_fluid_mass = sim.total_fluid_mass();

        for _ in 0..50 {
            sim.step(0.01);
        }

        sim.sync_to_cpu();

        let final_sed_mass = sim.total_sediment_mass();
        let final_fluid_mass = sim.total_fluid_mass();

        // 1. Fluid mass conservation (allowing for minor pore absorption as wave spreads into marginally dried cells)
        let fluid_diff = (initial_fluid_mass - final_fluid_mass).abs();
        assert!(fluid_diff < 0.1, "GPU fluid mass not conserved! Diff: {}", fluid_diff);

        // 2. Sediment total mass conservation
        let sed_diff = (initial_sed_mass - final_sed_mass).abs();
        let rel_error = sed_diff / initial_sed_mass;
        assert!(
            rel_error < 5e-3,
            "GPU sediment mass not conserved! Initial: {}, Final: {}, Diff: {}, Rel: {}",
            initial_sed_mass,
            final_sed_mass,
            sed_diff,
            rel_error
        );

        // 3. Confirm erosion and transport took place
        let max_c = sim.current_state().sediment_c.iter().copied().fold(0.0f32, f32::max);
        let min_z = sim.current_state().z_bed.iter().copied().fold(f32::INFINITY, f32::min);
        assert!(max_c > 0.001, "GPU erosion failed to suspend sediment! Max C: {}", max_c);
        assert!(min_z < 1.499, "GPU flow did not erode bed! Min Z: {}", min_z);

        // 4. Confirm bedrock limit
        for (i, &z) in sim.current_state().z_bed.iter().enumerate() {
            let b = sim.current_state().bedrock_z[i];
            assert!(z >= b - 1e-5, "GPU erosion breached bedrock! Z: {}, Bedrock: {}", z, b);
        }
    }

    #[test]
    fn test_wgpu_beach_waves_scenario() {
        use sim_core::Scenarios;

        let desc = SimDomainDescriptor {
            grid_res_x: 64,
            grid_res_y: 64,
            extent_x: 50.0,
            extent_y: 50.0,
            ..Default::default()
        };

        let grid = Scenarios::beach_sandcastle_waves(desc);
        let mut sim: Box<dyn SimulationBackend> = Box::new(WgpuSimulator::new_sync(grid));

        // Step simulation on GPU for 40 steps (0.4s of physical time with wave surge)
        for _ in 0..40 {
            sim.step(0.01);
        }

        sim.sync_to_cpu();
        let state = sim.current_state();

        // 1. Verify positivity and no NaNs across all cells
        for (i, &h) in state.h.iter().enumerate() {
            assert!(h >= 0.0, "GPU wave fluid depth negative at {}: {}", i, h);
            assert!(!h.is_nan(), "GPU wave fluid depth NaN at {}", i);
            let z = state.z_bed[i];
            let b = state.bedrock_z[i];
            assert!(z >= b - 1e-4, "GPU bed below bedrock at {}: z={}, b={}", i, z, b);
            assert!(!z.is_nan(), "GPU bed elevation NaN at {}", i);
        }

        // 2. Verify wave has surged onshore into South cells
        let swash_idx = state.idx(32, 50); // row 50 is near the coastline
        assert!(
            state.h[swash_idx] > 0.01,
            "GPU wave failed to lap onto beach at row 50! Depth: {}",
            state.h[swash_idx]
        );
    }

    #[test]
    fn test_wgpu_porous_infiltration() {
        let desc = SimDomainDescriptor {
            grid_res_x: 16,
            grid_res_y: 16,
            extent_x: 16.0,
            extent_y: 16.0,
            ..Default::default()
        };

        let mut grid = DoubleBufferedGrid::new(desc);

        // Dry sand bed of 0.5m over bedrock at 0.0m
        for i in 0..grid.current.z_bed.len() {
            grid.current.z_bed[i] = 0.5;
            grid.current.bedrock_z[i] = 0.0;
            grid.current.soil_sat[i] = 0.0;
        }

        // Add shallow water in center 4x4 block
        for y in 6..10 {
            for x in 6..10 {
                let idx = grid.current.idx(x, y);
                grid.current.h[idx] = 0.04;
            }
        }

        // Impervious stone cell at (4, 4) with bedrock == z_bed (0 soil depth)
        let stone_idx = grid.current.idx(4, 4);
        grid.current.z_bed[stone_idx] = 0.5;
        grid.current.bedrock_z[stone_idx] = 0.5;
        grid.current.h[stone_idx] = 0.04;

        let mut sim: Box<dyn SimulationBackend> = Box::new(WgpuSimulator::new_sync(grid));
        let initial_surface_water = sim.total_fluid_mass();
        let initial_total_water = sim.current_state().interior_total_water_mass(0.40);

        assert!(initial_surface_water > 0.0);

        // Step 5 times on GPU
        for _ in 0..5 {
            sim.step(0.05);
        }

        sim.sync_to_cpu();

        let post_surface_water = sim.total_fluid_mass();
        let post_total_water = sim.current_state().interior_total_water_mass(0.40);
        let sample_idx = sim.current_state().idx(7, 7);

        // 1. Surface water h must strictly decrease on GPU as it soaks into dry sand
        assert!(
            post_surface_water < initial_surface_water,
            "GPU porous infiltration did not drain surface water! Initial: {}, Post: {}",
            initial_surface_water,
            post_surface_water
        );

        // 2. Soil moisture W_sat must increase
        assert!(
            sim.current_state().soil_sat[sample_idx] > 0.0,
            "GPU soil saturation did not increase! Got: {}",
            sim.current_state().soil_sat[sample_idx]
        );

        // 3. Strict total water mass conservation (surface h + pore moisture)
        let total_diff = (initial_total_water - post_total_water).abs();
        assert!(
            total_diff < 0.05,
            "GPU total water mass not conserved! Initial: {}, Post: {}, Diff: {}",
            initial_total_water,
            post_total_water,
            total_diff
        );
    }
}

