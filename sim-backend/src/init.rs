use std::borrow::Cow;
use std::sync::Arc;
use wgpu::util::DeviceExt;
use sim_core::state::DoubleBufferedGrid;
use crate::params::{BrushParams, GpuStepParams};
use crate::simulator::WgpuSimulator;

impl WgpuSimulator {
    /// Initializes a GPU-accelerated hydraulic & sediment simulator from an existing wgpu Device and Queue.
    /// This enables zero-copy texture sharing with renderers and window surfaces on the exact same GPU context.
    pub fn from_device(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>, mut cpu_grid: DoubleBufferedGrid) -> Self {
        cpu_grid.current.apply_reflective_boundaries();
        cpu_grid.next.apply_reflective_boundaries();

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
        let export_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Export Textures Shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("shaders/export_textures.wgsl"))),
        });
        let brush_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Brush Shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("shaders/brush.wgsl"))),
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

        let export_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Export Pipeline"),
            layout: None,
            module: &export_shader,
            entry_point: "main",
            compilation_options: Default::default(),
        });
        let export_bg_layout = export_pipeline.get_bind_group_layout(0);

        let brush_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Brush Pipeline"),
            layout: None,
            module: &brush_shader,
            entry_point: "main",
            compilation_options: Default::default(),
        });
        let brush_bg_layout = brush_pipeline.get_bind_group_layout(0);

        // Uniform buffers
        let buf_domain = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Domain Buffer"),
            contents: bytemuck::bytes_of(&cpu_grid.descriptor),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let initial_params = GpuStepParams::new(0.016, cpu_grid.time, &cpu_grid.boundaries, false, false);
        let buf_dt = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Step Params Buffer"),
            contents: bytemuck::bytes_of(&initial_params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let buf_brush = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Brush Params Buffer"),
            size: std::mem::size_of::<BrushParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
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

        // 2D Textures for Zero-Copy rendering (AGENTS.md contract)
        let width = cpu_grid.descriptor.grid_res_x;
        let height = cpu_grid.descriptor.grid_res_y;

        let make_tex = |label: &str, format: wgpu::TextureFormat| {
            let tex = device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            });
            let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
            (tex, view)
        };

        let (tex_elevation, view_elevation) = make_tex("ElevationMap", wgpu::TextureFormat::R32Float);
        let (tex_water, view_water) = make_tex("WaterMap", wgpu::TextureFormat::Rgba32Float);
        let (tex_velocity, view_velocity) = make_tex("VelocityMap", wgpu::TextureFormat::Rgba32Float);
        let (tex_sed_sat, view_sed_sat) = make_tex("SedimentWetnessMap", wgpu::TextureFormat::Rgba32Float);

        let bg_export = [
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Export BG 0"),
                layout: &export_bg_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: buf_domain.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: buf_h[0].as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: buf_u[0].as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: buf_v[0].as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: buf_z[0].as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 5, resource: buf_c[0].as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 6, resource: buf_sat[0].as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 7, resource: buf_bedrock.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 8, resource: wgpu::BindingResource::TextureView(&view_elevation) },
                    wgpu::BindGroupEntry { binding: 9, resource: wgpu::BindingResource::TextureView(&view_water) },
                    wgpu::BindGroupEntry { binding: 10, resource: wgpu::BindingResource::TextureView(&view_velocity) },
                    wgpu::BindGroupEntry { binding: 11, resource: wgpu::BindingResource::TextureView(&view_sed_sat) },
                ],
            }),
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Export BG 1"),
                layout: &export_bg_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: buf_domain.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: buf_h[1].as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: buf_u[1].as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: buf_v[1].as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: buf_z[1].as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 5, resource: buf_c[1].as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 6, resource: buf_sat[1].as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 7, resource: buf_bedrock.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 8, resource: wgpu::BindingResource::TextureView(&view_elevation) },
                    wgpu::BindGroupEntry { binding: 9, resource: wgpu::BindingResource::TextureView(&view_water) },
                    wgpu::BindGroupEntry { binding: 10, resource: wgpu::BindingResource::TextureView(&view_velocity) },
                    wgpu::BindGroupEntry { binding: 11, resource: wgpu::BindingResource::TextureView(&view_sed_sat) },
                ],
            }),
        ];

        let bg_brush = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Brush BG"),
            layout: &brush_bg_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: buf_brush.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: buf_h[0].as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: buf_h[1].as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: buf_z[0].as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: buf_z[1].as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: buf_bedrock.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: buf_sat[0].as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: buf_sat[1].as_entire_binding() },
            ],
        });

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

        let mut sim = Self {
            device,
            queue,
            swe_pipeline,
            swe_boundary_pipeline,
            sat_pipeline,
            exner_pipeline,
            talus_pipeline,
            sed_boundary_pipeline,
            export_pipeline,
            brush_pipeline,
            bg_swe,
            bg_sat,
            bg_exner,
            bg_talus,
            bg_sed_boundary,
            bg_export,
            bg_brush,
            tex_elevation,
            view_elevation,
            tex_water,
            view_water,
            tex_velocity,
            view_velocity,
            tex_sed_sat,
            view_sed_sat,
            cpu_grid,
            _buf_domain: buf_domain,
            buf_dt,
            buf_brush,
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
            stream_inflow_active: false,
            coastal_sink_active: false,
            wind_speed: 0.0,
            wind_dir: [0.0, -1.0],
            wind_turbulence: 0.45,
            wind_shelter: 0.70,
        };

        sim.export_textures();
        sim
    }

    /// Asynchronously initializes a new GPU-accelerated hydraulic & sediment simulator.
    pub async fn new(cpu_grid: DoubleBufferedGrid) -> Self {
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

        Self::from_device(Arc::new(device), Arc::new(queue), cpu_grid)
    }

    /// Synchronously initializes the GPU simulator via blocking execution.
    pub fn new_sync(cpu_grid: DoubleBufferedGrid) -> Self {
        pollster::block_on(Self::new(cpu_grid))
    }
}
