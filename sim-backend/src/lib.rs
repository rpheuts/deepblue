// use sim_core::domain::SimDomainDescriptor;
use sim_core::state::DoubleBufferedGrid;
use wgpu::util::DeviceExt;
use std::borrow::Cow;

pub struct WgpuSimulator {
    device: wgpu::Device,
    queue: wgpu::Queue,
    compute_pipeline: wgpu::ComputePipeline,
    boundary_pipeline: wgpu::ComputePipeline,
    bind_group_0: wgpu::BindGroup,
    bind_group_1: wgpu::BindGroup,
    // We hold the CPU state so we can sync it back when requested
    pub cpu_grid: DoubleBufferedGrid,
    
    // GPU Buffers
    _buf_domain: wgpu::Buffer,
    buf_dt: wgpu::Buffer,
    
    // Ping-pong buffers
    buf_h: [wgpu::Buffer; 2],
    _buf_u: [wgpu::Buffer; 2],
    _buf_v: [wgpu::Buffer; 2],
    _buf_z: [wgpu::Buffer; 2],
    
    ping_pong: usize,
}

impl WgpuSimulator {
    pub async fn new(cpu_grid: DoubleBufferedGrid) -> Self {
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

        // Load shader
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("SWE Shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("shaders/swe.wgsl"))),
        });

        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("SWE Compute Pipeline"),
            layout: None,
            module: &shader,
            entry_point: "main",
            compilation_options: Default::default(),
        });

        let bind_group_layout = compute_pipeline.get_bind_group_layout(0);
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Shared Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let boundary_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("SWE Boundary Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "boundary",
            compilation_options: Default::default(),
        });

        // Create uniform buffers
        let buf_domain = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Domain Buffer"),
            contents: bytemuck::bytes_of(&cpu_grid.descriptor),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let dt = 0.016f32;
        let buf_dt = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Dt Buffer"),
            contents: bytemuck::bytes_of(&dt),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Create storage buffers
        let _size = (cpu_grid.descriptor.grid_res_x * cpu_grid.descriptor.grid_res_y) as usize * 4;
        
        let create_storage = |label: &str, data: &[f32]| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::cast_slice(data),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            })
        };

        let buf_h = [
            create_storage("h_0", &cpu_grid.current.h),
            create_storage("h_1", &cpu_grid.next.h),
        ];
        let buf_u = [
            create_storage("u_0", &cpu_grid.current.u),
            create_storage("u_1", &cpu_grid.next.u),
        ];
        let buf_v = [
            create_storage("v_0", &cpu_grid.current.v),
            create_storage("v_1", &cpu_grid.next.v),
        ];
        let buf_z = [
            create_storage("z_0", &cpu_grid.current.z_bed),
            create_storage("z_1", &cpu_grid.next.z_bed),
        ];

        // Create Bind Groups (Ping-Pong)
        
        let create_bg = |in_idx: usize, out_idx: usize| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("SWE Bind Group"),
                layout: &bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: buf_domain.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: buf_dt.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: buf_h[in_idx].as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: buf_u[in_idx].as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: buf_v[in_idx].as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 5, resource: buf_z[in_idx].as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 6, resource: buf_h[out_idx].as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 7, resource: buf_u[out_idx].as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 8, resource: buf_v[out_idx].as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 9, resource: buf_z[out_idx].as_entire_binding() },
                ],
            })
        };

        let bind_group_0 = create_bg(0, 1); // Read 0, Write 1
        let bind_group_1 = create_bg(1, 0); // Read 1, Write 0

        Self {
            device,
            queue,
            compute_pipeline,
            boundary_pipeline,
            bind_group_0,
            bind_group_1,
            cpu_grid,
            _buf_domain: buf_domain,
            buf_dt,
            buf_h,
            _buf_u: buf_u,
            _buf_v: buf_v,
            _buf_z: buf_z,
            ping_pong: 0,
        }
    }

    pub fn step(&mut self, dt: f32) {
        self.queue.write_buffer(&self.buf_dt, 0, bytemuck::bytes_of(&dt));

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
            cpass.set_pipeline(&self.compute_pipeline);
            
            if self.ping_pong == 0 {
                cpass.set_bind_group(0, &self.bind_group_0, &[]);
            } else {
                cpass.set_bind_group(0, &self.bind_group_1, &[]);
            }

            let workgroups_x = (self.cpu_grid.descriptor.grid_res_x + 15) / 16;
            let workgroups_y = (self.cpu_grid.descriptor.grid_res_y + 15) / 16;
            
            cpass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
            
            // Boundary pass
            cpass.set_pipeline(&self.boundary_pipeline);
            cpass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
        }

        self.queue.submit(Some(encoder.finish()));
        self.ping_pong = 1 - self.ping_pong;
    }

    pub async fn sync_to_cpu(&mut self) {
        // Find which buffer holds the latest data
        let out_idx = self.ping_pong;
        
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        
        let size = (self.cpu_grid.descriptor.grid_res_x * self.cpu_grid.descriptor.grid_res_y) as u64 * 4;
        let staging_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Let's just copy 'h' for now to demonstrate synchronization
        // In a real scenario, we'd copy all buffers if needed, or map them asynchronously
        encoder.copy_buffer_to_buffer(&self.buf_h[out_idx], 0, &staging_buf, 0, size);
        self.queue.submit(Some(encoder.finish()));

        let buffer_slice = staging_buf.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());

        self.device.poll(wgpu::Maintain::Wait);

        if receiver.recv().unwrap().is_ok() {
            let data = buffer_slice.get_mapped_range();
            let floats: &[f32] = bytemuck::cast_slice(&data);
            
            // Update CPU grid
            self.cpu_grid.current.h.copy_from_slice(floats);
            drop(data);
            staging_buf.unmap();
        }
    }
}
