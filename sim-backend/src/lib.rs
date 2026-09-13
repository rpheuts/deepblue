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
    
    // Cached CPU state for readback
    pub cpu_grid: DoubleBufferedGrid,
    
    // GPU Uniform Buffers
    _buf_domain: wgpu::Buffer,
    buf_dt: wgpu::Buffer,
    
    // Ping-pong storage buffers
    buf_h: [wgpu::Buffer; 2],
    _buf_u: [wgpu::Buffer; 2],
    _buf_v: [wgpu::Buffer; 2],
    _buf_z: [wgpu::Buffer; 2],

    // Persistent staging buffer for readback (avoid per-frame allocation)
    staging_h: wgpu::Buffer,
    buffer_byte_size: u64,
    
    ping_pong: usize,
}

impl WgpuSimulator {
    pub async fn new(mut cpu_grid: DoubleBufferedGrid) -> Self {
        // Ensure reflective boundary halo cells are valid on both buffers
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

        // Load SWE compute shader
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
        // Pad uniform buffer to 16 bytes for strict WebGPU/Vulkan alignment
        let mut dt_padded = [0u8; 16];
        dt_padded[0..4].copy_from_slice(bytemuck::bytes_of(&dt));
        let buf_dt = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Dt Buffer"),
            contents: &dt_padded,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Create storage buffers
        let cell_count = (cpu_grid.descriptor.grid_res_x * cpu_grid.descriptor.grid_res_y) as usize;
        let buffer_byte_size = (cell_count * std::mem::size_of::<f32>()) as u64;

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

        let staging_h = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer H"),
            size: buffer_byte_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

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
            staging_h,
            buffer_byte_size,
            ping_pong: 0,
        }
    }

    pub fn step(&mut self, dt: f32) {
        self.queue.write_buffer(&self.buf_dt, 0, bytemuck::bytes_of(&dt));

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        let workgroups_x = (self.cpu_grid.descriptor.grid_res_x + 15) / 16;
        let workgroups_y = (self.cpu_grid.descriptor.grid_res_y + 15) / 16;

        // Pass 1: Interior SWE kernel dispatch
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("SWE Main Pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.compute_pipeline);
            if self.ping_pong == 0 {
                cpass.set_bind_group(0, &self.bind_group_0, &[]);
            } else {
                cpass.set_bind_group(0, &self.bind_group_1, &[]);
            }
            cpass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
        }

        // Pass 2: Boundary reflection kernel dispatch
        // Running in a separate compute pass guarantees a full GPU execution and memory barrier
        {
            let mut bpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("SWE Boundary Pass"),
                timestamp_writes: None,
            });
            bpass.set_pipeline(&self.boundary_pipeline);
            if self.ping_pong == 0 {
                bpass.set_bind_group(0, &self.bind_group_0, &[]);
            } else {
                bpass.set_bind_group(0, &self.bind_group_1, &[]);
            }
            bpass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
        }

        self.queue.submit(Some(encoder.finish()));
        self.ping_pong = 1 - self.ping_pong;
    }

    pub async fn sync_to_cpu(&mut self) {
        // The most recently written buffer is indexed by self.ping_pong
        let out_idx = self.ping_pong;

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Sync Encoder"),
        });

        encoder.copy_buffer_to_buffer(&self.buf_h[out_idx], 0, &self.staging_h, 0, self.buffer_byte_size);
        self.queue.submit(Some(encoder.finish()));

        let buffer_slice = self.staging_h.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());

        self.device.poll(wgpu::Maintain::Wait);

        if receiver.recv().unwrap().is_ok() {
            let data = buffer_slice.get_mapped_range();
            let floats: &[f32] = bytemuck::cast_slice(&data);

            self.cpu_grid.current.h.copy_from_slice(floats);
            drop(data);
            self.staging_h.unmap();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sim_core::domain::SimDomainDescriptor;

    #[test]
    fn test_wgpu_simulator_step_and_conservation() {
        pollster::block_on(async {
            let mut desc = SimDomainDescriptor::default();
            desc.grid_res_x = 32;
            desc.grid_res_y = 32;
            desc.extent_x = 32.0;
            desc.extent_y = 32.0;

            let mut grid = DoubleBufferedGrid::new(desc);

            // Water block in the center
            for y in 10..22 {
                for x in 10..22 {
                    let idx = grid.current.idx(x, y);
                    grid.current.h[idx] = 3.0;
                }
            }

            let initial_mass = grid.current.interior_mass();

            let mut sim = WgpuSimulator::new(grid).await;

            let dt = 0.005;
            for _ in 0..50 {
                sim.step(dt);
            }

            sim.sync_to_cpu().await;

            let final_mass = sim.cpu_grid.current.interior_mass();
            let diff = (initial_mass - final_mass).abs();

            assert!(
                diff < 1e-2,
                "GPU simulation mass not conserved! Initial: {}, Final: {}, Diff: {}",
                initial_mass,
                final_mass,
                diff
            );
        });
    }
}
