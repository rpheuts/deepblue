use crate::params::{BrushParams, GpuStepParams};
use crate::simulator::WgpuSimulator;

/// Applies a brush operation directly to a CPU GridState bathymetry/water field,
/// matching the exact arithmetic and falloff of `shaders/brush.wgsl`.
pub fn apply_brush_to_grid(
    grid: &mut sim_core::state::GridState,
    desc: &sim_core::domain::SimDomainDescriptor,
    center_x: f32,
    center_y: f32,
    radius: f32,
    strength: f32,
    tool_type: u32,
) {
    let norm_x = center_x / desc.extent_x;
    let norm_y = center_y / desc.extent_y;
    let gx = (norm_x * (desc.grid_res_x - 1) as f32).round() as i32;
    let gy = (norm_y * (desc.grid_res_y - 1) as f32).round() as i32;

    let radius_cells = ((radius / desc.extent_x) * (desc.grid_res_x as f32)).round().max(1.0) as i32;
    let r2 = radius_cells * radius_cells;

    for dy in -radius_cells..=radius_cells {
        for dx in -radius_cells..=radius_cells {
            let dist2 = dx * dx + dy * dy;
            if dist2 <= r2 {
                let px = (gx + dx).clamp(1, desc.grid_res_x as i32 - 2) as u32;
                let py = (gy + dy).clamp(1, desc.grid_res_y as i32 - 2) as u32;
                let idx = grid.idx(px, py);
                let falloff = 1.0 - (dist2 as f32 / r2 as f32).sqrt();

                match tool_type {
                    1 => {
                        grid.h[idx] += strength * 1.20 * falloff;
                    }
                    2 => {
                        grid.z_bed[idx] += strength * 0.60 * falloff;
                        grid.soil_sat[idx] = 0.25;
                    }
                    3 => {
                        grid.z_bed[idx] += strength * 0.75 * falloff;
                        grid.bedrock_z[idx] = grid.z_bed[idx];
                    }
                    4 => {
                        let cur_z = grid.z_bed[idx];
                        let bedrock = grid.bedrock_z[idx];
                        grid.z_bed[idx] = (cur_z - strength * 0.70 * falloff).max(bedrock);
                    }
                    _ => {}
                }
            }
        }
    }
}

impl WgpuSimulator {
    /// Enqueues the full 6-pass coupled hydrodynamic & sediment compute sequence into a command encoder.
    pub(crate) fn record_physics_passes(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        pp: usize,
        workgroups_x: u32,
        workgroups_y: u32,
    ) {
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
    }

    /// Advances the GPU compute simulation forward by a single time step `dt`.
    pub fn step(&mut self, dt: f32) {
        self.cpu_grid.time += dt;
        let mut gpu_params = GpuStepParams::new(
            dt,
            self.cpu_grid.time,
            &self.cpu_grid.boundaries,
            self.stream_inflow_active,
            self.coastal_sink_active,
        );
        gpu_params.wind_speed = self.wind_speed;
        gpu_params.wind_dir_x = self.wind_dir[0];
        gpu_params.wind_dir_y = self.wind_dir[1];
        gpu_params.wind_drag_coeff = 0.00025;
        gpu_params.wind_turbulence = self.wind_turbulence;
        gpu_params.wind_shelter = self.wind_shelter;
        gpu_params._step_param_pad1 = 0.0;
        gpu_params._step_param_pad2 = 0.0;
        self.queue.write_buffer(&self.buf_dt, 0, bytemuck::bytes_of(&gpu_params));

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        let workgroups_x = self.cpu_grid.descriptor.grid_res_x.div_ceil(16);
        let workgroups_y = self.cpu_grid.descriptor.grid_res_y.div_ceil(16);
        let pp = self.ping_pong;

        self.record_physics_passes(&mut encoder, pp, workgroups_x, workgroups_y);

        self.ping_pong = 1 - self.ping_pong;

        // Pass 7: Texture Export Pass (Zero-Copy VRAM)
        {
            let mut exp_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Export Textures Pass"),
                timestamp_writes: None,
            });
            exp_pass.set_pipeline(&self.export_pipeline);
            exp_pass.set_bind_group(0, &self.bg_export[self.ping_pong], &[]);
            exp_pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
        }

        self.queue.submit(Some(encoder.finish()));
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

        let mut gpu_params = GpuStepParams::new(
            dt,
            self.cpu_grid.time,
            &self.cpu_grid.boundaries,
            self.stream_inflow_active,
            self.coastal_sink_active,
        );
        gpu_params.wind_speed = self.wind_speed;
        gpu_params.wind_dir_x = self.wind_dir[0];
        gpu_params.wind_dir_y = self.wind_dir[1];
        gpu_params.wind_drag_coeff = 0.00025;
        gpu_params.wind_turbulence = self.wind_turbulence;
        gpu_params.wind_shelter = self.wind_shelter;
        gpu_params._step_param_pad1 = 0.0;
        gpu_params._step_param_pad2 = 0.0;
        self.queue.write_buffer(&self.buf_dt, 0, bytemuck::bytes_of(&gpu_params));

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Subdivided Compute Passes"),
        });
        let workgroups_x = self.cpu_grid.descriptor.grid_res_x.div_ceil(16);
        let workgroups_y = self.cpu_grid.descriptor.grid_res_y.div_ceil(16);

        for _ in 0..steps {
            let pp = self.ping_pong;
            self.record_physics_passes(&mut encoder, pp, workgroups_x, workgroups_y);
            self.ping_pong = 1 - self.ping_pong;
        }

        // Pass 7: Texture Export Pass (Zero-Copy VRAM)
        {
            let mut exp_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Export Textures Pass"),
                timestamp_writes: None,
            });
            exp_pass.set_pipeline(&self.export_pipeline);
            exp_pass.set_bind_group(0, &self.bg_export[self.ping_pong], &[]);
            exp_pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
        }

        self.queue.submit(Some(encoder.finish()));
    }

    /// Dispatches an in-situ GPU compute brush operation directly mutating active VRAM storage buffers.
    /// Also updates the host-side `cpu_grid.current` mirror to keep CPU raycasting and inspection 100% in sync.
    pub fn apply_brush(
        &mut self,
        center_x: f32,
        center_y: f32,
        radius: f32,
        strength: f32,
        tool_type: u32,
    ) {
        let params = BrushParams {
            center_x,
            center_y,
            radius,
            strength,
            tool_type,
            grid_res_x: self.cpu_grid.descriptor.grid_res_x,
            grid_res_y: self.cpu_grid.descriptor.grid_res_y,
            extent_x: self.cpu_grid.descriptor.extent_x,
            extent_y: self.cpu_grid.descriptor.extent_y,
            pad0: 0.0,
            pad1: 0.0,
            pad2: 0.0,
        };
        self.queue.write_buffer(&self.buf_brush, 0, bytemuck::bytes_of(&params));

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Brush Compute Command Encoder"),
        });

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Brush Compute Pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.brush_pipeline);
            cpass.set_bind_group(0, &self.bg_brush, &[]);
            let workgroups_x = self.cpu_grid.descriptor.grid_res_x.div_ceil(16);
            let workgroups_y = self.cpu_grid.descriptor.grid_res_y.div_ceil(16);
            cpass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
        }

        self.queue.submit(Some(encoder.finish()));
        self.export_textures();

        // Synchronize local host mirror for CPU raycasting / inspection
        apply_brush_to_grid(
            &mut self.cpu_grid.current,
            &self.cpu_grid.descriptor,
            center_x,
            center_y,
            radius,
            strength,
            tool_type,
        );
    }

    /// Explicitly executes the texture export pass to ensure 2D textures match current storage buffers.
    pub fn export_textures(&mut self) {
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Manual Export Encoder"),
        });
        let workgroups_x = self.cpu_grid.descriptor.grid_res_x.div_ceil(16);
        let workgroups_y = self.cpu_grid.descriptor.grid_res_y.div_ceil(16);
        {
            let mut exp_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Export Textures Pass"),
                timestamp_writes: None,
            });
            exp_pass.set_pipeline(&self.export_pipeline);
            exp_pass.set_bind_group(0, &self.bg_export[self.ping_pong], &[]);
            exp_pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
        }
        self.queue.submit(Some(encoder.finish()));
    }
}
