use crate::simulator::WgpuSimulator;

impl WgpuSimulator {
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
        self.export_textures();
    }

    /// Uploads host CPU fluid depth modifications (`h`) to the active GPU compute storage buffer.
    pub fn upload_water_depth(&mut self) {
        let in_idx = self.ping_pong;
        self.queue.write_buffer(&self.buf_h[in_idx], 0, bytemuck::cast_slice(&self.cpu_grid.current.h));
        self.export_textures();
    }
}
